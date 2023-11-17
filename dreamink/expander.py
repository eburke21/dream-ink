"""GPT-4 scene expansion: raw dream notes -> rich visual scene descriptions."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from dreamink.config import AppConfig, get_config
from dreamink.models import SceneExpansion, TokenUsage
from dreamink.utils import extract_token_usage, get_openai_client, retry_api_call

logger = logging.getLogger(__name__)

SCENE_EXPANSION_SYSTEM_PROMPT = """You translate raw dream journal notes into vivid visual
scene descriptions suitable for image generation. Your job is to add visual richness while
preserving the dream's internal logic.

RULES:
- PRESERVE surreal, illogical elements exactly as described. If someone's mother is made
  of stained glass, describe the light refracting through her. Don't rationalize it.
- ADD visual details the dreamer likely experienced but didn't write down: lighting
  conditions, color palette, spatial perspective, atmospheric quality, textures.
- MAINTAIN the emotional tone. If the notes say "felt peaceful not scary," the scene
  description should evoke serenity even if the imagery is objectively strange.
- DO NOT add narrative or plot. Dreams are moments, not stories. Describe a single
  frozen scene, not a sequence of events.
- DO NOT add people or characters not mentioned in the notes.
- OUTPUT exactly 3-4 sentences. First sentence establishes the setting and dominant
  visual. Second adds surreal/dream-specific elements. Third captures atmosphere and
  lighting. Optional fourth for a specific striking detail.

You are describing a painting, not narrating a story."""

_METADATA_SYSTEM_PROMPT = """Given a visual scene description, return a JSON object with:
- "dominant_colors": array of 2-4 color names (e.g. ["amber", "cobalt blue", "deep red"])
- "mood": a single word or short phrase for the emotional tone (e.g. "peaceful", "unsettling", "nostalgic")

Return ONLY valid JSON, no other text."""


def expand_dream_notes(
    raw_notes: str,
    emotional_tone: str | None = None,
    client: OpenAI | None = None,
    config: AppConfig | None = None,
) -> SceneExpansion:
    """Expand fragmentary dream notes into a rich visual scene description.

    Args:
        raw_notes: Raw morning dream notes (typically 1-4 sentences, fragmentary).
        emotional_tone: Optional override for emotional quality. If None, inferred.
        client: OpenAI client. If None, creates one from config.
        config: App config. If None, loads defaults.

    Returns:
        SceneExpansion with description, dominant_colors, mood, and token_usage.
    """
    config = config or get_config()
    client = client or get_openai_client(config)

    user_message = raw_notes
    if emotional_tone:
        user_message += f"\n\n[Emotional tone: {emotional_tone}]"

    # Stage 1: expand the dream notes into a scene description
    expansion_response = retry_api_call(
        lambda: client.chat.completions.create(
            model=config.expansion_model,
            temperature=config.expansion_temperature,
            messages=[
                {"role": "system", "content": SCENE_EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        ),
        max_retries=config.max_retries,
    )

    description = expansion_response.choices[0].message.content.strip()
    expansion_usage = extract_token_usage(expansion_response)

    # Stage 2: extract dominant colors and mood from the scene
    metadata_response = retry_api_call(
        lambda: client.chat.completions.create(
            model=config.expansion_model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": _METADATA_SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
        ),
        max_retries=config.max_retries,
    )

    metadata_usage = extract_token_usage(metadata_response)
    metadata_text = metadata_response.choices[0].message.content.strip()

    dominant_colors = []
    mood = ""
    try:
        metadata = json.loads(metadata_text)
        dominant_colors = metadata.get("dominant_colors", [])
        mood = metadata.get("mood", "")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse metadata JSON: %s", metadata_text)

    total_usage = TokenUsage(
        prompt_tokens=expansion_usage.prompt_tokens + metadata_usage.prompt_tokens,
        completion_tokens=expansion_usage.completion_tokens
        + metadata_usage.completion_tokens,
    )

    return SceneExpansion(
        description=description,
        dominant_colors=dominant_colors,
        mood=mood,
        token_usage=total_usage,
    )
