"""GPT-4 tag extraction: scene descriptions -> thematic dream tags."""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from dreamink.config import AppConfig, get_config
from dreamink.models import TokenUsage
from dreamink.utils import extract_token_usage, get_openai_client, retry_api_call

logger = logging.getLogger(__name__)

TAG_EXTRACTION_PROMPT = """Extract 3-5 thematic tags from this dream scene description.
Tags should capture recurring dream motifs, not plot details.

Good tags: water, childhood-home, flight, transformation, being-lost, light, animals
Bad tags: kitchen, tuesday, blue-shirt (too specific/literal)

Return ONLY a JSON array of lowercase hyphenated strings. Example: ["water", "childhood-home", "flight"]"""

_TAG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def extract_tags(
    scene_description: str,
    client: OpenAI | None = None,
    config: AppConfig | None = None,
) -> tuple[list[str], TokenUsage]:
    """Extract thematic dream tags from a scene description.

    Args:
        scene_description: Expanded scene from the expander module.
        client: OpenAI client. If None, creates one from config.
        config: App config. If None, loads defaults.

    Returns:
        Tuple of (tags list, token_usage).
    """
    config = config or get_config()
    client = client or get_openai_client(config)

    response = retry_api_call(
        lambda: client.chat.completions.create(
            model=config.expansion_model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": TAG_EXTRACTION_PROMPT},
                {"role": "user", "content": scene_description},
            ],
        ),
        max_retries=config.max_retries,
    )

    usage = extract_token_usage(response)
    raw_text = response.choices[0].message.content.strip()

    tags = _parse_tags(raw_text)

    if not tags:
        # Fallback: regex extraction from response text
        logger.warning("JSON parse failed, falling back to regex: %s", raw_text)
        tags = _regex_extract_tags(raw_text)

    # Validate, deduplicate, limit to 5
    tags = _validate_and_dedupe(tags)

    return tags, usage


def _parse_tags(text: str) -> list[str]:
    """Try to parse tags from JSON array."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(t).strip().lower() for t in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _regex_extract_tags(text: str) -> list[str]:
    """Extract tags from text using regex as fallback."""
    # Look for quoted strings that look like tags
    matches = re.findall(r'"([a-z0-9]+(?:-[a-z0-9]+)*)"', text.lower())
    if matches:
        return matches
    # Try comma-separated words
    words = re.findall(r'[a-z]+(?:-[a-z]+)*', text.lower())
    return [w for w in words if len(w) > 2 and w not in ("the", "and", "for", "with", "from")]


def _validate_and_dedupe(tags: list[str]) -> list[str]:
    """Validate tag format, deduplicate, limit to 5."""
    seen = set()
    valid = []
    for tag in tags:
        tag = tag.strip().lower()
        if _TAG_PATTERN.match(tag) and tag not in seen:
            seen.add(tag)
            valid.append(tag)
        if len(valid) >= 5:
            break
    return valid
