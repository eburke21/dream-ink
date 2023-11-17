"""DALL-E 3 image generation from expanded scene descriptions."""

from __future__ import annotations

import logging

import openai
from openai import OpenAI

from dreamink.config import AppConfig, get_config
from dreamink.models import GeneratedImage, StyleConfig
from dreamink.utils import get_openai_client, retry_api_call

logger = logging.getLogger(__name__)

# DALL-E 3 prompt limit
_MAX_PROMPT_LENGTH = 4000

_SANITIZE_PROMPT = """Rewrite this dream scene description to remove any potentially
objectionable elements (violence, gore, nudity, distressing imagery) while preserving
the dream's structure, atmosphere, and surreal logic. Keep the same length (3-4 sentences)
and visual detail level. Return ONLY the rewritten description, no explanation."""


def sanitize_scene(
    scene_description: str,
    client: OpenAI,
    config: AppConfig,
) -> str:
    """Ask GPT-4 to rewrite a scene removing content-policy-triggering elements."""
    response = retry_api_call(
        lambda: client.chat.completions.create(
            model=config.expansion_model,
            temperature=0.5,
            messages=[
                {"role": "system", "content": _SANITIZE_PROMPT},
                {"role": "user", "content": scene_description},
            ],
        ),
        max_retries=config.max_retries,
    )
    return response.choices[0].message.content.strip()


def generate_illustration(
    scene_description: str,
    style: StyleConfig,
    client: OpenAI | None = None,
    config: AppConfig | None = None,
    allow_sanitize_retry: bool = True,
) -> GeneratedImage:
    """Generate a dream illustration via DALL-E 3.

    Constructs a prompt from the scene description + style suffix,
    calls DALL-E 3, and returns the result with the revised prompt.

    On content policy rejection, if allow_sanitize_retry is True, asks
    GPT-4 to sanitize the scene and retries once with the cleaned prompt.

    Args:
        scene_description: Expanded scene from the expander module.
        style: Style preset with suffix to append to the prompt.
        client: OpenAI client. If None, creates one from config.
        config: App config. If None, loads defaults.
        allow_sanitize_retry: If True, attempt one sanitized retry on
            content policy rejection.

    Returns:
        GeneratedImage with url, revised_prompt, and metadata.
        If content policy rejects the image, returns with rejected=True.
    """
    config = config or get_config()
    client = client or get_openai_client(config)

    prompt = f"{scene_description}. {style.suffix}"

    # Truncate scene if prompt exceeds DALL-E 3 limit
    if len(prompt) > _MAX_PROMPT_LENGTH:
        max_scene_len = _MAX_PROMPT_LENGTH - len(style.suffix) - 2  # ". " separator
        prompt = f"{scene_description[:max_scene_len]}. {style.suffix}"
        logger.warning(
            "Prompt truncated to %d chars (DALL-E 3 limit: %d)",
            len(prompt),
            _MAX_PROMPT_LENGTH,
        )

    try:
        response = retry_api_call(
            lambda: client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=config.image_size,
                quality=config.image_quality,
                n=1,
            ),
            max_retries=config.max_retries,
        )

        image_data = response.data[0]
        return GeneratedImage(
            url=image_data.url,
            revised_prompt=image_data.revised_prompt or "",
            generation_metadata={
                "model": "dall-e-3",
                "size": config.image_size,
                "quality": config.image_quality,
                "style_name": style.name,
                "original_prompt_length": len(prompt),
            },
        )

    except openai.BadRequestError as e:
        error_body = getattr(e, "body", {}) or {}
        error_code = ""
        error_message = str(e)
        if isinstance(error_body, dict):
            error_obj = error_body.get("error", {})
            if isinstance(error_obj, dict):
                error_code = error_obj.get("code", "")
                error_message = error_obj.get("message", str(e))

        if error_code == "content_policy_violation":
            logger.warning("Content policy rejection: %s", error_message)

            if allow_sanitize_retry:
                logger.info("Attempting sanitized retry...")
                try:
                    sanitized = sanitize_scene(scene_description, client, config)
                    return generate_illustration(
                        sanitized, style,
                        client=client,
                        config=config,
                        allow_sanitize_retry=False,
                    )
                except Exception as retry_err:
                    logger.warning("Sanitized retry also failed: %s", retry_err)

            return GeneratedImage(
                url="",
                rejected=True,
                rejection_reason=error_message,
                generation_metadata={
                    "model": "dall-e-3",
                    "style_name": style.name,
                    "error_code": error_code,
                },
            )
        raise
