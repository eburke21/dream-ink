"""Tests for DALL-E 3 image generation module."""

from unittest.mock import MagicMock

import openai
import pytest

from dreamink.config import AppConfig
from dreamink.illustrator import _MAX_PROMPT_LENGTH, generate_illustration
from dreamink.models import GeneratedImage, StyleConfig


@pytest.fixture
def mock_config():
    return AppConfig(
        api_key_env="OPENAI_API_KEY",
        expansion_model="gpt-4-0613",
        expansion_temperature=0.9,
        max_retries=0,
        image_size="1792x1024",
        image_quality="hd",
        cost_per_image_usd=0.08,
        output_path="journal",
        database_path="data/dreamink.json",
        default_style="watercolor",
    )


@pytest.fixture
def watercolor_style():
    return StyleConfig(
        name="watercolor",
        label="Watercolor Journal",
        suffix="loose watercolor illustration with visible brushstrokes",
    )


def _make_image_response(url="https://example.com/image.png", revised_prompt="revised"):
    response = MagicMock()
    response.data = [MagicMock()]
    response.data[0].url = url
    response.data[0].revised_prompt = revised_prompt
    return response


class TestGenerateIllustration:
    def test_constructs_correct_prompt(self, mock_config, watercolor_style):
        mock_client = MagicMock()
        mock_client.images.generate.return_value = _make_image_response()

        generate_illustration(
            "A vivid scene.", watercolor_style, client=mock_client, config=mock_config
        )

        call_args = mock_client.images.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert prompt == "A vivid scene.. loose watercolor illustration with visible brushstrokes"

    def test_prompt_format_is_scene_dot_space_suffix(self, mock_config, watercolor_style):
        mock_client = MagicMock()
        mock_client.images.generate.return_value = _make_image_response()

        scene = "A kitchen submerged underwater"
        generate_illustration(scene, watercolor_style, client=mock_client, config=mock_config)

        prompt = mock_client.images.generate.call_args.kwargs["prompt"]
        assert prompt.startswith(scene + ". ")
        assert prompt.endswith(watercolor_style.suffix)

    def test_prompt_under_4000_chars(self, mock_config, watercolor_style):
        mock_client = MagicMock()
        mock_client.images.generate.return_value = _make_image_response()

        long_scene = "A" * 5000
        generate_illustration(long_scene, watercolor_style, client=mock_client, config=mock_config)

        prompt = mock_client.images.generate.call_args.kwargs["prompt"]
        assert len(prompt) <= _MAX_PROMPT_LENGTH

    def test_captures_revised_prompt(self, mock_config, watercolor_style):
        mock_client = MagicMock()
        mock_client.images.generate.return_value = _make_image_response(
            revised_prompt="DALL-E rewrote this prompt"
        )

        result = generate_illustration(
            "A scene.", watercolor_style, client=mock_client, config=mock_config
        )

        assert result.revised_prompt == "DALL-E rewrote this prompt"

    def test_returns_generated_image(self, mock_config, watercolor_style):
        mock_client = MagicMock()
        mock_client.images.generate.return_value = _make_image_response(
            url="https://cdn.openai.com/image123.png"
        )

        result = generate_illustration(
            "A scene.", watercolor_style, client=mock_client, config=mock_config
        )

        assert isinstance(result, GeneratedImage)
        assert result.url == "https://cdn.openai.com/image123.png"
        assert result.rejected is False
        assert result.generation_metadata["style_name"] == "watercolor"

    def test_content_policy_rejection_without_retry(self, mock_config, watercolor_style):
        mock_client = MagicMock()

        error_body = {
            "error": {
                "code": "content_policy_violation",
                "message": "Your request was rejected by content policy.",
            }
        }
        error = openai.BadRequestError(
            message="Bad request",
            response=MagicMock(status_code=400, headers={}),
            body=error_body,
        )
        mock_client.images.generate.side_effect = error

        result = generate_illustration(
            "A violent scene.", watercolor_style, client=mock_client, config=mock_config,
            allow_sanitize_retry=False,
        )

        assert result.rejected is True
        assert "content policy" in result.rejection_reason.lower()
        assert result.url == ""

    def test_content_policy_sanitize_retry_succeeds(self, mock_config, watercolor_style):
        mock_client = MagicMock()

        error_body = {
            "error": {
                "code": "content_policy_violation",
                "message": "Rejected",
            }
        }
        error = openai.BadRequestError(
            message="Bad request",
            response=MagicMock(status_code=400, headers={}),
            body=error_body,
        )

        # First images.generate call fails, sanitize call succeeds, second images.generate succeeds
        mock_client.images.generate.side_effect = [
            error,
            _make_image_response(url="https://example.com/sanitized.png"),
        ]
        # Chat call for sanitization
        sanitize_resp = MagicMock()
        sanitize_resp.choices = [MagicMock()]
        sanitize_resp.choices[0].message.content = "A gentle scene."
        sanitize_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_client.chat.completions.create.return_value = sanitize_resp

        result = generate_illustration(
            "A violent scene.", watercolor_style, client=mock_client, config=mock_config,
            allow_sanitize_retry=True,
        )

        assert result.rejected is False
        assert result.url == "https://example.com/sanitized.png"

    def test_non_content_policy_error_raises(self, mock_config, watercolor_style):
        mock_client = MagicMock()

        error = openai.BadRequestError(
            message="Invalid size",
            response=MagicMock(status_code=400, headers={}),
            body={"error": {"code": "invalid_size", "message": "Invalid size"}},
        )
        mock_client.images.generate.side_effect = error

        with pytest.raises(openai.BadRequestError):
            generate_illustration(
                "A scene.", watercolor_style, client=mock_client, config=mock_config
            )

    def test_passes_correct_dalle_params(self, mock_config, watercolor_style):
        mock_client = MagicMock()
        mock_client.images.generate.return_value = _make_image_response()

        generate_illustration(
            "A scene.", watercolor_style, client=mock_client, config=mock_config
        )

        call_args = mock_client.images.generate.call_args.kwargs
        assert call_args["model"] == "dall-e-3"
        assert call_args["size"] == "1792x1024"
        assert call_args["quality"] == "hd"
        assert call_args["n"] == 1
