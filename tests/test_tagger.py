"""Tests for tag extraction module."""

from unittest.mock import MagicMock

import pytest

from dreamink.config import AppConfig
from dreamink.tagger import (
    TAG_EXTRACTION_PROMPT,
    _parse_tags,
    _regex_extract_tags,
    _validate_and_dedupe,
    extract_tags,
)


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


def _make_chat_response(content, prompt_tokens=50, completion_tokens=20):
    """Create a mock chat completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestExtractTags:
    def test_returns_tags_and_usage(self, mock_config):
        resp = _make_chat_response('["water", "childhood-home", "flight"]')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        tags, usage = extract_tags(
            "A kitchen submerged in water with fish swimming through cabinets.",
            client=mock_client,
            config=mock_config,
        )

        assert tags == ["water", "childhood-home", "flight"]
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 20

    def test_uses_tag_extraction_prompt(self, mock_config):
        resp = _make_chat_response('["water"]')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        extract_tags("some scene", client=mock_client, config=mock_config)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == TAG_EXTRACTION_PROMPT

    def test_uses_low_temperature(self, mock_config):
        resp = _make_chat_response('["water"]')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        extract_tags("some scene", client=mock_client, config=mock_config)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.3

    def test_scene_description_is_user_message(self, mock_config):
        resp = _make_chat_response('["water"]')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        extract_tags("A vivid underwater scene.", client=mock_client, config=mock_config)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "A vivid underwater scene."

    def test_fallback_to_regex_on_bad_json(self, mock_config):
        resp = _make_chat_response('Here are some tags: "water", "flight", "transformation"')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        tags, _ = extract_tags("some scene", client=mock_client, config=mock_config)

        assert "water" in tags
        assert "flight" in tags
        assert "transformation" in tags

    def test_limits_to_five_tags(self, mock_config):
        resp = _make_chat_response(
            '["water", "flight", "childhood-home", "transformation", "light", "animals", "forest"]'
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        tags, _ = extract_tags("some scene", client=mock_client, config=mock_config)

        assert len(tags) <= 5

    def test_deduplicates_tags(self, mock_config):
        resp = _make_chat_response('["water", "water", "flight", "flight"]')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp

        tags, _ = extract_tags("some scene", client=mock_client, config=mock_config)

        assert tags == ["water", "flight"]


class TestParseTags:
    def test_valid_json_array(self):
        assert _parse_tags('["water", "flight"]') == ["water", "flight"]

    def test_empty_array(self):
        assert _parse_tags("[]") == []

    def test_invalid_json(self):
        assert _parse_tags("not json") == []

    def test_non_array_json(self):
        assert _parse_tags('{"tag": "water"}') == []

    def test_mixed_types_coerced_to_string(self):
        result = _parse_tags('[1, "water", true]')
        assert result == ["1", "water", "true"]


class TestRegexExtractTags:
    def test_extracts_quoted_tags(self):
        text = 'The tags are "water", "childhood-home", and "flight".'
        result = _regex_extract_tags(text)
        assert "water" in result
        assert "childhood-home" in result
        assert "flight" in result

    def test_falls_back_to_comma_separated(self):
        text = "water, flight, transformation"
        result = _regex_extract_tags(text)
        assert "water" in result
        assert "flight" in result
        assert "transformation" in result

    def test_filters_short_and_common_words(self):
        text = "the water and flight for joy"
        result = _regex_extract_tags(text)
        assert "the" not in result
        assert "and" not in result
        assert "for" not in result
        assert "water" in result


class TestValidateAndDedupe:
    def test_valid_tags_pass(self):
        assert _validate_and_dedupe(["water", "flight"]) == ["water", "flight"]

    def test_removes_duplicates(self):
        assert _validate_and_dedupe(["water", "water"]) == ["water"]

    def test_limits_to_five(self):
        tags = ["a", "b", "c", "d", "e", "f", "g"]
        assert len(_validate_and_dedupe(tags)) == 5

    def test_rejects_invalid_format(self):
        # "UPPER" lowercases to "upper" which is valid; "has space" is rejected
        assert _validate_and_dedupe(["valid", "UPPER", "has space", "ok-tag"]) == ["valid", "upper", "ok-tag"]

    def test_strips_and_lowercases(self):
        assert _validate_and_dedupe(["  Water  ", " FLIGHT "]) == ["water", "flight"]

    def test_empty_input(self):
        assert _validate_and_dedupe([]) == []
