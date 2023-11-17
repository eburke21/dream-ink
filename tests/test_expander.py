"""Tests for scene expansion module."""

from unittest.mock import MagicMock, patch

import pytest

from dreamink.config import AppConfig
from dreamink.expander import (
    SCENE_EXPANSION_SYSTEM_PROMPT,
    expand_dream_notes,
)
from dreamink.models import SceneExpansion


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


def _make_chat_response(content, prompt_tokens=100, completion_tokens=50):
    """Create a mock chat completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestExpandDreamNotes:
    def test_returns_scene_expansion(self, mock_config):
        expansion_resp = _make_chat_response(
            "A childhood kitchen submerged in clear, still water. "
            "Cabinet doors drift open as silver fish navigate the shelves. "
            "Amber underwater light fills the room with warmth.",
            prompt_tokens=150,
            completion_tokens=60,
        )
        metadata_resp = _make_chat_response(
            '{"dominant_colors": ["amber", "silver"], "mood": "peaceful"}',
            prompt_tokens=80,
            completion_tokens=20,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp,
            metadata_resp,
        ]

        result = expand_dream_notes(
            "underwater kitchen dream with fish",
            client=mock_client,
            config=mock_config,
        )

        assert isinstance(result, SceneExpansion)
        assert "kitchen" in result.description.lower()
        assert result.dominant_colors == ["amber", "silver"]
        assert result.mood == "peaceful"
        assert result.token_usage.prompt_tokens == 230  # 150 + 80
        assert result.token_usage.completion_tokens == 80  # 60 + 20

    def test_uses_correct_system_prompt(self, mock_config):
        expansion_resp = _make_chat_response("A vivid scene.")
        metadata_resp = _make_chat_response('{"dominant_colors": [], "mood": ""}')
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp,
            metadata_resp,
        ]

        expand_dream_notes("test", client=mock_client, config=mock_config)

        call_args = mock_client.chat.completions.create.call_args_list[0]
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SCENE_EXPANSION_SYSTEM_PROMPT
        assert "PRESERVE surreal" in messages[0]["content"]

    def test_uses_configured_model_and_temperature(self, mock_config):
        expansion_resp = _make_chat_response("A scene.")
        metadata_resp = _make_chat_response('{"dominant_colors": [], "mood": ""}')
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp,
            metadata_resp,
        ]

        expand_dream_notes("test", client=mock_client, config=mock_config)

        call_args = mock_client.chat.completions.create.call_args_list[0]
        assert call_args.kwargs["model"] == "gpt-4-0613"
        assert call_args.kwargs["temperature"] == 0.9

    def test_emotional_tone_appended_to_user_message(self, mock_config):
        expansion_resp = _make_chat_response("A scene.")
        metadata_resp = _make_chat_response('{"dominant_colors": [], "mood": ""}')
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp,
            metadata_resp,
        ]

        expand_dream_notes(
            "some dream",
            emotional_tone="anxious",
            client=mock_client,
            config=mock_config,
        )

        call_args = mock_client.chat.completions.create.call_args_list[0]
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "[Emotional tone: anxious]" in user_msg

    def test_handles_malformed_metadata_json(self, mock_config):
        expansion_resp = _make_chat_response("A vivid scene description.")
        metadata_resp = _make_chat_response("not valid json {{{")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp,
            metadata_resp,
        ]

        result = expand_dream_notes("test", client=mock_client, config=mock_config)

        assert result.description == "A vivid scene description."
        assert result.dominant_colors == []
        assert result.mood == ""

    def test_three_different_dreams(self, mock_config):
        """Feed 3 sample dreams and verify each produces a valid SceneExpansion."""
        dreams = [
            "underwater kitchen, fish in cabinets, mom made of stained glass",
            "flying over city made of books, pages fluttering as wings",
            "forest where all the trees were clocks, ticking sound everywhere",
        ]
        for dream in dreams:
            expansion_resp = _make_chat_response(f"Scene for: {dream}")
            metadata_resp = _make_chat_response(
                '{"dominant_colors": ["blue"], "mood": "dreamy"}'
            )
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                expansion_resp,
                metadata_resp,
            ]

            result = expand_dream_notes(
                dream, client=mock_client, config=mock_config
            )
            assert isinstance(result, SceneExpansion)
            assert len(result.description) > 0
