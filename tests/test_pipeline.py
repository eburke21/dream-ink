"""Tests for the pipeline orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from dreamink.config import AppConfig
from dreamink.models import DreamEntry, TokenUsage
from dreamink.pipeline import process_dream


@pytest.fixture
def mock_config(tmp_path):
    db_path = str(tmp_path / "dreamink.json")
    output_path = str(tmp_path / "journal")
    return AppConfig(
        api_key_env="OPENAI_API_KEY",
        expansion_model="gpt-4-0613",
        expansion_temperature=0.9,
        max_retries=0,
        image_size="1792x1024",
        image_quality="hd",
        cost_per_image_usd=0.08,
        output_path=output_path,
        database_path=db_path,
        default_style="watercolor",
    )


def _make_chat_response(content, prompt_tokens=100, completion_tokens=50):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def _make_image_response(url="https://example.com/image.png", revised_prompt="revised"):
    response = MagicMock()
    response.data = [MagicMock()]
    response.data[0].url = url
    response.data[0].revised_prompt = revised_prompt
    return response


class TestProcessDream:
    @patch("dreamink.pipeline.download_and_save")
    @patch("dreamink.pipeline.get_styles")
    def test_full_pipeline_returns_entry(self, mock_styles, mock_download, mock_config, tmp_path):
        from dreamink.models import StyleConfig

        mock_styles.return_value = {
            "watercolor": StyleConfig(
                name="watercolor", label="Watercolor", suffix="watercolor style"
            )
        }

        mock_download.return_value = MagicMock(
            image_path=str(tmp_path / "image.png"),
            thumb_path=str(tmp_path / "thumb.png"),
            file_size=1024,
        )

        expansion_resp = _make_chat_response("A vivid dream scene description.", 150, 60)
        metadata_resp = _make_chat_response('{"dominant_colors": ["blue"], "mood": "peaceful"}', 80, 20)
        tag_resp = _make_chat_response('["water", "light"]', 50, 10)
        image_resp = _make_image_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp,
            metadata_resp,
            tag_resp,
        ]
        mock_client.images.generate.return_value = image_resp

        entry = process_dream(
            raw_notes="underwater kitchen dream with fish in cabinets and peaceful feeling",
            date="2023-11-15",
            config=mock_config,
            client=mock_client,
        )

        assert isinstance(entry, DreamEntry)
        assert entry.date == "2023-11-15"
        assert entry.id == "2023-11-15-001"
        assert "dream scene" in entry.scene_description.lower()
        assert entry.tags == ["water", "light"]
        assert entry.mood == "peaceful"
        assert len(entry.illustrations) == 1
        assert entry.illustrations[0].style == "watercolor"

    @patch("dreamink.pipeline.download_and_save")
    @patch("dreamink.pipeline.get_styles")
    def test_content_policy_rejection_saves_entry_without_image(
        self, mock_styles, mock_download, mock_config
    ):
        import openai
        from dreamink.models import StyleConfig

        mock_styles.return_value = {
            "watercolor": StyleConfig(
                name="watercolor", label="Watercolor", suffix="watercolor style"
            )
        }

        expansion_resp = _make_chat_response("A scene description.", 150, 60)
        metadata_resp = _make_chat_response('{"dominant_colors": [], "mood": "calm"}', 80, 20)
        tag_resp = _make_chat_response('["water"]', 50, 10)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp,
            metadata_resp,
            tag_resp,
        ]

        # Simulate content policy rejection
        error_body = {"error": {"code": "content_policy_violation", "message": "rejected"}}
        mock_client.images.generate.side_effect = openai.BadRequestError(
            message="rejected",
            response=MagicMock(status_code=400, headers={}),
            body=error_body,
        )

        entry = process_dream(
            raw_notes="a dream about a storm with lightning and destruction everywhere",
            date="2023-11-15",
            config=mock_config,
            client=mock_client,
        )

        assert entry.scene_description == "A scene description."
        assert entry.illustrations == []
        mock_download.assert_not_called()

    @patch("dreamink.pipeline.download_and_save")
    @patch("dreamink.pipeline.get_styles")
    def test_long_notes_truncated(self, mock_styles, mock_download, mock_config, tmp_path):
        from dreamink.models import StyleConfig

        mock_styles.return_value = {
            "watercolor": StyleConfig(
                name="watercolor", label="Watercolor", suffix="watercolor style"
            )
        }
        mock_download.return_value = MagicMock(
            image_path=str(tmp_path / "image.png"),
            thumb_path=str(tmp_path / "thumb.png"),
            file_size=1024,
        )

        expansion_resp = _make_chat_response("A truncated scene.", 150, 60)
        metadata_resp = _make_chat_response('{"dominant_colors": [], "mood": ""}', 80, 20)
        tag_resp = _make_chat_response('["water"]', 50, 10)
        image_resp = _make_image_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp, metadata_resp, tag_resp,
        ]
        mock_client.images.generate.return_value = image_resp

        # 600 words input
        long_notes = " ".join(["word"] * 600)
        entry = process_dream(
            raw_notes=long_notes,
            date="2023-11-15",
            config=mock_config,
            client=mock_client,
        )

        # The entry should be created (truncation logged but not blocking)
        assert isinstance(entry, DreamEntry)

    @patch("dreamink.pipeline.download_and_save")
    @patch("dreamink.pipeline.get_styles")
    def test_cost_tracking(self, mock_styles, mock_download, mock_config, tmp_path):
        from dreamink.models import StyleConfig

        mock_styles.return_value = {
            "watercolor": StyleConfig(
                name="watercolor", label="Watercolor", suffix="watercolor style"
            )
        }
        mock_download.return_value = MagicMock(
            image_path=str(tmp_path / "image.png"),
            thumb_path=str(tmp_path / "thumb.png"),
            file_size=1024,
        )

        expansion_resp = _make_chat_response("A scene.", 200, 150)
        metadata_resp = _make_chat_response('{"dominant_colors": [], "mood": ""}', 100, 30)
        tag_resp = _make_chat_response('["water"]', 200, 30)
        image_resp = _make_image_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp, metadata_resp, tag_resp,
        ]
        mock_client.images.generate.return_value = image_resp

        entry = process_dream(
            raw_notes="underwater kitchen dream with fish in cabinets and peaceful light",
            date="2023-11-15",
            config=mock_config,
            client=mock_client,
        )

        assert "expansion" in entry.token_usage
        assert "tagging" in entry.token_usage
        assert entry.token_usage["expansion"].prompt_tokens == 300  # 200 + 100
        assert entry.token_usage["tagging"].prompt_tokens == 200
        assert entry.illustrations[0].cost_usd == 0.08

    @patch("dreamink.pipeline.get_styles")
    def test_unknown_style_raises(self, mock_styles, mock_config):
        mock_styles.return_value = {"watercolor": MagicMock()}
        mock_client = MagicMock()

        with pytest.raises(ValueError, match="Unknown style"):
            process_dream(
                raw_notes="a dream about flying over mountains and valleys with eagles",
                date="2023-11-15",
                style_name="nonexistent",
                config=mock_config,
                client=mock_client,
            )

    @patch("dreamink.pipeline.download_and_save")
    @patch("dreamink.pipeline.get_styles")
    def test_sequence_number_increments(self, mock_styles, mock_download, mock_config, tmp_path):
        from dreamink.models import StyleConfig
        from dreamink.storage import add_entry, load_database, save_database

        mock_styles.return_value = {
            "watercolor": StyleConfig(
                name="watercolor", label="Watercolor", suffix="watercolor style"
            )
        }
        mock_download.return_value = MagicMock(
            image_path=str(tmp_path / "image.png"),
            thumb_path=str(tmp_path / "thumb.png"),
            file_size=1024,
        )

        # Pre-populate database with one entry for the same date
        db = load_database(mock_config.database_path)
        existing = DreamEntry(
            id="2023-11-15-001", date="2023-11-15",
            raw_notes="first dream about the ocean and waves and sand",
        )
        db = add_entry(db, existing)
        save_database(db, mock_config.database_path)

        expansion_resp = _make_chat_response("A scene.", 100, 50)
        metadata_resp = _make_chat_response('{"dominant_colors": [], "mood": ""}', 80, 20)
        tag_resp = _make_chat_response('["water"]', 50, 10)
        image_resp = _make_image_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            expansion_resp, metadata_resp, tag_resp,
        ]
        mock_client.images.generate.return_value = image_resp

        entry = process_dream(
            raw_notes="second dream about flying over clouds and mountains and birds",
            date="2023-11-15",
            config=mock_config,
            client=mock_client,
        )

        assert entry.id == "2023-11-15-002"
