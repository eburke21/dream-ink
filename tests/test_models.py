"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from dreamink.models import (
    DreamEntry,
    Illustration,
    JournalDatabase,
    SceneExpansion,
    StyleConfig,
    TokenUsage,
)


class TestTokenUsage:
    def test_defaults(self):
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0

    def test_with_values(self):
        t = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert t.prompt_tokens == 100


class TestSceneExpansion:
    def test_minimal(self):
        s = SceneExpansion(description="A vivid scene")
        assert s.description == "A vivid scene"
        assert s.dominant_colors == []
        assert s.mood == ""

    def test_full(self):
        s = SceneExpansion(
            description="A vivid scene",
            dominant_colors=["amber", "blue"],
            mood="peaceful",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )
        assert len(s.dominant_colors) == 2
        assert s.token_usage.prompt_tokens == 100


class TestStyleConfig:
    def test_creation(self):
        s = StyleConfig(
            name="watercolor",
            label="Watercolor Journal",
            suffix="loose watercolor...",
        )
        assert s.name == "watercolor"
        assert s.best_for == ""


class TestIllustration:
    def test_valid(self):
        i = Illustration(style="watercolor", image_path="images/test.png")
        assert i.cost_usd == 0.0

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            Illustration(
                style="watercolor", image_path="images/test.png", cost_usd=-1.0
            )


class TestDreamEntry:
    def test_valid_entry(self):
        e = DreamEntry(
            id="2023-11-15-001",
            date="2023-11-15",
            raw_notes="underwater kitchen dream",
            tags=["water", "childhood-home"],
        )
        assert e.id == "2023-11-15-001"
        assert len(e.tags) == 2

    def test_invalid_date_format(self):
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            DreamEntry(
                id="test", date="Nov 15 2023", raw_notes="some notes"
            )

    def test_empty_notes_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            DreamEntry(id="test", date="2023-11-15", raw_notes="   ")

    def test_invalid_tag_format(self):
        with pytest.raises(ValidationError, match="lowercase-hyphenated"):
            DreamEntry(
                id="test",
                date="2023-11-15",
                raw_notes="some notes",
                tags=["Water"],
            )

    def test_tag_with_spaces_rejected(self):
        with pytest.raises(ValidationError, match="lowercase-hyphenated"):
            DreamEntry(
                id="test",
                date="2023-11-15",
                raw_notes="some notes",
                tags=["childhood home"],
            )

    def test_serialization_roundtrip(self):
        e = DreamEntry(
            id="2023-11-15-001",
            date="2023-11-15",
            raw_notes="underwater kitchen dream",
            tags=["water", "light"],
            token_usage={
                "expansion": TokenUsage(prompt_tokens=100, completion_tokens=50)
            },
        )
        json_str = e.model_dump_json()
        e2 = DreamEntry.model_validate_json(json_str)
        assert e == e2


class TestJournalDatabase:
    def test_empty_database(self):
        db = JournalDatabase()
        assert db.version == "1.0"
        assert db.entries == []
        assert db.tag_index == {}

    def test_serialization_roundtrip(self):
        db = JournalDatabase(
            entries=[
                DreamEntry(
                    id="2023-11-15-001",
                    date="2023-11-15",
                    raw_notes="test dream",
                )
            ],
            tag_index={"water": ["2023-11-15-001"]},
        )
        json_str = db.model_dump_json()
        db2 = JournalDatabase.model_validate_json(json_str)
        assert len(db2.entries) == 1
        assert db2.tag_index["water"] == ["2023-11-15-001"]
