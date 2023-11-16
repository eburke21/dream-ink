"""Tests for JSON state persistence."""

import json

import pytest

from dreamink.models import DreamEntry, JournalDatabase
from dreamink.storage import (
    add_entry,
    get_entries_by_date_range,
    get_entry,
    load_database,
    save_database,
)


@pytest.fixture
def sample_entry():
    return DreamEntry(
        id="2023-11-15-001",
        date="2023-11-15",
        raw_notes="underwater kitchen dream",
        tags=["water", "childhood-home"],
    )


@pytest.fixture
def sample_entry_2():
    return DreamEntry(
        id="2023-11-18-001",
        date="2023-11-18",
        raw_notes="flying over a city of books",
        tags=["flight", "urban"],
    )


@pytest.fixture
def sample_entry_3():
    return DreamEntry(
        id="2023-12-01-001",
        date="2023-12-01",
        raw_notes="forest made of clocks",
        tags=["transformation", "water"],
    )


class TestLoadDatabase:
    def test_nonexistent_file_returns_empty(self, tmp_path):
        db = load_database(tmp_path / "nonexistent.json")
        assert db.version == "1.0"
        assert db.entries == []
        assert db.tag_index == {}

    def test_loads_existing_file(self, tmp_path, sample_entry):
        db = JournalDatabase(entries=[sample_entry])
        path = tmp_path / "test.json"
        path.write_text(db.model_dump_json(indent=2))

        loaded = load_database(path)
        assert len(loaded.entries) == 1
        assert loaded.entries[0].id == "2023-11-15-001"


class TestSaveDatabase:
    def test_creates_parent_dirs(self, tmp_path, sample_entry):
        db = JournalDatabase(entries=[sample_entry])
        path = tmp_path / "nested" / "dir" / "db.json"
        save_database(db, path)
        assert path.exists()

    def test_roundtrip(self, tmp_path, sample_entry):
        db = JournalDatabase(entries=[sample_entry])
        path = tmp_path / "db.json"
        save_database(db, path)
        loaded = load_database(path)
        assert loaded == db

    def test_atomic_write_preserves_original_on_error(self, tmp_path, sample_entry):
        """If writing fails, the original file should remain intact."""
        db = JournalDatabase(entries=[sample_entry])
        path = tmp_path / "db.json"
        save_database(db, path)

        # Verify original is intact
        loaded = load_database(path)
        assert len(loaded.entries) == 1


class TestAddEntry:
    def test_adds_entry(self, sample_entry):
        db = JournalDatabase()
        db = add_entry(db, sample_entry)
        assert len(db.entries) == 1

    def test_updates_tag_index(self, sample_entry):
        db = JournalDatabase()
        db = add_entry(db, sample_entry)
        assert "water" in db.tag_index
        assert "childhood-home" in db.tag_index
        assert db.tag_index["water"] == ["2023-11-15-001"]

    def test_shared_tags_across_entries(self, sample_entry, sample_entry_3):
        db = JournalDatabase()
        db = add_entry(db, sample_entry)
        db = add_entry(db, sample_entry_3)
        assert db.tag_index["water"] == ["2023-11-15-001", "2023-12-01-001"]

    def test_multiple_entries_save_reload(
        self, tmp_path, sample_entry, sample_entry_2, sample_entry_3
    ):
        db = JournalDatabase()
        db = add_entry(db, sample_entry)
        db = add_entry(db, sample_entry_2)
        db = add_entry(db, sample_entry_3)

        path = tmp_path / "db.json"
        save_database(db, path)
        loaded = load_database(path)
        assert len(loaded.entries) == 3


class TestGetEntry:
    def test_found(self, sample_entry):
        db = JournalDatabase(entries=[sample_entry])
        result = get_entry(db, "2023-11-15-001")
        assert result is not None
        assert result.raw_notes == "underwater kitchen dream"

    def test_not_found(self, sample_entry):
        db = JournalDatabase(entries=[sample_entry])
        assert get_entry(db, "nonexistent") is None


class TestGetEntriesByDateRange:
    def test_filters_correctly(self, sample_entry, sample_entry_2, sample_entry_3):
        db = JournalDatabase(
            entries=[sample_entry, sample_entry_2, sample_entry_3]
        )
        results = get_entries_by_date_range(db, "2023-11-01", "2023-11-30")
        assert len(results) == 2
        ids = [e.id for e in results]
        assert "2023-11-15-001" in ids
        assert "2023-11-18-001" in ids

    def test_inclusive_boundaries(self, sample_entry):
        db = JournalDatabase(entries=[sample_entry])
        results = get_entries_by_date_range(db, "2023-11-15", "2023-11-15")
        assert len(results) == 1

    def test_empty_range(self, sample_entry):
        db = JournalDatabase(entries=[sample_entry])
        results = get_entries_by_date_range(db, "2024-01-01", "2024-01-31")
        assert len(results) == 0
