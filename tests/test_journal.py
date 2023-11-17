"""Tests for Markdown journal generation."""

import pytest

from dreamink.journal import append_to_journal, rebuild_journal, render_entry
from dreamink.models import DreamEntry, Illustration, JournalDatabase, TokenUsage


@pytest.fixture
def full_entry():
    return DreamEntry(
        id="2023-11-15-001",
        date="2023-11-15",
        created_at="2023-11-15T19:45:12Z",
        raw_notes="was in my childhood kitchen but it was underwater. fish swimming through the cabinets.",
        scene_description="A childhood kitchen submerged in clear, still water. Cabinet doors drift open as silver fish navigate the shelves.",
        tags=["water", "childhood-home"],
        mood="peaceful",
        illustrations=[
            Illustration(
                style="watercolor",
                image_path="journal/2023-11/images/2023-11-15_watercolor.png",
                thumb_path="journal/2023-11/images/thumbs/2023-11-15_watercolor.png",
                revised_prompt="A watercolor illustration of...",
                generated_at="2023-11-15T19:45:12Z",
                cost_usd=0.08,
            )
        ],
    )


@pytest.fixture
def no_illustration_entry():
    return DreamEntry(
        id="2023-11-16-001",
        date="2023-11-16",
        raw_notes="violent storm destroying everything",
        scene_description="A massive storm engulfing a cityscape.",
        tags=["weather", "destruction"],
        mood="anxious",
    )


@pytest.fixture
def multi_illustration_entry():
    return DreamEntry(
        id="2023-11-17-001",
        date="2023-11-17",
        raw_notes="flying over a city made of books",
        scene_description="A vast city built entirely from open books.",
        tags=["flight", "urban"],
        mood="exhilarating",
        illustrations=[
            Illustration(
                style="watercolor",
                image_path="journal/2023-11/images/2023-11-17_watercolor.png",
                cost_usd=0.08,
            ),
            Illustration(
                style="manga",
                image_path="journal/2023-11/images/2023-11-17_manga.png",
                cost_usd=0.08,
            ),
        ],
    )


class TestRenderEntry:
    def test_full_entry_has_date_header(self, full_entry):
        md = render_entry(full_entry)
        assert "## November 15, 2023" in md

    def test_full_entry_has_raw_notes(self, full_entry):
        md = render_entry(full_entry)
        assert "**Raw notes:** was in my childhood kitchen" in md

    def test_full_entry_has_scene(self, full_entry):
        md = render_entry(full_entry)
        assert "**Scene:** A childhood kitchen submerged" in md

    def test_full_entry_has_style(self, full_entry):
        md = render_entry(full_entry)
        # Should show the style label (from styles.toml or fallback)
        assert "**Style:**" in md

    def test_full_entry_has_tags_as_inline_code(self, full_entry):
        md = render_entry(full_entry)
        assert "`water`" in md
        assert "`childhood-home`" in md

    def test_full_entry_has_image_embed(self, full_entry):
        md = render_entry(full_entry)
        assert "![dream-2023-11-15-watercolor]" in md
        assert "journal/2023-11/images/2023-11-15_watercolor.png" in md

    def test_full_entry_has_horizontal_rule(self, full_entry):
        md = render_entry(full_entry)
        assert "---" in md

    def test_no_illustration_shows_unavailable(self, no_illustration_entry):
        md = render_entry(no_illustration_entry)
        assert "*[illustration unavailable]*" in md
        assert "**Scene:**" in md
        # Should still have tags
        assert "`weather`" in md

    def test_multi_illustration_shows_all(self, multi_illustration_entry):
        md = render_entry(multi_illustration_entry)
        assert "2023-11-17_watercolor.png" in md
        assert "2023-11-17_manga.png" in md
        # Should have multiple style labels
        assert md.count("**Style:**") == 2

    def test_entry_with_no_tags(self):
        entry = DreamEntry(
            id="2023-11-18-001",
            date="2023-11-18",
            raw_notes="a simple dream",
            scene_description="A simple scene.",
            illustrations=[
                Illustration(
                    style="watercolor",
                    image_path="journal/2023-11/images/2023-11-18_watercolor.png",
                    cost_usd=0.08,
                )
            ],
        )
        md = render_entry(entry)
        assert "**Tags:**" not in md
        assert "**Scene:**" in md


class TestAppendToJournal:
    def test_creates_file_if_not_exists(self, tmp_path, full_entry):
        journal_path = str(tmp_path / "dream_journal.md")
        append_to_journal(full_entry, journal_path)

        content = (tmp_path / "dream_journal.md").read_text()
        assert content.startswith("# Dream Journal")
        assert "## November 15, 2023" in content

    def test_appends_to_existing_file(self, tmp_path, full_entry, no_illustration_entry):
        journal_path = str(tmp_path / "dream_journal.md")
        append_to_journal(full_entry, journal_path)
        append_to_journal(no_illustration_entry, journal_path)

        content = (tmp_path / "dream_journal.md").read_text()
        assert "## November 15, 2023" in content
        assert "## November 16, 2023" in content

    def test_creates_parent_dirs(self, tmp_path, full_entry):
        journal_path = str(tmp_path / "nested" / "dir" / "journal.md")
        append_to_journal(full_entry, journal_path)
        assert (tmp_path / "nested" / "dir" / "journal.md").exists()


class TestRebuildJournal:
    def test_sorts_by_date_descending(self, tmp_path, full_entry, no_illustration_entry):
        db = JournalDatabase(entries=[full_entry, no_illustration_entry])
        journal_path = str(tmp_path / "dream_journal.md")
        rebuild_journal(db, journal_path)

        content = (tmp_path / "dream_journal.md").read_text()
        pos_nov16 = content.index("November 16")
        pos_nov15 = content.index("November 15")
        assert pos_nov16 < pos_nov15  # Nov 16 appears first (newer)

    def test_rebuilds_from_scratch(self, tmp_path, full_entry):
        journal_path = str(tmp_path / "dream_journal.md")

        # Write some garbage first
        (tmp_path / "dream_journal.md").write_text("old content")

        db = JournalDatabase(entries=[full_entry])
        rebuild_journal(db, journal_path)

        content = (tmp_path / "dream_journal.md").read_text()
        assert "old content" not in content
        assert "# Dream Journal" in content
        assert "## November 15, 2023" in content

    def test_empty_database(self, tmp_path):
        db = JournalDatabase()
        journal_path = str(tmp_path / "dream_journal.md")
        rebuild_journal(db, journal_path)

        content = (tmp_path / "dream_journal.md").read_text()
        assert content == "# Dream Journal\n\n"
