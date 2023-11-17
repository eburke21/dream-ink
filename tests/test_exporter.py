"""Tests for HTML export module."""

import pytest

from dreamink.exporter import export_html, _render_entry_html
from dreamink.models import DreamEntry, Illustration, JournalDatabase


@pytest.fixture
def sample_entry():
    return DreamEntry(
        id="2023-11-15-001",
        date="2023-11-15",
        raw_notes="underwater kitchen dream",
        scene_description="A kitchen submerged in amber light.",
        tags=["water", "light"],
        mood="peaceful",
        illustrations=[
            Illustration(
                style="watercolor",
                image_path="nonexistent.png",
                thumb_path="",
                cost_usd=0.08,
            )
        ],
    )


@pytest.fixture
def styles():
    return {"watercolor": "Watercolor Journal", "manga": "Atmospheric Manga"}


class TestRenderEntryHtml:
    def test_contains_date_header(self, sample_entry, styles):
        html = _render_entry_html(sample_entry, styles)
        assert "November 15, 2023" in html

    def test_contains_raw_notes(self, sample_entry, styles):
        html = _render_entry_html(sample_entry, styles)
        assert "underwater kitchen dream" in html

    def test_contains_scene(self, sample_entry, styles):
        html = _render_entry_html(sample_entry, styles)
        assert "A kitchen submerged in amber light." in html

    def test_contains_tags(self, sample_entry, styles):
        html = _render_entry_html(sample_entry, styles)
        assert '<span class="tag">water</span>' in html
        assert '<span class="tag">light</span>' in html

    def test_contains_mood(self, sample_entry, styles):
        html = _render_entry_html(sample_entry, styles)
        assert "peaceful" in html

    def test_contains_style_label(self, sample_entry, styles):
        html = _render_entry_html(sample_entry, styles)
        assert "Watercolor Journal" in html

    def test_missing_image_shows_unavailable(self, sample_entry, styles):
        html = _render_entry_html(sample_entry, styles)
        assert "image not available" in html

    def test_no_illustration_entry(self, styles):
        entry = DreamEntry(
            id="test", date="2023-11-15",
            raw_notes="a dream about something",
            scene_description="A scene.",
        )
        html = _render_entry_html(entry, styles)
        assert "illustration unavailable" in html


class TestExportHtml:
    def test_creates_html_file(self, tmp_path, sample_entry, styles):
        db = JournalDatabase(entries=[sample_entry])
        output = str(tmp_path / "journal.html")
        result = export_html(db, output, styles)

        assert result == output
        assert (tmp_path / "journal.html").exists()

    def test_html_contains_doctype(self, tmp_path, sample_entry, styles):
        db = JournalDatabase(entries=[sample_entry])
        output = str(tmp_path / "journal.html")
        export_html(db, output, styles)

        content = (tmp_path / "journal.html").read_text()
        assert "<!DOCTYPE html>" in content

    def test_html_contains_css(self, tmp_path, sample_entry, styles):
        db = JournalDatabase(entries=[sample_entry])
        output = str(tmp_path / "journal.html")
        export_html(db, output, styles)

        content = (tmp_path / "journal.html").read_text()
        assert "<style>" in content
        assert "font-family" in content

    def test_html_is_responsive(self, tmp_path, sample_entry, styles):
        db = JournalDatabase(entries=[sample_entry])
        output = str(tmp_path / "journal.html")
        export_html(db, output, styles)

        content = (tmp_path / "journal.html").read_text()
        assert "viewport" in content
        assert "@media" in content

    def test_html_entry_count(self, tmp_path, styles):
        entries = [
            DreamEntry(id=f"e{i}", date=f"2023-11-{15+i:02d}", raw_notes=f"dream {i}")
            for i in range(3)
        ]
        db = JournalDatabase(entries=entries)
        output = str(tmp_path / "journal.html")
        export_html(db, output, styles)

        content = (tmp_path / "journal.html").read_text()
        assert "3 entries" in content

    def test_html_sorts_newest_first(self, tmp_path, styles):
        entries = [
            DreamEntry(id="e1", date="2023-11-15", raw_notes="first dream"),
            DreamEntry(id="e2", date="2023-11-20", raw_notes="second dream"),
        ]
        db = JournalDatabase(entries=entries)
        output = str(tmp_path / "journal.html")
        export_html(db, output, styles)

        content = (tmp_path / "journal.html").read_text()
        pos_nov20 = content.index("November 20")
        pos_nov15 = content.index("November 15")
        assert pos_nov20 < pos_nov15

    def test_creates_parent_dirs(self, tmp_path, sample_entry, styles):
        db = JournalDatabase(entries=[sample_entry])
        output = str(tmp_path / "deep" / "nested" / "journal.html")
        export_html(db, output, styles)
        assert (tmp_path / "deep" / "nested" / "journal.html").exists()

    def test_base64_encodes_existing_image(self, tmp_path, styles):
        # Create a real tiny PNG
        img_path = tmp_path / "test.png"
        from PIL import Image
        img = Image.new("RGB", (10, 10), "red")
        img.save(str(img_path))

        entry = DreamEntry(
            id="test", date="2023-11-15",
            raw_notes="a dream about colors and light",
            illustrations=[
                Illustration(
                    style="watercolor",
                    image_path=str(img_path),
                    thumb_path=str(img_path),
                    cost_usd=0.08,
                )
            ],
        )
        db = JournalDatabase(entries=[entry])
        output = str(tmp_path / "journal.html")
        export_html(db, output, styles)

        content = (tmp_path / "journal.html").read_text()
        assert "data:image/png;base64," in content
