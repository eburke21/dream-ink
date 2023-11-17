"""Markdown journal assembly and file management."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from dreamink.config import get_styles
from dreamink.models import DreamEntry, JournalDatabase

logger = logging.getLogger(__name__)

_JOURNAL_HEADER = "# Dream Journal\n\n"


def render_entry(entry: DreamEntry) -> str:
    """Render a single DreamEntry as a Markdown string.

    Matches the spec format: H2 date header, raw notes, scene description,
    style, tags, image embed(s), horizontal rule.
    """
    dt = datetime.strptime(entry.date, "%Y-%m-%d")
    date_str = dt.strftime("%B %d, %Y")  # "November 15, 2023"

    lines = [f"## {date_str}", ""]

    # Raw notes
    lines.append(f"**Raw notes:** {entry.raw_notes}")
    lines.append("")

    # Scene description
    if entry.scene_description:
        lines.append(f"**Scene:** {entry.scene_description}")
        lines.append("")

    # Style(s) and images
    styles = _load_styles_safe()

    if entry.illustrations:
        for illust in entry.illustrations:
            style_label = styles.get(illust.style, illust.style)
            lines.append(f"**Style:** {style_label}")
            lines.append("")

            if entry.tags:
                tags_str = " ".join(f"`{t}`" for t in entry.tags)
                lines.append(f"**Tags:** {tags_str}")
                lines.append("")

            img_name = f"dream-{entry.date}-{illust.style}"
            lines.append(f"![{img_name}]({illust.image_path})")
            lines.append("")
    else:
        # No illustration
        if entry.tags:
            tags_str = " ".join(f"`{t}`" for t in entry.tags)
            lines.append(f"**Tags:** {tags_str}")
            lines.append("")
        lines.append("*[illustration unavailable]*")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def _load_styles_safe() -> dict[str, str]:
    """Load style name -> label mapping, or return empty on failure."""
    try:
        return {name: s.label for name, s in get_styles().items()}
    except FileNotFoundError:
        return {}


def append_to_journal(entry: DreamEntry, journal_path: str) -> None:
    """Append a rendered entry to the main journal file.

    Creates the file with a header if it doesn't exist.
    """
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text(_JOURNAL_HEADER)

    rendered = render_entry(entry)

    with open(path, "a") as f:
        f.write(rendered)


def rebuild_journal(db: JournalDatabase, journal_path: str) -> list[str]:
    """Regenerate the entire journal file from the database.

    Entries are sorted by date descending (newest first).
    Detects missing image files and logs warnings.

    Returns:
        List of missing image paths (empty if all images exist).
    """
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_entries = sorted(db.entries, key=lambda e: e.date, reverse=True)

    missing_images = []
    for entry in sorted_entries:
        for illust in entry.illustrations:
            if illust.image_path and not Path(illust.image_path).exists():
                missing_images.append(illust.image_path)
                logger.warning("Missing image file: %s", illust.image_path)

    parts = [_JOURNAL_HEADER]
    for entry in sorted_entries:
        parts.append(render_entry(entry))

    path.write_text("".join(parts))
    return missing_images
