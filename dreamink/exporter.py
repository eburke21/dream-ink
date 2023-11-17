"""HTML export: generate a self-contained HTML journal file."""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path

from dreamink.models import DreamEntry, JournalDatabase

logger = logging.getLogger(__name__)

_CSS = """
body {
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem 1rem;
    background: #fafaf8;
    color: #2c2c2c;
    line-height: 1.7;
}
h1 {
    font-size: 2rem;
    border-bottom: 2px solid #e0d8cc;
    padding-bottom: 0.5rem;
    color: #3a3a3a;
}
h2 {
    font-size: 1.4rem;
    color: #4a4a4a;
    margin-top: 2.5rem;
}
.entry {
    margin-bottom: 2rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #e0d8cc;
}
.raw-notes {
    font-style: italic;
    color: #666;
    margin: 0.5rem 0;
}
.scene {
    margin: 1rem 0;
}
.tag {
    display: inline-block;
    background: #e8e4dd;
    color: #555;
    padding: 2px 8px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    margin: 2px 2px;
}
.style-label {
    color: #888;
    font-size: 0.9rem;
}
.mood {
    color: #888;
    font-size: 0.9rem;
}
img.illustration {
    max-width: 100%;
    border-radius: 4px;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.meta {
    font-size: 0.85rem;
    color: #999;
}
@media (max-width: 600px) {
    body { padding: 1rem 0.5rem; }
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.2rem; }
}
"""


def _encode_image(image_path: str) -> str | None:
    """Base64-encode an image file for inline HTML embedding."""
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image not found for export: %s", image_path)
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_entry_html(entry: DreamEntry, styles: dict[str, str]) -> str:
    """Render a single entry as an HTML fragment."""
    dt = datetime.strptime(entry.date, "%Y-%m-%d")
    date_str = dt.strftime("%B %d, %Y")

    parts = [f'<div class="entry">', f"<h2>{date_str}</h2>"]

    parts.append(f'<p class="raw-notes"><strong>Raw notes:</strong> {entry.raw_notes}</p>')

    if entry.scene_description:
        parts.append(f'<p class="scene"><strong>Scene:</strong> {entry.scene_description}</p>')

    if entry.mood:
        parts.append(f'<p class="mood"><strong>Mood:</strong> {entry.mood}</p>')

    if entry.tags:
        tags_html = " ".join(f'<span class="tag">{t}</span>' for t in entry.tags)
        parts.append(f"<p>{tags_html}</p>")

    for illust in entry.illustrations:
        style_label = styles.get(illust.style, illust.style)
        parts.append(f'<p class="style-label"><strong>Style:</strong> {style_label}</p>')

        # Try thumbnail first (smaller file), fall back to full image
        img_path = illust.thumb_path or illust.image_path
        data_uri = _encode_image(img_path)
        if data_uri:
            parts.append(f'<img class="illustration" src="{data_uri}" alt="{illust.style}">')
        else:
            parts.append(f'<p><em>[image not available: {illust.image_path}]</em></p>')

    if not entry.illustrations:
        parts.append("<p><em>[illustration unavailable]</em></p>")

    parts.append("</div>")
    return "\n".join(parts)


def export_html(db: JournalDatabase, output_path: str, styles: dict[str, str]) -> str:
    """Export the journal database as a self-contained HTML file.

    Args:
        db: The journal database.
        output_path: Path to write the HTML file.
        styles: Mapping of style name -> style label.

    Returns:
        The output file path.
    """
    sorted_entries = sorted(db.entries, key=lambda e: e.date, reverse=True)

    entry_html = "\n".join(
        _render_entry_html(entry, styles) for entry in sorted_entries
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dream Journal</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Dream Journal</h1>
<p class="meta">{len(db.entries)} entries</p>
{entry_html}
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)

    return str(path)
