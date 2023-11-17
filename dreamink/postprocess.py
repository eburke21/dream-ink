"""Image download, thumbnails, and metadata embedding."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, PngImagePlugin

logger = logging.getLogger(__name__)

_THUMBNAIL_WIDTH = 400
_DOWNLOAD_TIMEOUT = 60.0


@dataclass
class ProcessedImage:
    """Result of downloading and processing a generated image."""

    image_path: str
    thumb_path: str
    file_size: int


def download_and_save(
    image_url: str,
    entry_date: str,
    style: str,
    output_root: str = "journal",
    raw_notes: str = "",
) -> ProcessedImage:
    """Download a generated image, create thumbnail, embed metadata.

    Args:
        image_url: Temporary URL from DALL-E 3 (expires in ~1 hour).
        entry_date: Entry date in YYYY-MM-DD format.
        style: Style name (e.g. "watercolor") for the filename.
        output_root: Root journal directory.
        raw_notes: Original dream notes for hash metadata.

    Returns:
        ProcessedImage with local paths and file size.
    """
    # Build paths
    year_month = entry_date[:7]  # "2023-11"
    images_dir = Path(output_root) / year_month / "images"
    thumbs_dir = images_dir / "thumbs"
    images_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{entry_date}_{style}.png"
    image_path = images_dir / filename
    thumb_path = thumbs_dir / filename

    # Download the image
    logger.info("Downloading image from DALL-E URL...")
    response = httpx.get(image_url, timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    image_data = response.content

    # Open with Pillow for processing
    img = Image.open(BytesIO(image_data))

    # Embed metadata in PNG text chunks
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("dreamink:date", entry_date)
    png_info.add_text("dreamink:style", style)
    if raw_notes:
        notes_hash = hashlib.sha256(raw_notes.encode()).hexdigest()[:8]
        png_info.add_text("dreamink:notes_hash", notes_hash)

    # Save full-size image with metadata
    img.save(str(image_path), pnginfo=png_info)
    file_size = image_path.stat().st_size

    # Generate thumbnail (400px wide, maintain aspect ratio)
    aspect_ratio = img.height / img.width
    thumb_height = int(_THUMBNAIL_WIDTH * aspect_ratio)
    thumb = img.resize((_THUMBNAIL_WIDTH, thumb_height), Image.LANCZOS)
    thumb.save(str(thumb_path), pnginfo=png_info)

    logger.info("Saved image: %s (%d bytes), thumbnail: %s", image_path, file_size, thumb_path)

    return ProcessedImage(
        image_path=str(image_path),
        thumb_path=str(thumb_path),
        file_size=file_size,
    )
