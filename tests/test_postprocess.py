"""Tests for image download and post-processing."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, PngImagePlugin

from dreamink.postprocess import ProcessedImage, download_and_save


def _make_test_png(width=1792, height=1024) -> bytes:
    """Create a minimal valid PNG image in memory."""
    img = Image.new("RGB", (width, height), color="blue")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDownloadAndSave:
    def test_saves_image_to_correct_path(self, tmp_path):
        png_data = _make_test_png()

        with patch("dreamink.postprocess.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.content = png_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_response

            result = download_and_save(
                image_url="https://example.com/image.png",
                entry_date="2023-11-15",
                style="watercolor",
                output_root=str(tmp_path / "journal"),
                raw_notes="test dream notes",
            )

        assert isinstance(result, ProcessedImage)
        expected_path = str(tmp_path / "journal/2023-11/images/2023-11-15_watercolor.png")
        assert result.image_path == expected_path

        from pathlib import Path
        assert Path(result.image_path).exists()

    def test_creates_thumbnail_at_400px_width(self, tmp_path):
        png_data = _make_test_png(1792, 1024)

        with patch("dreamink.postprocess.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.content = png_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_response

            result = download_and_save(
                image_url="https://example.com/image.png",
                entry_date="2023-11-15",
                style="watercolor",
                output_root=str(tmp_path / "journal"),
            )

        from pathlib import Path
        assert Path(result.thumb_path).exists()

        thumb = Image.open(result.thumb_path)
        assert thumb.width == 400

    def test_thumbnail_maintains_aspect_ratio(self, tmp_path):
        png_data = _make_test_png(1792, 1024)

        with patch("dreamink.postprocess.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.content = png_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_response

            result = download_and_save(
                image_url="https://example.com/image.png",
                entry_date="2023-11-15",
                style="watercolor",
                output_root=str(tmp_path / "journal"),
            )

        thumb = Image.open(result.thumb_path)
        expected_height = int(400 * (1024 / 1792))
        assert thumb.height == expected_height

    def test_embeds_png_metadata(self, tmp_path):
        png_data = _make_test_png()

        with patch("dreamink.postprocess.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.content = png_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_response

            result = download_and_save(
                image_url="https://example.com/image.png",
                entry_date="2023-11-15",
                style="manga",
                output_root=str(tmp_path / "journal"),
                raw_notes="underwater kitchen dream",
            )

        img = Image.open(result.image_path)
        metadata = img.info
        assert metadata["dreamink:date"] == "2023-11-15"
        assert metadata["dreamink:style"] == "manga"
        assert len(metadata["dreamink:notes_hash"]) == 8

    def test_thumb_path_in_thumbs_subdir(self, tmp_path):
        png_data = _make_test_png()

        with patch("dreamink.postprocess.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.content = png_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_response

            result = download_and_save(
                image_url="https://example.com/image.png",
                entry_date="2023-12-01",
                style="surrealist",
                output_root=str(tmp_path / "journal"),
            )

        assert "/thumbs/" in result.thumb_path
        expected = str(tmp_path / "journal/2023-12/images/thumbs/2023-12-01_surrealist.png")
        assert result.thumb_path == expected

    def test_file_size_is_positive(self, tmp_path):
        png_data = _make_test_png()

        with patch("dreamink.postprocess.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.content = png_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_response

            result = download_and_save(
                image_url="https://example.com/image.png",
                entry_date="2023-11-15",
                style="watercolor",
                output_root=str(tmp_path / "journal"),
            )

        assert result.file_size > 0
