"""Tests for configuration loader."""

import pytest
from pathlib import Path

from dreamink.config import get_config, get_styles


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class TestGetConfig:
    def test_loads_default_config(self):
        config = get_config()
        assert config.api_key_env == "OPENAI_API_KEY"
        assert config.expansion_model == "gpt-4-0613"
        assert config.expansion_temperature == 0.9
        assert config.image_size == "1792x1024"
        assert config.image_quality == "hd"
        assert config.default_style == "watercolor"

    def test_config_is_frozen(self):
        config = get_config()
        with pytest.raises(AttributeError):
            config.expansion_model = "gpt-4-turbo"

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            get_config(tmp_path / "nonexistent.toml")


class TestGetStyles:
    def test_loads_all_styles(self):
        styles = get_styles()
        assert len(styles) == 5
        expected = {"surrealist", "watercolor", "film_still", "manga", "collage"}
        assert set(styles.keys()) == expected

    def test_style_fields(self):
        styles = get_styles()
        wc = styles["watercolor"]
        assert wc.label == "Watercolor Journal"
        assert "watercolor" in wc.suffix.lower()
        assert wc.name == "watercolor"

    def test_all_styles_have_suffix(self):
        styles = get_styles()
        for name, style in styles.items():
            assert style.suffix, f"Style '{name}' has empty suffix"

    def test_missing_styles_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Styles file not found"):
            get_styles(tmp_path / "nonexistent.toml")
