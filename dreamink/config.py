"""TOML-based configuration loader for DreamInk."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from dreamink.models import StyleConfig

# Default config directory: project_root/config/
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from dreamink.toml."""

    api_key_env: str
    expansion_model: str
    expansion_temperature: float
    max_retries: int
    image_size: str
    image_quality: str
    cost_per_image_usd: float
    output_path: str
    database_path: str
    default_style: str


def get_config(config_path: Path | None = None) -> AppConfig:
    """Load application config from TOML file.

    Args:
        config_path: Path to dreamink.toml. Defaults to config/dreamink.toml.

    Returns:
        Frozen AppConfig dataclass.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = config_path or (_CONFIG_DIR / "dreamink.toml")
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Create config/dreamink.toml or pass --config."
        )

    with open(path, "rb") as f:
        data = tomllib.load(f)

    api = data.get("api", {})
    models = data.get("models", {})
    images = data.get("images", {})
    journal = data.get("journal", {})

    return AppConfig(
        api_key_env=api.get("api_key_env", "OPENAI_API_KEY"),
        expansion_model=models.get("expansion_model", "gpt-4-0613"),
        expansion_temperature=models.get("expansion_temperature", 0.9),
        max_retries=models.get("max_retries", 2),
        image_size=images.get("size", "1792x1024"),
        image_quality=images.get("quality", "hd"),
        cost_per_image_usd=images.get("cost_per_image_usd", 0.080),
        output_path=journal.get("output_path", "journal"),
        database_path=journal.get("database_path", "data/dreamink.json"),
        default_style=journal.get("default_style", "watercolor"),
    )


def get_styles(styles_path: Path | None = None) -> dict[str, StyleConfig]:
    """Load style presets from TOML file.

    Args:
        styles_path: Path to styles.toml. Defaults to config/styles.toml.

    Returns:
        Dict mapping style name to StyleConfig.

    Raises:
        FileNotFoundError: If the styles file doesn't exist.
    """
    path = styles_path or (_CONFIG_DIR / "styles.toml")
    if not path.exists():
        raise FileNotFoundError(
            f"Styles file not found: {path}. "
            "Create config/styles.toml with style presets."
        )

    with open(path, "rb") as f:
        data = tomllib.load(f)

    styles_data = data.get("styles", {})
    styles = {}
    for name, preset in styles_data.items():
        styles[name] = StyleConfig(
            name=name,
            label=preset["label"],
            suffix=preset["suffix"],
            best_for=preset.get("best_for", ""),
        )

    return styles
