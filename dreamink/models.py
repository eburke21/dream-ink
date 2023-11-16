"""Pydantic data models for DreamInk."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class TokenUsage(BaseModel):
    """Token counts from a GPT-4 API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


class SceneExpansion(BaseModel):
    """Output of the GPT-4 scene expansion stage."""

    description: str
    dominant_colors: list[str] = Field(default_factory=list)
    mood: str = ""
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class GeneratedImage(BaseModel):
    """Output of a DALL-E 3 image generation call."""

    url: str
    revised_prompt: str = ""
    local_path: str = ""
    rejected: bool = False
    rejection_reason: str = ""
    generation_metadata: dict = Field(default_factory=dict)


class StyleConfig(BaseModel):
    """A single style preset loaded from TOML config."""

    name: str
    label: str
    suffix: str
    best_for: str = ""


class Illustration(BaseModel):
    """A persisted illustration record attached to a dream entry."""

    style: str
    image_path: str
    thumb_path: str = ""
    revised_prompt: str = ""
    generated_at: str = ""
    cost_usd: float = 0.0

    @field_validator("cost_usd")
    @classmethod
    def cost_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("cost_usd must be non-negative")
        return v


class DreamEntry(BaseModel):
    """A single dream journal entry."""

    id: str
    date: str
    created_at: str = ""
    raw_notes: str
    scene_description: str = ""
    tags: list[str] = Field(default_factory=list)
    mood: str = ""
    illustrations: list[Illustration] = Field(default_factory=list)
    token_usage: dict[str, TokenUsage] = Field(default_factory=dict)

    @field_validator("date")
    @classmethod
    def date_must_be_valid(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"date must be YYYY-MM-DD format, got '{v}'")
        return v

    @field_validator("raw_notes")
    @classmethod
    def notes_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raw_notes must not be empty")
        return v

    @field_validator("tags")
    @classmethod
    def tags_must_be_lowercase_hyphenated(cls, v: list[str]) -> list[str]:
        pattern = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
        for tag in v:
            if not pattern.match(tag):
                raise ValueError(
                    f"tags must be lowercase-hyphenated, got '{tag}'"
                )
        return v


class JournalDatabase(BaseModel):
    """Root model for the JSON state file."""

    version: str = "1.0"
    entries: list[DreamEntry] = Field(default_factory=list)
    tag_index: dict[str, list[str]] = Field(default_factory=dict)
