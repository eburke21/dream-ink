# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest tests/ -v

# Run a single test file
poetry run pytest tests/test_expander.py -v

# Run a single test by name
poetry run pytest tests/ -k "test_name" -v

# CLI usage (all commands via Click)
poetry run dreamink add --date 2024-01-15 --style watercolor
poetry run dreamink generate --all
poetry run dreamink compare --styles surrealist,watercolor,manga
poetry run dreamink tags --stats
poetry run dreamink journal
poetry run dreamink export
```

## Architecture

DreamInk is a two-stage LLM pipeline that transforms raw morning dream notes into illustrated journal entries using GPT-4 and DALL-E 3.

### Pipeline Stages

1. **Scene Expansion** (`expander.py`) — GPT-4 transforms 2-4 sentence dream notes into 3-4 sentence visual scene descriptions. A second call extracts dominant colors and mood as JSON metadata. Temperature is 0.9 by default, lowered to 0.7 for short input (<10 words).

2. **Tag Extraction** (`tagger.py`) — GPT-4 extracts 3-5 thematic tags (recurring motifs like "water", "flight", "transformation") in lowercase-hyphenated format. Uses JSON parsing with regex fallback.

3. **Image Generation** (`illustrator.py`) — DALL-E 3 generates illustrations using one of 5 style presets defined in `styles.toml` (surrealist, watercolor, film_still, manga, collage). Handles content policy violations by asking GPT-4 to sanitize the scene and retrying once.

4. **Post-Processing** (`postprocess.py`) — Downloads images via httpx, generates 400px-wide thumbnails with Pillow, embeds PNG metadata (date, style, notes hash).

5. **Journal/Export** (`journal.py`, `exporter.py`) — Builds Markdown journal (chronological, newest first) and exports self-contained HTML with base64-encoded images.

### Orchestration

`pipeline.py` coordinates all stages and tracks cumulative costs. `cli.py` is the Click-based entry point.

### Data Layer

- **Models** (`models.py`) — Pydantic v2 models: `DreamEntry`, `Illustration`, `SceneExpansion`, `TokenUsage`, `StyleConfig`, etc.
- **Storage** (`storage.py`) — Flat JSON file at `data/dreamink.json`. Writes atomically via temp file + rename.
- **Config** (`config.py`) — Loads `dreamink.toml` (API settings, costs) and `styles.toml` (style presets).

### Key Design Decisions

- **No LangChain** — Two-call pipeline doesn't need a framework. Direct OpenAI API via `utils.py` (client factory, retry with exponential backoff).
- **"Preserve surreal" prompting** — LLMs default to coherence; explicit instructions prevent rationalizing impossible dream elements.
- **Watercolor default style** — Gracefully handles dream vagueness; photorealistic felt uncanny.
- **Flat JSON storage** — Personal tool with <500 entries/year; human-readable, zero-dependency.

### Error Handling

- Exponential backoff (2^attempt seconds) for HTTP 429/5xx errors
- Non-retryable: HTTP 400/401
- Content policy violations: auto-sanitize scene via GPT-4, retry once
- Database writes are atomic (temp file + rename)

### Tests

119 tests across 10 modules, all fully mocked (no API calls). Test files mirror source files: `test_expander.py`, `test_illustrator.py`, `test_pipeline.py`, etc.
