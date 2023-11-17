"""Pipeline orchestrator: raw dream notes -> illustrated journal entry."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from openai import OpenAI

from dreamink.config import AppConfig, get_config, get_styles
from dreamink.expander import expand_dream_notes
from dreamink.illustrator import generate_illustration
from dreamink.models import DreamEntry, Illustration, TokenUsage
from dreamink.postprocess import download_and_save
from dreamink.storage import add_entry, load_database, save_database
from dreamink.utils import calculate_llm_cost, get_openai_client

logger = logging.getLogger(__name__)


def process_dream(
    raw_notes: str,
    date: str,
    style_name: str | None = None,
    config: AppConfig | None = None,
    client: OpenAI | None = None,
) -> DreamEntry:
    """Run the full pipeline: expand -> generate -> download -> persist.

    Args:
        raw_notes: Raw morning dream notes.
        date: Entry date in YYYY-MM-DD format.
        style_name: Style preset name. If None, uses config default.
        config: App config. If None, loads defaults.
        client: OpenAI client. If None, creates one from config.

    Returns:
        The completed DreamEntry, persisted to the database.
    """
    config = config or get_config()
    client = client or get_openai_client(config)
    styles = get_styles()

    style_name = style_name or config.default_style
    if style_name not in styles:
        raise ValueError(
            f"Unknown style '{style_name}'. Available: {', '.join(styles.keys())}"
        )
    style = styles[style_name]

    # Generate entry ID: date + sequence number
    db = load_database(config.database_path)
    existing_for_date = [e for e in db.entries if e.date == date]
    seq = len(existing_for_date) + 1
    entry_id = f"{date}-{seq:03d}"

    now = datetime.now(timezone.utc).isoformat()

    # Stage 1: Scene expansion
    logger.info("Stage 1: Expanding dream notes...")
    t0 = time.time()
    scene = expand_dream_notes(raw_notes, client=client, config=config)
    t1 = time.time()
    logger.info("Expansion completed in %.1fs: %s", t1 - t0, scene.description[:80])

    expansion_cost = calculate_llm_cost(scene.token_usage)

    # Stage 2: Image generation
    illustrations = []
    logger.info("Stage 2: Generating illustration (style: %s)...", style.label)
    t2 = time.time()
    generated = generate_illustration(scene.description, style, client=client, config=config)
    t3 = time.time()
    logger.info("Generation completed in %.1fs", t3 - t2)

    if generated.rejected:
        logger.warning(
            "Image generation rejected by content policy: %s",
            generated.rejection_reason,
        )
    else:
        # Stage 3: Download and post-process
        logger.info("Stage 3: Downloading and processing image...")
        t4 = time.time()
        processed = download_and_save(
            image_url=generated.url,
            entry_date=date,
            style=style_name,
            output_root=config.output_path,
            raw_notes=raw_notes,
        )
        t5 = time.time()
        logger.info("Download completed in %.1fs", t5 - t4)

        illustrations.append(
            Illustration(
                style=style_name,
                image_path=processed.image_path,
                thumb_path=processed.thumb_path,
                revised_prompt=generated.revised_prompt,
                generated_at=now,
                cost_usd=config.cost_per_image_usd,
            )
        )

    total_cost = expansion_cost + sum(i.cost_usd for i in illustrations)

    entry = DreamEntry(
        id=entry_id,
        date=date,
        created_at=now,
        raw_notes=raw_notes,
        scene_description=scene.description,
        tags=[],  # Tags added in Phase 4
        mood=scene.mood,
        illustrations=illustrations,
        token_usage={"expansion": scene.token_usage},
    )

    # Persist
    db = add_entry(db, entry)
    save_database(db, config.database_path)
    logger.info(
        "Entry %s saved. Total cost: $%.3f (expansion: $%.3f, image: $%.3f)",
        entry_id,
        total_cost,
        expansion_cost,
        config.cost_per_image_usd if illustrations else 0,
    )

    return entry
