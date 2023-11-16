"""JSON state persistence for DreamInk journal database."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from dreamink.models import DreamEntry, JournalDatabase


def load_database(db_path: str | Path) -> JournalDatabase:
    """Load the journal database from JSON file.

    Returns an empty database if the file doesn't exist.
    """
    path = Path(db_path)
    if not path.exists():
        return JournalDatabase()

    with open(path) as f:
        data = json.load(f)

    return JournalDatabase.model_validate(data)


def save_database(db: JournalDatabase, db_path: str | Path) -> None:
    """Atomically write the database to JSON.

    Writes to a temp file first, then renames to prevent corruption.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=".dreamink_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(db.model_dump_json(indent=2))
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def add_entry(db: JournalDatabase, entry: DreamEntry) -> JournalDatabase:
    """Add a dream entry and update the tag index.

    Returns the updated database (does not persist — call save_database).
    """
    db.entries.append(entry)

    for tag in entry.tags:
        if tag not in db.tag_index:
            db.tag_index[tag] = []
        if entry.id not in db.tag_index[tag]:
            db.tag_index[tag].append(entry.id)

    return db


def get_entry(db: JournalDatabase, entry_id: str) -> DreamEntry | None:
    """Look up a dream entry by ID."""
    for entry in db.entries:
        if entry.id == entry_id:
            return entry
    return None


def get_entries_by_date_range(
    db: JournalDatabase, start: str, end: str
) -> list[DreamEntry]:
    """Get entries within a date range (inclusive).

    Args:
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    return [
        entry
        for entry in db.entries
        if start_dt <= datetime.strptime(entry.date, "%Y-%m-%d") <= end_dt
    ]
