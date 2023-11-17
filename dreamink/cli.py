"""Click CLI entry point for DreamInk."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone

import click

from dreamink.config import get_config, get_styles
from dreamink.utils import calculate_llm_cost


def _styled(text: str, color: str) -> str:
    return click.style(text, fg=color)


def _success(msg: str) -> None:
    click.echo(_styled(msg, "green"))


def _warn(msg: str) -> None:
    click.echo(_styled(msg, "yellow"), err=True)


def _error(msg: str) -> None:
    click.echo(_styled(f"Error: {msg}", "red"), err=True)


@click.group()
@click.version_option(package_name="dreamink")
def cli():
    """DreamInk -- Dream Journal Illustrator.

    Transform raw dream notes into rich visual illustrations
    using a GPT-4 + DALL-E 3 pipeline.
    """


@cli.command()
@click.option(
    "--date", "entry_date",
    default=None,
    help="Entry date (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--style", "style_name",
    default=None,
    help="Illustration style preset (e.g. watercolor, manga, surrealist).",
)
@click.option(
    "--skip-image",
    is_flag=True,
    default=False,
    help="Save the entry without generating an illustration.",
)
def add(entry_date: str | None, style_name: str | None, skip_image: bool):
    """Add a new dream entry.

    Prompts for dream notes interactively, or accepts piped input.

    Examples:

        dreamink add --date 2023-11-15

        echo "flying over a city" | dreamink add --date 2023-11-20
    """
    # Resolve date
    if entry_date is None:
        entry_date = date.today().isoformat()
    else:
        try:
            datetime.strptime(entry_date, "%Y-%m-%d")
        except ValueError:
            _error(f"Invalid date format: '{entry_date}'. Use YYYY-MM-DD.")
            sys.exit(1)

    # Get dream notes from stdin (piped) or interactive prompt
    if not sys.stdin.isatty():
        raw_notes = sys.stdin.read().strip()
    else:
        click.echo("Enter your dream notes (press Enter twice to finish):")
        lines = []
        while True:
            line = click.get_text_stream("stdin").readline()
            if line.strip() == "" and lines:
                break
            lines.append(line.rstrip())
        raw_notes = "\n".join(lines).strip()

    if not raw_notes:
        _error("No dream notes provided.")
        sys.exit(1)

    click.echo(f"\nDate: {entry_date}")
    click.echo(f"Notes: {raw_notes[:80]}{'...' if len(raw_notes) > 80 else ''}")
    click.echo()

    try:
        config = get_config()
        styles = get_styles()

        if skip_image:
            _run_add_without_image(raw_notes, entry_date, config)
        else:
            _run_add_with_image(raw_notes, entry_date, style_name, config, styles)

    except RuntimeError as e:
        if "Missing API key" in str(e):
            _error(f"Authentication failed -- check your OPENAI_API_KEY environment variable.\n  {e}")
        else:
            _error(str(e))
        sys.exit(1)
    except Exception as e:
        _error(f"Pipeline failed: {e}")
        sys.exit(1)


def _run_add_without_image(raw_notes, entry_date, config):
    """Add an entry with scene expansion but no illustration."""
    from dreamink.expander import expand_dream_notes
    from dreamink.journal import append_to_journal
    from dreamink.models import DreamEntry
    from dreamink.storage import add_entry, load_database, save_database
    from dreamink.utils import get_openai_client

    client = get_openai_client(config)

    click.echo(_styled("Expanding dream scene...", "cyan"))
    scene = expand_dream_notes(raw_notes, client=client, config=config)
    expansion_cost = calculate_llm_cost(scene.token_usage)

    db = load_database(config.database_path)
    existing = [e for e in db.entries if e.date == entry_date]
    seq = len(existing) + 1
    entry_id = f"{entry_date}-{seq:03d}"

    entry = DreamEntry(
        id=entry_id,
        date=entry_date,
        created_at=datetime.now(timezone.utc).isoformat(),
        raw_notes=raw_notes,
        scene_description=scene.description,
        mood=scene.mood,
        token_usage={"expansion": scene.token_usage},
    )

    db = add_entry(db, entry)
    save_database(db, config.database_path)

    journal_path = f"{config.output_path}/dream_journal.md"
    append_to_journal(entry, journal_path)

    _print_summary(entry, expansion_cost, 0.0)
    _warn("Image generation skipped (--skip-image). Use 'dreamink generate' to add illustrations later.")


def _run_add_with_image(raw_notes, entry_date, style_name, config, styles):
    """Add an entry with the full pipeline (expand + generate + download)."""
    from dreamink.journal import append_to_journal
    from dreamink.pipeline import process_dream
    from dreamink.utils import get_openai_client

    client = get_openai_client(config)

    resolved_style = style_name or config.default_style
    if resolved_style not in styles:
        _error(f"Unknown style '{resolved_style}'. Available: {', '.join(styles.keys())}")
        sys.exit(1)

    click.echo(_styled("Expanding dream scene...", "cyan"))
    click.echo(_styled(f"Generating illustration ({styles[resolved_style].label})...", "cyan"))
    click.echo(_styled("Downloading and processing image...", "cyan"))

    entry = process_dream(
        raw_notes=raw_notes,
        date=entry_date,
        style_name=style_name,
        config=config,
        client=client,
    )

    journal_path = f"{config.output_path}/dream_journal.md"
    append_to_journal(entry, journal_path)

    expansion_cost = calculate_llm_cost(entry.token_usage.get("expansion", __import__("dreamink.models", fromlist=["TokenUsage"]).TokenUsage()))
    image_cost = sum(i.cost_usd for i in entry.illustrations)

    _print_summary(entry, expansion_cost, image_cost)

    if not entry.illustrations:
        _warn("Image generation was rejected by content policy. Entry saved with scene description only.")
        _warn("Use 'dreamink generate' to retry with a different style.")


def _print_summary(entry, expansion_cost: float, image_cost: float):
    """Print a formatted summary after adding an entry."""
    total_cost = expansion_cost + image_cost

    click.echo()
    _success(f"Entry {entry.id} saved!")
    click.echo()
    click.echo(_styled("Scene: ", "bright_white") + entry.scene_description.split(".")[0] + ".")
    if entry.mood:
        click.echo(_styled("Mood: ", "bright_white") + entry.mood)
    if entry.tags:
        tags_str = " ".join(f"`{t}`" for t in entry.tags)
        click.echo(_styled("Tags: ", "bright_white") + tags_str)
    if entry.illustrations:
        click.echo(_styled("Image: ", "bright_white") + entry.illustrations[0].image_path)
    click.echo(_styled("Cost: ", "bright_white") + f"${total_cost:.3f} (expansion: ${expansion_cost:.3f}, image: ${image_cost:.3f})")


@cli.command()
@click.option(
    "--date", "entry_date",
    default=None,
    help="Generate for a specific date (YYYY-MM-DD).",
)
@click.option(
    "--all", "gen_all",
    is_flag=True,
    default=False,
    help="Generate illustrations for all un-illustrated entries.",
)
@click.option(
    "--style", "style_name",
    default=None,
    help="Style preset to use. Adds a new illustration (doesn't replace existing).",
)
def generate(entry_date: str | None, gen_all: bool, style_name: str | None):
    """Generate illustrations for existing entries.

    Use for entries saved with --skip-image or where generation previously failed.

    Examples:

        dreamink generate --date 2023-11-15

        dreamink generate --all

        dreamink generate --style manga --date 2023-11-15
    """
    if not entry_date and not gen_all:
        _error("Specify --date or --all.")
        sys.exit(1)

    try:
        config = get_config()
        styles = get_styles()
        resolved_style = style_name or config.default_style
        if resolved_style not in styles:
            _error(f"Unknown style '{resolved_style}'. Available: {', '.join(styles.keys())}")
            sys.exit(1)
        style = styles[resolved_style]

        from dreamink.illustrator import generate_illustration
        from dreamink.journal import append_to_journal, rebuild_journal
        from dreamink.models import Illustration
        from dreamink.postprocess import download_and_save
        from dreamink.storage import load_database, save_database
        from dreamink.utils import get_openai_client

        client = get_openai_client(config)
        db = load_database(config.database_path)

        if entry_date:
            entries = [e for e in db.entries if e.date == entry_date]
            if not entries:
                _error(f"No entries found for date {entry_date}.")
                sys.exit(1)
        else:
            # --all: find entries without illustrations
            entries = [e for e in db.entries if not e.illustrations]
            if not entries:
                _success("All entries already have illustrations.")
                return

        generated_count = 0
        for entry in entries:
            # Skip if already has this style (unless --all with no style specified)
            existing_styles = {i.style for i in entry.illustrations}
            if resolved_style in existing_styles and style_name is not None:
                _warn(f"Entry {entry.id} already has a {resolved_style} illustration. Skipping.")
                continue

            if not entry.scene_description:
                _warn(f"Entry {entry.id} has no scene description. Skipping.")
                continue

            click.echo(_styled(f"Generating {style.label} for {entry.id}...", "cyan"))

            result = generate_illustration(
                entry.scene_description, style, client=client, config=config
            )

            if result.rejected:
                _warn(f"Content policy rejection for {entry.id}: {result.rejection_reason}")
                continue

            click.echo(_styled("Downloading image...", "cyan"))
            processed = download_and_save(
                image_url=result.url,
                entry_date=entry.date,
                style=resolved_style,
                output_root=config.output_path,
                raw_notes=entry.raw_notes,
            )

            entry.illustrations.append(
                Illustration(
                    style=resolved_style,
                    image_path=processed.image_path,
                    thumb_path=processed.thumb_path,
                    revised_prompt=result.revised_prompt,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    cost_usd=config.cost_per_image_usd,
                )
            )
            generated_count += 1
            _success(f"Saved: {processed.image_path}")

        save_database(db, config.database_path)

        # Rebuild journal to reflect new illustrations
        journal_path = f"{config.output_path}/dream_journal.md"
        rebuild_journal(db, journal_path)

        click.echo()
        _success(f"Generated {generated_count} illustration(s). Journal updated.")

    except RuntimeError as e:
        if "Missing API key" in str(e):
            _error(f"Authentication failed -- check your OPENAI_API_KEY environment variable.\n  {e}")
        else:
            _error(str(e))
        sys.exit(1)
    except Exception as e:
        _error(f"Generation failed: {e}")
        sys.exit(1)
