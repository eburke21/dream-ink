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

    word_count = len(raw_notes.split())
    if word_count > 500:
        _warn(f"Long input ({word_count} words) — will be truncated to 500 words.")
    elif word_count < 10:
        _warn(f"Short input ({word_count} words) — expansion will use lower temperature for accuracy.")

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
    from dreamink.tagger import extract_tags
    from dreamink.utils import get_openai_client

    client = get_openai_client(config)

    click.echo(_styled("Expanding dream scene...", "cyan"))
    scene = expand_dream_notes(raw_notes, client=client, config=config)
    expansion_cost = calculate_llm_cost(scene.token_usage)

    click.echo(_styled("Extracting tags...", "cyan"))
    tags, tag_usage = extract_tags(scene.description, client=client, config=config)

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
        tags=tags,
        mood=scene.mood,
        token_usage={"expansion": scene.token_usage, "tagging": tag_usage},
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


@cli.command()
@click.option(
    "--date", "entry_date",
    required=True,
    help="Date of the entry to compare (YYYY-MM-DD).",
)
@click.option(
    "--styles", "style_list",
    default=None,
    help="Comma-separated list of styles (default: all 5).",
)
def compare(entry_date: str, style_list: str | None):
    """Generate the same dream in multiple styles for comparison.

    Examples:

        dreamink compare --date 2023-11-15

        dreamink compare --date 2023-11-15 --styles watercolor,manga,film_still
    """
    try:
        from dreamink.illustrator import generate_illustration
        from dreamink.journal import rebuild_journal
        from dreamink.models import Illustration
        from dreamink.postprocess import download_and_save
        from dreamink.storage import load_database, save_database
        from dreamink.utils import get_openai_client

        config = get_config()
        all_styles = get_styles()
        client = get_openai_client(config)
        db = load_database(config.database_path)

        entries = [e for e in db.entries if e.date == entry_date]
        if not entries:
            _error(f"No entries found for date {entry_date}.")
            sys.exit(1)
        entry = entries[0]

        if not entry.scene_description:
            _error(f"Entry {entry.id} has no scene description.")
            sys.exit(1)

        # Determine which styles to generate
        if style_list:
            requested = [s.strip() for s in style_list.split(",")]
            for s in requested:
                if s not in all_styles:
                    _error(f"Unknown style '{s}'. Available: {', '.join(all_styles.keys())}")
                    sys.exit(1)
        else:
            requested = list(all_styles.keys())

        existing_styles = {i.style for i in entry.illustrations}
        to_generate = [s for s in requested if s not in existing_styles]

        if not to_generate:
            _success(f"Entry {entry.id} already has all requested styles.")
            return

        click.echo(f"Comparing {len(to_generate)} styles for entry {entry.id}:")
        total_cost = 0.0

        for style_name in to_generate:
            style = all_styles[style_name]
            click.echo(_styled(f"  Generating {style.label}...", "cyan"))

            result = generate_illustration(
                entry.scene_description, style, client=client, config=config
            )

            if result.rejected:
                _warn(f"  Content policy rejection for {style.label}: {result.rejection_reason}")
                continue

            processed = download_and_save(
                image_url=result.url,
                entry_date=entry.date,
                style=style_name,
                output_root=config.output_path,
                raw_notes=entry.raw_notes,
            )

            entry.illustrations.append(
                Illustration(
                    style=style_name,
                    image_path=processed.image_path,
                    thumb_path=processed.thumb_path,
                    revised_prompt=result.revised_prompt,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    cost_usd=config.cost_per_image_usd,
                )
            )
            total_cost += config.cost_per_image_usd
            _success(f"  Saved: {processed.image_path}")

        save_database(db, config.database_path)

        # Generate comparison Markdown
        _write_comparison_md(entry, config.output_path, all_styles)

        # Rebuild main journal
        journal_path = f"{config.output_path}/dream_journal.md"
        rebuild_journal(db, journal_path)

        click.echo()
        _success(f"Comparison complete. Total cost: ${total_cost:.2f}")

    except RuntimeError as e:
        if "Missing API key" in str(e):
            _error(f"Authentication failed -- check your OPENAI_API_KEY environment variable.\n  {e}")
        else:
            _error(str(e))
        sys.exit(1)
    except Exception as e:
        _error(f"Compare failed: {e}")
        sys.exit(1)


def _write_comparison_md(entry, output_root: str, styles: dict):
    """Write a comparison Markdown file with all illustrations for an entry."""
    from pathlib import Path

    year_month = entry.date[:7]
    comp_dir = Path(output_root) / year_month / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_path = comp_dir / f"{entry.date}_comparison.md"

    dt = datetime.strptime(entry.date, "%Y-%m-%d")
    lines = [
        f"# Style Comparison: {dt.strftime('%B %d, %Y')}",
        "",
        f"**Raw notes:** {entry.raw_notes}",
        "",
        f"**Scene:** {entry.scene_description}",
        "",
        "---",
        "",
    ]

    for illust in entry.illustrations:
        label = styles.get(illust.style, type("", (), {"label": illust.style})).label
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"![{illust.style}]({illust.image_path})")
        lines.append("")

    comp_path.write_text("\n".join(lines))


@cli.command()
@click.option("--since", default=None, help="Show tags from entries after this date (YYYY-MM-DD).")
@click.option("--top", default=None, type=int, help="Show only the N most common tags.")
@click.option("--stats", is_flag=True, default=False, help="Show cost statistics alongside tags.")
def tags(since: str | None, top: int | None, stats: bool):
    """Show tag frequency and co-occurrence across all entries.

    Examples:

        dreamink tags

        dreamink tags --since 2023-11-01 --top 10

        dreamink tags --stats
    """
    from dreamink.storage import load_database
    from dreamink.utils import calculate_llm_cost as calc_cost
    from dreamink.models import TokenUsage

    config = get_config()
    db = load_database(config.database_path)

    entries = db.entries
    if since:
        try:
            since_dt = datetime.strptime(since, "%Y-%m-%d")
        except ValueError:
            _error(f"Invalid date format: '{since}'. Use YYYY-MM-DD.")
            sys.exit(1)
        entries = [e for e in entries if datetime.strptime(e.date, "%Y-%m-%d") >= since_dt]

    if not entries:
        _warn("No entries found.")
        return

    # Tag frequency
    tag_counts: dict[str, int] = {}
    for entry in entries:
        for tag in entry.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if not tag_counts:
        _warn("No tags found on any entries.")
        return

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    if top:
        sorted_tags = sorted_tags[:top]

    max_count = sorted_tags[0][1] if sorted_tags else 1
    max_name_len = max(len(t[0]) for t in sorted_tags)

    click.echo(_styled("Tag Frequency", "bright_white"))
    click.echo()
    for tag, count in sorted_tags:
        bar_len = int((count / max_count) * 20)
        bar = "\u2588" * bar_len
        click.echo(f"  {tag:<{max_name_len}}  {bar} {count}")

    # Co-occurrence
    click.echo()
    click.echo(_styled("Co-occurrence (tags appearing together)", "bright_white"))
    click.echo()

    pair_counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        tags_sorted = sorted(entry.tags)
        for i in range(len(tags_sorted)):
            for j in range(i + 1, len(tags_sorted)):
                pair = (tags_sorted[i], tags_sorted[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    shown = 0
    for (t1, t2), count in sorted_pairs:
        if count < 2:
            break
        click.echo(f"  {t1} + {t2}: {count}")
        shown += 1
        if shown >= 10:
            break

    if shown == 0:
        click.echo("  (no co-occurring pairs yet)")

    # Cost stats
    if stats:
        _print_cost_stats(entries, config)


def _print_cost_stats(entries, config):
    """Print cost statistics for a set of entries."""
    from dreamink.models import TokenUsage
    from dreamink.utils import calculate_llm_cost as calc_cost

    click.echo()
    click.echo(_styled("Cost Statistics", "bright_white"))
    click.echo()

    total_expansion_cost = 0.0
    total_tagging_cost = 0.0
    total_image_cost = 0.0

    for entry in entries:
        if "expansion" in entry.token_usage:
            total_expansion_cost += calc_cost(entry.token_usage["expansion"])
        if "tagging" in entry.token_usage:
            total_tagging_cost += calc_cost(entry.token_usage["tagging"])
        for illust in entry.illustrations:
            total_image_cost += illust.cost_usd

    total_cost = total_expansion_cost + total_tagging_cost + total_image_cost
    avg_cost = total_cost / len(entries) if entries else 0

    click.echo(f"  Total entries: {len(entries)}")
    click.echo(f"  Total cost: ${total_cost:.2f}")
    click.echo(f"  Average cost/entry: ${avg_cost:.3f}")
    click.echo(f"  Breakdown:")
    click.echo(f"    Expansion: ${total_expansion_cost:.3f}")
    click.echo(f"    Tagging:   ${total_tagging_cost:.3f}")
    click.echo(f"    Images:    ${total_image_cost:.2f}")

    # Projected monthly cost
    if len(entries) >= 2:
        first_date = datetime.strptime(min(e.date for e in entries), "%Y-%m-%d")
        last_date = datetime.strptime(max(e.date for e in entries), "%Y-%m-%d")
        days_span = (last_date - first_date).days or 1
        entries_per_month = len(entries) / days_span * 30
        projected = avg_cost * entries_per_month
        click.echo(f"  Projected monthly ({entries_per_month:.0f} entries): ${projected:.2f}")


@cli.command("journal")
def journal_cmd():
    """Rebuild journal index and monthly index pages.

    Regenerates the main journal file, monthly index pages with
    thumbnail tables and tag frequency charts, and a root index
    linking to each month.

    Examples:

        dreamink journal
    """
    from dreamink.journal import rebuild_journal
    from dreamink.storage import load_database

    config = get_config()
    db = load_database(config.database_path)

    if not db.entries:
        _warn("No entries in the database.")
        return

    # Rebuild main journal
    journal_path = f"{config.output_path}/dream_journal.md"
    rebuild_journal(db, journal_path)
    _success(f"Rebuilt {journal_path}")

    # Build monthly indexes
    _build_monthly_indexes(db, config)

    _success("Journal rebuild complete.")


def _build_monthly_indexes(db, config):
    """Generate monthly index pages and root index."""
    from collections import defaultdict
    from pathlib import Path

    from dreamink.utils import calculate_llm_cost as calc_cost

    styles = get_styles()

    # Group entries by month
    months: dict[str, list] = defaultdict(list)
    for entry in db.entries:
        ym = entry.date[:7]
        months[ym].append(entry)

    root_index_lines = ["# Dream Journal Index", ""]

    for ym in sorted(months.keys(), reverse=True):
        entries = sorted(months[ym], key=lambda e: e.date)
        dt = datetime.strptime(ym + "-01", "%Y-%m-%d")
        month_label = dt.strftime("%B %Y")

        root_index_lines.append(f"- [{month_label}]({ym}/index.md) ({len(entries)} entries)")

        # Build monthly index
        lines = [f"# {month_label} Dreams", ""]
        lines.append("| Date | Tags | Style | Thumbnail |")
        lines.append("|------|------|-------|-----------|")

        for entry in entries:
            entry_dt = datetime.strptime(entry.date, "%Y-%m-%d")
            date_short = entry_dt.strftime("%b %d")
            tags_str = ", ".join(entry.tags) if entry.tags else ""
            if entry.illustrations:
                illust = entry.illustrations[0]
                style_label = styles.get(illust.style, type("", (), {"label": illust.style})).label
                thumb = f"![thumb]({illust.thumb_path})" if illust.thumb_path else ""
            else:
                style_label = ""
                thumb = ""
            lines.append(f"| {date_short} | {tags_str} | {style_label} | {thumb} |")

        lines.append("")

        # Tag frequency for this month
        tag_counts: dict[str, int] = {}
        for entry in entries:
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if tag_counts:
            lines.append(f"### Tag Frequency ({month_label})")
            lines.append("")
            max_count = max(tag_counts.values())
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                bar = "\u2588" * int((count / max_count) * 10)
                lines.append(f"{tag}: {bar} {count}")
            lines.append("")

        # Monthly cost
        total_cost = 0.0
        for entry in entries:
            for usage in entry.token_usage.values():
                total_cost += calc_cost(usage)
            for illust in entry.illustrations:
                total_cost += illust.cost_usd

        lines.append(f"**Entries:** {len(entries)} | **Total cost:** ${total_cost:.2f}")
        lines.append("")

        month_dir = Path(config.output_path) / ym
        month_dir.mkdir(parents=True, exist_ok=True)
        (month_dir / "index.md").write_text("\n".join(lines))
        click.echo(f"  Built {ym}/index.md ({len(entries)} entries)")

    root_index_lines.append("")
    root_path = Path(config.output_path) / "index.md"
    root_path.write_text("\n".join(root_index_lines))


@cli.command("export")
def export_cmd():
    """Export journal as a self-contained HTML file.

    Generates a single HTML file with all journal entries and
    base64-encoded thumbnail images. The file can be opened in any
    browser and shared without external dependencies.

    Examples:

        dreamink export
    """
    from dreamink.exporter import export_html
    from dreamink.storage import load_database

    config = get_config()
    styles = get_styles()
    db = load_database(config.database_path)

    if not db.entries:
        _warn("No entries in the database.")
        return

    style_labels = {name: s.label for name, s in styles.items()}
    output_path = f"{config.output_path}/dream_journal.html"

    result_path = export_html(db, output_path, style_labels)
    _success(f"Exported {len(db.entries)} entries to {result_path}")
