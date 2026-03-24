"""Rich-based progress indicators for DreamInk CLI commands."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.status import Status
from rich.theme import Theme

# Theme for progress bar colors
_THEME = Theme({
    "bar.complete": "cyan",
    "bar.finished": "green",
})

# Shared console instance — themed, reused across spinner and progress bar
_console = Console(theme=_THEME)

# Styling constants (VHS-friendly)
SPINNER_TYPE = "dots"
SPINNER_STYLE = "cyan"
REFRESH_PER_SECOND = 8


@contextmanager
def dreamink_spinner(
    message: str,
    *,
    console: Console | None = None,
) -> Generator[Status, None, None]:
    """Animated spinner for single long-running operations.

    Usage:
        with dreamink_spinner("Generating watercolor illustration..."):
            result = generate_illustration(...)
    """
    con = console or _console
    with con.status(
        message,
        spinner=SPINNER_TYPE,
        spinner_style=SPINNER_STYLE,
        refresh_per_second=REFRESH_PER_SECOND,
    ) as status:
        yield status


BAR_WIDTH = 30


class DreamInkProgress:
    """Combined progress bar + spinner for batch operations.

    Usage:
        with DreamInkProgress(total=5, label="styles") as progress:
            for item in items:
                progress.update_step(f"Generating {item}...")
                do_work()
                progress.log(f"[green]✓[/green] Saved: {path}")
                progress.advance()
    """

    def __init__(
        self,
        total: int,
        label: str = "items",
        *,
        console: Console | None = None,
    ) -> None:
        self._console = console or _console
        self._total = total
        self._label = label
        self._progress: Progress | None = None
        self._task_id: int | None = None

    def __enter__(self) -> DreamInkProgress:
        self._progress = Progress(
            BarColumn(bar_width=BAR_WIDTH),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MofNCompleteColumn(),
            TextColumn("{task.fields[label]}"),
            SpinnerColumn(SPINNER_TYPE, style=SPINNER_STYLE),
            TextColumn("{task.fields[step]}"),
            console=self._console,
            refresh_per_second=REFRESH_PER_SECOND,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            "",
            total=self._total,
            label=self._label,
            step="",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._progress:
            self._progress.stop()

    def update_step(self, step: str) -> None:
        """Update the spinner text (e.g., 'Generating Watercolor...')."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, step=step)

    def advance(self) -> None:
        """Advance the progress bar by one unit."""
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id)

    def log(self, message: str) -> None:
        """Print a message above the live progress display."""
        if self._progress:
            self._progress.console.print(message)
