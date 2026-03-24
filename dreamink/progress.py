"""Rich-based progress indicators for DreamInk CLI commands."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
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
