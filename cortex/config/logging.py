"""Where the structlog level is decided.

Nothing configured structlog before this, so every entry point inherited the library
default, which prints everything including `debug`. That is the right default for a
developer watching their own run: `llm.cache` emits one line per request, and it is what
makes cache behaviour attributable to a *call* rather than to a run.

It is the wrong default for exactly one situation — a run somebody else will read. In a
recorded terminal session the per-request diagnostics interleave with the report itself,
and a reader who does not know the codebase cannot tell the instrumentation from the
answer. `CORTEX_LOG_LEVEL=info` turns them off for that run only.

Deliberately not a CLI flag. The level is a property of how a process is being run, not of
the question being asked, and a flag would have to be threaded through every entry point
that might one day be recorded.
"""

from __future__ import annotations

import logging

import structlog

from cortex.config.settings import get_settings

#: structlog's filtering wrapper takes a stdlib level number, not a name.
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging() -> None:
    """Apply `CORTEX_LOG_LEVEL`. Safe to call more than once."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(_LEVELS[get_settings().log_level])
    )
