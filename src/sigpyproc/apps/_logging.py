"""Shared CLI logging options for spp_* entrypoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

import rich_click as click

from sigpyproc.logging import configure_logging

C = TypeVar("C", bound=Callable[..., object])


def setup_cli_logging(verbose: int, *, quiet: bool) -> None:
    """Map Click verbosity flags to :func:`sigpyproc.logging.configure_logging`."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO
    configure_logging(level)


def logging_options[C: Callable[..., object]](cmd: C) -> C:
    """Attach ``-v`` / ``--verbose`` and ``-q`` / ``--quiet`` to a Click command."""
    cmd = click.option(
        "-q",
        "--quiet",
        is_flag=True,
        default=False,
        help="Only warnings and errors (overrides --verbose).",
    )(cmd)
    return click.option(
        "-v",
        "--verbose",
        count=True,
        default=0,
        help="Increase logging verbosity (-v → DEBUG).",
    )(cmd)
