"""Shared CLI logging options for spp_* entrypoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

import rich_click as click

from sigpyproc.core.logging import configure_logging

C = TypeVar("C", bound=Callable[..., object])


def setup_cli_logging(verbose: int = 0, *, quiet: bool = False) -> None:
    """Map Click verbosity flags to :func:`sigpyproc.core.logging.configure_logging`."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO
    configure_logging(level)


def _guard_invoke(invoke: Callable[..., object]) -> Callable[..., object]:
    def wrapped(ctx: click.Context) -> object:
        try:
            return invoke(ctx)
        except click.ClickException:
            raise
        except (ValueError, IndexError, OSError, NotImplementedError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapped


def logging_options[C: Callable[..., object]](cmd: C) -> C:
    """Attach ``-v`` / ``--verbose`` and ``-q`` / ``--quiet`` to a Click command.

    ValueError, IndexError, OSError, and NotImplementedError raised by the
    command become a Click error so the process exits with status 1.
    """
    if isinstance(cmd, click.Command):
        cmd.invoke = _guard_invoke(cmd.invoke)  # ty: ignore[invalid-assignment]
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
