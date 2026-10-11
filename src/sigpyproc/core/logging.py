"""Logging configuration for sigpyproc."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from sigpyproc.utils import validate_path

LOG = logging.getLogger("sigpyproc")

_LOG_FORMAT = "- %(name)s - %(message)s"
_RICH_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_LOG_FORMAT)


def _rich_handler() -> RichHandler:
    handler = RichHandler(
        show_path=False,
        rich_tracebacks=True,
        log_time_format=_RICH_TIME_FORMAT,
    )
    handler.setFormatter(_formatter())
    return handler


def _has_rich_handler(logger: logging.Logger) -> bool:
    return any(isinstance(hndlr, RichHandler) for hndlr in logger.handlers)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """Configure Rich logging for the sigpyproc package.

    Parameters
    ----------
    level : int or str, optional
        Logging level for the ``sigpyproc`` logger, by default ``logging.INFO``.
    log_file : str, optional
        Optional path to append plain-text log output.
    force : bool, optional
        If True, replace existing Rich handlers on the package logger.
    """
    pkg = LOG
    pkg.setLevel(level)

    if force:
        for hndlr in pkg.handlers[:]:
            if isinstance(hndlr, RichHandler):
                pkg.removeHandler(hndlr)
                hndlr.close()

    if not _has_rich_handler(pkg):
        for hndlr in pkg.handlers[:]:
            if isinstance(hndlr, logging.NullHandler):
                pkg.removeHandler(hndlr)
        pkg.addHandler(_rich_handler())

    if log_file:
        log_path = validate_path(log_file, exists=False, writable=True)
        path_str = log_path.as_posix()
        if not any(
            isinstance(hndlr, logging.FileHandler) and hndlr.baseFilename == path_str
            for hndlr in pkg.handlers
        ):
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(_formatter())
            pkg.addHandler(file_handler)
