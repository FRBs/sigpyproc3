import logging
from pathlib import Path

import pytest
from rich.logging import RichHandler

from sigpyproc.core.logging import LOG, configure_logging


@pytest.fixture(autouse=True)
def _reset_sigpyproc_logging() -> None:
    for hndlr in LOG.handlers[:]:
        LOG.removeHandler(hndlr)
        hndlr.close()
    LOG.setLevel(logging.NOTSET)


class TestConfigureLogging:
    def test_default_rich_handler(self) -> None:
        configure_logging()
        assert LOG.level == logging.INFO
        assert len([h for h in LOG.handlers if isinstance(h, RichHandler)]) == 1

    def test_verbose_level(self) -> None:
        configure_logging(logging.DEBUG)
        assert LOG.level == logging.DEBUG

    def test_idempotent(self) -> None:
        configure_logging()
        configure_logging()
        assert len([h for h in LOG.handlers if isinstance(h, RichHandler)]) == 1

    def test_force_replaces_rich_handler(self) -> None:
        configure_logging()
        first = next(h for h in LOG.handlers if isinstance(h, RichHandler))
        configure_logging(force=True)
        rich_handlers = [h for h in LOG.handlers if isinstance(h, RichHandler)]
        assert len(rich_handlers) == 1
        assert rich_handlers[0] is not first

    def test_log_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "sigpyproc.log"
        configure_logging(log_file=str(log_path))
        file_handlers = [h for h in LOG.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == str(log_path.resolve())
