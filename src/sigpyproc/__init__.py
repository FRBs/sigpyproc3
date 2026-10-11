"""sigpyproc: Python package for pulsar data analysis."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

import lazy_loader as lazy

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("sigpyproc")
except PackageNotFoundError:
    __version__ = "0+unknown"

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
