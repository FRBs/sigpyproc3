"""Kernel functions for sigpyproc (Numba JIT accelerated).

This package contains all computational Numba kernels for sigpyproc, lazily
loaded following Scientific Python SPEC 1.
"""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
