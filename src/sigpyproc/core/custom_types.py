"""Custom type aliases and protocols for sigpyproc."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Buffer, Callable, Iterator

    import numpy as np

    from sigpyproc.header import Header

type FilterMethods = Literal["median", "mean"]
type LocMethods = Literal["median", "mean"]
type ScaleMethods = Literal[
    "std",
    "iqr",
    "mad",
    "doublemad",
    "diffcov",
    "biweight",
    "qn",
    "sn",
    "gapper",
]
type MaskMethods = Literal["iqrm", "mad"]
type SpecSimulMethods = Literal[
    "flat",
    "power_law",
    "smooth_envelope",
    "gaussian",
    "polynomial_peaks",
    "scintillation",
    "gaussian_blobs",
    "random",
]
type MatchFilterMethods = Literal["boxcar", "gaussian", "lorentzian"]


@runtime_checkable
class PlanReader(Protocol):
    """Protocol for filterbank streaming readers yielding plan chunks."""

    @property
    def header(self) -> Header: ...

    def read_plan(
        self,
        gulp: int = 16384,
        start: int = 0,
        nsamps: int | None = None,
        skipback: int = 0,
        description: str | None = None,
        *,
        quiet: bool = False,
        allocator: Callable[[int], Buffer] | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]: ...
