"""Custom type aliases for sigpyproc."""

from typing import Literal

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
