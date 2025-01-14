from typing import Literal

FilterMethods = Literal["median", "mean"]
LocMethods = Literal["median", "mean"]
ScaleMethods = Literal[
    "iqr",
    "mad",
    "doublemad",
    "diffcov",
    "biweight",
    "qn",
    "sn",
    "gapper",
]
MaskMethods = Literal["iqrm", "mad"]
SpecSimulMethods = Literal[
    "flat",
    "power_law",
    "smooth_envelope",
    "gaussian",
    "polynomial_peaks",
    "scintillation",
    "gaussian_blobs",
    "random",
]
