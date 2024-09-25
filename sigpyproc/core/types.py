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
