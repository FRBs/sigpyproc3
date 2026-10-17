<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/logo-tagline-dark.svg">
    <img alt="sigpyproc: pulsar and FRB I/O" src="docs/_static/logo/logo-tagline.svg" width="480">
  </picture>
</p>

<div align="center">

[![GitHub CI](https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg)](https://github.com/FRBs/sigpyproc3/actions)
[![PyPI](https://img.shields.io/pypi/v/sigpyproc)](https://pypi.org/project/sigpyproc/)
[![Docs](https://readthedocs.org/projects/sigpyproc3/badge/?version=latest)](https://sigpyproc3.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/FRBs/sigpyproc3/branch/main/graph/badge.svg)](https://codecov.io/gh/FRBs/sigpyproc3)
[![License](https://img.shields.io/github/license/FRBs/sigpyproc3)](https://github.com/FRBs/sigpyproc3/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Scientific Python Ecosystem Coordination](https://img.shields.io/badge/SPEC-0,1-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/)

</div>

`sigpyproc` is a Python library for pulsar and FRB data. It reads [SIGPROC filterbank](http://sigproc.sourceforge.net), [PSRFITS](https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html), time series, and Fourier series as objects, and runs the numerical work with [Numba](https://numba.pydata.org/). Use it to inspect a file, try a processing step, and prototype a search ahead of a survey pipeline.

## Installation

Python 3.12 or newer, on Linux or macOS.

```bash
python -m pip install sigpyproc
```

`uv pip install sigpyproc` works the same way. The first Numba call compiles; later calls reuse that code. See the [installation page](https://sigpyproc3.readthedocs.io/en/latest/install.html) for a virtual environment or a development checkout.

## Usage

```python
from sigpyproc import FilReader

with FilReader("observation.fil") as obs:
    print(obs.header)
    series = obs.dedisperse(dm=30.0)

candidates = series.search(nfolds=4, pmax=10.0)
candidates.cluster(snr_min=8.0).to_csv("candidates.csv")
```

From the terminal:

```bash
spp_header print observation.fil
spp_view observation.fil
spp_search fft observation.tim --nfolds 4 --pmax 10 -o candidates.csv
```

Each `spp_*` command has `--help`. The [command reference](https://sigpyproc3.readthedocs.io/en/latest/cmd.html) lists the rest. The [quickstart](https://sigpyproc3.readthedocs.io/en/latest/tutorials/quickstart.html) runs the Python example on a synthetic file.

## Older releases

2.0.0 is the stable API. The `legacy` branch and the last 0.x release, 0.5.5, keep the previous interface.

## Contributing

See the [developer documentation](https://sigpyproc3.readthedocs.io/en/latest/dev.html). sigpyproc is released under the [MIT license](https://github.com/FRBs/sigpyproc3/blob/main/LICENSE).
