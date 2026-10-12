# sigpyproc

[![GitHub CI](https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg)](https://github.com/FRBs/sigpyproc3/actions)
[![Docs](https://readthedocs.org/projects/sigpyproc3/badge/?version=latest)](https://sigpyproc3.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/FRBs/sigpyproc3/branch/main/graph/badge.svg)](https://codecov.io/gh/FRBs/sigpyproc3)
[![License](https://img.shields.io/github/license/FRBs/sigpyproc3)](https://github.com/FRBs/sigpyproc3/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Scientific Python Ecosystem Coordination](https://img.shields.io/badge/SPEC-0,1-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/)

`sigpyproc` is a pulsar and FRB data analysis library for python. It provides an OOP approach to pulsar data handling through the use of
objects representing different data types (e.g. [SIGPROC filterbank](http://sigproc.sourceforge.net),
[PSRFITS](https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html), time-series, fourier-series, etc.).
As pulsar data processing is often time critical, speed is maintained using
the excellent [numba](https://numba.pydata.org) library.

## Installation

The package requires Python 3.12 or newer.

```bash
pip install sigpyproc
```

Until 2.0.0 is on PyPI, install the release candidate from git:

```bash
pip install -U git+https://github.com/FRBs/sigpyproc3
```

See the [installation page](https://sigpyproc3.readthedocs.io/en/latest/install.html) for a development checkout.

## Older releases

2.0.0 is the stable API. The ``legacy`` branch and the last 0.x release, 0.5.5, keep the previous interface.

## Usage

```python
from sigpyproc import FilReader, PFITSReader, TimeSeries

fil = FilReader("tutorial.fil")
fits = PFITSReader("tutorial.fits")

ts = TimeSeries.from_tim("observation.tim")
hits = ts.search(nfolds=4, pmax=10.0)
hits.cluster(snr_min=8.0).to_csv("cands.csv")
```

Check out the [filterbank quickstart](https://sigpyproc3.readthedocs.io/en/latest/tutorials/quickstart.html),
the [search tutorial](https://sigpyproc3.readthedocs.io/en/latest/tutorials/search.html),
and the [API docs](https://sigpyproc3.readthedocs.io) for more.

## Contributing

Check out [the developer documentation](https://sigpyproc3.readthedocs.io/en/latest/dev.html) for more info about getting started.
