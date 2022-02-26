# sigpyproc

[![GitHub CI](https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg)](https://github.com/FRBs/sigpyproc3/actions)
[![Docs](https://readthedocs.org/projects/sigpyproc3/badge/?version=latest)](https://sigpyproc3.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/FRBs/sigpyproc3/branch/main/graph/badge.svg)](https://codecov.io/gh/FRBs/sigpyproc3)
[![License](https://img.shields.io/github/license/FRBs/sigpyproc3)](https://github.com/FRBs/sigpyproc3/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`sigpyproc` is a pulsar and FRB data analysis library for python. It provides an OOP approach to pulsar data handling through the use of
objects representing different data types (e.g. [SIGPROC filterbank](http://sigproc.sourceforge.net),
[PSRFITS](https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html), time-series, fourier-series, etc.).
As pulsar data processing is often time critical, speed is maintained using
the excellent [numba](https://numba.pydata.org) library.

## Installation

The quickest way to install the package is to use [pip](https://pip.pypa.io):

```bash
pip install -U git+https://github.com/FRBs/sigpyproc3
```

Note that you will need Python (>=3.8) installed to use `sigpyproc`.
Also check out the [installation documentation page](https://sigpyproc3.readthedocs.io/en/latest/install.html) for more options.

## Legacy Version

`sigpyproc` is currently undergoing major developements which will modify the existing API in order to be a modern python
replacemet for [SIGPROC](http://sigproc.sourceforge.net). To use the older API, you can install the ``legacy``
branch of this repo, or install the last released version 0.5.5.

## Usage

```python
from sigpyproc.readers import FilReader, PFITSReader

fil = FilReader("tutorial.fil")
fits = PFITSReader("tutorial.fits")
```

Check out the tutorials and API docs on [the docs page](https://sigpyproc3.readthedocs.io) for example usage and more info.

## Contributing

Check out [the developer documentation](https://sigpyproc3.readthedocs.io/en/latest/dev.html) for more info about getting started.
