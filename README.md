# sigpyproc

[![GitHub CI](https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg)](https://github.com/FRBs/sigpyproc3/actions)
[![Docs](https://readthedocs.org/projects/sigpyproc3/badge/?version=latest)](https://sigpyproc3.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/FRBs/sigpyproc3/branch/master/graph/badge.svg)](https://codecov.io/gh/FRBs/sigpyproc3)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`sigpyproc` is a pulsar and FRB data analysis library for python. It provides an OOP approach to pulsar data handling through the use of objects representing different data types (e.g. [SIGPROC filterbank](http://sigproc.sourceforge.net), time-series, fourier-series, etc.).

As pulsar data processing is often time critical, speed is maintained through the use of compiled C++ code that are accessed via the excellent `pybind11 <https://pybind11.readthedocs.io/>`_ library. Additional performance increases are obtained via the use of multi-threading with OpenMP, a threading library standard to most linux and mac systems.

`sigpyproc` was initially intended to be an Python wrapper for the [SIGPROC](http://sigproc.sourceforge.net) pulsar signal processing toolbox, but over time it has developed and become an independent project in its own right. Unlike [SIGPROC](http://sigproc.sourceforge.net) and [PRESTO](https://github.com/scottransom/presto), `sigpyproc` does not currently have full capabilities as a piece of FRB/pulsar searching software. Instead, `sigpyproc` provides data manipulation routines which are well suited to preprocessing and micro-management of pulsar data. The structure of the package also makes it an ideal development environment, with a simple plug-and-play system with new modules and extensions.

## Basic Usage

```python
from sigpyproc.readers import FilReader
myFil = FilReader("tutorial.fil")

```

## Installation

You need Python 3.8 or later to run sigpyproc.
Install via pip:

```bash
pip install git+https://github.com/FRBs/sigpyproc3
```

Or, download / clone this repository, and then run

```bash
python -m pip install .
```

### Test the installation

You can execute some unit and benchmark tests using [pytest](https://docs.pytest.org) to make sure that the installation went alright. In the root directory of the source code
execute the following command:

```bash
python -m pytest -v tests
```
