# sigpyproc

**sigpyproc** is a FRB/pulsar data analysis library for Python. It provides an
OOP approach to pulsar data handling through the use of objects representing
different data types (e.g.
[SIGPROC filterbank](http://sigproc.sourceforge.net),
[PSRFITS](https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html),
time series, fourier series, etc.).

As pulsar data processing is often time critical, speed is maintained using
the excellent [numba](https://numba.pydata.org/) library.

`sigpyproc` is intended to be an Python alternative for the
[SIGPROC filterbank](http://sigproc.sourceforge.net) pulsar signal processing toolbox.
Over time it has also developed and become an independent project in its own right.
Unlike [SIGPROC](http://sigproc.sourceforge.net) and [PRESTO](https://github.com/scottransom/presto),
`sigpyproc` does not currently have full capabilities as a piece of FRB/pulsar searching software.
Instead, `sigpyproc` provides data manipulation routines which are well suited to preprocessing
and micro-management of pulsar data. The structure of the package also makes it an ideal development
environment, with a simple plug-and-play system with new modules and extensions.

[![GitHub CI](https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg)](https://github.com/FRBs/sigpyproc3/actions)
[![Docs](https://readthedocs.org/projects/sigpyproc3/badge/?version=latest)](https://sigpyproc3.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/FRBs/sigpyproc3/branch/main/graph/badge.svg)](https://codecov.io/gh/FRBs/sigpyproc3)
[![License](https://img.shields.io/github/license/FRBs/sigpyproc3)](https://github.com/FRBs/sigpyproc3/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Contents

```{toctree}
:caption: User Guide
:maxdepth: 1

install
cmd
modules
dev
changes
```

```{toctree}
:caption: Tutorials
:maxdepth: 1

tutorials/quickstart.ipynb
```
