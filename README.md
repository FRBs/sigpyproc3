# sigpyproc

`sigpyproc` is a pulsar and FRB data analysis library for python.

[![GitHub CI](https://github.com/FRBs/sigpyproc3/workflows/GitHub%20CI/badge.svg)](https://github.com/FRBs/sigpyproc3/actions)
[![Docs](https://readthedocs.org/projects/sigpyproc3/badge/?version=latest)](https://sigpyproc3.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/FRBs/sigpyproc3/branch/master/graph/badge.svg)](https://codecov.io/gh/FRBs/sigpyproc3)

## Usage

```python
from sigpyproc.Readers import FilReader
myFil = FilReader("tutorial.fil")

```

## Installation

You need Python 3.6 or later to run sigpyproc. Additionally,
[FFTW3](http://www.fftw.org) and [OpenMP](https://www.openmp.org)
should be installed/enabled on your system.

### Step-by-step guide

Once you have all the requirements installed, you can install this via pip:

```bash
pip install git+https://github.com/FRBs/sigpyproc3
```

Or, download / clone this repository, and then run

```bash
python -m pip install .
```

<!---
### Docker

This repo now comes with a `Dockerfile`, so you can build a simple docker container with `sigpyproc` in it. To do so, clone this directory, cd into it, and then run on your command line:

```
docker build --tag sigpyproc .
```

You can then run the container with

```
docker run --rm -it sigpyproc
```

(Have a read of docker tutorials and documentation for more details!)
--->
