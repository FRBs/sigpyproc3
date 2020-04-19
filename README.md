# sigpyproc

`sigpyproc` is a pulsar and FRB data analysis library for python.

[![Build Status](https://travis-ci.com/pravirkr/sigpyproc3.svg?branch=master)](https://travis-ci.com/pravirkr/sigpyproc3)


Usage
-----

```python
from sigpyproc.Readers import FilReader
myFil = FilReader("tutorial.fil")

```

Installation
------------

### Requirements

You need Python 3.6 or later to run sigpyproc.

    * FFTW3
    * OpenMP

### Step-by-step guide

Once you have all the requirements installed, you can install this via pip:

```
pip install git+https://github.com/pravirkr/sigpyproc3
```

Or, download / clone this repository, and then run

```
python setup.py install
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