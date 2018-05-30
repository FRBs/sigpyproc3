sigpyproc
=========

Installation
------------

### Requirements

	
        * numpy 
        * ctypes 
        * FFTW3
        * OpenMP

### Step-by-step guide

Once you have all the requirements installed, you can install this via pip:

```
pip install git+https://github.com/telegraphic/sigpyproc
```

Or, download / clone this repository, and then run

```
python setup.py install
```

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


[![Build Status](https://travis-ci.org/telegraphic/sigpyproc.svg?branch=master)](https://travis-ci.org/telegraphic/sigpyproc)


