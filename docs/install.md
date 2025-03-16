(install)=

# Installation

:::{note}
`sigpyproc` requires Python 3.11 and later.
:::

## Using pip

The recommended method of installing *sigpyproc* is with [pip](https://pip.pypa.io):

```bash
python -m pip install -U git+https://github.com/FRBs/sigpyproc3
```

Some of the required dependencies are:

- [numpy](https://numpy.org)
- [numba](https://numba.pydata.org)
- [astropy](https://www.astropy.org)
- [matplotlib](https://matplotlib.org)
- [bottleneck](https://bottleneck.readthedocs.io)
- [h5py](https://docs.h5py.org/en/stable/)
- [attrs](https://attrs.org)
- [rich](https://rich.readthedocs.io)
- [click](https://click.palletsprojects.com/en/latest/)

(source)=

## From Source

The source code for *sigpyproc* can be downloaded and installed [from GitHub](https://github.com/FRBs/sigpyproc3) by running

```bash
git clone https://github.com/FRBs/sigpyproc3.git
cd sigpyproc3
python -m pip install -e .
```

## Testing

To run the unit tests with [pytest](https://docs.pytest.org),
first install the testing dependencies using pip:

```bash
python -m pip install -e ".[tests]"
```

and then execute:

```bash
python -m pytest -v tests
```

Normally, all of the tests should pass.
