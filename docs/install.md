(install)=

# Installation

```{note}
**Requirements:** Python 3.12 or newer.
```

`sigpyproc` is not currently published on PyPI. Installation is performed directly from the GitHub repository.

---

## Quick Start (Recommended)

The recommended installation method uses [uv](https://docs.astral.sh/uv/):

```bash
uv pip install git+https://github.com/FRBs/sigpyproc3
```

Verify the installation:

```bash
python -c "import sigpyproc; print(sigpyproc.__version__)"
```

After installation, browse the {doc}`generated/sigpyproc` for the full API reference.

---

## Development Installation

Clone the repository and install development dependencies:

```bash
git clone https://github.com/FRBs/sigpyproc3.git
cd sigpyproc3
uv sync --extra tests --extra dev --extra docs
```

This installs:

- Core runtime dependencies  
- Testing tools  
- Linting and formatting tools  
- Documentation build requirements  

---

## Running the Test Suite

From the repository root:

```bash
pytest --cov=src --cov-report=html -v
```

All tests should pass on a supported Python version.

---

## Building Documentation Locally

To build the documentation:

```bash
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Open `docs/_build/html/index.html` in your browser.

---

## Core Dependencies

sigpyproc depends on:

- [NumPy](https://numpy.org)  
- [Numba](https://numba.pydata.org)  
- [Astropy](https://www.astropy.org)  
- [Matplotlib](https://matplotlib.org)  
- [bottleneck](https://bottleneck.readthedocs.io)
- [h5py](https://docs.h5py.org/en/stable/)  
- [attrs](https://attrs.org)
- [rich](https://rich.readthedocs.io)
- [click](https://click.palletsprojects.com/)  

All required dependencies are installed automatically via `uv sync` or `uv pip install`.
