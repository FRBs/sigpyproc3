(install)=

# Installation

```{note}
**Requirements:** Python 3.12 or newer.
```

`sigpyproc` 2.0.0 is the stable release. Install it from PyPI, or from git until that upload is live.

---

## Quick Start (Recommended)

The recommended installation method uses [uv](https://docs.astral.sh/uv/):

```bash
pip install sigpyproc
```

Or, before the PyPI upload:

```bash
uv pip install git+https://github.com/FRBs/sigpyproc3
```

Verify the installation:

```bash
python -c "import sigpyproc; print(sigpyproc.__version__)"
```

After installation, browse the {doc}`generated/sigpyproc` for the full API reference,
or jump to {doc}`/tutorials/quickstart` (filterbanks) and
{doc}`/tutorials/search` (FFT / FFA / pulses).

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
- [seaborn](https://seaborn.pydata.org)
- [rich](https://rich.readthedocs.io) and [rich-click](https://github.com/ewels/rich-click)
- [click](https://click.palletsprojects.com/)
- [rocket-fft](https://github.com/styfenschaer/rocket-fft)
- [bidict](https://github.com/jab/bidict)
- [lazy-loader](https://github.com/scientific-python/lazy-loader)  

All required dependencies are installed automatically via `uv sync` or `uv pip install`.

---

## Next steps

- {doc}`/tutorials/quickstart` — load a filterbank, inspect the header, dedisperse.
- {doc}`/tutorials/search` — search a time series and work with `CandidateTable`.
- {doc}`/cmd` — every `spp_*` command.
- {doc}`/generated/sigpyproc` — full API.
