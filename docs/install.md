(install)=

# Installation

```{note}
**Requirements:** Python 3.12 or newer.
```

The 2.0 release requires Python 3.12 or newer on Linux or macOS. It ships a
Python wheel, with native numerical dependencies supplied by platform wheels.

## Install from PyPI

Install sigpyproc in your Python environment:

```bash
python -m pip install sigpyproc
```

If you manage environments with [uv](https://docs.astral.sh/uv/), use
`uv pip install sigpyproc` inside an activated environment.

For a fresh environment with Python's standard tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install sigpyproc
```

Runtime dependencies are installed automatically. Numba compiles numerical routines on first use, so an
initial call can take longer than subsequent calls.

Verify the installation:

```bash
python -c "import sigpyproc; print(sigpyproc.__version__)"
```

After installation, browse the {doc}`generated/sigpyproc` for the full API reference,
or jump to {doc}`/tutorials/basics` (objects and headers),
{doc}`/tutorials/quickstart` (plots), and {doc}`/tutorials/search` (FFT / pulses).

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
pytest -v
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

- {doc}`/tutorials/basics` — `FilReader`, headers, dedisperse, FFT, fold.
- {doc}`/tutorials/quickstart` — plots and a short periodicity search.
- {doc}`/tutorials/search` — search a time series and work with `CandidateTable`.
- {doc}`/tutorials/fake` — write a mock FRB, or inject one into a filterbank.
- {doc}`/cmd` — every `spp_*` command.
- {doc}`/generated/sigpyproc` — full API.
