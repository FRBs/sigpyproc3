# Development Guide

This page outlines the workflow, conventions, and extension points for contributing to **sigpyproc**.

---

## Development Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/FRBs/sigpyproc3.git
cd sigpyproc3
uv sync --extra dev --extra docs
```

This installs:

- Runtime dependencies  
- Test suite requirements  
- Linting and static typing tools  
- Documentation build dependencies  

---

## Contribution Workflow

Typical development cycle:

1. Create a feature branch.
2. Implement changes.
3. Add or update tests.
4. Run quality checks.
5. Build documentation locally to verify changes.
6. Open a pull request.

All contributions should maintain API consistency and pass the full test suite.

---

## Extending sigpyproc

The architecture separates:

- **User-facing API** (high-level classes)
- **Core numerical kernels** (Numba-accelerated)
- **Structured metadata handling**

New functionality typically involves:

1. Adding a method to a data object (e.g. {py:class}`~sigpyproc.base.Filterbank` class)
2. Implementing or extending a Numba kernel
3. Writing tests
4. Documenting the public API

---

## Example: Adding a new function: `bandpass()`

**Goal:** Add a new function called `bandpass`, which will return the total power
as a function of frequency for the filterbank data.

### Step 1 — Add the API Method

As this function will run on data with both time and frequency resolution, it belongs in the
{py:class}`~sigpyproc.base.Filterbank` class.

```python
def bandpass(self, gulp: int = 16384, **plan_kwargs: Unpack[PlanKwargs]) -> TimeSeries:
    bpass_ar = np.zeros(self.header.nchans, dtype=np.float32)
    num_samples = 0

    for nsamps, _, data in self.read_plan(**plan_kwargs):
        kernels.extract_bpass(data, bpass_ar, self.header.nchans, nsamps)
        num_samples += nsamps

    bpass_ar /= num_samples

    hdr_changes = {"nchans": 1, "nsamples": len(bpass_ar)}
    return TimeSeries(bpass_ar, self.header.new_header(hdr_changes))
```

This method:

- Streams data in blocks  
- Calls a compiled kernel  
- Returns a new {py:class}`~sigpyproc.timeseries.TimeSeries` instance  

Public-facing methods should remain clean and explicit.

---

### Step 2 — Implement the Numba Kernel

Add the corresponding kernel to `sigpyproc/core/kernels.py`:

```python
@njit(
    ["void(u1[:], f4[:], i4, i4)", "void(f4[:], f4[:], i4, i4)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def extract_bpass(inarray, outarray, nchans, nsamps):
    for ichan in prange(nchans):
        for isamp in range(nsamps):
            outarray[ichan] += inarray[nchans * isamp + ichan]
```

This kernel:

- Accumulates power along the time axis  
- Uses `parallel=True` to enable Numba's parallel execution  
- Is intentionally simple and memory-linear

We keep kernels small, predictable, and side-effect free.

---

## Quality Checks

From the repository root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

All checks must pass before submitting a pull request.

---

## Building Documentation

```bash
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Open `docs/_build/html/index.html` in your browser to verify the changes.
Documentation builds must complete without warnings.

---

## Reporting Issues

Please report bugs or feature requests via the [GitHub issue tracker](https://github.com/FRBs/sigpyproc3/issues).

Include:

- Minimal reproducible example  
- Python version  
- Operating system  
- Relevant stack traces  

---

## Documentation Policy

- API documentation lives in `src/sigpyproc/**` using NumPy-style docstrings.
- Narrative documentation lives in `docs/*.md`.
- Per-module API pages are generated via autosummary.

---

## Contributing code or documentation

An excellent place to start is the
[AstroPy developer docs](https://docs.astropy.org/en/latest/index_dev.html).
