# Development guide

This page outlines the workflow and conventions for contributing to **sigpyproc**.

## Get a working tree

```bash
git clone https://github.com/FRBs/sigpyproc3.git
cd sigpyproc3
uv sync --extra tests --extra dev --extra docs
```

That pulls in the library, pytest, ruff, ty, and the Sphinx/MyST stack for
documentation.

## Usual workflow

Branch from `main`, make your change, add or adjust tests, run the checks below,
and open a PR. New public API should match existing patterns (data objects,
headers, streaming reads) and the full test suite should pass.

## Where new code goes

User-facing code lives on the data classes — {py:class}`~sigpyproc.base.Filterbank`,
{py:class}`~sigpyproc.timeseries.TimeSeries`, {py:class}`~sigpyproc.core.candidates.CandidateTable`,
and friends. Heavy lifting is in Numba kernels under `sigpyproc.kernels`.
Metadata stays on {py:class}`~sigpyproc.header.Header` and friends in `io/`.

Search is a good example of the split: `TimeSeries.search`, `find_pulses`, and
`PowerSpectrum.find_peaks` are methods on those objects, but the algorithms sit
in `sigpyproc.algorithms` on plain arrays. Single-pulse paths go through
`box.py` (PRESTO-style stages in `sps.py`, optional CPU Heimdall in
`heimdall.py`); periodic picking is in `period.py`; shared width banks and
validation are in `common.py`. Put the algorithm in the right module, keep
`apps/` and `io/` thin — `spp_search` is only Click glue.

## Example: a `bandpass()` method

Suppose you want total power per channel across the whole filterbank, returned
as a one-channel {py:class}`~sigpyproc.timeseries.TimeSeries`. That belongs on
{py:class}`~sigpyproc.base.Filterbank` because you still have frequency
resolution while streaming in time.

**API method** (on `Filterbank`):

```python
def bandpass(self, gulp: int = 16384, **plan_kwargs: Unpack[PlanKwargs]) -> TimeSeries:
    bpass_ar = np.zeros(self.header.nchans, dtype=np.float32)
    num_samples = 0

    for nsamps, _, data in self.read_plan(gulp=gulp, **plan_kwargs):
        kernels.extract_bpass(data, bpass_ar, self.header.nchans, nsamps)
        num_samples += nsamps

    bpass_ar /= num_samples

    hdr_changes = {"nchans": 1, "nsamples": len(bpass_ar)}
    return TimeSeries(bpass_ar, self.header.new_header(hdr_changes))
```

Stream blocks, call a kernel, return a new object with an updated header — same
shape as other filterbank methods.

**Kernel** in `sigpyproc/kernels/filterbank.py`:

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

Kernels stay small, side-effect free, and linear in memory. Prefer `parallel=True`
only when it actually helps.

## Checks before you push

From the repo root:

```bash
ruff check
ruff format --check
ty check
pytest -v
```

For coverage of compiled kernels, point Numba at a fresh cache each run:

```bash
NUMBA_CACHE_DIR="$(mktemp -d)" NUMBA_JIT_COVERAGE=1 MPLBACKEND=Agg NUMBA_NUM_THREADS=2 \
  pytest --cov=src --cov-report=html --cov-report=term-missing
```

CI expects all of the above to pass.

## Build the docs

```bash
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Open `docs/_build/html/index.html`. We treat warnings as errors (`-W`).

**Runnable tutorials** use MyST Notebook (`myst_nb`): front matter
`file_format: mystnb` and `{code-cell} ipython3` fences. Use `%matplotlib inline`
and `plt.show()` — Agg alone will not embed figures. `header.md` is static
Markdown; `io.md` runs against `tests/data/parkes_4bit.sf` in a git checkout.
Seeded sims and temp dirs keep builds reproducible; a failing cell fails the
build. Force a full re-execute with `sphinx-build -E -a ...`.

In MyST prose, write math as `$...$` or `$$...$$`, not `\(...\)`. Avoid fussy
LaTeX inside tables; put the formula in the next paragraph.

The README cards use PNGs exported from the viewer and folding notebooks. After
you change those tutorials:

```bash
python docs/export_figures.py \
  docs/_build/jupyter_execute/tutorials/viewers.ipynb \
  docs/_build/jupyter_execute/tutorials/folding.ipynb
sphinx-build -b html docs docs/_build/html -W --keep-going
```

The script reads cells tagged `preview-waterfall` and `preview-fold`; check
`docs/_static/figures/` and the rendered HTML.

## API vs narrative docs

- Docstrings in `src/sigpyproc/` (NumPy style) feed the API reference.
- Guides and tutorials are `docs/*.md`.
- Autosummary generates the per-module HTML under `generated/`.

## Bugs and feature requests

Use the [GitHub issue tracker](https://github.com/FRBs/sigpyproc3/issues). A
useful report has a small reproducer, Python version, OS, and a traceback if
something crashed.

## Further reading

The [Astropy developer guide](https://docs.astropy.org/en/latest/index_dev.html)
is a solid model for Python astronomy packaging and contribution norms.
