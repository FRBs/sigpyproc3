---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started

New to sigpyproc? Start here. This page is about **objects and metadata**: opening a
file, reading the header, calling methods, and chaining steps. The
{doc}`filterbank quickstart <quickstart>` adds plots and a short search; other
tutorials go deeper on folding, viewers, and simulation.

Everything below uses a **small synthetic filterbank** in a temporary directory, so
you do not need the git checkout test files or your own observation files. The first
Numba call may take a moment to compile.

## Create a file to work with

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank, make_header

workspace = TemporaryDirectory()
path = Path(workspace.name) / "tutorial.fil"
header = make_header(
    filename=str(path),
    nchans=16,
    nbits=32,
    tsamp=0.001,
    tobs=4.096,
    fch1=1400.0,
    foff=-8.0,
)
_ = fake_filterbank(header, PulsarParams(period=0.128, dm=30.0, snr=12.0), seed=4)
print(path)
```

## Open a filterbank

Use {class}`~sigpyproc.readers.FilReader` for SIGPROC `.fil` files. A context
manager closes the file when you are done; for a quick look you can also keep the
reader open as in the cells below.

```{code-cell} ipython3
from rich.pretty import Pretty
from sigpyproc.readers import FilReader

myfil = FilReader(path)
myfil
```

Observational metadata lives on `myfil.header`. It is a
{class}`~sigpyproc.header.Header` instance — a structured object, not a plain
dict. `rich.pretty.Pretty` prints the fields in a readable layout:

```{code-cell} ipython3
Pretty(myfil.header)
```

```{code-cell} ipython3
type(myfil.header)
```

All values stored in the myFil.header attribute may be accessed as attributes, i.e.:

```{code-cell} ipython3
myfil.header.nchans
```

```{code-cell} ipython3
myfil.header.chan_freqs[[0, -1]]  # MHz, first and last channel
```

Field names match SIGPROC keys (`fch1`, `foff`, `tsamp`, …). Derived values such
as `chan_freqs` are computed when you ask for them.

## Dedisperse to a time series

Dedispersion removes frequency-dependent delays. The result is a
{class}`~sigpyproc.timeseries.TimeSeries`:

```{code-cell} ipython3
mytim = myfil.dedisperse(dm=30.0, quiet=True)
print(type(mytim))
mytim.data
```

```{code-cell} ipython3
float(mytim.data.sum()), float(mytim.data.max()), float(mytim.data.min())
```

`TimeSeries` holds a NumPy array in `.data` plus a matching `.header`. It is **not**
an `ndarray` subclass — pass `mytim.data` to NumPy and other libraries, as above.

## Fourier transform and spectrum

```{code-cell} ipython3
myfs = mytim.rfft()
print(type(myfs))
myfs.data
```

```{code-cell} ipython3
myfs_red = myfs.deredden()
myfs_red.data
```

```{code-cell} ipython3
myspec = myfs_red.form_spec(interpolate=True)
myspec.data
```

{class}`~sigpyproc.fourierseries.PowerSpectrum` helpers map physical units to FFT
bins (spacing is `1 / myspec.header.tobs` Hz):

```{code-cell} ipython3
myspec.period2bin(0.128), myspec.freq2bin(5.0)
```

Harmonic folding sums power at integer multiples of a trial frequency. The
injected pulsar period is 0.128 s:

```{code-cell} ipython3
folds = myspec.harmonic_fold(3)
len(folds), [fold.data.shape for fold in folds]
```

## Fold the filterbank

Folding stacks the band on a trial period and DM. Here we use the known injection
parameters (0.128 s, 30 pc cm⁻³):

```{code-cell} ipython3
cube = myfil.fold(
    period=0.128,
    dm=30.0,
    accel=0.0,
    nbins=32,
    nints=8,
    nbands=8,
    quiet=True,
)
type(cube), cube.data.shape
```

{class}`~sigpyproc.foldedcube.FoldedData` exposes slices and sums — for example
`get_profile()`, `get_subint()`, `get_subband()`, `get_time_phase()`, and
`get_freq_phase()`. Small DM and period tweaks are exploratory nudges, not a full
refold:

```{code-cell} ipython3
cube.update_dm(32.0)
cube.update_period(0.12801)
profile = cube.get_profile()
profile.data.shape, profile.tsamp
```

See {doc}`folding` for coordinates, archives, and figures.

## Tips in the REPL or a notebook

- **Tab completion** — after `myfil.` or `mytim.`, press Tab in IPython or
  Jupyter to list methods and attributes.
- **Docstrings** — `myfil.dedisperse?` or `help(myfil.dedisperse)` in
  an interactive session; API pages are in {doc}`../generated/sigpyproc`.
- **Chaining** — for quick experiments you can pipe steps on one line (keep longer
  workflows readable by splitting cells):

```{code-cell} ipython3
from sigpyproc.readers import FilReader

chained = (
    FilReader(path)
    .dedisperse(dm=30.0, quiet=True)
    .rfft()
    .deredden()
    .form_spec(interpolate=True)
)
chained.data.shape
```

## Next steps

- {doc}`quickstart` — waterfalls, dedispersed traces, and an FFT plot  
- {doc}`search` — candidate tables from periodic and single-pulse search  
- {doc}`viewers` — `spp_view` and batch PNG export  

```{code-cell} ipython3
:tags: [remove-cell]

myfil.close()
workspace.cleanup()
```
