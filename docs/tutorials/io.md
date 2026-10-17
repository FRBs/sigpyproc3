---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Read and convert PSRFITS

This page walks through **search-mode PSRFITS** — the `.fits` / `.sf` files
pulsar backends often write — and converting one to SIGPROC with `digifil`.
Use your own observation the same way; keep the original file and pick a new
output name.

sigpyproc reads **single-file search mode** only. Fold-mode files, irregular
channel grids, row gaps, and other odd layouts are rejected instead of
guessing a time axis. You need at least two frequency channels.

The runnable cells below use the Parkes test file shipped in the repository
(`tests/data/parkes_4bit.sf`). That path exists in a git checkout and on
Read the Docs; it is **not** installed with the PyPI wheel. For your data,
point `PFITSReader` at your `.sf` or `.fits` file.

## Find the sample file and peek at the header

From a clone you can run:

```bash
spp_header print tests/data/parkes_4bit.sf
spp_digifil --help
```

The next cell prints the same summary the `print` command uses (not the longer
`--long` pretty-print).

```{code-cell} ipython3
import os
from pathlib import Path

from sigpyproc.header import Header

SF = Path(os.environ["SIGPYPROC_PROJECT_ROOT"]) / "tests/data/parkes_4bit.sf"
if not SF.is_file():
    msg = f"Sample file missing: {SF} (need a full git checkout to run this page)"
    raise FileNotFoundError(msg)

print(Header.from_pfits(SF).to_string())
```

## Read a short stretch of data

`DAT_FREQ` sets channel order. `DAT_SCL`, `DAT_OFFS`, and `DAT_WTS` enter the
calibration when samples are read. Defaults and knobs live on
{py:class}`~sigpyproc.io.pfits.PFITSFile` and
{py:class}`~sigpyproc.io.digifil.DigifilConfig`.

```{code-cell} ipython3
import numpy as np
from sigpyproc import PFITSReader

WINDOW = 512

with PFITSReader(SF) as observation:
    header = observation.header
    nread = min(WINDOW, header.nsamples)
    block = observation.read_block(0, nread)

print(f"{header.nchans} channels, {header.nsamples} samples in the file")
print(f"tsamp = {header.tsamp:g} s, tstart = {header.tstart}")
print(f"Band edges (MHz): {header.chan_freqs[0]:.2f} … {header.chan_freqs[-1]:.2f}")
print(f"Window shape {block.data.shape}  (channels × samples)")
assert np.isfinite(block.data).all()
```

```{code-cell} ipython3
%matplotlib inline

import matplotlib.pyplot as plt
from sigpyproc.viz.scaling import frequency_oriented


def plot_window(block, title: str) -> None:
    """Waterfall for a (nchan, nsamp) block — shared by PSRFITS and SIGPROC."""
    t0, t1 = 0.0, block.nsamples * block.header.tsamp
    image, f_lo, f_hi = frequency_oriented(
        block.data, block.header.chan_freqs, foff=block.header.foff,
    )
    nchan, nsamp = image.shape
    time_edges = np.linspace(t0, t1, nsamp + 1)
    freq_edges = np.linspace(f_lo, f_hi, nchan + 1)
    vmin, vmax = np.percentile(block.data, [2, 98])

    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    mesh = ax.pcolormesh(
        time_edges, freq_edges, image, shading="flat", cmap="magma", vmin=vmin, vmax=vmax,
    )
    ax.set(xlabel="Time from window start (s)", ylabel="Frequency (MHz)", title=title)
    fig.colorbar(mesh, ax=ax, label="Intensity")
    plt.show()
    plt.close(fig)


plot_window(block, title=f"PSRFITS — first {nread} samples of {SF.name}")
```

## Convert to SIGPROC and spot-check

For a first conversion, float32 output with **no running rescale** avoids
extra quantisation while you compare headers:

```{code-cell} ipython3
from tempfile import TemporaryDirectory

from sigpyproc import FilReader
from sigpyproc.io.digifil import DigifilConfig, digifil

workspace = TemporaryDirectory()
converted = Path(workspace.name) / "parkes_4bit.fil"

digifil(
    str(SF),
    str(converted),
    config=DigifilConfig(nbits=-32, rescale_seconds=0.0),
)

with FilReader(converted) as out:
    print(f"Wrote {converted.name}")
    print(f"tstart {out.header.tstart}, tsamp {out.header.tsamp:g} s")
    print(f"MHz: {out.header.chan_freqs[0]:.2f} … {out.header.chan_freqs[-1]:.2f}")
    print(f"{out.header.nsamples} samples in the .fil")
```

The converter writes a **negative** `foff`, reversing ascending input channels.
Frequency scrunch must divide the input channel count (zero means no scrunch).
Time scrunch can drop a partial group at the end, so the output length may be
slightly shorter than a naive estimate.

## Read the same window from the `.fil`

After conversion, `FilReader` is the usual entry point. The block below reads
the same number of samples from the start of the file and plots them with the
same helper as above so you can compare PSRFITS and SIGPROC side by side in
the built docs.

```{code-cell} ipython3
with FilReader(converted) as observation:
    fil_header = observation.header
    fil_nread = min(WINDOW, fil_header.nsamples)
    fil_block = observation.read_block(0, fil_nread)

print(f"{fil_header.nchans} channels, {fil_header.nsamples} samples in the .fil")
print(f"tsamp = {fil_header.tsamp:g} s, tstart = {fil_header.tstart}")
print(f"Band edges (MHz): {fil_header.chan_freqs[0]:.2f} … {fil_header.chan_freqs[-1]:.2f}")
print(f"Window shape {fil_block.data.shape}")
assert np.isfinite(fil_block.data).all()
```

```{code-cell} ipython3
plot_window(fil_block, title=f"SIGPROC — first {fil_nread} samples of {converted.name}")
```

## Other formats and header quirks

SIGPROC supports 1/2/4/8/16/32-bit output. Signed 8-bit keeps sign; a 16-bit
**numerical** stream is promoted to float32 (every `uint16` value is exact).
The [header rules](https://github.com/FRBs/sigpyproc3/blob/main/docs/header.md)
in the repo explain which metadata is on disk versus inferred.

Packed 2- and 4-bit SIGPROC samples are LSB-first. If the last packing group is
incomplete, conversion raises rather than dropping samples silently.

## Next steps

- {doc}`header` — edit SIGPROC headers after conversion  
- {doc}`quickstart` — dedisperse and plot a `.fil`  
- {doc}`../cmd` — `spp_digifil` flags  

```{code-cell} ipython3
:tags: [remove-cell]

plt.close("all")
workspace.cleanup()
```
