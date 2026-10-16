---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(search-tutorial)=

# Searching time series

Once you have a dedispersed time series — usually from
{py:meth}`~sigpyproc.base.Filterbank.dedisperse` or a SIGPROC `.tim` file —
you can hunt for periodic power with an FFT or for individual bursts with a
boxcar search. Results land in one {py:class}`~sigpyproc.core.candidates.CandidateTable`
that you can filter, cluster, and export to CSV. The `spp_search` commands call
the same methods shown here; they are a convenience layer, not a separate
search engine.

This page picks up where {doc}`/tutorials/quickstart` leaves off. It describes a
modern Python API inspired by SIGPROC's `seek` / `best` / `peak`, but it is not
a drop-in replacement for those tools.

---

## What maps to what

| SIGPROC | sigpyproc | Notes |
| --- | --- | --- |
| `seek` | {py:meth}`~sigpyproc.timeseries.TimeSeries.search` / `spp_search fft` | Whitened FFT + harmonic folds. No `.prd`, no birdie files. |
| `find` | same as `spp_search fft` | Historical `find` was renamed `seek`; we do not ship `spp_find`. |
| `best` | {py:meth}`~sigpyproc.core.candidates.CandidateTable.cluster` / `spp_search cluster` | Harmonic families only. No PGPLOT, no `.bst` / `.fld`. |
| `peak` | {py:meth}`~sigpyproc.timeseries.TimeSeries.find_pulses` / `spp_search pulses` | Selectable PRESTO-inspired and original matched-filter boxcar searches. |
| `giant` | — | Interactive PGPLOT viewer; out of scope. |

Exports are **CSV**, not `.prd`.

## Load a time series

From a `.tim` in the repo test data:

```python
from sigpyproc.timeseries import TimeSeries

ts = TimeSeries.from_tim("tests/data/GBT_J1807-0847.tim")
ts.header.tsamp, ts.nsamples, ts.header.dm
```

From a filterbank, collapse to one DM first (see {doc}`/tutorials/quickstart`):

```python
from sigpyproc.readers import FilReader

ts = FilReader("observation.fil").dedisperse(dm=30.0)
ts.to_tim("observation.tim")  # optional; search also accepts the object
```

You can also build a series in memory. That is the easiest way to
experiment:

```python
import numpy as np
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries

nsamp, tsamp, period = 2**14, 1e-3, 0.05
rng = np.random.default_rng(0)
data = rng.normal(0.0, 1.0, nsamp).astype(np.float32)
t = np.arange(nsamp) * tsamp
data += 3.0 * np.sin(2 * np.pi * t / period).astype(np.float32)

header = Header(
    filename="demo.tim",
    rawdatafile="demo.tim",
    data_type="time series",
    nchans=1,
    foff=1.0,
    fch1=1400.0,
    nbits=32,
    tsamp=tsamp,
    tstart=50000.0,
    nsamples=nsamp,
)
ts = TimeSeries(data, header)
```

## FFT search

{py:meth}`~sigpyproc.timeseries.TimeSeries.search` follows the same steps SIGPROC
`seek` used, in order:

1. Optional acceleration resample
2. Real FFT
3. {py:meth}`~sigpyproc.fourierseries.FourierSeries.deredden` (running
   median on the **complex** series — not
   {py:meth}`~sigpyproc.timeseries.TimeSeries.deredden`)
4. Amplitude spectrum, with inter-bin interpolation by default
5. Harmonic folds $2, 4, \ldots, 2^{N}$ **in addition to** the unfolded spectrum
6. Local-maxima peak pick → one {py:class}`~sigpyproc.core.candidates.CandidateTable`

```python
hits = ts.search(nfolds=4, pmax=10.0, snr_min=6.0)
print(len(hits), hits)
print(hits.values("period_s")[:5])
print(hits.values("snr")[:5])
print(hits.table["nharm"][:5])
```

`nfolds` counts **extra** harmonic stacks, not “search harmonics 1 through n”.
With `nfolds=4` you get the unfolded spectrum plus folds at 2, 4, 8, and 16.
How those folds are combined depends on `method`:

- `lyne` (default): SIGPROC Lyne-Ashworth stretch, with a half-bin neighbour add.
- `simple`: integer bins `h*n` only. No drift. Fast, and the least sensitive off-bin.
- `presto`: stock PRESTO sum, even harmonics, nearest bin.
- `greedy`: follow the louder neighbour as the harmonic order increases.

Everything stays on the original Fourier grid, so a sharp pulse can still
peak on a harmonic of the true period. For one-off bursts, use `find_pulses` instead.

### Cluster harmonic families

One pulsar often appears several times — $P$, $P/2$, $2P$, and so on.
{py:meth}`~sigpyproc.core.candidates.CandidateTable.cluster` keeps the strongest
row in each $k/n$ family (by default $k, n \le 16$):

```python
best = hits.cluster(snr_min=8.0, period_tol=1e-4)
best.to_csv("fft_clustered.csv")
```

Single-pulse rows (`kind="pulse"`) are never treated as harmonics of
each other. Slice first if you want to cluster only FFT hits:

```python
fft_only = hits[hits.values("kind") == "fft"]
```

## Single pulses

{py:meth}`~sigpyproc.timeseries.TimeSeries.find_pulses` follows the broad
idea of [PRESTO’s single-pulse search](https://github.com/scottransom/presto/blob/master/bin/single_pulse_search.py):
baseline removal, block noise estimates, boxcars at several widths, then prune
weak neighbours. Normalisation and tie-breaking differ from stock PRESTO.

```python
pulses = ts.find_pulses(snr_min=6.0, max_width=256)
fft_pulses = ts.find_pulses(snr_min=6.0, max_width=256, engine="fft")
original = ts.find_pulses(method="matched_filter", max_width=256)
```

Both PRESTO engines compute the same statistic:

$$
\mathrm{SNR} = \frac{\sum_{i \in \mathrm{window}} z_i}{\sqrt{w}},
$$

where samples $z_i$ are baseline-subtracted and normalized by their block's
noise estimate, and $w$ is the width in samples.

### Widths and preprocessing

The width bank is a **ceiling**, not a promise to search that exact
width.

```python
# Same bank for comparisons between pipelines.
widths = [1, 3, 8, 17, 32]
a = ts.find_pulses(widths=widths, max_width=32)
b = ts.find_pulses(method="matched_filter", widths=widths, max_width=32)

# Geometric bank: 1, 2, 4, ..., 256.
c = ts.find_pulses(width_style="geometric", spacing_factor=2, max_width=256)

# Median subtraction provides another baseline experiment.
d = ts.find_pulses(block_size=16000, detrend="median", chunk_size=8192)
```

## CPU Heimdall behaviour

`method="heimdall"` is a CPU sketch of Heimdall’s single-DM stages for trying
ideas on an already dedispersed series. Heimdall parity is not guaranteed.

```python
from sigpyproc.algorithms.heimdall import HeimdallConfig

native = ts.find_pulses(method="heimdall", max_width=4096)
renormalized = ts.find_pulses(
    method="heimdall",
    max_width=4096,
    heimdall_config=HeimdallConfig(
        gulp_size=262144,
        baseline_length=2.0,       # seconds
        min_tscrunch_width=4096,
        cand_sep_time=3,
        cand_sep_filter=3,
        boxcar_renorm=True,
    ),
)
native.to_pulse_csv("heimdall.csv", ts.header.tsamp)
```

## Command line

Search lives under one Click group (`spp_search`); there is no umbrella `spp`
command:

```bash
spp_search fft observation.tim --nfolds 4 --pmax 10 -o fft.csv
spp_search pulses observation.tim --snr 6 --max-width 256 -o pulses.csv
spp_search pulses observation.tim --engine fft --width 3 --width 8 -o fft_pulses.csv
spp_search pulses observation.tim --method matched_filter -o original_pulses.csv
spp_search pulses observation.dat --snr 5 --max-width 150 -o dat_pulses.csv
spp_search cluster fft.csv --snr 8 -o best.csv
```

Without `-o`, outputs default to `<stem>_fft.csv` (and similar). `-o -` prints
CSV to stdout. Pulse CSVs only carry event columns (`time_s`, `width_bins`, …),
not period trials. See {doc}`/cmd` for every flag.

## A complete walk-through

The cells below build a noisy periodic pulse train in memory, run an FFT search,
cluster harmonic families, and plot the amplitude spectrum. No test-data files
are required.

```{code-cell} ipython3
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries

nsamp, tsamp, period = 2**14, 1e-3, 0.05
rng = np.random.default_rng(1)
data = rng.normal(0.0, 1.0, nsamp).astype(np.float32)
phase = np.mod(np.arange(nsamp) * tsamp, period) / period
data[phase < 0.04] += 5.0

ts = TimeSeries(
    data,
    Header(
        filename="fake.tim",
        rawdatafile="fake.tim",
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=tsamp,
        tstart=50000.0,
        nsamples=nsamp,
    ),
)

cands = ts.search(nfolds=2, pmax=1.0, snr_min=6.0)
best = cands.cluster(snr_min=8.0)
print(f"Top clustered period {best.values('period_s')[0]:.4f} s, SNR {best.values('snr')[0]:.1f}")
print(f"{len(cands)} raw FFT hits before clustering")
```

```{code-cell} ipython3
spectrum = ts.rfft().form_spec(interpolate=False)
frequencies = np.arange(spectrum.data.size) / spectrum.header.tobs
fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
ax.plot(frequencies[1:], spectrum.data[1:], linewidth=0.8)
ax.axvline(1 / period, color="tab:orange", linestyle="--", label="Injected fundamental")
ax.set(xlim=(0, 150), xlabel="Frequency (Hz)", ylabel="Spectral amplitude",
       title="Amplitude spectrum of the synthetic series")
ax.legend()
plt.show()
plt.close(fig)
```

```{code-cell} ipython3
pulses = ts.find_pulses(snr_min=5.0, max_width=64)
times = np.arange(ts.nsamples) * tsamp
fig, ax = plt.subplots(figsize=(8, 2.8), layout="constrained")
ax.plot(times[:4000], data[:4000], color="0.35", linewidth=0.6, label="Data (first 4 s)")
for centre, width in zip(
    pulses.values("bin")[:8] * tsamp,
    pulses.values("width_s")[:8],
    strict=True,
):
    ax.axvspan(centre - width / 2, centre + width / 2, color="tab:red", alpha=0.15)
ax.set(xlabel="Time (s)", ylabel="Intensity", title="Single-pulse detections (first eight)")
plt.show()
plt.close(fig)
print(f"{len(pulses)} pulses above threshold")
```

`cluster` keeps the loudest member of a harmonic family. For this rectangular
pulse that row is often a harmonic, not $0.05\,\mathrm{s}$. Browse `cands` for
the injected period. A match within about one Fourier bin is
$P^{2} / T_{\mathrm{obs}}$. Treat the first row of `best` as a hint, not the
fundamental by definition.

To start from a filterbank instead of a time series, use
{py:func}`~sigpyproc.simulation.pulsar.fake_filterbank` or `spp_fake pulsar`,
then {py:meth}`~sigpyproc.base.Filterbank.dedisperse`. Mock FRBs and injection
are covered in {doc}`/tutorials/fake`.

---

## What this is not

Search here is for exploratory work on a single time series. It is not a
drop-in survey pipeline and it is not bit-identical to SIGPROC.

```{code-cell} ipython3
:tags: [remove-cell]

plt.close("all")
```
