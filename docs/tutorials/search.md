(search-tutorial)=

# Searching time series

```{note}
Search is a **library-first** API. The `spp_search` CLI is a thin wrapper
around {py:class}`~sigpyproc.timeseries.TimeSeries` and
{py:class}`~sigpyproc.core.candidates.CandidateTable`. This is a modern
extension of SIGPROC, not a clone of `seek` / `best` / `ffa` / `peak`.
```

sigpyproc can now look for **periodic signals** (FFT and Fast Folding
Analysis) and **single pulses** in a time series, and keep the hits in
one candidate table. Input is a SIGPROC `.tim` file or an in-memory
{py:class}`~sigpyproc.timeseries.TimeSeries` — typically the output of
{py:meth}`~sigpyproc.base.Filterbank.dedisperse`.

This page is the “how to use it” companion to {doc}`/tutorials/quickstart`.

---

## What maps to what

| SIGPROC | sigpyproc | Notes |
| --- | --- | --- |
| `seek` | {py:meth}`~sigpyproc.timeseries.TimeSeries.search` / `spp_search fft` | Whitened FFT + harmonic folds. No `.prd`, no birdie files. |
| `find` | same as `spp_search fft` | Historical `find` was renamed `seek`; we do not ship `spp_find`. |
| `best` | {py:meth}`~sigpyproc.core.candidates.CandidateTable.cluster` / `spp_search cluster` | Harmonic families only. No PGPLOT, no `.bst` / `.fld`. |
| `peak` | {py:meth}`~sigpyproc.timeseries.TimeSeries.find_pulses` / `spp_search pulses` | Selectable PRESTO-inspired and original matched-filter boxcar searches. |
| `ffa` | {py:meth}`~sigpyproc.timeseries.TimeSeries.ffa` / `spp_search ffa` | Staelin / Mueller tree. Periods in **seconds** on the API. |
| `giant` | — | Interactive PGPLOT viewer; out of scope. |

The public interchange format is **CSV**, not `.prd`.

---

## Load a time series

From a SIGPROC `.tim` (the GBT example ships with the test suite):

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

---

## FFT search

{py:meth}`~sigpyproc.timeseries.TimeSeries.search` is the `seek`-shaped
pipeline, on purpose, in a fixed order:

1. Optional acceleration resample
2. Real FFT
3. {py:meth}`~sigpyproc.fourierseries.FourierSeries.deredden` (running
   median on the **complex** series — not
   {py:meth}`~sigpyproc.timeseries.TimeSeries.deredden`)
4. Amplitude spectrum, with inter-bin interpolation by default
5. Harmonic folds \(2, 4, \ldots, 2^{N}\) **in addition to** the
   unfolded spectrum
6. Local-maxima peak pick → one {py:class}`~sigpyproc.core.candidates.CandidateTable`

```python
hits = ts.search(nfolds=4, pmax=10.0, snr_min=6.0)
print(len(hits), hits)
print(hits.values("period_s")[:5])
print(hits.values("snr")[:5])
print(hits.table["nharm"][:5])
```

`nfolds` is the number of **folds**, not “use harmonics 1…n”.
`nfolds=4` searches the unfolded spectrum plus 2, 4, 8 and 16 harmonics.
`method` chooses how those harmonics are added:

- `lyne` (default): SIGPROC Lyne-Ashworth stretch, with a half-bin neighbour add.
- `simple`: integer bins `h*n` only. No drift. Fast, and the least sensitive off-bin.
- `presto`: stock PRESTO sum, even harmonics, nearest bin.
- `greedy`: follow the louder neighbour as the harmonic order increases.

Folded spectra stay on the original Fourier grid. A narrow pulse can still
peak on a harmonic of the true period. {py:meth}`~sigpyproc.timeseries.TimeSeries.ffa`
and `find_pulses` remain the tools for that case.

### Cluster harmonic families

Several folds of the same pulsar show up as \(P\), \(P/2\), \(2P\), …
{py:meth}`~sigpyproc.core.candidates.CandidateTable.cluster` keeps the
loudest member of each \(k/n\) family (default \(k, n \le 16\)):

```python
best = hits.cluster(snr_min=8.0, period_tol=1e-4)
best.to_csv("fft_clustered.csv")
```

Single-pulse rows (`kind="pulse"`) are never treated as harmonics of
each other. Slice first if you want to cluster only FFT hits:

```python
fft_only = hits[hits.values("kind") == "fft"]
```

---

## Single pulses

{py:meth}`~sigpyproc.timeseries.TimeSeries.find_pulses` defaults to a
PRESTO-inspired search. It removes a baseline and estimates noise in blocks,
searches full-resolution boxcars, then suppresses nearby weaker detections
within and across widths. This is a modern adaptation of
[PRESTO single_pulse_search.py](https://github.com/scottransom/presto/blob/master/bin/single_pulse_search.py),
with different normalization and deterministic greedy pruning.

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
noise estimate, and $w$ is the width in samples. The `running_sum` engine
(default) uses Numba rolling sums with float64 accumulation. The `fft` engine
uses zero-padded linear convolution through rocket-fft inside Numba. Numerical
rounding can change results very close to an SNR threshold or a tie.

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

---

## CPU Heimdall behavior

`method="heimdall"` models the single-DM stages of the supplied Heimdall CUDA
pipeline on a CPU. It is intended for experimentation on an already dedispersed
series; GPU parity is unverified.

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

---

## Fast Folding Analysis

{py:meth}`~sigpyproc.timeseries.TimeSeries.ffa` is the long-period
companion to the FFT search. Periods on the **Python API are in
seconds**. The CLI accepts SIGPROC-style milliseconds and converts
immediately.

```python
ffa_hits = ts.ffa(pmin=1.0, pmax=1.5, snr_min=8.0, npeaks=20)
print(ffa_hits.values("period_s")[:5])
```

Useful flags:

- `nrebin` — mean-downsample first (wider effective pulse)
- `search_width=True` — loop rebin in `{1, 2, 4, 8, 16}` (exclusive
  with `nrebin != 1`, as in `ffa.f -w` / `-rb`)
- `zero_pad=True` — pad to the next power of two after the \(2^{24}\)
  sample cap

One candidate is kept **per integer trial period**. The fractional FFA
tree offset is used only when it clearly outranks the unshifted fold —
for a wide pulse every shift looks equally loud, and the integer period
is the honest answer.

---

## Command line

Each `spp_*` tool is a standalone script — there is no umbrella `spp`
command. Search is one Click group:

```bash
spp_search fft observation.tim --nfolds 4 --pmax 10 -o fft.csv
spp_search pulses observation.tim --snr 6 --max-width 256 -o pulses.csv
spp_search pulses observation.tim --engine fft --width 3 --width 8 -o fft_pulses.csv
spp_search pulses observation.tim --method matched_filter -o original_pulses.csv
spp_search pulses observation.dat --snr 5 --max-width 150 -o dat_pulses.csv
spp_search ffa observation.tim --pmin-ms 1000 --pmax-ms 20000 -o ffa.csv
spp_search cluster fft.csv --snr 8 -o best.csv
```

Defaults write `<stem>_{fft,pulses,ffa,cluster}.csv`. Use `-o -` to
print CSV on stdout. Pulse CSVs contain only event fields, including event
`time_s` and `width_bins`, rather than periodic-candidate columns. Full flag
lists live in {doc}`/cmd`.

---

## A complete walk-through

Inject a rectangular pulsar, FFT-search it, and cluster:

```python
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
print(best.values("period_s")[0], best.values("snr")[0])
```

The recovered period should sit within about one Fourier bin of
\(0.05\,\mathrm{s}\) (\(\Delta P \approx P^{2} / T_{\mathrm{obs}}\)).

To invent a filterbank instead of a time series, use
{py:func}`~sigpyproc.simulation.pulsar.fake_filterbank` /
`spp_fake pulsar`, then {py:meth}`~sigpyproc.base.Filterbank.dedisperse`.
`spp_fake furby` writes a mock FRB whose ``--snr`` is the matched-filter
signal-to-noise ratio of the dedispersed sum, including 1- to 16-bit files.
`spp_fake inject` stamps the same pulses into a filterbank that `spp_fake`
already wrote. The walk-through is {doc}`/tutorials/fake`.

---

## What this is not

Search here is for **exploratory** work on a single time series. It is
not a drop-in survey pipeline and it is not bit-identical to SIGPROC.
