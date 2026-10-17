---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(fake-tutorial)=

# Simulating filterbank pulses

Need a filterbank with a known pulsar, a single pulse, or a mock FRB? `spp_fake`
writes SIGPROC `.fil` files filled with unit Gaussian noise and optional
signals. The same generators are available in Python if you prefer to script
injection or build test catalogues.

Mock FRBs use the [furby](https://arxiv.org/abs/2011.10191) model: the pulse is
stored dedispersed at the file sampling rate, and per-channel delays are applied
when it is written. That keeps memory use modest on wide bands and high DMs.
For a worked search example on synthetic data, see {doc}`search`.

## What each command does

| Command | Output | Meaning of `--snr` |
| --- | --- | --- |
| `spp_fake noise` | Unit Gaussian noise | — |
| `spp_fake pulsar` | Periodic rectangular pulse | In-channel peak SNR (scaled; see note below) |
| `spp_fake pulse` | One rectangular pulse | Same as `pulsar` |
| `spp_fake furby` | Gaussian or boxcar FRB (smearing, scattering, spectrum) | Matched-filter SNR of the frequency sum |
| `spp_fake inject` | FRBs added to an existing `.fil` | Same matched-filter SNR |

For `pulsar` and `pulse`, `--snr` is the peak SNR in one channel divided by
$\sqrt{N_{\mathrm{chan}}}$.

`--nbits 32` stores float32. For 1–16 bit files the fixed dspsr digitiser
(mean and scale for unit-variance noise, no running rescale) keeps the injected
amplitude meaningful after packing.

## Build a furby and look at it

The example below uses the same timing as the CLI recipe further down: arrival
0.05 s at 1400 MHz, DM 200 pc cm⁻³, 4 ms FWHM, matched-filter SNR 20. Figures
are generated when the docs build; locally you can step through the cells in
any MyST notebook viewer.

```{code-cell} ipython3
%matplotlib inline

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from sigpyproc.simulation.furby import FurbyGenerator, PulseParams
from sigpyproc.simulation.pulsar import make_header

workspace = TemporaryDirectory()
out = Path(workspace.name)

header = make_header(
    filename=str(out / "burst.fil"),
    nchans=64,
    nbits=32,
    tsamp=1e-3,
    tobs=0.3,
    fch1=1400.0,
    foff=-1.0,
    dm=200.0,
)
params = PulseParams(
    dm=200.0,
    snr=20.0,
    width=4e-3,
    tau0=1e-3,
    tau_ref_freq=1000.0,
    spec_kind="power_law",
    spec_idx=-1.5,
)
# Pad to the file length so plots show the burst in context, not only the
# few dozen samples the intrinsic pulse needs.
plot_nsamps = int(header.nsamples)
pulse = FurbyGenerator(header, params, nsamps_out=plot_nsamps, seed=1).generate()
duration = plot_nsamps * header.tsamp
print(f"Profile shape {pulse.profile.data.shape} ({duration:g} s window)")
print(f"Target SNR {params.snr:.1f}; generator reports {pulse.stats_hdr.box_snr:.2f}")
```

`profile` is the dedispersed `(nchans, nsamples)` array at native sampling.
`delays` lists how many samples later each channel arrives once dispersion is
applied. Injection uses that compact form directly; it does not materialise a
full bow-tie cube in memory.

```{code-cell} ipython3
fig = pulse.profile.plot(figsize=(10, 4), scaler="percentile")
fig.suptitle("Dedispersed furby (noise-free)")
for ax in fig.axes:
    if ax.images:
        ax.set_xlim(0, duration)
plt.show()
plt.close(fig)

# realize() is for inspection and for to_filterbank(): each row is shifted by
# the rounded DM delay. The array is mostly zeros outside the sweep; zscore
# scaling looks harsh, so use percentile or zscale.
dispersed = pulse.realize()
fig = dispersed.plot(figsize=(10, 4), scaler="percentile")
fig.suptitle("Dispersed view (per-channel delays applied)")
for ax in fig.axes:
    if ax.images:
        ax.set_xlim(0, dispersed.nsamples * dispersed.header.tsamp)
plt.show()
plt.close(fig)

aligned = dispersed.dedisperse(
    pulse.params_hdr.dm, ref_freq="fch1", only_valid_samples=True,
)
fig = aligned.plot(figsize=(10, 4), scaler="percentile")
fig.suptitle("realize() then dedisperse at injected DM")
for ax in fig.axes:
    if ax.images:
        ax.set_xlim(0, aligned.nsamples * aligned.header.tsamp)
plt.show()
plt.close(fig)
```

With `foff < 0`, channel index 0 is the high-frequency edge, so the burst
sweeps toward lower MHz at later times. The middle panel is what you would see
in a filterbank before dedispersion; the third panel checks that shifting the
channels back lines them up again.

## Write a noisy filterbank

To add unit Gaussian noise and pack the result into a `.fil` file, pass an
{py:class}`~sigpyproc.simulation.inject.Injector` to
{py:func}`~sigpyproc.simulation.pulsar.fake_filterbank`:

```{code-cell} ipython3
from sigpyproc import FilReader
from sigpyproc.simulation.inject import Injector
from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank

fil_path = out / "burst.fil"
arrival = int(np.rint(0.05 / header.tsamp))
injector = Injector.from_furbies([pulse], [arrival])
fake_filterbank(header, PulsarParams(), str(fil_path), seed=1, injector=injector)

with FilReader(fil_path) as observation:
    window = observation.read_block(0, observation.header.nsamples)
    series = observation.dedisperse(200.0, quiet=True)

peak = int(series.data.argmax())
print(f"Dedispersed peak at sample {peak} ({peak * series.header.tsamp:.3f} s)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
times = np.arange(window.nsamples) * window.header.tsamp
freqs = window.header.chan_freqs
vmin, vmax = np.percentile(window.data, [2, 98])
axes[0].pcolormesh(
    times, freqs, window.data, shading="auto", cmap="magma", vmin=vmin, vmax=vmax,
)
axes[0].set(
    title="Raw filterbank (+ noise)", xlabel="Time (s)", ylabel="MHz",
    xlim=(0, duration),
)
ts_times = np.arange(series.nsamples) * series.header.tsamp
axes[1].plot(ts_times, series.data, color="tab:blue", linewidth=0.9)
axes[1].axvline(0.05, color="tab:orange", linestyle="--", label="Injected arrival")
axes[1].set(
    title="Dedispersed sum at DM = 200", xlabel="Time (s)", ylabel="Intensity",
    xlim=(0, duration),
)
axes[1].legend()
plt.show()
plt.close(fig)
```

`box_snr` refers to the noise-free dedispersed sum. A short stretch of noisy
data will not measure exactly 20 unless you average over many realisations.

Compact HDF5 storage is available via `pulse.save("burst.h5")` and
{py:meth}`~sigpyproc.simulation.furby.Furby.from_file`.

## One mock FRB from the terminal

`-t` is arrival time in seconds at the reference frequency (`--ref-freq`,
default `fch1`). `-w` is Gaussian FWHM in seconds.

```bash
spp_fake furby \
  -t 0.05 \
  -d 200 \
  -w 0.004 \
  -s 20 \
  -o burst.fil \
  --nchans 64 \
  --nbits 32 \
  --tsamp 0.001 \
  --tobs 0.3 \
  --fch1 1400 \
  --foff -1 \
  --seed 1
```

Scattering uses `--tau` in seconds at 1 GHz by default (`--tau-ref-freq` changes
the reference). Spectrum shape comes from `--spec-kind` and `--spec-index`; use
`--shape boxcar` for a top-hat pulse. `--nosmear` skips intra-channel smearing.

```bash
spp_fake furby \
  -t 0.05 -d 200 -w 0.004 -s 20 \
  --tau 0.001 --tau-ref-freq 1000.0 \
  --spec-kind power_law --spec-index -1.5 \
  --shape gaussian \
  -o scattered.fil \
  --nchans 64 --nbits 8 --tsamp 0.001 --tobs 0.3 \
  --fch1 1400 --foff -1 --seed 1
```

Repeat `-t` for several bursts that share `--dm`, `--width`, and `--snr`. The
header DM matches that shared value and the source name becomes `DM 200` unless
you pass `--source`.

## Catalogues

List one pulse per line: `time dm width snr`, with optional `spec_idx` and
`tau0` (seconds at 1 GHz):

```text
# time_s  dm  width_s  snr  [spec_idx]  [tau0_s]
0.05   50  0.004  20   -1.5  0.001
0.20  120  0.008  15   -2.0  0.002
```

```bash
spp_fake furby --catalogue pulses.txt -o bursts.fil \
  --nchans 64 --nbits 32 --tsamp 0.001 --tobs 0.4 \
  --fch1 1400 --foff -1 --seed 1
```

## Random populations

Draw burst parameters from ranges and laws, then write a catalogue text file:

```bash
spp_fake population -n 20 -o catalogue.txt \
  --tobs 10.0 --dm-min 50 --dm-max 1000 \
  --snr-min 10 --snr-max 100 --snr-law euclidean \
  --tau-min 0.0 --tau-max 0.005 --seed 42
```

```{code-cell} ipython3
from sigpyproc.simulation.population import BurstDistribution, RandomBurstPopulation

dist = BurstDistribution(
    dm_range=(50.0, 800.0),
    snr_range=(10.0, 50.0),
    snr_law="euclidean",
    tau_range=(0.0, 0.002),
)
catalogue = out / "population.txt"
RandomBurstPopulation(dist, seed=123).write_catalogue(
    str(catalogue), count=5, tobs=2.0,
)
print(catalogue.read_text())
```

In a longer script, feed the catalogue to `spp_fake furby --catalogue` or to
{py:func}`~sigpyproc.simulation.inject.read_pulse_catalogue`.

## Inject into an existing file

`spp_fake inject` streams the input in blocks (`--gulp`) and adds pulses in one
pass. For files `spp_fake` created, integer samples are inverted with the fixed
digitiser before injection:

```bash
spp_fake noise -o noise.fil \
  --nchans 64 --nbits 8 --tsamp 0.001 --tobs 0.3 \
  --fch1 1400 --foff -1 --seed 1

spp_fake inject noise.fil -o injected.fil \
  -t 0.05 -d 200 -w 0.004 -s 20 --seed 1
```

On real telescope data, match the bandpass statistics:

```bash
spp_fake inject telescope_data.fil -o injected.fil \
  -t 0.05 -d 200 -w 0.004 -s 20 \
  --preserve-bandpass --seed 1
```

The Python equivalent is {py:func}`~sigpyproc.simulation.inject.inject_file`
(with `preserve_bandpass=True` for observations).

```{code-cell} ipython3
:tags: [remove-cell]

plt.close("all")
workspace.cleanup()
```
