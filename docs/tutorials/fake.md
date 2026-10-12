(fake-tutorial)=

# Simulating filterbank pulses

```{note}
Simulation is a **library-first** API. `spp_fake` is a thin wrapper
around {py:func}`~sigpyproc.simulation.pulsar.fake_filterbank`,
{py:class}`~sigpyproc.simulation.furby.FurbyGenerator`, and
{py:func}`~sigpyproc.simulation.inject.inject_file`.
```

`spp_fake` writes a SIGPROC filterbank of Gaussian noise, with an
optional pulsar, a rectangular single pulse, or one or more mock FRBs.
The FRB path is a [furby](https://arxiv.org/abs/2011.10191): a
noise-free pulse stored dedispersed at the file sampling time.
Dispersion is a per-channel delay applied when the pulse is written,
so the generator does not allocate the dispersed sweep.

This page is the companion to the one-line pointer in
{doc}`/tutorials/search`.

---

## Which command

| Command | What it writes | What `--snr` means |
| --- | --- | --- |
| `spp_fake noise` | Unit Gaussian noise | — |
| `spp_fake pulsar` | Periodic rectangular pulse | Peak SNR in one channel, then divided by \(\sqrt{N_{\mathrm{chan}}}\) |
| `spp_fake pulse` | One rectangular pulse | Same peak SNR as `pulsar` |
| `spp_fake furby` | Gaussian or boxcar FRB, with optional smearing, scattering, and a spectrum | Matched-filter SNR of the frequency sum |
| `spp_fake inject` | Those FRBs added to an existing sigproc file | Same matched-filter SNR |

`--nbits 32` stores float32. Narrower files use the fixed dspsr
digitiser (mean and scale for unit-variance input, no running
rescale), so the injected amplitude survives 1-, 2-, 4-, 8-, and
16-bit packing.

---

## One mock FRB

Arrival time is seconds from the start of the file, at the reference
frequency (`--ref-freq`, default `fch1`). Width is the intrinsic
Gaussian FWHM in seconds.

```bash
spp_fake furby \
  -t 0.05 \
  -d 50 \
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

The pulse is centred near sample 50. Channel 0 (1400 MHz) arrives
then. Lower channels arrive later. Scattering is off unless you pass
`--tau` in seconds, defined by default at 1 GHz (1000 MHz, configurable via
`--tau-ref-freq`):

```bash
spp_fake furby \
  -t 0.05 -d 50 -w 0.004 -s 20 \
  --tau 0.001 --tau-ref-freq 1000.0 \
  --spec-kind power_law --spec-index -1.5 \
  --shape gaussian \
  -o scattered.fil \
  --nchans 64 --nbits 8 --tsamp 0.001 --tobs 0.3 \
  --fch1 1400 --foff -1 --seed 1
```

`--spec-kind flat` ignores `--spec-index`. The other shapes
(`power_law`, `smooth_envelope`, `gaussian`, `polynomial_peaks`,
`scintillation`, `gaussian_blobs`, `random`) use the index when it is
not zero. `--shape boxcar` replaces the Gaussian. Intra-channel dispersion
smearing is centered around zero delay so channel arrival centroids
faithfully match nominal cold-plasma dispersion delay without artificial
delay curvature. `--nosmear` skips smearing entirely.

Repeat `-t` for several arrivals that share `--dm`, `--width`, and
`--snr`. The header DM is that shared value, and the source name
becomes `DM 50` unless you pass `--source`.

---

## A catalogue

A text file can give each pulse its own parameters. Blank lines and `#`
comments are ignored. Files can be whitespace- or comma-delimited. Four
columns are required (`time dm width snr`), with optional 5th and 6th
columns for `spec_idx` and `tau0` (in seconds at 1 GHz):

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

---

## Generating random FRB populations

You can generate realistic random burst populations using the CLI or the
Python API:

```bash
spp_fake population -n 20 -o catalogue.txt \
  --tobs 10.0 --dm-min 50 --dm-max 1000 \
  --snr-min 10 --snr-max 100 --snr-law euclidean \
  --tau-min 0.0 --tau-max 0.005 --seed 42
```

In Python, use {py:class}`~sigpyproc.simulation.population.BurstDistribution`
and {py:class}`~sigpyproc.simulation.population.RandomBurstPopulation`:

```python
from sigpyproc.simulation.population import BurstDistribution, RandomBurstPopulation

dist = BurstDistribution(
    dm_range=(50.0, 800.0),
    snr_range=(10.0, 50.0),
    snr_law="euclidean",
    tau_range=(0.0, 0.002),
    tau_ref_freq=1000.0,
)
pop = RandomBurstPopulation(dist, seed=123)
pop.write_catalogue("population.txt", count=50, tobs=60.0)
```

---

## The same pulse from Python

{py:meth}`~sigpyproc.simulation.furby.FurbyGenerator.generate` returns
a {py:class}`~sigpyproc.simulation.furby.Furby`. `profile` and `block`
are the same array, shape `(nchans, nsamples)`, at the header sampling
time. Channels are aligned at the intrinsic epoch. `delays` says how
many native samples later each channel arrives.

```python
from sigpyproc.simulation.furby import FurbyGenerator, PulseParams
from sigpyproc.simulation.pulsar import make_header

header = make_header(
    filename="burst.fil",
    nchans=64,
    nbits=32,
    tsamp=1e-3,
    tobs=0.3,
    fch1=1400.0,
    foff=-1.0,
    dm=50.0,
)
params = PulseParams(
    dm=50.0,
    snr=20.0,
    width=4e-3,
    tau0=1e-3,          # seconds at 1 GHz (tau_ref_freq); 0 leaves scattering off
    tau_ref_freq=1000.0,
    spec_kind="power_law",
    spec_idx=-1.5,
)
pulse = FurbyGenerator(header, params, seed=1).generate()
pulse.profile.data.shape
pulse.stats_hdr.box_snr       # matches params.snr, to about 0.1%
pulse.save("burst.h5")        # compact HDF5; Furby.from_file reads it back
```

`pulse.plot()` draws the dedispersed profile. `pulse.realize()` builds
the dispersed waterfall at native resolution. That array is the thing
the compact form avoids: on a wide band at high DM it can be hundreds
of megabytes. Call it when you want the dynamic spectrum, and skip it
when you only want to write a filterbank.

To write the noisy file yourself, build an
{py:class}`~sigpyproc.simulation.inject.Injector` and pass it to
{py:func}`~sigpyproc.simulation.pulsar.fake_filterbank`:

```python
import numpy as np
from sigpyproc.simulation.inject import Injector
from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank

arrival = int(np.rint(0.05 / header.tsamp))
injector = Injector.from_furbies([pulse], [arrival])
fake_filterbank(header, PulsarParams(), "burst.fil", seed=1, injector=injector)
```

`box_snr` is the SNR of the noise-free sum. The file adds unit
Gaussian noise per channel, so a short excerpt will not measure 20
exactly.

---

## Inject into an existing file

`spp_fake inject` reads a sigproc filterbank, adds pulses, and writes a new
file in one streaming pass using blocks (`--gulp`).

For synthetic filterbanks generated with `spp_fake`, integer samples are
inverted to sigma units using the fixed digitiser:

```bash
spp_fake noise -o noise.fil \
  --nchans 64 --nbits 8 --tsamp 0.001 --tobs 0.3 \
  --fch1 1400 --foff -1 --seed 1

spp_fake inject noise.fil -o injected.fil \
  -t 0.05 -d 50 -w 0.004 -s 20 --seed 1
```

For real telescope observations, pass `--preserve-bandpass`:

```bash
spp_fake inject telescope_data.fil -o injected.fil \
  -t 0.05 -d 50 -w 0.004 -s 20 \
  --preserve-bandpass --seed 1
```

When `--preserve-bandpass` is enabled, per-channel empirical median and
standard deviation are measured from the input filterbank so the pulse can be
added without disrupting the telescope bandpass.

From Python the same step is
{py:func}`~sigpyproc.simulation.inject.inject_file` (with
`preserve_bandpass=True`). A catalogue is
{py:func}`~sigpyproc.simulation.inject.read_pulse_catalogue` and
{py:func}`~sigpyproc.simulation.inject.injector_from_rows`.

---

## Read it back

Dedisperse at the DM you injected. The peak of the sum sits near the
arrival time at `fch1`.

```python
from sigpyproc.readers import FilReader

fil = FilReader("burst.fil")
ts = fil.dedisperse(50.0)
peak = int(ts.data.argmax())
print(peak, peak * fil.header.tsamp)
```

For an 8-bit file the samples are digitiser levels (mean 127.5), not
sigma. The pulse is still the bright outlier after dedispersion.

---

## What this does not do

- No fluence or SEFD calibration. `--snr` is defined against noise of
  rms 1 per channel.
- No measured SNR on the noisy file, and no threshold that drops a
  weak pulse.
- The spectral shapes are a gain per channel. They are not a
  time-frequency dynamic scintillation screen.
- `spp_fake pulse` remains the rectangular SIGPROC-style pulse. It
  does not grow a scattering tail or a spectrum.
