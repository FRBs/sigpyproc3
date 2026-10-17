---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---


# Filterbank quickstart

If you are new to the package objects (`FilReader`, `Header`, `TimeSeries`, …),
work through {doc}`basics` first — it covers pretty-printing metadata, attribute
access, folding, and method chaining without the plotting focus below.

This page builds a tiny synthetic pulsar observation, reads it back, dedisperses
it, and runs a quick periodicity search. Everything runs in a temporary
directory.

```{code-cell} ipython3
%matplotlib inline
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib.pyplot as plt
from sigpyproc import FilReader
from sigpyproc.simulation.pulsar import make_header, fake_filterbank, PulsarParams

workspace = TemporaryDirectory()
path = Path(workspace.name) / "demo.fil"
header = make_header(filename=str(path), nchans=16, nbits=32, tsamp=0.001,
                     tobs=4.096, fch1=1400.0, foff=-8.0)
_ = fake_filterbank(header, PulsarParams(period=0.128, dm=30.0, snr=12.0), seed=4)
```

## Read a window and dedisperse

Open the file with a context manager so it closes cleanly. A block is
`(nchans, nsamples)`; times are in seconds, frequencies in MHz, DM in
pc cm⁻³.

```{code-cell} ipython3
with FilReader(path) as observation:
    block = observation.read_block(0, 512)
    series = observation.dedisperse(dm=30.0, quiet=True)

print(f"{block.nchans} channels, {block.nsamples} selected samples")
print(f"{series.nsamples} dedispersed samples at {series.header.tsamp:g} s")
```

## See the frequency and time axes

Before dedispersion the pulse sweeps across the band; lower frequencies
arrive later. Axis values come from the block header, not from raw indices.

```{code-cell} ipython3
times = np.arange(block.nsamples) * block.header.tsamp
frequencies = block.header.chan_freqs
fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
image = ax.pcolormesh(times, frequencies, block.data, shading="auto", cmap="magma")
ax.set(xlabel="Time from window start (s)", ylabel="Frequency (MHz)",
       title="Simulated pulsar before dedispersion")
fig.colorbar(image, ax=ax, label="Intensity (arbitrary units)")
plt.show()
plt.close(fig)
```

## Work with arrays explicitly

`TimeSeries`, `FourierSeries`, and `FilterbankBlock` wrap a `.data` array and
a `.header` — they are not NumPy ndarray subclasses. Pass `.data` into NumPy; methods
return new objects with updated arrays and matching metadata.

```{code-cell} ipython3
print(f"Mean: {series.data.mean():.3f}; standard deviation: {series.data.std():.3f}")
fig, ax = plt.subplots(figsize=(8, 3))
times = np.arange(series.nsamples) * series.header.tsamp
ax.plot(times[:512], series.data[:512])
ax.set(xlabel="Time (s)", ylabel="Summed intensity")
ax.set_title("Dedispersed frequency sum at DM = 30 pc cm⁻³")
fig.tight_layout()
plt.show()
plt.close(fig)
```

## Form a spectrum

Next we take the real FFT (`rfft()`), which keeps the transform length the
series needs (including any zero-padding to a sensible FFT size). Running
`deredden()` on that complex series knocks down red noise; `form_spec()` turns
it into an amplitude spectrum. With the default local noise estimate set to
zero, those amplitudes are left as-is rather than rescaled. The search below
also folds harmonics, so a bright harmonic or subharmonic can outrank the true
fundamental; treat the top row as a lead, not a period measurement.

```{code-cell} ipython3
fourier = series.rfft()
spectrum = fourier.deredden().form_spec(interpolate=True)
print(f"Frequency spacing: {1 / spectrum.header.tobs:.4f} Hz")
candidates = series.search(nfolds=2, pmax=1.0, snr_min=6.0)
print(f"{len(candidates)} exploratory FFT candidates")
```

```{code-cell} ipython3
frequencies = np.arange(spectrum.data.size) / spectrum.header.tobs
fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
ax.plot(frequencies[1:], spectrum.data[1:], linewidth=0.8)
ax.axvline(1 / 0.128, color="tab:orange", linestyle="--", label="Injected fundamental")
ax.set(xlim=(0, 100), xlabel="Frequency (Hz)", ylabel="Spectral amplitude",
       title="Whitened spectrum and pulse harmonics")
ax.legend()
plt.show()
plt.close(fig)
```

## Continue exploring

- {doc}`folding` — phase bins, tweak DM/period on a fold, save archives  
- {doc}`search` — candidate tables and clustering  
- {doc}`io` — PSRFITS to `.fil`  
- {doc}`viewers` — `spp_view` and batch PNGs  
- {doc}`../capabilities` — what is supported and where it stops  

```{code-cell} ipython3
plt.close("all")
workspace.cleanup()
```
