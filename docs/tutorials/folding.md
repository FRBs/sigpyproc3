---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Fold, save and plot

Here we fold a short synthetic pulsar (period **0.128 s**, DM **10 pc cm⁻³**),
look at the cube, nudge period and DM a little, and save the result to an NPZ
archive. Those numbers are what we injected — not fitted from the data. The
notebook needs only an installed package and a writable temp directory.

## Generate and fold an observation

```{code-cell} ipython3
%matplotlib inline

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from sigpyproc import FilReader, FoldedData
from sigpyproc.simulation.pulsar import make_header, fake_filterbank, PulsarParams

workspace = TemporaryDirectory()
out = Path(workspace.name)
header = make_header(filename=str(out / "demo.fil"), nchans=32, nbits=32,
                     tsamp=0.001, tobs=4.096, fch1=1400.0, foff=-4.0)
fake_filterbank(
    header, PulsarParams(period=0.128, width=0.008, dm=10.0, snr=12.0), seed=5,
)

with FilReader(header.filename) as observation:
    cube = observation.fold(period=0.128, dm=10.0, nbins=32, nints=8,
                            nbands=8, quiet=True)

print(f"Cube shape: {cube.data.shape} (subintegrations, subbands, phase bins)")
print(f"Phase-bin duration: {cube.period / cube.nbins:g} s")
```

## See the folded signal

The three panels are the integrated profile, time versus phase, and subband
versus phase. They should line up with the period and DM we put in at the start.

```{code-cell} ipython3
:tags: [preview-fold]

centered = cube.centre()
phase = (np.arange(centered.nbins) + 0.5) / centered.nbins
profile = centered.get_profile()
time_phase = centered.get_time_phase().data
freq_phase = centered.get_freq_phase().data

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
axes[0].plot(phase, profile.data, drawstyle="steps-mid", color="tab:blue")
axes[0].set(xlabel="Pulse phase (cycles)", ylabel="Summed intensity",
            title="Integrated profile", xlim=(0, 1))
axes[1].imshow(time_phase, aspect="auto", origin="lower", cmap="magma",
               extent=(0, 1, -0.5, time_phase.shape[0] - 0.5))
axes[1].set(xlabel="Pulse phase (cycles)", ylabel="Subintegration index",
            title="Time / phase")
axes[2].imshow(freq_phase, aspect="auto", origin="lower", cmap="magma",
               extent=(0, 1, -0.5, freq_phase.shape[0] - 0.5))
axes[2].set(xlabel="Pulse phase (cycles)", ylabel="Subband index",
            title="Frequency / phase")
plt.show()
plt.close(fig)
```

Row labels in the heatmaps are subintegration and subband indices. Times and
frequencies are still on `cube.header` if you need physical axes for a given
geometry.

## Adjust and save a fold

`update_dm` and `update_period` take the value you want, not a delta — calling
them again with the same number does not stack corrections. We centre the cube,
write an NPZ, and load it back to confirm nothing was lost.

```{code-cell} ipython3
cube.update_dm(12.0)
cube.update_period(0.12801)
once = cube.data.copy()
cube.update_period(0.12801)
np.testing.assert_array_equal(cube.data, once)
centered = cube.centre()
archive = centered.to_npz(str(out / "fold"))
restored = FoldedData.from_npz(archive)
np.testing.assert_array_equal(restored.data, centered.data)
print(f"Reloaded shape: {restored.data.shape}")
print(f"Profile sample interval: {restored.get_profile().tsamp:g} s")
```

## Coordinates and correction limits

Cube axes are `(subintegration, subband, phase_bin)`. A profile's sampling
interval is **period / number of bins**, not the original file sample time.
DM and period corrections are circular integer-bin shifts measured from the
original fold. They are useful for small exploratory adjustments; refold the
original samples when a more precise phase model is needed. Reversing a
correction restores its shifts, and `centre()` keeps the correction basis
and acceleration. Invalid periods and nonfinite DMs are rejected before
changing the cube.

Subintegrations are laid out across the **whole** file unless you fold a slice.
If you only care about part of the observation, extract that piece to its own
`.fil` first so the subint boundaries mean what you expect.

## Archives and visualization

Fold archives are versioned NPZ files (no pickle): data, fold geometry,
acceleration, and the DM/period correction state. Sample counts stay 64-bit.
They do not duplicate every optional SIGPROC header field — keep the original
`.fil` if you need coordinates or custom keys.

The waterfall exporter and `batch=True` spectrum/series viewers use Agg and
write images without `show()`. The CLI equivalents are:

```bash
spp_view demo.fil --batch --stop 1024 -g 1024 -o plots
spp_view observation.tim --batch -o plots
spp_view observation.spec --batch -o plots
```

Complex and amplitude `.spec` files store their type and real transform length.
Use `FourierSeries.from_spec` for complex products and `PowerSpectrum.from_spec`
for amplitude products; `spp_view` dispatches automatically. Legacy complex
products infer an even real length. Historical SIGPROC seek spectra use a
different format.

```{code-cell} ipython3
:tags: [remove-cell]

workspace.cleanup()
```
