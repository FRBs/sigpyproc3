---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# View observations and export figures

`spp_view` opens waterfalls, dedispersed time series, and spectra in Matplotlib.
You can change DM and scaling from the keyboard while you look at the data.
We use a fake pulsar here (period **0.256 s**, DM **120 pc cm⁻³**) — numbers we
chose up front, not measured from the plot.

Figures in this page are snapshots from the same code paths the CLI uses. On your
machine, run `spp_view observation.fil` for the live controls. On a headless
machine or in batch processing, use `--batch` or the Python viewers with
`batch=True` to write PNGs without opening a window.

## Generate an observation

Data lives in a temp directory with a fixed seed so the docs build gets the
same noise every time.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from IPython.display import Image, display

from sigpyproc import FilReader
from sigpyproc.core.logging import set_progress_disabled
from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank, make_header
from sigpyproc.viz.batch import render_waterfall_batch
from sigpyproc.viz.series import SeriesViewer, SpectrumViewer
from sigpyproc.viz.waterfall import WaterfallViewer

# Keep transient progress widgets out of the static figures.
set_progress_disabled(disabled=True)

workspace = TemporaryDirectory()
out = Path(workspace.name)
header = make_header(
    filename=str(out / "pulsar.fil"), nchans=32, nbits=32,
    tsamp=0.001, tobs=4.096, fch1=1400.0, foff=-8.0,
)
_ = fake_filterbank(
    header, PulsarParams(period=0.256, width=0.012, dm=120.0, snr=25.0), seed=7,
)
```

## The waterfall viewer

Layout matches `spp_view`: summed time series on top, frequency–time in the
middle, bandpass on the right. At DM = 0 the pulse still sweeps across the
band because dispersion has not been removed.

For this static preview, `batch=True` selects a headless backend and
`gui=True` constructs the full viewer figure, including its controls.
Displaying that figure captures the same layout used by `spp_view`; the
controls in the documentation image are not interactive.

```{code-cell} ipython3
:tags: [preview-waterfall]

viewer = WaterfallViewer(header.filename, gulp=1024, dm=0.0, batch=True, gui=True)
try:
    viewer.fig.savefig(out / "waterfall-viewer.png", dpi=130)
    display(Image(filename=str(out / "waterfall-viewer.png")))
finally:
    viewer.close()
```

Locally, open the same interface with:

```bash
spp_view pulsar.fil --gulp 1024 --dm 0
```

Press `?` in the live viewer for shortcuts. Flag details are in {doc}`../cmd`.

## Dedisperse and export a waterfall

At the injected DM, the pulses align across frequency. Batch exports show
the data and projections without the interactive control panel. This is useful
for reports, remote sessions and processing several windows of an observation.

```{code-cell} ipython3
with FilReader(header.filename) as observation:
    images = render_waterfall_batch(
        observation, out / "plots", start=0, stop=1024, gulp=1024,
        dm=120.0, jobs=1,
    )
    series = observation.dedisperse(dm=120.0, quiet=True)

display(Image(filename=str(images[0])))
```

The equivalent CLI export is:

```bash
spp_view pulsar.fil --batch --stop 1024 --gulp 1024 --dm 120 -o plots
```

The selected window must leave enough samples for the dispersion delays;
exports near the end of an observation may contain fewer usable samples.

## Inspect a time series

The overview shows the whole dedispersed series. Selecting a range in a local
viewer changes the detail panel. Here we display the first second, with a
pulse train separated by the injected 0.256 s period.

```{code-cell} ipython3
viewer = SeriesViewer(series, batch=True, gui=True)
try:
    viewer.end_samp = 1024
    viewer.update_detail()
    viewer.fig.savefig(out / "series-viewer.png", dpi=130)
    display(Image(filename=str(out / "series-viewer.png")))
finally:
    viewer.close()
```

Write a time series and open it locally with `spp_view pulsar.tim`:

```{code-cell} ipython3
series.to_tim(str(out / "pulsar.tim"))
print(f"{series.nsamples} samples at {series.header.tsamp:g} s per sample")
```

## Inspect a spectrum

A narrow periodic pulse produces harmonics as well as a fundamental. The
spectrum viewer supports frequency inspection and harmonic markers in an
interactive session. This preview shows the first 100 Hz of the whitened
amplitude spectrum; see {doc}`search` for candidate interpretation.

```{code-cell} ipython3
spectrum = series.rfft().deredden().form_spec()
viewer = SpectrumViewer(spectrum, batch=True, gui=True)
try:
    viewer.ax.set_xlim(0, 100)
    viewer.fig.savefig(out / "spectrum-viewer.png", dpi=130)
    display(Image(filename=str(out / "spectrum-viewer.png")))
finally:
    viewer.close()
```

For a minimal PNG, `save_png` on a viewer with `batch=True` skips the control
widgets. The CLI also opens `.tim` and `.spec` files directly:

```{code-cell} ipython3
spectrum.to_spec(str(out / "pulsar.spec"))
viewer = SpectrumViewer(spectrum, batch=True)
try:
    path = viewer.save_png(str(out / "spectrum.png"))
    print(f"Exported {Path(path).name}")
finally:
    viewer.close()
```

```bash
spp_view pulsar.tim --batch -o plots
spp_view pulsar.spec --batch -o plots
```

For phase-aligned profiles and frequency/phase plots, continue with
{doc}`folding`. Generated files above live in a temporary directory and are
removed when this example finishes; use a directory of your choice to keep
exports in your own workflow.

```{code-cell} ipython3
:tags: [remove-cell]

plt.close("all")
workspace.cleanup()
```
