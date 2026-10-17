from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigpyproc.fourierseries import FourierSeries, PowerSpectrum
from sigpyproc.simulation.pulsar import make_header
from sigpyproc.timeseries import TimeSeries
from sigpyproc.viz.series import SeriesViewer, SpectrumViewer


class DummyEvent:
    def __init__(
        self,
        key: str | None = None,
        inaxes: object = None,
        xdata: float | None = None,
        ydata: float | None = None,
    ) -> None:
        self.key = key
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata


def test_series_viewer(timfile: str, tmp_path: Path) -> None:
    tim = TimeSeries.from_tim(timfile)
    viewer = SeriesViewer(tim, batch=True, gui=True)
    assert viewer.total_samps == len(tim.data)
    assert viewer.start_samp == 0

    # Pan right
    viewer.pan_right()
    assert viewer.start_samp > 0

    # Pan left
    viewer.pan_left()
    assert viewer.start_samp == 0

    # Zoom in & out
    viewer.zoom_in()
    assert viewer.end_samp < 16384
    viewer.zoom_out()
    viewer.reset_view()
    assert viewer.start_samp == 0

    # Span selection
    viewer.on_span_select(0.01, 0.05)
    assert viewer.start_samp >= 0

    viewer.end_samp = viewer.total_samps
    viewer.update_detail()
    viewer.start_samp = 0
    viewer.end_samp = min(64, viewer.total_samps)
    viewer.update_detail()
    assert viewer.line_detail.axes is viewer.ax_detail
    assert viewer.line_detail.get_visible()

    out_png = tmp_path / "series.png"
    viewer.fig.savefig(out_png)
    assert out_png.exists()
    viewer.close()


def test_spectrum_viewer(fftfile: str, tmp_path: Path) -> None:
    fs = FourierSeries.from_fft(fftfile)
    spec = fs.form_spec()
    viewer = SpectrumViewer(spec, batch=True, gui=True)
    assert viewer.total_bins == len(spec.data)

    # Click harmonic markers
    ev_click = DummyEvent(inaxes=viewer.ax, xdata=10.5)
    viewer.on_click(ev_click)  # type: ignore[arg-type]
    for line in viewer.marker_lines:
        assert line.get_visible() is True

    # Toggle log-y
    viewer.toggle_logy()
    assert viewer.logy is True
    viewer.toggle_logy()
    assert viewer.logy is False

    # Save
    out_png = tmp_path / "spectrum.png"
    viewer.fig.savefig(out_png)
    assert out_png.exists()
    viewer.close()


def test_timeseries_view_method(timfile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(SeriesViewer, "show", lambda _self: shown.append(True))
    tim = TimeSeries.from_tim(timfile)
    viewer = tim.view(batch=True)
    assert isinstance(viewer, SeriesViewer)
    assert shown == []


def test_fourierseries_and_powerspectrum_view_methods(
    fftfile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(SpectrumViewer, "show", lambda _self: shown.append(True))
    fs = FourierSeries.from_fft(fftfile)
    viewer = fs.view(batch=True)
    assert isinstance(viewer, SpectrumViewer)

    spec = fs.form_spec()
    viewer_spec = spec.view(batch=True)
    assert isinstance(viewer_spec, SpectrumViewer)
    assert shown == []


def test_spectrum_spacing_uses_the_transform_length(tmp_path: Path) -> None:
    header = make_header(
        filename="tone.spec",
        nchans=1,
        nbits=32,
        tsamp=1e-3,
        tobs=0.016,
        tstart=50000.0,
    )
    data = np.zeros(9, dtype=np.float32)
    data[1] = 1.0
    viewer = SpectrumViewer(PowerSpectrum(data, header), batch=True)
    assert viewer.df == pytest.approx(62.5)
    assert viewer.freqs[1] == pytest.approx(62.5)
    out = tmp_path / "tone.png"
    assert viewer.save_png(str(out)) == str(out)
    assert out.exists()


def test_large_spectrum_envelope_preserves_isolated_line(tmp_path: Path) -> None:
    from matplotlib import pyplot as plt

    header = make_header(nchans=1, nbits=32, tsamp=0.001, tobs=32.768)
    data = np.ones(16385, dtype=np.float32)
    data[1234] = 100
    viewer = SpectrumViewer(PowerSpectrum(data, header), batch=True, gui=True)
    try:
        assert not viewer.line.get_visible()
        assert viewer._spec_fill is not None
        assert viewer.ax.get_ylim()[1] >= 100
        viewer.update_plot()
        viewer.on_click(DummyEvent())
        assert not any(line.get_visible() for line in viewer.marker_lines)
        viewer.on_click(DummyEvent(inaxes=viewer.ax, xdata=12.5))
        for harmonic, line in enumerate(viewer.marker_lines, 1):
            np.testing.assert_allclose(line.get_xdata(), 12.5 * harmonic)
        path = viewer.save_png(str(tmp_path / "large.png"))
        assert Path(path).stat().st_size > 1000
    finally:
        viewer.close()
        plt.close("all")


def test_batch_series_exports_selected_interval(tmp_path: Path) -> None:
    header = make_header(nchans=1, nbits=32, tsamp=0.001, tobs=0.128)
    data = np.zeros(128, dtype=np.float32)
    data[32] = 10
    viewer = SeriesViewer(TimeSeries(data, header), batch=True)
    viewer.start_samp, viewer.end_samp = 16, 64
    viewer.update_detail()
    path = viewer.save_png(str(tmp_path / "window.png"))
    assert Path(path).stat().st_size > 1000
    before = viewer.start_samp, viewer.end_samp
    viewer.on_span_select(0.0, 0.001)
    assert (viewer.start_samp, viewer.end_samp) == before


def test_headless_spectrum_log_toggle_does_not_require_widgets() -> None:
    header = make_header(nchans=1, nbits=32, tsamp=0.001, tobs=0.128)
    viewer = SpectrumViewer(
        PowerSpectrum(np.ones(65, dtype=np.float32), header), batch=True
    )
    viewer.update_plot()
    viewer.toggle_logy()
    assert viewer.logy
    viewer.close()
