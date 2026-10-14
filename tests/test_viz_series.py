from __future__ import annotations

from typing import TYPE_CHECKING

from sigpyproc.fourierseries import FourierSeries
from sigpyproc.timeseries import TimeSeries
from sigpyproc.viz.series import SeriesViewer, SpectrumViewer

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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
