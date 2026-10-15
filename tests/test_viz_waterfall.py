"""Unit tests for WaterfallViewer, interactive events, and batch rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigpyproc.readers import open_filterbank
from sigpyproc.viz.batch import render_waterfall_batch
from sigpyproc.viz.waterfall import WaterfallModel, WaterfallViewer


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


def test_waterfall_viewer_init(filfile_8bit_1: str) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gulp=1024)
    assert viewer.state.nsamps == 1024
    assert viewer.state.start == 0
    assert viewer.state.dm == 0.0
    assert len(viewer.file_list) == 1


def test_filterbank_view_batch_does_not_show(
    filfile_8bit_1: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(WaterfallViewer, "show", lambda _self: shown.append(True))
    fil = open_filterbank(filfile_8bit_1)
    viewer = fil.view(batch=True, gulp=128)
    assert isinstance(viewer, WaterfallViewer)
    assert shown == []


def test_waterfall_viewer_navigation(filfile_8bit_1: str) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gulp=512)
    # Next gulp
    viewer.next_gulp()
    assert viewer.state.start == 512

    # Step gulp (+0.5)
    viewer.step_gulp(0.5)
    assert viewer.state.start == 512 + 256

    # Prev gulp
    viewer.prev_gulp()
    assert viewer.state.start == 256

    # Start of file
    viewer.goto_start()
    assert viewer.state.start == 0

    # End of file
    viewer.goto_end()
    assert viewer.state.start > 0

    # History back
    viewer.history_back()
    assert viewer.state.start == 0


def test_waterfall_viewer_dm_control(filfile_8bit_1: str) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True)
    viewer.step_dm(5.0)
    assert np.isclose(viewer.state.dm, 5.0)

    viewer.step_dm(-2.0)
    assert np.isclose(viewer.state.dm, 3.0)

    # Toggle DM
    viewer.toggle_dm()
    assert viewer.state.dm == 0.0
    viewer.toggle_dm()
    assert np.isclose(viewer.state.dm, 3.0)


def test_waterfall_viewer_display_toggles(filfile_8bit_1: str) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gui=True)
    initial_cmap = viewer.state.cmap
    viewer.cycle_cmap()
    assert viewer.state.cmap != initial_cmap

    viewer.toggle_invert()
    assert viewer.state.cmap.endswith("_r")

    viewer.step_time_downsample(larger=True)
    assert viewer.state.tfactor > 1
    assert viewer.state.nsamps % viewer.state.tfactor == 0
    viewer.step_freq_downsample(larger=True)
    assert viewer.state.ffactor > 1
    assert viewer.fil.header.nchans % viewer.state.ffactor == 0

    assert viewer.state.theme == "light"
    viewer.toggle_theme()
    assert viewer.state.theme == "dark"
    assert viewer.fig.get_facecolor()[:3] == pytest.approx(
        (24 / 255, 24 / 255, 24 / 255),
        abs=0.02,
    )

    assert viewer.state.hires is False
    viewer.toggle_hires()
    assert viewer.state.hires is True

    labels = [button.label.get_text() for button in viewer._buttons]
    for name in ("Start", "Prev", "Next", "End", "Fit DM", "Help", "Save mask"):
        assert name in labels
    viewer.close()


def test_waterfall_viewer_bandpass_zapping(filfile_8bit_1: str) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gui=True)
    freqs = viewer.fil.header.chan_freqs

    # Simulate bandpass press and release
    ev_press = DummyEvent(inaxes=viewer.ax_bp, ydata=float(freqs[10]))
    viewer.on_button_press(ev_press)  # type: ignore[arg-type]
    assert viewer._bp_pressed is True

    ev_release = DummyEvent(inaxes=viewer.ax_bp, ydata=float(freqs[12]))
    viewer.on_button_release(ev_release)  # type: ignore[arg-type]
    assert viewer._bp_pressed is False
    assert len(viewer.state.zapped_channels) > 0
    viewer.close()


def test_dm_fit_waits_until_armed(filfile_8bit_1: str) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gui=True, gulp=256)
    first = DummyEvent(inaxes=viewer.ax_im, xdata=0.0, ydata=1400.0)
    second = DummyEvent(inaxes=viewer.ax_im, xdata=0.2, ydata=1200.0)
    viewer.on_button_press(first)  # type: ignore[arg-type]
    viewer.on_button_press(second)  # type: ignore[arg-type]
    assert viewer.state.dm == 0.0

    viewer.start_dm_fit()
    viewer.on_button_press(first)  # type: ignore[arg-type]
    viewer.on_button_press(second)  # type: ignore[arg-type]
    assert viewer.state.dm > 0.0
    assert viewer.fit_curve.get_xdata().size > 0
    viewer.close()


def test_file_key_is_not_the_scaler(filfile_8bit_1: str) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gulp=256)
    assert viewer.keymap._bindings["N"][0] == viewer.next_file
    assert viewer.keymap._bindings["n"][0] == viewer.cycle_scaler
    assert "n, N" not in viewer.keymap.format_help()


def test_waterfall_viewer_save_and_export(filfile_8bit_1: str, tmp_path: Path) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gui=True)
    out_png = tmp_path / "test.png"
    viewer.fig.savefig(out_png)
    assert out_png.exists()
    viewer.close()


def test_export_mask_keeps_channel_mean(filfile_8bit_1: str, tmp_path: Path) -> None:
    from sigpyproc.core.rfi import RFIMask

    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gulp=256)
    viewer.state.zapped_channels.add(3)
    out = tmp_path / "zap.h5"
    viewer.export_rfimask(str(out))
    loaded = RFIMask.from_file(str(out))
    assert loaded.chan_mask[3]
    assert np.any(loaded.chan_mean != 0)


def test_render_waterfall_batch_png(filfile_8bit_1: str, tmp_path: Path) -> None:
    fil = open_filterbank(filfile_8bit_1)
    out_files = render_waterfall_batch(
        fil,
        outdir=tmp_path,
        start=0,
        stop=2048,
        gulp=1024,
        out_format="png",
        jobs=1,
    )
    assert len(out_files) == 2
    for f in out_files:
        assert Path(f).exists()


def test_render_waterfall_batch_respects_mask(
    filfile_8bit_1: str,
    maskfile: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    real_init = WaterfallModel.__init__

    def _spy(self: WaterfallModel, fil: object, **kwargs: object) -> None:
        seen["mask"] = kwargs.get("mask")
        real_init(self, fil, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WaterfallModel, "__init__", _spy)
    fil = open_filterbank(filfile_8bit_1)
    render_waterfall_batch(
        fil,
        outdir=tmp_path,
        start=0,
        stop=512,
        gulp=512,
        mask=maskfile,
        jobs=1,
    )
    assert seen["mask"] is not None


def test_render_waterfall_batch_jobs(filfile_8bit_1: str, tmp_path: Path) -> None:
    fil = open_filterbank(filfile_8bit_1)
    out_files = render_waterfall_batch(
        fil,
        outdir=tmp_path,
        start=0,
        stop=1024,
        gulp=512,
        out_format="png",
        jobs=2,
    )
    assert len(out_files) == 2
    for path in out_files:
        assert Path(path).is_file()


def test_render_waterfall_batch_pdf_multi(filfile_8bit_1: str, tmp_path: Path) -> None:
    fil = open_filterbank(filfile_8bit_1)
    out_files = render_waterfall_batch(
        fil,
        outdir=tmp_path,
        start=0,
        stop=2048,
        gulp=1024,
        out_format="pdf-multi",
    )
    assert len(out_files) == 1
    assert Path(out_files[0]).exists()


def test_save_npz_uses_the_selected_block(filfile_8bit_1: str, tmp_path: Path) -> None:
    fil = open_filterbank(filfile_8bit_1)
    viewer = WaterfallViewer(fil, batch=True, start=20, gulp=32)
    out = tmp_path / "block.npz"
    viewer.save_npz(str(out))
    loaded = np.load(out, allow_pickle=False)
    assert loaded["nsamples"] == 32
    assert loaded["tstart"] == pytest.approx(fil.header.mjd_after_nsamps(20))
    assert loaded["tsamp"] == pytest.approx(fil.header.tsamp)
    assert loaded["foff"] == pytest.approx(fil.header.foff)
    assert loaded["dm"] == pytest.approx(0.0)
    half = abs(fil.header.foff) / 2
    products = viewer.model.products(viewer.state)
    assert products.extent[2] == pytest.approx(
        float(np.min(fil.header.chan_freqs)) - half
    )
    assert products.extent[3] == pytest.approx(
        float(np.max(fil.header.chan_freqs)) + half
    )
