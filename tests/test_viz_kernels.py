"""Unit tests for visualization Numba kernels and viz foundation modules."""

from __future__ import annotations

import numpy as np
import pytest

from sigpyproc import kernels
from sigpyproc.viz._backend import viewer_rc
from sigpyproc.viz.keymap import KeyMap
from sigpyproc.viz.scaling import SCALERS
from sigpyproc.viz.waterfall import History, WaterfallState


def test_decimate_2d_mean() -> None:
    data = np.ones((4, 4), dtype=np.float32)
    dec = kernels.decimate_2d(data, 2, 2, mode=0)
    assert dec.shape == (2, 2)
    np.testing.assert_allclose(dec, 1.0)


def test_decimate_2d_non_integer_ratio() -> None:
    data = np.arange(35, dtype=np.float32).reshape(5, 7)
    dec = kernels.decimate_2d(data, 2, 3, mode=0)
    assert dec.shape == (2, 3)
    assert not np.isnan(dec).any()


def test_decimate_2d_max_spike() -> None:
    data = np.zeros((10, 10), dtype=np.float32)
    data[3, 7] = 42.0
    dec = kernels.decimate_2d(data, 2, 2, mode=1)
    assert dec.shape == (2, 2)
    assert np.max(dec) == 42.0


def test_decimate_2d_nan_handling() -> None:
    """A partly masked tile is the mean of the finite samples."""
    data = np.ones((4, 4), dtype=np.float32)
    data[0, :] = np.nan
    dec = kernels.decimate_2d(data, 2, 2, mode=0)
    np.testing.assert_allclose(dec, 1.0)

    partial = np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float32)
    mean = kernels.decimate_2d(partial, 1, 1, mode=0)
    np.testing.assert_allclose(mean, 1.0)
    assert np.isnan(kernels.decimate_2d(np.full((2, 2), np.nan), 1, 1, mode=0)).all()

    spiked = np.array([[np.nan, 1.0], [-3.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(kernels.decimate_2d(spiked, 1, 1, mode=1), 2.0)
    np.testing.assert_allclose(kernels.decimate_2d(spiked, 1, 1, mode=2), -3.0)


def test_minmax_envelope_skips_nans() -> None:
    x = np.array([np.nan, 1.0, 3.0, np.nan], dtype=np.float32)
    mins, maxs = kernels.minmax_envelope(x, 0, 4, 1)
    np.testing.assert_allclose(mins, 1.0)
    np.testing.assert_allclose(maxs, 3.0)
    empty_mins, empty_maxs = kernels.minmax_envelope(
        np.array([np.nan, np.nan]), 0, 2, 1
    )
    assert np.isnan(empty_mins).all()
    assert np.isnan(empty_maxs).all()


def test_minmax_envelope() -> None:
    x = np.sin(np.linspace(0, 10, 100)).astype(np.float32)
    mins, maxs = kernels.minmax_envelope(x, 0, 100, 10)
    assert len(mins) == 10
    assert len(maxs) == 10
    assert (mins <= maxs).all()


def test_robust_chan_norm_ignores_a_single_spike() -> None:
    data = np.ones((1, 32), dtype=np.float32) * 5
    data[0, 0] = np.float32(1.0e6)
    out = np.empty_like(data)
    kernels.robust_chan_norm(data, out, subsample=1)
    assert abs(float(out[0, 1])) < 1e-3


def test_robust_chan_norm() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=15.0, size=(8, 100)).astype(np.float32)
    out = np.empty_like(data)
    kernels.robust_chan_norm(data, out, subsample=2)
    assert np.allclose(np.median(out, axis=1), 0.0, atol=0.5)


def test_scalers() -> None:
    data = np.array([[-1.0, 0.0], [1.0, 10.0]], dtype=np.float32)
    for scaler in SCALERS:
        prep = scaler.prepare(data)
        vmin, vmax = scaler.clim(prep)
        assert vmin < vmax


def test_state_and_history() -> None:
    hist = History()
    s0 = WaterfallState(start=0, dm=10.0)
    hist.push(s0)

    s1 = WaterfallState(start=4096, dm=15.0)
    back_s = hist.back(s1)
    assert back_s is not None
    assert back_s.start == 0
    assert back_s.dm == 10.0

    fwd_s = hist.forward(back_s)
    assert fwd_s is not None
    assert fwd_s.start == 4096


def test_keymap() -> None:
    km = KeyMap()
    called = []
    km.bind(["right", "l"], lambda: called.append(1), "Next", "Nav")
    help_txt = km.format_help()
    assert "right, l" in help_txt
    assert "Next" in help_txt


def test_viewer_rc() -> None:
    rc_light = viewer_rc("light")
    assert rc_light["keymap.back"] == []
    assert rc_light["figure.facecolor"] == "#f8f9fa"

    rc_dark = viewer_rc("dark")
    assert rc_dark["figure.facecolor"] == "#181818"


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_masked_pixels_keep_real_zeros_and_extreme_values(mode: int) -> None:
    data = np.array([[np.nan, np.inf, 0], [-1e32, 1e32, -3]], dtype=np.float64)
    valid = data[np.isfinite(data)]
    expected = [valid.mean(), valid.max(), valid[np.argmax(np.abs(valid))]][mode]
    np.testing.assert_allclose(
        kernels.decimate_2d(data, 1, 1, mode), expected, atol=1e-6
    )
    low, high = kernels.minmax_envelope(data.ravel(), 0, data.size, 1)
    np.testing.assert_allclose(low, valid.min())
    np.testing.assert_allclose(high, valid.max())


def test_masked_normalization_matches_independent_median_mad() -> None:
    values = np.array([[1, np.nan, 3, np.inf, 5, 7], [np.nan] * 6], dtype=np.float32)
    out = np.empty_like(values)
    kernels.robust_chan_norm(values, out, subsample=1)
    finite = np.isfinite(values[0])
    data = values[0, finite]
    med = np.median(data)
    expected = (data - med) / (1.4826 * np.median(np.abs(data - med)))
    np.testing.assert_allclose(out[0, finite], expected, rtol=1e-6)
    assert np.isnan(out[0, ~finite]).all()
    assert np.isnan(out[1]).all()


def test_visualization_boundaries_preserve_missing_and_empty_results() -> None:
    assert kernels.decimate_2d(np.empty((0, 2)), 1, 1).shape == (1, 1)
    assert np.isnan(kernels.decimate_2d(np.empty((0, 2)), 1, 1)).all()
    assert kernels.decimate_2d(np.ones((2, 2)), 0, 1).size == 0
    for data, rows, cols, mode in [
        (np.ones(2), 1, 1, 0),
        (np.ones((2, 2)), -1, 1, 0),
        (np.ones((2, 2)), 1, 1, 3),
    ]:
        with pytest.raises(ValueError, match="decimation"):
            kernels.decimate_2d(data, rows, cols, mode)
    low, high = kernels.minmax_envelope(np.ones(3), 1, 1, 2)
    assert np.isnan(low).all()
    assert np.isnan(high).all()
    with pytest.raises(ValueError, match="slice"):
        kernels.minmax_envelope(np.ones(3), -1, 2, 1)
    with pytest.raises(ValueError, match="same-shaped"):
        kernels.robust_chan_norm(np.ones((2, 2)), np.empty(2))


def test_rebinding_keys_updates_dispatch_and_help() -> None:
    from types import SimpleNamespace

    keymap = KeyMap()
    calls = []
    keymap.bind(["a", "b"], lambda: calls.append("old"), "Old action")
    keymap.bind("a", lambda: calls.append("new"), "New action")
    assert keymap.handle(SimpleNamespace(key="a"))
    assert keymap.handle(SimpleNamespace(key="b"))
    assert not keymap.handle(SimpleNamespace(key="unknown"))
    assert calls == ["new", "old"]
    assert keymap.help_table() == [
        ("General", "b", "Old action"),
        ("General", "a", "New action"),
    ]


@pytest.mark.parametrize("use_axes", [False, True])
def test_blitting_updates_real_canvas_artists(use_axes: bool) -> None:
    from types import SimpleNamespace

    from matplotlib import pyplot as plt

    from sigpyproc.viz.keymap import BlitManager

    fig, ax = plt.subplots()
    try:
        (line,) = ax.plot([0, 1], [0, 1], color="red")
        manager = BlitManager(fig, [line], axes=ax if use_axes else None)
        assert line.get_animated()
        manager.on_draw(SimpleNamespace(canvas=object()))
        assert manager._bg is None
        manager.update()
        before = np.asarray(fig.canvas.buffer_rgba()).copy()
        line.set_ydata([1, 0])
        manager.update()
        assert np.any(before != np.asarray(fig.canvas.buffer_rgba()))
    finally:
        plt.close(fig)


def test_preferred_gui_backend_failure_is_a_viewer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sigpyproc.viz import _backend

    def fail(_name: str, **_kwargs: object) -> None:
        msg = "backend unavailable"
        raise ImportError(msg)

    monkeypatch.setattr(_backend.mpl, "use", fail)
    with pytest.raises(_backend.ViewerError, match="preferred backend"):
        _backend.ensure_interactive_backend("QtAgg")


def test_unavailable_gui_backends_report_a_headless_workaround(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sigpyproc.viz import _backend

    def fail(_name: str, **_kwargs: object) -> None:
        msg = "backend unavailable"
        raise ImportError(msg)

    monkeypatch.setattr(_backend.mpl, "use", fail)
    monkeypatch.setattr(_backend.mpl, "get_backend", lambda: "Agg")
    monkeypatch.setattr(_backend.os, "uname", lambda: SimpleNamespace(sysname="Linux"))
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(_backend.ViewerError, match="--batch"):
        _backend.ensure_interactive_backend()
    monkeypatch.setenv("DISPLAY", ":1")
    with pytest.raises(_backend.ViewerError, match="Tkinter or PyQt"):
        _backend.ensure_interactive_backend()
