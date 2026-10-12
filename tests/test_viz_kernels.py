"""Unit tests for visualization Numba kernels and viz foundation modules."""

from __future__ import annotations

import numpy as np

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
    data = np.ones((4, 4), dtype=np.float32)
    data[0, :] = np.nan
    dec = kernels.decimate_2d(data, 2, 2, mode=0)
    assert np.isnan(dec[0, :]).all()
    assert np.isclose(dec[1, :], 1.0).all()


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
