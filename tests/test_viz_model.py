"""Unit tests for WaterfallModel."""

from __future__ import annotations

import numpy as np

from sigpyproc.core.rfi import RFIMask
from sigpyproc.readers import FilReader
from sigpyproc.viz.scaling import SCALERS
from sigpyproc.viz.waterfall import WaterfallModel, WaterfallState


def test_model_clamping(filfile_8bit_1: str) -> None:
    fil = FilReader(filfile_8bit_1)
    model = WaterfallModel(fil)
    state = WaterfallState(start=fil.header.nsamples + 5000, nsamps=10000000)
    clamped = model.clamp(state)
    assert clamped.start < fil.header.nsamples
    assert clamped.nsamps <= fil.header.nsamples


def test_model_products(filfile_8bit_1: str) -> None:
    fil = FilReader(filfile_8bit_1)
    model = WaterfallModel(fil)
    state = WaterfallState(start=0, nsamps=1024, dm=5.0)
    products = model.products(state, screen_size=(128, 256))

    assert products.image.shape == (128, 256)
    assert len(products.ts) == 1024
    assert len(products.bp_mean) == fil.header.nchans
    assert products.extent[0] == 0.0
    assert products.extent[1] > 0.0
    assert not np.isnan(products.image).all()


def test_model_with_mask_and_zapping(filfile_8bit_1: str) -> None:
    fil = FilReader(filfile_8bit_1)
    # Create empty dummy mask with channel 5 masked
    mean = np.zeros(fil.header.nchans)
    var = np.ones(fil.header.nchans)
    mask = RFIMask(
        threshold=3.0,
        header=fil.header,
        chan_mean=mean,
        chan_var=var,
        chan_skew=mean,
        chan_kurt=mean,
        chan_maxima=mean,
        chan_minima=mean,
    )
    mask.chan_mask[5] = True

    model = WaterfallModel(fil, mask=mask)
    state = WaterfallState(start=0, nsamps=512, zapped_channels={10})
    blk, _ = model.get_block(state)

    assert np.isnan(blk.data[5, :]).all()
    assert np.isnan(blk.data[10, :]).all()


def test_frequency_orientation_high_freq_first(filfile_8bit_1: str) -> None:
    fil = FilReader(filfile_8bit_1)
    assert fil.header.foff < 0
    model = WaterfallModel(fil)
    nsamps = 128
    state = WaterfallState(start=0, nsamps=nsamps, scaler_idx=len(SCALERS) - 1)
    products = model.products(state, screen_size=(fil.header.nchans, nsamps))
    raw = model._raw_block(state)
    np.testing.assert_allclose(products.image[0], raw.data[-1], atol=1e-5)
    np.testing.assert_allclose(products.image[-1], raw.data[0], atol=1e-5)
    assert products.extent[2] < products.extent[3]


def test_zap_does_not_reread(filfile_8bit_1: str) -> None:
    fil = FilReader(filfile_8bit_1)
    model = WaterfallModel(fil)
    state = WaterfallState(start=0, nsamps=128, zapped_channels={1})
    blocked, _ = model.get_block(state)
    assert np.isnan(blocked.data[1]).all()
    assert len(model._cache) == 1
    state.zapped_channels.add(4)
    blocked, _ = model.get_block(state)
    assert np.isnan(blocked.data[4]).all()
    assert len(model._cache) == 1
    cached = next(iter(model._cache.values()))
    assert np.isfinite(cached.data).all()


def test_scrunch_snaps_to_a_divisor(filfile_8bit_1: str) -> None:
    fil = FilReader(filfile_8bit_1)
    model = WaterfallModel(fil)
    state = WaterfallState(start=0, nsamps=1000, tfactor=3, ffactor=3)
    clamped = model.clamp(state)
    assert clamped.nsamps % clamped.tfactor == 0
    assert fil.header.nchans % clamped.ffactor == 0


def test_fit_dm() -> None:
    # Dispersion delay formula: dt = DM * 4148.808 * (f2^-2 - f1^-2)
    f1, f2 = 1400.0, 1200.0
    dt = 0.25
    expected_dm = dt / (4148.808 * ((f2**-2) - (f1**-2)))
    fitted = WaterfallModel.fit_dm(0.0, f1, dt, f2)
    assert np.isclose(fitted, expected_dm, rtol=1e-3)


def test_history_has_detached_undo_redo_and_bounded_capacity() -> None:
    from sigpyproc.viz.waterfall import History

    history = History(maxlen=2)
    state = WaterfallState(zapped_channels={1})
    assert history.back(state) is None
    assert history.forward(state) is None
    for start in [0, 10, 20]:
        state.start = start
        history.push(state)
    state.zapped_channels.add(2)
    previous = history.back(state)
    assert previous.start == 20
    assert previous.zapped_channels == {1}
    restored = history.forward(previous)
    assert restored.zapped_channels == {1, 2}
    history.push(WaterfallState(start=30))
    assert history.forward(state) is None
    assert len(history._past) == 2


def test_model_cache_evicts_old_windows_and_owns_data(filfile_8bit_1: str) -> None:
    with FilReader(filfile_8bit_1) as reader:
        model = WaterfallModel(reader, cache_blocks=1)
        first = model._raw_block(WaterfallState(start=0, nsamps=64))
        expected = first.data.copy()
        second = model._raw_block(WaterfallState(start=64, nsamps=64))
        assert second is not first
        assert len(model._cache) == 1
        np.testing.assert_array_equal(first.data, expected)
        assert model._raw_block(WaterfallState(start=64, nsamps=64)) is second


def test_zero_dm_removes_common_mode_without_changing_cached_input(
    filfile_8bit_1: str,
) -> None:
    with FilReader(filfile_8bit_1) as reader:
        model = WaterfallModel(reader)
        state = WaterfallState(nsamps=128)
        original = model._raw_block(state).data.copy()
        state.zerodm = True
        filtered, _ = model.get_block(state)
        np.testing.assert_allclose(filtered.data.mean(axis=0), 0, atol=1e-4)
        np.testing.assert_array_equal(model._raw_block(state).data, original)
