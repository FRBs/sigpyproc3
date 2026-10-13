"""Independent references for the composable pulse-search pipeline."""

import numpy as np
import pytest
from scipy.signal import detrend as scipy_detrend

from sigpyproc import kernels
from sigpyproc.algorithms import box
from sigpyproc.algorithms.box import pulse_search
from sigpyproc.algorithms.common import (
    PRESTO_WIDTHS,
    PulseDetections,
    make_width_bank,
    pulse_candidates,
)
from sigpyproc.algorithms.sps import (
    BoxcarScores,
    PreparedPulseData,
    detect_boxcar_peaks,
    iter_boxcar_scores,
    preprocess_pulse_data,
    prune_pulse_detections,
)
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries


def _prepared(data: np.ndarray, valid: np.ndarray | None = None) -> PreparedPulseData:
    return PreparedPulseData(
        data.astype(np.float32),
        np.ones(data.size, dtype=np.bool_) if valid is None else valid,
        np.array([0]),
        np.array([data.size]),
        np.array([1.0]),
    )


def _direct_scores(data: np.ndarray, valid: np.ndarray, width: int) -> np.ndarray:
    out = np.full(data.size, -np.inf)
    for start in range(data.size - width + 1):
        if valid[start : start + width].all():
            out[start + width // 2] = np.sum(
                data[start : start + width],
                dtype=np.float64,
            ) / np.sqrt(width)
    return out


def _score_arrays(
    prepared: PreparedPulseData,
    widths: list[int],
    *,
    engine: str = "running_sum",
    chunk_size: int = 8192,
) -> dict[int, np.ndarray]:
    arrays = {w: np.full(prepared.data.size, np.nan) for w in widths}
    for scores in iter_boxcar_scores(
        prepared, widths, engine=engine, chunk_size=chunk_size
    ):
        arrays[scores.width][scores.offset : scores.offset + scores.snrs.size] = (
            scores.snrs
        )
    return arrays


class TestWidthBanks:
    def test_presets(self) -> None:
        np.testing.assert_array_equal(make_width_bank(300), PRESTO_WIDTHS)
        np.testing.assert_array_equal(make_width_bank(1000), PRESTO_WIDTHS)
        np.testing.assert_array_equal(make_width_bank(1), [1])
        np.testing.assert_array_equal(
            make_width_bank(13, style="geometric"), [1, 2, 4, 8]
        )
        np.testing.assert_array_equal(
            make_width_bank(9, style="geometric", spacing_factor=1.5),
            [1, 2, 3, 4, 6, 9],
        )
        np.testing.assert_array_equal(
            make_width_bank(64, widths=[16, 3, 3, 1]), [1, 3, 16]
        )
        np.testing.assert_array_equal(
            make_width_bank(64, style="geometric", spacing_factor=1e308), [1]
        )

    @pytest.mark.parametrize(
        "widths", [[], [0], [-1], [1.0], [True], [65], [[1, 2]], [np.nan]]
    )
    def test_invalid_explicit_widths(self, widths: list) -> None:
        with pytest.raises(ValueError, match="widths"):
            make_width_bank(64, widths=widths)


class TestPreprocessing:
    @pytest.mark.parametrize("length", [0, 1, 2, 7, 17, 18, 31])
    @pytest.mark.parametrize("method", ["linear", "median"])
    def test_blocks_and_statistics(self, length: int, method: str) -> None:
        data = np.random.default_rng(2).normal(size=length).astype(np.float32)
        data += np.arange(length, dtype=np.float32) * 0.3 + 100
        original = data.copy()
        prepared = preprocess_pulse_data(data, block_size=8, detrend=method)
        assert prepared.data.size == length
        if length:
            assert prepared.block_starts[0] == 0
            assert prepared.block_ends[-1] == length
            np.testing.assert_array_equal(
                prepared.block_starts[1:], prepared.block_ends[:-1]
            )
        for i, (lo, hi) in enumerate(
            zip(prepared.block_starts, prepared.block_ends, strict=True)
        ):
            raw = data[lo:hi].astype(np.float64)
            expected = (
                scipy_detrend(raw)
                if method == "linear" and raw.size > 1
                else raw - np.median(raw)
            )
            q1, q3 = np.percentile(expected, [25, 75])
            scale = (q3 - q1) / 1.3489795003921634
            if scale == 0:
                scale = np.std(expected)
            if scale > 1e-10:
                assert prepared.noise[i] == pytest.approx(scale, abs=2e-6)
                np.testing.assert_allclose(
                    prepared.data[lo:hi], expected / scale, atol=2e-6
                )
            else:
                assert not prepared.valid[lo:hi].any()
        np.testing.assert_array_equal(data, original)

    @pytest.mark.parametrize("size", [2, 8000, 20001, 65536])
    @pytest.mark.parametrize("offset", [0.0, 1e6, -1e6])
    def test_large_offset_detrending_precision(self, size: int, offset: float) -> None:
        x = np.arange(size)
        data = (offset + x * 0.7 + np.sin(x / 10)).astype(np.float32)
        expected = scipy_detrend(data.astype(np.float64))
        np.testing.assert_allclose(kernels.detrend_1d(data), expected, atol=1e-6)

    def test_noise_rejection_and_bright_pulses(self) -> None:
        data = np.random.default_rng(5).normal(size=24 * 512).astype(np.float32)
        data[5 * 512 : 6 * 512] *= 20
        data[8 * 512 : 9 * 512] = 0
        data[10 * 512 + 100 : 10 * 512 + 108] += 50
        default = preprocess_pulse_data(data, block_size=512)
        masked = preprocess_pulse_data(data, block_size=512, mask_bad_blocks=True)
        assert default.valid[5 * 512 : 6 * 512].all()
        assert not masked.valid[5 * 512 : 6 * 512].any()
        assert not default.valid[8 * 512 : 9 * 512].any()
        assert default.valid[10 * 512 : 11 * 512].all()
        median_masked = preprocess_pulse_data(
            data,
            block_size=512,
            detrend="median",
            mask_bad_blocks=True,
        )
        assert median_masked.valid[10 * 512 : 11 * 512].all()
        assert np.isfinite(masked.data).all()

    def test_iqr_fallback_and_small_block_count(self) -> None:
        data = np.zeros(32, dtype=np.float32)
        data[10] = 8
        prepared = preprocess_pulse_data(data, detrend="median", mask_bad_blocks=True)
        assert prepared.noise[0] == pytest.approx(np.std(data, dtype=np.float64))
        assert prepared.valid.all()

    def test_zero_noise_dispersion(self) -> None:
        data = np.tile(np.array([-1, 1], np.float32), 64)
        prepared = preprocess_pulse_data(
            data, block_size=8, detrend="median", mask_bad_blocks=True
        )
        assert prepared.valid.all()


class TestBoxcarScoring:
    @pytest.mark.parametrize("engine", ["running_sum", "fft"])
    @pytest.mark.parametrize("chunk_size", [1, 7, 64, 1000])
    def test_matches_direct_reference(self, engine: str, chunk_size: int) -> None:
        data = np.random.default_rng(3).normal(size=129)
        valid = np.ones(data.size, dtype=np.bool_)
        valid[21:25] = False
        valid[65] = False
        prepared = _prepared(data, valid)
        widths = [1, 2, 3, 4, 17, 128, 129]
        arrays = _score_arrays(prepared, widths, engine=engine, chunk_size=chunk_size)
        for width in widths:
            expected = _direct_scores(prepared.data, valid, width)
            np.testing.assert_array_equal(
                np.isfinite(arrays[width]), np.isfinite(expected)
            )
            np.testing.assert_allclose(arrays[width], expected, atol=2e-12, rtol=2e-12)

    @pytest.mark.parametrize("engine", ["running_sum", "fft"])
    @pytest.mark.parametrize("width", [1, 2, 3, 8, 17])
    def test_all_pulse_phases_and_edges(self, engine: str, width: int) -> None:
        for start in [0, 1, 30, 31, 32, 33, 128 - width]:
            data = np.zeros(128, np.float32)
            data[start : start + width] = 2
            prepared = _prepared(data)
            scores = _score_arrays(prepared, [width], engine=engine, chunk_size=32)[
                width
            ]
            assert scores[start + width // 2] == pytest.approx(
                2 * np.sqrt(width), abs=1e-12
            )
            assert np.argmax(scores) == start + width // 2
            np.testing.assert_allclose(
                scores, _direct_scores(data, prepared.valid, width), atol=2e-12
            )

    def test_threshold_and_owned_score_buffers(self) -> None:
        scores = BoxcarScores(10, 3, np.array([5, 6, np.inf, np.nan, -np.inf, 7]))
        detected = detect_boxcar_peaks(scores)
        np.testing.assert_array_equal(detected.bins, [11, 15])
        np.testing.assert_array_equal(detected.snrs, [6, 7])
        np.testing.assert_array_equal(detected.widths, [3, 3])
        iterator = iter_boxcar_scores(_prepared(np.arange(20)), [1, 2], chunk_size=3)
        first = next(iterator)
        original = first.snrs.copy()
        list(iterator)
        np.testing.assert_array_equal(first.snrs, original)


class TestPruning:
    def test_chain_only_accepted_candidates_suppress(self) -> None:
        detected = PulseDetections(
            np.array([0, 2, 4]), np.array([10.0, 9.0, 8.0]), np.array([4, 4, 4])
        )
        pruned = prune_pulse_detections(detected)
        np.testing.assert_array_equal(pruned.bins, [0, 4])

    def test_cross_widths_and_ties(self) -> None:
        detected = PulseDetections(
            np.array([10, 15, 20, 40, 41, 60, 60]),
            np.array([10.0, 9.0, 8.0, 7.0, 7.0, 6.0, 6.0]),
            np.array([2, 12, 2, 1, 1, 4, 2]),
        )
        separate = prune_pulse_detections(detected, across_widths=False)
        assert 15 in separate.bins
        assert 40 in separate.bins
        assert 41 in separate.bins
        pruned = prune_pulse_detections(separate)
        np.testing.assert_array_equal(pruned.bins, [10, 20, 40, 60])
        np.testing.assert_array_equal(pruned.widths, [2, 2, 1, 2])
        np.testing.assert_array_equal(
            prune_pulse_detections(separate, deadtime=20).bins, [10, 40]
        )

    def test_unsorted_arrays_and_empty(self) -> None:
        detected = PulseDetections(
            np.array([20, 10, 11]), np.array([7.0, 8.0, 9.0]), np.array([2, 2, 2])
        )
        np.testing.assert_array_equal(prune_pulse_detections(detected).bins, [11, 20])
        assert prune_pulse_detections(PulseDetections.empty()).bins.size == 0


class TestPipeline:
    @pytest.mark.parametrize("engine", ["running_sum", "fft"])
    def test_injections_across_blocks_and_final_chunk(self, engine: str) -> None:
        data = np.random.default_rng(10).normal(size=10003).astype(np.float32)
        starts, widths = [0, 1996, 8189, 9970], [3, 8, 17, 9]
        for start, width in zip(starts, widths, strict=True):
            data[start : start + width] += 6
        before = data.copy()
        cands = pulse_search(
            data,
            0.001,
            engine=engine,
            widths=widths,
            max_width=32,
            block_size=2000,
            dm=12,
            accel=3,
        )
        for start, width in zip(starts, widths, strict=True):
            assert np.min(np.abs(cands.values("bin") - (start + width // 2))) <= 2
        assert (cands.values("kind") == "pulse").all()
        assert np.isnan(cands.values("period_s")).all()
        assert (cands.values("dm") == 12).all()
        assert (cands.values("accel") == 3).all()
        assert (np.diff(cands.values("snr")) <= 0).all()
        np.testing.assert_array_equal(data, before)

    def test_methods_engines_and_object_wrapper(self) -> None:
        data = np.random.default_rng(8).normal(size=10003).astype(np.float32)
        data[8190:8198] += 8
        header = Header(
            filename="pulse.tim",
            data_type="time series",
            nchans=1,
            foff=1.0,
            fch1=1400.0,
            nbits=32,
            tsamp=0.001,
            tstart=50000.0,
            nsamples=data.size,
            dm=12.0,
        )
        ts = TimeSeries(data, header)
        running = ts.find_pulses(widths=[8])
        fft = ts.find_pulses(widths=[8], engine="fft", chunk_size=317)
        direct = box.pulse_search(data, 0.001, widths=[8], dm=12.0)
        for col in ("bin", "snr", "width_s", "dm"):
            np.testing.assert_allclose(running.values(col), fft.values(col), rtol=1e-6)
            np.testing.assert_array_equal(running.values(col), direct.values(col))
        assert running.values("bin")[0] == 8194
        old = box.boxcar_pulse_search(
            data,
            0.001,
            snr_min=6.0,
            max_width=64,
            spacing_factor=2.0,
            deadtime=8,
            dm=12.0,
            accel=0.0,
        )
        retained = ts.find_pulses(method="matched_filter", max_width=64)
        for col in ("bin", "snr", "width_s"):
            np.testing.assert_array_equal(old.values(col), retained.values(col))
        assert retained.values("bin")[0] == 8190

    @pytest.mark.parametrize("method", ["presto", "matched_filter"])
    def test_empty_and_constant(self, method: str) -> None:
        assert len(pulse_search([], 0.001, method=method)) == 0
        assert len(pulse_search(np.ones(512), 0.001, method=method)) == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"method": "unknown"},
            {"engine": "unknown"},
            {"width_style": "unknown"},
            {"width_style": ""},
            {"detrend": "unknown"},
            {"max_width": 0},
            {"max_width": True},
            {"max_width": 2.5},
            {"max_width": 1000},
            {"deadtime": -1},
            {"block_size": 1},
            {"chunk_size": 0},
            {"spacing_factor": 1},
            {"spacing_factor": np.inf},
            {"snr_min": 0},
            {"snr_min": np.nan},
            {"snr_min": np.inf},
        ],
    )
    def test_invalid_options(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            pulse_search(np.ones(64), 0.001, **kwargs)

    @pytest.mark.parametrize(
        "data", [[np.nan], [np.inf], [-np.inf], [[1, 2]], [1j], [1e100], ["1"]]
    )
    def test_invalid_samples(self, data: list) -> None:
        with pytest.raises(ValueError, match="data"):
            pulse_search(data, 0.001, max_width=1)

    @pytest.mark.parametrize("tsamp", [0, -1, np.nan, np.inf])
    def test_invalid_sampling_interval(self, tsamp: float) -> None:
        with pytest.raises(ValueError, match="tsamp"):
            pulse_search(np.ones(16), tsamp, max_width=1)

    def test_invalid_stage_inputs(self) -> None:
        prepared = _prepared(np.ones(16))
        with pytest.raises(ValueError, match="engine"):
            list(iter_boxcar_scores(prepared, [1], engine="bad"))
        with pytest.raises(ValueError, match="widths"):
            list(iter_boxcar_scores(prepared, []))
        bad_mask = _prepared(np.ones(16), np.ones(3, np.bool_))
        with pytest.raises(ValueError, match="valid"):
            list(iter_boxcar_scores(bad_mask, [1]))
        bad = PulseDetections(np.array([1]), np.array([np.nan]), np.array([1]))
        with pytest.raises(ValueError, match="detections"):
            prune_pulse_detections(bad)
        with pytest.raises(ValueError, match="detections"):
            pulse_candidates(bad, 0.001)

    def test_candidate_tie_order(self) -> None:
        detected = PulseDetections(
            np.array([30, 10, 20]), np.array([8.0, 8.0, 8.0]), np.array([4, 2, 2])
        )
        table = pulse_candidates(detected, 0.002)
        np.testing.assert_array_equal(table.values("bin"), [10, 20, 30])
        np.testing.assert_allclose(table.values("width_s"), [0.004, 0.004, 0.008])
