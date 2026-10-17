"""CPU source-model tests; these do not claim validation against a GPU run."""

from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sigpyproc.algorithms.box import pulse_search
from sigpyproc.algorithms.common import make_width_bank
from sigpyproc.algorithms.heimdall import (
    HeimdallConfig,
    HeimdallGiants,
    HeimdallPrepared,
    HeimdallScores,
    cluster_heimdall_giants,
    find_heimdall_giants,
    heimdall_candidates,
    heimdall_pulse_search,
    iter_heimdall_gulps,
    iter_heimdall_scores,
    preprocess_heimdall,
)
from sigpyproc.apps.spp_search import main
from sigpyproc.core.candidates import (
    HEIMDALL_CANDIDATE_NAMES,
    PULSE_CANDIDATE_NAMES,
    CandidateTable,
)
from sigpyproc.header import Header
from sigpyproc.kernels import search as search_kernels
from sigpyproc.timeseries import TimeSeries


def _reference_scrunch(data: np.ndarray) -> np.ndarray:
    if data.size < 5:
        return np.array([np.median(data)])
    return np.median(data[: data.size // 5 * 5].reshape(-1, 5), axis=1)


def _reference_stretch(data: np.ndarray, size: int) -> np.ndarray:
    # Evaluate a piecewise linear function at the source's phase-shifted
    # coordinates, including linear extrapolation outside the knot range.
    step = (size - 1) / (data.size - 1)
    coordinates = np.arange(size) / step - int(step / 2) / step
    left = np.clip(coordinates.astype(np.int64), 0, data.size - 2)
    return data[left] + (coordinates - left) * np.diff(data)[left]


def _reference_baseline(data: np.ndarray, radius: int) -> np.ndarray:
    knots = int(data.size / radius + 0.5)
    if knots <= 1:
        return np.zeros(data.size)
    levels = int(np.floor(np.log(data.size // knots) / np.log(5)))
    values = _reference_stretch(data, knots * 5**levels)
    for _ in range(levels):
        values = _reference_scrunch(values)
    expanded = _reference_stretch(values, 2 * knots)
    anchors = np.concatenate(
        ([2 * expanded[0] - expanded[1]], expanded, [2 * expanded[-1] - expanded[-2]])
    )
    return _reference_stretch(anchors, data.size)


def _series(data: np.ndarray, tsamp: float = 0.001) -> TimeSeries:
    header = Header(
        filename="heimdall.tim",
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=tsamp,
        tstart=50000.0,
        nsamples=data.size,
        dm=42.0,
    )
    return TimeSeries(data.astype(np.float32), header)


def _giants(bins: list[int], snrs: list[float], widths: list[int]) -> HeimdallGiants:
    indices = np.array(bins, np.int64)
    return HeimdallGiants(
        indices,
        np.array(snrs, np.float64),
        np.array(widths, np.int64),
        indices - 1,
        indices + 2,
        np.ones(len(bins), np.int64),
    )


class TestHeimdallStatistics:
    @pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 24, 25, 26, 127, 262144])
    def test_median_tree_rms(self, length: int) -> None:
        data = np.random.default_rng(20).normal(size=length)
        np.testing.assert_allclose(
            search_kernels.heimdall_median_scrunch(data), _reference_scrunch(data)
        )
        reduced = np.abs(data)
        while reduced.size > 1:
            reduced = _reference_scrunch(reduced)
        assert search_kernels.heimdall_rms(data) == pytest.approx(
            1.4862 * reduced[0], abs=1e-12
        )

    def test_rms_remainder_is_discarded(self) -> None:
        assert search_kernels.heimdall_rms(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1e9])
        ) == pytest.approx(1.4862 * 3)
        # Hierarchical medians are intentionally not an exact global MAD.
        data = np.array(
            [
                0.0,
                0.0,
                10.0,
                11.0,
                12.0,
                0.0,
                0.0,
                10.0,
                11.0,
                12.0,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
            ]
        )
        assert search_kernels.heimdall_rms(data) != pytest.approx(
            1.4862 * np.median(np.abs(data))
        )

    def test_stretch_phase_and_safe_short_cases(self) -> None:
        np.testing.assert_allclose(
            search_kernels.heimdall_linear_stretch(np.array([0.0, 10.0]), 4),
            [-10 / 3, 0, 10 / 3, 20 / 3],
        )
        np.testing.assert_array_equal(
            search_kernels.heimdall_linear_stretch(np.array([3.0]), 4), [3, 3, 3, 3]
        )
        assert search_kernels.heimdall_rms(np.array([])) == 0
        assert search_kernels.heimdall_median_scrunch(np.array([])).size == 0

    @pytest.mark.parametrize(
        ("length", "radius"), [(125, 25), (1000, 50), (1031, 125), (14, 2), (200, 500)]
    )
    def test_baseline_matches_source_model(self, length: int, radius: int) -> None:
        data = np.random.default_rng(21).normal(size=length) + np.arange(length) * 0.01
        np.testing.assert_allclose(
            search_kernels.heimdall_baseline(data, radius),
            _reference_baseline(data, radius),
            atol=1e-12,
        )

    def test_preprocessing_and_no_input_mutation(self) -> None:
        data = (100 + np.random.default_rng(22).normal(size=8192)).astype(np.float32)
        original = data.copy()
        prepared = preprocess_heimdall(data, 0.001, baseline_length=0.2)
        baseline = _reference_baseline(data.astype(np.float64), 100)
        np.testing.assert_allclose(prepared.baseline, baseline)
        reduced = np.abs(data - baseline)
        while reduced.size > 1:
            reduced = _reference_scrunch(reduced)
        expected_scale = 1.4862 * reduced[0]
        assert prepared.noise == pytest.approx(expected_scale)
        np.testing.assert_allclose(prepared.data, (data - baseline) / expected_scale)
        np.testing.assert_array_equal(data, original)

    @pytest.mark.parametrize("data", [[], [1.0], np.ones(1024), np.zeros(1024)])
    def test_degenerate_preprocessing(self, data: list | np.ndarray) -> None:
        prepared = preprocess_heimdall(data, 0.001)
        assert prepared.noise == 0
        assert not np.any(prepared.data)


class TestHeimdallScores:
    @pytest.mark.parametrize("minimum", [2, 8, 4096])
    @pytest.mark.parametrize("ceiling", [16, 17])
    def test_prefix_sums_and_strides(self, minimum: int, ceiling: int) -> None:
        data = np.random.default_rng(23).normal(size=127)
        prepared = HeimdallPrepared(data, np.zeros(data.size), 1.0)
        for scores in iter_heimdall_scores(
            prepared, [1, 2, 4, 8, 16], max_width=ceiling, min_tscrunch_width=minimum
        ):
            stride = max(1, 2 * scores.width // minimum)
            centers = np.arange(
                ceiling // 2, ceiling // 2 + data.size + 1 - ceiling, stride
            )
            expected = np.array(
                [
                    np.sum(
                        data[
                            b - scores.width // 2 : b - scores.width // 2 + scores.width
                        ]
                    )
                    / np.sqrt(scores.width)
                    for b in centers
                ]
            )
            assert scores.offset == ceiling // 2
            assert scores.stride == stride
            np.testing.assert_allclose(scores.snrs, expected, atol=1e-12)

    def test_filter_renormalization(self) -> None:
        data = np.random.default_rng(24).normal(size=1024)
        prepared = HeimdallPrepared(data, np.zeros(data.size), 1.0)
        ordinary = next(iter_heimdall_scores(prepared, [16], max_width=64))
        renorm = next(
            iter_heimdall_scores(prepared, [16], max_width=64, boxcar_renorm=True)
        )
        raw = ordinary.snrs * 4
        np.testing.assert_allclose(
            renorm.snrs, raw / search_kernels.heimdall_rms(raw), atol=1e-12
        )

    def test_invalid_widths_and_prepared_data(self) -> None:
        prepared = HeimdallPrepared(np.ones(16), np.zeros(16), 1.0)
        for widths in ([3], [32], []):
            with pytest.raises(ValueError):
                list(iter_heimdall_scores(prepared, widths, max_width=16))
        assert (
            list(
                iter_heimdall_scores(
                    HeimdallPrepared(np.zeros(16), np.zeros(16), 0.0), [1], max_width=16
                )
            )
            == []
        )


class TestHeimdallGiants:
    def test_strict_threshold_gaps_ties_and_bounds(self) -> None:
        scores = HeimdallScores(
            100,
            4,
            2,
            np.array([6.0, 7.0, 0.0, 8.0, 8.0, 0.0, 0.0, 9.0, np.inf, np.nan]),
        )
        found = find_heimdall_giants(scores, snr_min=6, cand_sep_time=1)
        np.testing.assert_array_equal(found.bins, [106, 114])
        np.testing.assert_array_equal(found.snrs, [8, 9])
        np.testing.assert_array_equal(found.begins, [102, 114])
        np.testing.assert_array_equal(found.ends, [110, 116])
        np.testing.assert_array_equal(found.members, [1, 1])
        # First peak wins a tie; it does not move to the later equal sample.
        np.testing.assert_array_equal(found.widths, [4, 4])

    def test_zero_tolerance_separates_adjacent_crossings(self) -> None:
        found = find_heimdall_giants(
            HeimdallScores(0, 1, 1, np.array([7.0, 8.0, 9.0])), cand_sep_time=0
        )
        np.testing.assert_array_equal(found.bins, [0, 1, 2])

    def test_transitive_components_and_members(self) -> None:
        giants = _giants([40, 10, 20, 80], [7.0, 8.0, 9.0, 10.0], [16, 4, 8, 1])
        clustered = cluster_heimdall_giants(giants, cand_sep_time=2, cand_sep_filter=1)
        np.testing.assert_array_equal(clustered.bins, [80, 20])
        np.testing.assert_array_equal(clustered.members, [1, 3])
        # Bounds belong to the strongest member, rather than the union.
        np.testing.assert_array_equal(clustered.begins, [79, 19])
        np.testing.assert_array_equal(clustered.ends, [82, 22])
        np.testing.assert_array_equal(giants.bins, [40, 10, 20, 80])

    def test_filter_tolerance_and_tie_order(self) -> None:
        giants = _giants([20, 10, 10], [8.0, 8.0, 8.0], [16, 4, 2])
        separate = cluster_heimdall_giants(giants, cand_sep_filter=0)
        assert separate.bins.size == 3
        joined = cluster_heimdall_giants(giants, cand_sep_filter=3)
        np.testing.assert_array_equal(joined.bins, [10])
        np.testing.assert_array_equal(joined.widths, [2])
        np.testing.assert_array_equal(joined.members, [3])


class TestHeimdallPipeline:
    def test_gulp_overlap_and_partial_tail(self) -> None:
        original = np.arange(100, dtype=np.float32)
        gulps = list(iter_heimdall_gulps(original, gulp_size=32, max_width=8))
        assert [(offset, len(data)) for offset, data in gulps] == [
            (0, 32),
            (24, 40),
            (56, 40),
            (88, 12),
        ]
        for offset, data in gulps:
            np.testing.assert_array_equal(data, original[offset : offset + len(data)])
            assert not data.flags.writeable
        assert original.flags.writeable

    def test_boundary_pulses_and_wrapper(self) -> None:
        data = np.random.default_rng(25).normal(size=16431).astype(np.float32)
        centers = [4064, 4096, 16390]
        # Use separate tests per position so broad filters cannot connect two
        # intentionally nearby pulses under Heimdall's clustering rules.
        config = HeimdallConfig(gulp_size=4096, baseline_length=0.2)
        for center in centers:
            injected = data.copy()
            injected[center - 4 : center + 4] += 6
            tim = _series(injected)
            via = tim.find_pulses(
                method="heimdall", max_width=16, heimdall_config=config
            )
            direct = heimdall_pulse_search(
                injected, 0.001, max_width=16, config=config, dm=42.0
            )
            assert np.min(np.abs(via.values("bin") - center)) <= 2
            for column in ("bin", "snr", "width_s", *HEIMDALL_CANDIDATE_NAMES):
                np.testing.assert_array_equal(via.values(column), direct.values(column))
            np.testing.assert_array_equal(tim.data, injected)

    def test_native_common_guard_excludes_observation_edges(self) -> None:
        data = np.random.default_rng(26).normal(size=8192).astype(np.float32)
        data[0] += 40
        data[-1] += 40
        found = pulse_search(data, 0.001, method="heimdall", max_width=64, widths=[1])
        assert len(found) == 0

    def test_all_widths_cluster_one_interior_pulse(self) -> None:
        data = np.random.default_rng(27).normal(size=16384).astype(np.float32)
        data[4000:4016] += 6
        found = pulse_search(data, 0.001, method="heimdall", max_width=64)
        assert len(found) == 1
        assert abs(int(found.values("bin")[0]) - 4008) <= 1
        assert int(found.values("members")[0]) > 1
        assert int(found.values("filter_index")[0]) == 4
        assert np.isnan(found.values("period_s")[0])

    @pytest.mark.parametrize("data", [[], np.zeros(512), np.ones(512)])
    def test_empty_constant_and_csv_roundtrip(
        self, data: list | np.ndarray, tmp_path: Path
    ) -> None:
        found = heimdall_pulse_search(data, 0.001, max_width=64)
        assert len(found) == 0
        output = tmp_path / "empty.csv"
        found.to_pulse_csv(output, 0.001)
        assert output.read_text().strip().split(",") == [
            *PULSE_CANDIDATE_NAMES,
            *HEIMDALL_CANDIDATE_NAMES,
        ]
        loaded = CandidateTable.from_csv(output)
        assert len(loaded) == 0
        assert set(HEIMDALL_CANDIDATE_NAMES) <= set(loaded.table.colnames)

    def test_native_metadata_roundtrip(self, tmp_path: Path) -> None:
        giants = _giants([20, 10], [8.0, 9.0], [16, 4])
        found = heimdall_candidates(giants, 0.001, dm=42.0)
        for export in ("to_csv", "to_pulse_csv"):
            output = tmp_path / f"{export}.csv"
            if export == "to_csv":
                found.to_csv(output)
            else:
                found.to_pulse_csv(output, 0.001)
            loaded = CandidateTable.from_csv(output)
            for column in ("bin", "snr", "width_s", *HEIMDALL_CANDIDATE_NAMES):
                np.testing.assert_array_equal(
                    loaded.values(column), found.values(column)
                )
        stream = StringIO()
        found.to_pulse_csv(stream, 0.001)
        assert "period" not in stream.getvalue().splitlines()[0]

    def test_empty_table_operations_preserve_metadata(self, tmp_path: Path) -> None:
        empty = heimdall_candidates(HeimdallGiants.empty(), 0.001)
        operations = (
            empty.sort_by_snr(),
            empty.cluster(),
            CandidateTable.concatenate([CandidateTable.empty(), empty]),
        )
        for i, result in enumerate(operations):
            assert result.table.colnames == empty.table.colnames
            for name in HEIMDALL_CANDIDATE_NAMES:
                assert result.table[name].dtype == np.dtype("int64")
            path = tmp_path / f"empty_{i}.csv"
            result.to_pulse_csv(path, 0.001)
            restored = CandidateTable.from_csv(path)
            assert set(HEIMDALL_CANDIDATE_NAMES) <= set(restored.table.colnames)
        assert empty.sort_by_snr().table is not empty.table

    def test_mixed_table_preserves_optional_columns(self, tmp_path: Path) -> None:
        native = heimdall_candidates(_giants([10], [8.0], [4]), 0.001)
        periodic = CandidateTable.from_rows(
            [{"period_s": 1.0, "snr": 7.0, "kind": "fft"}]
        )
        combined = native + periodic
        path = tmp_path / "mixed.csv"
        combined.to_csv(path)
        restored = CandidateTable.from_csv(path)
        assert restored.values("kind").tolist() == ["pulse", "fft"]
        assert int(restored.table["members"][0]) == 1
        assert restored.table["members"].mask.tolist() == [False, True]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gulp_size": 1},
            {"baseline_length": 0},
            {"baseline_length": np.nan},
            {"min_tscrunch_width": 3},
            {"min_tscrunch_width": 1},
            {"cand_sep_time": -1},
            {"cand_sep_filter": -1},
            {"boxcar_renorm": "yes"},
        ],
    )
    def test_invalid_config(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            HeimdallConfig(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"engine": "fft"},
            {"deadtime": 0},
            {"block_size": 1024},
            {"chunk_size": 4096},
            {"detrend": "median"},
            {"mask_bad_blocks": True},
            {"widths": [3]},
            {"heimdall_config": HeimdallConfig(gulp_size=64)},
            {"heimdall_config": HeimdallConfig(baseline_length=0.0001)},
        ],
    )
    def test_invalid_method_controls(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            pulse_search(np.ones(512), 0.001, method="heimdall", max_width=64, **kwargs)
        with pytest.raises(ValueError, match="only valid"):
            pulse_search(np.ones(512), 0.001, heimdall_config=HeimdallConfig())

    def test_cli_controls_and_event_columns(self, tmp_path: Path) -> None:
        data = np.random.default_rng(28).normal(size=16384).astype(np.float32)
        data[4000:4016] += 6
        tim = _series(data)
        path = tmp_path / "in.tim"
        tim.to_tim(str(path))
        result = CliRunner().invoke(
            main,
            [
                "pulses",
                str(path),
                "--method",
                "heimdall",
                "--max-width",
                "64",
                "--gulp-size",
                "4096",
                "--baseline-length",
                ".2",
                "--min-tscrunch-width",
                "32",
                "--cand-sep-time",
                "2",
                "--cand-sep-filter",
                "2",
                "--boxcar-renorm",
                "-o",
                "-",
            ],
        )
        assert result.exit_code == 0, result.output
        outfile = tmp_path / "out.csv"
        outfile.write_text(result.output)
        actual = CandidateTable.from_csv(outfile)
        expected = tim.find_pulses(
            method="heimdall",
            max_width=64,
            heimdall_config=HeimdallConfig(
                gulp_size=4096,
                baseline_length=0.2,
                min_tscrunch_width=32,
                cand_sep_time=2,
                cand_sep_filter=2,
                boxcar_renorm=True,
            ),
        )
        assert len(actual) > 0
        assert result.output.splitlines()[0].split(",") == [
            *PULSE_CANDIDATE_NAMES,
            *HEIMDALL_CANDIDATE_NAMES,
            "bin_origin",
        ]
        assert np.all(actual.values("bin_origin") == "center")
        np.testing.assert_array_equal(actual.values("bin"), expected.values("bin"))
        np.testing.assert_array_equal(
            actual.values("members"), expected.values("members")
        )
        rejected = CliRunner().invoke(
            main, ["pulses", str(path), "--gulp-size", "4096", "-o", "-"]
        )
        assert rejected.exit_code != 0
        assert "Heimdall options require" in rejected.output

    def test_binary_width_style(self) -> None:
        np.testing.assert_array_equal(
            make_width_bank(100, style="powers_of_two", spacing_factor=1.5),
            [1, 2, 4, 8, 16, 32, 64],
        )
