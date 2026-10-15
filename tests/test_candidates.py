from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from sigpyproc.core.candidates import (
    CANDIDATE_NAMES,
    PULSE_CANDIDATE_NAMES,
    CandidateTable,
)


def _row(
    period: float,
    snr: float,
    *,
    nharm: int = 1,
    kind: str = "fft",
    dm: float = 0.0,
) -> dict[str, object]:
    return {
        "period_s": period,
        "snr": snr,
        "nharm": nharm,
        "kind": kind,
        "dm": dm,
    }


class TestCandidateTable:
    def test_empty(self) -> None:
        cands = CandidateTable.empty()
        assert len(cands) == 0
        assert cands.table.colnames == list(CANDIDATE_NAMES)
        assert "CandidateTable" in repr(cands)

    def test_from_rows_infers_freq(self) -> None:
        cands = CandidateTable.from_rows([_row(0.5, 12.0)])
        assert len(cands) == 1
        np.testing.assert_allclose(cands.table["freq_hz"].value, 2.0)
        np.testing.assert_allclose(cands.table["period_s"].value, 0.5)

    def test_from_rows_infers_period(self) -> None:
        cands = CandidateTable.from_rows([{"freq_hz": 10.0, "snr": 9.0}])
        np.testing.assert_allclose(cands.table["period_s"].value, 0.1)

    def test_from_rows_empty(self) -> None:
        assert len(CandidateTable.from_rows([])) == 0

    def test_getitem_and_add(self) -> None:
        cands = CandidateTable.from_rows([_row(1.0, 10.0), _row(0.5, 8.0)])
        assert len(cands[0]) == 1
        assert len(cands[:1]) == 1
        mask = np.array([False, True])
        assert len(cands[mask]) == 1
        stacked = cands + CandidateTable.from_rows([_row(2.0, 7.0)])
        assert len(stacked) == 3

    def test_csv_roundtrip(self, tmp_path: Path) -> None:
        cands = CandidateTable.from_rows(
            [_row(1.234, 15.0, dm=12.5), {"kind": "pulse", "snr": 9.0, "bin": 8}],
        )
        path = tmp_path / "cands.csv"
        cands.to_csv(path)
        loaded = CandidateTable.from_csv(path)
        assert len(loaded) == 2
        np.testing.assert_allclose(loaded.table["period_s"].value[0], 1.234)
        np.testing.assert_allclose(loaded.table["dm"].value[0], 12.5)
        assert loaded.table["kind"][1] == "pulse"
        assert int(loaded.table["bin"][1]) == 8

    def test_from_csv_missing_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("snr,period_s\n10,1.0\n")
        with pytest.raises(ValueError, match="missing columns"):
            CandidateTable.from_csv(path)

    def test_missing_columns_on_init(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            CandidateTable(QTable({"snr": [1.0]}))

    def test_cluster_drops_harmonic(self) -> None:
        cands = CandidateTable.from_rows(
            [
                _row(1.0, 20.0),
                _row(0.5, 12.0),
                _row(2.0, 9.0),
                _row(0.3333, 8.5),
                _row(1.37, 18.0),
            ],
        )
        clustered = cands.cluster(snr_min=8.0)
        periods = np.sort(clustered.values("period_s"))
        np.testing.assert_allclose(periods, [1.0, 1.37], rtol=1e-3)
        assert len(clustered) == 2

    def test_cluster_snr_min(self) -> None:
        cands = CandidateTable.from_rows([_row(1.0, 20.0), _row(0.7, 5.0)])
        clustered = cands.cluster(snr_min=8.0)
        assert len(clustered) == 1
        np.testing.assert_allclose(clustered.table["period_s"].value[0], 1.0)

    def test_cluster_keeps_pulses(self) -> None:
        cands = CandidateTable.from_rows(
            [
                _row(1.0, 20.0),
                {"kind": "pulse", "snr": 11.0, "bin": 10},
                {"kind": "pulse", "snr": 10.0, "bin": 50},
            ],
        )
        clustered = cands.cluster(snr_min=8.0)
        assert len(clustered) == 3

    def test_cluster_invalid(self) -> None:
        cands = CandidateTable.from_rows([_row(1.0, 10.0)])
        with pytest.raises(ValueError, match="period_tol"):
            cands.cluster(period_tol=0.0)
        with pytest.raises(ValueError, match="harm_max"):
            cands.cluster(harm_max=0)

    def test_values_strips_units(self) -> None:
        cands = CandidateTable.from_rows([_row(0.5, 12.0)])
        period = cands.values("period_s")
        assert period.dtype == np.float64
        np.testing.assert_allclose(period, [0.5])

    def test_sort_by_snr(self) -> None:
        cands = CandidateTable.from_rows([_row(1.0, 5.0), _row(2.0, 15.0)])
        ordered = cands.sort_by_snr()
        np.testing.assert_allclose(ordered.table["snr"][0], 15.0)


class TestPulseCSV:
    @pytest.mark.parametrize("empty", [False, True])
    def test_file_and_stream_roundtrip(self, tmp_path: Path, empty: bool) -> None:
        cands = (
            CandidateTable.empty()
            if empty
            else CandidateTable.from_rows(
                [
                    {
                        "kind": "pulse",
                        "snr": 12.5,
                        "dm": 42.0,
                        "bin": 10,
                        "width_s": 0.003,
                    },
                    {
                        "kind": "pulse",
                        "snr": 8.0,
                        "dm": 42.0,
                        "bin": 30,
                        "width_s": 0.008,
                    },
                ]
            )
        )
        original = cands.table.copy()
        stream = StringIO()
        cands.to_pulse_csv(stream, 0.001)
        lines = stream.getvalue().splitlines()
        assert lines[0].split(",") == list(PULSE_CANDIDATE_NAMES)
        assert "period" not in lines[0]
        assert "freq" not in lines[0]
        assert "nharm" not in lines[0]
        assert "accel" not in lines[0]
        path = tmp_path / "pulses.csv"
        cands.to_pulse_csv(path, 0.001)
        assert path.read_text() == stream.getvalue()
        loaded = CandidateTable.from_csv(path)
        assert len(loaded) == len(cands)
        for col in ("snr", "dm", "bin", "width_s"):
            np.testing.assert_array_equal(loaded.values(col), cands.values(col))
        assert np.all(loaded.values("kind") == "pulse")
        assert np.all(np.isnan(loaded.values("period_s")))
        assert cands.table.colnames == original.colnames
        if not empty:
            assert lines[1].split(",")[2] == "0.01"
            assert lines[1].split(",")[4] == "3"
            assert lines[2].split(",")[4] == "8"

    @pytest.mark.parametrize("tsamp", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_sampling_interval(self, tsamp: float) -> None:
        with pytest.raises(ValueError, match="tsamp"):
            CandidateTable.empty().to_pulse_csv(StringIO(), tsamp)

    def test_reject_periodic_and_invalid_widths(self) -> None:
        with pytest.raises(ValueError, match="only pulse"):
            CandidateTable.from_rows([_row(1.0, 8.0)]).to_pulse_csv(StringIO(), 0.001)
        for width in (np.nan, 0.0, 0.0015):
            cands = CandidateTable.from_rows([{"kind": "pulse", "width_s": width}])
            with pytest.raises(ValueError, match="whole numbers"):
                cands.to_pulse_csv(StringIO(), 0.001)

    def test_quantities_are_converted_before_export_and_clustering(
        self, tmp_path: Path
    ) -> None:
        loud = CandidateTable.from_rows(
            [
                {
                    "period_s": 1000 * u.ms,
                    "freq_hz": 1 * u.Hz,
                    "snr": 20.0,
                    "dm": (1 * u.pc / u.m**3),
                    "width_s": 2 * u.ms,
                    "accel": 100 * u.cm / u.s**2,
                    "bin_origin": "center",
                },
            ],
        )
        quiet = CandidateTable.from_rows(
            [{"period_s": 1.0 * u.s, "snr": 9.0, "bin_origin": "start"}],
        )
        np.testing.assert_allclose(loud.values("period_s"), 1.0)
        np.testing.assert_allclose(loud.values("dm"), 1e-6)
        np.testing.assert_allclose(loud.values("width_s"), 0.002)
        np.testing.assert_allclose(loud.values("accel"), 1.0)
        assert loud.table["bin_origin"][0] == "center"
        clustered = (loud + quiet).cluster(snr_min=1.0)
        assert len(clustered) == 1
        np.testing.assert_allclose(clustered.values("snr"), 20.0)

        path = tmp_path / "units.csv"
        loud.to_csv(path)
        loaded = CandidateTable.from_csv(path)
        np.testing.assert_allclose(loaded.values("period_s"), 1.0)
        np.testing.assert_allclose(loaded.values("accel"), 1.0)
        assert loaded.table["bin_origin"][0] == "center"

        with pytest.raises(ValueError, match="period_s"):
            CandidateTable.from_rows([{"period_s": 1.0 * u.m, "snr": 1.0}])
        with pytest.raises(ValueError, match="scalar"):
            CandidateTable.from_rows([{"period_s": [1.0, 2.0] * u.s, "snr": 1.0}])
