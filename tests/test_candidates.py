from pathlib import Path

import numpy as np
import pytest
from astropy.table import QTable

from sigpyproc.core.candidates import CANDIDATE_NAMES, CandidateTable


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
