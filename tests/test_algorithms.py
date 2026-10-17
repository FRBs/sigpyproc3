from pathlib import Path

import numpy as np
import pytest

from sigpyproc.algorithms import period
from sigpyproc.core.dispersion import DMPlan
from sigpyproc.header import Header
from sigpyproc.readers import FilReader
from sigpyproc.timeseries import TimeSeries

TSAMP = 1e-3
NSAMP = 2**16


def _series(data: np.ndarray, tsamp: float = TSAMP) -> TimeSeries:
    header = Header(
        filename="robust.tim",
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=tsamp,
        tstart=50000.0,
        nsamples=data.size,
    )
    return TimeSeries(data.astype(np.float32), header)


def _pulse_train(period: float, duty: float, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(NSAMP) * TSAMP
    phase = np.mod(t, period) / period
    data = rng.normal(0.0, 1.0, NSAMP)
    data[phase < duty] += amp
    return data


class TestPulseRobustness:
    def test_recovers_injected_boxcars(self) -> None:
        rng = np.random.default_rng(11)
        data = rng.normal(0.0, 1.0, NSAMP)
        starts = [5000, 20000, 40000]
        for start in starts:
            data[start : start + 8] += 3.0
        cands = _series(data).find_pulses(snr_min=6.0, max_width=64, deadtime=16)
        bins = cands.values("bin")
        for start in starts:
            assert np.min(np.abs(bins - start)) <= 8

    def test_boxcar_snr_matches_theory(self) -> None:
        rng = np.random.default_rng(12)
        data = rng.normal(0.0, 1.0, NSAMP)
        data[20000:20008] += 3.0
        cands = _series(data).find_pulses(snr_min=6.0, max_width=64, deadtime=16)
        near = np.abs(cands.values("bin") - 20000) <= 8
        assert near.any()
        # Amplitude 3 over 8 unit-variance samples gives SNR 3 * 8 / sqrt(8).
        assert cands.values("snr")[near].max() == pytest.approx(8.5, abs=1.5)

    def test_pure_noise_has_no_pulses(self) -> None:
        rng = np.random.default_rng(13)
        cands = _series(rng.normal(0.0, 1.0, NSAMP)).find_pulses(snr_min=6.0)
        assert len(cands) == 0


class TestFFTRobustness:
    @pytest.mark.parametrize("nsamples", [0, 1, 2])
    def test_spectrum_without_interior_bins(self, nsamples: int) -> None:
        candidates = period.pick_spectral_peaks(
            np.zeros(nsamples),
            1.0,
            npeaks=10,
            snr_min=6.0,
            pmax=1.0,
            dm=0.0,
            accel=0.0,
            nharm=1,
        )
        assert len(candidates) == 0

    @pytest.mark.parametrize("method", ["simple", "greedy"])
    def test_fft_search_recovers_duty_cycle_pulse_train(self, method: str) -> None:
        period = 0.05
        data = _pulse_train(period, duty=0.05, amp=4.0, seed=3)
        cands = _series(data).search(nfolds=4, pmax=1.0, snr_min=6.0, method=method)
        best = cands.cluster(snr_min=8.0).values("period_s")[0]
        assert abs(best - period) <= 2 * TSAMP

    def test_fft_search_recovers_sinusoid(self) -> None:
        period = 0.05
        t = np.arange(NSAMP) * TSAMP
        rng = np.random.default_rng(14)
        data = rng.normal(0.0, 1.0, NSAMP) + 3.0 * np.sin(2 * np.pi * t / period)
        cands = _series(data).search(nfolds=2, pmax=1.0, snr_min=6.0)
        best = cands.cluster(snr_min=8.0).values("period_s")[0]
        tobs = NSAMP * TSAMP
        assert abs(best - period) <= 2.0 * period**2 / tobs


class TestDedispersionConsistency:
    def test_plan_trials_match_dedisperse(
        self, filfile_8bit_1: str, tmpdir: str
    ) -> None:
        src = FilReader(filfile_8bit_1)
        band = src.extract_bands(
            chanstart=0,
            nchans=32,
            chanpersub=32,
            outfile_base=str(Path(tmpdir) / "band"),
        )[0]
        fil = FilReader(band)
        full = DMPlan.from_header(fil.header, dm_max=300.0, ti=40e-6)
        plan = DMPlan(dms=full.dms[:3])
        paths = fil.dedisperse_plan(
            plan,
            gulp=1024,
            outfile_base=str(Path(tmpdir) / "plan"),
        )
        for idx in (1, 2):
            trial = TimeSeries.from_tim(paths[idx]).data
            ref = fil.dedisperse(float(plan.dms[idx]), "fch1").data
            np.testing.assert_array_equal(trial, ref[: trial.size])
