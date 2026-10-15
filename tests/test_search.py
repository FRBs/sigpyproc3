import numpy as np
import pytest

from sigpyproc.fourierseries import FourierSeries, PowerSpectrum
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries


def _header(nsamples: int, tsamp: float = 0.001, dm: float = 0.0) -> Header:
    return Header(
        filename="search.tim",
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=tsamp,
        tstart=50000.0,
        nsamples=nsamples,
        dm=dm,
    )


class TestFindPeaks:
    def test_recovers_injected_bin(self) -> None:
        nfreq = 4096
        tsamp = 0.001
        nsamples = (nfreq - 1) * 2
        data = np.zeros(nfreq, dtype=np.float32)
        rng = np.random.default_rng(0)
        data += rng.normal(0.0, 1.0, nfreq).astype(np.float32)
        injected = 200
        data[injected] = 40.0
        spec = PowerSpectrum(data, _header(nsamples, tsamp))
        cands = spec.find_peaks(snr_min=6.0, pmax=100.0)
        assert len(cands) >= 1
        assert abs(int(cands.table["bin"][0]) - injected) <= 1
        assert cands.table["kind"][0] == "fft"
        assert int(cands.table["nharm"][0]) == 1

    def test_skips_long_periods(self) -> None:
        nfreq = 1024
        data = np.zeros(nfreq, dtype=np.float32)
        data[2] = 50.0
        spec = PowerSpectrum(data, _header(2048, tsamp=1.0))
        # tobs = 2048 s, period at bin 2 is 1024 s >> pmax
        cands = spec.find_peaks(snr_min=1.0, pmax=10.0)
        assert len(cands) == 0

    def test_invalid_args(self) -> None:
        spec = PowerSpectrum(np.ones(64, dtype=np.float32), _header(128))
        with pytest.raises(ValueError, match="npeaks"):
            spec.find_peaks(npeaks=0)
        with pytest.raises(ValueError, match="pmax"):
            spec.find_peaks(pmax=0.0)
        with pytest.raises(ValueError, match="nharm"):
            spec.find_peaks(nharm=0)


class TestTimeSeriesSearch:
    def test_recovers_injected_period(self) -> None:
        nsamp = 2**14
        tsamp = 1e-3
        period = 0.05
        rng = np.random.default_rng(1)
        data = rng.normal(0.0, 1.0, nsamp).astype(np.float32)
        t = np.arange(nsamp) * tsamp
        data += 3.0 * np.sin(2 * np.pi * t / period).astype(np.float32)
        tim = TimeSeries(data, _header(nsamp, tsamp))
        cands = tim.search(nfolds=2, pmax=1.0, snr_min=6.0)
        assert len(cands) >= 1
        best = float(cands.values("period_s")[0])
        tobs = nsamp * tsamp
        # one Fourier bin in period units is P^2 / T
        assert abs(best - period) <= 2.0 * (period**2 / tobs)

    def test_nfolds_zero_is_unfolded_only(self) -> None:
        nsamp = 2048
        rng = np.random.default_rng(2)
        tim = TimeSeries(rng.normal(0.0, 1.0, nsamp).astype(np.float32), _header(nsamp))
        cands = tim.search(nfolds=0, pmax=1.0, snr_min=100.0)
        assert len(cands) == 0

    def test_folded_period_is_fundamental(self) -> None:
        nsamp = 2**13
        tsamp = 1e-3
        period = 0.04
        rng = np.random.default_rng(11)
        data = rng.normal(0.0, 1.0, nsamp).astype(np.float32)
        t = np.arange(nsamp) * tsamp
        data += 4.0 * np.sin(2 * np.pi * t / period).astype(np.float32)
        tim = TimeSeries(data, _header(nsamp, tsamp))
        spec0 = tim.rfft().deredden().form_spec(interpolate=True)
        fold = spec0.harmonic_fold(1)[0]
        cands = fold.find_peaks(snr_min=6.0, pmax=1.0, nharm=2)
        assert len(cands) >= 1
        best = float(cands.values("period_s")[0])
        tobs = nsamp * tsamp
        assert abs(best - period) <= 2.0 * (period**2 / tobs)
        assert int(cands.table["nharm"][0]) == 2

    def test_nfolds_negative(self) -> None:
        tim = TimeSeries(np.ones(64, dtype=np.float32), _header(64))
        with pytest.raises(ValueError, match="nfolds"):
            tim.search(nfolds=-1)


class TestFindPulses:
    def test_injected_boxcar(self) -> None:
        nsamp = 4096
        tsamp = 1e-3
        rng = np.random.default_rng(3)
        data = rng.normal(0.0, 1.0, nsamp).astype(np.float32)
        width = 8
        peak = 1000
        snr = 12.0
        data[peak : peak + width] += np.float32(snr / np.sqrt(width))
        tim = TimeSeries(data, _header(nsamp, tsamp, dm=12.0))
        cands = tim.find_pulses(
            method="matched_filter", snr_min=6.0, max_width=64, deadtime=8
        )
        assert len(cands) >= 1
        assert abs(int(cands.table["bin"][0]) - peak) <= 2
        width_s = float(cands.values("width_s")[0])
        # bank is 1, 2, 4, 8, 16, … — one bank step around width 8
        assert width_s / tsamp in {4, 8, 16}
        assert cands.table["kind"][0] == "pulse"
        np.testing.assert_allclose(cands.table["dm"].value[0], 12.0)

    def test_invalid_args(self) -> None:
        tim = TimeSeries(np.ones(64, dtype=np.float32), _header(64))
        with pytest.raises(ValueError, match="max_width"):
            tim.find_pulses(max_width=0)
        with pytest.raises(ValueError, match="deadtime"):
            tim.find_pulses(deadtime=-1)


def test_resample_rejects_a_mapping_outside_the_array() -> None:
    tim = TimeSeries(np.ones(64, dtype=np.float32), _header(64, tsamp=1.0))
    with pytest.raises(ValueError, match="does not fit"):
        tim.resample(1e16)


def test_zero_spectrum_deredden_stays_finite() -> None:
    series = FourierSeries(np.zeros(64, dtype=np.complex64), _header(128))
    whitened = series.deredden()
    assert np.all(np.isfinite(whitened.data))
    with pytest.raises(ValueError, match="window"):
        series.deredden(start_width=0)
