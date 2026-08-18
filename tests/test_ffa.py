import numpy as np
import pytest

from sigpyproc.core import kernels
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries


def _header(nsamples: int, tsamp: float = 1e-3) -> Header:
    return Header(
        filename="ffa.tim",
        data_type="time series",
        nchans=1,
        foff=1.0,
        fch1=1400.0,
        nbits=32,
        tsamp=tsamp,
        tstart=50000.0,
        nsamples=nsamples,
        dm=0.0,
    )


class TestFfaStage:
    def test_aligned_rows_sum(self) -> None:
        row = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        block = np.vstack([row, row])
        profiles = kernels.ffa_stage(block)
        assert profiles.shape[0] == 2
        np.testing.assert_allclose(profiles[0], [2.0, 0.0, 0.0, 0.0])


class TestTimeSeriesFFA:
    def test_recovers_rectangular_pulse(self) -> None:
        nsamp = 2**16
        tsamp = 1e-3
        period = 1.234
        duty = 0.04
        peak_snr = 15.0
        rng = np.random.default_rng(4)
        data = rng.normal(0.0, 1.0, nsamp).astype(np.float32)
        t = np.arange(nsamp) * tsamp
        phase = np.mod(t, period) / period
        on = phase < duty
        data[on] += np.float32(peak_snr)
        tim = TimeSeries(data, _header(nsamp, tsamp))
        cands = tim.ffa(1.0, 1.5, snr_min=8.0, npeaks=20)
        assert len(cands) >= 1
        best = float(cands.values("period_s")[0])
        n_rows = nsamp // round(period / tsamp)
        n_pad = 1 << (n_rows - 1).bit_length()
        period_step = tsamp / n_pad
        assert abs(best - period) <= period_step
        assert float(cands.table["snr"][0]) >= 8.0
        assert cands.table["kind"][0] == "ffa"

    def test_search_width_and_zero_pad(self) -> None:
        nsamp = 2048
        tsamp = 1e-3
        rng = np.random.default_rng(9)
        tim = TimeSeries(
            rng.normal(0.0, 1.0, nsamp).astype(np.float32), _header(nsamp, tsamp)
        )
        cands = tim.ffa(0.02, 0.04, search_width=True, zero_pad=True, snr_min=50.0)
        assert all(kind == "ffa" for kind in cands.table["kind"])

    def test_search_width_exclusive_with_rebin(self) -> None:
        tim = TimeSeries(np.ones(64, dtype=np.float32), _header(64))
        with pytest.raises(ValueError, match="mutually exclusive"):
            tim.ffa(0.01, 0.02, nrebin=2, search_width=True)

    def test_invalid_period_window(self) -> None:
        tim = TimeSeries(np.ones(64, dtype=np.float32), _header(64))
        with pytest.raises(ValueError, match="pmin"):
            tim.ffa(0.02, 0.01)
        with pytest.raises(ValueError, match="nrebin"):
            tim.ffa(0.01, 0.02, nrebin=0)
        with pytest.raises(ValueError, match="npeaks"):
            tim.ffa(0.01, 0.02, npeaks=0)
