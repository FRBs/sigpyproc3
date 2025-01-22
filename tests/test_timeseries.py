from pathlib import Path

import numpy as np
import pytest

from sigpyproc.foldedcube import FoldedData
from sigpyproc.fourierseries import FourierSeries
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries


class TestTimeSeries:
    def test_init_fail(self, tim_data: np.ndarray, tim_header: dict) -> None:
        with pytest.raises(TypeError):
            TimeSeries(tim_data, tim_data)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            TimeSeries(tim_data[0], Header(**tim_header))
        with pytest.raises(ValueError):
            TimeSeries(tim_data[:-1], Header(**tim_header))
        with pytest.raises(ValueError):
            TimeSeries([], Header(**tim_header))

    def test_timeseries(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        np.testing.assert_equal(tim.header.nbits, 32)
        np.testing.assert_almost_equal(tim.data.mean(), 128, decimal=0)

    def test_fold(self, tim_data: np.ndarray, tim_header: dict) -> None:
        period = 1
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_folded = tim.fold(period=period, nints=16)
        assert isinstance(tim_folded, FoldedData)

    def test_rfft(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_rfft = tim.rfft()
        assert isinstance(tim_rfft, FourierSeries)

    def test_deredden(self, tim_data: np.ndarray, tim_header: dict) -> None:
        window = 101
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_deredden = tim.deredden(window=window)
        np.testing.assert_equal(tim_deredden.data.size, tim.data.size)
        np.testing.assert_allclose(tim_deredden.data.mean(), 0, atol=0.1)
        with pytest.raises(ValueError):
            tim.deredden(window=-5)

    def test_apply_boxcar(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_boxcar = tim.apply_boxcar(width=5)
        assert isinstance(tim_boxcar, TimeSeries)
        np.testing.assert_equal(tim_boxcar.data.size, tim.data.size)
        with pytest.raises(ValueError):
            tim.apply_boxcar(width=0)

    def test_downsample(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tfactor = 16
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_decimated = tim.downsample(factor=tfactor)
        assert isinstance(tim_decimated, TimeSeries)
        np.testing.assert_equal(tim_decimated.data.size, tim.data.size // tfactor)
        np.testing.assert_allclose(
            tim.data[:tfactor].mean(),
            tim_decimated.data[0],
            atol=0.01,
        )
        tim_decimated_1 = tim.downsample(factor=1)
        np.testing.assert_equal(tim_decimated_1.data, tim.data)
        with pytest.raises(ValueError):
            tim.downsample(factor=0)
        with pytest.raises(ValueError):
            tim.downsample(factor=1.5) # type: ignore[arg-type]

    def test_pad(self, tim_data: np.ndarray, tim_header: dict) -> None:
        npad = 100
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_padded = tim.pad(npad=npad)
        assert isinstance(tim_padded, TimeSeries)
        np.testing.assert_equal(tim_padded.data.size, tim.data.size + npad)

    def test_resample(self, tim_data: np.ndarray, tim_header: dict) -> None:
        accel = 1
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_resampled = tim.resample(accel=accel)
        assert isinstance(tim_resampled, TimeSeries)
        np.testing.assert_equal(tim_resampled.header.accel, accel)

    def test_correlate(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_corr = tim.correlate(tim_data)
        assert isinstance(tim_corr, TimeSeries)

    def test_to_dat(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        datfile = tim.to_dat()
        datfile_path = Path(datfile)
        inffile_path = datfile_path.with_suffix(".inf")
        assert datfile_path.is_file()
        assert inffile_path.is_file()
        datfile_path.unlink()
        inffile_path.unlink()

    def test_to_tim(
        self,
        tim_data: np.ndarray,
        tim_header: dict,
        tmpfile: str,
    ) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        outfile = tim.to_tim(tmpfile)
        assert Path(outfile).is_file()
        outfile = tim.to_tim()
        outpath = Path(outfile)
        assert outpath.is_file()
        outpath.unlink()


    def test_from_dat(
        self,
        datfile: str,
        datfile_mean: float,
        datfile_std: float,
    ) -> None:
        tim = TimeSeries.from_dat(datfile)
        np.testing.assert_allclose(tim.data.mean(), datfile_mean, atol=1)
        np.testing.assert_allclose(tim.data.std(), datfile_std, atol=1)
        with pytest.raises(FileNotFoundError):
            TimeSeries.from_dat("nonexistent.dat")

    def test_from_tim(
        self,
        timfile: str,
        timfile_mean: float,
        timfile_std: float,
    ) -> None:
        tim = TimeSeries.from_tim(timfile)
        np.testing.assert_allclose(tim.data.mean(), timfile_mean, atol=1)
        np.testing.assert_allclose(tim.data.std(), timfile_std, atol=1)
