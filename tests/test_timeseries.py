from pathlib import Path

import numpy as np

from sigpyproc.foldedcube import FoldedData
from sigpyproc.fourierseries import FourierSeries
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries


class TestTimeSeries:
    def test_timeseries(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        np.testing.assert_equal(tim.header.nbits, 32)
        np.testing.assert_almost_equal(tim.mean(), 128, decimal=0)

    def test_fold(self, tim_data: np.ndarray, tim_header: dict) -> None:
        period = 1
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_folded = tim.fold(period=period, nints=16)
        assert isinstance(tim_folded, FoldedData)

    def test_rfft(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_rfft = tim.rfft()
        assert isinstance(tim_rfft, FourierSeries)

    def test_running_mean(self, tim_data: np.ndarray, tim_header: dict) -> None:
        window = 101
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_filtered_mean = tim.running_mean(window=window)
        np.testing.assert_equal(tim_filtered_mean.size, tim.size)
        np.testing.assert_allclose(tim_filtered_mean.mean(), tim.mean(), atol=0.1)

    def test_running_median(self, tim_data: np.ndarray, tim_header: dict) -> None:
        window = 101
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_filtered_median = tim.running_median(window=window)
        np.testing.assert_equal(tim_filtered_median.size, tim.size)
        np.testing.assert_allclose(tim_filtered_median.mean(), tim.mean(), atol=0.2)

    def test_apply_boxcar(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_boxcar = tim.apply_boxcar(width=5)
        assert isinstance(tim_boxcar, TimeSeries)
        np.testing.assert_equal(tim_boxcar.size, tim.size)

    def test_downsample(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tfactor = 16
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_decimated = tim.downsample(factor=tfactor)
        assert isinstance(tim_decimated, TimeSeries)
        np.testing.assert_equal(tim_decimated.size, tim.size // tfactor)
        np.testing.assert_allclose(tim[:tfactor].mean(), tim_decimated[0], atol=0.01)

    def test_pad(self, tim_data: np.ndarray, tim_header: dict) -> None:
        npad = 100
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_padded = tim.pad(npad=npad)
        assert isinstance(tim_padded, TimeSeries)
        np.testing.assert_equal(tim_padded.size, tim.size + npad)

    def test_resample(self, tim_data: np.ndarray, tim_header: dict) -> None:
        accel = 1
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_resampled = tim.resample(accel=accel)
        assert isinstance(tim_resampled, TimeSeries)
        np.testing.assert_equal(tim_resampled.header.accel, accel)

    def test_correlate(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_corr = tim.correlate(tim)
        assert isinstance(tim_corr, TimeSeries)

    def test_to_dat(self, tim_data: np.ndarray, tim_header: dict) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        datfile, inffile = tim.to_dat(basename="temp_test")
        datfile_path = Path(datfile)
        inffile_path = Path(inffile)
        assert datfile_path.is_file()
        assert inffile_path.is_file()
        datfile_path.unlink()
        inffile_path.unlink()

    def test_to_file(
        self,
        tim_data: np.ndarray,
        tim_header: dict,
        tmpfile: str,
    ) -> None:
        tim = TimeSeries(tim_data, Header(**tim_header))
        outfile = tim.to_file(tmpfile)
        assert Path(outfile).is_file()

    def test_read_dat(
        self,
        datfile: str,
        datfile_mean: float,
        datfile_std: float,
    ) -> None:
        tim = TimeSeries.read_dat(datfile)
        np.testing.assert_allclose(tim.mean(), datfile_mean, atol=1)
        np.testing.assert_allclose(tim.std(), datfile_std, atol=1)

    def test_read_tim(
        self,
        timfile: str,
        timfile_mean: float,
        timfile_std: float,
    ) -> None:
        tim = TimeSeries.read_tim(timfile)
        np.testing.assert_allclose(tim.mean(), timfile_mean, atol=1)
        np.testing.assert_allclose(tim.std(), timfile_std, atol=1)
