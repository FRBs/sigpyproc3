import numpy as np
from pathlib import Path

from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries


class TestTimeSeries(object):
    def test_timeseries(self, tim_data, tim_header):
        tim = TimeSeries(tim_data, Header(**tim_header))
        assert tim.header.nbits == 32
        np.testing.assert_allclose(tim.mean(), 128, atol=0.1)

    def test_running_mean(self, tim_data, tim_header):
        window = 101
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_filtered_mean = tim.running_mean(window=window)
        assert tim_filtered_mean.size == tim.size
        np.testing.assert_allclose(tim_filtered_mean.mean(), tim.mean(), atol=0.1)

    def test_running_median(self, tim_data, tim_header):
        window = 101
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_filtered_median = tim.running_median(window=window)
        assert tim_filtered_median.size == tim.size
        np.testing.assert_allclose(tim_filtered_median.mean(), tim.mean(), atol=0.2)

    def test_downsample(self, tim_data, tim_header):
        tfactor = 16
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_decimated = tim.downsample(factor=tfactor)
        assert tim_decimated.size == tim.size // tfactor
        np.testing.assert_allclose(tim[:tfactor].mean(), tim_decimated[0], atol=0.01)

    def test_pad(self, tim_data, tim_header):
        npad = 100
        tim = TimeSeries(tim_data, Header(**tim_header))
        tim_padded = tim.pad(npad=npad)
        assert tim_padded.size == tim.size + npad

    def test_to_dat(self, tim_data, tim_header):
        tim = TimeSeries(tim_data, Header(**tim_header))
        datfile, inffile = tim.to_dat(basename="temp_test")
        datfile_path = Path(datfile)
        inffile_path = Path(inffile)
        assert datfile_path.is_file()
        assert inffile_path.is_file()
        datfile_path.unlink()
        inffile_path.unlink()

    def test_to_file(self, tim_data, tim_header, tmpfile):
        tim = TimeSeries(tim_data, Header(**tim_header))
        outfile = tim.to_file(tmpfile)
        assert Path(outfile).is_file()

    def test_read_dat(self, datfile, datfile_mean, datfile_std):
        tim = TimeSeries.read_dat(datfile)
        np.testing.assert_allclose(tim.mean(), datfile_mean, atol=1)
        np.testing.assert_allclose(tim.std(), datfile_std, atol=1)

    def test_read_tim(self, timfile, timfile_mean, timfile_std):
        tim = TimeSeries.read_tim(timfile)
        np.testing.assert_allclose(tim.mean(), timfile_mean, atol=1)
        np.testing.assert_allclose(tim.std(), timfile_std, atol=1)
