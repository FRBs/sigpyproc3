import os
import numpy as np
from sigpyproc.TimeSeries import TimeSeries
from sigpyproc.Header import Header


class TestTimeSeries:
    def test_timeseries(self, tim_data, tim_header):
        myTim = TimeSeries(tim_data, Header(tim_header))
        assert myTim.header.nbits == 32
        assert myTim.header.source_name == "test"
        np.testing.assert_allclose(np.mean(myTim), 128, atol=0.1)

    def test_runningMean(self, tim_data, tim_header):
        myTim = TimeSeries(tim_data, Header(tim_header))
        mean_filter = myTim.runningMean(window=101)
        np.testing.assert_allclose(np.mean(mean_filter), 128, atol=0.1)

    def test_runningMedian(self, tim_data, tim_header):
        myTim = TimeSeries(tim_data, Header(tim_header))
        median_filter = myTim.runningMedian(window=101)
        np.testing.assert_allclose(np.mean(median_filter), 128, atol=0.1)
    """
    def test_downsample(self, tim_data, tim_header):
        myTim = TimeSeries(tim_data, Header(tim_header))
        downsampled = myTim.downsample(factor=16)
        np.testing.assert_allclose(np.mean(downsampled), 128, atol=0.1)
    """
    def test_toDat(self, tim_data, tim_header):
        myTim = TimeSeries(tim_data, Header(tim_header))
        datfile, inffile = myTim.toDat(basename="temp_test")
        assert os.path.isfile(datfile)
        assert os.path.isfile(inffile)
        os.remove(datfile)
        os.remove(inffile)

    def test_toFile(self, tim_data, tim_header):
        myTim = TimeSeries(tim_data, Header(tim_header))
        outfile = myTim.toFile()
        assert os.path.isfile(outfile)
        os.remove(outfile)

    def test_readDat(self, tim_data, tim_header):
        myTim = TimeSeries(tim_data, Header(tim_header))
        datfile, inffile = myTim.toDat(basename="temp_test")
        mynewTim = TimeSeries.readDat(filename=datfile)
        assert mynewTim.header.nbits == 32
        assert mynewTim.header.source_name == "test"
        np.testing.assert_allclose(np.mean(mynewTim), 128, atol=0.1)
        os.remove(datfile)
        os.remove(inffile)

    def test_readTim(self, tim_data, tim_header):
        myTim = TimeSeries(tim_data, Header(tim_header))
        outfile = myTim.toFile()
        mynewTim = TimeSeries.readTim(filename=outfile)
        assert mynewTim.header.nbits == 32
        assert mynewTim.header.source_name == "test"
        np.testing.assert_allclose(np.mean(mynewTim), 128, atol=0.1)
        os.remove(outfile)
