import os
import numpy as np
from sigpyproc.FourierSeries import FourierSeries
from sigpyproc.Header import Header


class TestFourierSeries:
    def test_fourierSeries(self, fourier_data, tim_header):
        myFS = FourierSeries(fourier_data, Header(tim_header))
        assert myFS.header.nbits == 32
        assert myFS.header.source_name == "test"
        assert myFS.shape == (10002,)
        np.testing.assert_allclose(np.mean(myFS), 135.61371, atol=0.1)

    def test_iFFT(self, fourier_data, tim_header, tim_data):
        myFS  = FourierSeries(fourier_data, Header(tim_header))
        myTim = myFS.iFFT()
        np.testing.assert_allclose(myTim, tim_data, atol=0.1)

    def test_toFFTFile(self, fourier_data, tim_header):
        myFS = FourierSeries(fourier_data, Header(tim_header))
        fftfile, inffile = myFS.toFFTFile(basename="temp_test")
        assert os.path.isfile(fftfile)
        assert os.path.isfile(inffile)
        os.remove(fftfile)
        os.remove(inffile)

    def test_toFile(self, fourier_data, tim_header):
        myFS = FourierSeries(fourier_data, Header(tim_header))
        outfile = myFS.toFile()
        assert os.path.isfile(outfile)
        os.remove(outfile)

    def test_readFFT(self, fourier_data, tim_header):
        myFS = FourierSeries(fourier_data, Header(tim_header))
        fftfile, inffile = myFS.toFFTFile(basename="temp_test")
        mynewFS = FourierSeries.readFFT(filename=fftfile)
        assert mynewFS.header.nbits == 32
        assert mynewFS.header.source_name == "test"
        np.testing.assert_allclose(np.mean(mynewFS), 135.61371, atol=0.1)
        os.remove(fftfile)
        os.remove(inffile)
