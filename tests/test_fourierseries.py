import numpy as np
from pathlib import Path

from sigpyproc.header import Header
from sigpyproc.fourierseries import FourierSeries


class TestFourierSeries(object):
    def test_fourierseries(self, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        assert fs.header.nbits == 32

    def test_to_file(self, fourier_data, tim_header, tmpfile):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        outfile = fs.to_file(tmpfile)
        assert Path(outfile).is_file()

    def test_to_fftfile(self, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        fftfile, inffile = fs.to_fftfile(basename="temp_test")
        fftfile_path = Path(fftfile)
        inffile_path = Path(inffile)
        assert fftfile_path.is_file()
        assert inffile_path.is_file()
        fftfile_path.unlink()
        inffile_path.unlink()

    def test_read_fft(self, fftfile, fftfile_mean):
        fs = FourierSeries.read_fft(fftfile)
        np.testing.assert_allclose(fs.mean(), fftfile_mean, atol=1)
