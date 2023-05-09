import numpy as np
from pathlib import Path

from sigpyproc.header import Header
from sigpyproc.fourierseries import FourierSeries, PowerSpectrum
from sigpyproc.foldedcube import Profile


class TestFourierSeries(object):
    def test_fourierseries(self, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        assert isinstance(fs, FourierSeries)
        assert isinstance(fs.header, Header)
        np.testing.assert_equal(fs.header.nbits, 32)
        np.testing.assert_equal(fs.dtype, np.complex64)

    def test_ifft(self, tim_data, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        tim_ar = fs.ifft()
        np.testing.assert_allclose(tim_ar, tim_data, atol=0.01)

    def test_form_spec(self, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        assert isinstance(spec, PowerSpectrum)
        np.testing.assert_equal(spec.size, fs.size)

    #def test_remove_rednoise(self, fourier_data, tim_header):
    #    fs = FourierSeries(fourier_data, Header(**tim_header))
    #    spec_red = fs.remove_rednoise()
    #    assert isinstance(spec_red, FourierSeries)
    #    np.testing.assert_equal(spec_red.size, fs.size)

    def test_recon_prof(self, fourier_data, tim_header):
        freq = 1.2
        fs = FourierSeries(fourier_data, Header(**tim_header))
        profile = fs.recon_prof(freq)
        assert isinstance(profile, Profile)

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

    def test_read_fft(self, fftfile):
        fs = FourierSeries.read_fft(fftfile)
        assert isinstance(fs, FourierSeries)
        np.testing.assert_equal(fs.dtype, np.complex64)

    def test_read_spec(self, fourier_data, tim_header, tmpfile):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        outfile = fs.to_file(tmpfile)
        fs_read = FourierSeries.read_spec(outfile)
        assert isinstance(fs_read, FourierSeries)
        np.testing.assert_equal(fs_read.dtype, np.complex64)
        np.testing.assert_equal(fs_read.size, fs.size)


class TestPowerSpectrum(object):
    def test_powerspectrum(self, fourier_data, tim_header):
        spec = PowerSpectrum(np.abs(fourier_data), Header(**tim_header))
        assert isinstance(spec, PowerSpectrum)
        assert isinstance(spec.header, Header)
        np.testing.assert_equal(spec.header.nbits, 32)
        np.testing.assert_equal(spec.dtype, np.float32)

    def test_bin2freq(self, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        np.testing.assert_equal(spec.bin2freq(0), 0)
        np.testing.assert_equal(spec.bin2freq(1), 1 / spec.header.tobs)

    def test_bin2period(self, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        np.testing.assert_equal(spec.bin2period(1), spec.header.tobs)
        with np.testing.assert_raises(ZeroDivisionError):
            spec.bin2period(0)

    def test_freq2bin(self, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        np.testing.assert_equal(spec.freq2bin(0), 0)
        np.testing.assert_equal(spec.freq2bin(1 / spec.header.tobs), 1)

    def test_period2bin(self, fourier_data, tim_header):
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        np.testing.assert_equal(spec.period2bin(spec.header.tobs), 1)

    def test_harmonic_fold(self, fourier_data, tim_header):
        nfolds = 2
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        folds = spec.harmonic_fold(nfolds)
        assert isinstance(folds[0], PowerSpectrum)
        np.testing.assert_equal(len(folds), nfolds)
