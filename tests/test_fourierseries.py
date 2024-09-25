from pathlib import Path

import numpy as np
import pytest

from sigpyproc.foldedcube import Profile
from sigpyproc.fourierseries import FourierSeries, PowerSpectrum
from sigpyproc.header import Header


class TestFourierSeries:
    def test_init_fail(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        with pytest.raises(TypeError):
            FourierSeries(fourier_data, fourier_data)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            FourierSeries(fourier_data[0], Header(**tim_header))
        with pytest.raises(ValueError):
            FourierSeries([], Header(**tim_header))

    def test_fourierseries(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        assert isinstance(fs, FourierSeries)
        assert isinstance(fs.header, Header)
        np.testing.assert_equal(fs.header.nbits, 32)
        np.testing.assert_equal(fs.data.dtype, np.complex64)

    def test_ifft(
        self,
        tim_data: np.ndarray,
        fourier_data: np.ndarray,
        tim_header: dict,
    ) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        tim_ar = fs.ifft()
        np.testing.assert_allclose(tim_ar.data, tim_data, atol=0.01)

    def test_form_spec(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        assert isinstance(spec, PowerSpectrum)
        np.testing.assert_equal(spec.data.size, fs.data.size)
        spec_interp = fs.form_spec(interpolate=True)
        np.testing.assert_equal(spec_interp.data.size, fs.data.size)

    def test_deredden(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec_red = fs.deredden()
        assert isinstance(spec_red, FourierSeries)
        np.testing.assert_equal(spec_red.data.size, fs.data.size)

    def test_recon_prof(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        freq = 1.2
        fs = FourierSeries(fourier_data, Header(**tim_header))
        profile = fs.recon_prof(freq)
        assert isinstance(profile, Profile)

    def test_multiply(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        fs_mult = fs.multiply(fourier_data)
        assert isinstance(fs_mult, FourierSeries)
        np.testing.assert_equal(fs_mult.data.size, fs.data.size)

    def test_to_spec(
        self,
        fourier_data: np.ndarray,
        tim_header: dict,
        tmpfile: str,
    ) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        outfile = fs.to_spec(tmpfile)
        assert Path(outfile).is_file()
        outfile = fs.to_spec()
        outpath = Path(outfile)
        assert outpath.is_file()
        outpath.unlink()

    def test_to_fft(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        fftfile = fs.to_fft()
        fftfile_path = Path(fftfile)
        inffile_path = fftfile_path.with_suffix(".inf")
        assert fftfile_path.is_file()
        assert inffile_path.is_file()
        fftfile_path.unlink()
        inffile_path.unlink()

    def test_from_fft(self, fftfile: str) -> None:
        fs = FourierSeries.from_fft(fftfile)
        assert isinstance(fs, FourierSeries)
        np.testing.assert_equal(fs.data.dtype, np.complex64)
        with pytest.raises(FileNotFoundError):
            FourierSeries.from_fft("nonexistent.fft")

    def test_from_spec(
        self,
        fourier_data: np.ndarray,
        tim_header: dict,
        tmpfile: str,
    ) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        outfile = fs.to_spec(tmpfile)
        fs_read = FourierSeries.from_spec(outfile)
        assert isinstance(fs_read, FourierSeries)
        np.testing.assert_equal(fs_read.data.dtype, np.complex64)
        np.testing.assert_equal(fs_read.data.size, fs.data.size)


class TestPowerSpectrum:
    def test_powerspectrum(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        spec = PowerSpectrum(np.abs(fourier_data), Header(**tim_header))
        assert isinstance(spec, PowerSpectrum)
        assert isinstance(spec.header, Header)
        np.testing.assert_equal(spec.header.nbits, 32)
        np.testing.assert_equal(spec.data.dtype, np.float32)

    def test_bin2freq(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        np.testing.assert_equal(spec.bin2freq(0), 0)
        np.testing.assert_equal(spec.bin2freq(1), 1 / spec.header.tobs)

    def test_bin2period(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        np.testing.assert_equal(spec.bin2period(1), spec.header.tobs)
        with np.testing.assert_raises(ZeroDivisionError):
            spec.bin2period(0)

    def test_freq2bin(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        np.testing.assert_equal(spec.freq2bin(0), 0)
        np.testing.assert_equal(spec.freq2bin(1 / spec.header.tobs), 1)

    def test_period2bin(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        np.testing.assert_equal(spec.period2bin(spec.header.tobs), 1)

    def test_harmonic_fold(self, fourier_data: np.ndarray, tim_header: dict) -> None:
        nfolds = 2
        fs = FourierSeries(fourier_data, Header(**tim_header))
        spec = fs.form_spec()
        folds = spec.harmonic_fold(nfolds)
        assert isinstance(folds[0], PowerSpectrum)
        np.testing.assert_equal(len(folds), nfolds)
