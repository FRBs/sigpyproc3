"""Numerical checks for every kernel in ``sigpyproc.fourier.fourier``."""

from __future__ import annotations

import numpy as np
import pytest
from numba import typed

from sigpyproc.kernels import fourier
from sigpyproc.kernels import simulation as ism_kernels


def _interp_amplitude(spectrum: np.ndarray) -> np.ndarray:
    """Interbin amplitude: larger of the bin power and the half-bin difference."""
    out = np.empty(spectrum.size, dtype=np.float64)
    prev = 0j
    for index, sample in enumerate(spectrum):
        power = sample.real**2 + sample.imag**2
        diff = 0.5 * (abs(sample - prev) ** 2)
        out[index] = np.sqrt(max(power, diff))
        prev = sample
    return out


def _lyne_reference(spec: np.ndarray, nfolds: int) -> np.ndarray:
    """SIGPROC ``sumHarms`` with the stage-to-stage pointer hand-off."""
    nfreqs = len(spec)
    out = np.zeros((nfolds, nfreqs), dtype=np.float64)
    folded = spec.astype(np.float64).copy()
    previous = 0
    for ifold in range(nfolds):
        nharm = 2 ** (ifold + 1)
        origin0 = min(nfreqs, max(1, nharm * previous - nharm // 2))
        offsets = [
            step * block // nharm
            for block in range(nharm)
            for step in range(1, nharm, 2)
        ]
        pointers = [
            (step * origin0 + nharm // 2) // nharm for step in range(1, nharm, 2)
        ]
        nptr = nharm // 2
        origin = origin0
        while origin < nfreqs - (nharm - 1):
            for neighbour in range(nharm):
                for ptr in range(nptr):
                    src = pointers[ptr] + offsets[neighbour * nptr + ptr]
                    if 0 <= src < nfreqs:
                        folded[origin + neighbour] += spec[src]
            pointers = [value + (2 * ptr + 1) for ptr, value in enumerate(pointers)]
            origin += nharm
        out[ifold] = folded
        previous = origin0
    return out


def _simple_reference(spec: np.ndarray, nfolds: int) -> np.ndarray:
    nfreqs = len(spec)
    out = np.zeros((nfolds, nfreqs), dtype=np.float64)
    for ifold in range(nfolds):
        nharm = 2 ** (ifold + 1)
        for fund in range(1, nfreqs):
            total = 0.0
            for harm in range(1, nharm + 1):
                bin_h = harm * fund
                if bin_h >= nfreqs:
                    break
                total += spec[bin_h]
            out[ifold, fund] = total
    return out


def _presto_reference(spec: np.ndarray, nfolds: int) -> np.ndarray:
    nfreqs = len(spec)
    out = np.zeros((nfolds, nfreqs), dtype=np.float64)
    for ifold in range(nfolds):
        nharm = 2 ** (ifold + 1)
        for end in range(nfreqs):
            fund = int(end / nharm + 0.5)
            if fund <= 0 or fund >= nfreqs:
                continue
            total = float(spec[fund])
            for harm in range(2, nharm + 1, 2):
                bin_h = int(harm * end / nharm + 0.5)
                if bin_h >= nfreqs:
                    break
                total += spec[bin_h]
            out[ifold, fund] = max(out[ifold, fund], total)
    return out


def _greedy_reference(spec: np.ndarray, nfolds: int) -> np.ndarray:
    nfreqs = len(spec)
    out = np.zeros((nfolds, nfreqs), dtype=np.float64)
    for ifold in range(nfolds):
        nharm = 2 ** (ifold + 1)
        for fund in range(1, nfreqs):
            drift = 0
            if fund + 1 < nfreqs and spec[fund + 1] > spec[fund]:
                drift = 1
                total = float(spec[fund + 1])
            else:
                total = float(spec[fund])
            for harm in range(2, nharm + 1):
                base = harm * fund + drift
                if base >= nfreqs:
                    break
                chosen = base
                if base + 1 < nfreqs and spec[base + 1] > spec[base]:
                    drift += 1
                    chosen = base + 1
                total += spec[chosen]
            out[ifold, fund] = total
    return out


class TestAmplitudeSpectra:
    def test_form_mspec_is_modulus(self) -> None:
        spectrum = np.array([1 + 1j, 3 + 4j, 0j, -5 + 12j], dtype=np.complex64)
        got = fourier.form_mspec(np.ascontiguousarray(spectrum))
        np.testing.assert_allclose(got, np.abs(spectrum), rtol=1e-5)
        assert got.dtype == np.float32

    def test_form_interp_mspec_takes_the_louder_neighbour(self) -> None:
        spectrum = np.array([10 + 0j, 0j, 10 + 0j], dtype=np.complex64)
        got = fourier.form_interp_mspec(np.ascontiguousarray(spectrum))
        np.testing.assert_allclose(got, _interp_amplitude(spectrum), rtol=1e-5)
        # The empty bin between two loud bins is filled from the half-bin step.
        assert got[1] > abs(spectrum[1])


class TestReddening:
    def test_running_median_normalises_power_and_keeps_phase(self) -> None:
        phase = np.linspace(0, 8 * np.pi, 256, dtype=np.float64)
        spectrum = np.exp(1j * phase).astype(np.complex64)
        whitened = fourier.fs_running_median(spectrum, 8, 16, 128)
        assert whitened.shape == spectrum.shape
        assert whitened[0] == 1 + 0j
        power = np.abs(whitened[1:]) ** 2
        assert 0.5 < np.median(power) < 2.0
        np.testing.assert_allclose(np.angle(whitened), np.angle(spectrum), atol=1e-5)

    def test_running_median_preserves_a_loud_bin(self) -> None:
        spectrum = np.ones(128, dtype=np.complex64)
        spectrum[40] = 30 + 0j
        whitened = fourier.fs_running_median(spectrum, 4, 8, 64)
        assert np.argmax(np.abs(whitened)) == 40


class TestHarmonicSums:
    def test_harmonic_counts(self) -> None:
        assert [fourier._harmonic_count(ifold) for ifold in range(4)] == [2, 4, 8, 16]

    def test_lyne_matches_sumharms(self) -> None:
        rng = np.random.default_rng(4)
        spec = rng.random(96).astype(np.float32)
        got = fourier.sum_harmonics_lyne(spec, 3)
        np.testing.assert_allclose(got, _lyne_reference(spec, 3), rtol=1e-5)
        assert not np.allclose(got[0], got[1])

    def test_lyne_tables_for_the_first_stretch(self) -> None:
        offsets, pointers = fourier._lyne_tables(2, 1)
        np.testing.assert_array_equal(offsets, [0, 0])
        np.testing.assert_array_equal(pointers, [1])

    def test_simple_is_the_integer_multiple_sum(self) -> None:
        spec = np.arange(40, dtype=np.float32)
        got = fourier.sum_harmonics_simple(spec, 3)
        np.testing.assert_allclose(got, _simple_reference(spec, 3), rtol=1e-5)
        assert got[0, 0] == 0

    def test_presto_keeps_the_largest_sum_on_a_shared_fundamental(self) -> None:
        spec = np.zeros(32, dtype=np.float32)
        spec[4] = 3
        spec[8] = 5
        spec[10] = 1
        got = fourier.sum_harmonics_presto(spec, 2)
        np.testing.assert_allclose(got, _presto_reference(spec, 2), rtol=1e-5)
        # H=2, end-bin 8: fundamental 4, sum spec[4] + spec[8] = 8.
        assert got[0, 4] == pytest.approx(8)

    def test_greedy_follows_the_louder_neighbour(self) -> None:
        spec = np.zeros(128, dtype=np.float32)
        spec[21] = 5
        spec[41] = 4
        spec[62] = 3
        spec[82] = 2
        got = fourier.sum_harmonics_greedy(spec, 2)
        np.testing.assert_allclose(got, _greedy_reference(spec, 2), rtol=1e-5)
        # Path from bin 20 steps onto 21, 41, 62, 82.
        assert got[1, 20] == pytest.approx(14)
        assert got[1, 20] > fourier.sum_harmonics_simple(spec, 2)[1, 20]

    def test_sums_stop_at_the_end_of_a_short_spectrum(self) -> None:
        spec = np.ones(8, dtype=np.float32)
        for kernel in (
            fourier.sum_harmonics_lyne,
            fourier.sum_harmonics_simple,
            fourier.sum_harmonics_presto,
            fourier.sum_harmonics_greedy,
        ):
            folded = kernel(spec, 4)
            assert folded.shape == (4, 8)
            assert np.isfinite(folded).all()

class TestLyne:
    def test_matches_sumharms_transcription(self) -> None:
        rng = np.random.default_rng(0)
        spec = rng.random(64).astype(np.float32)
        got = fourier.sum_harmonics_lyne(spec, 3)
        np.testing.assert_allclose(got, _lyne_reference(spec, 3), rtol=1e-5)

    def test_later_fold_is_not_a_copy_of_the_first(self) -> None:
        rng = np.random.default_rng(1)
        spec = rng.random(128).astype(np.float32)
        folded = fourier.sum_harmonics_lyne(spec, 2)
        assert not np.allclose(folded[0], folded[1])


class TestSimple:
    def test_integer_multiples(self) -> None:
        spec = np.arange(32, dtype=np.float32)
        got = fourier.sum_harmonics_simple(spec, 2)
        for fund in range(1, 32):
            expected = 0.0
            for harm in range(1, 5):
                bin_h = harm * fund
                if bin_h >= 32:
                    break
                expected += spec[bin_h]
            assert got[1, fund] == expected


class TestDrift:
    def test_half_bin_greedy_beats_integer_bins(self) -> None:
        spec = np.zeros(128, dtype=np.float32)
        spec[21] = 5.0
        spec[41] = 4.0
        spec[62] = 3.0
        spec[82] = 2.0
        simple = fourier.sum_harmonics_simple(spec, 2)
        greedy = fourier.sum_harmonics_greedy(spec, 2)
        presto = fourier.sum_harmonics_presto(spec, 2)
        window = slice(18, 24)
        assert greedy[1, window].max() > simple[1, window].max()
        assert presto[1, window].max() > simple[1, window].max()

    def test_short_spectrum_does_not_read_past_the_end(self) -> None:
        spec = np.ones(8, dtype=np.float32)
        for kernel in (
            fourier.sum_harmonics_lyne,
            fourier.sum_harmonics_simple,
            fourier.sum_harmonics_presto,
            fourier.sum_harmonics_greedy,
        ):
            folded = kernel(spec, 4)
            assert folded.shape == (4, 8)
            assert np.isfinite(folded).all()


class TestFft:
    @pytest.mark.parametrize("size", [1, 7, 16, 937])
    def test_good_size_is_at_least_the_request_and_stable(self, size: int) -> None:
        for real in (False, True):
            got = fourier.nb_fft_good_size(size, real=real)
            assert got >= size
            assert fourier.nb_fft_good_size(got, real=real) == got

    def test_real_roundtrip(self) -> None:
        data = np.linspace(-1.0, 2.0, 48, dtype=np.float64)
        spectrum = fourier.nb_rfft(data)
        np.testing.assert_allclose(spectrum, np.fft.rfft(data), rtol=1e-6, atol=1e-6)
        restored = fourier.nb_irfft(spectrum, data.size)
        np.testing.assert_allclose(restored, data, rtol=1e-6, atol=1e-6)

    def test_complex_roundtrip(self) -> None:
        data = np.array([1 + 2j, -3 + 0.5j, 4 - 1j, 0j], dtype=np.complex128)
        spectrum = fourier.nb_fft(data)
        np.testing.assert_allclose(spectrum, np.fft.fft(data), rtol=1e-6, atol=1e-6)
        restored = fourier.nb_ifft(spectrum)
        np.testing.assert_allclose(restored, data, rtol=1e-6, atol=1e-6)

    def test_fftconvolve_matches_direct_convolution(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        kernel = np.array([1.0, 0.0, -1.0], dtype=np.float64)
        got = fourier.fftconvolve(data, kernel)
        expected = np.convolve(data, kernel, mode="full")
        np.testing.assert_allclose(got, expected, rtol=1e-5)

    def test_fftconvolve_empty_and_rank(self) -> None:
        data = np.arange(4, dtype=np.float64)
        empty = np.zeros(0, dtype=np.float64)
        assert fourier.fftconvolve(data, empty).size == 0
        assert fourier.fftconvolve(empty, data).size == 0
        with pytest.raises(ValueError, match="1D"):
            fourier.fftconvolve(data.reshape(2, 2), data)


class TestTemplates:
    def test_circular_pad_repeats(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        padded = fourier.circular_pad_goodsize(data)
        assert padded.size == fourier.nb_fft_good_size(data.size, real=True)
        repeated = np.take(data, np.arange(padded.size) % data.size)
        np.testing.assert_array_equal(padded, repeated)

    def test_normalize_template_is_zero_mean_unit_power(self) -> None:
        template = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        normalised = fourier.normalize_template(template)
        np.testing.assert_allclose(normalised.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.sum(normalised**2), 1.0, rtol=1e-5)
        zeros = np.zeros(5, dtype=np.float64)
        np.testing.assert_array_equal(fourier.normalize_template(zeros), 0.0)

    def test_convolve_templates_peaks_on_the_matching_lag(self) -> None:
        data = np.zeros(32, dtype=np.float64)
        data[10] = 1.0
        templates = typed.List()
        templates.append(np.array([1.0], dtype=np.float64))
        refs = typed.List()
        refs.append(0)
        folded = fourier.convolve_templates(data, templates, refs)
        assert folded.shape == (1, data.size)
        assert int(np.argmax(folded[0])) == 10




class TestFourierKernels:
    def test_form_mspec(self, random_normal_1d_complex: np.ndarray) -> None:
        mspec = fourier.form_mspec(random_normal_1d_complex)
        np.testing.assert_array_almost_equal(
            mspec,
            np.abs(random_normal_1d_complex),
            decimal=5,
        )
        fourier.form_mspec.py_func(random_normal_1d_complex)

    def test_form_interp_mspec(self, random_normal_1d_complex: np.ndarray) -> None:
        mspec = fourier.form_interp_mspec(random_normal_1d_complex)
        assert mspec.shape == random_normal_1d_complex.shape
        fourier.form_interp_mspec.py_func(random_normal_1d_complex)

    def test_fs_running_median(self) -> None:
        arr = np.exp(1j * np.linspace(0, 10 * np.pi, 1000, dtype=np.float32))
        out = fourier.fs_running_median(arr, 10, 20, 500)
        pow_spec = np.abs(out) ** 2
        # Check that the power spectrum is normalized
        assert 0.5 < np.mean(pow_spec) < 2.0
        # Check that the phase is preserved
        np.testing.assert_array_almost_equal(
            np.angle(out),
            np.angle(arr),
            decimal=5,
        )
        fourier.fs_running_median.py_func(arr, 10, 20, 500)

    def test_rfft(self, random_normal_1d: np.ndarray) -> None:
        expected = np.fft.rfft(random_normal_1d)
        result = fourier.nb_rfft.py_func(random_normal_1d)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        inv_result = fourier.nb_irfft.py_func(result)
        np.testing.assert_array_almost_equal(inv_result, random_normal_1d, decimal=5)

    def test_fft(self, random_normal_1d_complex: np.ndarray) -> None:
        expected = np.fft.fft(random_normal_1d_complex)
        result = fourier.nb_fft.py_func(random_normal_1d_complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        inv_result = fourier.nb_ifft.py_func(result)
        np.testing.assert_array_almost_equal(
            inv_result,
            random_normal_1d_complex,
            decimal=5,
        )

    def test_fftconvolve(self, random_normal_1d: np.ndarray) -> None:
        kernel = np.ones(10, dtype=np.float32)
        expected = np.convolve(random_normal_1d, kernel, mode="full")
        result = fourier.fftconvolve.py_func(random_normal_1d, kernel)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        kernel_empty = np.zeros(0, dtype=np.float32)
        result = fourier.fftconvolve.py_func(random_normal_1d, kernel_empty)
        np.testing.assert_array_almost_equal(result, kernel_empty, decimal=5)

    def test_fftconvolve_fail(self, random_normal_2d: np.ndarray) -> None:
        kernel = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError):
            fourier.fftconvolve(random_normal_2d, kernel)
        with pytest.raises(ValueError):
            fourier.fftconvolve.py_func(random_normal_2d, kernel)
        kernel = np.ones(0, dtype=np.float32)
        with pytest.raises(ValueError):
            fourier.fftconvolve(random_normal_2d, kernel)
        with pytest.raises(ValueError):
            fourier.fftconvolve.py_func(random_normal_2d, kernel)

    def test_simulate_ism(self, random_normal_1d: np.ndarray) -> None:
        nchans = 512
        signal = random_normal_1d
        nsamps = len(signal)
        spectrum = np.ones(nchans, dtype=np.float32)
        dm_smear = np.ones(nchans, dtype=np.float32)
        tau_nus = np.ones(nchans, dtype=np.float32)
        simulate = ism_kernels.simulate_ism.py_func(
            signal,
            spectrum,
            dm_smear,
            tau_nus,
            1,
        )
        assert simulate.shape[0] == nchans
        assert simulate.shape[1] >= nsamps
        assert isinstance(simulate, np.ndarray)

    def test_normalize_template(self) -> None:
        arr = np.ones(10, dtype=np.float32)
        temp = np.pad(arr, (0, 100 - arr.size), mode="constant")
        temp_norm = fourier.normalize_template(temp)
        np.testing.assert_equal(temp_norm.size, temp.size)
        np.testing.assert_array_almost_equal(np.mean(temp_norm), 0.0, decimal=5)
        np.testing.assert_array_almost_equal(
            np.sqrt(np.sum(temp_norm**2)),
            1.0,
            decimal=5,
        )
        temp_norm = fourier.normalize_template.py_func(temp)
        temp_zeros = np.zeros(10, dtype=np.float32)
        temp_norm_zeros = fourier.normalize_template.py_func(temp_zeros)
        np.testing.assert_array_equal(temp_zeros, temp_norm_zeros)

    def test_circular_pad_goodsize(self, random_normal_1d: np.ndarray) -> None:
        bad_size = 937
        good_size = fourier.nb_fft_good_size(bad_size, real=True)
        arr = random_normal_1d[:bad_size]
        padded = fourier.circular_pad_goodsize(arr)
        np.testing.assert_equal(padded.size, good_size)
        np.testing.assert_array_equal(padded[: arr.size], arr)
        np.testing.assert_array_equal(padded[arr.size :], arr[: good_size - arr.size])
        fourier.circular_pad_goodsize.py_func(random_normal_1d)

    def test_convolve_templates(self, random_normal_1d: np.ndarray) -> None:
        samp_temps = typed.List([np.array([0.5, 1.0, 0.5]), np.array([1.0, -1.0])])
        ref_bins = typed.List([1, 0])
        result = fourier.convolve_templates(random_normal_1d, samp_temps, ref_bins)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(samp_temps), len(random_normal_1d))
        fourier.convolve_templates.py_func(random_normal_1d, samp_temps, ref_bins)
