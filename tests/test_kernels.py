import numpy as np
import pytest
from numba import typed
from scipy import signal, stats

from sigpyproc.core import kernels


class TestKernelsPackUnpack:
    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    def test_unpack(self, nbits: int, bitorder: str) -> None:
        input_arr = np.array([7, 23], dtype=np.uint8)
        if nbits == 1 and bitorder == "big":
            expected = np.array(
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                dtype=np.uint8,
            )
        elif nbits == 1 and bitorder == "little":
            expected = np.array(
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                dtype=np.uint8,
            )
        elif nbits == 2 and bitorder == "big":
            expected = np.array([0, 0, 1, 3, 0, 1, 1, 3], dtype=np.uint8)
        elif nbits == 2 and bitorder == "little":
            expected = np.array([3, 1, 0, 0, 3, 1, 1, 0], dtype=np.uint8)
        elif nbits == 4 and bitorder == "big":
            expected = np.array([0, 7, 1, 7], dtype=np.uint8)
        elif nbits == 4 and bitorder == "little":
            expected = np.array([7, 0, 7, 1], dtype=np.uint8)
        unpacked = np.empty_like(expected)
        bitorder_str = "big" if bitorder[0] == "b" else "little"
        unpack_func = getattr(
            kernels,
            f"unpack{nbits:d}_8_{bitorder_str}",
        )
        unpack_func(input_arr, unpacked)
        np.testing.assert_array_equal(unpacked, expected, strict=True)
        unpack_func.py_func(input_arr, unpacked)
        np.testing.assert_array_equal(unpacked, expected, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    def test_pack(self, nbits: int, bitorder: str) -> None:
        rng = np.random.default_rng()
        arr = rng.integers(255, size=2**10, dtype=np.uint8)
        unpack_func = getattr(kernels, f"unpack{nbits:d}_8_{bitorder}")
        pack_func = getattr(kernels, f"pack{nbits:d}_8_{bitorder}")
        unpacked = np.zeros(arr.size * 8 // nbits, dtype=np.uint8)
        unpack_func(arr, unpacked)
        packed = np.empty_like(arr)
        pack_func(unpacked, packed)
        np.testing.assert_array_equal(packed, arr, strict=True)
        unpack_func.py_func(arr, unpacked)
        pack_func.py_func(unpacked, packed)
        np.testing.assert_array_equal(packed, arr, strict=True)

    @pytest.mark.parametrize("bitorder", ["big", "little"])
    def test_pack1_8_vect(self, bitorder: str) -> None:
        rng = np.random.default_rng()
        arr = rng.integers((1 << 1), size=2**10, dtype=np.uint8)
        expected = np.packbits(arr, bitorder=bitorder)  # type: ignore[arg-type]
        packed = np.empty_like(expected)
        big_endian = bitorder == "big"
        kernels.pack1_8_vect(arr, packed, big_endian=big_endian)
        np.testing.assert_array_equal(packed, expected, strict=True)
        kernels.pack1_8_vect.py_func(arr, packed, big_endian=big_endian)
        np.testing.assert_array_equal(packed, expected, strict=True)


class TestMoments:
    nchans = 10
    nsamps = 1000
    decimal_precision = 2

    def test_update_moments(self, random_normal_1d: np.ndarray) -> None:
        m1, m2, m3, m4, n = 0, 0, 0, 0, 0
        for val in random_normal_1d:
            m1, m2, m3, m4, n = kernels.update_moments(val, m1, m2, m3, m4, n)
        np.testing.assert_almost_equal(
            m1,
            stats.moment(random_normal_1d, axis=0, center=0, order=1),
            decimal=self.decimal_precision,
        )
        np.testing.assert_almost_equal(
            m2,
            stats.moment(random_normal_1d, axis=0, order=2) * n,
            decimal=self.decimal_precision,
        )
        np.testing.assert_almost_equal(
            m3,
            stats.moment(random_normal_1d, axis=0, order=3) * n,
            decimal=self.decimal_precision,
        )
        np.testing.assert_almost_equal(
            m4,
            stats.moment(random_normal_1d, axis=0, order=4) * n,
            decimal=self.decimal_precision,
        )
        kernels.update_moments.py_func(random_normal_1d, m1, m2, m3, m4, n)

    def test_update_moments_basic(self, random_normal_1d: np.ndarray) -> None:
        m1, m2, n = 0, 0, 0
        for val in random_normal_1d:
            m1, m2, n = kernels.update_moments_basic(val, m1, m2, n)
        np.testing.assert_almost_equal(
            m1,
            stats.moment(random_normal_1d, axis=0, center=0, order=1),
            decimal=self.decimal_precision,
        )
        np.testing.assert_almost_equal(
            m2,
            stats.moment(random_normal_1d, axis=0, order=2) * n,
            decimal=self.decimal_precision,
        )
        kernels.update_moments_basic.py_func(random_normal_1d, m1, m2, n)

    def test_compute_moments_basic(self, random_normal_2d: np.ndarray) -> None:
        moments = np.zeros(self.nchans, dtype=kernels.moments_dtype)
        kernels.compute_online_moments_basic(
            random_normal_2d.T.ravel(),
            moments,
            startflag=0,
        )
        np.testing.assert_array_almost_equal(
            moments["max"],
            np.max(random_normal_2d, axis=1),
        )
        np.testing.assert_array_almost_equal(
            moments["min"],
            np.min(random_normal_2d, axis=1),
        )
        np.testing.assert_array_almost_equal(
            moments["m1"],
            stats.moment(random_normal_2d, axis=1, center=0, order=1),
        )
        np.testing.assert_array_almost_equal(
            moments["m2"],
            stats.moment(random_normal_2d, axis=1, order=2) * self.nsamps,
            decimal=self.decimal_precision,
        )
        kernels.compute_online_moments_basic.py_func(
            random_normal_2d.T.ravel(),
            moments,
            startflag=0,
        )

    def test_compute_moments(self, random_normal_2d: np.ndarray) -> None:
        moments = np.zeros(self.nchans, dtype=kernels.moments_dtype)
        kernels.compute_online_moments(random_normal_2d.T.ravel(), moments, startflag=0)
        np.testing.assert_array_almost_equal(
            moments["max"],
            np.max(random_normal_2d, axis=1),
        )
        np.testing.assert_array_almost_equal(
            moments["min"],
            np.min(random_normal_2d, axis=1),
        )
        np.testing.assert_array_almost_equal(
            moments["m1"],
            stats.moment(random_normal_2d, axis=1, center=0, order=1),
        )
        np.testing.assert_array_almost_equal(
            moments["m2"],
            stats.moment(random_normal_2d, axis=1, order=2) * self.nsamps,
            decimal=2,
        )
        np.testing.assert_array_almost_equal(
            moments["m3"],
            stats.moment(random_normal_2d, axis=1, order=3) * self.nsamps,
            decimal=2,
        )
        np.testing.assert_array_almost_equal(
            moments["m4"],
            stats.moment(random_normal_2d, axis=1, order=4) * self.nsamps,
            decimal=2,
        )
        kernels.compute_online_moments.py_func(
            random_normal_2d.T.ravel(),
            moments,
            startflag=0,
        )

    def test_compute_moments_add(self, random_normal_2d: np.ndarray) -> None:
        moments1 = np.zeros(self.nchans, dtype=kernels.moments_dtype)
        moments2 = np.zeros(self.nchans, dtype=kernels.moments_dtype)
        moments_exp = np.zeros(self.nchans, dtype=kernels.moments_dtype)
        moments_out = np.zeros(self.nchans, dtype=kernels.moments_dtype)
        kernels.compute_online_moments(
            random_normal_2d[:, : self.nsamps // 2].T.ravel(),
            moments1,
            startflag=0,
        )
        kernels.compute_online_moments(
            random_normal_2d[:, self.nsamps // 2 :].T.ravel(),
            moments2,
            startflag=0,
        )
        kernels.compute_online_moments(
            random_normal_2d.T.ravel(),
            moments_exp,
            startflag=0,
        )
        kernels.add_online_moments(moments1, moments2, moments_out)
        np.testing.assert_array_almost_equal(moments_out["max"], moments_exp["max"])
        np.testing.assert_array_almost_equal(moments_out["min"], moments_exp["min"])
        np.testing.assert_array_almost_equal(
            moments_out["m1"],
            moments_exp["m1"],
            decimal=self.decimal_precision,
        )
        np.testing.assert_array_almost_equal(
            moments_out["m2"],
            moments_exp["m2"],
            decimal=self.decimal_precision,
        )
        np.testing.assert_array_almost_equal(
            moments_out["m3"],
            moments_exp["m3"],
            decimal=self.decimal_precision,
        )
        np.testing.assert_array_almost_equal(
            moments_out["m4"],
            moments_exp["m4"],
            decimal=self.decimal_precision,
        )
        kernels.add_online_moments.py_func(moments1, moments2, moments_out)


class TestKernels:
    @pytest.mark.parametrize("factor", [1, 2, 4, 7, 10])
    def test_downsample_1d_mean(
        self,
        random_normal_1d: np.ndarray,
        factor: int,
    ) -> None:
        nsamps_new = (random_normal_1d.size // factor) * factor
        expected = np.mean(random_normal_1d[:nsamps_new].reshape(-1, factor), axis=1)
        result = kernels.downsample_1d_mean(random_normal_1d, factor)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        result = kernels.downsample_1d_mean.py_func(random_normal_1d, factor)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    @pytest.mark.parametrize(
        ("factor1", "factor2"),
        [(1, 1), (1, 3), (2, 6), (7, 7), (10, 23)],
    )
    def test_downsample_2d_mean_flat(
        self,
        random_normal_2d: np.ndarray,
        factor1: int,
        factor2: int,
    ) -> None:
        dim1, dim2 = random_normal_2d.shape
        new_dim1 = dim1 // factor1
        new_dim2 = dim2 // factor2
        new_shape = (new_dim1, factor1, new_dim2, factor2)
        expected = np.mean(
            random_normal_2d[: new_dim1 * factor1, : new_dim2 * factor2].reshape(
                new_shape,
            ),
            axis=(1, 3),
        )
        result = kernels.downsample_2d_mean_flat(
            random_normal_2d.ravel(),
            factor1,
            factor2,
            dim1,
            dim2,
        )
        np.testing.assert_array_almost_equal(result, expected.ravel(), decimal=5)
        result = kernels.downsample_2d_mean_flat.py_func(
            random_normal_2d.ravel(),
            factor1,
            factor2,
            dim1,
            dim2,
        )
        np.testing.assert_array_almost_equal(result, expected.ravel(), decimal=5)

    def test_extract_tim(self, random_normal_2d: np.ndarray) -> None:
        out = np.zeros(random_normal_2d.shape[1], dtype=np.float32)
        kernels.extract_tim(random_normal_2d.T.ravel(), out, 10, 1000, 0)
        expected = random_normal_2d.sum(axis=0)
        np.testing.assert_array_almost_equal(out, expected, decimal=2)
        kernels.extract_tim.py_func(random_normal_2d.T.ravel(), out, 10, 1000, 0)

    def test_extract_bpass(self, random_normal_2d: np.ndarray) -> None:
        out = np.zeros(random_normal_2d.shape[0], dtype=np.float32)
        kernels.extract_bpass(random_normal_2d.T.ravel(), out, 10, 1000)
        expected = random_normal_2d.sum(axis=1)
        np.testing.assert_array_almost_equal(out, expected, decimal=2)
        kernels.extract_bpass.py_func(random_normal_2d.T.ravel(), out, 10, 1000)

    def test_mask_channels(self, random_normal_2d: np.ndarray) -> None:
        arr = random_normal_2d.copy()
        arr_ravel = arr.T.ravel()
        rng = np.random.default_rng()
        mask = rng.choice([True, False], size=random_normal_2d.shape[0])
        maskvalue = 0
        kernels.mask_channels(arr_ravel, mask, maskvalue, 10, 1000)
        out = arr_ravel.reshape(random_normal_2d.T.shape).T
        for ichan in range(random_normal_2d.shape[0]):
            if mask[ichan]:
                np.testing.assert_array_equal(out[ichan], np.zeros_like(out[ichan]))
            else:
                np.testing.assert_array_equal(out[ichan], random_normal_2d[ichan])
        kernels.mask_channels.py_func(arr.T.ravel(), mask, maskvalue, 10, 1000)

    def test_invert_freq(self, random_normal_2d: np.ndarray) -> None:
        inverted = kernels.invert_freq(random_normal_2d.T.ravel(), 10, 1000)
        reinverted = kernels.invert_freq(inverted, 10, 1000)
        np.testing.assert_array_equal(reinverted, random_normal_2d.T.ravel())
        kernels.invert_freq.py_func(random_normal_2d.T.ravel(), 10, 1000)

    def test_detrend_1d(self) -> None:
        arr = np.ones(100, dtype=np.float32)
        np.testing.assert_array_almost_equal(
            kernels.detrend_1d(arr),
            signal.detrend(arr),
            decimal=3,
        )
        kernels.detrend_1d.py_func(arr)
        arr = np.arange(100, dtype=np.float32)
        np.testing.assert_array_almost_equal(
            kernels.detrend_1d(arr),
            signal.detrend(arr),
            decimal=3,
        )
        kernels.detrend_1d.py_func(arr)
        arr = np.array([1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            kernels.detrend_1d(arr),
            signal.detrend(arr),
            decimal=3,
        )
        kernels.detrend_1d.py_func(arr)

    def test_detrend_1d_fail(self) -> None:
        with pytest.raises(ValueError):
            kernels.detrend_1d(np.array([]))
        with pytest.raises(ValueError):
            kernels.detrend_1d.py_func(np.array([]))

    def test_normalize_template(self) -> None:
        arr = np.ones(10, dtype=np.float32)
        temp = np.pad(arr, (0, 100 - arr.size), mode="constant")
        temp_norm = kernels.normalize_template(temp)
        np.testing.assert_equal(temp_norm.size, temp.size)
        np.testing.assert_array_almost_equal(np.mean(temp_norm), 0.0, decimal=5)
        np.testing.assert_array_almost_equal(
            np.sqrt(np.sum(temp_norm**2)),
            1.0,
            decimal=5,
        )
        temp_norm = kernels.normalize_template.py_func(temp)
        temp_zeros = np.zeros(10, dtype=np.float32)
        temp_norm_zeros = kernels.normalize_template.py_func(temp_zeros)
        np.testing.assert_array_equal(temp_zeros, temp_norm_zeros)

    def test_circular_pad_goodsize(self, random_normal_1d: np.ndarray) -> None:
        bad_size = 937
        good_size = kernels.nb_fft_good_size(bad_size, real=True)
        arr = random_normal_1d[:bad_size]
        padded = kernels.circular_pad_goodsize(arr)
        np.testing.assert_equal(padded.size, good_size)
        np.testing.assert_array_equal(padded[: arr.size], arr)
        np.testing.assert_array_equal(padded[arr.size :], arr[: good_size - arr.size])
        kernels.circular_pad_goodsize.py_func(random_normal_1d)

    def test_convolve_templates(self, random_normal_1d: np.ndarray) -> None:
        samp_temps = typed.List([np.array([0.5, 1.0, 0.5]), np.array([1.0, -1.0])])
        ref_bins = typed.List([1, 0])
        result = kernels.convolve_templates(random_normal_1d, samp_temps, ref_bins)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(samp_temps), len(random_normal_1d))
        kernels.convolve_templates.py_func(random_normal_1d, samp_temps, ref_bins)


class TestFourierKernels:
    def test_form_mspec(self, random_normal_1d_complex: np.ndarray) -> None:
        mspec = kernels.form_mspec(random_normal_1d_complex)
        np.testing.assert_array_almost_equal(
            mspec,
            np.abs(random_normal_1d_complex),
            decimal=5,
        )
        kernels.form_mspec.py_func(random_normal_1d_complex)

    def test_form_interp_mspec(self, random_normal_1d_complex: np.ndarray) -> None:
        mspec = kernels.form_interp_mspec(random_normal_1d_complex)
        assert mspec.shape == random_normal_1d_complex.shape
        kernels.form_interp_mspec.py_func(random_normal_1d_complex)

    def test_fs_running_median(self) -> None:
        arr = np.exp(1j * np.linspace(0, 10 * np.pi, 1000, dtype=np.float32))
        out = kernels.fs_running_median(arr, 10, 20, 500)
        pow_spec = np.abs(out) ** 2
        # Check that the power spectrum is normalized
        assert 0.5 < np.mean(pow_spec) < 2.0
        # Check that the phase is preserved
        np.testing.assert_array_almost_equal(
            np.angle(out),
            np.angle(arr),
            decimal=5,
        )
        kernels.fs_running_median.py_func(arr, 10, 20, 500)

    def test_rfft(self, random_normal_1d: np.ndarray) -> None:
        expected = np.fft.rfft(random_normal_1d)
        result = kernels.nb_rfft.py_func(random_normal_1d)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        inv_result = kernels.nb_irfft.py_func(result)
        np.testing.assert_array_almost_equal(inv_result, random_normal_1d, decimal=5)

    def test_fft(self, random_normal_1d_complex: np.ndarray) -> None:
        expected = np.fft.fft(random_normal_1d_complex)
        result = kernels.nb_fft.py_func(random_normal_1d_complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        inv_result = kernels.nb_ifft.py_func(result)
        np.testing.assert_array_almost_equal(
            inv_result,
            random_normal_1d_complex,
            decimal=5,
        )

    def test_fftconvolve(self, random_normal_1d: np.ndarray) -> None:
        kernel = np.ones(10, dtype=np.float32)
        expected = np.convolve(random_normal_1d, kernel, mode="full")
        result = kernels.fftconvolve.py_func(random_normal_1d, kernel)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        kernel_empty = np.zeros(0, dtype=np.float32)
        result = kernels.fftconvolve.py_func(random_normal_1d, kernel_empty)
        np.testing.assert_array_almost_equal(result, kernel_empty, decimal=5)

    def test_fftconvolve_fail(self, random_normal_2d: np.ndarray) -> None:
        kernel = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError):
            kernels.fftconvolve(random_normal_2d, kernel)
        with pytest.raises(ValueError):
            kernels.fftconvolve.py_func(random_normal_2d, kernel)
        kernel = np.ones(0, dtype=np.float32)
        with pytest.raises(ValueError):
            kernels.fftconvolve(random_normal_2d, kernel)
        with pytest.raises(ValueError):
            kernels.fftconvolve.py_func(random_normal_2d, kernel)

    def test_simulate_ism(self, random_normal_1d: np.ndarray) -> None:
        nchans = 512
        signal = random_normal_1d
        nsamps = len(signal)
        spectrum = np.ones(nchans, dtype=np.float32)
        dm_smear = np.ones(nchans, dtype=np.float32)
        tau_nus = np.ones(nchans, dtype=np.float32)
        simulate = kernels.simulate_ism.py_func(
            signal,
            spectrum,
            dm_smear,
            tau_nus,
            1,
        )
        assert simulate.shape[0] == nchans
        assert simulate.shape[1] >= nsamps
        assert isinstance(simulate, np.ndarray)


class TestRollingKernels:
    def test_nb_roll(self) -> None:
        arr = np.zeros((10, 10), dtype=np.float32)
        shift = 10
        rolled = kernels.nb_roll.py_func(arr, shift, axis=0)
        np.testing.assert_array_equal(rolled, np.roll(arr, shift, axis=0))
        rolled = kernels.nb_roll.py_func(arr, shift, axis=1)
        np.testing.assert_array_equal(rolled, np.roll(arr, shift, axis=1))
        rolled = kernels.nb_roll.py_func(arr, shift)
        np.testing.assert_array_equal(rolled, np.roll(arr, shift))

    def test_roll_block(self) -> None:
        arr = np.zeros((10, 10), dtype=np.float32)
        arr[:, 5] = 1.0
        shifts = np.arange(arr.shape[0])
        rolled = kernels.roll_block.py_func(arr, shifts)
        unrolled = kernels.roll_block.py_func(rolled, -shifts)
        np.testing.assert_array_equal(unrolled, arr)
        with pytest.raises(ValueError):
            kernels.roll_block.py_func(shifts, 1)
        with pytest.raises(ValueError):
            kernels.roll_block.py_func(arr, shifts[:-1])

    def test_dmt_block(self) -> None:
        arr = np.zeros((10, 10), dtype=np.float32)
        arr[:, 5] = 1.0
        dm_delays = np.expand_dims(np.arange(arr.shape[0]), 0)
        dmt = kernels.dmt_block.py_func(arr, dm_delays)
        assert dmt.shape == (len(dm_delays), arr.shape[1])
        expected = np.ones(arr.shape[1], dtype=np.float32)
        np.testing.assert_array_equal(dmt[0], expected)
        with pytest.raises(ValueError):
            kernels.dmt_block.py_func(arr[0], dm_delays)
        with pytest.raises(ValueError):
            kernels.dmt_block.py_func(arr, dm_delays[:, :-1])

    def test_roll_block_valid_positive_shifts(self) -> None:
        arr = np.zeros((10, 19), dtype=np.float32)
        arr[:, 9] = 1.0
        shifts = np.arange(arr.shape[0])
        rolled = kernels.roll_block_valid.py_func(arr, shifts)
        valid_samples = arr.shape[1] - np.abs(shifts).max()
        assert rolled.shape == (arr.shape[0], valid_samples)
        expected = np.ones(valid_samples)
        np.testing.assert_array_equal(rolled.sum(axis=0), expected)

    def test_roll_block_valid_negative_shifts(self) -> None:
        arr = np.zeros((10, 19), dtype=np.float32)
        arr[:, 9] = 1.0
        shifts = -np.arange(arr.shape[0])
        rolled = kernels.roll_block_valid.py_func(arr, shifts)
        valid_samples = arr.shape[1] - np.abs(shifts).max()
        assert rolled.shape == (arr.shape[0], valid_samples)
        expected = np.ones(valid_samples)
        np.testing.assert_array_equal(rolled.sum(axis=0), expected)

    def test_roll_block_valid_fail(self) -> None:
        arr = np.zeros((10, 19), dtype=np.float32)
        arr[:, 9] = 1.0
        shifts = np.arange(arr.shape[0])
        with pytest.raises(ValueError):
            kernels.roll_block_valid.py_func(arr[0], shifts)
        with pytest.raises(ValueError):
            kernels.roll_block_valid.py_func(arr, np.arange(arr.shape[1] + 5))
        with pytest.raises(ValueError):
            kernels.roll_block_valid.py_func(arr, np.arange(arr.shape[0]) * 5)

    def test_dmt_block_valid(self) -> None:
        arr = np.zeros((10, 19), dtype=np.float32)
        arr[:, 9] = 1.0
        dm_delays = np.expand_dims(np.arange(arr.shape[0]), 0)
        dmt = kernels.dmt_block_valid.py_func(arr, dm_delays)
        assert dmt.shape == (len(dm_delays), arr.shape[1] - dm_delays.max())
        expected = np.ones(arr.shape[1] - dm_delays.max())
        np.testing.assert_array_equal(dmt[0], expected)
        with pytest.raises(ValueError):
            kernels.dmt_block_valid.py_func(arr[0], dm_delays)
        with pytest.raises(ValueError):
            kernels.dmt_block_valid.py_func(arr, dm_delays[:, :-1])
        with pytest.raises(ValueError):
            kernels.dmt_block_valid.py_func(
                arr,
                np.expand_dims(np.arange(arr.shape[0]) * 5, 0),
            )
