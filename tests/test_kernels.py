import numpy as np
import pytest
from scipy import stats

from sigpyproc.core import kernels


class TestKernels:
    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_unpack1_8(self, nbits: int, bitorder: str, parallel: bool) -> None:
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
        parallel_str = "" if parallel else "_serial"
        unpack_func = getattr(
            kernels,
            f"unpack{nbits:d}_8_{bitorder_str}{parallel_str}",
        )
        unpack_func(input_arr, unpacked)
        np.testing.assert_array_equal(unpacked, expected, strict=True)
        unpack_func.py_func(input_arr, unpacked)
        np.testing.assert_array_equal(unpacked, expected, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_pack(self, nbits: int, bitorder: str, parallel: bool) -> None:
        rng = np.random.default_rng()
        arr = rng.integers(255, size=2**10, dtype=np.uint8)
        parallel_str = "" if parallel else "_serial"
        unpack_func = getattr(kernels, f"unpack{nbits:d}_8_{bitorder}{parallel_str}")
        pack_func = getattr(kernels, f"pack{nbits:d}_8_{bitorder}{parallel_str}")
        unpacked = np.zeros(arr.size * 8 // nbits, dtype=np.uint8)
        unpack_func(arr, unpacked)
        packed = np.empty_like(arr)
        pack_func(unpacked, packed)
        np.testing.assert_array_equal(packed, arr, strict=True)
        unpack_func.py_func(arr, unpacked)
        pack_func.py_func(unpacked, packed)
        np.testing.assert_array_equal(packed, arr, strict=True)

    def test_np_mean(self) -> None:
        rng = np.random.default_rng()
        arr = rng.normal(size=(128, 256))
        np.testing.assert_array_almost_equal(
            kernels.np_mean(arr, axis=0),
            np.mean(arr, axis=0),
        )
        np.testing.assert_array_almost_equal(
            kernels.np_mean(arr, axis=1),
            np.mean(arr, axis=1),
        )
        with pytest.raises(ValueError):
            kernels.np_mean(arr, axis=2)
        arr = rng.normal(size=(128, 256, 512))
        with pytest.raises(ValueError):
            kernels.np_mean(arr, axis=0)


class TestMoments:
    def test_compute_moments_basic(self) -> None:
        nchans = 128
        nsamps = 256
        rng = np.random.default_rng()
        arr = rng.normal(size=(nchans, nsamps)).astype(np.float32)
        bag = kernels.MomentsBag(nchans)
        kernels.compute_online_moments_basic(arr.T.ravel(), bag, nsamps, startflag=0)
        np.testing.assert_array_almost_equal(bag.max, np.max(arr, axis=1))
        np.testing.assert_array_almost_equal(bag.min, np.min(arr, axis=1))
        np.testing.assert_array_almost_equal(
            bag.m1,
            stats.moment(arr, axis=1, center=0, order=1),
        )
        np.testing.assert_array_almost_equal(
            bag.m2,
            stats.moment(arr, axis=1, order=2) * nsamps,
            decimal=2,
        )
        kernels.compute_online_moments_basic.py_func(
            arr.T.ravel(),
            bag,
            nsamps,
            startflag=0,
        )

    def test_compute_moments(self) -> None:
        nchans = 128
        nsamps = 256
        rng = np.random.default_rng()
        arr = rng.normal(size=(nchans, nsamps)).astype(np.float32)
        bag = kernels.MomentsBag(nchans)
        kernels.compute_online_moments(arr.T.ravel(), bag, nsamps, startflag=0)
        np.testing.assert_array_almost_equal(bag.max, np.max(arr, axis=1))
        np.testing.assert_array_almost_equal(bag.min, np.min(arr, axis=1))
        np.testing.assert_array_almost_equal(
            bag.m1,
            stats.moment(arr, axis=1, center=0, order=1),
        )
        np.testing.assert_array_almost_equal(
            bag.m2,
            stats.moment(arr, axis=1, order=2) * nsamps,
            decimal=2,
        )
        np.testing.assert_array_almost_equal(
            bag.m3,
            stats.moment(arr, axis=1, order=3) * nsamps,
            decimal=2,
        )
        np.testing.assert_array_almost_equal(
            bag.m4,
            stats.moment(arr, axis=1, order=4) * nsamps,
            decimal=2,
        )
        kernels.compute_online_moments.py_func(
            arr.T.ravel(),
            bag,
            nsamps,
            startflag=0,
        )

    def test_compute_moments_add(self) -> None:
        nchans = 128
        nsamps = 256
        rng = np.random.default_rng()
        arr1 = rng.normal(size=(nchans, nsamps)).astype(np.float32)
        arr2 = rng.normal(size=(nchans, nsamps)).astype(np.float32)
        arr_expected = np.concatenate((arr1, arr2), axis=1).astype(np.float32)
        bag1 = kernels.MomentsBag(nchans)
        bag2 = kernels.MomentsBag(nchans)
        bag_out = kernels.MomentsBag(nchans)
        bag_expected = kernels.MomentsBag(nchans)
        kernels.compute_online_moments(arr1.T.ravel(), bag1, nsamps, startflag=0)
        kernels.compute_online_moments(arr2.T.ravel(), bag2, nsamps, startflag=0)
        kernels.compute_online_moments(
            arr_expected.T.ravel(),
            bag_expected,
            2 * nsamps,
            startflag=0,
        )
        kernels.add_online_moments(bag1, bag2, bag_out)
        np.testing.assert_array_almost_equal(bag_out.max, bag_expected.max)
        np.testing.assert_array_almost_equal(bag_out.min, bag_expected.min)
        np.testing.assert_array_almost_equal(bag_out.m1, bag_expected.m1)
        np.testing.assert_array_almost_equal(bag_out.m2, bag_expected.m2, decimal=2)
        np.testing.assert_array_almost_equal(bag_out.m3, bag_expected.m3, decimal=2)
        np.testing.assert_array_almost_equal(bag_out.m4, bag_expected.m4, decimal=2)
