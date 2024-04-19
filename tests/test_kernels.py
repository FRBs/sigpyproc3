import numpy as np
import pytest

from sigpyproc.core import kernels, rfi, stats
from sigpyproc.header import Header


class TestKernels:
    def test_unpack1_8(self) -> None:
        input_arr = np.array([7, 23], dtype=np.uint8)
        expected_big = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            dtype=np.uint8,
        )
        expected_little = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
            dtype=np.uint8,
        )
        unpacked_big = np.empty_like(expected_big)
        unpacked_little = np.empty_like(expected_little)
        kernels.unpack1_8_big(input_arr, unpacked_big)
        kernels.unpack1_8_little(input_arr, unpacked_little)
        np.testing.assert_array_equal(unpacked_big, expected_big, strict=True)
        np.testing.assert_array_equal(unpacked_little, expected_little, strict=True)

    def test_unpack2_8(self) -> None:
        input_arr = np.array([7, 23], dtype=np.uint8)
        expected_big = np.array(
            [0, 0, 1, 3, 0, 1, 1, 3],
            dtype=np.uint8,
        )
        expected_little = np.array(
            [3, 1, 0, 0, 3, 1, 1, 0],
            dtype=np.uint8,
        )
        unpacked_big = np.empty_like(expected_big)
        unpacked_little = np.empty_like(expected_little)
        kernels.unpack2_8_big(input_arr, unpacked_big)
        kernels.unpack2_8_little(input_arr, unpacked_little)
        np.testing.assert_array_equal(unpacked_big, expected_big, strict=True)
        np.testing.assert_array_equal(unpacked_little, expected_little, strict=True)

    def test_unpack4_8(self) -> None:
        input_arr = np.array([7, 23], dtype=np.uint8)
        expected_big = np.array([0, 7, 1, 7], dtype=np.uint8)
        expected_little = np.array([7, 0, 7, 1], dtype=np.uint8)
        unpacked_big = np.empty_like(expected_big)
        unpacked_little = np.empty_like(expected_little)
        kernels.unpack4_8_big(input_arr, unpacked_big)
        kernels.unpack4_8_little(input_arr, unpacked_little)
        np.testing.assert_array_equal(unpacked_big, expected_big, strict=True)
        np.testing.assert_array_equal(unpacked_little, expected_little, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_pack(self, nbits: int, bitorder: str, parallel: bool) -> None:  # noqa: FBT001
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


class TestStats(object):
    def test_zscore_mad(self):
        input_arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        desired = np.array(
            [-1.01173463, -0.33724488, 0.33724488, 1.01173463], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(
            desired, stats.zscore_mad(input_arr), decimal=4
        )

    def test_zscore_double_mad(self):
        input_arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        desired = np.array(
            [-1.01173463, -0.33724488, 0.33724488, 1.01173463], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(
            desired, stats.zscore_double_mad(input_arr), decimal=4
        )


class TestRFI(object):
    def test_double_mad_mask(self):
        input_arr = np.array([1, 2, 3, 4, 5, 20], dtype=np.uint8)
        desired = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        np.testing.assert_array_equal(desired, rfi.double_mad_mask(input_arr))


class TestRFIMask(object):
    def test_from_file(self, maskfile):
        mask = rfi.RFIMask.from_file(maskfile)
        assert isinstance(mask.header, Header)
        np.testing.assert_equal(mask.num_masked, 83)
        np.testing.assert_equal(mask.chan_mask.size, mask.header.nchans)
        np.testing.assert_almost_equal(mask.masked_fraction, 9.97, decimal=1)
