import numpy as np
import pytest

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
