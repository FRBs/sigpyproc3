import numpy as np
import pytest

from sigpyproc.io import bits


class TestUnpacking:
    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_unpack_empty(self, nbits: int) -> None:
        input_arr = np.empty((0,), dtype=np.uint8)
        output = bits.unpack(input_arr, nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)

    @pytest.mark.parametrize("nbits", [1])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_unpack_1bit(self, nbits: int, bitorder: str, parallel: bool) -> None:
        rng = np.random.default_rng()
        arr = rng.integers(255, size=2**10, dtype=np.uint8)
        expected = np.unpackbits(arr, bitorder=bitorder)  # type: ignore [arg-type]
        output_buff = np.zeros(arr.size * 8 // nbits, dtype=np.uint8)
        output_buff = bits.unpack(
            arr,
            nbits,
            unpacked=output_buff,
            bitorder=bitorder,
            parallel=parallel,
        )
        output_return = bits.unpack(
            arr,
            nbits,
            unpacked=None,
            bitorder=bitorder,
            parallel=parallel,
        )
        np.testing.assert_array_equal(output_buff, expected, strict=True)
        np.testing.assert_array_equal(output_return, expected, strict=True)

    @pytest.mark.parametrize("nbits", [1])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_pack_1bit(self, nbits: int, bitorder: str, parallel: bool) -> None:
        rng = np.random.default_rng()
        arr = rng.integers((1 << nbits) - 1, size=2**10, dtype=np.uint8)
        expected = np.packbits(arr, bitorder=bitorder)  # type: ignore [arg-type]
        output_buff = np.zeros(arr.size // 8, dtype=np.uint8)
        output_buff = bits.pack(
            arr,
            nbits,
            packed=output_buff,
            bitorder=bitorder,
            parallel=parallel,
        )
        output_return = bits.pack(
            arr,
            nbits,
            packed=None,
            bitorder=bitorder,
            parallel=parallel,
        )
        np.testing.assert_array_equal(output_buff, expected, strict=True)
        np.testing.assert_array_equal(output_return, expected, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_packunpack(self, nbits: int, bitorder: str, parallel: bool) -> None:
        rng = np.random.default_rng()
        arr = rng.integers(255, size=2**10, dtype=np.uint8)
        tmp_unpack = np.zeros(arr.size * 8 // nbits, dtype=np.uint8)
        tmp_unpack = bits.unpack(
            arr,
            nbits=nbits,
            unpacked=tmp_unpack,
            bitorder=bitorder,
            parallel=parallel,
        )
        output_buff = np.zeros_like(arr)
        output_buff = bits.pack(
            tmp_unpack,
            nbits=nbits,
            packed=output_buff,
            bitorder=bitorder,
            parallel=parallel,
        )
        output_return = bits.pack(
            bits.unpack(arr, nbits=nbits, bitorder=bitorder, parallel=parallel),
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
        )
        np.testing.assert_array_equal(output_buff, arr, strict=True)
        np.testing.assert_array_equal(output_return, arr, strict=True)

    @pytest.mark.parametrize(
        ("nbits", "dtype", "bitorder"),
        [(10, np.uint8, "big"), (4, np.float32, "little"), (4, np.uint8, "invalid")],
    )
    def test_unpack_fail(self, nbits: int, dtype: np.dtype, bitorder: str) -> None:
        input_arr = np.arange(255, dtype=dtype)
        with np.testing.assert_raises(ValueError):
            bits.unpack(input_arr, nbits=nbits, bitorder=bitorder)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_unpack_fail_unpacked(self, nbits: int) -> None:
        input_arr = np.arange(255, dtype=np.uint8)
        unpacked = np.zeros_like(input_arr)
        with np.testing.assert_raises(ValueError):
            bits.unpack(input_arr, nbits=nbits, unpacked=unpacked)

    @pytest.mark.parametrize(
        ("nbits", "dtype", "bitorder"),
        [(10, np.uint8, "big"), (4, np.float32, "little"), (4, np.uint8, "invalid")],
    )
    def test_pack_fail(self, nbits: int, dtype: np.dtype, bitorder: str) -> None:
        rng = np.random.default_rng()
        input_arr = rng.integers((1 << nbits) - 1, size=2**10).astype(dtype)
        with np.testing.assert_raises(ValueError):
            bits.pack(input_arr, nbits=nbits, bitorder=bitorder)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_pack_fail_packed(self, nbits: int) -> None:
        rng = np.random.default_rng()
        input_arr = rng.integers((1 << nbits) - 1, size=2**10, dtype=np.uint8)
        packed = np.zeros_like(input_arr)
        with np.testing.assert_raises(ValueError):
            bits.pack(input_arr, nbits=nbits, packed=packed)


class TestBitsInfo:
    def test_nbits_4(self) -> None:
        bitsinfo = bits.BitsInfo(4)
        np.testing.assert_equal(bitsinfo.nbits, 4)
        np.testing.assert_equal(bitsinfo.dtype, np.uint8)
        np.testing.assert_equal(bitsinfo.itemsize, 1)
        np.testing.assert_equal(bitsinfo.unpack, True)
        np.testing.assert_equal(bitsinfo.bitfact, 2)
        np.testing.assert_equal(bitsinfo.digi_min, 0)
        np.testing.assert_equal(bitsinfo.digi_max, 15)
        np.testing.assert_equal(bitsinfo.digi_mean, 7.5)
        np.testing.assert_equal(bitsinfo.digi_scale, 7.5 / bitsinfo.digi_sigma)
        info = bitsinfo.to_dict()
        np.testing.assert_equal(info["nbits"], 4)

    def test_quantize(self) -> None:
        arr_norm = np.array(
            [-15.5, -10.5, -5.5, -0.5, 0.0, 0.5, 5.5, 10.5, 15.5],
            dtype=np.float32,
        )
        arr_quant_16 = np.array(
            [0, 0, 2731, 30037, 32768, 35498, 62804, 65535, 65535],
            dtype=np.uint16,
        )
        arr_quant_8 = np.array([0, 0, 11, 117, 128, 138, 244, 255, 255], dtype=np.uint8)
        arr_quant_4 = np.array([0, 0, 1, 7, 8, 8, 14, 15, 15], dtype=np.uint8)
        arr_quant_2 = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3], dtype=np.uint8)
        arr_quant_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(
            bits.BitsInfo(16).quantize(arr_norm),
            arr_quant_16,
        )
        np.testing.assert_array_equal(bits.BitsInfo(8).quantize(arr_norm), arr_quant_8)
        np.testing.assert_array_equal(bits.BitsInfo(4).quantize(arr_norm), arr_quant_4)
        np.testing.assert_array_equal(bits.BitsInfo(2).quantize(arr_norm), arr_quant_2)
        np.testing.assert_array_equal(bits.BitsInfo(1).quantize(arr_norm), arr_quant_1)
