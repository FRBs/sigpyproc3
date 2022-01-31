import pytest
import numpy as np
from sigpyproc.io import bits


class TestUnpacking(object):
    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_unpack_empty(self, nbits):
        input_arr = np.empty((0,), dtype=np.uint8)
        output = bits.unpack(input_arr, nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_packunpack(self, nbits):
        input_arr = np.arange(255, dtype=np.uint8)
        output = bits.pack(bits.unpack(input_arr, nbits=nbits), nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)

    def test_unpack_fail(self):
        nbits = 10
        input_arr = np.arange(255, dtype=np.uint8)
        with np.testing.assert_raises(ValueError):
            bits.unpack(input_arr, nbits=nbits)


class TestBitsInfo(object):
    def test_nbits_4(self):
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
