import pytest
import numpy as np
from sigpyproc.io import bits


class Testlibcpp(object):
    def test_unpackbits(self):
        input_arr = np.array([0, 2, 7, 23], dtype=np.uint8)
        expected_bit1 = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8
        )
        expected_bit2 = np.array(
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 3, 0, 1, 1, 3], dtype=np.uint8
        )
        expected_bit4 = np.array([0, 0, 0, 2, 0, 7, 1, 7], dtype=np.uint8)
        np.testing.assert_array_equal(expected_bit1, bits.unpack(input_arr, nbits=1))
        np.testing.assert_array_equal(expected_bit2, bits.unpack(input_arr, nbits=2))
        np.testing.assert_array_equal(expected_bit4, bits.unpack(input_arr, nbits=4))

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_unpackbits_empty(self, nbits):
        input_arr = np.empty((0,), dtype=np.uint8)
        output = bits.unpack(input_arr, nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_packbits(self, nbits):
        input_arr = np.arange(255, dtype=np.uint8)
        output = bits.pack(bits.unpack(input_arr, nbits=nbits), nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)
