import numpy as np
from sigpyproc.core import kernels


class TestKernels(object):
    def test_unpack1_8(self):
        input_arr = np.array([0, 2, 7, 23], dtype=np.uint8)
        expected_bit1 = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8
        )
        np.testing.assert_array_equal(expected_bit1, kernels.unpack1_8(input_arr))

    def test_unpack2_8(self):
        input_arr = np.array([0, 2, 7, 23], dtype=np.uint8)
        expected_bit2 = np.array(
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 3, 0, 1, 1, 3], dtype=np.uint8
        )
        np.testing.assert_array_equal(expected_bit2, kernels.unpack2_8(input_arr))

    def test_unpack4_8(self):
        input_arr = np.array([0, 2, 7, 23], dtype=np.uint8)
        expected_bit4 = np.array([0, 0, 0, 2, 0, 7, 1, 7], dtype=np.uint8)
        np.testing.assert_array_equal(expected_bit4, kernels.unpack4_8(input_arr))

    def test_pack2_8(self):
        input_arr = np.arange(255, dtype=np.uint8)
        output = kernels.pack2_8(kernels.unpack2_8(input_arr))
        np.testing.assert_array_equal(input_arr, output)

    def test_pack4_8(self):
        input_arr = np.arange(255, dtype=np.uint8)
        output = kernels.pack4_8(kernels.unpack4_8(input_arr))
        np.testing.assert_array_equal(input_arr, output)
