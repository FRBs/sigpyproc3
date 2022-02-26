import numpy as np
from sigpyproc.core import kernels, stats, rfi
from sigpyproc.header import Header


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
        desired = np.array([0, 0, 0, 0, 0, 1], dtype=np.bool)
        np.testing.assert_array_equal(desired, rfi.double_mad_mask(input_arr))


class TestRFIMask(object):
    def test_from_file(self, maskfile):
        mask = rfi.RFIMask.from_file(maskfile)
        assert isinstance(mask.header, Header)
        np.testing.assert_equal(mask.num_masked, 83)
        np.testing.assert_equal(mask.chan_mask.size, mask.header.nchans)
        np.testing.assert_almost_equal(mask.masked_fraction, 9.97, decimal=1)
