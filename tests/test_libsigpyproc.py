import pytest
import numpy as np
from sigpyproc import libSigPyProc as lib

numpy_types = [
    np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64,
    np.float32, np.float64]


def fft1(array):
    array_len = len(array)
    phase = -2j * np.pi * (np.arange(array_len) / float(array_len))
    phase = np.arange(array_len).reshape(-1, 1) * phase
    return np.sum(array * np.exp(phase), axis=1)


class TestLibSigPyProc:
    def test_unpackbits(self):
        input_arr = np.array([0, 2, 7, 23], dtype=np.uint8)
        expected_bit1 = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 1, 0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 1, 0, 1, 0, 0, 0], dtype=np.uint8)
        expected_bit2 = np.array([0, 0, 0, 0, 0, 0, 0, 2,
                                  0, 0, 1, 3, 0, 1, 1, 3], dtype=np.uint8)
        expected_bit4 = np.array([0, 0, 0, 2,
                                  0, 7, 1, 7], dtype=np.uint8)
        np.testing.assert_array_equal(expected_bit1, lib.unpack(input_arr, nbits=1))
        np.testing.assert_array_equal(expected_bit2, lib.unpack(input_arr, nbits=2))
        np.testing.assert_array_equal(expected_bit4, lib.unpack(input_arr, nbits=4))

    @pytest.mark.parametrize('nbits', [1, 2, 4])
    def test_unpackbits_empty(self, nbits):
        input_arr = np.empty((0,), dtype=np.uint8)
        output = lib.unpack(input_arr, nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)

    @pytest.mark.parametrize('nbits', [1, 2, 4])
    def test_packbits(self, nbits):
        input_arr = np.arange(255, dtype=np.uint8)
        output = lib.pack(lib.unpack(input_arr, nbits=nbits), nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)

    @pytest.mark.parametrize('dtype', numpy_types)
    def test_runningMean_size1(self, dtype):
        array = np.array([2, 4, 6], dtype)
        output = lib.runningMean(array, 1, array.size)
        np.testing.assert_allclose(array, output)

    @pytest.mark.parametrize('dtype', numpy_types)
    def test_runningMean_size2(self, dtype):
        array = np.array([2, 4, 6], dtype)
        expected = [3, 5, 6]
        output = lib.runningMean(array, 2, array.size)
        np.testing.assert_allclose(expected, output)

    def test_runningMean_roundoff_errors(self):
        in_ = np.repeat([0, 1, 0], [9, 9, 9])
        for window_size in range(3, 10):
            out = lib.runningMean(in_, window_size)
            np.testing.assert_equal(out.sum(), 10 - window_size)

    @pytest.mark.parametrize('dtype', numpy_types)
    def test_runningMedian_size1(self, dtype):
        array = np.array([3, 2, 5, 1, 4], dtype)
        output = lib.runningMedian(array, 1, array.size)
        np.testing.assert_allclose(array, output)

    @pytest.mark.parametrize('dtype', numpy_types)
    def test_runningMedian_size2(self, dtype):
        array = np.array([3, 2, 5, 1, 4], dtype)
        expected = [3, 3, 2, 4, 4]
        output = lib.runningMedian(array, 2, array.size)
        np.testing.assert_allclose(expected, output)

    @pytest.mark.parametrize('dtype', numpy_types)
    def test_runningMedian_size3(self, dtype):
        array = np.array([3, 2, 5, 1, 4], dtype)
        expected = [3, 3, 2, 4, 4]
        output = lib.runningMedian(array, 3, array.size)
        np.testing.assert_allclose(expected, output)

    def test_rfft(self):
        array = np.random.random(30)
        for npoints in (array.size, 2 * array.size):
            expected = fft1(array)[:(npoints // 2 + 1)]
            output = lib.rfft(array, npoints)
            np.testing.assert_allclose(expected, output)

    def test_ifft(self):
        array = np.random.random(30)
        output = lib.ifft(lib.rfft(array, 30))
        np.testing.assert_allclose(array, output)
