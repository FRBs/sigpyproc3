import numpy as np
from sigpyproc import utils


class TestUtils(object):
    def test_roll_array(self):
        arr = np.arange(10)
        np.testing.assert_equal(utils.roll_array(arr, 0), arr)
        np.testing.assert_equal(utils.roll_array(arr, 1), np.roll(arr, -1))
        np.testing.assert_equal(utils.roll_array(arr, -1), np.roll(arr, 1))
        np.testing.assert_equal(utils.roll_array(arr, 10), arr)
        np.testing.assert_equal(utils.roll_array(arr, -10), arr)

    def test_nearest_factor(self):
        np.testing.assert_equal(utils.nearest_factor(10, 0), 1)
        np.testing.assert_equal(utils.nearest_factor(10, 3), 2)
        np.testing.assert_equal(utils.nearest_factor(10, 7), 5)
        np.testing.assert_equal(utils.nearest_factor(10, 11), 10)

    def test_duration_string(self):
        np.testing.assert_equal(utils.duration_string(0), "0.0 seconds")
        np.testing.assert_equal(utils.duration_string(60), "1.0 minutes")
        np.testing.assert_equal(utils.duration_string(3600), "1.0 hours")
        np.testing.assert_equal(utils.duration_string(86400), "1.0 days")


class TestFrequencyChannels(object):
    def test_from_sig(self):
        fch1 = 1500
        foff = -1
        nchans = 1024
        freqs = utils.FrequencyChannels.from_sig(fch1, foff, nchans)
        np.testing.assert_equal(freqs.fch1.value, fch1)
        np.testing.assert_equal(freqs.foff.value, foff)
        np.testing.assert_equal(freqs.nchans, nchans)

    def test_from_pfits(self):
        fcenter = 1500
        bandwidth = -1024
        nchans = 1024
        freqs = utils.FrequencyChannels.from_pfits(fcenter, bandwidth, nchans)
        np.testing.assert_equal(freqs.fcenter.value, fcenter)
        np.testing.assert_equal(freqs.bandwidth.value, -bandwidth)
        np.testing.assert_equal(freqs.nchans, nchans)

    def test_fail(self):
        arr = []
        with np.testing.assert_raises(AssertionError):
            utils.FrequencyChannels(arr)
        arr = [1, 2, 4, 7]
        with np.testing.assert_raises(AssertionError):
            utils.FrequencyChannels(arr)
