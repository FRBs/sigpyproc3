import numpy as np

from sigpyproc import profile


class TestProfileHelpers(object):
    def test_get_box_snr(self, profile_data, gaus_data_snr):
        snr = profile.get_box_snr(profile_data, on_pulse=(497, 504))
        np.testing.assert_allclose(snr, gaus_data_snr, atol=1)

    def test_get_template(self, profile_data, gaus_data_snr):
        width = 7
        temp = profile.get_template(width=width, kind="boxcar")
        assert len(temp.array) == width

    def test_matched_filter(self, profile_data):
        fit = profile.MatchedFilter(profile_data, widths=range(1, 32), kind="boxcar")
        assert fit.best_width == 3
        assert fit.peak_bin == 500
