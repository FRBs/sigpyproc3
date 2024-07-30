import numpy as np
import pytest
import scipy

from sigpyproc.core import stats


@pytest.fixture(scope="module", autouse=True)
def random_normal() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=5, scale=2, size=1000)


@pytest.fixture(scope="module", autouse=True)
def skewed_normal() -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 900), rng.normal(10, 1, 100)])


@pytest.fixture(scope="module", autouse=True)
def filter_samples() -> np.ndarray:
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture(scope="module", autouse=True)
def random_normal_2d() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=5, scale=2, size=(10, 1000)).astype(np.float32)


class TestEstimateLoc:
    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_estimate_loc_methods(
        self,
        random_normal: np.ndarray,
        method: str,
    ) -> None:
        np.testing.assert_almost_equal(
            5,
            stats.estimate_loc(random_normal, method=method),
            decimal=1,
        )

    def test_estimate_loc_empty(self) -> None:
        with pytest.raises(ValueError):
            stats.estimate_loc(np.array([]))

    def test_estimate_loc_invalid_method(self, random_normal: np.ndarray) -> None:
        with pytest.raises(ValueError):
            stats.estimate_loc(random_normal, method="invalid")

    def test_estimate_loc_single_element(self) -> None:
        assert stats.estimate_loc(np.array([42])) == 42


class TestEstimateScale:
    @pytest.mark.parametrize(
        "method",
        ["iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    def test_estimate_scale_methods(
        self,
        random_normal: np.ndarray,
        method: str,
    ) -> None:
        np.testing.assert_allclose(
            2,
            stats.estimate_scale(random_normal, method=method),
            atol=0.3,
        )

    def test_estimate_scale_double_mad(self, random_normal: np.ndarray) -> None:
        scale = stats.estimate_scale(random_normal, method="doublemad")
        assert isinstance(scale, np.ndarray)
        assert scale.shape == random_normal.shape
        np.testing.assert_allclose(
            2,
            scale.mean(),
            atol=0.3,
        )

    @pytest.mark.parametrize(
        "method",
        ["iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    def test_estimate_scale_skewed(
        self,
        skewed_normal: np.ndarray,
        method: str,
    ) -> None:
        scale = stats.estimate_scale(skewed_normal, method=method)
        assert scale > 0
        assert scale < 10

    def test_estimate_scale_skewed_double_mad(self, skewed_normal: np.ndarray) -> None:
        scale = stats.estimate_scale(skewed_normal, method="doublemad")
        assert isinstance(scale, np.ndarray)
        assert scale.shape == skewed_normal.shape
        assert scale.mean() > 0
        assert scale.mean() < 10

    def test_estimate_scale_empty(self) -> None:
        with pytest.raises(ValueError):
            stats.estimate_scale(np.array([]))

    def test_estimate_scale_invalid_method(self, random_normal: np.ndarray) -> None:
        with pytest.raises(ValueError):
            stats.estimate_scale(random_normal, method="invalid")

    def test_estimate_scale_single_element(self) -> None:
        assert stats.estimate_scale(np.array([42])) == 0

    def test_estimate_scale_constant_array(self) -> None:
        assert stats.estimate_scale(np.full(100, 5)) == 0


class TestZScore:
    @pytest.mark.parametrize("loc_method", ["mean", "median"])
    @pytest.mark.parametrize(
        "scale_method",
        ["iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    def test_zscore_methods(
        self,
        random_normal: np.ndarray,
        loc_method: str,
        scale_method: str,
    ) -> None:
        zscore_re = stats.zscore(
            random_normal,
            loc_method=loc_method,
            scale_method=scale_method,
        )
        assert isinstance(zscore_re, stats.ZScoreResult)
        np.testing.assert_almost_equal(5, zscore_re.loc, decimal=1)
        np.testing.assert_allclose(2, zscore_re.scale, atol=0.3)
        assert zscore_re.zscores.shape == random_normal.shape
        np.testing.assert_almost_equal(0, zscore_re.zscores.mean(), decimal=1)
        np.testing.assert_almost_equal(1, zscore_re.zscores.std(), decimal=1)

    def test_zscore_double_mad(self, random_normal: np.ndarray) -> None:
        zscore_re = stats.zscore(random_normal, scale_method="doublemad")
        assert isinstance(zscore_re, stats.ZScoreResult)
        np.testing.assert_almost_equal(5, zscore_re.loc, decimal=1)
        assert isinstance(zscore_re.scale, np.ndarray)
        np.testing.assert_allclose(2, zscore_re.scale.mean(), atol=0.3)
        assert zscore_re.scale.shape == random_normal.shape
        assert zscore_re.zscores.shape == random_normal.shape
        np.testing.assert_almost_equal(0, zscore_re.zscores.mean(), decimal=1)
        np.testing.assert_almost_equal(1, zscore_re.zscores.std(), decimal=1)

    def test_zscore_empty(self) -> None:
        with pytest.raises(ValueError):
            stats.zscore(np.array([]))

    def test_zscore_invalid(self, random_normal: np.ndarray) -> None:
        with pytest.raises(ValueError):
            stats.zscore(random_normal, loc_method="invalid")
        with pytest.raises(ValueError):
            stats.zscore(random_normal, scale_method="invalid")


class TestRunningFilter:
    def test_running_mean(self, filter_samples: np.ndarray) -> None:
        result = stats.running_filter(filter_samples, 3, "mean")
        expected = np.array([1.3, 2, 3, 4, 5, 6, 7, 8, 9, 9.7])
        np.testing.assert_almost_equal(result, expected, decimal=1)

    def test_running_median(self, filter_samples: np.ndarray) -> None:
        result = stats.running_filter(filter_samples, 3, "median")
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_almost_equal(result, expected)

    def test_running_even_window(self, filter_samples: np.ndarray) -> None:
        result = stats.running_filter(filter_samples, 4, "mean")
        expected = np.array([1.5, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.25])
        np.testing.assert_almost_equal(result, expected)

    def test_runnin_window_size_one(self, filter_samples: np.ndarray) -> None:
        result = stats.running_filter(filter_samples, 1, "mean")
        np.testing.assert_almost_equal(result, filter_samples)

    def test_running_invalid_method(self, filter_samples: np.ndarray) -> None:
        with pytest.raises(ValueError):
            stats.running_filter(filter_samples, 3, "invalid")

    def test_running_empty(self) -> None:
        with pytest.raises(ValueError):
            stats.running_filter(np.array([]), 3, "mean")


class TestChannelStats:
    nchans = 10
    nsamps = 1000
    deimal_precision = 3

    def test_initialization(self) -> None:
        chan_stats = stats.ChannelStats(nchans=self.nchans, nsamps=self.nsamps)
        assert chan_stats.nchans == self.nchans
        assert chan_stats.nsamps == self.nsamps
        assert chan_stats.moments.shape == (self.nchans,)

    @pytest.mark.parametrize("mode", ["basic", "advanced"])
    def test_push_data_basic(self, random_normal_2d: np.ndarray, mode: str) -> None:
        chan_stats = stats.ChannelStats(nchans=self.nchans, nsamps=self.nsamps)
        chan_stats.push_data(random_normal_2d.T.ravel(), start_index=0, mode=mode)
        assert np.all(chan_stats.mean > 0)
        assert np.all(chan_stats.var > 0)

    def test_properties(self, random_normal_2d: np.ndarray) -> None:
        chan_stats = stats.ChannelStats(nchans=self.nchans, nsamps=self.nsamps)
        chan_stats.push_data(random_normal_2d.T.ravel(), start_index=0, mode="advanced")
        np.testing.assert_almost_equal(
            chan_stats.mean,
            np.mean(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.var,
            np.var(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.std,
            np.std(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.maxima,
            np.max(random_normal_2d, axis=1),
        )
        np.testing.assert_almost_equal(
            chan_stats.minima,
            np.min(random_normal_2d, axis=1),
        )
        np.testing.assert_almost_equal(
            chan_stats.skew,
            scipy.stats.skew(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.kurtosis,
            scipy.stats.kurtosis(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )

    def test_addition(self, random_normal_2d: np.ndarray) -> None:
        chan_stats1 = stats.ChannelStats(nchans=self.nchans, nsamps=self.nsamps // 2)
        chan_stats2 = stats.ChannelStats(nchans=self.nchans, nsamps=self.nsamps // 2)
        chan_stats1.push_data(
            random_normal_2d[:, : self.nsamps // 2].T.ravel(),
            start_index=0,
            mode="advanced",
        )
        chan_stats2.push_data(
            random_normal_2d[:, self.nsamps // 2 :].T.ravel(),
            start_index=0,
            mode="advanced",
        )
        chan_stats = chan_stats1 + chan_stats2
        assert chan_stats.nchans == self.nchans
        assert chan_stats.nsamps == self.nsamps
        np.testing.assert_almost_equal(
            chan_stats.mean,
            np.mean(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.var,
            np.var(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.maxima,
            np.max(random_normal_2d, axis=1),
        )
        np.testing.assert_almost_equal(
            chan_stats.minima,
            np.min(random_normal_2d, axis=1),
        )

    def test_addition_invalid(self, random_normal_2d: np.ndarray) -> None:
        chan_stats = stats.ChannelStats(nchans=self.nchans, nsamps=self.nsamps)
        with pytest.raises(TypeError):
            chan_stats + random_normal_2d

    def test_push_data_stream(self, random_normal_2d: np.ndarray) -> None:
        chan_stats = stats.ChannelStats(nchans=self.nchans, nsamps=self.nsamps)
        for i in range(0, self.nsamps, 100):
            chan_stats.push_data(
                random_normal_2d[:, i : i + 100].T.ravel(),
                start_index=i,
            )
        np.testing.assert_almost_equal(
            chan_stats.mean,
            np.mean(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.var,
            np.var(random_normal_2d, axis=1),
            decimal=self.deimal_precision,
        )

    def test_constant_data(self) -> None:
        chan_stats = stats.ChannelStats(nchans=self.nchans, nsamps=self.nsamps)
        chan_stats.push_data(np.full((5, 100), 5).ravel(), start_index=0)
        np.testing.assert_almost_equal(
            chan_stats.mean,
            np.full(10, 5),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.var,
            np.zeros(10),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.std,
            np.zeros(10),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.skew,
            np.zeros(10),
            decimal=self.deimal_precision,
        )
        np.testing.assert_almost_equal(
            chan_stats.kurtosis,
            np.full(10, -3),
            decimal=self.deimal_precision,
        )
