from __future__ import annotations

import numpy as np
import pytest
import scipy

from sigpyproc.core import stats


class TestEstimateLoc:
    @pytest.mark.parametrize("method", ["mean", "median"])
    @pytest.mark.parametrize("axis", [None, 0, (0,)])
    def test_estimate_loc_1d(
        self,
        random_normal_1d: np.ndarray,
        method: str,
        axis: int | tuple[int, ...] | None,
    ) -> None:
        result = stats.estimate_loc(random_normal_1d, method=method, axis=axis)
        assert isinstance(result, np.floating)
        np.testing.assert_almost_equal(result, random_normal_1d.mean(), decimal=1)
        result = stats.estimate_loc(
            random_normal_1d,
            method=method,
            axis=axis,
            keepdims=True,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        np.testing.assert_almost_equal(result[0], random_normal_1d.mean(), decimal=1)

    @pytest.mark.parametrize("method", ["mean", "median"])
    @pytest.mark.parametrize("axis", [None, (0, 1)])
    def test_estimate_loc_2d(
        self,
        random_normal_2d: np.ndarray,
        method: str,
        axis: int | tuple[int, ...] | None,
    ) -> None:
        result = stats.estimate_loc(random_normal_2d, method=method, axis=axis)
        assert isinstance(result, np.floating)
        np.testing.assert_almost_equal(result, random_normal_2d.mean(), decimal=1)
        result = stats.estimate_loc(
            random_normal_2d,
            method=method,
            axis=axis,
            keepdims=True,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        np.testing.assert_almost_equal(result[0, 0], random_normal_2d.mean(), decimal=1)

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_estimate_loc_2d_axis(
        self,
        random_normal_2d: np.ndarray,
        method: str,
    ) -> None:
        result = stats.estimate_loc(random_normal_2d, method=method, axis=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1000,)
        np.testing.assert_almost_equal(
            result.mean(),
            random_normal_2d.mean(),
            decimal=1,
        )
        result = stats.estimate_loc(random_normal_2d, method=method, axis=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)
        np.testing.assert_almost_equal(
            result.mean(),
            random_normal_2d.mean(),
            decimal=1,
        )

    def test_estimate_loc_empty(self) -> None:
        with pytest.raises(ValueError):
            stats.estimate_loc(np.array([]))

    def test_estimate_loc_invalid_method(self, random_normal_1d: np.ndarray) -> None:
        with pytest.raises(ValueError):
            stats.estimate_loc(random_normal_1d, method="invalid")

    def test_estimate_loc_single_element(self) -> None:
        assert stats.estimate_loc(np.array([42])) == 42


class TestEstimateScale:
    @pytest.mark.parametrize(
        "method",
        ["std", "iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    @pytest.mark.parametrize("axis", [None, 0, (0,)])
    def test_estimate_scale_1d(
        self,
        random_normal_1d: np.ndarray,
        method: str,
        axis: int | tuple[int, ...] | None,
    ) -> None:
        result = stats.estimate_scale(random_normal_1d, method=method, axis=axis)
        assert isinstance(result, np.float64)
        np.testing.assert_almost_equal(result, random_normal_1d.std(), decimal=0)
        result = stats.estimate_scale(
            random_normal_1d,
            method=method,
            axis=axis,
            keepdims=True,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        np.testing.assert_almost_equal(result[0], random_normal_1d.std(), decimal=0)

    @pytest.mark.parametrize(
        "method",
        ["std", "iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    @pytest.mark.parametrize("axis", [None, (0, 1)])
    def test_estimate_scale_2d(
        self,
        random_normal_2d: np.ndarray,
        method: str,
        axis: int | tuple[int, ...] | None,
    ) -> None:
        result = stats.estimate_scale(random_normal_2d, method=method, axis=axis)
        assert isinstance(result, np.floating)
        np.testing.assert_almost_equal(result, random_normal_2d.std(), decimal=1)
        result = stats.estimate_scale(
            random_normal_2d,
            method=method,
            axis=axis,
            keepdims=True,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        np.testing.assert_almost_equal(result[0, 0], random_normal_2d.std(), decimal=1)

    @pytest.mark.parametrize(
        "method",
        ["std", "iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    def test_estimate_scale_2d_axis(
        self,
        random_normal_2d: np.ndarray,
        method: str,
    ) -> None:
        result = stats.estimate_scale(random_normal_2d, method=method, axis=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1000,)
        result = stats.estimate_scale(random_normal_2d, method=method, axis=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)
        np.testing.assert_almost_equal(result.mean(), random_normal_2d.std(), decimal=1)

    def test_estimate_scale_double_mad(self, random_normal_1d: np.ndarray) -> None:
        scale = stats.estimate_scale(random_normal_1d, method="doublemad")
        assert isinstance(scale, np.ndarray)
        assert scale.shape == random_normal_1d.shape
        np.testing.assert_almost_equal(scale.mean(), random_normal_1d.std(), decimal=1)

    @pytest.mark.parametrize(
        "method",
        ["std", "iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    def test_estimate_scale_skewed(
        self,
        skewed_normal_1d: np.ndarray,
        method: str,
    ) -> None:
        scale = stats.estimate_scale(skewed_normal_1d, method=method)
        assert scale > 0
        assert scale < 10

    def test_estimate_scale_skewed_double_mad(
        self,
        skewed_normal_1d: np.ndarray,
    ) -> None:
        scale = stats.estimate_scale(skewed_normal_1d, method="doublemad")
        assert isinstance(scale, np.ndarray)
        assert scale.shape == skewed_normal_1d.shape
        assert scale.mean() > 0
        assert scale.mean() < 10

    def test_estimate_scale_empty(self) -> None:
        with pytest.raises(ValueError):
            stats.estimate_scale(np.array([]))

    def test_estimate_scale_invalid_method(self, random_normal_1d: np.ndarray) -> None:
        with pytest.raises(ValueError):
            stats.estimate_scale(random_normal_1d, method="invalid")

    def test_estimate_scale_single_element(self) -> None:
        assert stats.estimate_scale(np.array([42])) == 0

    def test_estimate_scale_constant_array(self) -> None:
        assert stats.estimate_scale(np.full(100, 5)) == 0


class TestZScore:
    @pytest.mark.parametrize("loc_method", ["mean", "median"])
    @pytest.mark.parametrize(
        "scale_method",
        ["std", "iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    def test_zscore_1d(
        self,
        random_normal_1d: np.ndarray,
        loc_method: str,
        scale_method: str,
    ) -> None:
        zscore_re = stats.estimate_zscore(
            random_normal_1d,
            loc_method=loc_method,
            scale_method=scale_method,
        )
        assert isinstance(zscore_re, stats.ZScoreResult)
        assert isinstance(zscore_re.loc, np.ndarray)
        assert isinstance(zscore_re.scale, np.ndarray)
        np.testing.assert_almost_equal(
            zscore_re.loc,
            random_normal_1d.mean(),
            decimal=1,
        )
        np.testing.assert_almost_equal(
            zscore_re.scale,
            random_normal_1d.std(),
            decimal=0,
        )
        assert zscore_re.data.shape == random_normal_1d.shape
        np.testing.assert_almost_equal(zscore_re.data.mean(), 0, decimal=1)
        np.testing.assert_almost_equal(zscore_re.data.std(), 1, decimal=1)

    @pytest.mark.parametrize("loc_method", ["mean", "median"])
    @pytest.mark.parametrize(
        "scale_method",
        ["std", "iqr", "mad", "diffcov", "biweight", "qn", "sn", "gapper"],
    )
    def test_zscore_2d(
        self,
        random_normal_2d: np.ndarray,
        loc_method: str,
        scale_method: str,
    ) -> None:
        zscore_re = stats.estimate_zscore(
            random_normal_2d,
            loc_method=loc_method,
            scale_method=scale_method,
            axis=1,
        )
        assert isinstance(zscore_re, stats.ZScoreResult)
        assert isinstance(zscore_re.loc, np.ndarray)
        assert isinstance(zscore_re.scale, np.ndarray)
        np.testing.assert_almost_equal(
            zscore_re.loc,
            random_normal_2d.mean(),
            decimal=1,
        )
        np.testing.assert_almost_equal(
            zscore_re.scale,
            random_normal_2d.std(),
            decimal=0,
        )
        assert zscore_re.loc.shape == (10, 1)
        assert zscore_re.scale.shape == (10, 1)
        assert zscore_re.data.shape == random_normal_2d.shape
        np.testing.assert_almost_equal(zscore_re.data.mean(), 0, decimal=1)
        np.testing.assert_almost_equal(zscore_re.data.std(), 1, decimal=1)

    def test_zscore_double_mad(self, random_normal_1d: np.ndarray) -> None:
        zscore_re = stats.estimate_zscore(random_normal_1d, scale_method="doublemad")
        assert isinstance(zscore_re, stats.ZScoreResult)
        np.testing.assert_almost_equal(
            zscore_re.loc,
            random_normal_1d.mean(),
            decimal=1,
        )
        assert isinstance(zscore_re.scale, np.ndarray)
        np.testing.assert_almost_equal(
            zscore_re.scale.mean(),
            random_normal_1d.std(),
            decimal=0,
        )
        assert zscore_re.scale.shape == random_normal_1d.shape
        assert zscore_re.data.shape == random_normal_1d.shape
        np.testing.assert_almost_equal(zscore_re.data.mean(), 0, decimal=1)
        np.testing.assert_almost_equal(zscore_re.data.std(), 1, decimal=1)

    def test_zscore_constant_array(self) -> None:
        data = np.full(100, 5)
        zscore_re = stats.estimate_zscore(data)
        assert isinstance(zscore_re, stats.ZScoreResult)
        np.testing.assert_almost_equal(zscore_re.loc, 5, decimal=1)
        np.testing.assert_almost_equal(zscore_re.scale, 1, decimal=1)
        np.testing.assert_array_equal(zscore_re.data, np.zeros_like(data))

    def test_zscore_empty(self) -> None:
        with pytest.raises(ValueError):
            stats.estimate_zscore(np.array([]))

    def test_zscore_invalid(self, random_normal_1d: np.ndarray) -> None:
        with pytest.raises(ValueError):
            stats.estimate_zscore(random_normal_1d, loc_method="invalid")
        with pytest.raises(ValueError):
            stats.estimate_zscore(random_normal_1d, scale_method="invalid")


class TestRunningFilter:
    filter_samples = np.arange(1, 11)

    def test_running_mean(self) -> None:
        result = stats.running_filter(self.filter_samples, 3, "mean")
        expected = np.array([1.3, 2, 3, 4, 5, 6, 7, 8, 9, 9.7])
        np.testing.assert_almost_equal(result, expected, decimal=1)
        result_fast = stats.running_filter_fast(self.filter_samples, 3, "mean")
        np.testing.assert_almost_equal(result_fast, expected, decimal=1)

    def test_running_median(self) -> None:
        result = stats.running_filter(self.filter_samples, 3, "median")
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_almost_equal(result, expected)

    def test_running_even_window(self) -> None:
        result = stats.running_filter(self.filter_samples, 4, "mean")
        expected = np.array([1.5, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.25])
        np.testing.assert_almost_equal(result, expected)

    def test_runnin_window_size_one(self) -> None:
        result = stats.running_filter(self.filter_samples, 1, "mean")
        np.testing.assert_almost_equal(result, self.filter_samples)

    def test_running_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            stats.running_filter(self.filter_samples, 3, "invalid")

    def test_running_empty(self) -> None:
        with pytest.raises(ValueError):
            stats.running_filter(np.array([]), 3, "mean")

    def test_running_fast(self, random_normal_1d: np.ndarray) -> None:
        result = stats.running_filter_fast(random_normal_1d, 250, "mean")
        assert result.shape == random_normal_1d.shape


class TestDownsample:
    @pytest.mark.parametrize("factor", [1, 2, 4, 7, 10])
    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_downsample_1d(
        self,
        random_normal_1d: np.ndarray,
        factor: int,
        method: str,
    ) -> None:
        np_op = getattr(np, method)
        nsamps_new = (random_normal_1d.size // factor) * factor
        expected = np_op(random_normal_1d[:nsamps_new].reshape(-1, factor), axis=1)
        result = stats.downsample_1d(random_normal_1d, factor, method=method)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_test_downsample_1d_fail(
        self,
        random_normal_1d: np.ndarray,
        random_normal_2d: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError):
            stats.downsample_1d(random_normal_1d, 0)
        with pytest.raises(ValueError):
            stats.downsample_1d(random_normal_1d, 5.6)
        with pytest.raises(ValueError):
            stats.downsample_1d(5, 5)
        with pytest.raises(ValueError):
            stats.downsample_1d(random_normal_2d, 5)
        with pytest.raises(ValueError):
            stats.downsample_1d(random_normal_1d, len(random_normal_1d) + 1)
        with pytest.raises(ValueError):
            stats.downsample_1d(random_normal_1d, 5, method="invalid")

    @pytest.mark.parametrize(
        ("factor1", "factor2"),
        [(1, 1), (1, 3), (2, 6), (7, 7), (10, 23)],
    )
    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_downsample_2d(
        self,
        random_normal_2d: np.ndarray,
        factor1: int,
        factor2: int,
        method: str,
    ) -> None:
        np_op = getattr(np, method)
        dim1, dim2 = random_normal_2d.shape
        new_dim1 = dim1 // factor1
        new_dim2 = dim2 // factor2
        new_shape = (new_dim1, factor1, new_dim2, factor2)
        expected = np_op(
            random_normal_2d[: new_dim1 * factor1, : new_dim2 * factor2].reshape(
                new_shape,
            ),
            axis=(1, 3),
        )
        result_flat = stats.downsample_2d_flat(
            random_normal_2d.ravel(),
            factor1,
            factor2,
            dim1,
            dim2,
            method=method,
        )
        np.testing.assert_array_almost_equal(result_flat, expected.ravel(), decimal=5)
        result = stats.downsample_2d(
            random_normal_2d,
            (factor1, factor2),
            method=method,
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_downsample_2d_fail(self, random_normal_2d: np.ndarray) -> None:
        with pytest.raises(ValueError):
            stats.downsample_2d(random_normal_2d.ravel(), (1, 1))
        with pytest.raises(ValueError):
            stats.downsample_2d(random_normal_2d, (0, 1))
        with pytest.raises(ValueError):
            stats.downsample_2d(random_normal_2d, (1, 0))
        with pytest.raises(ValueError):
            stats.downsample_2d(random_normal_2d, (1, 1), method="invalid")

    def test_downsample_2d_flat_fail(self, random_normal_2d: np.ndarray) -> None:
        arr = random_normal_2d.ravel()
        dim1, dim2 = random_normal_2d.shape
        with pytest.raises(ValueError):
            stats.downsample_2d_flat(random_normal_2d, 1, 1, dim1, dim2)
        with pytest.raises(ValueError):
            stats.downsample_2d_flat(arr, 0, 1, dim1, dim2)
        with pytest.raises(ValueError):
            stats.downsample_2d_flat(arr, 1, 0, dim1, dim2)
        with pytest.raises(ValueError):
            stats.downsample_2d_flat(arr, 1, 1, dim1, dim2 + 1)
        with pytest.raises(ValueError):
            stats.downsample_2d_flat(arr, 1, 1, dim1, dim2, method="invalid")


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
