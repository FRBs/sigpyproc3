import numpy as np

from sigpyproc.core import stats


class TestStats:
    def test_zscore_mad(self) -> None:
        input_arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        desired = np.array(
            [-1.01173463, -0.33724488, 0.33724488, 1.01173463],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(
            desired,
            stats.zscore_mad(input_arr),
            decimal=4,
        )

    def test_zscore_double_mad(self) -> None:
        input_arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        desired = np.array(
            [-1.01173463, -0.33724488, 0.33724488, 1.01173463],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(
            desired,
            stats.zscore_double_mad(input_arr),
            decimal=4,
        )
