import numpy as np

from sigpyproc.io.rescale import Rescale


class TestRescale:
    def test_counts_samples_not_channels(self) -> None:
        nchans = 4
        rng = np.random.default_rng(0)
        data = rng.normal(100.0, 10.0, (1000, nchans)).astype(np.float32)
        rescaler = Rescale(nchans, rescale_samples=500)
        out = rescaler.execute(data.ravel()).reshape(-1, nchans)
        assert rescaler.isample == 1000
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-3)
        np.testing.assert_allclose(out.std(axis=0), 1.0, rtol=1e-3)

    def test_constant_channel_is_not_scaled(self) -> None:
        data = np.ones((64, 2), dtype=np.float32)
        data[:, 1] = 5.0
        rescaler = Rescale(2)
        out = rescaler.execute(data.ravel()).reshape(-1, 2)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)
