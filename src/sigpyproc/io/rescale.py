import numpy as np
from numpy.typing import NDArray


class Rescale:
    """A class to rescale streaming data to zero mean and unit variance.

    Parameters
    ----------
    nchans : int
        Number of channels in the data.
    rescale_samples : int, optional
        Number of samples (per channel) to use for stats computation, by default 16384.
    constant_offset_scale : bool, optional
        Whether to the offset/scale computed at first update for all, by default False.
    """

    def __init__(
        self,
        nchans: int,
        rescale_samples: int = 16384,
        *,
        constant_offset_scale: bool = False,
    ) -> None:
        self.nchans = nchans
        self.rescale_samples = rescale_samples
        self.constant_offset_scale = constant_offset_scale

        self.first_call = True
        self.sum_ar = np.zeros(self.nchans, dtype=np.float32)
        self.sumsq_ar = np.zeros(self.nchans, dtype=np.float32)
        self.scale = np.ones(self.nchans, dtype=np.float32)
        self.offset = np.zeros(self.nchans, dtype=np.float32)
        self.isample = 0

    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Rescale the data to zero mean and unit variance.

        Parameters
        ----------
        data : NDArray[float32]
            Flattened data array.

        Returns
        -------
        NDArray[float32]
            Rescaled data array.
        """
        if data.size % self.nchans != 0:
            msg = f"Data size {data.size} is not a multiple of nchans {self.nchans}"
            raise ValueError(msg)
        data = np.ascontiguousarray(data).reshape(-1, self.nchans)

        if not self.constant_offset_scale or self.first_call:
            self._compute_stats(data)

        normdata = (data + self.offset) * self.scale
        self.first_call = False
        return normdata.ravel()

    def _compute_stats(self, data: NDArray[np.float32]) -> None:
        _, nsamples = data.shape
        self.sum_ar += np.sum(data, axis=0)
        self.sumsq_ar += np.sum(data**2, axis=0)
        self.isample += nsamples

        if self.isample >= self.rescale_samples or self.first_call:
            mean = self.sum_ar / self.isample
            variance = self.sumsq_ar / self.isample - mean * mean
            self.offset = -mean
            self.scale = np.where(
                np.isclose(variance, 0, atol=1e-5),
                1,
                1.0 / np.sqrt(variance),
            )
