import numpy as np
from numpy.typing import NDArray


class Rescale:
    """A class to rescale data to zero mean and unit variance.

    Parameters
    ----------
    tsamp : float
        sampling time, by default None
    nchans : int
        number of channels, by default None
    rescale_seconds : float, optional
        sample interval used for quantization in seconds, by default 10
    constant_offset_scale : bool, optional
        whether to use constant offset and scale, by default False
    """

    def __init__(
        self,
        tsamp: float,
        nchans: int,
        *,
        rescale_seconds: float = 10,
        constant_offset_scale: bool = False,
    ) -> None:
        self.tsamp = tsamp
        self.nchans = nchans
        self.rescale_seconds = rescale_seconds
        self.constant_offset_scale = constant_offset_scale

        self.first_call = True
        self._initialize_arr()

    @property
    def rescale_samples(self) -> int:
        return round(self.rescale_seconds / self.tsamp)

    def rescale(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        data = data.reshape(-1, self.nchans)

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
            self._initialize_arr()

    def _initialize_arr(self) -> None:
        self.sum_ar = np.zeros(self.nchans, dtype=np.float32)
        self.sumsq_ar = np.zeros(self.nchans, dtype=np.float32)
        self.scale = np.ones(self.nchans, dtype=np.float32)
        self.offset = np.zeros(self.nchans, dtype=np.float32)
        self.isample = 0
