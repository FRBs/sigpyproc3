"""Data scaling and contrast normalization strategies for viz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from astropy.visualization import ZScaleInterval

from sigpyproc import kernels

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Scaler(Protocol):
    """Protocol for image data scalers."""

    name: str

    def prepare(self, data: np.ndarray) -> np.ndarray:
        """Prepare or transform the data array (e.g. per-channel normalize)."""
        ...

    def clim(self, data: np.ndarray) -> tuple[float, float]:
        """Compute (vmin, vmax) contrast limits for the array."""
        ...


class GlobalMinMax:
    """Plain min/max scaling."""

    name: str = "minmax"

    def prepare(self, data: np.ndarray) -> np.ndarray:
        return data

    def clim(self, data: np.ndarray) -> tuple[float, float]:
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return 0.0, 1.0
        return float(np.min(valid)), float(np.max(valid))


class ChannelZScore:
    """Robust per-channel median/MAD normalization."""

    name: str = "zscore"

    def prepare(self, data: np.ndarray) -> np.ndarray:
        if data.dtype == np.float32 and data.flags.c_contiguous:
            src = data
        else:
            src = np.ascontiguousarray(data, dtype=np.float32)
        out = np.empty_like(src)
        kernels.robust_chan_norm(src, out, 8)
        return out

    def clim(self, data: np.ndarray) -> tuple[float, float]:
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return -3.0, 5.0
        med = float(np.median(valid[:10000]))
        return med - 2.0, med + 5.0


class ZScale:
    """IRAF-style ZScale interval."""

    name: str = "zscale"

    def __init__(self, contrast: float = 0.25) -> None:
        self.contrast = contrast
        self._interval = ZScaleInterval(contrast=contrast)

    def prepare(self, data: np.ndarray) -> np.ndarray:
        return data

    def clim(self, data: np.ndarray) -> tuple[float, float]:
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return 0.0, 1.0
        sample: NDArray[np.floating] = (
            valid[:: max(1, valid.size // 20000)] if valid.size > 20000 else valid
        )
        try:
            vmin, vmax = self._interval.get_limits(sample)
            return float(vmin), float(vmax)
        except (ValueError, TypeError, ZeroDivisionError):
            return float(np.min(sample)), float(np.max(sample))


class Percentile:
    """Percentile-based clipping."""

    name: str = "percentile"

    def __init__(self, q_lo: float = 1.0, q_hi: float = 99.5) -> None:
        self.q_lo = q_lo
        self.q_hi = q_hi

    def prepare(self, data: np.ndarray) -> np.ndarray:
        return data

    def clim(self, data: np.ndarray) -> tuple[float, float]:
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return 0.0, 1.0
        sample = valid[:: max(1, valid.size // 20000)] if valid.size > 20000 else valid
        vlo, vhi = np.percentile(sample, [self.q_lo, self.q_hi])
        return float(vlo), float(vhi)


class SigmaClip:
    """Mean +/- N sigma clipping."""

    name: str = "clip"

    def __init__(self, n: float = 3.0) -> None:
        self.n = n

    def prepare(self, data: np.ndarray) -> np.ndarray:
        return data

    def clim(self, data: np.ndarray) -> tuple[float, float]:
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return 0.0, 1.0
        sample = valid[:: max(1, valid.size // 20000)] if valid.size > 20000 else valid
        mean = float(np.mean(sample))
        std = float(np.std(sample))
        return mean - self.n * std, mean + self.n * std


SCALERS: list[Scaler] = [
    ChannelZScore(),
    ZScale(),
    Percentile(1.0, 99.5),
    SigmaClip(3.0),
    GlobalMinMax(),
]


def frequency_oriented(
    image: np.ndarray,
    freqs: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return ``image`` with row 0 at the lowest frequency.

    Filterbank blocks store channel 0 at ``fch1``. When ``foff`` is negative
    that row is the top of the band. ``imshow(..., origin="lower")`` pins row 0
    to the bottom of the extent, so this flips that layout. The returned
    frequencies are ``(f_low, f_high)`` in MHz.

    Parameters
    ----------
    image : np.ndarray
        2-D array whose rows follow ``freqs``.
    freqs : np.ndarray
        Channel frequencies in MHz, one per row.

    Returns
    -------
    tuple of ndarray, float, float
        Oriented image, lowest frequency, highest frequency.
    """
    if freqs.size == 0:
        return image, 0.0, 1.0
    f_lo = float(np.min(freqs))
    f_hi = float(np.max(freqs))
    if float(freqs[0]) > float(freqs[-1]):
        return np.ascontiguousarray(image[::-1]), f_lo, f_hi
    return image, f_lo, f_hi
