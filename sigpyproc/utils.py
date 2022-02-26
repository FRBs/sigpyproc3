from __future__ import annotations
import numpy as np
import logging

from numpy import typing as npt

from datetime import datetime
from rich.text import Text
from rich.logging import RichHandler
from astropy import units
from astropy.time import Time, TimeDelta


class AttrDict(dict):  # noqa:WPS600
    def __init__(self, *args, **kwargs) -> None:
        """Dict class to expose keys as attributes."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def roll_array(arr: npt.ArrayLike, shift: int, axis: int = 0) -> np.ndarray:
    """Roll the elements in the array by `shift` positions along the given axis.

    Parameters
    ----------
    arr : :py:obj:`~numpy.typing.ArrayLike`
        input array to roll
    shift : int
        number of bins to shift by
    axis : int
        axis to roll along, by default 0

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        shifted numpy array
    """
    arr = np.asanyarray(arr)
    arr_size = arr.shape[axis]
    shift %= arr_size
    return arr.take(np.concatenate((np.arange(shift, arr_size), np.arange(shift))), axis)


def nearest_factor(num: int, fac: int) -> int:
    """Find nearest factor Calculates the factor of `num`, which is closest to `fac`.

    Parameters
    ----------
    num : int
        number that we wish to factor
    fac : int
        number around which we wish to find factor

    Returns
    -------
    int
        nearest factor
    """
    factors = {
        factor
        for check in range(1, int(num**0.5) + 1)
        for factor in (check, num // check)
        if num % check == 0
    }
    return min(factors, key=lambda val: abs(val - fac))


def get_logger(
    name: str, level: int | str = logging.INFO, quiet: bool = False
) -> logging.Logger:
    """Get a fancy logging utility using Rich library.

    Parameters
    ----------
    name : str
        logger name
    level : int or str, optional
        logging level, by default logging.INFO
    quiet : bool, optional
        if True set `level` as logging.WARNING, by default False

    Returns
    -------
    logging.Logger
        a logging object
    """
    logger = logging.getLogger(name)
    if quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(level)

    logformat = "- %(name)s - %(message)s"
    formatter = logging.Formatter(fmt=logformat)

    if not logger.hasHandlers():
        handler = RichHandler(
            show_level=False,
            show_path=False,
            rich_tracebacks=True,
            log_time_format=_time_formatter,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _time_formatter(timestamp: datetime) -> Text:
    return Text(timestamp.isoformat(sep=" ", timespec="milliseconds"))


def time_after_nsamps(tstart: float, tsamp: float, nsamps: int = 0) -> Time:
    """Get precise time nsamps after input tstart. If nsamps is not given then just return tstart.

    Parameters
    ----------
    tstart : float
        starting mjd.
    tsamp : float
        sampling time in seconds.
    nsamps : int, optional
        number of samples, by default 0

    Returns
    -------
    :class:`~astropy.time.Time`
        Astropy Time object after given nsamps
    """
    precision = int(np.ceil(abs(np.log10(tsamp))))
    tstart = Time(tstart, format="mjd", scale="utc", precision=precision)
    return tstart + TimeDelta(nsamps * tsamp, format="sec")


def duration_string(duration: float) -> str:
    """Convert duration in seconds to human readable string.

    Parameters
    ----------
    duration : float
        duration in seconds

    Returns
    -------
    str
        human readable duration string
    """
    if duration < 60:
        return f"{duration:.1f} seconds"
    elif duration < 3600:
        return f"{duration / 60:.1f} minutes"
    elif duration < 86400:
        return f"{duration / 3600:.1f} hours"
    return f"{duration / 86400:.1f} days"


class FrequencyChannels(object):
    """FrequencyChannels class to handle frequency channels.

    Parameters
    ----------
    freqs : :py:obj:`~numpy.typing.ArrayLike`
        array of frequencies
    """

    def __init__(self, freqs: npt.ArrayLike) -> None:
        freqs = np.asanyarray(freqs, dtype=np.float64)
        self._check_freqs(freqs)
        self._array = units.Quantity(freqs, units.MHz, copy=False)
        self._fch1 = self._array[0]
        self._nchans = len(self._array)
        self._foff = self._array[1] - self._array[0]

    @property
    def array(self) -> np.ndarray:
        """:py:obj:`~numpy.ndarray`: Get the frequency array."""
        return self._array

    @property
    def nchans(self) -> int:
        """float: Number of channels."""
        return self._nchans

    @property
    def fch1(self) -> float:
        """float: Central frequency of the first channel."""
        return self._fch1

    @property
    def foff(self) -> float:
        """float: Channel width."""
        return self._foff

    @property
    def ftop(self) -> float:
        """float: Frequency (edge) of the top channel."""
        return self.fch1 - 0.5 * self.foff

    @property
    def fcenter(self) -> float:
        """float: Central frequency of the whole band."""
        return self.ftop + 0.5 * self.foff * self.nchans

    @property
    def fbottom(self) -> float:
        """float: Frequency (edge) of the bottom channel."""
        return self.ftop + self.foff * self.nchans

    @property
    def bandwidth(self) -> float:
        """float: Bandwidth."""
        return abs(self.foff) * self.nchans

    @classmethod
    def from_sig(cls, fch1: float, foff: float, nchans: int) -> FrequencyChannels:
        array = np.arange(nchans, dtype=np.float64) * foff + fch1
        return cls(array)

    @classmethod
    def from_pfits(
        cls, fcenter: float, bandwidth: float, nchans: int
    ) -> FrequencyChannels:
        foff = bandwidth / nchans
        fch1 = fcenter - 0.5 * foff * (nchans - 1)
        array = np.arange(nchans, dtype=np.float64) * foff + fch1
        return cls(array)

    def _check_freqs(self, freqs) -> None:
        assert len(freqs) > 0, "Frequency array cannot be empty."
        diff = np.diff(freqs)
        assert np.all(np.isclose(diff, diff[0])), "Frequencies must be equally spaced."
