from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units
from astropy.time import Time, TimeDelta
from numpy import typing as npt
from rich.logging import RichHandler

if TYPE_CHECKING:
    import inspect


def roll_array(arr: npt.ArrayLike, shift: int, axis: int = 0) -> np.ndarray:
    """
    Roll the elements in the array by `shift` positions along the given axis.

    The shift direction is from the end towards the beginning of the axis,
    opposite to the shift direction of :py:func:`~numpy.roll`.

    Parameters
    ----------
    arr : ArrayLike
        Input array to roll.
    shift : int
        Number of bins to shift by.
    axis : int
        Axis to roll along, by default 0.

    Returns
    -------
    ndarray
        Shifted numpy array.
    """
    arr = np.asanyarray(arr)
    arr_size = arr.shape[axis]
    shift %= arr_size
    return arr.take(
        np.concatenate((np.arange(shift, arr_size), np.arange(shift))),
        axis,
    )


def nearest_factor(num: int, fac: int) -> int:
    """
    Find nearest factor Calculates the factor of `num`, which is closest to `fac`.

    Parameters
    ----------
    num : int
        Number that we wish to factor.
    fac : int
        Number around which we wish to find factor.

    Returns
    -------
    int
        Nearest factor.
    """
    factors = {
        factor
        for check in range(1, int(num**0.5) + 1)
        for factor in (check, num // check)
        if num % check == 0
    }
    return min(factors, key=lambda val: abs(val - fac))


def next2_to_n(x: int) -> int:
    """
    Find the next power of 2 greater than or equal to `x`.

    Parameters
    ----------
    x : int
        Number to find the next power of 2 for.

    Returns
    -------
    int
        Next power of 2 greater than or equal to `x`.

    Raises
    ------
    ValueError
        If `x` is not positive.
    """
    if x <= 0:
        msg = "Input must be positive."
        raise ValueError(msg)
    return 1 << (x - 1).bit_length()


def get_logger(
    name: str,
    *,
    level: int | str = logging.INFO,
    quiet: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Get a fancy configured logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int or str, optional
        Logging level, by default logging.INFO.
    quiet : bool, optional
        If True set `level` as logging.WARNING, by default False.
    log_file : str, optional
        Path to log file, by default None.

    Returns
    -------
    logging.Logger
        A logging object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING if quiet else level)
    logformat = "- %(name)s - %(message)s"
    formatter = logging.Formatter(fmt=logformat)
    if not logger.hasHandlers():
        handler = RichHandler(
            show_path=False,
            rich_tracebacks=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if log_file and not any(
        isinstance(hndlr, logging.FileHandler)
        and hndlr.baseFilename == Path(log_file).resolve().as_posix()
        for hndlr in logger.handlers
    ):
        file_handler = logging.FileHandler(Path(log_file).resolve().as_posix())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_callerfunc(stack: list[inspect.FrameInfo]) -> str:
    for i in range(len(stack)):
        if stack[i].function == "<module>":
            return stack[i - 1].function
    return stack[1].function


def time_after_nsamps(tstart: float, tsamp: float, nsamps: int = 0) -> Time:
    """
    Get time after given nsamps. If nsamps is not given then just return tstart.

    Parameters
    ----------
    tstart : float
        Starting mjd.
    tsamp : float
        Sampling time in seconds.
    nsamps : int, optional
        Number of samples, by default 0.

    Returns
    -------
    :class:`~astropy.time.Time`
        Astropy Time object after given nsamps.
    """
    precision = int(np.ceil(abs(np.log10(tsamp))))
    tstart = Time(tstart, format="mjd", scale="utc", precision=precision)
    return tstart + TimeDelta(nsamps * tsamp, format="sec")


def duration_string(duration: float) -> str:
    """
    Convert duration in seconds to human readable string.

    Parameters
    ----------
    duration : float
        Duration in seconds.

    Returns
    -------
    str
        Human readable duration string.
    """
    if duration < 60:
        return f"{duration:.1f} seconds"
    if duration < 3600:
        return f"{duration / 60:.1f} minutes"
    if duration < 86400:
        return f"{duration / 3600:.1f} hours"
    return f"{duration / 86400:.1f} days"


class FrequencyChannels:
    """
    FrequencyChannels class to handle frequency channels.

    Parameters
    ----------
    freqs : :py:obj:`~numpy.ndarray`
        Array of frequencies.
    """

    def __init__(self, freqs: np.ndarray) -> None:
        freqs = np.asanyarray(freqs, dtype=np.float64)
        self._check_freqs(freqs)
        self._array = units.Quantity(freqs, units.MHz, copy=False)
        self._fch1 = self._array[0]
        self._nchans = len(self._array)
        self._foff = self._array[1] - self._array[0]

    @property
    def array(self) -> units.Quantity:
        """:py:obj:`~numpy.ndarray`: Get the frequency array."""
        return self._array

    @property
    def nchans(self) -> int:
        """int: Number of channels."""
        return self._nchans

    @property
    def fch1(self) -> units.Quantity:
        """float: Central frequency of the first channel."""
        return self._fch1

    @property
    def foff(self) -> units.Quantity:
        """float: Channel width."""
        return self._foff

    @property
    def ftop(self) -> units.Quantity:
        """float: Frequency (edge) of the top channel."""
        return self.fch1 - 0.5 * self.foff

    @property
    def fcenter(self) -> units.Quantity:
        """float: Central frequency of the whole band."""
        return self.ftop + 0.5 * self.foff * self.nchans

    @property
    def fbottom(self) -> units.Quantity:
        """float: Frequency (edge) of the bottom channel."""
        return self.ftop + self.foff * self.nchans

    @property
    def bandwidth(self) -> units.Quantity:
        """float: Bandwidth."""
        return abs(self.foff) * self.nchans

    @classmethod
    def from_sig(cls, fch1: float, foff: float, nchans: int) -> FrequencyChannels:
        array = np.arange(nchans, dtype=np.float64) * foff + fch1
        return cls(array)

    @classmethod
    def from_pfits(
        cls,
        fcenter: float,
        bandwidth: float,
        nchans: int,
    ) -> FrequencyChannels:
        foff = bandwidth / nchans
        fch1 = fcenter - 0.5 * foff * (nchans - 1)
        array = np.arange(nchans, dtype=np.float64) * foff + fch1
        return cls(array)

    def _check_freqs(self, freqs: np.ndarray) -> None:
        if len(freqs) == 0:
            msg = "Frequency array empty."
            raise ValueError(msg)
        diff = np.diff(freqs)
        if not np.all(np.isclose(diff, diff[0])):
            msg = "Frequencies must be equally spaced."
            raise ValueError(msg)
