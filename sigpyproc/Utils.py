import numpy as np
import logging

from numpy import typing as npt
from datetime import datetime
from rich.text import Text
from rich.logging import RichHandler
from typing import Union


def roll_array(arr: npt.ArrayLike, shift: int, axis: int) -> np.ndarray:
    """Roll the elements in the array by `shift` positions along the given axis.

    Parameters
    ----------
    arr : npt.ArrayLike
        input array to roll
    shift : int
        number of bins to shift by
    axis : int
        axis to roll along

    Returns
    -------
    np.ndarray
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
        for check in range(1, int(num ** 0.5) + 1)
        for factor in (check, num // check)
        if num % check == 0
    }
    return min(factors, key=lambda val: abs(val - fac))


def get_logger(
    name: str,
    level: Union[int, str] = logging.INFO,
    quiet: bool = False,
) -> logging.Logger:
    """Get a fancy logging utility using Rich library.

    Parameters
    ----------
    name : str
        logger name
    level : Union[int, str], optional
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
