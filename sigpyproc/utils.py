from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np
from astropy import units
from astropy.time import Time, TimeDelta
from rich.logging import RichHandler

if TYPE_CHECKING:
    import inspect


def detect_file_type(filename: str | Path) -> str:
    """
    Detect file type based on file extension.

    Supported extensions:

    - sigproc (.fil).
    - pfits (.fits, .sf).
    - fbh5 (.h5).

    Parameters
    ----------
    filename : str | Path
        File name to detect format for.

    Returns
    -------
    str
        File type name.
    """
    filename = Path(filename)
    ext = filename.suffix.lower()
    if ext == ".fil":
        return "sigproc"
    if ext in (".fits", ".sf"):
        return "pfits"
    if ext == ".h5":
        return "fbh5"
    msg = f"Unknown file format for file {filename}"
    raise ValueError(msg)


def nearest_factor(num: int, fac: int) -> int:
    """Calculate the factor of ``num`` closest to ``fac``.

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
    Find the next power of 2 greater than or equal to ``x``.

    Parameters
    ----------
    x : int
        Number to find the next power of 2 for.

    Returns
    -------
    int
        Next power of 2 greater than or equal to ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not positive.
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
        Logging level, by default `logging.INFO`.
    quiet : bool, optional
        If True set ``level`` as `logging.WARNING`, by default False.
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
    astropy.time.Time
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


def validate_path(
    path: str | Path,
    *,
    exists: bool = True,
    file_okay: bool = True,
    dir_okay: bool = False,
    readable: bool = True,
    writable: bool = False,
    resolve_path: bool = True,
) -> Path:
    """Validate a path based on various criteria.

    Parameters
    ----------
    path : str | Path
        Path to validate.
    exists : bool, optional
        Whether the path must exist, by default True.
    file_okay : bool, optional
        Whether a file path is acceptable, by default True.
    dir_okay : bool, optional
        Whether a directory path is acceptable, by default False.
    readable : bool, optional
        Whether the path must be readable, by default True.
    writable : bool, optional
        Whether the path must be writable, by default False.
    resolve_path : bool, optional
        Whether to resolve the path to an absolute path, by default True.

    Returns
    -------
    Path
        Validated path.

    Raises
    ------
    ValueError
        If neither ``file_okay`` nor ``dir_okay`` is True.
    FileNotFoundError
        If ``exists`` is True and the path does not exist.
    NotADirectoryError
        If a directory path is expected but a file path was found.
    IsADirectoryError
        If a file path is expected but a directory path was found.
    PermissionError
        If the path lacks the required permissions.
    """
    if not (file_okay or dir_okay):
        msg = "At least one of file_okay or dir_okay must be True."
        raise ValueError(msg)
    path = Path(path).resolve() if resolve_path else Path(path)
    if exists:
        if not path.exists():
            msg = f"Path does not exist: {path}"
            raise FileNotFoundError(msg)
        if path.is_file() and not file_okay:
            msg = f"Expected a directory but got a file: {path}"
            raise NotADirectoryError(msg)
        if path.is_dir() and not dir_okay:
            msg = f"Expected a file but got a directory: {path}"
            raise IsADirectoryError(msg)
        if readable or writable:
            mode = (os.R_OK if readable else 0) | (os.W_OK if writable else 0)
            if not os.access(path, mode):
                perms = []
                if readable and not os.access(path, os.R_OK):
                    perms.append("read")
                if writable and not os.access(path, os.W_OK):
                    perms.append("write")
                msg = f"Path {path} lacks {' and '.join(perms)} permission(s)."
                raise PermissionError(msg)
    return path


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class FrequencyChannels:
    """A class to handle frequency channels.

    Parameters
    ----------
    freqs : ArrayLike
        Central frequencies of the channels in MHz.

    Attributes
    ----------
    freqs : ndarray
        Central frequencies of the channels in MHz.
    array : ~astropy.units.Quantity
        Frequency array in Astropy Quantity format.

    nchans
    fch1
    foff
    ftop
    fcenter
    fbottom
    bandwidth
    """

    freqs: np.ndarray = attrs.field(converter=np.asarray)
    array: units.Quantity = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "array", units.Quantity(self.freqs, units.MHz))

    @freqs.validator
    def _check_freqs(self, attribute: attrs.Attribute, value: np.ndarray) -> None:
        attr_name = attribute.name
        if len(value) == 0:
            msg = f"{attr_name} must not be empty."
            raise ValueError(msg)
        diff = np.diff(value)
        if not np.all(np.isclose(diff, diff[0])):
            msg = f"{attr_name} must have a constant difference between elements."
            raise ValueError(msg)

    @property
    def nchans(self) -> int:
        """Number of frequency channels.

        Returns
        -------
        int
            Number of channels.
        """
        return len(self.array)

    @property
    def fch1(self) -> units.Quantity:
        """Central frequency of the first channel.

        Returns
        -------
        ~astropy.units.Quantity
            Central frequency of the first channel in MHz.
        """
        return self.array[0]

    @property
    def foff(self) -> units.Quantity:
        """Frequency offset between channels.

        Returns
        -------
        ~astropy.units.Quantity
            Frequency offset between channels in MHz.
        """
        return self.array[1] - self.array[0]

    @property
    def ftop(self) -> units.Quantity:
        """Edge frequency of the top channel.

        Returns
        -------
        ~astropy.units.Quantity
            Edge frequency of the top channel in MHz.
        """
        return self.fch1 - 0.5 * self.foff

    @property
    def fcenter(self) -> units.Quantity:
        """Central frequency of the entire band.

        Returns
        -------
        ~astropy.units.Quantity
            Central frequency of the entire band in MHz.
        """
        return self.ftop + 0.5 * self.foff * self.nchans

    @property
    def fbottom(self) -> units.Quantity:
        """Edge frequency of the bottom channel.

        Returns
        -------
        ~astropy.units.Quantity
            Edge frequency of the bottom channel in MHz.
        """
        return self.ftop + self.foff * self.nchans

    @property
    def bandwidth(self) -> units.Quantity:
        """Bandwidth of the entire band.

        Returns
        -------
        ~astropy.units.Quantity
            Bandwidth in MHz.
        """
        return abs(self.foff) * self.nchans

    @classmethod
    def from_sig(cls, fch1: float, foff: float, nchans: int) -> FrequencyChannels:
        """Create from sigproc parameters.

        Parameters
        ----------
        fch1 : float
            Central frequency of the first channel.
        foff : float
            Frequency offset between channels.
        nchans : int
            Number of frequency channels.

        Returns
        -------
        FrequencyChannels
            FrequencyChannels object.
        """
        array = np.arange(nchans, dtype=np.float64) * foff + fch1
        return cls(array)

    @classmethod
    def from_pfits(
        cls,
        fcenter: float,
        bandwidth: float,
        nchans: int,
    ) -> FrequencyChannels:
        """Create from pfits parameters.

        Parameters
        ----------
        fcenter : float
            Central frequency of the band.
        bandwidth : float
            Total bandwidth in MHz.
        nchans : int
            Number of frequency channels.

        Returns
        -------
        FrequencyChannels
            FrequencyChannels object.
        """
        foff = bandwidth / nchans
        fch1 = fcenter - 0.5 * foff * (nchans - 1)
        array = np.arange(nchans, dtype=np.float64) * foff + fch1
        return cls(array)
