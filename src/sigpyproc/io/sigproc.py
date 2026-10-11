"""Sigproc-style header parsing and editing.

This module contains functions for parsing and editing Sigproc-style headers.
"""

from __future__ import annotations

import logging
import math
import re
import struct
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from bidict import bidict

from sigpyproc import params
from sigpyproc.utils import validate_path

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class HeaderField:
    """Class to handle sigproc header schema fields.

    Attributes
    ----------
    fmt : struct.Struct or None
        Format struct; None for string fields.
    doc : str
        Human-readable description of the field.
    """

    fmt: struct.Struct | None  # None for string fields
    doc: str


_UINT = struct.Struct("<I")  # 4 bytes, unsigned int
_DOUBLE = struct.Struct("<d")  # 8 bytes, double
_BYTE = struct.Struct("b")  # 1 byte, signed byte

SIGPROC_SCHEMA: dict[str, HeaderField] = {
    "telescope_id": HeaderField(fmt=_UINT, doc="Numeric telescope identifier"),
    "machine_id": HeaderField(fmt=_UINT, doc="Numeric machine identifier"),
    "data_type": HeaderField(fmt=_UINT, doc="Numeric data type identifier"),
    "rawdatafile": HeaderField(fmt=None, doc="Name of the original data file"),
    "source_name": HeaderField(fmt=None, doc="Name of the observed source"),
    "barycentric": HeaderField(
        fmt=_UINT, doc="Whether the data are barycentric (1) or otherwise (0)"
    ),
    "pulsarcentric": HeaderField(
        fmt=_UINT, doc="Whether the data are pulsarcentric (1) or otherwise (0)"
    ),
    "az_start": HeaderField(
        fmt=_DOUBLE, doc="Telescope azimuth at start of scan (degrees)"
    ),
    "za_start": HeaderField(
        fmt=_DOUBLE, doc="Telescope zenith angle at start of scan (degrees)"
    ),
    "src_raj": HeaderField(
        fmt=_DOUBLE,
        doc="Right ascension (J2000) of source (HHMMSS.S)",
    ),
    "src_dej": HeaderField(
        fmt=_DOUBLE,
        doc="Declination of source (DDMMSS.S)",
    ),
    "tstart": HeaderField(fmt=_DOUBLE, doc="Time stamp (MJD) of first sample"),
    "tsamp": HeaderField(fmt=_DOUBLE, doc="Time interval between samples (s)"),
    "nbits": HeaderField(fmt=_UINT, doc="Number of bits per time sample"),
    "nsamples": HeaderField(
        fmt=_UINT, doc="Number of time samples in the data file (rarely used)"
    ),
    "fch1": HeaderField(
        fmt=_DOUBLE, doc="Centre frequency (MHz) of first filterbank channel"
    ),
    "foff": HeaderField(fmt=_DOUBLE, doc="Filterbank channel bandwidth (MHz)"),
    "nchans": HeaderField(fmt=_UINT, doc="Number of frequency channels"),
    "nifs": HeaderField(fmt=_UINT, doc="Number of seperate IF channels"),
    "refdm": HeaderField(fmt=_DOUBLE, doc="Reference Dispersion Measure (pc/cm^3)"),
    "period": HeaderField(fmt=_DOUBLE, doc="Pulsar folding period (s)"),
    "signed": HeaderField(fmt=_BYTE, doc="Sign convention of 8-bit data samples"),
    "ibeam": HeaderField(fmt=_UINT, doc="Beam number of the observation"),
    "nbeams": HeaderField(fmt=_UINT, doc="Number of beams in the observation"),
}
"""Header keys recognised by the sigproc package."""


telescope_ids = bidict(
    {
        "Fake": 0,
        "Arecibo": 1,
        "Ooty": 2,
        "Nancay": 3,
        "Parkes": 4,
        "Jodrell": 5,
        "GBT": 6,
        "GMRT": 7,
        "Effelsberg": 8,
        "Effelsberg LOFAR": 9,
        "SRT": 10,
        "LOFAR": 11,
        "VLA": 12,
        "CHIME": 20,
        "MWA": 30,
        "MeerKAT": 64,
        "NC": 40,
        "NGNC": 41,
    },
)
"""Telescope IDs recognised by the sigproc package."""

machine_ids = bidict(
    {
        "FAKE": 0,
        "PSPM": 1,
        "WAPP": 2,
        "AOFTM": 3,
        "BPP": 4,
        "OOTY": 5,
        "SCAMP": 6,
        "GMRTFB": 7,
        "PULSAR2000": 8,
        "PARSPEC": 9,
        "BPSR": 10,
        "COBALT": 11,
        "GMRTNEW": 14,
        "CHIME": 20,
        "MWA-VCS": 30,
        "MWAX-VCS": 31,
        "MWAX-RTB": 32,
        "ADU": 40,
        "iTPM": 41,
    },
)
"""Machine IDs recognised by the sigproc package."""


@dataclass(frozen=True, kw_only=True, slots=True)
class FileInfo:
    """Class to handle individual file information."""

    filename: str
    hdrlen: int
    datalen: int
    nsamples: int
    tstart: float
    tsamp: float

    @property
    def tend(self) -> float:
        """Get the end time of the file."""
        return self.tstart + ((self.nsamples - 1) * self.tsamp) / 86400

    @classmethod
    def from_dict(cls, info: dict[str, int | float | str]) -> FileInfo:
        return cls(
            filename=str(info["filename"]),
            hdrlen=int(info["hdrlen"]),
            datalen=int(info["datalen"]),
            nsamples=int(info["nsamples"]),
            tstart=float(info["tstart"]),
            tsamp=float(info["tsamp"]),
        )


@dataclass(slots=True)
class StreamInfo:
    """Class to handle stream information as a list of FileInfo objects."""

    entries: list[FileInfo] = field(default_factory=list)

    @property
    def cumsum_datalens(self) -> np.ndarray:
        """Get the cumulative sum of datalen for all entries."""
        return np.cumsum(self.get_info_list("datalen"))

    @property
    def time_gaps(self) -> np.ndarray:
        """Get the time gaps between files."""
        tstart = np.array(self.get_info_list("tstart"))
        tend = np.array(self.get_info_list("tend"))
        if len(tstart) == 1:
            return np.array([0])
        return (tstart[1:] - tend[:-1]) * 86400

    def add_entry(self, finfo: FileInfo) -> None:
        """Add a FileInfo entry to the StreamInfo object."""
        self.entries.append(finfo)

    def get_combined(self, key: str) -> int | float:
        """Get the combined value of a key for all entries."""
        return sum(self.get_info_list(key))

    def get_info_list(self, key: str) -> list:
        """Get list of values for a given key for all entries."""
        return [getattr(entry, key) for entry in self.entries]

    def check_contiguity(self) -> bool:
        """Check if the files in the stream are contiguous/sequential."""
        tsamp_list = self.get_info_list("tsamp")
        tsamp_check = len(set(tsamp_list)) == 1
        contiguous = np.allclose(self.time_gaps, tsamp_list[0], rtol=0.1)
        return tsamp_check and contiguous


# Keys that fix how the samples are laid out. Changing them in place would make the
# data unreadable, so they are refused. Use spp_digifil to change them instead.
LAYOUT_KEYS = frozenset({"nbits", "nchans", "nifs"})

_NBITS_VALUES = (1, 2, 4, 8, 16, 32)
_NIFS_VALUES = (1, 2, 4)
_FLAG_KEYS = frozenset({"barycentric", "pulsarcentric", "signed"})
_STRING_KEYS = frozenset({"source_name", "rawdatafile"})
_POSITIVE_KEYS = frozenset({"nchans", "fch1", "tsamp", "tstart"})
_NON_NEGATIVE_KEYS = frozenset(
    {"nsamples", "telescope_id", "machine_id", "ibeam", "nbeams", "period", "refdm"}
)
_ANGLE_LIMITS = {"az_start": (0, 360), "za_start": (0, 180)}


def _coerce_value(key: str, value: float | str | None) -> float | str:
    """Convert a user-supplied value to the type the key is stored as."""
    if value is None:
        msg = f"'{key}' needs a value, got None."
        raise ValueError(msg)
    field = SIGPROC_SCHEMA[key]
    if field.fmt is None:
        if not isinstance(value, str):
            msg = f"'{key}' is a string key; got {value!r}."
            raise TypeError(msg)
        return value
    if field.fmt is _DOUBLE:
        try:
            number = float(value)
        except (TypeError, ValueError):
            msg = f"'{key}' needs a number, got {value!r}."
            raise ValueError(msg) from None
        if not math.isfinite(number):
            msg = f"'{key}' must be finite, got {value!r}."
            raise ValueError(msg)
        return number
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        msg = f"'{key}' needs a whole number, got {value!r}."
        raise ValueError(msg)
    try:
        number = int(value)
    except (TypeError, ValueError):
        msg = f"'{key}' needs a whole number, got {value!r}."
        raise ValueError(msg) from None
    low, high = (-128, 127) if field.fmt is _BYTE else (0, 2**32 - 1)
    if not low <= number <= high:
        msg = f"'{key}' must be between {low} and {high}, got {number}."
        raise ValueError(msg)
    return number


def _sexagesimal_ok(value: float, *, max_degrees: float) -> bool:
    """Return True if a HHMMSS.S or DDMMSS.S value has valid fields."""
    if value < 0 and max_degrees == 24:
        return False
    head, rest = divmod(abs(value), 10000)
    minutes, seconds = divmod(rest, 100)
    if not (0 <= minutes < 60 and 0 <= seconds < 60):
        return False
    if max_degrees == 24:
        return head < 24
    return head < 90 or (head == 90 and minutes == 0 and seconds == 0)


def _allowed_values(key: str) -> tuple[float, ...] | None:
    """Return the only values a key may take, or None if it has no fixed set."""
    if key == "nbits":
        return _NBITS_VALUES
    if key == "nifs":
        return _NIFS_VALUES
    if key == "data_type":
        return tuple(params.data_types)
    return None


def _position_problem(key: str, value: float) -> str | None:
    """Return what an angle or sky position must be, or None if it is valid."""
    if key in _ANGLE_LIMITS:
        low, high = _ANGLE_LIMITS[key]
        if not low <= value <= high:
            return f"between {low} and {high} degrees"
    if key == "src_raj" and not _sexagesimal_ok(value, max_degrees=24):
        return "a valid HHMMSS.S right ascension"
    if key == "src_dej" and not _sexagesimal_ok(value, max_degrees=90):
        return "a valid DDMMSS.S declination"
    return None


def _bound_problem(key: str, value: float) -> str | None:
    """Return what ``value`` must be for a numeric range check, or None if valid."""
    if key in _POSITIVE_KEYS and value <= 0:
        return "greater than 0"
    if key == "foff" and value == 0:
        return "non-zero"
    if key in _NON_NEGATIVE_KEYS and value < 0:
        return "zero or greater"
    return _position_problem(key, value)


def _range_problem(key: str, value: float | str) -> str | None:
    """Return a description of what ``value`` must be, or None if it is valid."""
    if isinstance(value, str):
        return None
    allowed = _allowed_values(key)
    if allowed is not None and value not in allowed:
        return f"one of {allowed}"
    if key in _FLAG_KEYS and value not in (0, 1):
        return "0 or 1"
    return _bound_problem(key, value)


def _check_value(key: str, value: float | str) -> None:
    """Raise ValueError if a value cannot describe a sigproc file."""
    problem = _range_problem(key, value)
    if problem is not None:
        msg = f"'{key}' must be {problem}, got {value}."
        raise ValueError(msg)


def _fit_string(key: str, value: str, old: str) -> str:
    """Return ``value`` padded to the byte length of ``old``, or raise if too long."""
    if not value:
        msg = f"'{key}' cannot be empty."
        raise ValueError(msg)
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        msg = f"New value for '{key}' contains non-ASCII characters."
        raise ValueError(msg) from None
    room = len(old.encode("utf-8"))
    if len(value) > room:
        msg = (
            f"New '{key}' is {len(value)} bytes but the file has room for {room}. "
            "Use a shorter value."
        )
        raise ValueError(msg)
    return value.ljust(room)


def _read_raw_header(filepath: Path) -> dict[str, Any]:
    """Read the header as stored in the file, without derived values."""
    with filepath.open("rb") as fp:
        return _read_sigproc_header(fp, filepath)


def _plan_edits(
    filepath: Path,
    updates: Mapping[str, float | str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate ``updates`` against the file and return ``(before, after)``."""
    if not updates:
        msg = "No header keys to update."
        raise ValueError(msg)
    before = _read_raw_header(filepath)
    after = dict(before)
    for key, value in updates.items():
        if key not in SIGPROC_SCHEMA:
            msg = f"Key '{key}' is not a valid sigproc key."
            raise ValueError(msg)
        if key in LAYOUT_KEYS:
            msg = (
                f"'{key}' cannot be changed in place: it sets how the samples are "
                "laid out. Use spp_digifil to requantise or reshape the data."
            )
            raise ValueError(msg)
        if key not in before:
            msg = (
                f"'{key}' is not in {filepath.name}. In-place edits cannot add keys, "
                "because that changes the header size."
            )
            raise ValueError(msg)
        new = _coerce_value(key, value)
        _check_value(key, new)
        if key in _STRING_KEYS:
            new = _fit_string(key, str(new), str(before[key]))
        if key == "signed" and new == 1:
            logger.warning("signed=1 is rejected by some readers.")
        after[key] = new
    return before, after


def _encode_checked(
    filepath: Path,
    before: dict[str, Any],
    after: dict[str, Any],
) -> bytes:
    """Encode the edited header and check that it reads back as requested."""
    new_hdr = encode_header(after, allow_nsamples_overwrite=True)
    if len(new_hdr) != before["hdrlen"]:
        msg = (
            f"New header length {len(new_hdr)} != original {before['hdrlen']} "
            f"for file {filepath}."
        )
        raise ValueError(msg)
    check = _read_sigproc_header(BytesIO(new_hdr), filepath)
    expected = {key: after[key] for key in SIGPROC_SCHEMA if key in after}
    got = {key: check.get(key) for key in expected}
    if got != expected or check.get("unknown_keys") != before.get("unknown_keys"):
        msg = "The edited header did not read back as requested. File not changed."
        raise ValueError(msg)
    return new_hdr


def prepare_header_edits(
    filename: str | Path,
    updates: Mapping[str, float | str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(before, after)`` headers for a would-be in-place edit.

    Values are checked against the file and the result is checked by reading it back.
    Nothing is written. ``before`` and ``after`` hold the values as stored in the
    file, so ``nsamples`` is the declared value, not the one computed from the data.

    Parameters
    ----------
    filename : str | Path
        Sigproc file to edit.
    updates : Mapping[str, float | int | str]
        Header keys and their new values.

    Returns
    -------
    tuple[dict, dict]
        Header values before and after the edit.

    Raises
    ------
    ValueError
        If a key is unknown, cannot be edited in place, is absent from the file, or
        has a value the file cannot store.
    """
    filepath = validate_path(filename)
    before, after = _plan_edits(filepath, updates)
    _encode_checked(filepath, before, after)
    return before, after


def prepare_header_edit(
    filename: str | Path,
    key: str,
    value: float | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(before, after)`` headers for a single would-be in-place edit.

    See :func:`prepare_header_edits`.
    """
    return prepare_header_edits(filename, {key: value})


def edit_headers(
    filename: str | Path,
    updates: Mapping[str, float | str],
) -> None:
    """Edit several keys of a sigproc header in place, all or nothing.

    Every value is checked, and the new header is read back, before the file is
    changed. If any check fails, the file is left as it was.

    Parameters
    ----------
    filename : str | Path
        Name of the sigproc file to modify.
    updates : Mapping[str, float | int | str]
        Header keys and their new values.

    Raises
    ------
    ValueError
        If a key is unknown, cannot be edited in place, is absent from the file, or
        has a value the file cannot store.

    Notes
    -----
    Edits never change the header size. Strings shorter than the stored value are
    padded with spaces. Longer strings are refused. Changing a key in place rewrites
    the file, so only use this on files you can afford to lose.
    """
    filepath = validate_path(filename, writable=True)
    before, after = _plan_edits(filepath, updates)
    new_hdr = _encode_checked(filepath, before, after)
    with filepath.open("rb+") as fp:
        fp.seek(0)
        fp.write(new_hdr)


def edit_header(filename: str | Path, key: str, value: float | str) -> None:
    """Edit one key of a sigproc header directly in place.

    See :func:`edit_headers` for the rules and the errors raised.

    Parameters
    ----------
    filename : str | Path
        Name of the sigproc file to modify.
    key : str
        Name of parameter to change (must be a valid sigproc key).
    value : int or float or str
        New value to enter into the header.
    """
    edit_headers(filename, {key: value})


def parse_header_multi(
    filenames: str | Path | Sequence[str | Path],
    *,
    check_contiguity: bool = True,
) -> tuple[dict, StreamInfo]:
    """Parse the metadata from Sigproc-style file/sequential files.

    Parameters
    ----------
    filenames : str | Path | Sequence[str | Path]
        sigproc filterbank files containing the header

    Returns
    -------
    tuple[dict[str, float | int | str], StreamInfo]
        observational metadata and stream info

    """
    if isinstance(filenames, str | Path):
        filenames = [filenames]

    header = parse_header(filenames[0])
    sinfo = StreamInfo([FileInfo.from_dict(header)])
    if len(filenames) > 1:
        for filename in filenames[1:]:
            hdr = parse_header(filename)
            match_header(header, hdr)
            sinfo.add_entry(FileInfo.from_dict(hdr))

        if check_contiguity and not sinfo.check_contiguity():
            msg = f"Files {filenames} are not contiguous"
            raise ValueError(msg)
    header["nsamples"] = sinfo.get_combined("nsamples")
    return header, sinfo


def parse_header(filename: str | Path) -> dict[str, float | int | str]:
    """Parse the metadata from a single Sigproc-style filterbank file.

    Parameters
    ----------
    filename : str | Path
        Path to the sigproc filterbank file.

    Returns
    -------
    dict[str, float | int | str]
        Observational metadata, plus computed keys. Unrecognised sigproc keys are
        kept as ``(key, raw value bytes)`` pairs under ``unknown_keys``.

    Raises
    ------
    OSError
        If the file header is not in sigproc format, is empty, or is
        truncated mid-header.
    ValueError
        If required keys (nbits, nchans) are missing or nsamples is zero.
    """
    filepath = validate_path(filename)
    with filepath.open("rb") as fp:
        header = _read_sigproc_header(fp, filepath)
        fp.seek(0, 2)
        filelen = fp.tell()
        header["filelen"] = filelen
    hdrlen = int(header["hdrlen"])
    datalen = filelen - hdrlen
    if datalen < 0:
        msg = (
            f"{filepath}: File length {filelen} is smaller than header "
            f"length {hdrlen}. File is likely corrupt."
        )
        raise OSError(msg)
    header["datalen"] = datalen
    nsamples = _compute_nsamples(header)
    if nsamples == 0:
        logger.warning(
            f"{filepath}: Computed nsamples is zero — data block may be empty."
        )
    header["nsamples"] = nsamples
    header["filename"] = filepath.as_posix()

    return header


def match_header(header1: dict, header2: dict) -> None:
    """Match header keywords between two parsed sigproc headers.

    A schema key must be present in both headers with the same value, or absent from
    both. ``tstart``, ``rawdatafile`` and ``nsamples`` are not compared, since they
    legitimately differ between sequential files.

    Parameters
    ----------
    header1 : dict
        parsed header from file 1.
    header2 : dict
        parsed header from file 2.

    Raises
    ------
    ValueError
        if key values do not match.
    """
    keys_nomatch = {"tstart", "rawdatafile", "nsamples"}
    keys = [
        key
        for key in SIGPROC_SCHEMA
        if key not in keys_nomatch and (key in header1 or key in header2)
    ]
    for key in keys:
        value1 = header1.get(key)
        value2 = header2.get(key)
        if value1 != value2:
            msg = (
                f"Header key ({key} = {value1}) and ({key} = {value2}) "
                f"do not match for file {header2.get('filename', '')}"
            )
            raise ValueError(msg)


def encode_header(
    header: dict[str, Any],
    *,
    allow_nsamples_overwrite: bool = False,
) -> bytes:
    """Encode sigproc header dict in binary format.

    Schema keys present in ``header`` are written in schema order, followed by any
    raw unknown keys stored under ``header["unknown_keys"]``.

    Returns
    -------
    bytes
        header in binary format
    """
    hdr_encoded = _encode_string("HEADER_START")

    # Deterministic ordering
    for key in SIGPROC_SCHEMA:
        # nsamples is legacy keyword and only for backward compatibility.
        if key == "nsamples" and not allow_nsamples_overwrite:
            continue
        if key not in header:
            continue
        hdr_encoded += encode_key(key, value=header[key])

    for key, raw_value in header.get("unknown_keys", ()):
        hdr_encoded += _encode_string(key) + raw_value

    hdr_encoded += _encode_string("HEADER_END")
    return hdr_encoded


def encode_key(
    key: str,
    value: float | str | None = None,
) -> bytes:
    """Encode single SIGPROC header key to a bytes string.

    Parameters
    ----------
    key : str
        header key
    value : int or float or str, optional
        value of the header key, by default None

    Returns
    -------
    bytes
        bytes encoded string
    """
    field_bytes = _encode_string(key)
    if value is None:
        return field_bytes
    if key not in SIGPROC_SCHEMA:
        msg = f"Unknown SIGPROC header key {key!r} cannot be encoded."
        raise KeyError(msg)
    field = SIGPROC_SCHEMA[key]
    if field.fmt is None:
        if not isinstance(value, str):
            msg = f"Key {key!r} expects a string value, got {value}."
            raise TypeError(msg)
        return field_bytes + _encode_string(value)
    return field_bytes + field.fmt.pack(value)


def parse_radec(src_raj: float, src_dej: float) -> SkyCoord:
    """Parse Sigproc format RADEC float as Astropy SkyCoord.

    Parameters
    ----------
    src_raj : float
        Sigproc style HHMMSS.SSSS right ascension
    src_dej : float
        Sigproc style DDMMSS.SSSS declination

    Returns
    -------
    :class:`~astropy.coordinates.SkyCoord`
        Astropy coordinate class
    """
    ho, mi = divmod(src_raj, 10000)
    mi, se = divmod(mi, 100)

    sign = "-" if src_dej < 0 else ""
    de, ami = divmod(abs(src_dej), 10000)
    ami, ase = divmod(ami, 100)

    # Fixed-point seconds avoid exponent tokens such as 1e-08, which Astropy rejects.
    # The sign goes on the degree field so that declinations above -1 degree keep it.
    radec_str = f"{int(ho)} {int(mi)} {se:.10f} {sign}{int(de)} {int(ami)} {ase:.10f}"
    return SkyCoord(radec_str, unit=(units.hourangle, units.deg))


def _encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _UINT.pack(len(encoded)) + encoded


def _read_sigproc_header(fp: BinaryIO, filepath: Path) -> dict[str, Any]:
    header: dict[str, Any] = {}
    unknown: list[tuple[str, bytes]] = []
    try:
        key = _read_string(fp)
    except (struct.error, UnicodeDecodeError, ValueError) as exc:
        msg = f"{filepath}: Header is not in sigproc format... Is file empty?."
        raise OSError(msg) from exc
    if key != "HEADER_START":
        msg = f"{filepath}: Header is not in sigproc format."
        raise OSError(msg)
    while True:
        try:
            key = _read_string(fp)
        except (struct.error, UnicodeDecodeError, ValueError) as exc:
            msg = (
                f"{filepath}: Header truncated while reading key at "
                f"byte offset {fp.tell()}."
            )
            raise OSError(msg) from exc
        if key == "HEADER_END":
            break
        if key == "FREQUENCY_START":
            _consume_frequency_start(fp, filepath)
            msg = (
                f"{filepath}: irregular frequency axis (FREQUENCY_START); "
                "linear fch1/foff only"
            )
            raise NotImplementedError(msg)
        if key not in SIGPROC_SCHEMA:
            # Unknown key: warn and attempt best-effort skip, keeping the raw bytes.
            logger.warning(
                f"{filepath}: Unknown header key {key} at offset {fp.tell()} — "
                f"attempting skip.",
            )
            value_start = fp.tell()
            _skip_unknown_key(fp, filepath)
            value_end = fp.tell()
            fp.seek(value_start)
            unknown.append((key, fp.read(value_end - value_start)))
            continue

        field = SIGPROC_SCHEMA[key]
        try:
            if field.fmt is None:
                header[key] = _read_string(fp)
            else:
                header[key] = _read_numeric(fp, field.fmt)
        except (struct.error, UnicodeDecodeError, ValueError) as exc:
            msg = (
                f"{filepath}: Header truncated while reading value for key {key!r} "
                f"at byte offset {fp.tell()}."
            )
            raise OSError(msg) from exc
    header["hdrlen"] = fp.tell()
    header["unknown_keys"] = tuple(unknown)
    return header


def _nsamples_in_binary_header(filename: str | Path) -> bool:
    """Return True if 'nsamples' key is physically present in the binary header."""
    filepath = validate_path(filename)
    with filepath.open("rb") as fp:
        _read_string(fp)  # HEADER_START — already validated upstream
        while True:
            key = _read_string(fp)
            if key == "HEADER_END":
                return False
            if key == "nsamples":
                return True
            # Skip value — reuse existing field dispatch
            field = SIGPROC_SCHEMA.get(key)
            if field is None:
                _skip_unknown_key(fp, filepath)
            elif field.fmt is None:
                _read_string(fp)
            else:
                fp.read(field.fmt.size)


def _consume_frequency_start(fp: BinaryIO, filepath: Path) -> None:
    """Skip a SIGPROC FREQUENCY_START / fchannel / FREQUENCY_END block."""
    key = _read_string(fp)
    if key != "nchans":
        msg = f"{filepath}: expected 'nchans' after FREQUENCY_START, got {key!r}"
        raise OSError(msg)
    nchans = int(_read_numeric(fp, _UINT))
    for _ in range(nchans):
        key = _read_string(fp)
        if key != "fchannel":
            msg = f"{filepath}: expected 'fchannel', got {key!r}"
            raise OSError(msg)
        _read_numeric(fp, _DOUBLE)
    key = _read_string(fp)
    if key != "FREQUENCY_END":
        msg = f"{filepath}: expected 'FREQUENCY_END', got {key!r}"
        raise OSError(msg)


def _read_numeric(fp: BinaryIO, fmt: struct.Struct) -> int | float:
    """Read and unpack a single fixed-width numeric value.

    Parameters
    ----------
    fp : BinaryIO
        Open file object positioned at the numeric value.
    fmt : struct.Struct
        Precompiled struct describing the binary layout (must include endian).

    Returns
    -------
    int | float
        The unpacked numeric value.

    Raises
    ------
    struct.error
        If EOF is encountered before reading the required number of bytes.
    """
    data = fp.read(fmt.size)
    if len(data) != fmt.size:
        msg = (
            f"EOF while reading numeric field: expected {fmt.size} bytes, "
            f"got {len(data)}."
        )
        raise struct.error(msg)
    return fmt.unpack(data)[0]


def _read_string(fp: BinaryIO) -> str:
    """Read a length-prefixed SIGPROC string: [uint32 length][raw bytes].

    Strings in SIGPROC headers are short ASCII identifiers
    (e.g., HEADER_START, source_name, etc.).

    Raises
    ------
    struct.error
        If EOF occurs while reading the length or string body.
    ValueError
        If the decoded length is implausible, indicating corruption
        or file misalignment.
    UnicodeDecodeError
        If the string is not valid UTF-8/ASCII.
    """
    # Read length field
    raw_len = fp.read(_UINT.size)
    if len(raw_len) != _UINT.size:
        msg = (
            f"EOF while reading string length: expected {_UINT.size} bytes, "
            f"got {len(raw_len)}."
        )
        raise struct.error(msg)

    strlen = _UINT.unpack(raw_len)[0]
    # SIGPROC strings are short (keywords, source names, etc.)
    # A large value almost certainly indicates corruption.
    if not (1 <= strlen <= 4096):
        msg = f"Implausible string length {strlen}; file likely corrupt or misaligned."
        raise ValueError(msg)

    # Read string body
    data = fp.read(strlen)
    if len(data) != strlen:
        msg = (
            f"EOF while reading string body: expected {strlen} bytes, got {len(data)}."
        )
        raise struct.error(msg)
    return data.decode("utf-8", errors="strict")


_PLAUSIBLE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_MAX_STRING_LEN = 4096
_MAX_UNKNOWN_KEYS = 64


def _value_candidates(fp: BinaryIO, start: int) -> list[int]:
    """Return the end offsets of the value that may start at ``start``, in probe order.

    Candidates are a 4-, 8- or 1-byte number, then a length-prefixed string. Only
    candidates that fit inside the file are returned.
    """
    fp.seek(0, 2)
    size = fp.tell()
    ends = [
        start + fmt.size for fmt in (_UINT, _DOUBLE, _BYTE) if start + fmt.size <= size
    ]
    fp.seek(start)
    raw_len = fp.read(_UINT.size)
    if len(raw_len) == _UINT.size:
        strlen = _UINT.unpack(raw_len)[0]
        if 1 <= strlen <= _MAX_STRING_LEN and start + _UINT.size + strlen <= size:
            ends.append(start + _UINT.size + strlen)
    return ends


def _header_reaches_end(fp: BinaryIO, pos: int, depth: int = 0) -> bool:
    """Return True if the header tokens from ``pos`` parse as far as HEADER_END.

    Known keys are read with their schema width. Unknown keys are accepted only with a
    plausible name, and their value candidates are tried recursively. A
    ``FREQUENCY_START`` block is accepted as a boundary, because the reader rejects it
    with a clear error when it reaches it.
    """
    if depth > _MAX_UNKNOWN_KEYS:
        return False
    while True:
        fp.seek(pos)
        try:
            key = _read_string(fp)
        except (struct.error, UnicodeDecodeError, ValueError):
            return False
        pos = fp.tell()
        if key in ("HEADER_END", "FREQUENCY_START"):
            return True
        if key in SIGPROC_SCHEMA:
            field = SIGPROC_SCHEMA[key]
            try:
                if field.fmt is None:
                    _read_string(fp)
                else:
                    _read_numeric(fp, field.fmt)
            except (struct.error, UnicodeDecodeError, ValueError):
                return False
            pos = fp.tell()
            continue
        if not _PLAUSIBLE_KEY.match(key):
            return False
        return any(
            _header_reaches_end(fp, end, depth + 1)
            for end in _value_candidates(fp, pos)
        )


def _skip_unknown_key(fp: BinaryIO, filepath: Path) -> None:
    """Skip an unknown key's value and leave the file at the next key.

    Each value candidate is tried in order. A candidate is committed only when the
    rest of the header, read from its end, reaches ``HEADER_END``. If none does,
    raise OSError.
    """
    start = fp.tell()
    for end in _value_candidates(fp, start):
        if _header_reaches_end(fp, end):
            fp.seek(end)
            logger.debug(f"Skipped unknown header key value at offset {start}.")
            return
    msg = (
        f"{filepath}: Cannot determine size of unknown header key value "
        f"at offset {start}. File may be corrupt or use an unsupported extension."
    )
    raise OSError(msg)


def _compute_nsamples(header: dict[str, int | float | str]) -> int:
    """Compute the number of samples from parsed header."""
    required_keys = frozenset({"nbits", "nchans", "datalen"})
    missing = required_keys - header.keys()
    if missing:
        msg = f"Cannot compute nsamples: missing required header keys {missing}."
        raise ValueError(msg)
    nbits = int(header["nbits"])
    nchans = int(header["nchans"])
    datalen = int(header["datalen"])
    if nbits <= 0 or nchans <= 0:
        msg = f"Invalid header values: nbits={nbits}, nchans={nchans} (must be > 0)."
        raise ValueError(msg)

    # Samples derived purely from file size and header parameters.
    nsamples_from_file = (8 * datalen) // nbits // nchans

    if "nsamples" in header:
        nsamples_declared = int(header["nsamples"])
        if nsamples_declared != nsamples_from_file:
            logger.warning(
                f"Header nsamples={nsamples_declared} disagrees with file-derived "
                f"nsamples={nsamples_from_file}. Using the smaller value.",
            )
        return min(nsamples_declared, nsamples_from_file)
    return nsamples_from_file
