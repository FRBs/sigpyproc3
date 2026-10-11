"""Sigproc-style header parsing and editing.

This module contains functions for parsing and editing Sigproc-style headers.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from bidict import bidict

from sigpyproc.utils import get_logger, validate_path

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


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


def prepare_header_edit(
    filename: str | Path,
    key: str,
    value: float | str,
) -> tuple[dict[str, float | int | str], dict[str, float | int | str]]:
    """Return ``(before, after)`` parsed headers for a would-be in-place edit.

    Does not write to disk. String keys are padded/truncated to the existing
    byte length, matching :func:`edit_header`.
    """
    if key not in SIGPROC_SCHEMA:
        msg = f"Key '{key}' is not a valid sigproc key."
        raise ValueError(msg)
    before = parse_header(filename)
    after = dict(before)
    if key in ("source_name", "rawdatafile") and isinstance(value, str):
        old_value = str(before.get(key, ""))
        old_bytes = old_value.encode("ascii")
        try:
            new_bytes = value.encode("ascii")
        except UnicodeEncodeError as exc:
            msg = (
                f"New value for '{key}' contains non-ASCII characters, which is "
                f"not allowed."
            )
            raise ValueError(msg) from exc
        oldlen = len(old_bytes)
        if len(new_bytes) > oldlen:
            logger.warning(
                f"New value for '{key}' is too long ({len(new_bytes)} bytes) to "
                f"fit in existing header space ({oldlen} bytes). Value will be "
                f"truncated.",
            )
            new_bytes = new_bytes[:oldlen]
        value = new_bytes.ljust(oldlen, b" ").decode("ascii")
    after[key] = value
    return before, after


def edit_header(filename: str | Path, key: str, value: float | str) -> None:
    """Edit a sigproc style header directly in place for the given file.

    Parameters
    ----------
    filename : str | Path
        Name of the sigproc file to modify header.
    key : str
        Name of parameter to change (must be a valid sigproc key)
    value : int or float or str
        new value to enter into header

    Raises
    ------
    ValueError
        if the new header to be written to file is longer or shorter than
        the header that was previously in the file.

    Notes
    -----
       It is up to the user to be responsible with this function, as it will directly
       change the file on which it is being operated.
    """
    _before, header = prepare_header_edit(filename, key, value)
    # nsamples is always injected by parse_header (computed), but may or may
    # not have existed in the original binary header. We must not encode it
    # unless it was genuinely there — otherwise we silently grow the header.
    nsamples_in_file = _nsamples_in_binary_header(filename)
    new_hdr = encode_header(header, allow_nsamples_overwrite=nsamples_in_file)
    filepath = validate_path(filename, writable=True)
    if len(new_hdr) != int(header["hdrlen"]):
        msg = (
            f"New header length {len(new_hdr)} != original "
            f"{header['hdrlen']} for file {filename}."
        )
        raise ValueError(msg)
    with filepath.open("rb+") as fp:
        fp.seek(0)
        fp.write(new_hdr)


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
