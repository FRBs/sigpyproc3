from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import attrs
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from bidict import bidict

from sigpyproc.utils import validate_path

header_keys = {
    "signed": "b",
    "telescope_id": "I",
    "ibeam": "I",
    "nbeams": "I",
    "refdm": "d",
    "nifs": "I",
    "nchans": "I",
    "foff": "d",
    "fch1": "d",
    "nbits": "I",
    "tsamp": "d",
    "tstart": "d",
    "src_dej": "d",
    "src_raj": "d",
    "za_start": "d",
    "az_start": "d",
    "source_name": "str",
    "rawdatafile": "str",
    "data_type": "I",
    "machine_id": "I",
    "barycentric": "I",
    "pulsarcentric": "I",
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
        "MeerKAT": 64,
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
    },
)
"""Machine IDs recognised by the sigproc package."""


@attrs.define(frozen=True, kw_only=True)
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
    def from_dict(cls, info: dict) -> FileInfo:
        """Create FileInfo object from a dictionary."""
        info_filtered = {key: info[key] for key in attrs.fields_dict(cls)}
        return cls(**info_filtered)


@attrs.define(frozen=True)
class StreamInfo:
    """Class to handle stream information as a list of FileInfo objects."""

    entries: list[FileInfo] = attrs.Factory(list)

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
        if not isinstance(finfo, FileInfo):
            msg = f"Input must be a FileInfo object, got {type(finfo)}"
            raise TypeError(msg)
        self.entries.append(finfo)

    def get_combined(self, key: str) -> int:
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
    if key not in header_keys:
        msg = f"Key '{key}' is not a valid sigproc key."
        raise ValueError(msg)
    header = parse_header(filename)
    if key == "source_name" and isinstance(value, str):
        oldlen = len(header["source_name"])
        value = value[:oldlen] + " " * (oldlen - len(value))

    hdr = header.copy()
    hdr.update({key: value})
    new_hdr = encode_header(hdr)
    filepath = validate_path(filename, writable=True)
    if header["hdrlen"] == len(new_hdr):
        with filepath.open("rb+") as fp:
            fp.seek(0)
            fp.write(new_hdr)
    else:
        msg = f"New header is too long/short for file {filename}"
        raise ValueError(msg)


def parse_header_multi(
    filenames: str | Path | list[str | Path],
    *,
    check_contiguity: bool = True,
) -> dict:
    """Parse the metadata from Sigproc-style file/sequential files.

    Parameters
    ----------
    filenames : str | Path | list[str | Path]
        sigproc filterbank files containing the header

    Returns
    -------
    dict
        observational metadata

    """
    if isinstance(filenames, (str, Path)):
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

    header["stream_info"] = sinfo
    header["nsamples"] = header["stream_info"].get_combined("nsamples")
    return header


def parse_header(filename: str | Path) -> dict:
    """
    Parse the metadata from a single Sigproc-style filterbank file.

    Parameters
    ----------
    filename : str | Path
        Path to the sigproc filterbank file containing the header

    Returns
    -------
    dict
        observational metadata

    Raises
    ------
    OSError
        If file header is not in sigproc format
    """
    filepath = validate_path(filename)
    with filepath.open("rb") as fp:
        header: dict[str, float | str] = {}
        try:
            key = _read_string(fp)
        except struct.error:
            msg = f"File {filename} Header is not in sigproc format... Is file empty?."
            raise OSError(msg) from None
        if key != "HEADER_START":
            msg = f"File {filename} Header is not in sigproc format."
            raise OSError(msg)
        while True:
            key = _read_string(fp)
            if key == "HEADER_END":
                break

            key_fmt = header_keys[key]
            if key_fmt == "str":
                header[key] = _read_string(fp)
            else:
                header[key] = struct.unpack(key_fmt, fp.read(struct.calcsize(key_fmt)))[
                    0
                ]
        header["hdrlen"] = fp.tell()
        fp.seek(0, 2)
        header["filelen"] = fp.tell()
        header["datalen"] = int(header["filelen"]) - int(header["hdrlen"])
        header["nsamples"] = (
            8 * int(header["datalen"]) // int(header["nbits"]) // int(header["nchans"])
        )
        fp.seek(0)
        header["filename"] = filepath.as_posix()

    return header


def match_header(header1: dict, header2: dict) -> None:
    """Match header keywords between two parsed sigproc headers.

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
    keys_nomatch = {"tstart", "rawdatafile"}
    for key, value in header1.items():
        if key in keys_nomatch or key not in header_keys:
            continue
        if value != header2[key]:
            msg = (
                f"Header key ({key} = {value}) and ({key} = {header2[key]}) "
                f"do not match for file {header2['filename']}"
            )
            raise ValueError(msg)


def encode_header(header: dict) -> bytes:
    """Get sigproc format header in binary format.

    Returns
    -------
    bytes
        header in binary format
    """
    hdr_encoded = encode_key("HEADER_START")

    for key in header:
        if key not in header_keys:
            continue
        hdr_encoded += encode_key(key, value=header[key], value_type=header_keys[key])

    hdr_encoded += encode_key("HEADER_END")
    return hdr_encoded


def encode_key(
    key: str,
    value: float | str | None = None,
    value_type: str = "str",
) -> bytes:
    """Encode given header key to a bytes string.

    Parameters
    ----------
    key : str
        header key
    value : int or float or str, optional
        value of the header key, by default None
    value_type : str, optional
        type of the header key, by default "str"

    Returns
    -------
    bytes
        bytes encoded string
    """
    if value is None:
        return struct.pack("I", len(key)) + key.encode()

    if value_type == "str" and isinstance(value, str):
        return (
            struct.pack("I", len(key))
            + key.encode()
            + struct.pack("I", len(value))
            + value.encode()
        )
    return struct.pack("I", len(key)) + key.encode() + struct.pack(value_type, value)


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

    sign = -1 if src_dej < 0 else 1
    de, ami = divmod(abs(src_dej), 10000)
    ami, ase = divmod(ami, 100)

    radec_str = f"{int(ho)} {int(mi)} {se} {sign* int(de)} {int(ami)} {ase}"
    return SkyCoord(radec_str, unit=(units.hourangle, units.deg))


def _read_string(fp: BinaryIO) -> str:
    """Read the next sigproc-format string in the file.

    Parameters
    ----------
    fp : file object
        file object to read from.

    Returns
    -------
    str
        read value from the file
    """
    strlen = struct.unpack("I", fp.read(struct.calcsize("I")))[0]
    return fp.read(strlen).decode()
