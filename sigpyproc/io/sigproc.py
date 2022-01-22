import struct
import numpy as np

from typing import List, Dict, Union, Optional

from bidict import bidict
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy import units

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
        "Unknown": 11,
        "CHIME": 20,
    }
)

machine_ids = bidict(
    {
        "FAKE": 0,
        "PSPM": 1,
        "Wapp": 2,
        "AOFTM": 3,
        "BCPM1": 4,
        "OOTY": 5,
        "SCAMP": 6,
        "GBT Pulsar Spigot": 7,
        "PFFTS": 8,
        "Unknown": 9,
        "CHIME": 20,
    }
)


def edit_header(filename: str, key: str, value: Union[int, float, str]) -> None:
    """Edit a sigproc style header directly in place for the given file.

    Parameters
    ----------
    filename : str
        name of the sigproc file to modify header.
    key : str
        name of parameter to change (must be a valid sigproc key)
    value : Union[int, float, str]
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
        raise ValueError(f"Key '{key}' is not a valid sigproc key.")
    header = parse_header(filename)
    if key == "source_name" and isinstance(value, str):
        oldlen = len(header["source_name"])
        value = value[:oldlen] + " " * (oldlen - len(value))

    hdr = header.copy()
    hdr.update({key: value})
    new_hdr = encode_header(hdr)
    if header["hdrlen"] == len(new_hdr):
        with open(filename, "rb+") as fp:
            fp.seek(0)
            fp.write(new_hdr)
    else:
        raise ValueError("New header is too long/short for file")


def parse_header_multi(
    filenames: Union[str, List[str]], check_contiguity: bool = True
) -> Dict:
    """Parse the metadata from Sigproc-style file/sequential files.

    Parameters
    ----------
    filenames : list
        sigproc filterbank files containing the header

    Returns
    -------
    Dict
        observational metadata

    """
    if isinstance(filenames, str):
        filenames = [filenames]

    header = parse_header(filenames[0])
    # Set multifile header values
    header["hdrlens"] = [header["hdrlen"]]
    header["datalens"] = [header["datalen"]]
    header["nsamples_files"] = [header["nsamples"]]
    header["tstart_files"] = [header["tstart"]]
    header["filenames"] = [header["filename"]]

    if len(filenames) > 1:
        for filename in filenames[1:]:
            hdr = parse_header(filename)
            match_header(header, hdr)

            header["hdrlens"].append(hdr["hdrlen"])
            header["datalens"].append(hdr["datalen"])
            header["nsamples_files"].append(hdr["nsamples"])
            header["tstart_files"].append(hdr["tstart"])
            header["filenames"].append(hdr["filename"])

        if check_contiguity:
            ensure_contiguity(header)

    header["nsamples"] = sum(header["nsamples_files"])
    return header


def parse_header(filename: str) -> Dict:
    """Parse the metadata from a single Sigproc-style file.

    Parameters
    ----------
    filename : str
        sigproc filterbank file containing the header

    Returns
    -------
    Dict
        observational metadata

    Raises
    ------
    IOError
        If file header is not in sigproc format
    """
    with open(filename, "rb") as fp:
        header = {}
        try:
            key = _read_string(fp)
        except struct.error:
            raise IOError("File Header is not in sigproc format... Is file empty?")
        if key != "HEADER_START":
            raise IOError("File Header is not in sigproc format")
        while True:
            key = _read_string(fp)
            if key == "HEADER_END":
                break

            key_fmt = header_keys[key]
            if key_fmt == "str":
                header[key] = _read_string(fp)
            else:
                header[key] = struct.unpack(key_fmt, fp.read(struct.calcsize(key_fmt)))[0]
        header["hdrlen"] = fp.tell()
        fp.seek(0, 2)
        header["filelen"] = fp.tell()
        header["datalen"] = header["filelen"] - header["hdrlen"]
        header["nsamples"] = 8 * header["datalen"] // header["nbits"] // header["nchans"]
        fp.seek(0)
        header["filename"] = filename

    return header


def match_header(header1: Dict, header2: Dict) -> None:
    """Match header keywords between two parsed sigproc headers.

    Parameters
    ----------
    header1 : Dict
        parsed header from file 1.
    header2 : Dict
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
            raise ValueError(
                f'Header key "{key} = {value} and {header2[key]}"'
                + f'do not match for file {header2["filename"]}'
            )


def ensure_contiguity(header: Dict) -> None:
    """Check if list of sigproc files are contiguous/sequential.

    Parameters
    ----------
    header : Dict
        parsed header of sigproc files

    Raises
    ------
    ValueError
        if files are not contiguous
    """
    for ifile, _file in enumerate(header["filenames"][:-1]):
        precision = int(np.ceil(abs(np.log10(header["tsamp"]))))
        tstart = Time(
            header["tstart_files"][ifile], format="mjd", scale="utc", precision=precision
        )
        end_time = tstart + TimeDelta(
            header["nsamples_files"][ifile] * header["tsamp"], format="sec"
        )
        end_mjd = end_time.mjd
        difference = header["tstart_files"][ifile + 1] - end_mjd
        if abs(difference) > header["tsamp"]:
            samp_diff = int(abs(difference) / header["tsamp"])
            raise ValueError(
                f"files {header['filenames'][ifile]} and {header['filenames'][ifile + 1]} "
                + f"are off by at least {samp_diff} samples."
            )


def encode_header(header: Dict) -> bytes:
    """Get sigproc format header in binary format.

    Returns
    -------
    bytes
        header in binary format
    """
    hdr_encoded = encode_key("HEADER_START")

    for key in header.keys():
        if key not in header_keys:
            continue
        hdr_encoded += encode_key(key, value=header[key], value_type=header_keys[key])

    hdr_encoded += encode_key("HEADER_END")
    return hdr_encoded


def encode_key(
    key: str, value: Optional[Union[int, float, str]] = None, value_type: str = "str"
) -> bytes:
    """Encode given header key to a bytes string.

    Parameters
    ----------
    key : str
        header key
    value : Optional[Union[int, float, str]], optional
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
    SkyCoord
        Astropy coordinate class
    """
    ho, mi = divmod(src_raj, 10000)  # noqa: WPS432
    mi, se = divmod(mi, 100)

    sign = -1 if src_dej < 0 else 1
    de, ami = divmod(abs(src_dej), 10000)  # noqa: WPS432
    ami, ase = divmod(ami, 100)

    radec_str = f"{int(ho)} {int(mi)} {se} {sign* int(de)} {int(ami)} {ase}"
    return SkyCoord(radec_str, unit=(units.hourangle, units.deg))


def _read_string(fp):
    """Read the next sigproc-format string in the file.

    Parameters
    ----------
    fp : file
        file object to read from.

    Returns
    -------
    str
        read value from the file
    """
    strlen = struct.unpack("I", fp.read(struct.calcsize("I")))[0]
    return fp.read(strlen).decode()
