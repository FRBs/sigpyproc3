from __future__ import annotations
import struct
import pathlib
import erfa
import numpy as np

from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass, asdict

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units, constants

from sigpyproc import HeaderParams as conf
from sigpyproc.io.fileio import FileWriter


DM_CONSTANT_LK = 4.148808e3  # L&K Handbook of Pulsar Astronomy
DM_CONSTANT_MT = 1 / 0.000241  # TEMPO2 Manchester & Taylor (1972)  # noqa: WPS432
DM_CONSTANT_SI = (
    (constants.e.esu ** 2 / (2 * np.pi * constants.m_e * constants.c)).to(
        units.s * units.MHz ** 2 * units.cm ** 3 / units.pc
    )
).value  # Precise SI constants


@dataclass
class Header(object):
    """Container object to handle observation metadata.

    Parameters
    ----------
    rawdatafile : str, optional
        [description], by default 1
    data_type : str, optional
        [description], by default 'abc'
    fch1 : float
        central frequency of the first channel in MHz
    foff : float
        channel width in MHz
    """

    rawdatafile: str
    data_type: int
    nchans: int
    foff: float
    fch1: float
    nbits: int
    tsamp: float
    tstart: float
    ibeam: int = 0
    nbeams: int = 1
    refdm: float = 0
    nifs: int = 1
    telescope_id: int = 0
    machine_id: int = 0
    src_dej: float = 0
    src_raj: float = 0
    za_start: float = 0
    az_start: float = 0
    source_name: str = "None"
    signed: bool = False

    filename: Optional[str] = None
    hdrlen: int = 0
    nsamples: int = 0
    period: float = 0
    accel: float = 0

    def __post_init__(self) -> None:
        self._coord = parse_radec(self.src_raj, self.src_dej)

    @property
    def basename(self) -> Optional[str]:
        """Basename of header filename (`str` or None, read-only)."""
        if self.filename is None:
            return None
        return pathlib.Path(self.filename).stem

    @property
    def extension(self) -> Optional[str]:
        """Extension of header filename (`str` or None, read-only)."""
        if self.filename is None:
            return None
        return pathlib.Path(self.filename).suffix

    @property
    def telescope(self) -> str:
        """Telescope name (`str`, read-only)."""
        return conf.ids_to_telescope[self.telescope_id]

    @property
    def machine(self) -> str:
        """Backend name (`str`, read-only)."""
        return conf.ids_to_machine[self.machine_id]

    @property
    def bandwidth(self) -> float:
        """Bandwidth in MHz (`float`, read-only)."""
        return abs(self.foff) * self.nchans

    @property
    def ftop(self) -> float:
        """Frequency (boundary) of the top channel (`float`, read-only)."""
        return self.fch1 - 0.5 * self.foff

    @property
    def fbottom(self) -> float:
        """Frequency (boundary) of the bottom channel (`float`, read-only)."""
        return self.ftop + self.foff * self.nchans

    @property
    def fcenter(self) -> float:
        """Central frequency of the whole band (`float`, read-only)."""
        return self.ftop + 0.5 * self.foff * self.nchans

    @property
    def chan_freqs(self) -> np.ndarray:
        """Frequency (center) of each channel(`np.ndarray`, read-only)."""
        return np.arange(self.nchans, dtype="float128") * self.foff + self.fch1

    @property
    def dtype(self) -> np.dtype:
        """Type of the data (`np.dtype`, read-only)."""
        return conf.bits_info[self.nbits].dtype

    @property
    def tobs(self) -> float:
        """Total time of the observation (`float`, read-only)."""
        return self.tsamp * self.nsamples

    @property
    def coord(self) -> SkyCoord:
        """Sky coordinate (`astropy.coordinates.SkyCoord`, read-only)."""
        return self._coord

    @property
    def ra(self) -> str:
        """Right Ascension (`str`, read-only)."""
        return self._coord.ra.to_string(unit="hourangle", sep=":", pad=True)

    @property
    def dec(self) -> str:
        """Declination (`str`, read-only)."""
        return self._coord.dec.to_string(unit="deg", sep=":", pad=True)

    @property
    def ra_rad(self) -> float:
        """Right Ascension in radians (`float`, read-only)."""
        return self._coord.ra.rad

    @property
    def dec_rad(self) -> float:
        """Declination in radians (`float`, read-only)."""
        return self._coord.dec.rad

    @property
    def ra_deg(self) -> float:
        """Right Ascension in degrees (`float`, read-only)."""
        return self._coord.ra.deg

    @property
    def dec_deg(self) -> float:
        """Declination in degrees (`float`, read-only)."""
        return self._coord.dec.deg

    @property
    def obs_date(self) -> str:
        """Observation date and time (`str`, read-only)."""
        precision = int(np.ceil(abs(np.log10(self.tsamp))))
        return Time(self.tstart, format="mjd", scale="utc", precision=precision).iso

    def to_dict(self) -> Dict[str, Any]:
        """Get a dict of all attributes including property attributes.

        Returns
        -------
        dict
            attributes
        """
        prop = {
            key: getattr(self, key)
            for key, value in vars(type(self)).items()  # noqa: WPS421
            if isinstance(value, property)
        }
        attributes = asdict(self)
        attributes.update(prop)
        return attributes

    def mjd_after_nsamps(self, nsamps: int) -> float:
        """Find the Modified Julian Date after nsamps have elapsed.

        Parameters
        ----------
        nsamps : int
            number of samples elapsed since start of observation.

        Returns
        -------
        float
            Modified Julian Date
        """
        return self.tstart + ((nsamps * self.tsamp) / erfa.DAYSEC)

    def new_header(self, update_dict: Optional[Dict[str, Any]] = None) -> Header:
        """Get a new instance of :class:`~sigpyproc.Header.Header`.

        Parameters
        ----------
        update_dict : dict, optional
            values to overide existing header values, by default None

        Returns
        -------
        :class:`~sigpyproc.Header.Header`
            new header information
        """
        new = asdict(self)
        if update_dict is not None:
            new.update(update_dict)
            new_check = {
                key: value for key, value in new.items() if key in asdict(self).keys()
            }
        return Header(**new_check)

    def dedispersed_header(self, dm: float) -> Header:
        """Get a dedispersed version of the current header.

        Parameters
        ----------
        dm : float
            dispersion measure we are dedispersing to

        Returns
        -------
        :class:`~sigpyproc.Header.Header`
            A dedispersed version of the header
        """
        return self.new_header({"refdm": dm, "nchans": 1, "data_type": 2, "nbits": 32})

    def spp_header(self, back_compatible=True):
        """Get Sigproc/sigpyproc format binary header.

        Parameters
        ----------
        back_compatible : bool, optional
            Flag for returning Sigproc compatible header (legacy code), by default True

        Returns
        -------
        str
            header in binary format
        """
        header = self._write_header("HEADER_START")

        for key in list(self.keys()):
            if back_compatible and key not in conf.sigproc_keys:
                continue
            elif not back_compatible and key not in conf.header_keys:
                continue

            key_fmt = conf.header_keys[key]
            header += self._write_header(key, value=self[key], value_type=key_fmt)

        header += self._write_header("HEADER_END")
        return header

    def get_dmdelays(self, dm: float, in_samples: bool = True) -> np.ndarray:
        """For a given dispersion measure get the dispersive ISM delay for middle of each frequency channel.

        Parameters
        ----------
        dm : float
            dispersion measure to calculate delays for
        in_samples : bool, optional
            flag to return delays as numbers of samples, by default True

        Returns
        -------
        :py:obj:`numpy.ndarray`
            delays for middle of each channel (highest frequency first)
        """
        delays = dm * DM_CONSTANT_LK * ((self.chan_freqs ** -2) - (self.fch1 ** -2))
        if in_samples:
            return (delays / self.tsamp).round().astype("int32")
        return delays

    def prep_outfile(
        self,
        filename: str,
        update_dict: Optional[Dict[str, Any]] = None,
        nbits: Optional[int] = None,
        back_compatible: bool = True,
        quantize: bool = False,
        interval_seconds: float = 10,
        constant_offset_scale: bool = False,
        **kwargs,
    ) -> FileWriter:
        """Prepare a file to have sigproc format data written to it.

        Parameters
        ----------
        filename : str
            name of new file
        update_dict : dict, optional
            values to overide existing header values, by default None
        nbits : int, optional
            the bitsize of data points that will written to this file (1,2,4,8,32),
            by default None
        back_compatible : bool, optional
            flag for making file Sigproc compatible, by default True

        Returns
        -------
        :class:`~sigpyproc.Utils.File`
            a prepared file
        """
        if nbits is None:
            nbits = self.nbits
        new = self.new_header(update_dict)
        new.nbits = nbits
        out_file = FileWriter(
            filename,
            mode="w+",
            nbits=nbits,
            tsamp=new.tsamp,
            nchans=new.nchans,
            quantize=quantize,
            interval_seconds=interval_seconds,
            constant_offset_scale=constant_offset_scale,
            **kwargs,
        )
        out_file.write(new.spp_header(back_compatible=back_compatible))
        return out_file

    def make_inf(self, outfile=None):
        """Make a presto format ``.inf`` file.

        Parameters
        ----------
        outfile : str, optional
            a filename to write to, by default None

        Returns
        -------
        str
            if outfile is unspecified ``.inf`` data is returned as string
        """
        inf_dict = self.to_dict()
        # Central freq of low channel (Mhz)
        inf_dict["freq_low"] = self.fbottom + 0.5 * abs(self.foff)
        inf_dict["barycentric"] = 0
        inf_dict["observer"] = "Robotic overlords"
        inf_dict["analyser"] = "sigpyproc"
        inf = [
            f" {desc:<38} =  {inf_dict[key]:{keyformat}}"
            for desc, (key, _keytype, keyformat) in conf.presto_inf.items()
        ]
        inf = "\n".join(inf)
        if outfile is None:
            return inf
        with open(outfile, "w+") as fp:
            fp.write(inf)
        return None

    @classmethod
    def from_inffile(cls, filename: str) -> Header:
        """Parse the metadata from a presto ``.inf`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.inf`` file containing the header

        Returns
        -------
        :class:`~sigpyproc.Header.Header`
            observational metadata
        """
        header: Dict[str, Any] = {}
        with open(filename, "r") as fp:
            lines = fp.readlines()

        for line in lines:
            desc = line.split("=")[0].strip()
            val = line.split("=")[-1].strip()
            if desc not in list(conf.presto_inf.keys()):
                continue
            else:
                key, keytype, _keyformat = conf.presto_inf[desc]
                header[key] = keytype(val)

        header["src_raj"] = float(str(header["ra"]).replace(":", ""))
        header["src_dej"] = float(str(header["dec"]).replace(":", ""))
        header["telescope_id"] = conf.telescope_ids.get(header["telescope"], 10)
        header["machine_id"] = conf.machine_ids.get(header["machine"], 9)
        header.update({"data_type": 2, "nbits": 32, "nchans": 1, "hdrlen": 0})
        header_check = {
            key: value for key, value in header.items() if key in asdict(cls).keys()
        }
        return cls(**header_check)

    @classmethod
    def from_sigproc(
        cls, filenames: Union[str, List[str]], check_contiguity: bool = True
    ) -> Header:
        """Parse the metadata from Sigproc-style file/sequential files.

        Parameters
        ----------
        filenames : list
            sigproc filterbank files containing the header

        Returns
        -------
        :class:`~sigpyproc.Header.Header`
            observational metadata

        """
        if isinstance(filenames, str):
            filenames = [filenames]

        header = parse_sigproc_header(filenames[0])
        header["hdrlens"] = [header["hdrlen"]]
        header["datalens"] = [header["nbytes"]]
        header["nsamples_list"] = [header["nsamples"]]
        header["tstart_list"] = [header["tstart"]]
        header["filenames"] = [header["filename"]]
        if len(filenames) > 1:
            for filename in filenames[1:]:
                hdr = parse_sigproc_header(filename)
                for key in conf.sigproc_keys:
                    if key in {"tstart", "rawdatafile"}:
                        continue
                    if key in header:  # TODO Fix later
                        assert (
                            hdr[key] == header[key]
                        ), f"Header value '{hdr[key]}' do not match for file {filename}"
                header["hdrlens"].append(hdr["hdrlen"])
                header["datalens"].append(hdr["nbytes"])
                header["nsamples_list"].append(hdr["nsamples"])
                header["tstart_list"].append(hdr["tstart"])
                header["filenames"].append(hdr["filename"])

            if check_contiguity:
                cls._ensure_contiguity(header)

        return cls(**header)

    @staticmethod
    def _ensure_contiguity(header):
        """Check if list of sigproc files are contiguous/sequential.

        Parameters
        ----------
        header : dict
            A dict of sigproc header keys for input files

        Raises
        ------
        ValueError
            if files are not contiguous
        """
        filenames = header["filenames"]
        for ii, _file in enumerate(filenames[:-1]):
            end_time = (
                header["tstart_list"][ii]
                + header["nsamples_list"][ii] * header["tsamp"] / 86400
            )
            difference = header["tstart_list"][ii + 1] - end_time
            if abs(difference) > 0.9 * header["tsamp"]:
                samp_diff = int(abs(difference) / header["tsamp"])
                raise ValueError(
                    f"files {filenames[ii]} and {filenames[ii + 1]} are off by "
                    f"at least {samp_diff} samples."
                )

    @staticmethod
    def _write_header(key, value=None, value_type="str"):
        """Encode the header key to a bytes string.

        Parameters
        ----------
        key : str
            header key
        value : int, float, str, optional
            value of the header key, by default None
        value_type : str, optional
            type of the header key, by default "str"

        Returns
        -------
        str
            bytes string
        """
        if value is None:
            return struct.pack("I", len(key)) + key.encode()

        if value_type == "str":
            return (
                struct.pack("I", len(key))
                + key.encode()
                + struct.pack("I", len(value))
                + value.encode()
            )
        return struct.pack("I", len(key)) + key.encode() + struct.pack(value_type, value)


def parse_sigproc_header(filename: str) -> Dict[str, Any]:
    """Parse the metadata from a single Sigproc-style file.

    Parameters
    ----------
    filename : str
        sigproc filterbank file containing the header

    Returns
    -------
    dict
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

            if key not in conf.header_keys:
                raise IOError(f"'{key}' is not a recognised sigproc header param")

            key_fmt = conf.header_keys[key]
            if key_fmt == "str":
                header[key] = _read_string(fp)
            else:
                header[key] = struct.unpack(key_fmt, fp.read(struct.calcsize(key_fmt)))[0]
        header["hdrlen"] = fp.tell()
        fp.seek(0, 2)
        header["filelen"] = fp.tell()
        header["nbytes"] = header["filelen"] - header["hdrlen"]
        header["nsamples"] = 8 * header["nbytes"] // header["nbits"] // header["nchans"]
        fp.seek(0)
        header["filename"] = filename
    return header


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


def edit_header(filename, key, value):
    """Edit a sigproc style header in place for the given file.

    Parameters
    ----------
    filename : str
        name of the sigproc file to modify header.
    key : str
        name of parameter to change (must be a valid sigproc key)
    value : int, float or str
        new value to enter into header

    Raises
    ------
    ValueError
        [description]

    Notes
    -----
       It is up to the user to be responsible with this function, as it will directly
       change the file on which it is being operated. The only fail contition of
       editInplace comes when the new header to be written to file is longer or shorter than the
       header that was previously in the file.
    """
    header = Header.parseSigprocHeader(filename)
    if key == "source_name":
        oldlen = len(header.source_name)
        value = value[:oldlen] + " " * (oldlen - len(value))
    header[key] = value
    new_header = header.SPPHeader(back_compatible=True)
    if header.hdrlen == len(new_header):
        with open(filename, "r+") as fp:
            fp.seek(0)
            fp.write(new_header)
    else:
        raise ValueError("New header is too long/short for file")


def parse_radec(src_raj: float, src_dej: float) -> SkyCoord:
    """Parse Sigproc format RADEC float to Astropy SkyCoord.

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
