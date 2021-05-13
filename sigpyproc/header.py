from __future__ import annotations
import struct
import pathlib
import attr
import numpy as np

from typing import Dict, List, Any, Union, Optional

from astropy.time import Time, TimeDelta
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


@attr.s(auto_attribs=True, kw_only=True)
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

    filename: str
    data_type: int
    nchans: int
    foff: float
    fch1: float
    nbits: int
    tsamp: float
    tstart: float
    nsamples: int

    rawdatafile: Optional[str] = None
    ibeam: int = 0
    nbeams: int = 0
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
    barycentric: bool = False
    pulsarcentric: bool = False
    period: float = 0
    accel: float = 0
    hdrlen: int = 0
    datalen: int = 0

    hdrlens: List[int] = attr.Factory(list)
    datalens: List[int] = attr.Factory(list)
    filenames: List[str] = attr.Factory(list)
    nsamples_files: List[int] = attr.Factory(list)
    tstart_files: List[float] = attr.Factory(list)

    def __attrs_post_init__(self) -> None:
        self._coord = parse_radec(self.src_raj, self.src_dej)

    @property
    def basename(self) -> str:
        """Basename of header filename (`str`, read-only)."""
        return pathlib.Path(self.filename).stem

    @property
    def extension(self) -> str:
        """Extension of header filename (`str`, read-only)."""
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
        return get_time_after_nsamps(self.tstart, self.tsamp).iso

    def to_dict(self, with_properties=True) -> Dict[str, Any]:
        """Get a dict of all attributes including property attributes.

        Returns
        -------
        dict
            attributes
        """
        attributes = attr.asdict(self)
        if with_properties:
            prop = {
                key: getattr(self, key)
                for key, value in vars(type(self)).items()  # noqa: WPS421
                if isinstance(value, property)
            }
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
        return get_time_after_nsamps(self.tstart, self.tsamp, nsamps).mjd

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
        new = attr.asdict(self)
        if update_dict is not None:
            new.update(update_dict)
            new_checked = {
                key: value
                for key, value in new.items()
                if key in attr.asdict(self).keys()
            }
        return Header(**new_checked)

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
        header = encode_header("HEADER_START")

        for key in attr.asdict(self).keys():
            if back_compatible and key not in conf.sigproc_keys:
                continue
            elif not back_compatible and key not in conf.header_keys:
                continue

            key_fmt = conf.header_keys[key]
            header += encode_header(key, value=self[key], value_type=key_fmt)

        header += encode_header("HEADER_END")
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

        header["fch1"] = header["freq_low"] + header["foff"] * header["nchans"]
        header["src_raj"] = float(str(header["ra"]).replace(":", ""))
        header["src_dej"] = float(str(header["dec"]).replace(":", ""))
        header["telescope_id"] = conf.telescope_ids.get(header["telescope"], 11)
        header["machine_id"] = conf.machine_ids.get(header["machine"], 9)
        header.update({"data_type": 2, "nbits": 32, "nchans": 1, "hdrlen": 0})
        header_check = {
            key: value
            for key, value in header.items()
            if key in attr.fields_dict(cls).keys()
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
        # Set multifile header values
        header.hdrlens = [header.hdrlen]
        header.datalens = [header.datalen]
        header.nsamples_files = [header.nsamples]
        header.tstart_files = [header.tstart]
        header.filenames = [header.filename]

        if len(filenames) > 1:
            for filename in filenames[1:]:
                hdr = parse_sigproc_header(filename)
                match_header(header, hdr)

                header.hdrlens.append(hdr.hdrlen)
                header.datalens.append(hdr.datalen)
                header.nsamples_files.append(hdr.nsamples)
                header.tstart_files.append(hdr.tstart)
                header.filenames.append(hdr.filename)

            if check_contiguity:
                ensure_contiguity(header)

        header.nsamples = sum(header.nsamples_files)
        return cls(**header.to_dict(with_properties=False))


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
    header = Header.from_sigproc(filename)
    if key == "source_name" and isinstance(value, str):
        oldlen = len(header.source_name)
        value = value[:oldlen] + " " * (oldlen - len(value))
    new_hdr = header.new_header({key: value})
    new_header = new_hdr.spp_header(back_compatible=True)
    if header.hdrlens[0] == len(new_header):
        with open(filename, "r+") as fp:
            fp.seek(0)
            fp.write(new_header)
    else:
        raise ValueError("New header is too long/short for file")


def parse_sigproc_header(filename: str) -> Header:
    """Parse the metadata from a single Sigproc-style file.

    Parameters
    ----------
    filename : str
        sigproc filterbank file containing the header

    Returns
    -------
    Header
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

            key_fmt = conf.header_keys[key]
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

    header_check = {
        key: value
        for key, value in header.items()
        if key in attr.fields_dict(Header).keys()
    }
    return Header(**header_check)


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


def encode_header(
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


def match_header(header1: Header, header2: Header) -> None:
    """Match header keywords between two parsed sigproc headers.

    Parameters
    ----------
    header1 : Header
        parsed header from file 1.
    header2 : Header
        parsed header from file 2.

    Raises
    ------
    ValueError
        if key values do not match.
    """
    keys_nomatch = {"tstart", "rawdatafile"}
    for key in conf.sigproc_keys:
        if key in keys_nomatch:
            continue
        if getattr(header1, key) != getattr(header2, key):
            raise ValueError(
                f'Header key "{key} = {getattr(header1, key)} and {getattr(header2, key)}"'
                f"do not match for file {header2.filename}"
            )


def ensure_contiguity(header: Header) -> None:
    """Check if list of sigproc files are contiguous/sequential.

    Parameters
    ----------
    header : Header
        parsed header of sigproc files

    Raises
    ------
    ValueError
        if files are not contiguous
    """
    for ifile, _file in enumerate(header.filenames[:-1]):
        end_time = get_time_after_nsamps(
            header.tstart_files[ifile], header.tsamp, header.nsamples_files[ifile]
        )
        end_mjd = end_time.mjd
        difference = header.tstart_files[ifile + 1] - end_mjd
        if abs(difference) > header.tsamp:
            samp_diff = int(abs(difference) / header.tsamp)
            raise ValueError(
                f"files {header.filenames[ifile]} and {header.filenames[ifile + 1]} are off by "
                f"at least {samp_diff} samples."
            )


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


def get_time_after_nsamps(
    tstart: float, tsamp: float, nsamps: Optional[int] = None
) -> Time:
    """Get precise time nsamps after input tstart. If nsamps is not given then just return tstart.

    Parameters
    ----------
    tstart : float
        starting mjd.
    tsamp : float
        sampling time in seconds.
    nsamps : Optional[int], optional
        number of samples, by default None

    Returns
    -------
    Time
        Astropy Time object after given nsamps
    """
    precision = int(np.ceil(abs(np.log10(tsamp))))
    tstart = Time(tstart, format="mjd", scale="utc", precision=precision)
    if nsamps:
        return tstart + TimeDelta(nsamps * tsamp, format="sec")
    return tstart
