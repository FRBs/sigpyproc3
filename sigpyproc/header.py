from __future__ import annotations
import pathlib
import attr
import numpy as np

from typing import Dict, List, Any, Union, Optional

from astropy import units
from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle, SkyCoord

from sigpyproc import params
from sigpyproc.io import sigproc
from sigpyproc.io.bits import BitsInfo
from sigpyproc.io.fileio import FileWriter


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
    coord: SkyCoord
        Sky coordinate.
    """

    filename: str
    data_type: str
    nchans: int
    foff: float
    fch1: float
    nbits: int
    tsamp: float
    tstart: float
    nsamples: int
    nifs: int
    coord: SkyCoord

    azimuth: Angle = Angle("0d")
    zenith: Angle = Angle("0d")
    telescope: str = "Fake"
    backend: str = "Fake"
    source: str = "Fake"
    ibeam: int = 0
    nbeams: int = 0
    dm: float = 0
    period: float = 0
    accel: float = 0
    signed: bool = False
    barycentric: bool = False
    pulsarcentric: bool = False

    rawdatafile: Optional[str] = None

    hdrlens: List[int] = attr.Factory(list)
    datalens: List[int] = attr.Factory(list)
    filenames: List[str] = attr.Factory(list)
    nsamples_files: List[int] = attr.Factory(list)
    tstart_files: List[float] = attr.Factory(list)

    @property
    def basename(self) -> str:
        """Basename of header filename (`str`, read-only)."""
        return pathlib.Path(self.filename).stem

    @property
    def extension(self) -> str:
        """Extension of header filename (`str`, read-only)."""
        return pathlib.Path(self.filename).suffix

    @property
    def telescope_id(self) -> int:
        """Telescope id (`int`, read-only)."""
        return sigproc.telescope_ids[self.telescope]

    @property
    def machine_id(self) -> int:
        """Machine id (`str`, read-only)."""
        return sigproc.machine_ids[self.backend]

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
        return BitsInfo(self.nbits).dtype

    @property
    def tobs(self) -> float:
        """Total time of the observation (`float`, read-only)."""
        return self.tsamp * self.nsamples

    @property
    def ra(self) -> str:
        """Right Ascension (`str`, read-only)."""
        return self.coord.ra.to_string(unit="hourangle", sep=":", pad=True)

    @property
    def dec(self) -> str:
        """Declination (`str`, read-only)."""
        return self.coord.dec.to_string(unit="deg", sep=":", pad=True)

    @property
    def ra_rad(self) -> float:
        """Right Ascension in radians (`float`, read-only)."""
        return self.coord.ra.rad

    @property
    def dec_rad(self) -> float:
        """Declination in radians (`float`, read-only)."""
        return self.coord.dec.rad

    @property
    def ra_deg(self) -> float:
        """Right Ascension in degrees (`float`, read-only)."""
        return self.coord.ra.deg

    @property
    def dec_deg(self) -> float:
        """Declination in degrees (`float`, read-only)."""
        return self.coord.dec.deg

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
        return self.new_header(
            {"dm": dm, "nchans": 1, "data_type": "time series", "nbits": 32}
        )

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
        delays = (
            dm * params.DM_CONSTANT_LK * ((self.chan_freqs ** -2) - (self.fch1 ** -2))
        )
        if in_samples:
            return (delays / self.tsamp).round().astype("int32")
        return delays

    def spp_header(self, as_dict=False) -> Union[Dict, bytes]:
        """Get sigproc format header binary header.

        Returns
        -------
        str
            header in binary format
        """
        header = attr.asdict(self)
        if as_dict:
            return header

        return sigproc.encode_header(header)

    def prep_outfile(
        self,
        filename: str,
        update_dict: Optional[Dict[str, Any]] = None,
        nbits: Optional[int] = None,
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

        Returns
        -------
        FileWriter
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
        out_file.write(new.spp_header())
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
            for desc, (key, _keytype, keyformat) in params.presto_inf.items()
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
            if desc not in list(params.presto_inf.keys()):
                continue
            else:
                key, keytype, _keyformat = params.presto_inf[desc]
                header[key] = keytype(val)

        hdr_update = {
            "data_type": "time series",
            "fch1": header["freq_low"] + header["foff"] * header["nchans"],
            "nbits": 32,
            "nchans": 1,
            "coord": SkyCoord(
                header["ra"], header["dec"], unit=(units.hourangle, units.deg)
            ),
        }
        header.update(hdr_update)
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
        header = sigproc.parse_header_multi(filenames, check_contiguity=check_contiguity)
        hdr_update = {
            "data_type": params.data_types[header["data_type"]],
            "telescope": sigproc.telescope_ids.inverse[header["data_type"]],
            "backend": sigproc.machine_ids.inverse[header["machine_id"]],
            "source": header["source_name"],
            "dm": header["refdm"],
            "coord": sigproc.parse_radec(header["src_raj"], header["src_dej"]),
            "azimuth": Angle(header["az_start"] * units.deg),
            "zenith": Angle(header["za_start"] * units.deg),
        }
        header.update(hdr_update)
        header_check = {
            key: value
            for key, value in header.items()
            if key in attr.fields_dict(cls).keys()
        }
        return cls(**header_check)


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
