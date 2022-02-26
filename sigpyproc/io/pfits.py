from __future__ import annotations
import attrs
import numpy as np

from astropy import units
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle, EarthLocation, SkyCoord

from sigpyproc.io.bits import BitsInfo, unpack
from sigpyproc.utils import FrequencyChannels

pol_type_to_state = {
    "XXYY": "PPQQ",
    "LLRR": "PPQQ",
    "AABB": "PPQQ",
    "STOKE": "Stokes",
    "XXYYCRCI": "Coherence",
    "LLRRCRCI": "Coherence",
    "AABBCRCI": "Coherence",
    "INTEN": "Intensity",
    "AA+BB": "Intensity",
}

npol_to_state = {1: "Intensity", 2: "PPQQ", 4: "Stokes"}


@attrs.define(auto_attribs=True, frozen=True, slots=True, kw_only=True)
class Receiver(object):
    """Receiver information.

    Attributes
    ----------
    name : str
        Name of the receiver.
    nrcvr : int
        Number of receptors.
    basis : str
        Basis of receptors.
    hand: int
        Hand of receptor basis.
    sa: float
        Symmetry angle of receptor basis.
    rph: float
        Reference source phase.
    tracking_mode: str
        Tracking mode of the receiver platform.
    tracking_angle: float
        Position angle tracked by the receiver.
    """

    name: str
    nrcvr: int
    basis: str
    hand: int
    sa: float
    rph: float
    tracking_mode: str
    tracking_angle: float


@attrs.define(auto_attribs=True, frozen=True, slots=True, kw_only=True)
class Backend(object):
    """Backend information.

    Attributes
    ----------
    name : str
        Name of the backend instrument.
    phase: float
        Phase convention of backend.
    dcc: float
        Downconversion conjugation corrected.
    delay: float
        Backend propn delay from digi. input.
    tcycle: float
        Native cycle time of correlation system.
    configfile: str
        Name of a configuration file used to set up the backend system.
    """

    name: str
    phase: float
    dcc: int
    delay: float
    tcycle: float
    configfile: str


class PrimaryHdr(object):
    def __init__(self, filename: str) -> None:
        header = fits.getheader(filename, extname="PRIMARY")
        self._check_header(header)
        self._header = header

    @property
    def header(self) -> fits.Header:
        """astropy.io.fits.Header: Primary header."""
        return self._header

    @property
    def observer(self) -> str:
        """str: Observer name(s)."""
        return self.header["OBSERVER"]

    @property
    def project_id(self) -> str:
        """str: Project name."""
        return self.header["PROJID"]

    @property
    def telescope(self) -> str:
        """str: Telescope name."""
        return self.header["TELESCOP"]

    @property
    def location(self) -> EarthLocation:
        """astropy.coordinates.EarthLocation: Antenna location."""
        return EarthLocation.from_geocentric(
            self.header["ANT_X"], self.header["ANT_Y"], self.header["ANT_Z"], unit=units.m
        )

    @property
    def receiver(self) -> Receiver:
        """sigpyproc.io.hdu.Receiver: Receiver information."""
        return Receiver(
            name=self.header["FRONTEND"],
            nrcvr=self.header["NRCVR"],
            basis=self.header["FD_POLN"],
            hand=self.header["FD_HAND"],
            sa=self.header["FD_SANG"],
            rph=self.header["FD_XYPH"],
            tracking_mode=self.header["FD_MODE"],
            tracking_angle=self.header["FA_REQ"],
        )

    @property
    def backend(self) -> Backend:
        """sigpyproc.io.hdu.Backend: Backend information."""
        return Backend(
            name=self.header["BACKEND"],
            phase=self.header["BE_PHASE"],
            dcc=self.header["BE_DCC"],
            delay=self.header["BE_DELAY"],
            tcycle=self.header["TCYCLE"],
            configfile=self.header["BECONFIG"],
        )

    @property
    def ibeam(self) -> int:
        """int: Beam number."""
        return self.header.get("IBEAM", 0)

    @property
    def obs_mode(self) -> str:
        """str: Observation mode."""
        return self.header["OBS_MODE"]

    @property
    def date_obs(self) -> Time:
        """astropy.time.Time: Observation start date."""
        return Time(
            self.header["DATE-OBS"], format="isot", scale="utc", location=self.location
        )

    @property
    def freqs(self) -> FrequencyChannels:
        """sigpyproc.utils.FrequencyChannels: Frequency channels."""
        return FrequencyChannels.from_pfits(
            self.header["OBSFREQ"], self.header["OBSBW"], self.header["OBSNCHAN"]
        )

    @property
    def chan_dm(self) -> float:
        """float: Dispersion measure value used for on-line (normally coherent) dedispersion."""
        return self.header.get("CHAN_DM", 0)

    @property
    def source(self) -> str:
        """str: Source name or label for an observation."""
        return self.header["SRC_NAME"]

    @property
    def coord(self) -> SkyCoord:
        """astropy.coordinates.SkyCoord: Source coordinates."""
        return SkyCoord(
            self.header["RA"], self.header["DEC"], unit=(units.hourangle, units.deg)
        )

    @property
    def tstart(self):
        """astropy.time.Time: Observation start time."""
        return Time(
            self.header["STT_IMJD"], format="mjd", scale="utc", location=self.location
        ) + TimeDelta(
            float(self.header["STT_SMJD"]), float(self.header["STT_OFFS"]), format="sec"
        )

    def _check_header(self, header):
        assert header["FITSTYPE"] == "PSRFITS", "Not a PSRFITS file."
        assert header["OBS_MODE"] == "SEARCH", "Not a search-mode file."


class SubintHdr(object):
    def __init__(self, filename: str) -> None:
        with fits.open(filename) as hdul:
            header = hdul["SUBINT"].header
            sub_data = hdul["SUBINT"].data[0]
        self._check_header(header)
        self._parse_data(sub_data)
        self._header = header

    @property
    def header(self) -> fits.Header:
        """astropy.io.fits.Header: SUBINT header."""
        return self._header

    @property
    def subint_width(self) -> int:
        """int: Width of subint table in bytes."""
        return self.header["NAXIS1"]

    @property
    def nsubint(self) -> int:
        """int: Number of rows (subints) in subint table."""
        return self.header["NAXIS2"]

    @property
    def poln_type(self) -> str:
        """str: Polarisation identifier (e.g., AABBCRCI, AA+BB)."""
        return self.header["POL_TYPE"]

    @property
    def poln_state(self) -> str:
        """str: Polarisation state (e.g., Coherence)."""
        state = pol_type_to_state.get(self.poln_type, None)
        if state is None:
            state = npol_to_state[self.npol]
        return state

    @property
    def npol(self) -> int:
        """int: Number of polarisation products in the DATA table."""
        return self.header["NPOL"]

    @property
    def tsamp(self) -> float:
        """float: Time per bin or sample (seconds)."""
        return self.header["TBIN"]

    @property
    def nbits(self) -> int:
        """int: Number of bits per sample."""
        return self.header["NBITS"]

    @property
    def zero_off(self) -> float:
        """float: Zero offset for data (the nominal data zero in digitiser units)."""
        return self.header.get("ZERO_OFF", 0)

    @property
    def signint(self) -> int:
        """int: Flag to indicate that the data values are signed integers."""
        return self.header["SIGNINT"]

    @property
    def subint_offset(self) -> int:
        """int: Subint offset for contiguous SEARCH-mode files."""
        return self.header["NSUBOFFS"]

    @property
    def nchans(self) -> int:
        """int: Number of frequency channels for each polarisation in the DATA table."""
        return self.header["NCHAN"]

    @property
    def chan_bw(self) -> float:
        """float: Channel bandwidth in MHz."""
        return self.header["CHAN_BW"]

    @property
    def channel_offset(self) -> int:
        """int: Channel offset for contiguous SEARCH-mode files."""
        return self.header.get("NCHNOFFS", 0)

    @property
    def subint_samples(self) -> int:
        """int: Number of samples per subint or table row."""
        return self.header["NSBLK"]

    @property
    def nsamples(self) -> int:
        """int: Total number of valid samples in a search-mode file."""
        return self.header.get("NSTOT", self.subint_samples * self.nsubint)

    @property
    def subint_shape(self) -> tuple[int, int, int]:
        return (self.subint_samples, self.npol, self.nchans)

    @property
    def sub_hdr(self) -> dict:
        return self._sub_hdr

    @property
    def tsubint(self) -> float:
        """float: Time per subint or table row (seconds)."""
        return self.sub_hdr.get("TSUBINT", self.subint_samples * self.tsamp)

    @property
    def offs_sub(self) -> float:
        """float: Time since the observation start at the centre of first sub-integration (seconds)."""
        return self.sub_hdr.get("OFFS_SUB", 0)

    @property
    def azimuth(self) -> Angle:
        """astropy.coordinates.Angle: Azimuth (degrees)."""
        return Angle(self.sub_hdr.get("TEL_AZ", 0), unit=units.deg)

    @property
    def zenith(self) -> Angle:
        """astropy.coordinates.Angle: Zenith (degrees)."""
        return Angle(self.sub_hdr.get("TEL_ZEN", 0), unit=units.deg)

    @property
    def freqs(self) -> FrequencyChannels:
        """sigpyproc.utils.FrequencyChannels: Frequency channels for first subint."""
        return FrequencyChannels(self._sub_freqs[: self.nchans])

    def _parse_data(self, sub_data: fits.FITS_record) -> None:
        unwanted_keys = {"DAT_WTS", "DAT_OFFS", "DAT_SCL", "DAT_FREQ", "DATA"}
        wanted_keys = [key for key in sub_data.array.names if key not in unwanted_keys]
        self._sub_hdr = {key: sub_data.field(key) for key in wanted_keys}
        self._sub_freqs = sub_data.field("DAT_FREQ")

    def _check_header(self, header: fits.Header) -> None:
        assert header["EXTNAME"] == "SUBINT", "Not a subint hdu."


class PFITSFile(object):
    """Handle a PSRFITS file.

    Parameters
    ----------
    filename : str
        Filename of the PSRFITS file.
    """

    def __init__(self, filename) -> None:
        self._filename = filename
        self._fits = fits.open(filename, mode="readonly", memmap=True)
        self._primary_hdr = PrimaryHdr(filename)
        self._subint_hdr = SubintHdr(filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # del self._fits['SUBINT'].data  # noqa:
        self._fits.close()

    @property
    def filename(self) -> str:
        """str: Name of file."""
        return self._filename

    @property
    def pri_hdr(self) -> PrimaryHdr:
        """PrimaryHdr: Primary header."""
        return self._primary_hdr

    @property
    def sub_hdr(self) -> SubintHdr:
        """SubintHdr: Subint header."""
        return self._subint_hdr

    @property
    def bitsinfo(self) -> BitsInfo:
        """BitsInfo: Bits information."""
        return BitsInfo(self.sub_hdr.nbits)

    def read_subints(
        self,
        startsub: int,
        nsubs: int,
        poln_select: int = 1,
        scloffs: bool = True,
        weights: bool = True,
    ) -> np.ndarray:
        """Read the digitised data in a given polarization format from a range of PSRFITS SUBINT table.

        Parameters
        ----------
        startsub : int
            index of start row (subint) to read from the SUBINT table
        nsubs : int
            number of subints to read from the SUBINT table
        poln_select : int, optional
            1=PP+QQ, 2=PP,QQ, 3=(PP+QQ)^2 4=PP,QQ,PQ,QP, by default 1
        scloffs : bool, optional
            apply scales and offsets when unpacking data, by default False
        weights : bool, optional
            apply weights when unpacking data, by default False

        Returns
        -------
        :py:obj:`numpy.ndarray`
            subint (row) data in float32 if scloffs or weights applied,
            otherwise in uint8 with shape (nsamps, nchan).
        """
        data_list = []
        for isub in range(startsub, startsub + nsubs):
            sdata = self.read_subint_pol(
                isub, poln_select=poln_select, scloffs=scloffs, weights=weights
            )
            data_list.append(sdata)
        data = np.concatenate(data_list)

        if self.sub_hdr.freqs.foff > 0:
            data = np.fliplr(data)

        return data

    def read_subint_pol(
        self, isub: int, poln_select: int = 1, scloffs: bool = True, weights: bool = True
    ) -> np.ndarray:
        """Read the digitised data in a given polarization format from the PSRFITS SUBINT table.

        Parameters
        ----------
        isub : int
            index of row (subint) to read from the SUBINT table
        poln_select : int, optional
            1=PP+QQ, 2=PP,QQ, 3=(PP+QQ)^2 4=PP,QQ,PQ,QP, by default 1
        scloffs : bool, optional
            apply scales and offsets when unpacking data, by default True
        weights : bool, optional
            apply weights when unpacking data, by default True

        Returns
        -------
        :py:obj:`numpy.ndarray`
            subint (row) data in float32 if scloffs or weights applied,
            otherwise in uint8 with shape (nsamps, nchan).
        """
        sdata = self.read_subint(isub, scloffs=scloffs, weights=weights)
        if self.sub_hdr.poln_state == "Coherence":
            scale = 1.0 / np.sqrt(2.0)
            data_shape = (self.sub_hdr.subint_samples, self.sub_hdr.nchans)
            if poln_select == 1:
                data = np.zeros(data_shape, dtype=np.float32)
                data = data + (sdata[:, 0, :] + sdata[:, 1, :]) * scale
            elif poln_select == 2:
                data = sdata[:, 0:2, :]
            elif poln_select == 3:
                data = np.zeros(data_shape, dtype=np.float32)
                data = data + (sdata[:, 0, :] + sdata[:, 1, :]) * scale
            elif poln_select == 4:
                data = sdata
        elif self.sub_hdr.poln_state == "Stokes":
            data = sdata[:, 0, :]
        elif self.sub_hdr.poln_state == "Intensity":
            data = sdata[:, 0, :].squeeze()

        return data

    def read_subint(
        self, isub: int, scloffs: bool = True, weights: bool = True
    ) -> np.ndarray:
        """Read the digitised data from the PSRFITS SUBINT table.

        Parameters
        ----------
        isub : int
            index of row (subint) to read from the SUBINT table
        scloffs : bool, optional
            apply scales and offsets when unpacking data, by default True
        weights : bool, optional
            apply weights when unpacking data, by default True

        Returns
        -------
        :py:obj:`numpy.ndarray`
            subint (row) data in float32 if scale_and_offset or weights applied,
            otherwise in uint8 with shape (nsamps, npol, nchan).
        """
        sdata = self._fits["SUBINT"].data[isub]["DATA"]  # noqa: WPS219
        sdata = sdata.squeeze()
        if self.bitsinfo.unpack:
            data = unpack(sdata.ravel(), self.bitsinfo.nbits)
            data = data.reshape(
                (sdata.shape[0] * self.bitsinfo.bitfact, sdata.shape[1], sdata.shape[2])
            )
        else:
            data = np.array(sdata)
        assert (
            data.shape == self.sub_hdr.subint_shape
        ), f"DATA column ordering {data.shape} is not TPF"

        if scloffs or weights:
            data = data.astype(np.float32, copy=False)
        if scloffs:
            data -= self.sub_hdr.zero_off  # TODO This will not work for 2-bit data.
            data = data * self.read_scales(isub) + self.read_offsets(isub)
        if weights:
            data *= self.read_weights(isub)

        # del self._fits['SUBINT'].data  # noqa: Magic happens here
        return data

    def read_freqs(self, isub: int) -> FrequencyChannels:
        """Read the Channel centre frequency from the PSRFITS SUBINT table.

        Parameters
        ----------
        isub : int
            index of row (subint) to read from the SUBINT table

        Returns
        -------
        FrequencyChannels
            Centre frequency for each channel in MHz (NCHAN)
        """
        freqs = self._fits["SUBINT"].data[isub]["DAT_FREQ"]  # noqa: WPS219
        return FrequencyChannels(freqs[: self.sub_hdr.nchans])

    def read_weights(self, isub: int) -> np.ndarray:
        """Read channel weight (DAT_WTS) column from the SUBINT table.

        Parameters
        ----------
        isub : int
            index of row (subint) to read from the SUBINT table

        Returns
        -------
        :py:obj:`numpy.ndarray`
            Weights for each channel in the range 0-1 (NCHAN)
        """
        weights = self._fits["SUBINT"].data[isub]["DAT_WTS"]  # noqa: WPS219
        return weights[: self.sub_hdr.nchans]

    def read_scales(self, isub: int) -> np.ndarray:
        """Read channel data scale factor (DAT_SCL) column from the SUBINT table.

        Parameters
        ----------
        isub : int
            index of row (subint) to read from the SUBINT table

        Returns
        -------
        :py:obj:`numpy.ndarray`
            Data scale factor for each channel (NCHAN*NPOL)
        """
        scales = self._fits["SUBINT"].data[isub]["DAT_SCL"]  # noqa: WPS219
        scales = scales[: self.sub_hdr.npol * self.sub_hdr.nchans]
        return scales.reshape(self.sub_hdr.npol, self.sub_hdr.nchans)

    def read_offsets(self, isub: int) -> np.ndarray:
        """Read channel data offset (DAT_OFFS) column from the SUBINT table.

        Parameters
        ----------
        isub : int
            index of row (subint) to read from the SUBINT table

        Returns
        -------
        :py:obj:`numpy.ndarray`
            Data offset for each channel (NCHAN*NPOL)
        """
        offsets = self._fits["SUBINT"].data[isub]["DAT_OFFS"]  # noqa: WPS219
        offsets = offsets[: self.sub_hdr.npol * self.sub_hdr.nchans]
        return offsets.reshape(self.sub_hdr.npol, self.sub_hdr.nchans)
