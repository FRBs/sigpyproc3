from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs
import numpy as np
from astropy import units
from astropy.coordinates import Angle, SkyCoord

from sigpyproc import params
from sigpyproc.io import pfits, sigproc
from sigpyproc.io.bits import BitsInfo
from sigpyproc.io.fileio import FileWriter
from sigpyproc.utils import duration_string, time_after_nsamps


@attrs.define(auto_attribs=True, kw_only=True)
class Header:
    """Container object to handle observation metadata.

    Parameters
    ----------
    filename: str
        filename
    data_type: str
        data type
    nchans: int
        Number of channels
    foff: float
        Channel width in MHz
    fch1: float
        Central frequency of the first channel in MHz
    nbits: int
        Number of bits of sample
    tsamp: float
        Sampling time
    tstart: float
        Start MJD
    nsamples: int
        Number of samples
    nifs: int
        Number of polarizations
    coord: :class:`~astropy.coordinates.SkyCoord`
        Sky coordinate
    azimuth: :class:`~astropy.coordinates.Angle`
        Telescope Azimuth
    zenith: :class:`~astropy.coordinates.Angle`
        Telescope Zenith angle
    telescope: str
        Telescope name
    backend: str
        Backend name
    source: str
        Source name
    frame: str
        Frame (Topocentric)
    ibeam: int
        Beam no
    nbeams: int
        Total beams
    dm: float
        Reference DM
    period: float
        Period
    accel: float
        Acceleration
    signed: bool
        if the data is signed
    rawdatafile: str
        Original file name
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

    nifs: int = 1
    coord: SkyCoord = SkyCoord(0, 0, unit="deg")
    azimuth: Angle = Angle("0d")
    zenith: Angle = Angle("0d")
    telescope: str = "Fake"
    backend: str = "FAKE"
    source: str = "Fake"
    frame: str = "topocentric"
    ibeam: int = 0
    nbeams: int = 0
    dm: float = 0
    period: float = 0
    accel: float = 0
    signed: bool = False
    rawdatafile: str = ""

    stream_info: sigproc.StreamInfo = sigproc.StreamInfo()

    @property
    def basename(self) -> str:
        """Basename of header filename (`str`, read-only)."""
        return Path(self.filename).stem

    @property
    def extension(self) -> str:
        """Extension of header filename (`str`, read-only)."""
        return Path(self.filename).suffix

    @property
    def telescope_id(self) -> int:
        """Telescope id (`int`, read-only)."""
        return sigproc.telescope_ids.get(self.telescope, 0)

    @property
    def machine_id(self) -> int:
        """Machine id (`str`, read-only)."""
        return sigproc.machine_ids.get(self.backend, 0)

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
        """Frequency (center) of each channel(:py:obj:`~numpy.ndarray`, read-only)."""
        return np.arange(self.nchans, dtype=np.float64) * self.foff + self.fch1

    @property
    def fmax(self) -> float:
        """Highest (center) frequency channel (`float`, read-only)."""
        return self.chan_freqs.max()

    @property
    def fmin(self) -> float:
        """Lowest (center) frequency channel (`float`, read-only)."""
        return self.chan_freqs.min()

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
    def obs_date(self) -> str:
        """Observation date and time (`str`, read-only)."""
        return time_after_nsamps(self.tstart, self.tsamp).iso

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
        return time_after_nsamps(self.tstart, self.tsamp, nsamps).mjd

    def get_dmdelays(
        self,
        dm: float,
        *,
        in_samples: bool = True,
        ref_freq: str = "ch1",
    ) -> np.ndarray:
        """Get ISM disperison delays at each channel frequency for a given DM.

        Parameters
        ----------
        dm : float
            dispersion measure to calculate delays for
        in_samples : bool, optional
            flag to return delays as numbers of samples, by default True
        ref_freq : str, optional
            reference frequency to calculate delays from, by default "ch1"

        Returns
        -------
        :py:obj:`~numpy.ndarray`
            delays for middle of each channel with respect to reference frequency
        """
        if ref_freq not in {"max", "min", "center", "ch1"}:
            msg = f"reference frequency {ref_freq} not defined"
            raise ValueError(msg)

        fch_ref = getattr(self, f"f{ref_freq}")
        return params.compute_dmdelays(
            self.chan_freqs,
            dm,
            self.tsamp,
            fch_ref,
            in_samples=in_samples,
        )

    def new_header(self, update_dict: dict[str, Any] | None = None) -> Header:
        """Get a new instance of :class:`~sigpyproc.header.Header`.

        Parameters
        ----------
        update_dict : dict, optional
            values to overide existing header values, by default None

        Returns
        -------
        :class:`~sigpyproc.header.Header`
            new header information
        """
        new = attrs.asdict(self)
        if update_dict is not None:
            new.update(update_dict)
        new_checked = {
            key: value for key, value in new.items() if key in attrs.asdict(self)
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
        :class:`~sigpyproc.header.Header`
            A dedispersed version of the header
        """
        return self.new_header(
            {"dm": dm, "nchans": 1, "data_type": "time series", "nbits": 32},
        )

    def to_dict(self, *, with_properties: bool = True) -> dict[str, Any]:
        """Get a dict of all attributes including property attributes.

        Returns
        -------
        dict
            attributes
        """
        attributes = attrs.asdict(self)
        if with_properties:
            prop = {
                key: getattr(self, key)
                for key, value in vars(type(self)).items()
                if isinstance(value, property)
            }
            attributes.update(prop)
        return attributes

    def to_sigproc(self) -> dict:
        """Get a dictionary of header values in Sigproc format.

        Returns
        -------
        dict
            header values in Sigproc format
        """
        header = self.to_dict()
        sig_header = {
            key: value for key, value in header.items() if key in sigproc.header_keys
        }
        hdr_update = {
            "data_type": params.data_types.inverse[sig_header["data_type"]],
            "pulsarcentric": 1 if self.frame == "pulsarcentric" else 0,
            "barycentric": 1 if self.frame == "barycentric" else 0,
            "source_name": self.source,
            "refdm": self.dm,
            "src_dej": float(self.dec.replace(":", "")),
            "src_raj": float(self.ra.replace(":", "")),
            "za_start": self.zenith.deg,
            "az_start": self.azimuth.deg,
        }
        sig_header.update(hdr_update)
        return sig_header

    def to_string(self) -> str:
        hdr = []
        temp = "{0:<33}: {1}"
        hdr.extend(
            [
                temp.format("Data file", self.filename),
                temp.format(
                    "Header size (bytes)",
                    self.stream_info.get_combined("hdrlen"),
                ),
                temp.format(
                    "Data size (bytes)",
                    self.stream_info.get_combined("datalen"),
                ),
                temp.format("Data type", f"{self.data_type} ({self.frame})"),
                temp.format("Telescope", self.telescope),
                temp.format("Datataking Machine", self.backend),
                temp.format("Source Name", self.source),
                temp.format("Source RA (J2000)", self.ra),
                temp.format("Source DEC (J2000)", self.dec),
                temp.format("Start AZ (deg)", self.azimuth.deg),
                temp.format("Start ZA (deg)", self.zenith.deg),
            ],
        )
        if self.data_type == "filterbank":
            hdr.extend(
                [
                    temp.format("Frequency of channel 1 (MHz)", self.fch1),
                    temp.format("Channel bandwidth      (MHz)", self.foff),
                    temp.format("Number of channels", self.nchans),
                    temp.format("Number of beams", self.nbeams),
                    temp.format("Beam number", self.ibeam),
                ],
            )
        elif self.data_type == "time series":
            hdr.extend(
                [
                    temp.format("Reference DM (pc/cc)", self.dm),
                    temp.format("Reference frequency    (MHz)", self.fch1),
                    temp.format("Number of channels", self.nchans),
                ],
            )
        print_dur, print_unit = duration_string(self.tobs).split()
        hdr.extend(
            [
                temp.format("Time stamp of first sample (MJD)", self.tstart),
                temp.format("Gregorian date (YYYY-MM-DD)", self.obs_date),
                temp.format(
                    "Sample time (us)",
                    (self.tsamp * units.second).to(units.microsecond).value,
                ),
                temp.format("Number of samples", self.nsamples),
                temp.format(f"Observation length {print_unit}", print_dur),
                temp.format("Number of bits per sample", self.nbits),
                temp.format("Number of IFs", self.nifs),
            ],
        )
        return "\n".join(hdr)

    def prep_outfile(
        self,
        filename: str,
        *,
        updates: dict[str, Any] | None = None,
        nbits: int | None = None,
        rescale: bool = False,
    ) -> FileWriter:
        """Prepare a file to have sigproc format data written to it.

        Parameters
        ----------
        filename : str
            name of new file
        updates : dict, optional
            values to overide existing header values, by default None
        nbits : int, optional
            Number of bits per output sample, by default None
        scale_fac : float, optional
            Additional scale factor to apply to data, by default 1.0
        rescale : bool, optional
            whether to rescale the data using the nbit-dependent values,
            by default True

        Returns
        -------
        :class:`~sigpyproc.io.fileio.FileWriter`
            a prepared file
        """
        if nbits is None:
            nbits = self.nbits
        if updates is None:
            updates = {}
        if nbits != self.nbits:
            updates["nbits"] = nbits
        new_hdr = self.new_header(updates)
        out_file = FileWriter(
            filename,
            mode="w+",
            nbits=nbits,
            rescale=rescale,
        )
        new_hdr_binary = sigproc.encode_header(new_hdr.to_sigproc())
        out_file.write(new_hdr_binary)
        return out_file

    def make_inf(self, outfile: str | None = None) -> str | None:
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
        inf_str = "\n".join(inf)
        if outfile is None:
            return inf_str
        with Path(outfile).open("w+") as fp:
            fp.write(inf_str)
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
        :class:`~sigpyproc.header.Header`
            observational metadata
        """
        header: dict[str, Any] = {}
        with Path(filename).open("r") as fp:
            lines = fp.readlines()

        for line in lines:
            desc = line.split("=")[0].strip()
            val = line.split("=")[-1].strip()
            if desc not in list(params.presto_inf.keys()):
                continue
            key, keytype, _keyformat = params.presto_inf[desc]
            header[key] = keytype(val)

        hdr_update = {
            "filename": header["basename"],
            "data_type": "time series",
            "fch1": header["freq_low"] + header["foff"] * header["nchans"],
            "nbits": 32,
            "nchans": 1,
            "nifs": 1,
            "coord": SkyCoord(
                header["ra"],
                header["dec"],
                unit=(units.hourangle, units.deg),
            ),
        }
        header.update(hdr_update)
        header_check = {
            key: value for key, value in header.items() if key in attrs.fields_dict(cls)
        }
        return cls(**header_check)

    @classmethod
    def from_sigproc(
        cls,
        filenames: str | list[str],
        *,
        check_contiguity: bool = True,
    ) -> Header:
        """Parse the metadata from Sigproc-style file/sequential files.

        Parameters
        ----------
        filenames : list
            sigproc filterbank files containing the header

        Returns
        -------
        :class:`~sigpyproc.header.Header`
            observational metadata

        """
        header = sigproc.parse_header_multi(
            filenames,
            check_contiguity=check_contiguity,
        )
        frame = "pulsarcentric" if header.get("pulsarcentric") else "topocentric"
        frame = "barycentric" if header.get("barycentric") else "topocentric"
        hdr_update = {
            "data_type": params.data_types[header.get("data_type", 1)],
            "telescope": sigproc.telescope_ids.inv.get(
                header.get("telescope_id", 0),
                "Fake",
            ),
            "backend": sigproc.machine_ids.inv.get(header.get("machine_id", 0), "FAKE"),
            "source": header.get("source_name", "Fake"),
            "dm": header.get("refdm", 0),
            "foff": header.get("foff", 0),
            "coord": sigproc.parse_radec(
                header.get("src_raj", 0),
                header.get("src_dej", 0),
            ),
            "azimuth": Angle(header.get("az_start", 0) * units.deg),
            "zenith": Angle(header.get("za_start", 0) * units.deg),
            "frame": frame,
        }
        header.update(hdr_update)
        header_check = {
            key: value for key, value in header.items() if key in attrs.fields_dict(cls)
        }
        return cls(**header_check)

    @classmethod
    def from_pfits(cls, filename: str) -> Header:
        """Parse the metadata from a PSRFITS file.

        Parameters
        ----------
        filename : str
            the name of the PSRFITS file containing the header

        Returns
        -------
        :class:`~sigpyproc.header.Header`
            observational metadata
        """
        primary_hdr = pfits.PrimaryHdr(filename)
        subint_hdr = pfits.SubintHdr(filename)

        header: dict[str, Any] = {}
        hdr_update = {
            "filename": filename,
            "data_type": "filterbank",
            "nchans": subint_hdr.nchans,
            "foff": subint_hdr.freqs.foff,
            "fch1": subint_hdr.freqs.fch1,
            "nbits": subint_hdr.nbits,
            "tsamp": subint_hdr.tsamp,
            "tstart": primary_hdr.tstart.mjd,
            "nsamples": subint_hdr.nsamples,
            "coord": primary_hdr.coord,
            "telescope": primary_hdr.telescope,
            "backend": primary_hdr.backend.name,
            "source": primary_hdr.source,
        }
        header.update(hdr_update)
        header_check = {
            key: value for key, value in header.items() if key in attrs.fields_dict(cls)
        }
        return cls(**header_check)
