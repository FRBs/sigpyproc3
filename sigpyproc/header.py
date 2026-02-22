from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
from astropy import units
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time, TimeDelta

from sigpyproc import params, utils
from sigpyproc.io import fbh5, pfits, sigproc
from sigpyproc.io.bits import BitsInfo
from sigpyproc.io.fileio import FileWriter

if TYPE_CHECKING:
    from collections.abc import Sequence


@attrs.frozen(auto_attribs=True, kw_only=True)
class Header:
    """Container object to handle observation metadata.

    Parameters
    ----------
    filename : str
        Name of the header file.
    data_type : str
        Type of data (filterbank, time series).
    nchans : int
        Number of frequency channels.
    foff : float
        Frequency channel width in MHz.
    fch1 : float
        Central frequency of the first channel in MHz.
    nbits : int
        Number of bits per sample.
    tsamp : float
        Sampling time in seconds.
    tstart : float
        Start time of the observation in MJD.
    nsamples : int
        Number of time samples in the observation.
    nifs : int, optional
        Number of polarizations, by default 1.
    coord : ~astropy.coordinates.SkyCoord, optional
        Source sky coordinates, by default (0, 0).
    azimuth : ~astropy.coordinates.Angle, optional
        Telescope Azimuth angle, by default 0.
    zenith : ~astropy.coordinates.Angle, optional
        Telescope Zenith angle, by default 0.
    telescope : str, optional
        Telescope name, by default "Fake".
    backend : str, optional
        Telescope backend name, by default "FAKE".
    source : str, optional
        Source name, by default "Fake".
    frame : str, optional
        Reference frame, by default "topocentric".
    ibeam : int, optional
        Telescope beam number, by default 0.
    nbeams : int, optional
        Number of beams, by default 0.
    dm : float, optional
        Reference Dispersion Measure, by default 0.
    period : float, optional
        Rotation period of the source, by default 0.
    accel : float, optional
        Binary acceleration of the source, by default 0.
    signed : bool, optional
        Whether the data is signed, by default False.
    rawdatafile : str, optional
        Name of the original raw data file, by default "".
    stream_info : :class:`~sigpyproc.io.sigproc.StreamInfo`, optional
        File stream information, by default `sigproc.StreamInfo()`.

    Attributes
    ----------
    basename
    extension
    telescope_id
    machine_id
    bandwidth
    ftop
    fbottom
    fcenter
    chan_freqs
    fmax
    fmin
    dtype
    tobs
    ra
    dec
    obs_date
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
    coord: SkyCoord = attrs.field(
        default=SkyCoord(0, 0, unit="deg"),
        validator=attrs.validators.instance_of(SkyCoord),
    )
    azimuth: Angle = attrs.field(
        default=Angle("0d"),
        validator=attrs.validators.instance_of(Angle),
    )
    zenith: Angle = attrs.field(
        default=Angle("0d"),
        validator=attrs.validators.instance_of(Angle),
    )
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

    stream_info: sigproc.StreamInfo = attrs.field(default=sigproc.StreamInfo())

    @property
    def basename(self) -> str:
        """Basename of header filename.

        Returns
        -------
        str
            Header filename without extension.
        """
        return Path(self.filename).stem

    @property
    def extension(self) -> str:
        """Extension of header filename.

        Returns
        -------
        str
            Header filename extension.
        """
        return Path(self.filename).suffix

    @property
    def telescope_id(self) -> int:
        """Telescope id in Sigproc format.

        Returns
        -------
        int
            Sigproc telescope id.
        """
        return sigproc.telescope_ids.get(self.telescope, 0)

    @property
    def machine_id(self) -> int:
        """Machine id in Sigproc format.

        Returns
        -------
        int
            Sigproc machine id.
        """
        return sigproc.machine_ids.get(self.backend, 0)

    @property
    def bandwidth(self) -> float:
        """Frequency bandwidth in MHz.

        Returns
        -------
        float
            Bandwidth in MHz.
        """
        return abs(self.foff) * self.nchans

    @property
    def ftop(self) -> float:
        """Edge frequency of the top frequency channel.

        Returns
        -------
        float
            Edge frequency at the top of the band.
        """
        return self.fch1 - 0.5 * self.foff

    @property
    def fbottom(self) -> float:
        """Edge frequency of the bottom frequency channel.

        Returns
        -------
        float
            Edge frequency at the bottom of the band.
        """
        return self.ftop + self.foff * self.nchans

    @property
    def fcenter(self) -> float:
        """Central frequency of the whole band.

        Returns
        -------
        float
            Central frequency.
        """
        return self.ftop + 0.5 * self.foff * self.nchans

    @property
    def chan_freqs(self) -> np.ndarray:
        """Center frequency of each channel.

        Returns
        -------
        ndarray
            Channel center frequencies.
        """
        return np.arange(self.nchans, dtype=np.float32) * self.foff + self.fch1

    @property
    def fmax(self) -> float:
        """Center frequency of the maximum frequency channel.

        Returns
        -------
        float
            Maximum frequency in the band.
        """
        return self.chan_freqs.max()

    @property
    def fmin(self) -> float:
        """Center frequency of the minimum frequency channel.

        Returns
        -------
        float
            Minimum frequency in the band.
        """
        return self.chan_freqs.min()

    @property
    def dtype(self) -> np.dtype:
        """Type of the data in file.

        Returns
        -------
        dtype
            Data type.
        """
        return BitsInfo(self.nbits).dtype

    @property
    def tobs(self) -> float:
        """Total time of the observation.

        Returns
        -------
        float
            Observation time.
        """
        return self.tsamp * self.nsamples

    @property
    def ra(self) -> str:
        """Right Ascension in string representation.

        Returns
        -------
        str
            Right Ascension.
        """
        return self.coord.ra.to_string(unit="hourangle", sep=":", pad=True)

    @property
    def dec(self) -> str:
        """Declination in string representation.

        Returns
        -------
        str
            Declination.
        """
        return self.coord.dec.to_string(unit="deg", sep=":", pad=True)

    @property
    def obs_time(self) -> Time:
        """Observation time in Astropy Time format.

        Returns
        -------
        Time
            Observation time.
        """
        precision = int(np.ceil(abs(np.log10(self.tsamp))))
        return Time(self.tstart, format="mjd", scale="utc", precision=precision)

    @property
    def obs_date(self) -> str:
        """Observation date and time in ISO format.

        Returns
        -------
        str
            Observation date and time.
        """
        return self.obs_time.iso

    def mjd_after_nsamps(self, nsamps: int) -> float:
        """Compute the MJD after nsamps have elapsed.

        Parameters
        ----------
        nsamps : int
            Number of samples elapsed since start of observation.

        Returns
        -------
        float
            Modified Julian Date.
        """
        new_time = self.obs_time + TimeDelta(nsamps * self.tsamp, format="sec")
        return new_time.mjd

    def get_dmdelays(
        self,
        dm: float | np.ndarray,
        ref_freq: str | float = "fch1",
        *,
        in_samples: bool = True,
    ) -> np.ndarray:
        """Get ISM dispersion delays for given DM value(s).

        Parameters
        ----------
        dm : float | np.ndarray
            Dispersion measure(s) to calculate delays for.
        ref_freq : str | float, optional
            Reference frequency to calculate delays from, by default "fch1".
            Accepted inputs are "fmax", "fmin", "fcenter", "fch1" or a number in MHz.
        in_samples : bool, optional
            Flag to return delays as numbers of samples, by default True.

        Returns
        -------
        ndarray
            Dispersion delays at middle of each channel with respect to ``ref_freq``.

            If dm is a scalar, returns a 1D array of delays. If dm is an array,
            returns a 2D array with shape ``(len(dm), len(freqs))``.
        """
        if isinstance(ref_freq, str):
            if ref_freq not in {"fmax", "fmin", "fcenter", "fch1"}:
                msg = f"reference frequency {ref_freq} not defined"
                raise ValueError(msg)
            fch_ref = float(getattr(self, f"{ref_freq}"))
        elif isinstance(ref_freq, int | float):
            fch_ref = float(ref_freq)
        else:
            msg = "ref_freq must be a string or a number"
            raise TypeError(msg)

        return params.compute_dmdelays(
            self.chan_freqs,
            dm,
            self.tsamp,
            fch_ref,
            in_samples=in_samples,
        )

    def get_dmsmearing(
        self,
        dm: float | np.ndarray,
        *,
        in_samples: bool = True,
    ) -> np.ndarray:
        """Get ISM smearing for given DM value(s).

        Parameters
        ----------
        dm : float | np.ndarray
            Dispersion measure(s) to calculate smearing for.
        in_samples : bool, optional
            Flag to return smearing as numbers of samples, by default True.

        Returns
        -------
        ndarray
            DM smearing in the frequency channels due to finite bandwidth.

            If dm is a scalar, returns a 1D array of smearing. If dm is an array,
            returns a 2D array with shape ``(len(dm), len(freqs))``.
        """
        return params.compute_dmsmearing(
            self.chan_freqs,
            dm,
            self.tsamp,
            in_samples=in_samples,
        )

    def new_header(self, update_dict: dict[str, Any] | None = None) -> Header:
        """Get a new updated instance of `Header`.

        This is the preferred way to update the header.

        Parameters
        ----------
        update_dict : dict, optional
            Values to overide existing header values, by default None.

        Returns
        -------
        Header
            A new instance of the header with updated values.
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
            Dispersion measure to dedisperse to.

        Returns
        -------
        :class:`~sigpyproc.header.Header`
            A new header with updated DM value.
        """
        return self.new_header(
            {"dm": dm, "nchans": 1, "data_type": "time series", "nbits": 32},
        )

    def to_dict(self, *, with_properties: bool = True) -> dict[str, Any]:
        """Get a dictionary of all attributes.

        Parameters
        ----------
        with_properties : bool, optional
            Whether to include properties in the output, by default True.

        Returns
        -------
        dict[str, Any]
            All attributes of the header.
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
        """Get header dictionary in Sigproc format.

        Returns
        -------
        dict
            Sigproc formatted header values.
        """
        header = self.to_dict()
        sig_header = {
            key: value for key, value in header.items() if key in sigproc.SIGPROC_SCHEMA
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
        """Get a string representation of the header.

        Returns
        -------
        str
            A string representation of the header.
        """
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
        print_dur, print_unit = utils.duration_string(self.tobs).split()
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
        """Prepare a file with Sigproc format data.

        This is the preferred way to write a new Sigproc file.

        Parameters
        ----------
        filename : str
            Name of new file.
        updates : dict, optional
            Values to overide existing header values, by default None.
        nbits : int, optional
            Number of bits per output sample, by default None.
        rescale : bool, optional
            Whether to rescale the data using the nbit-dependent values,
            by default True.

        Returns
        -------
        :class:`~sigpyproc.io.fileio.FileWriter`
            A file writer object to write data to.
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

    def make_inf(self, outfile: str | None = None) -> str:
        """Make a presto format ``.inf`` file.

        Parameters
        ----------
        outfile : str, optional
            Name of the output file, by default None.

        Returns
        -------
        str
            A string representation of ``.inf`` data.
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
        if outfile is not None:
            with Path(outfile).open("w+") as fp:
                fp.write(inf_str)
        return inf_str

    @classmethod
    def from_inffile(cls, filename: str | Path) -> Header:
        """Parse the metadata from a presto ``.inf`` file.

        Parameters
        ----------
        filename : str | Path
            Name of the ``.inf`` file containing the header.

        Returns
        -------
        Header
            Observational metadata.
        """
        filepath = utils.validate_path(filename)
        header: dict[str, Any] = {}
        with filepath.open("r") as fp:
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
        filenames: str | Path | Sequence[str | Path],
        *,
        check_contiguity: bool = True,
    ) -> Header:
        """Parse the metadata from Sigproc-style files.

        Parameters
        ----------
        filenames : str | Path | Sequence[str | Path]
            Sigproc filterbank files containing the header.
        check_contiguity : bool, optional
            Check if the files are contiguous, by default True.

        Returns
        -------
        Header
            Observational metadata.
        """
        header, sinfo = sigproc.parse_header_multi(
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
        return cls(**header_check, stream_info=sinfo)

    @classmethod
    def from_pfits(cls, filename: str) -> Header:
        """Parse the metadata from a PSRFITS file.

        Parameters
        ----------
        filename : str
            Name of the PSRFITS file containing the header.

        Returns
        -------
        Header
            Observational metadata.
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

    @classmethod
    def from_fbh5(cls, filename: str) -> Header:
        """Parse the metadata from a filterbank HDF5 file.

        Parameters
        ----------
        filename : str
            Name of the HDF5 file containing the header.

        Returns
        -------
        Header
            Observational metadata.
        """
        header = fbh5.parse_header(filename)
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
