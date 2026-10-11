"""Header class for handling observation metadata.

This module contains the Header class for handling observation metadata.
"""

from __future__ import annotations

import logging
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

    from bidict import bidict


logger = logging.getLogger(__name__)

# Header attributes and the Sigproc keys they are read from and written as. An
# attribute is written only when it is in ``Header.known_keys``.
_WRITTEN_KEYS: dict[str, str] = {
    "dm": "refdm",
    "period": "period",
    "signed": "signed",
    "ibeam": "ibeam",
    "nbeams": "nbeams",
    "rawdatafile": "rawdatafile",
    "azimuth": "az_start",
    "zenith": "za_start",
}

_ATTR_SOURCE_KEYS: dict[str, tuple[str, ...]] = {
    "telescope": ("telescope_id",),
    "backend": ("machine_id",),
    "source": ("source_name",),
    "coord": ("src_raj", "src_dej"),
    **{attr: (key,) for attr, key in _WRITTEN_KEYS.items()},
}

_FRAME_FLAGS = ("barycentric", "pulsarcentric")
_FRAMES = ("topocentric", *_FRAME_FLAGS)

# Attribute values that mean "not set". An attribute equal to its default is not
# written unless it is in ``known_keys``.
_DEFAULTS: dict[str, Any] = {
    "source": "Fake",
    "dm": 0,
    "period": 0,
    "signed": False,
    "ibeam": 0,
    "nbeams": 1,
    "rawdatafile": "",
}


def _is_default(attr: str, value: object) -> bool:
    """Return True if ``value`` is the default of optional attribute ``attr``."""
    if isinstance(value, SkyCoord):
        return value.ra.deg == 0 and value.dec.deg == 0
    if isinstance(value, Angle):
        return value.deg == 0
    return value == _DEFAULTS[attr]


def _default_known_keys(header: Header) -> frozenset[str]:
    """Return the attributes that differ from their defaults, for code-built headers."""
    known = {
        attr
        for attr in (*_DEFAULTS, "coord", "azimuth", "zenith")
        if not _is_default(attr, getattr(header, attr))
    }
    return frozenset(known)


def _name_from_id(table: bidict[str, int], code: int) -> str:
    return table.inv.get(code, "unknown")


def _sigproc_values(header: dict[str, Any]) -> dict[str, Any]:
    """Convert a parsed Sigproc header into ``Header`` attribute values.

    Only keys present in the file are set, so absent keys keep the ``Header``
    defaults and are not treated as known.
    """
    values = dict(header)
    values["data_type"] = params.data_types[header.get("data_type", 1)]
    values["foff"] = header.get("foff", 0)
    if "telescope_id" in header:
        values["telescope"] = _name_from_id(
            sigproc.telescope_ids, header["telescope_id"]
        )
        values["sigproc_telescope_id"] = header["telescope_id"]
    if "machine_id" in header:
        values["backend"] = _name_from_id(sigproc.machine_ids, header["machine_id"])
        values["sigproc_machine_id"] = header["machine_id"]
    if "source_name" in header:
        values["source"] = header["source_name"]
    if "src_raj" in header and "src_dej" in header:
        values["coord"] = sigproc.parse_radec(header["src_raj"], header["src_dej"])
        values["sigproc_src_raj"] = float(header["src_raj"])
        values["sigproc_src_dej"] = float(header["src_dej"])
    if "refdm" in header:
        values["dm"] = header["refdm"]
    if "signed" in header:
        values["signed"] = bool(header["signed"])
    if "az_start" in header:
        values["azimuth"] = Angle(header["az_start"] * units.deg)
    if "za_start" in header:
        values["zenith"] = Angle(header["za_start"] * units.deg)
    return values


def _known_from_sigproc(header: dict[str, Any]) -> frozenset[str]:
    """Return the attributes whose Sigproc keys are physically present in the file."""
    known = {
        attr
        for attr, keys in _ATTR_SOURCE_KEYS.items()
        if all(key in header for key in keys)
    }
    known.update(flag for flag in _FRAME_FLAGS if flag in header)
    return frozenset(known)


def _frame_from_sigproc(header: dict[str, Any]) -> str | None:
    """Return the frame recorded by the barycentric/pulsarcentric flags, if any.

    Barycentric wins when both flags are set. Returns None when neither flag is
    present, meaning the frame is unknown.
    """
    if header.get("barycentric"):
        return "barycentric"
    if header.get("pulsarcentric"):
        return "pulsarcentric"
    if any(flag in header for flag in _FRAME_FLAGS):
        return "topocentric"
    return None


def _infer_horizon(values: dict[str, Any], known: frozenset[str]) -> frozenset[str]:
    """Infer azimuth and zenith for attributes the source does not record.

    Inference needs a known telescope, a known source position and ``tstart``. Angles
    already in ``known`` are never replaced.

    Returns
    -------
    frozenset[str]
        The angle attributes that were inferred and stored in ``values``.
    """
    if not {"telescope", "coord"} <= known or "tstart" not in values:
        return frozenset()
    angles = utils.horizon_angles(
        values["telescope"], values["coord"], values["tstart"]
    )
    if angles is None:
        return frozenset()
    inferred = set()
    for attr, angle in zip(("azimuth", "zenith"), angles, strict=True):
        if attr not in known:
            values[attr] = angle
            inferred.add(attr)
    return frozenset(inferred)


def _sigproc_kwargs(
    cls: type[Header],
    header: dict[str, Any],
    *,
    first_write: bool,
) -> dict[str, Any]:
    """Return ``Header`` keyword arguments for a parsed Sigproc-style header.

    Parameters
    ----------
    cls : type[Header]
        Class the arguments are for.
    header : dict
        Parsed Sigproc header.
    first_write : bool
        If True, inferred angles become known and are written to new files. If False
        (a Sigproc file being rewritten) they are only available for use.
    """
    known = _known_from_sigproc(header)
    values = _sigproc_values(header)
    inferred = _infer_horizon(values, known)
    if first_write:
        known |= inferred
    values["known_keys"] = known
    values["frame"] = _frame_from_sigproc(header)
    values["frame_flags"] = frozenset(flag for flag in _FRAME_FLAGS if header.get(flag))
    return {
        key: value for key, value in values.items() if key in attrs.fields_dict(cls)
    }


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
        Reference frame (``"topocentric"``, ``"barycentric"`` or ``"pulsarcentric"``).
        None means the source does not say, by default None. Code that needs a frame
        uses ``effective_frame``, which treats None as topocentric. A None frame is
        never written to a Sigproc file.
    ibeam : int, optional
        Telescope beam number, by default 0.
    nbeams : int, optional
        Number of beams, by default 1.
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
    known_keys : frozenset of str, optional
        Attributes whose value is a fact for this header: read from the source,
        set explicitly, or inferred from complete inputs. Only these are written to
        Sigproc files. By default, attributes that differ from their defaults are
        known.
    sigproc_telescope_id, sigproc_machine_id : int, optional
        Numeric Sigproc IDs as read from a file. They are written back unchanged, even
        when the name is not in the Sigproc table. By default None, so the ID comes
        from the name.
    frame_flags : frozenset of str, optional
        Frame flags (``"barycentric"``, ``"pulsarcentric"``) that are set to 1.
    unknown_keys : tuple of (str, bytes), optional
        Sigproc keys not recognised by this package, kept with their raw value bytes
        so they survive a rewrite, by default empty.
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
    sigproc_telescope_id: int | None = None
    sigproc_machine_id: int | None = None
    sigproc_src_raj: float | None = None
    sigproc_src_dej: float | None = None
    frame: str | None = None
    frame_flags: frozenset[str] = attrs.field(factory=frozenset, converter=frozenset)
    ibeam: int = 0
    nbeams: int = 1
    dm: float = 0
    period: float = 0
    accel: float = 0
    signed: bool = False
    rawdatafile: str = ""
    known_keys: frozenset[str] = attrs.field(
        default=attrs.Factory(_default_known_keys, takes_self=True),
        converter=frozenset,
    )
    unknown_keys: tuple[tuple[str, bytes], ...] = attrs.field(
        factory=tuple,
        converter=tuple,
    )

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

        Returns the ID read from the file if there is one, otherwise the ID for
        ``telescope``, or 0 if the name is not in the Sigproc table.

        Returns
        -------
        int
            Sigproc telescope id.
        """
        if self.sigproc_telescope_id is not None:
            return self.sigproc_telescope_id
        return sigproc.telescope_ids.get(self.telescope, 0)

    @property
    def machine_id(self) -> int:
        """Machine id in Sigproc format.

        Returns the ID read from the file if there is one, otherwise the ID for
        ``backend``, or 0 if the name is not in the Sigproc table.

        Returns
        -------
        int
            Sigproc machine id.
        """
        if self.sigproc_machine_id is not None:
            return self.sigproc_machine_id
        return sigproc.machine_ids.get(self.backend, 0)

    @property
    def effective_frame(self) -> str:
        """Reference frame used for calculations.

        Returns
        -------
        str
            ``frame``, or ``"topocentric"`` when the frame is unknown.
        """
        return self.frame or "topocentric"

    def _tag(self, text: str, attr: str) -> str:
        """Append ``(derived)`` to ``text`` when ``attr`` is not a known fact."""
        return text if attr in self.known_keys else f"{text} (derived)"

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
        :class:`~astropy.time.Time`
            Observation time in Astropy Time format.
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
        new = attrs.asdict(self, recurse=False)
        known = set(self.known_keys or ())
        if update_dict is not None:
            update = {
                key: value
                for key, value in update_dict.items()
                if not (key == "frame" and value is None)
            }
            if "frame" in update and update["frame"] not in _FRAMES:
                msg = f"frame must be one of {_FRAMES}, got {update['frame']!r}."
                raise ValueError(msg)
            if "frame" in update:
                update.setdefault(
                    "frame_flags",
                    frozenset(f for f in (update["frame"],) if f in _FRAME_FLAGS),
                )
            if "telescope" in update:
                update.setdefault("sigproc_telescope_id", None)
            if "backend" in update:
                update.setdefault("sigproc_machine_id", None)
            if "coord" in update:
                update.setdefault("sigproc_src_raj", None)
                update.setdefault("sigproc_src_dej", None)
            if "rawdatafile" in update and not update["rawdatafile"]:
                msg = (
                    "rawdatafile cannot be empty; a zero-length string is not readable."
                )
                raise ValueError(msg)
            new.update(update)
            known.update(key for key in update if key in _ATTR_SOURCE_KEYS)
        new["known_keys"] = frozenset(known)
        new_checked = {
            key: value for key, value in new.items() if key in attrs.fields_dict(Header)
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

        Core keys are always included. Optional keys are included only when they are
        in ``known_keys``, so values the header does not know are not written. Frame
        flags are written from the source flags and from an explicit ``frame``.
        Unrecognised Sigproc keys read from a file are carried in ``unknown_keys``.

        Returns
        -------
        dict
            Sigproc formatted header values.
        """
        if (
            self.sigproc_telescope_id is None
            and self.telescope not in sigproc.telescope_ids
        ):
            logger.warning(
                f"Telescope {self.telescope!r} has no Sigproc id; "
                "writing telescope_id 0."
            )
        if self.sigproc_machine_id is None and self.backend not in sigproc.machine_ids:
            logger.warning(
                f"Backend {self.backend!r} has no Sigproc id; writing machine_id 0."
            )
        sig_header: dict[str, Any] = {
            "data_type": params.data_types.inverse[self.data_type],
            "telescope_id": self.telescope_id,
            "machine_id": self.machine_id,
            "tstart": self.tstart,
            "tsamp": self.tsamp,
            "nbits": self.nbits,
            "nsamples": self.nsamples,
            "fch1": self.fch1,
            "foff": self.foff,
            "nchans": self.nchans,
            "nifs": self.nifs,
            "unknown_keys": self.unknown_keys,
        }
        if "source" in self.known_keys and self.source:
            sig_header["source_name"] = self.source
        if "coord" in self.known_keys:
            if self.sigproc_src_raj is not None and self.sigproc_src_dej is not None:
                sig_header["src_raj"] = self.sigproc_src_raj
                sig_header["src_dej"] = self.sigproc_src_dej
            else:
                sig_header["src_raj"] = float(self.ra.replace(":", ""))
                sig_header["src_dej"] = float(self.dec.replace(":", ""))
        sig_header.update(self._optional_sigproc_values())
        sig_header.update(self._frame_sigproc_flags())
        return sig_header

    def _optional_sigproc_values(self) -> dict[str, Any]:
        """Return the optional keys whose attributes are in ``known_keys``."""
        values: dict[str, Any] = {}
        for attr, key in _WRITTEN_KEYS.items():
            if attr not in self.known_keys:
                continue
            value = getattr(self, attr)
            if isinstance(value, Angle):
                value = value.deg
            if attr == "rawdatafile" and not value:
                continue
            if attr == "signed":
                value = int(value)
                if value:
                    logger.warning("signed=1 is rejected by some readers")
            values[key] = value
        return values

    def _frame_sigproc_flags(self) -> dict[str, int]:
        """Return the frame flags to write.

        A flag is written if the source recorded it, with the value the source gave.
        An explicit barycentric or pulsarcentric frame sets that flag to 1.
        """
        flags = {
            flag: int(flag in self.frame_flags)
            for flag in _FRAME_FLAGS
            if flag in self.known_keys
        }
        if self.frame in _FRAME_FLAGS:
            flags[self.frame] = 1
        return flags

    def to_string(self) -> str:
        """Get a string representation of the header.

        Returns
        -------
        str
            A string representation of the header.
        """
        hdr = []
        temp = "{0:<33}: {1}"
        frame = self.effective_frame
        frame_text = frame if self.frame is not None else f"{frame} (derived)"
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
                temp.format("Data type", f"{self.data_type} ({frame_text})"),
                temp.format("Telescope", self._tag(self.telescope, "telescope")),
                temp.format("Datataking Machine", self._tag(self.backend, "backend")),
                temp.format("Source Name", self._tag(self.source, "source")),
                temp.format("Source RA (J2000)", self._tag(self.ra, "coord")),
                temp.format("Source DEC (J2000)", self._tag(self.dec, "coord")),
                temp.format(
                    "Start AZ (deg)", self._tag(f"{self.azimuth.deg}", "azimuth")
                ),
                temp.format(
                    "Start ZA (deg)", self._tag(f"{self.zenith.deg}", "zenith")
                ),
            ],
        )
        if self.data_type == "filterbank":
            hdr.extend(
                [
                    temp.format("Frequency of channel 1 (MHz)", self.fch1),
                    temp.format("Channel bandwidth      (MHz)", self.foff),
                    temp.format("Number of channels", self.nchans),
                    temp.format(
                        "Number of beams", self._tag(str(self.nbeams), "nbeams")
                    ),
                    temp.format("Beam number", self._tag(str(self.ibeam), "ibeam")),
                ],
            )
        elif self.data_type == "time series":
            hdr.extend(
                [
                    temp.format("Reference DM (pc/cc)", self._tag(str(self.dm), "dm")),
                    temp.format("Reference frequency    (MHz)", self.fch1),
                    temp.format("Number of channels", self.nchans),
                ],
            )
        print_dur, print_unit = utils.duration_string(self.tobs).split()
        signedness = self._tag("SIGNED" if self.signed else "UNSIGNED", "signed")
        hdr.extend(
            [
                temp.format("Signedness of 8-bit numbers", signedness),
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
        inf_dict["barycentric"] = 1 if self.frame == "barycentric" else 0
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
            desc, _, val = line.rpartition("=")
            desc = desc.strip()
            if desc in params.PRESTO_BARYCENTRIC_ALIASES:
                desc = params.PRESTO_BARYCENTRIC_LABEL
            if desc not in params.presto_inf:
                continue
            key, keytype, _keyformat = params.presto_inf[desc]
            header[key] = keytype(val.strip())

        known = {
            attr
            for attr in ("telescope", "backend", "source", "dm", "barycentric")
            if attr in header
        } | {"coord"}
        frame = None
        if "barycentric" in header:
            frame = "barycentric" if header["barycentric"] else "topocentric"
        values: dict[str, Any] = {
            **header,
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
            "frame": frame,
            "frame_flags": frozenset({"barycentric"})
            if header.get("barycentric")
            else frozenset(),
        }
        known_frozen = frozenset(known)
        known_frozen |= _infer_horizon(values, known_frozen)
        values["known_keys"] = known_frozen
        header_check = {
            key: value for key, value in values.items() if key in attrs.fields_dict(cls)
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
        return cls(**_sigproc_kwargs(cls, header, first_write=False), stream_info=sinfo)

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

        values: dict[str, Any] = {
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
            "rawdatafile": Path(filename).name,
        }
        known = {"coord", "telescope", "backend", "source", "rawdatafile"}
        if "IBEAM" in primary_hdr.header:
            values["ibeam"] = primary_hdr.ibeam
            known.add("ibeam")
        if "NBEAM" in primary_hdr.header:
            values["nbeams"] = int(primary_hdr.header["NBEAM"])
            known.add("nbeams")
        known_frozen = frozenset(known)
        known_frozen |= _infer_horizon(values, known_frozen)
        values["known_keys"] = known_frozen
        header_check = {
            key: value for key, value in values.items() if key in attrs.fields_dict(cls)
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
        return cls(**_sigproc_kwargs(cls, header, first_write=True))
