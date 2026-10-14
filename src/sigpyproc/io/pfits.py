"""PSRFITS search-mode header and row reader.

SUBINT rows are addressed from the FITS binary-table byte layout. The table is
never materialised through ``astropy.io.fits`` ``HDU.data``, which would read
every sub-integration, including the DATA cells, on first touch.

One SEARCH-mode file with at least two channels is supported. Fold mode,
single-channel files, multi-file streams, irregular frequency tables, gapped
sub-integrations and writing PSRFITS are not.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Self

import attrs
import numpy as np
from astropy import units
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta

from sigpyproc.io.bits import BitsInfo, unpack
from sigpyproc.kernels.pfits import decode_cube, decode_intensity
from sigpyproc.utils import FrequencyChannels

if TYPE_CHECKING:
    from typing import BinaryIO

logger = logging.getLogger(__name__)

# POL_TYPE values seen in search-mode files. There is no NPOL fallback:
# an unlisted label is an error, because NPOL=4 is both coherence and Stokes.
pol_type_to_state = {
    "XXYY": "PPQQ",
    "LLRR": "PPQQ",
    "AABB": "PPQQ",
    "STOKE": "Stokes",
    "STOKES": "Stokes",
    "IQUV": "Stokes",
    "XXYYCRCI": "Coherence",
    "LLRRCRCI": "Coherence",
    "AABBCRCI": "Coherence",
    "INTEN": "Intensity",
    "AA+BB": "Intensity",
}

# Kept for callers that inspect the historical NPOL guess. Parsing does not use it.
npol_to_state = {1: "Intensity", 2: "PPQQ", 4: "Stokes"}

_STATE_NPOL = {
    "Intensity": frozenset({1}),
    "PPQQ": frozenset({2}),
    "Coherence": frozenset({4}),
    "Stokes": frozenset({1, 4}),
}

_TFORM_RE = re.compile(r"(\d*)([A-Za-z])(.*)")
_TFORM_SIZE = {
    "L": 1,
    "B": 1,
    "A": 1,
    "I": 2,
    "J": 4,
    "K": 8,
    "E": 4,
    "D": 8,
}
_NUMERIC_CODES = frozenset({"B", "I", "J", "K", "E", "D"})
_DTYPE_OF = {"B": "u1", "I": "i2", "J": "i4", "K": "i8", "E": "f4", "D": "f8"}
_FREQ_ATOL_MHZ = 1e-4
_CHAN_BW_ATOL_MHZ = 1e-3
_SCALAR_COLUMNS = frozenset(
    {"INDEXVAL", "TSUBINT", "OFFS_SUB", "AUX_DM", "AUX_RM", "TEL_AZ", "TEL_ZEN"},
)


def polarisation_state(pol_type: str, npol: int) -> str:
    """Return the feed layout for a search-mode ``POL_TYPE``.

    Parameters
    ----------
    pol_type : str
        SUBINT ``POL_TYPE`` card.
    npol : int
        ``NPOL`` of the DATA column.

    Returns
    -------
    str
        ``Intensity``, ``PPQQ``, ``Coherence`` or ``Stokes``.

    Raises
    ------
    ValueError
        If ``pol_type`` is unknown or ``npol`` cannot carry that layout.
    """
    key = pol_type.strip()
    state = pol_type_to_state.get(key)
    if state is None:
        msg = f"Unsupported PSRFITS polarisation POL_TYPE={pol_type!r} with NPOL={npol}"
        raise ValueError(msg)
    allowed = _STATE_NPOL[state]
    if npol not in allowed:
        msg = (
            f"PSRFITS POL_TYPE={pol_type!r} ({state}) is inconsistent with NPOL={npol}"
        )
        raise ValueError(msg)
    return state


def _parse_tform(tform: str) -> tuple[int, str, int]:
    """Return ``(repeat, type_code, element_nbytes)`` for a binary TFORM."""
    match = _TFORM_RE.fullmatch(tform.strip())
    if match is None:
        msg = f"Cannot parse TFORM {tform!r}"
        raise ValueError(msg)
    repeat_text, code, rest = match.groups()
    code = code.upper()
    if code in {"P", "Q"} or rest.upper().startswith(("P", "Q")):
        msg = f"Variable-length TFORM {tform!r} is not supported"
        raise ValueError(msg)
    if code not in _TFORM_SIZE:
        msg = f"Unsupported TFORM {tform!r}"
        raise ValueError(msg)
    repeat = int(repeat_text) if repeat_text else 1
    return repeat, code, _TFORM_SIZE[code]


def _parse_tdim(text: str) -> tuple[int, ...]:
    body = text.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    if not body:
        msg = f"Empty TDIM {text!r}"
        raise ValueError(msg)
    return tuple(int(part) for part in body.split(","))


def _column_spans(header: fits.Header) -> dict[str, tuple[int, int, str]]:
    """Return ``name -> (byte_offset, nbytes, tform)`` and check ``NAXIS1``."""
    spans: dict[str, tuple[int, int, str]] = {}
    offset = 0
    nfields = int(header["TFIELDS"])
    for index in range(1, nfields + 1):
        name = str(header[f"TTYPE{index}"]).strip()
        tform = str(header[f"TFORM{index}"]).strip()
        repeat, _code, elem = _parse_tform(tform)
        nbytes = repeat * elem
        spans[name] = (offset, nbytes, tform)
        offset += nbytes
    naxis1 = int(header["NAXIS1"])
    if offset != naxis1:
        msg = f"SUBINT column widths sum to {offset} bytes, NAXIS1 is {naxis1}"
        raise ValueError(msg)
    return spans


def _view_numeric(buf: np.ndarray, code: str) -> np.ndarray:
    """Interpret a big-endian FITS column buffer as a native 1-d array."""
    dtype = np.dtype(_DTYPE_OF[code]).newbyteorder(">")
    count = buf.size // dtype.itemsize
    view = np.ndarray(count, dtype=dtype, buffer=np.ascontiguousarray(buf))
    return view.astype(dtype.newbyteorder("="), copy=False)


def _assign_axes(
    shape: tuple[int, ...],
    n_time: int,
    n_pol: int,
    n_chan: int,
) -> tuple[int, int, int]:
    """Return axis indices ``(time, pol, chan)`` for a C-order element cube.

    The standard order ``(time, pol, chan)`` wins when several axes share a
    length. Any other shape must match exactly one permutation.
    """
    if len(shape) != 3:
        msg = f"DATA TDIM has {len(shape)} axes {shape}, expected 3"
        raise ValueError(msg)
    wanted = (n_time, n_pol, n_chan)
    if shape == wanted:
        return (0, 1, 2)
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    options = [
        perm
        for perm in perms
        if (shape[perm[0]], shape[perm[1]], shape[perm[2]]) == wanted
    ]
    if len(options) != 1:
        msg = (
            f"DATA shape {shape} does not match "
            f"time={n_time}, pol={n_pol}, chan={n_chan}"
        )
        raise ValueError(msg)
    return options[0]


def arrange_data_bytes(
    raw: np.ndarray,
    tdim: tuple[int, ...],
    *,
    nbits: int,
    nsblk: int,
    npol: int,
    nchan: int,
) -> np.ndarray:
    """Reorder one DATA cell to ``(nsblk, npol, stride)`` ``uint8``.

    ``stride`` is ``nchan`` for 8-bit, ``2 * nchan`` for 16-bit and
    ``nchan * nbits / 8`` for packed samples. ``tdim`` is the Fortran TDIM.
    """
    element_count = int(np.prod(tdim))
    if element_count == 0 or raw.size % element_count != 0:
        msg = f"DATA has {raw.size} bytes and TDIM {tdim}"
        raise ValueError(msg)
    elem_size = raw.size // element_count
    cshape = tuple(reversed(tdim))
    if nbits < 8:
        time_len = nsblk // (8 // nbits)
        expected = 1
    elif nbits == 16:
        time_len = nsblk
        expected = 2
    else:
        time_len = nsblk
        expected = 1
    if elem_size != expected:
        msg = (
            f"{nbits}-bit DATA element size is {elem_size}, expected {expected} byte(s)"
        )
        raise ValueError(msg)
    time_ax, pol_ax, chan_ax = _assign_axes(cshape, time_len, npol, nchan)
    owned = np.ascontiguousarray(raw)
    if elem_size == 1:
        cube = owned.reshape(cshape)
        ordered = np.ascontiguousarray(np.transpose(cube, (time_ax, pol_ax, chan_ax)))
        stride = (nchan * nbits) // 8 if nbits < 8 else nchan
        return ordered.reshape(nsblk, npol, stride)
    dtype = np.dtype(f">i{elem_size}")
    samples = np.ndarray(cshape, dtype=dtype, buffer=owned)
    ordered = np.ascontiguousarray(np.transpose(samples, (time_ax, pol_ax, chan_ax)))
    return np.ascontiguousarray(ordered.view(np.uint8)).reshape(
        nsblk,
        npol,
        nchan * elem_size,
    )


class _SearchTable:
    """One open search-mode PSRFITS file and its SUBINT byte layout."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        header, sub_header, datloc, table_loaded = _read_headers(filename)
        if header.get("FITSTYPE") != "PSRFITS":
            msg = f"File {filename} is not a PSRFITS file."
            raise ValueError(msg)
        mode = header.get("OBS_MODE")
        if mode != "SEARCH":
            msg = (
                f"File {filename} has OBS_MODE={mode!r}. "
                "Only a single SEARCH-mode file is supported."
            )
            raise ValueError(msg)
        if sub_header.get("EXTNAME", "SUBINT") != "SUBINT":
            msg = f"File {filename} has no SUBINT table."
            raise ValueError(msg)
        self.primary_header = header
        self.sub_header = sub_header
        self.datloc = datloc
        self.table_loaded = table_loaded
        self.row_bytes = int(sub_header["NAXIS1"])
        self.nsubint = int(sub_header["NAXIS2"])
        self.spans = _column_spans(sub_header)
        self.npol = int(sub_header["NPOL"])
        self.nchan = int(sub_header["NCHAN"])
        self.nbits = int(sub_header["NBITS"])
        self.nsblk = int(sub_header["NSBLK"])
        self.pol_type = str(sub_header["POL_TYPE"]).strip()
        self.pol_state = polarisation_state(self.pol_type, self.npol)
        self.signint = int(sub_header.get("SIGNINT", 0))
        self.zero_off = float(sub_header.get("ZERO_OFF", 0.0))
        if self.nbits not in {1, 2, 4, 8, 16}:
            msg = f"Unsupported NBITS={self.nbits}"
            raise ValueError(msg)
        if self.nchan < 2:
            msg = (
                f"{filename} has NCHAN={self.nchan}; "
                "search files need at least two channels"
            )
            raise ValueError(msg)
        if min(self.npol, self.nsblk, self.nsubint) < 1:
            msg = (
                f"{filename} has NPOL={self.npol}, NSBLK={self.nsblk}, "
                f"NAXIS2={self.nsubint}; each must be positive"
            )
            raise ValueError(msg)
        if self.nbits < 8 and (self.nchan * self.nbits) % 8 != 0:
            msg = f"NCHAN={self.nchan} is not a whole number of {self.nbits}-bit bytes"
            raise ValueError(msg)
        self._check_nstot()
        self.tdim = self._data_tdim()
        self._fh: BinaryIO | None = Path(filename).open("rb")  # noqa: SIM115
        try:
            self.freqs = self._read_freqs()
            self.scalars = self._read_scalars()
            self._check_subint_spacing()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release the file."""
        handle = self._fh
        if handle is not None:
            handle.close()
            self._fh = None

    def row_slice(self, isub: int, column: str) -> np.ndarray:
        """Return a copy of one column cell as ``uint8``."""
        self._check_row(isub)
        offset, nbytes, _tform = self.spans[column]
        start = self.datloc + isub * self.row_bytes + offset
        handle = self._require_file()
        handle.seek(start)
        buf = handle.read(nbytes)
        if len(buf) != nbytes:
            msg = f"SUBINT row {isub} column {column} extends past the end of the file"
            raise ValueError(msg)
        return np.frombuffer(buf, dtype=np.uint8).copy()

    def column_array(self, isub: int, column: str) -> np.ndarray:
        """Return one numeric column cell in native byte order."""
        _offset, _nbytes, tform = self.spans[column]
        _repeat, code, _elem = _parse_tform(tform)
        if code not in _NUMERIC_CODES:
            msg = f"Column {column} TFORM {tform} is not numeric"
            raise ValueError(msg)
        return _view_numeric(self.row_slice(isub, column), code)

    def data_bytes(self, isub: int) -> np.ndarray:
        """Return DATA for one row as ``(nsblk, npol, stride)`` ``uint8``."""
        raw = self.row_slice(isub, "DATA")
        return arrange_data_bytes(
            raw,
            self.tdim,
            nbits=self.nbits,
            nsblk=self.nsblk,
            npol=self.npol,
            nchan=self.nchan,
        )

    def _require_file(self) -> BinaryIO:
        if self._fh is None:
            msg = f"PSRFITS file {self.filename} is closed"
            raise ValueError(msg)
        return self._fh

    def _check_row(self, isub: int) -> None:
        if isub < 0 or isub >= self.nsubint:
            msg = f"subint {isub} is outside 0..{self.nsubint - 1}"
            raise IndexError(msg)

    def _data_tdim(self) -> tuple[int, ...]:
        data_index = None
        nfields = int(self.sub_header["TFIELDS"])
        for index in range(1, nfields + 1):
            if str(self.sub_header[f"TTYPE{index}"]).strip() == "DATA":
                data_index = index
                break
        if data_index is None:
            msg = "SUBINT table has no DATA column"
            raise ValueError(msg)
        text = self.sub_header.get(f"TDIM{data_index}")
        if text:
            return _parse_tdim(str(text))
        packed_time = self.nsblk * self.nbits // 8 if self.nbits < 8 else self.nsblk
        return (self.nchan, self.npol, packed_time)

    def _read_freqs(self) -> FrequencyChannels:
        first = self._freq_row(0)
        for isub in range(1, self.nsubint):
            other = self._freq_row(isub)
            if not np.allclose(first, other, rtol=0.0, atol=_FREQ_ATOL_MHZ):
                msg = f"DAT_FREQ of subint {isub} differs from subint 0"
                raise ValueError(msg)
        if not np.all(np.isfinite(first)):
            msg = f"DAT_FREQ in {self.filename} is not finite"
            raise ValueError(msg)
        self._require_uniform_freqs(first)
        return FrequencyChannels(first)

    def _require_uniform_freqs(self, freqs: np.ndarray) -> None:
        """Reject an irregular channel grid. ``CHAN_BW`` is only a cross-check."""
        step = np.diff(freqs)
        if np.max(np.abs(step - step[0])) > _FREQ_ATOL_MHZ:
            msg = f"DAT_FREQ in {self.filename} is not uniform to {_FREQ_ATOL_MHZ} MHz"
            raise ValueError(msg)
        if "CHAN_BW" not in self.sub_header:
            return
        chan_bw = float(self.sub_header["CHAN_BW"])
        if abs(chan_bw - float(step[0])) > _CHAN_BW_ATOL_MHZ:
            logger.warning(
                "CHAN_BW=%s MHz disagrees with DAT_FREQ step %s MHz in %s",
                chan_bw,
                float(step[0]),
                self.filename,
            )

    def _check_nstot(self) -> None:
        """Reject ``NSTOT`` that points past the stored rows.

        A shorter ``NSTOT`` is a partial final row and is kept. ``NSUBOFFS``
        is not applied: ``STT_*`` is already the time of the first sample.
        """
        if "NSTOT" not in self.sub_header:
            return
        nstot = int(self.sub_header["NSTOT"])
        capacity = self.nsblk * self.nsubint
        if nstot < 0 or nstot > capacity:
            msg = (
                f"NSTOT={nstot} is outside 0..{capacity} "
                f"(NSBLK*NAXIS2) in {self.filename}"
            )
            raise ValueError(msg)

    def _check_subint_spacing(self) -> None:
        """Reject a gap or overlap between SUBINT rows."""
        if self.nsubint < 2:
            return
        if "OFFS_SUB" not in self.scalars:
            msg = (
                f"{self.filename} has {self.nsubint} sub-integrations and no "
                "OFFS_SUB; split files and row gaps are not supported"
            )
            raise ValueError(msg)
        if "TBIN" not in self.sub_header:
            msg = f"{self.filename} has no TBIN"
            raise ValueError(msg)
        tsamp = float(self.sub_header["TBIN"])
        if not np.isfinite(tsamp) or tsamp <= 0:
            msg = f"TBIN={tsamp} in {self.filename} is not a positive sample interval"
            raise ValueError(msg)
        centres = np.asarray(self.scalars["OFFS_SUB"], dtype=np.float64).reshape(-1)
        if centres.size != self.nsubint or not np.all(np.isfinite(centres)):
            msg = (
                f"OFFS_SUB in {self.filename} does not have "
                f"{self.nsubint} finite centres"
            )
            raise ValueError(msg)
        if "TSUBINT" in self.scalars:
            duration = np.asarray(self.scalars["TSUBINT"], dtype=np.float64).reshape(-1)
        else:
            duration = np.full(self.nsubint, self.nsblk * tsamp)
        if duration.size != self.nsubint or np.any(
            ~np.isfinite(duration) | (duration <= 0)
        ):
            msg = f"TSUBINT in {self.filename} must be positive for every row"
            raise ValueError(msg)
        expected = 0.5 * (duration[:-1] + duration[1:])
        if np.any(np.abs(np.diff(centres) - expected) > 0.5 * tsamp):
            msg = (
                f"{self.filename} sub-integrations are not contiguous. "
                "Split files and row gaps are not supported."
            )
            raise ValueError(msg)

    def _freq_row(self, isub: int) -> np.ndarray:
        values = self.column_array(isub, "DAT_FREQ")
        if values.size < self.nchan:
            msg = f"DAT_FREQ has {values.size} values, NCHAN is {self.nchan}"
            raise ValueError(msg)
        return np.asarray(values[: self.nchan], dtype=np.float64)

    def _read_scalars(self) -> dict[str, np.ndarray]:
        scalars: dict[str, np.ndarray] = {}
        for name, (_offset, _nbytes, tform) in self.spans.items():
            if name not in _SCALAR_COLUMNS:
                continue
            _repeat, code, _elem = _parse_tform(tform)
            if code not in _NUMERIC_CODES:
                continue
            rows = [self.column_array(isub, name) for isub in range(self.nsubint)]
            scalars[name] = np.stack([row.reshape(-1) for row in rows])
        return scalars


def _read_headers(
    filename: str,
) -> tuple[fits.Header, fits.Header, int, bool]:
    with fits.open(filename, mode="readonly", memmap=True) as hdul:
        if "SUBINT" not in hdul:
            msg = f"File {filename} has no SUBINT table."
            raise ValueError(msg)
        sub = hdul["SUBINT"]
        info = sub.fileinfo()
        if info is None or info.get("datLoc") is None:
            msg = f"File {filename} SUBINT extension has no data array"
            raise ValueError(msg)
        return (
            hdul[0].header.copy(),
            sub.header.copy(),
            int(info["datLoc"]),
            bool(sub._data_loaded),  # noqa: SLF001
        )


def _first_scalar(table: _SearchTable, name: str, default: float) -> float:
    values = table.scalars.get(name)
    if values is None or values.size == 0:
        return default
    return float(values.reshape(-1)[0])


@attrs.frozen(auto_attribs=True, kw_only=True)
class Receiver:
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


@attrs.frozen(auto_attribs=True, kw_only=True)
class Backend:
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


class PrimaryHdr:
    """Primary header information.

    Parameters
    ----------
    filename : str
        Filename of the PSRFITS file.

    Attributes
    ----------
    header
    observer
    project_id
    telescope
    location
    receiver
    backend
    ibeam
    obs_mode
    date_obs
    freqs
    chan_dm
    source
    coord
    tstart
    """

    def __init__(self, filename: str) -> None:
        table = _SearchTable(filename)
        try:
            self._bind(table)
        finally:
            table.close()

    @classmethod
    def _from_table(cls, table: _SearchTable) -> Self:
        header = cls.__new__(cls)
        header._bind(table)  # noqa: SLF001
        return header

    def _bind(self, table: _SearchTable) -> None:
        self._header = table.primary_header
        self._freqs = table.freqs

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
            self.header["ANT_X"],
            self.header["ANT_Y"],
            self.header["ANT_Z"],
            unit=units.m,
        )

    @property
    def receiver(self) -> Receiver:
        """:class:`Receiver`: Receiver information."""
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
        """:class:`Backend`: Backend information."""
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
            self.header["DATE-OBS"],
            format="isot",
            scale="utc",
            location=self.location,
        )

    @property
    def freqs(self) -> FrequencyChannels:
        """:class:`~sigpyproc.utils.FrequencyChannels`: Frequency channels.

        The grid is ``DAT_FREQ`` of the first sub-integration. ``OBSFREQ`` and
        ``OBSBW`` are not used: recorded search files often leave ``OBSBW`` at
        zero or set it to a nominal band that is not the channel grid.
        """
        return self._freqs

    @property
    def chan_dm(self) -> float:
        """float: On-line (normally coherent) dedispersion measure."""
        return self.header.get("CHAN_DM", 0)

    @property
    def source(self) -> str:
        """str: Source name or label for an observation."""
        return self.header["SRC_NAME"]

    @property
    def coord(self) -> SkyCoord:
        """astropy.coordinates.SkyCoord: Source coordinates."""
        return SkyCoord(
            self.header["RA"],
            self.header["DEC"],
            unit=(units.hourangle, units.deg),
        )

    @property
    def tstart(self) -> Time:
        """astropy.time.Time: Observation start time.

        ``STT_IMJD``, ``STT_SMJD`` and ``STT_OFFS`` are the time of the first
        sample in this file. ``NSUBOFFS`` is retained on the sub-integration
        header and is not added here.
        """
        return Time(
            self.header["STT_IMJD"],
            format="mjd",
            scale="utc",
            location=self.location,
        ) + TimeDelta(
            float(self.header["STT_SMJD"]),
            float(self.header["STT_OFFS"]),
            format="sec",
        )


class SubintHdr:
    """Sub-integration header information.

    Parameters
    ----------
    filename : str
        Filename of the PSRFITS file.

    Attributes
    ----------
    header
    subint_width
    nsubint
    poln_type
    poln_state
    npol
    tsamp
    nbits
    zero_off
    signint
    subint_offset
    nchans
    chan_bw
    channel_offset
    subint_samples
    nsamples
    subint_shape
    sub_hdr
    tsubint
    offs_sub
    azimuth
    zenith
    freqs
    """

    def __init__(self, filename: str) -> None:
        table = _SearchTable(filename)
        try:
            self._bind(table)
        finally:
            table.close()

    @classmethod
    def _from_table(cls, table: _SearchTable) -> Self:
        header = cls.__new__(cls)
        header._bind(table)  # noqa: SLF001
        return header

    def _bind(self, table: _SearchTable) -> None:
        self._header = table.sub_header
        self._poln_state = table.pol_state
        self._freqs = table.freqs
        self._sub_hdr = {
            name: float(values.reshape(-1)[0]) for name, values in table.scalars.items()
        }

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
        return str(self.header["POL_TYPE"]).strip()

    @property
    def poln_state(self) -> str:
        """str: Polarisation state (e.g., Coherence)."""
        return self._poln_state

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
        """int: Subint offset (``NSUBOFFS``) of a split search-mode file.

        This counts sub-integrations before this file in the original
        observation. The primary ``STT_*`` cards already give the time of the
        first sample stored here, so ``NSUBOFFS`` is not added to ``tstart``.
        """
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
        """float: Time since the observation start at the centre of first subint (s)."""
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
        return self._freqs


def _expand_subbyte(raw: np.ndarray, nbits: int) -> np.ndarray:
    """Expand 1-, 2- and 4-bit samples with the shared big-endian unpacker.

    ``raw`` is ``(nsamp, npol, packed_chan)``. The returned array is
    ``(nsamp, npol, nchan)`` ``uint8``. 8-bit and 16-bit rows are returned
    unchanged. Two-bit levels are applied later; this step only splits bits.
    """
    if nbits >= 8:
        return raw
    nsamp, npol, _packed = raw.shape
    flat = unpack(raw.reshape(-1), nbits, bitorder="big")
    nchan = flat.size // (nsamp * npol)
    return np.ascontiguousarray(flat).reshape(nsamp, npol, nchan)


def _calibration_flags(
    nbits: int,
    *,
    scloffs: bool,
) -> tuple[bool, bool]:
    """Return ``(subtract_zero, apply_scale)`` for the dspsr unpacker rules."""
    if not scloffs:
        return False, False
    # 2-bit levels are already the dspsr lookup. 16-bit is a signed integer
    # and dspsr does not subtract ZERO_OFF from it.
    subtract_zero = nbits not in {2, 16}
    return subtract_zero, True


class PFITSFile:
    """Handle a search-mode PSRFITS file.

    Parameters
    ----------
    filename : str
        Filename of the PSRFITS file.

    Attributes
    ----------
    filename
    pri_hdr
    sub_hdr
    bitsinfo
    """

    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._table = _SearchTable(filename)
        self._primary_hdr = PrimaryHdr._from_table(self._table)  # noqa: SLF001
        self._subint_hdr = SubintHdr._from_table(self._table)  # noqa: SLF001
        self._cache_key: tuple | None = None
        self._cache_val: np.ndarray | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        """Close the file memmap."""
        self._table.close()

    @property
    def filename(self) -> str:
        """Name of file.

        Returns
        -------
        str
            Filename of the PSRFITS file.
        """
        return self._filename

    @property
    def pri_hdr(self) -> PrimaryHdr:
        """Primary header object.

        Returns
        -------
        PrimaryHdr
            Primary header.
        """
        return self._primary_hdr

    @property
    def sub_hdr(self) -> SubintHdr:
        """Sub-integration header object.

        Returns
        -------
        SubintHdr
            Subint header.
        """
        return self._subint_hdr

    @property
    def bitsinfo(self) -> BitsInfo:
        """Bits information object.

        Returns
        -------
        :class:`~sigpyproc.io.bits.BitsInfo`
            Bits information.
        """
        return BitsInfo(self.sub_hdr.nbits)

    @property
    def _table_loaded(self) -> bool:
        """bool: whether opening the file materialised the SUBINT table."""
        return self._table.table_loaded

    def read_subints(
        self,
        startsub: int,
        nsubs: int,
        *,
        poln_select: int = 1,
        scloffs: bool = True,
        weights: bool = True,
    ) -> np.ndarray:
        """Read Stokes I from a range of SUBINT rows.

        Channel order is the ``DAT_FREQ`` order. The array is not reversed
        when the band increases with channel number; ``header.fch1`` and
        ``header.foff`` describe this same order.

        Parameters
        ----------
        startsub : int
            Index of start row (subint) to read from the SUBINT table.
        nsubs : int
            Number of subints to read from the SUBINT table.
        poln_select : int, optional
            Must be 1 (Stokes I). The dspsr ``-d`` codes that request more
            than one product are not part of this reader.
        scloffs : bool, optional
            Apply ``DAT_SCL`` and ``DAT_OFFS``, by default True.
        weights : bool, optional
            Apply ``DAT_WTS``, by default True.

        Returns
        -------
        ndarray
            ``float32`` array of shape ``(nsamps, nchan)``.
        """
        if startsub < 0 or nsubs < 0 or startsub + nsubs > self.sub_hdr.nsubint:
            msg = (
                f"subint range {startsub}:{startsub + nsubs} is outside "
                f"0..{self.sub_hdr.nsubint}"
            )
            raise IndexError(msg)
        rows = [
            self.read_subint_pol(
                isub,
                poln_select=poln_select,
                scloffs=scloffs,
                weights=weights,
            )
            for isub in range(startsub, startsub + nsubs)
        ]
        if not rows:
            return np.empty((0, self.sub_hdr.nchans), dtype=np.float32)
        return np.concatenate(rows)

    def read_subint_pol(
        self,
        isub: int,
        *,
        poln_select: int = 1,
        scloffs: bool = True,
        weights: bool = True,
    ) -> np.ndarray:
        """Read Stokes I from one SUBINT row.

        ``PPQQ`` and ``Coherence`` form ``P0 + P1`` with no extra scale.
        ``Intensity`` and ``Stokes`` (including ``IQUV``) return product 0,
        which is already Stokes I. This matches dspsr ``PScrunch`` for
        detected search data.

        Parameters
        ----------
        isub : int
            Index of row (subint) to read from the SUBINT table.
        poln_select : int, optional
            Must be 1, Stokes I. Other dspsr ``-d`` codes are rejected.
        scloffs : bool, optional
            Apply scales and offsets when unpacking data, by default True.
        weights : bool, optional
            Apply weights when unpacking data, by default True.

        Returns
        -------
        ndarray
            ``float32`` array of shape ``(nsamps, nchan)`` in ``DAT_FREQ`` order.
        """
        if poln_select != 1:
            msg = (
                f"poln_select={poln_select} is not supported; "
                "the search reader returns Stokes I only (poln_select=1)"
            )
            raise ValueError(msg)
        state = self.sub_hdr.poln_state
        if state in {"Coherence", "PPQQ"}:
            pol = -1
        elif state in {"Stokes", "Intensity"}:
            pol = 0
        else:
            msg = (
                f"Unsupported PSRFITS polarisation state {state!r} "
                f"(POL_TYPE={self.sub_hdr.poln_type!r})"
            )
            raise NotImplementedError(msg)
        return self.read_detected(isub, scloffs=scloffs, weights=weights, pol=pol)

    def read_detected(
        self,
        isub: int,
        *,
        scloffs: bool = False,
        weights: bool = False,
        pol: int = -1,
    ) -> np.ndarray:
        """Read one detected product from a SUBINT row.

        Parameters
        ----------
        isub : int
            Index of the sub-integration.
        scloffs : bool, optional
            Apply ``DAT_SCL`` and ``DAT_OFFS``, by default False.
        weights : bool, optional
            Apply ``DAT_WTS``, by default False.
        pol : int, optional
            Polarisation index to copy. ``-1`` sums the first two products.

        Returns
        -------
        ndarray
            ``float32`` array of shape ``(NSBLK, NCHAN)``.
        """
        if pol < -1 or pol >= self.sub_hdr.npol:
            msg = f"polarisation {pol} is outside 0..{self.sub_hdr.npol - 1}"
            raise ValueError(msg)
        if pol < 0 and self.sub_hdr.npol < 2:
            msg = "Summing feeds needs NPOL >= 2"
            raise ValueError(msg)
        key = ("detected", isub, scloffs, weights, pol)
        cached = self._recall(key)
        if cached is not None:
            return cached
        raw, zero_off, do_zero, scales, offsets, do_scale, weight, do_weight = (
            self._decode_inputs(isub, scloffs=scloffs, weights=weights)
        )
        out = np.empty(
            (self.sub_hdr.subint_samples, self.sub_hdr.nchans),
            dtype=np.float32,
        )
        decode_intensity(
            raw,
            self.sub_hdr.nbits,
            self._table.signint != 0,
            zero_off,
            do_zero,
            scales,
            offsets,
            do_scale,
            weight,
            do_weight,
            pol,
            out,
        )
        return self._store(key, out)

    def read_subint(
        self,
        isub: int,
        *,
        scloffs: bool = True,
        weights: bool = True,
    ) -> np.ndarray:
        """Read every polarisation product from one SUBINT row.

        Parameters
        ----------
        isub : int
            Index of row (subint) to read from the SUBINT table.
        scloffs : bool, optional
            Apply scales and offsets when unpacking data, by default True.
        weights : bool, optional
            Apply weights when unpacking data, by default True.

        Returns
        -------
        ndarray
            ``float32`` array of shape ``(NSBLK, NPOL, NCHAN)`` in ``DAT_FREQ``
            order. Two-bit values are the dspsr levels ``{-2, -0.5, 0.5, 2}``.
            ``ZERO_OFF`` is subtracted for 1-, 4- and 8-bit data only when
            ``scloffs`` is set. 16-bit data is signed and is not shifted by
            ``ZERO_OFF``.
        """
        key = ("cube", isub, scloffs, weights)
        cached = self._recall(key)
        if cached is not None:
            return cached
        raw, zero_off, do_zero, scales, offsets, do_scale, weight, do_weight = (
            self._decode_inputs(isub, scloffs=scloffs, weights=weights)
        )
        out = np.empty(self.sub_hdr.subint_shape, dtype=np.float32)
        decode_cube(
            raw,
            self.sub_hdr.nbits,
            self._table.signint != 0,
            zero_off,
            do_zero,
            scales,
            offsets,
            do_scale,
            weight,
            do_weight,
            out,
        )
        return self._store(key, out)

    def read_freqs(self, isub: int) -> FrequencyChannels:
        """Read the channel centre frequencies from one SUBINT row.

        Parameters
        ----------
        isub : int
            Index of row (subint) to read from the SUBINT table.

        Returns
        -------
        :class:`~sigpyproc.utils.FrequencyChannels`
            Centre frequency for each channel in MHz (``NCHAN``).
        """
        freqs = np.asarray(
            self._table.column_array(isub, "DAT_FREQ"),
            dtype=np.float64,
        ).reshape(-1)
        nchan = self.sub_hdr.nchans
        if freqs.size < nchan:
            msg = f"DAT_FREQ has {freqs.size} values, NCHAN is {nchan}"
            raise ValueError(msg)
        return FrequencyChannels(freqs[:nchan])

    def read_weights(self, isub: int) -> np.ndarray:
        """Read channel weight (DAT_WTS) column from the SUBINT table.

        Parameters
        ----------
        isub : int
            Index of row (subint) to read from the SUBINT table.

        Returns
        -------
        ndarray
            Weights for each channel in the range 0-1 (``NCHAN``).
        """
        nchan = self.sub_hdr.nchans
        try:
            weights = self._table.column_array(isub, "DAT_WTS")
        except KeyError as exc:
            msg = "SUBINT has no DAT_WTS column"
            raise ValueError(msg) from exc
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if weights.size < nchan:
            msg = f"DAT_WTS has {weights.size} values, NCHAN is {nchan}"
            raise ValueError(msg)
        return weights[:nchan]

    def read_scales(self, isub: int) -> np.ndarray:
        """Read channel data scale factor (DAT_SCL) column from the SUBINT table.

        Parameters
        ----------
        isub : int
            Index of row (subint) to read from the SUBINT table.

        Returns
        -------
        ndarray
            Data scale factor for each channel, shape ``(NPOL, NCHAN)``,
            channel fastest.
        """
        scales = self._table.column_array(isub, "DAT_SCL")
        count = self.sub_hdr.npol * self.sub_hdr.nchans
        if scales.size < count:
            msg = f"DAT_SCL has {scales.size} values, expected at least {count}"
            raise ValueError(msg)
        return np.asarray(scales[:count], dtype=np.float32).reshape(
            self.sub_hdr.npol,
            self.sub_hdr.nchans,
        )

    def read_offsets(self, isub: int) -> np.ndarray:
        """Read channel data offset (DAT_OFFS) column from the SUBINT table.

        Parameters
        ----------
        isub : int
            Index of row (subint) to read from the SUBINT table.

        Returns
        -------
        ndarray
            Data offset for each channel, shape ``(NPOL, NCHAN)``,
            channel fastest.
        """
        offsets = self._table.column_array(isub, "DAT_OFFS")
        count = self.sub_hdr.npol * self.sub_hdr.nchans
        if offsets.size < count:
            msg = f"DAT_OFFS has {offsets.size} values, expected at least {count}"
            raise ValueError(msg)
        return np.asarray(offsets[:count], dtype=np.float32).reshape(
            self.sub_hdr.npol,
            self.sub_hdr.nchans,
        )

    def _decode_inputs(
        self,
        isub: int,
        *,
        scloffs: bool,
        weights: bool,
    ) -> tuple[
        np.ndarray,
        np.float32,
        bool,
        np.ndarray,
        np.ndarray,
        bool,
        np.ndarray,
        bool,
    ]:
        raw = _expand_subbyte(
            np.ascontiguousarray(self._table.data_bytes(isub)),
            self.sub_hdr.nbits,
        )
        do_zero, do_scale = _calibration_flags(self.sub_hdr.nbits, scloffs=scloffs)
        if do_scale:
            scales = np.ascontiguousarray(self.read_scales(isub), dtype=np.float32)
            offsets = np.ascontiguousarray(self.read_offsets(isub), dtype=np.float32)
        else:
            scales = np.zeros((1, 1), dtype=np.float32)
            offsets = np.zeros((1, 1), dtype=np.float32)
        if weights:
            weight = np.ascontiguousarray(self.read_weights(isub), dtype=np.float32)
        else:
            weight = np.zeros(1, dtype=np.float32)
        return (
            raw,
            np.float32(self._table.zero_off),
            do_zero,
            scales,
            offsets,
            do_scale,
            weight,
            weights,
        )

    def _recall(self, key: tuple) -> np.ndarray | None:
        if self._cache_key == key and self._cache_val is not None:
            return self._cache_val
        return None

    def _store(self, key: tuple, value: np.ndarray) -> np.ndarray:
        self._cache_key = key
        self._cache_val = value
        return value
