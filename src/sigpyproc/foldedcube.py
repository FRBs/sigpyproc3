"""Classes for handling folded data cubes.

This module contains the Profile and FoldedData classes for handling 1D and 3D
data cubes, respectively.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sigpyproc.core import dispersion
from sigpyproc.core.filters import MatchedFilter
from sigpyproc.header import Header

_FOLD_NPZ_VERSION = 1

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from sigpyproc.core.custom_types import MatchFilterMethods


class Profile:
    """An array class to handle a 1-D pulse profile.

    Parameters
    ----------
    data : :py:obj:`~numpy.typing.ArrayLike`
        1-D pulse profile

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        1-D Pulse profile
    """

    def __init__(self, data: ArrayLike, tsamp: float) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._tsamp = float(tsamp)
        if self._data.ndim != 1 or self._data.size == 0:
            msg = "A profile must be a non-empty 1-D array"
            raise ValueError(msg)
        if not np.isfinite(self._tsamp) or self._tsamp <= 0:
            msg = f"tsamp must be positive and finite, got {tsamp}"
            raise ValueError(msg)

    @property
    def data(self) -> np.ndarray:
        """The pulse profile data (`numpy.ndarray`, read-only)."""
        return self._data

    @property
    def tsamp(self) -> float:
        """The sampling time of the profile (`float`, read-only)."""
        return self._tsamp

    def compute_mf(
        self,
        temp_kind: MatchFilterMethods = "boxcar",
        nbins_max: int = 32,
        spacing_factor: float = 1.5,
    ) -> MatchedFilter:
        return MatchedFilter(
            self.data,
            temp_kind=temp_kind,
            nbins_max=nbins_max,
            spacing_factor=spacing_factor,
        )


class FoldSlice:
    """An array class to handle a folded 2-D data slice.

    Parameters
    ----------
    input_array : :py:obj:`~numpy.typing.ArrayLike`
        2-D array with phase in x axis.

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        2-D array array

    """

    def __init__(self, data: np.ndarray, tsamp: float) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._tsamp = float(tsamp)
        if self._data.ndim != 2 or self._data.shape[1] == 0 or self._data.shape[0] == 0:
            msg = "A fold slice must be a non-empty 2-D array"
            raise ValueError(msg)
        if not np.isfinite(self._tsamp) or self._tsamp <= 0:
            msg = f"tsamp must be positive and finite, got {tsamp}"
            raise ValueError(msg)

    @property
    def data(self) -> np.ndarray:
        """The data slice (`numpy.ndarray`, read-only)."""
        return self._data

    @property
    def tsamp(self) -> float:
        """The sampling time of the slice (`float`, read-only)."""
        return self._tsamp

    @property
    def nbins(self) -> int:
        """Number of bins in the slice (`int`, read-only)."""
        return self.data.shape[1]

    def normalize(self) -> FoldSlice:
        """Normalise the slice by dividing each row by its mean.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldSlice`
            normalised version of slice
        """
        means = np.mean(self.data, axis=1, keepdims=True)
        norm_data = np.divide(
            self.data,
            means,
            out=np.zeros_like(self.data),
            where=means != 0,
        )
        return FoldSlice(norm_data, self.tsamp)

    def get_profile(self) -> Profile:
        """Return the pulse profile from the slice.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.Profile`
            a pulse profile
        """
        return Profile(np.sum(self.data, axis=0), self.tsamp)


class FoldedData:
    """An array class to handle a folded 3-D data cube.

    Parameters
    ----------
    data : :py:obj:`~numpy.ndarray`
        3-D array of folded data
    header : :class:`~sigpyproc.header.Header`
        observational metadata
    period : float
        period that data was folded with
    dm : float
        DM that data was folded with
    accel : float, optional
        accleration that data was folded with, by default 0

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        3-D array of folded data with header metadata

    Notes
    -----
    Input array should have the shape:
    (number of subintegrations, number of subbands, number of profile bins)
    """

    def __init__(
        self,
        data: np.ndarray,
        hdr: Header,
        period: float,
        dm: float,
        accel: float = 0,
    ) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._hdr = hdr
        self._period = period
        self._dm = dm
        self._period0 = period
        self._dm0 = dm
        self._accel = float(accel)
        self._check_input()
        self._tph_shifts = np.zeros(self.nsubints, dtype=np.int64)
        self._fph_shifts = np.zeros(self.nsubbands, dtype=np.int64)

    @property
    def data(self) -> np.ndarray:
        """The folded data cube (`numpy.ndarray`, read-only)."""
        return self._data

    @property
    def header(self) -> Header:
        """The observational metadata (`sigpyproc.header.Header`, read-only)."""
        return self._hdr

    @property
    def period(self) -> float:
        """The folding period (`float`, read-only)."""
        return self._period

    @property
    def dm(self) -> float:
        """The DM (`float`, read-only)."""
        return self._dm

    @property
    def accel(self) -> float:
        """The folding acceleration (`float`, read-only)."""
        return self._accel

    @property
    def nsubints(self) -> int:
        """Number of subintegrations in the data cube(`int`, read-only)."""
        return self.data.shape[0]

    @property
    def nsubbands(self) -> int:
        """Number of subbands in the data cube (`int`, read-only)."""
        return self.data.shape[1]

    @property
    def nbins(self) -> int:
        """Number of bins in the data cube(`int`, read-only)."""
        return self.data.shape[2]

    def get_subint(self, nsubint: int) -> FoldSlice:
        """Get a single subintegration from the data cube.

        Parameters
        ----------
        nsubint : int
            subintegration number (n=0 is first subintegration)

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldSlice`
            a 2-D array containing the subintegration
        """
        return FoldSlice(self.data[nsubint], self._bin_tsamp)

    def get_subband(self, nsubband: int) -> FoldSlice:
        """Get a single subband from the data cube.

        Parameters
        ----------
        nsubband : int
            subband number (n=0 is first subband)

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldSlice`
            a 2-D array containing the subband
        """
        return FoldSlice(self.data[:, nsubband], self._bin_tsamp)

    def get_profile(self) -> Profile:
        """Get the summed pulse profile from the data cube.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.Profile`
            a 1-D array containing the power as a function of phase
        """
        return Profile(np.sum(self.data, axis=(0, 1)), self._bin_tsamp)

    def get_time_phase(self) -> FoldSlice:
        """Return the data cube collapsed in frequency.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldSlice`
            a 2-D array containing the time vs. phase plane
        """
        return FoldSlice(np.sum(self.data, axis=1), self._bin_tsamp)

    def get_freq_phase(self) -> FoldSlice:
        """Return the data cube collapsed in time.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldSlice`
            a 2-D array containing the frequency vs. phase plane
        """
        return FoldSlice(self.data.sum(axis=0), self._bin_tsamp)

    def centre(self) -> FoldedData:
        """Roll the cube so the brightest phase bin sits at the centre.

        The shift is circular. A flat profile is left where it is.
        """
        profile = np.sum(self.data, axis=(0, 1))
        finite = profile[np.isfinite(profile)]
        if finite.size == 0 or np.max(finite) == np.min(finite):
            shift = 0
        else:
            pos = int(np.nanargmax(profile))
            shift = self.nbins // 2 - pos
        centred = FoldedData(
            np.roll(self.data, shift, axis=2),
            self.header,
            self.period,
            self.dm,
            self._accel,
        )
        centred._period0 = self._period0
        centred._dm0 = self._dm0
        centred._fph_shifts = self._fph_shifts.copy()
        centred._tph_shifts = self._tph_shifts.copy()
        return centred

    def to_ascii(self, filename: str) -> str:
        """Write the summed profile as one float per line."""
        np.savetxt(filename, self.get_profile().data)
        return filename

    def to_npz(self, filename: str) -> str:
        """Write a versioned fold archive and return the path actually written.

        NumPy appends ``.npz`` when the name does not already end with it.
        """
        path = Path(filename)
        if not path.name.endswith(".npz"):
            path = path.with_name(path.name + ".npz")
        header = self.header
        np.savez(
            path,
            version=np.int32(_FOLD_NPZ_VERSION),
            data=self.data,
            period=np.float64(self.period),
            dm=np.float64(self.dm),
            accel=np.float64(self._accel),
            period0=np.float64(self._period0),
            dm0=np.float64(self._dm0),
            tph_shifts=self._tph_shifts,
            fph_shifts=self._fph_shifts,
            filename=np.str_(header.filename),
            data_type=np.str_(header.data_type),
            nchans=np.int32(header.nchans),
            foff=np.float64(header.foff),
            fch1=np.float64(header.fch1),
            nbits=np.int32(header.nbits),
            tsamp=np.float64(header.tsamp),
            tstart=np.float64(header.tstart),
            nsamples=np.int64(header.nsamples),
            nifs=np.int32(header.nifs),
        )
        return str(path)

    @classmethod
    def from_npz(cls, filename: str) -> FoldedData:
        """Load a fold archive written by :meth:`to_npz`.

        Parameters
        ----------
        filename : str
            Path of the ``.npz`` archive.

        Returns
        -------
        FoldedData
            Cube with the saved data, header coordinates, and correction basis.

        Raises
        ------
        ValueError
            If the archive version is missing or not supported.
        """
        with np.load(filename, allow_pickle=False) as archive:
            if "version" not in archive.files:
                msg = f"{filename} has no fold archive version"
                raise ValueError(msg)
            version = int(archive["version"])
            if version != _FOLD_NPZ_VERSION:
                msg = f"Unsupported fold archive version {version}"
                raise ValueError(msg)
            required = {
                "data",
                "period",
                "dm",
                "accel",
                "period0",
                "dm0",
                "tph_shifts",
                "fph_shifts",
                "filename",
                "data_type",
                "nchans",
                "foff",
                "fch1",
                "nbits",
                "tsamp",
                "tstart",
                "nsamples",
                "nifs",
            }
            missing = required - set(archive.files)
            if missing:
                msg = f"Fold archive is missing fields: {sorted(missing)}"
                raise ValueError(msg)
            data = np.array(archive["data"], dtype=np.float32, copy=True)
            period = float(archive["period"])
            dm = float(archive["dm"])
            accel = float(archive["accel"])
            period0 = float(archive["period0"])
            dm0 = float(archive["dm0"])
            tph_shifts = np.array(archive["tph_shifts"], copy=True)
            fph_shifts = np.array(archive["fph_shifts"], copy=True)
            header = Header(
                filename=str(archive["filename"]),
                data_type=str(archive["data_type"]),
                nchans=int(archive["nchans"]),
                foff=float(archive["foff"]),
                fch1=float(archive["fch1"]),
                nbits=int(archive["nbits"]),
                tsamp=float(archive["tsamp"]),
                tstart=float(archive["tstart"]),
                nsamples=int(archive["nsamples"]),
                nifs=int(archive["nifs"]),
            )
        cube = cls(data, header, period, dm, accel)
        if not np.isfinite(period0) or period0 <= 0 or not np.isfinite(dm0):
            msg = "Fold archive has an invalid original period or DM"
            raise ValueError(msg)
        for shifts, size in ((tph_shifts, cube.nsubints), (fph_shifts, cube.nsubbands)):
            if shifts.shape != (size,) or not np.issubdtype(
                shifts.dtype, np.signedinteger
            ):
                msg = "Fold archive shift vectors must be signed integers matching axes"
                raise ValueError(msg)
        cube._period0 = period0
        cube._dm0 = dm0
        # Only circular offsets matter, and bounded shifts prevent subtraction
        # overflow in subsequent corrections, including archives written by v1.
        cube._tph_shifts = np.remainder(tph_shifts, cube.nbins).astype(np.int64)
        cube._fph_shifts = np.remainder(fph_shifts, cube.nbins).astype(np.int64)
        return cube

    def replace_nan(self) -> None:
        """Replace NaNs with the median of the finite samples, or 0 if none."""
        finite = self.data[np.isfinite(self.data)]
        fill = float(np.median(finite)) if finite.size else 0.0
        self.data[np.isnan(self.data)] = fill

    def update_dm(self, dm: float) -> None:
        """Install a new DM in the data cube.

        Parameters
        ----------
        dm : float
            The new DM to dedisperse to
        """
        if not np.isfinite(dm):
            msg = f"dm must be finite, got {dm}"
            raise ValueError(msg)
        dmdelays = self._get_dmdelays(dm)
        for isubint in range(self.nsubints):
            for isubband in range(self.nsubbands):
                self.data[isubint][isubband] = np.roll(
                    self.data[isubint][isubband],
                    -dmdelays[isubband],
                    axis=0,
                )
        self._dm = dm

    def update_period(self, period: float) -> None:
        """Install a new folding period in the data cube.

        Parameters
        ----------
        period : float
            The new period to fold with
        """
        if not np.isfinite(period) or period <= 0:
            msg = f"period must be positive and finite, got {period}"
            raise ValueError(msg)
        pdelays = self._get_pdelays(period)
        for isubint in range(self.nsubints):
            for isubband in range(self.nsubbands):
                self.data[isubint][isubband] = np.roll(
                    self.data[isubint][isubband],
                    -pdelays[isubint],
                    axis=0,
                )
        self._period = period

    @property
    def _bin_tsamp(self) -> float:
        """Duration of one phase bin at the current period."""
        return self.period / self.nbins

    def _get_dmdelays(self, newdm: float) -> np.ndarray:
        if newdm == self._dm:
            return np.zeros(self.nsubbands, dtype=np.int32)
        delta_dm = newdm - self._dm0
        chan_width = self.header.foff * self.header.nchans / self.nsubbands
        freqs = (
            np.arange(self.nsubbands, dtype=np.float64) * chan_width + self.header.fch1
        )
        drifts = np.asarray(
            dispersion.compute_dmdelays(
                freqs,
                delta_dm,
                self._period0 / self.nbins,
                self.header.fch1,
                in_samples=True,
            ),
            dtype=np.int64,
        )
        drifts %= self.nbins
        bin_drifts = drifts - self._fph_shifts
        self._fph_shifts = drifts
        return bin_drifts

    def _get_pdelays(self, newperiod: float) -> np.ndarray:
        if newperiod == self._period:
            return np.zeros(self.nsubints, dtype=np.int32)
        dbins = (
            (newperiod / self._period0 - 1)
            * self.header.tobs
            * self.nbins
            / self._period0
        )
        if dbins == 0:
            drifts = np.zeros(self.nsubints, dtype=np.int32)
        else:
            shifts = np.round(
                np.arange(self.nsubints, dtype=np.float64) / (self.nsubints / dbins),
            )
            if not np.isfinite(shifts).all():
                msg = "period correction exceeds the finite phase range"
                raise ValueError(msg)
            drifts = np.remainder(shifts, self.nbins).astype(np.int64)
        bin_drifts = drifts - self._tph_shifts
        self._tph_shifts = drifts
        return bin_drifts

    def _check_input(self) -> None:
        if not isinstance(self.header, Header):
            msg = "Input header is not a Header instance"
            raise TypeError(msg)
        if self.data.ndim != 3 or any(size == 0 for size in self.data.shape):
            msg = "Folded data must be a non-empty 3-D cube"
            raise ValueError(msg)
        if not np.isfinite(self._period) or self._period <= 0:
            msg = f"period must be positive and finite, got {self._period}"
            raise ValueError(msg)
        if not np.isfinite(self._dm):
            msg = f"dm must be finite, got {self._dm}"
            raise ValueError(msg)
        if not np.isfinite(self._accel):
            msg = f"accel must be finite, got {self._accel}"
            raise ValueError(msg)
