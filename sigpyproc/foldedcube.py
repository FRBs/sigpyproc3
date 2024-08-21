from __future__ import annotations

import numpy as np
from numpy import typing as npt

from sigpyproc import params
from sigpyproc.core.filters import MatchedFilter
from sigpyproc.header import Header
from sigpyproc.utils import roll_array


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

    def __init__(self, data: npt.ArrayLike, tsamp: float) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._tsamp = tsamp

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
        temp_kind: str = "boxcar",
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
        self._tsamp = tsamp

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
        norm_data = self.data / np.mean(self.data, axis=1, keepdims=True)
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
        self._accel = accel
        self._check_input()
        self._tph_shifts = np.zeros(self.nsubints, dtype=np.int32)
        self._fph_shifts = np.zeros(self.nsubbands, dtype=np.int32)

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
        return FoldSlice(self.data[nsubint], self.header.tsamp)

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
        return FoldSlice(self.data[:, nsubband], self.header.tsamp)

    def get_profile(self) -> Profile:
        """Get the summed pulse profile from the data cube.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.Profile`
            a 1-D array containing the power as a function of phase
        """
        return Profile(np.sum(self.data, axis=(0, 1)), self.header.tsamp)

    def get_time_phase(self) -> FoldSlice:
        """Return the data cube collapsed in frequency.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldSlice`
            a 2-D array containing the time vs. phase plane
        """
        return FoldSlice(np.sum(self.data, axis=1), self.header.tsamp)

    def get_freq_phase(self) -> FoldSlice:
        """Return the data cube collapsed in time.

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldSlice`
            a 2-D array containing the frequency vs. phase plane
        """
        return FoldSlice(self.data.sum(axis=0), self.header.tsamp)

    def centre(self) -> FoldedData:
        """Roll the data cube to center the pulse."""
        prof = self.get_profile()
        on_pulse_region = prof.compute_mf().on_pulse
        pos = int(np.mean(on_pulse_region))
        new_ar = roll_array(self.data, (pos - self.nbins // 2), 2)
        return FoldedData(new_ar, self.header, self.period, self.dm)

    def replace_nan(self) -> None:
        self.data[np.isnan(self.data)] = np.nanmedian(self.data)

    def dedisperse(self, dm: float) -> None:
        """Rotate the data cube to remove dispersion delay between subbands.

        Parameters
        ----------
        dm : float
            New DM to dedisperse to
        """
        dmdelays = self._get_dmdelays(dm)
        for isubint in range(self.nsubints):
            for isubband in range(self.nsubbands):
                self.data[isubint][isubband] = roll_array(
                    self.data[isubint][isubband],
                    dmdelays[isubband],
                    0,
                )
        self._dm = dm

    def update_period(self, period: float) -> None:
        """Install a new folding period in the data cube.

        Parameters
        ----------
        period : float
            the new period to fold with
        """
        pdelays = self._get_pdelays(period)
        for isubint in range(self.nsubints):
            for isubband in range(self.nsubbands):
                self.data[isubint][isubband] = roll_array(
                    self.data[isubint][isubband],
                    pdelays[isubint],
                    0,
                )
        self._period = period

    def _get_dmdelays(self, newdm: float) -> np.ndarray:
        delta_dm = newdm - self.dm
        if delta_dm == 0:
            drifts = -1 * self._fph_shifts
            self._fph_shifts.fill(0)
            return drifts
        chan_width = self.header.foff * self.header.nchans / self.nsubbands
        freqs = (
            np.arange(self.nsubbands, dtype=np.float64) * chan_width + self.header.fch1
        )
        tsamp = self.period / self.nbins
        drifts = params.compute_dmdelays(
            freqs,
            delta_dm,
            tsamp,
            self.header.fch1,
            in_samples=True,
        )
        bin_drifts = drifts - self._fph_shifts
        self._fph_shifts = drifts
        return bin_drifts

    def _get_pdelays(self, newperiod: float) -> np.ndarray:
        dbins = (
            (newperiod / self._period - 1)
            * self.header.tobs
            * self.nbins
            / self._period
        )
        if dbins == 0:
            drifts = -1 * self._tph_shifts
            self._tph_shifts.fill(0)
            return drifts
        drifts = np.arange(self.nsubints, dtype=np.float32)
        drifts = np.round(drifts / (self.nsubints / dbins)).astype(np.int32)
        bin_drifts = drifts - self._tph_shifts
        self._tph_shifts = drifts
        return bin_drifts

    def _check_input(self) -> None:
        if not isinstance(self.header, Header):
            msg = "Input header is not a Header instance"
            raise TypeError(msg)
        if self.data.ndim != 3:
            msg = "Input data is not 3 dimensional"
            raise ValueError(msg)
