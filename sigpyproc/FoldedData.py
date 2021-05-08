from __future__ import annotations
import numpy as np

from numpy import typing as npt

from sigpyproc.profile import PulseProfile
from sigpyproc.Header import Header, DM_CONSTANT_LK
from sigpyproc.Utils import roll_array


class FoldSlice(np.ndarray):
    """An array class to handle a 2-D slice of :class:`~sigpyproc.FoldedData.FoldedData`.

    Parameters
    ----------
    input_array : npt.ArrayLike
        2-D array with phase in x axis.
    header : Header
        observational metadata

    Returns
    -------
    :py:obj:`numpy.ndarray`
        1 dimensional time series with header

    """

    def __new__(cls, input_array: npt.ArrayLike, header: Header) -> FoldSlice:
        """Create a new FoldSlice array."""
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, "header", None)

    def normalize(self) -> FoldSlice:
        """Normalise the slice by dividing each row by its mean.

        Returns
        -------
        :class:`~sigpyproc.FoldedData.FoldSlice`
            normalised version of slice
        """
        return self / self.mean(axis=1).reshape(self.shape[0], 1)

    def get_profile(self) -> PulseProfile:
        """Return the pulse profile from the slice.

        Returns
        -------
        :class:`~sigpyproc.FoldedData.Profile`
            a pulse profile
        """
        return self.sum(axis=0).view(PulseProfile)


class FoldedData(np.ndarray):
    """An array class to handle a data cube produced by any of the folding methods.

    Parameters
    ----------
    input_array : npt.ArrayLike
        3-D array of folded data
    header : Header
        observational metadata
    period : float
        period that data was folded with
    dm : float
        DM that data was folded with
    accel : float, optional
        accleration that data was folded with, by default 0

    Returns
    -------
    :py:obj:`numpy.ndarray`
        3-D array of folded data with header metadata

    Notes
    -----
    Input array should have the shape:
    (number of subintegrations, number of subbands, number of profile bins)
    """

    def __new__(
        cls,
        input_array: npt.ArrayLike,
        header: Header,
        period: float,
        dm: float,
        accel: float = 0,
    ):
        """Construct Folded Data cube."""
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        obj.period = period
        obj.dm = dm
        obj.accel = accel
        obj._set_defaults()
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, "header", None)
        self.period = getattr(obj, "period", None)
        self.dm = getattr(obj, "dm", None)
        self.accel = getattr(obj, "accel", None)

    @property
    def nints(self) -> int:
        """Number of subintegrations in the data cube(`int`, read-only)."""
        return self.shape[0]

    @property
    def nbands(self) -> int:
        """Number of subbands in the data cube (`int`, read-only)."""
        return self.shape[1]

    @property
    def nbins(self) -> int:
        """Number of bins in the data cube(`int`, read-only)."""
        return self.shape[2]

    def get_subint(self, nint: int) -> FoldSlice:
        """Return a single subintegration from the data cube.

        Parameters
        ----------
        nint : int
            subintegration number (n=0 is first subintegration

        Returns
        -------
        :class:`~sigpyproc.FoldedData.FoldSlice`
            a 2-D array containing the subintegration
        """
        return self[nint].view(FoldSlice)

    def get_subband(self, nint: int) -> FoldSlice:
        """Return a single subband from the data cube.

        Parameters
        ----------
        nint : int
            subband number (n=0 is first subband)

        Returns
        -------
        :class:`~sigpyproc.FoldedData.FoldSlice`
            a 2-D array containing the subband
        """
        return self[:, nint].view(FoldSlice)

    def get_profile(self) -> PulseProfile:
        """Return a the data cube summed in time and frequency.

        Returns
        -------
        :class:`~sigpyproc.FoldedData.Profile`
            a 1-D array containing the power as a function of phase (pulse profile)
        """
        return self.sum(axis=0).sum(axis=0).view(PulseProfile)

    def get_time_phase(self) -> FoldSlice:
        """Return the data cube collapsed in frequency.

        Returns
        -------
        :class:`~sigpyproc.FoldedData.FoldSlice`
            a 2-D array containing the time vs. phase plane
        """
        return self.sum(axis=1).view(FoldSlice)

    def get_freq_phase(self) -> FoldSlice:
        """Return the data cube collapsed in time.

        Returns
        -------
        :class:`~sigpyproc.FoldedData.FoldSlice`
            a 2-D array containing the frequency vs. phase plane
        """
        return self.sum(axis=0).view(FoldSlice)

    def centre(self) -> FoldedData:
        """Roll the data cube to center the pulse."""
        prof = self.get_profile()
        on_pulse = prof.on_pulse
        peak = (on_pulse[0] + on_pulse[1]) // 2
        return roll_array(self, (peak - self.nbins // 2), 2).view(FoldedData)

    def update_dm(self, dm: float) -> None:
        """Install a new DM in the data cube.

        Parameters
        ----------
        dm : float
            the new DM to dedisperse to
        """
        dmdelays = self._get_dmdelays(dm)
        for iint in range(self.nints):
            for iband in range(self.nbands):
                self[iint][iband] = roll_array(self[iint][iband], dmdelays[iband], 0)
        self.dm = dm
        self.header.refdm = dm

    def update_period(self, period: float) -> None:
        """Install a new folding period in the data cube.

        Parameters
        ----------
        period : float
            the new period to fold with
        """
        pdelays = self._get_pdelays(period)
        for iint in range(self.nints):
            for iband in range(self.nbands):
                self[iint][iband] = roll_array(self[iint][iband], pdelays[iint], 0)
        self.period = period

    def _set_defaults(self) -> None:
        self._period = self.period
        self._dm = self.dm
        self._accel = self.accel
        self._tph_shifts = np.zeros(self.nints, dtype="int32")
        self._fph_shifts = np.zeros(self.nbands, dtype="int32")

    def _get_dmdelays(self, newdm: float) -> np.ndarray:
        delta_dm = newdm - self._dm
        if delta_dm == 0:
            drifts = -1 * self._fph_shifts
            self._fph_shifts.fill(0)
            return drifts
        chan_width = self.header.foff * self.header.nchans / self.nbands
        freqs = np.arange(self.nbands, dtype="float128") * chan_width + self.header.fch1
        drifts = (
            delta_dm
            * DM_CONSTANT_LK
            * ((freqs ** -2) - (self.header.fch1 ** -2))
            / ((self.period / self.nbins))
        )
        drifts = drifts.round().astype("int32")
        bin_drifts = drifts - self._fph_shifts
        self._fph_shifts = drifts
        return bin_drifts

    def _get_pdelays(self, newperiod: float) -> np.ndarray:
        dbins = (
            (newperiod / self._period - 1) * self.header.tobs * self.nbins / self._period
        )
        if dbins == 0:
            drifts = -1 * self._tph_shifts
            self._tph_shifts.fill(0)
            return drifts
        drifts = np.arange(self.nints, dtype="float32")
        drifts = np.round(drifts / (self.nints / dbins)).astype("int32")
        bin_drifts = drifts - self._tph_shifts
        self._tph_shifts = drifts
        return bin_drifts
