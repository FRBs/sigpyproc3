from __future__ import annotations

import numpy as np

from sigpyproc.core import kernels
from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries
from sigpyproc.utils import roll_array


class FilterbankBlock:
    """
    An array class to handle a discrete block of data in time-major order.

    Parameters
    ----------
    data : :py:obj:`~numpy.ndarray`
        2 dimensional array of shape (nchans, nsamples)
    header : :class:`~sigpyproc.header.Header`
        header object containing metadata
    dm : float, optional
        DM of the input_array, by default 0
    """

    def __init__(self, data: np.ndarray, hdr: Header, dm: float = 0) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._hdr = hdr
        self._dm = dm
        self._check_input()

    @property
    def header(self) -> Header:
        """Header object containing metadata."""
        return self._hdr

    @property
    def data(self) -> np.ndarray:
        """Data array."""
        return self._data

    @property
    def dm(self) -> float:
        """DM of the data."""
        return self._dm

    @property
    def nchans(self) -> int:
        """Number of frequency channels."""
        return self.data.shape[0]

    @property
    def nsamples(self) -> int:
        """Number of time samples."""
        return self.data.shape[1]

    def downsample(self, tfactor: int = 1, ffactor: int = 1) -> FilterbankBlock:
        """Downsample data block in frequency and/or time.

        Parameters
        ----------
        tfactor : int, optional
            factor by which to downsample in time, by default 1
        ffactor : int, optional
            factor by which to downsample in frequency, by default 1

        Returns
        -------
        FilterbankBlock
            2 dimensional array of downsampled data

        Raises
        ------
        ValueError
            If number of channels is not divisible by `ffactor`.
        ValueError
            If number of time samples is not divisible by `tfactor`.
        """
        if self.data.shape[0] % ffactor != 0:
            msg = f"Bad frequency factor given: {ffactor}"
            raise ValueError(msg)
        if self.data.shape[1] % tfactor != 0:
            msg = f"Bad time factor given: {tfactor}"
            raise ValueError(msg)
        ar = self.data.transpose().ravel().copy()
        new_ar = kernels.downsample_2d(
            ar,
            tfactor,
            ffactor,
            self.data.shape[0],
            self.data.shape[1],
        )
        new_ar = new_ar.reshape(
            self.data.shape[1] // tfactor,
            self.data.shape[0] // ffactor,
        ).transpose()
        changes = {
            "tsamp": self.header.tsamp * tfactor,
            "foff": self.header.foff * ffactor,
            "nsamples": self.header.nsamples // tfactor,
            "nchans": self.header.nchans // ffactor,
        }
        return FilterbankBlock(new_ar, self.header.new_header(changes))

    def normalise(
        self,
        loc_method: str = "mean",
        *,
        norm_chans: bool = True,
    ) -> FilterbankBlock:
        """Normalise the data block (Subtract mean/median, divide by std).

        Parameters
        ----------
        loc_method : str, optional
            method to estimate location to subtract, by default "mean"
        norm_chans : bool, optional
            if True, normalise each channel, by default True

        Returns
        -------
        FilterbankBlock
            A normalised version of the data block

        Raises
        ------
        ValueError
            if `loc_method` is not "mean" or "median"
        """
        if loc_method not in {"mean", "median"}:
            msg = f"loc_method must be 'mean' or 'median', got {loc_method}"
            raise ValueError(msg)
        np_op = getattr(np, loc_method)
        if norm_chans:
            norm_block = self.data - np_op(self.data, axis=1, keepdims=True)
            data_std = np.std(norm_block, axis=1, keepdims=True)
            norm_block /= np.where(np.isclose(data_std, 0, atol=1e-4), 1, data_std)
        else:
            norm_block = (self.data - np_op(self.data)) / np.std(self.data)
        return FilterbankBlock(norm_block, self.header.new_header())

    def pad_samples(
        self,
        nsamps_final: int,
        offset: int,
        pad_mode: str = "median",
    ) -> FilterbankBlock:
        """Pad the data block with the given mode.

        Parameters
        ----------
        nsamps_final : int
            Number of time samples to pad to
        offset : int
            Number of time samples to pad at the start
        pad_mode : str, optional
            Mode for padding the data, by default "median"

        Returns
        -------
        FilterbankBlock
            Padded data block.

        Raises
        ------
        ValueError
            If the pad_mode is not "mean" or "median".
        """
        if pad_mode not in {"mean", "median"}:
            msg = f"pad_mode must be 'mean' or 'median', got {pad_mode}"
            raise ValueError(msg)
        np_op = getattr(np, pad_mode)
        pad_arr = np_op(self.data, axis=1)
        data_pad = np.ones((self.data.shape[0], nsamps_final), dtype=self.data.dtype)
        data_pad *= pad_arr[:, None]
        data_pad[:, offset : offset + self.data.shape[1]] = self.data
        return FilterbankBlock(
            data_pad,
            self.header.new_header({"nsamples": nsamps_final}),
        )

    def get_tim(self) -> TimeSeries:
        """Sum across all frequencies for each time sample.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            Sum of all channels as timeseries
        """
        ts = self.data.sum(axis=0)
        return TimeSeries(ts, self.header.dedispersed_header(dm=self.dm))

    def get_bandpass(self) -> np.ndarray:
        """Average across each time sample for all frequencies.

        Returns
        -------
        :py:obj:`~numpy.typing.ArrayLike`
            the bandpass of the data
        """
        return self.data.sum(axis=1)

    def dedisperse(
        self,
        dm: float,
        *,
        only_valid_samples: bool = False,
        ref_freq: str = "ch1",
    ) -> FilterbankBlock:
        """Dedisperse the block.

        Parameters
        ----------
        dm : float
            dm to dedisperse to
        only_valid_samples : bool, optional
            return a FilterbankBlock with only time samples that
            contain the full bandwidth, by default False
        ref_freq : str, optional
            reference frequency to dedisperse to, by default "ch1"

        Returns
        -------
        FilterbankBlock
            a dedispersed version of the block

        Raises
        ------
        ValueError
            If there are not enough time samples to dedisperse.

        Notes
        -----
        Frequency dependent delays are applied as rotations to each
        channel in the block with respect to the reference frequency.
        """
        delays = self.header.get_dmdelays(dm, ref_freq=ref_freq)
        if only_valid_samples:
            valid_samps = self.data.shape[1] - delays[-1]
            if valid_samps < 0:
                msg = (
                    f"Insufficient time samples to dedisperse to {dm} (requires at "
                    f"least {delays[-1]} samples, given {self.data.shape[1]})."
                )
                raise ValueError(msg)
            new_ar = np.empty((self.data.shape[0], valid_samps), dtype=self.data.dtype)
            for ichan in range(self.data.shape[0]):
                new_ar[ichan] = self.data[
                    ichan,
                    delays[ichan] : delays[ichan] + valid_samps,
                ]
        else:
            new_ar = np.empty(self.data.shape, dtype=self.data.dtype)
            for ichan in range(self.data.shape[0]):
                new_ar[ichan] = roll_array(self.data[ichan], delays[ichan])
        return FilterbankBlock(
            new_ar,
            self.header.new_header({"nsamples": new_ar.shape[1]}),
            dm,
        )

    def dmt_transform(
        self,
        dm: float,
        dmsteps: int = 512,
        ref_freq: str = "ch1",
    ) -> DMTBlock:
        """Generate a DM-time transform by dedispersing data block at adjacent DMs.

        Parameters
        ----------
        dm : float
            Central DM to dedisperse to
        dmsteps : int, optional
            Number of adjacent DMs to dedisperse to, by default 512
        ref_freq : str, optional
            Reference frequency to dedisperse to, by default "ch1"

        Returns
        -------
        DMTBlock
            2 dimensional array of DM-time transform
        """
        dm_arr = dm + np.linspace(-dm, dm, dmsteps)
        new_ar = np.empty((dmsteps, self.data.shape[1]), dtype=self.data.dtype)
        for idm, dm_val in enumerate(dm_arr):
            new_ar[idm] = self.dedisperse(dm_val, ref_freq=ref_freq).get_tim().data
        return DMTBlock(new_ar, self.header.new_header({"nchans": 1}), dm_arr)

    def to_file(self, filename: str | None = None) -> str:
        """Write the data to file.

        Parameters
        ----------
        filename : str, optional
            name of the output file, by default ``basename_split_start_to_end.fil``

        Returns
        -------
        str
            name of output file
        """
        if filename is None:
            mjd_after = self.header.mjd_after_nsamps(self.data.shape[1])
            filename = (
                f"{self.header.basename}_{self.header.tstart:.12f}_"
                f"to_{mjd_after:.12f}.fil"
            )
        updates = {"nbits": 32}
        out_file = self.header.prep_outfile(filename, updates=updates, nbits=32)
        out_file.cwrite(self.data.transpose().ravel())
        return filename

    def _check_input(self) -> None:
        if not isinstance(self.header, Header):
            msg = "Input header is not a Header instance"
            raise TypeError(msg)
        if self.data.ndim != 2:
            msg = "Input data is not 2 dimensional"
            raise ValueError(msg)
        if self.nsamples != self.header.nsamples:
            msg = (
                f"Input data length ({self.nsamples}) does not match "
                f"header nsamples ({self.header.nsamples})"
            )
            raise ValueError(msg)


class DMTBlock:
    """An array class to handle a DM-time transform block of data in time-major order.

    Parameters
    ----------
    data : :py:obj:`~numpy.ndarray`
        2 dimensional array of shape (ndms, nsamples)
    header : :class:`~sigpyproc.header.Header`
        header object containing metadata
    dms : :py:obj:`~numpy.ndarray`
        array of DM values corresponding to each row of data

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        2 dimensional array of shape (nchans, nsamples) with header metadata
    """

    def __init__(self, data: np.ndarray, hdr: Header, dms: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._hdr = hdr
        self._dms = np.asarray(dms, dtype=np.float32)
        self._check_input()

    @property
    def header(self) -> Header:
        """Header object containing metadata."""
        return self._hdr

    @property
    def data(self) -> np.ndarray:
        """Data array."""
        return self._data

    @property
    def dms(self) -> np.ndarray:
        """DM values corresponding to each row of data."""
        return self._dms

    @property
    def ndms(self) -> int:
        """Number of DMs."""
        return self.data.shape[0]

    @property
    def nsamples(self) -> int:
        """Number of time samples."""
        return self.data.shape[1]

    def _check_input(self) -> None:
        if not isinstance(self.header, Header):
            msg = "Input header is not a Header instance"
            raise TypeError(msg)
        if self.data.ndim != 2:
            msg = "Input data is not 2 dimensional"
            raise ValueError(msg)
        if self.nsamples != self.header.nsamples:
            msg = (
                f"Input data length ({self.nsamples}) does not match "
                f"header nsamples ({self.header.nsamples})"
            )
            raise ValueError(msg)
        if self.ndms != self.dms.size:
            msg = (
                f"Number of DMs ({self.ndms}) does not match number of DM values "
                f"({self.dms.size})"
            )
            raise ValueError(msg)
