from __future__ import annotations
import numpy as np
from numpy import typing as npt

from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries
from sigpyproc.core import kernels
from sigpyproc.utils import roll_array


class FilterbankBlock(np.ndarray):
    """An array class to handle a discrete block of data in time-major order.

    Parameters
    ----------
    input_array : :py:obj:`~numpy.typing.ArrayLike`
        2 dimensional array of shape (nchans, nsamples)
    header : :class:`~sigpyproc.header.Header`
        observational metadata
    dm : float, optional
        DM of the input_array, by default 0

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        2 dimensional array of shape (nchans, nsamples) with header metadata

    Notes
    -----
    Data is converted to 32 bits regardless of original type.
    """

    def __new__(
        cls, input_array: npt.ArrayLike, header: Header, dm: float = 0
    ) -> FilterbankBlock:
        """Create a new block array."""
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        obj.dm = dm
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, "header", None)
        self.dm = getattr(obj, "dm", 0)

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
        """
        if self.shape[0] % ffactor != 0:
            raise ValueError("Bad frequency factor given")
        ar = self.transpose().ravel().copy()
        new_ar = kernels.downsample_2d(ar, tfactor, ffactor, self.shape[0], self.shape[1])
        new_ar = new_ar.reshape(
            self.shape[1] // tfactor, self.shape[0] // ffactor
        ).transpose()
        changes = {
            "tsamp": self.header.tsamp * tfactor,
            "foff": self.header.foff * ffactor,
            "nsamples": self.header.nsamples // tfactor,
            "nchans": self.header.nchans // ffactor,
        }
        return FilterbankBlock(new_ar, self.header.new_header(changes))

    def normalise(self, by: str = "mean", chans: bool = True) -> FilterbankBlock:
        """Normalise the data block (Subtract mean/median, divide by std).

        Parameters
        ----------
        by : str, optional
            measurement to subtract from each channel, by default "mean"
        chans : bool, optional
            if True, normalise each channel, by default True

        Returns
        -------
        FilterbankBlock
            A normalised version of the data block
        """
        np_op = getattr(np, by)
        if chans:
            norm_block = self - np_op(self, axis=1)[:, np.newaxis]
            data_std = np.std(norm_block, axis=1)
            norm_block /= np.where(np.isclose(data_std, 0, atol=1e-4), 1, data_std)[
                :, np.newaxis
            ]
        else:
            norm_block = (self - np_op(self)) / np.std(self)
        return FilterbankBlock(norm_block, self.header.new_header())

    def get_tim(self) -> TimeSeries:
        """Sum across all frequencies for each time sample.

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            Sum of all channels as timeseries
        """
        ts = self.sum(axis=0)
        return TimeSeries(ts, self.header.dedispersed_header(dm=self.dm))

    def get_bandpass(self) -> npt.ArrayLike:
        """Average across each time sample for all frequencies.

        Returns
        -------
        :py:obj:`~numpy.typing.ArrayLike`
            the bandpass of the data
        """
        return self.sum(axis=1)

    def dedisperse(
        self, dm: float, only_valid_samples: bool = False, ref_freq: str = "ch1"
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
            valid_samps = self.shape[1] - delays[-1]
            if valid_samps < 0:
                raise ValueError(
                    f"Insufficient time samples to dedisperse to {dm} (requires "
                    + f"at least {delays[-1]} samples, given {self.shape[1]})."
                )
            new_ar = np.empty((self.shape[0], valid_samps), dtype=self.dtype)
            for ichan in range(self.shape[0]):
                new_ar[ichan] = self[ichan, delays[ichan] : delays[ichan] + valid_samps]
        else:
            new_ar = np.empty(self.shape, dtype=self.dtype)
            for ichan in range(self.shape[0]):
                new_ar[ichan] = roll_array(self[ichan], delays[ichan])
        return FilterbankBlock(new_ar, self.header.new_header(), dm=dm)

    def dmt_transform(
        self, dm: float, dmsteps: int = 512, ref_freq: str = "ch1"
    ) -> FilterbankBlock:
        """Generate a DM-time transform of the data block by dedispersing at adjacent DM values.

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
        FilterbankBlock
            2 dimensional array of DM-time transform
        """        
        dm_arr = np.linspace(-dm, dm, dmsteps)
        new_ar = np.empty((dmsteps, self.shape[1]), dtype=self.dtype)
        for idm, dm in enumerate(dm_arr):
            new_ar[idm] = self.dedisperse(dm, ref_freq=ref_freq).get_tim()
        return FilterbankBlock(new_ar, self.header.new_header({"nchans": 1}), dm=dm)

    def to_file(self, filename: str = None) -> str:
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
            mjd_after = self.header.mjd_after_nsamps(self.shape[1])
            filename = (
                f"{self.header.basename}_{self.header.tstart:d}_to_{mjd_after:d}.fil"
            )
        changes = {"nbits": 32}
        out_file = self.header.prep_outfile(filename, changes, nbits=32)
        out_file.cwrite(self.transpose().ravel())
        return filename
