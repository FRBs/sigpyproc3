from __future__ import annotations
import numpy as np
from numpy import typing as npt

from sigpyproc.header import Header
from sigpyproc.timeseries import TimeSeries
from sigpyproc.utils import roll_array
from sigpyproc import libcpp  # type: ignore


class FilterbankBlock(np.ndarray):
    """An array class to handle a discrete block of data in time-major order.

    Parameters
    ----------
    input_array : npt.ArrayLike
        2 dimensional array of shape (nchans, nsamples)
    header : Header
        observational metadata
    dm : float, optional
        DM of the input_array, by default 0

    Returns
    -------
    :py:obj:`numpy.ndarray`
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
        :class:`~sigpyproc.Filterbank.FilterbankBlock`
            2 dimensional array of downsampled data

        Raises
        ------
        ValueError
            If number of channels is not divisible by `ffactor`.
        """
        if self.shape[0] % ffactor != 0:
            raise ValueError("Bad frequency factor given")
        newnsamps = self.shape[1] - self.shape[1] % tfactor
        new_ar = np.empty(
            newnsamps * self.shape[0] // ffactor // tfactor,
            dtype="float32",
        )
        ar = self.transpose().ravel().copy()
        libcpp.downsample(ar, new_ar, tfactor, ffactor, self.shape[0], newnsamps)
        new_ar = new_ar.reshape(
            newnsamps // tfactor, self.shape[0] // ffactor
        ).transpose()
        new_header = self.header.new_header(
            {
                "tsamp": self.header.tsamp * tfactor,
                "nchans": self.header.nchans // ffactor,
            }
        )
        return FilterbankBlock(new_ar, new_header)

    def to_file(self, filename: str = None, back_compatible: bool = True) -> str:
        """Write the data to file.

        Parameters
        ----------
        filename : str, optional
            name of the output file, by default ``basename_split_start_to_end.fil``
        back_compatible : bool, optional
            sigproc compatibility flag (legacy code), by default True

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
        new_header = {"nbits": 32}
        out_file = self.header.prep_outfile(
            filename, new_header, nbits=32, back_compatible=back_compatible
        )
        out_file.cwrite(self.transpose().ravel())
        return filename

    def normalise(self) -> FilterbankBlock:
        """Divide each frequency channel by its average.

        Returns
        -------
        :class:`~sigpyproc.Filterbank.FilterbankBlock`
            normalised version of the data
        """
        return self / self.mean(axis=1).reshape(self.shape[0], 1)

    def get_tim(self) -> TimeSeries:
        """Sum across all frequencies for each time sample.

        Returns
        -------
        TimeSeries
            Sum of all channels as timeseries
        """
        ts = self.sum(axis=0)
        return TimeSeries(ts, self.header.new_header())

    def get_bandpass(self) -> npt.ArrayLike:
        """Average across each time sample for all frequencies.

        Returns
        -------
        ArrayLike
            the bandpass of the data
        """
        return self.sum(axis=1)

    def dedisperse(self, dm: float, only_valid_samples: bool = False) -> FilterbankBlock:
        """Dedisperse the block.

        Parameters
        ----------
        dm : float
            dm to dedisperse to
        only_valid_samples : bool, optional
            return a FilterbankBlock with only time samples that
            contain the full bandwidth, by default False

        Returns
        -------
        :class:`~sigpyproc.Filterbank.FilterbankBlock`
            a dedispersed version of the block

        Raises
        ------
        ValueError
            If there are not enough time samples to dedisperse.

        Notes
        -----
        Frequency dependent delays are applied as rotations to each
        channel in the block.
        """
        delays = self.header.get_dmdelays(dm)
        if only_valid_samples:
            if self.shape[1] < delays[-1]:
                raise ValueError(
                    f"Insufficient time samples to dedisperse to {dm} (requires "
                    + f"at least {delays[-1]} samples, given {self.shape[1]})."
                )
            new_ar = FilterbankBlock(
                np.zeros(
                    (self.header.nchans, self.shape[1] - delays[-1]), dtype=self.dtype
                ),
                header=self.header,
            )
            end_samples = delays + new_ar.shape[1]

            slices = [
                np.arange(delay, end_sample)
                for delay, end_sample in zip(delays, end_samples)
            ]
            for idx, time_slice in enumerate(slices):
                new_ar[idx, :] = self[idx, time_slice]
        else:
            new_ar = self.copy()
            for ii in range(self.shape[0]):
                new_ar[ii] = roll_array(self[ii], delays[ii] % self.shape[1], 0)

        new_ar.dm = dm
        return new_ar
