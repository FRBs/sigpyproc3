from __future__ import annotations

import pathlib

import numpy as np
from numpy import typing as npt

try:
    from pyfftw.interfaces import numpy_fft
except ModuleNotFoundError:
    from numpy import fft as numpy_fft

from sigpyproc import foldedcube, fourierseries
from sigpyproc.core import kernels, stats
from sigpyproc.header import Header


class TimeSeries:
    """An array class to handle pulsar/FRB time series data.

    Parameters
    ----------
    data : :py:obj:`~numpy.typing.ArrayLike`
        1-D time series
    header : :class:`~sigpyproc.header.Header`
        header object containing metadata

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        1 dimensional time series array with header metadata
    """

    def __init__(self, data: npt.ArrayLike, hdr: Header) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._hdr = hdr
        self._check_input()

    @property
    def header(self) -> Header:
        """Header object containing metadata."""
        return self._hdr

    @property
    def data(self) -> npt.NDArray[np.float32]:
        """Time series data array."""
        return self._data

    @property
    def nsamples(self) -> int:
        """Number of samples in the time series."""
        return len(self.data)

    def downsample(self, factor: int) -> TimeSeries:
        """Downsample the time series.

        Parameters
        ----------
        factor : int
            factor by which time series will be downsampled

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            downsampled time series

        Raises
        ------
        TypeError
            If factor is not an integer
        ValueError
            If factor is less than or equal to 0

        Notes
        -----
        Returned time series is of size nsamples // factor
        """
        if not isinstance(factor, int):
            msg = "Downsample factor must be an integer"
            raise TypeError(msg)
        if factor <= 0:
            msg = "Downsample factor must be greater than 0"
            raise ValueError(msg)
        if factor == 1:
            return self
        tim_data = kernels.downsample_1d(self.data, factor)
        hdr_changes = {"tsamp": self.header.tsamp * factor, "nsamples": len(tim_data)}
        return TimeSeries(tim_data, self.header.new_header(hdr_changes))

    def pad(self, npad: int, mode: str = "mean", **pad_kwargs: dict) -> TimeSeries:
        """Pad a time series with mean valued data.

        Parameters
        ----------
        npad : int
            number of padding points (bins) to add at the end of the time series
        mode : str, optional
            mode of padding (as used by :py:func:`numpy.pad()`), by default 'mean'
        **pad_kwargs : dict
            Keyword arguments for :py:func:`numpy.pad()`

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            padded time series
        """
        tim_data = np.pad(self.data, (0, npad), mode=mode, **pad_kwargs)  # type: ignore[call-overload]
        hdr_changes = {"nsamples": len(tim_data)}
        return TimeSeries(tim_data, self.header.new_header(hdr_changes))

    def remove_rednoise(
        self,
        filter_func: str = "mean",
        window: float = 0.5,
    ) -> TimeSeries:
        """Remove low-frequency red noise from time series using a moving filter.

        Parameters
        ----------
        filter_func : str, optional
            Moving filter function to use, by default 'mean'
        window : int, optional
            width of moving filter window in seconds, by default 0.5 seconds

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            The de-reddened time series

        Raises
        ------
        ValueError
            If window size < 0
        """
        if window < 0:
            msg = "Window size must be greater than 0"
            raise ValueError(msg)
        window_bins = int(round(window / self.header.tsamp))
        tim_filter = stats.running_filter(
            self.data,
            window_bins,
            filter_func=filter_func,
        )
        tim_deredden = self.data - tim_filter
        return TimeSeries(tim_deredden, self.header)

    def fold(self, period, accel=0, nbins=50, nints=32):
        """Fold time series into discrete phase and subintegration bins.

        Parameters
        ----------
        period : float
            period in seconds to fold with
        accel : float, optional
            The acceleration to fold the time series, by default 0
        nbins : int, optional
            number of phase bins in output, by default 50
        nints : int, optional
            number of subintegrations in output, by default 32

        Returns
        -------
        :class:`~sigpyproc.foldedcube.FoldedData`
            data cube containing the folded data

        Raises
        ------
        ValueError
            If ``nbins * nints`` is too large for length of the data.
        """
        if self.size // (nbins * nints) < 10:
            raise ValueError("nbins x nints is too large for length of data")
        fold_ar = np.zeros(nbins * nints, dtype=np.float32)
        count_ar = np.zeros(nbins * nints, dtype=np.int32)
        kernels.fold(
            self,
            fold_ar,
            count_ar,
            np.array([0], dtype=np.int32),
            0,
            self.header.tsamp,
            period,
            accel,
            self.size,
            self.size,
            1,
            nbins,
            nints,
            1,
            0,
        )
        fold_ar /= count_ar
        fold_ar = fold_ar.reshape(nints, 1, nbins)
        return foldedcube.FoldedData(
            fold_ar,
            self.header.new_header(),
            period,
            self.header.dm,
            accel,
        )

    def rfft(self) -> fourierseries.FourierSeries:
        """Perform 1-D real to complex forward FFT.

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            output of One-Dimensional DFTs of Real Data
        """
        fftsize = self.nsamples - (self.nsamples % 2)
        fft_ar = numpy_fft.rfft(self.data, fftsize)
        return fourierseries.FourierSeries(fft_ar, self.header.new_header())

    def apply_boxcar(self, width: int) -> TimeSeries:
        """Apply a square-normalized boxcar filter to the time series.

        Parameters
        ----------
        width : int
            width of boxcar to apply in bins

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            filtered time series

        Raises
        ------
        ValueError
            If boxcar width < 1

        Notes
        -----
        Time series returned is normalized in units of S/N.
        """
        if width < 1:
            msg = f"invalid boxcar width: {width}"
            raise ValueError(msg)
        mean_ar = stats.running_filter(self.data, width, filter_func="mean")
        mean_ar_norm = mean_ar * np.sqrt(width)
        ref_bin = -width // 2 + 1 if width % 2 else -width // 2
        boxcar_ar = np.roll(mean_ar_norm, ref_bin)
        return TimeSeries(boxcar_ar, self.header.new_header())

    def resample(self, accel: float, jerk: float = 0) -> TimeSeries:
        """Perform time domain resampling to remove acceleration and jerk.

        Parameters
        ----------
        accel : float
            The acceleration to remove from the time series
        jerk : float, optional
            The jerk/jolt to remove from the time series, by default 0

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            resampled time series
        """
        tim_ar = kernels.resample_tim(self, accel, self.header.tsamp)
        hdr_changes = {"nsamples": tim_ar.size, "accel": accel}
        return TimeSeries(tim_ar, self.header.new_header(hdr_changes))

    def correlate(self, other: TimeSeries | npt.ArrayLike) -> TimeSeries:
        """Cross correlate with another time series of the same length.

        Parameters
        ----------
        other : TimeSeries or :py:obj:`~numpy.typing.ArrayLike`
            array to correlate with

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            time series containing the correlation

        Raises
        ------
        IOError
            if input array ``other`` is not array like
        """
        if not isinstance(other, TimeSeries):
            other = TimeSeries(other, self.header.new_header())
        corr_ar = self.rfft().multiply(other.rfft())
        return corr_ar.ifft()

    def to_dat(self, basename: str | None = None) -> str:
        """Write time series in presto ``.dat`` format.

        Parameters
        ----------
        basename : str, optional
            file basename for output ``.dat`` and ``.inf`` files, by default None

        Returns
        -------
        str
            output ``.dat`` file name

        Notes
        -----
        Method also writes a corresponding .inf file from the header data
        """
        if basename is None:
            basename = self.header.basename
        self.header.make_inf(outfile=f"{basename}.inf")
        out_filename = f"{basename}.dat"
        with self.header.prep_outfile(out_filename, nbits=32) as outfile:
            if self.nsamples % 2 == 0:
                outfile.cwrite(self.data)
            else:
                outfile.cwrite(self.data[:-1])
        return out_filename

    def to_tim(self, filename: str | None = None) -> str:
        """Write time series in sigproc format.

        Parameters
        ----------
        filename : str, optional
            name of file to write to, by default ``basename.tim``

        Returns
        -------
        str
            output ``.tim`` file name
        """
        if filename is None:
            filename = f"{self.header.basename}.tim"
        with self.header.prep_outfile(filename, nbits=32) as outfile:
            outfile.cwrite(self.data)
        return filename

    @classmethod
    def from_dat(cls, datfile: str, inffile: str | None = None) -> TimeSeries:
        """Read a presto format ``.dat`` file.

        Parameters
        ----------
        datfile : str
            the name of the ``.dat`` file to read
        inffile : str, optional
            the name of the corresponding ``.inf`` file, by default None

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            a new TimeSeries object

        Raises
        ------
        IOError
            If no ``.inf`` file found in the same directory of ``.dat`` file.

        Notes
        -----
        If inf is None, then the associated .inf file must be in the same directory.
        """
        datpath = pathlib.Path(datfile).resolve()
        if inffile is None:
            inffile = datpath.with_suffix(".inf").as_posix()
        if not pathlib.Path(inffile).is_file():
            msg = "No corresponding .inf file found"
            raise OSError(msg)
        data = np.fromfile(datfile, dtype=np.float32)
        header = Header.from_inffile(inffile)
        header.filename = datfile
        header.nsamples = data.size
        return cls(data, header)

    @classmethod
    def from_tim(cls, timfile: str) -> TimeSeries:
        """Read a sigproc format ``.tim`` file.

        Parameters
        ----------
        timfile : str
            the name of the ``.tim`` file to read

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            a new TimeSeries object
        """
        header = Header.from_sigproc(timfile)
        data = np.fromfile(
            timfile,
            dtype=header.dtype,
            offset=header.stream_info.entries[0].hdrlen,
        )
        return cls(data, header)

    def _check_input(self) -> None:
        if not isinstance(self.header, Header):
            msg = "Input header is not a Header instance"
            raise TypeError(msg)
        if self.data.ndim != 1:
            msg = "Input data is not 1D"
            raise ValueError(msg)
        if len(self.data) != self.header.nsamples:
            msg = (
                f"Input data length ({len(self.data)}) does not match "
                f"header nsamples ({self.header.nsamples})"
            )
            raise ValueError(msg)
        if self.data.size == 0:
            msg = "Input data is empty"
            raise ValueError(msg)
