from __future__ import annotations
import pathlib
import numpy as np

from typing import Optional, Tuple, Union
from numpy import typing as npt

from sigpyproc import foldedcube
from sigpyproc import fourierseries
from sigpyproc.header import Header
from sigpyproc import libcpp  # type: ignore


class TimeSeries(np.ndarray):
    """An array class to handle pulsar/FRB data in time series.

    Parameters
    ----------
    input_array : npt.ArrayLike
        1 dimensional array of shape (nsamples)
    header : Header
        observational metadata

    Returns
    -------
    :py:obj:`numpy.ndarray`
        1 dimensional time series with header

    Notes
    -----
    Data is converted to 32 bits regardless of original type.
    """

    def __new__(cls, input_array: npt.ArrayLike, header: Header) -> TimeSeries:
        """Create a new TimeSeries array."""
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, "header", None)

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
        :class:`~sigpyproc.FoldedData.FoldedData`
            data cube containing the folded data

        Raises
        ------
        ValueError
            If ``nbins * nints`` is too large for length of the data.
        """
        if self.size // (nbins * nints) < 10:
            raise ValueError("nbins x nints is too large for length of data")
        fold_ar = np.zeros(nbins * nints, dtype="float64")
        count_ar = np.zeros(nbins * nints, dtype="int32")
        libcpp.fold_tim(
            self,
            fold_ar,
            count_ar,
            self.header.tsamp,
            period,
            accel,
            self.size,
            nbins,
            nints,
        )
        fold_ar /= count_ar
        fold_ar = fold_ar.reshape(nints, 1, nbins)
        return foldedcube.FoldedData(
            fold_ar, self.header.new_header(), period, self.header.refdm, accel
        )

    def rfft(self) -> fourierseries.FourierSeries:
        """Perform 1-D real to complex forward FFT using FFTW3.

        Returns
        -------
        :class:`~sigpyproc.FourierSeries.FourierSeries`
            output of One-Dimensional DFTs of Real Data
        """
        if self.size % 2 == 0:
            fftsize = self.size
        else:
            fftsize = self.size - 1
        fft_ar = lib.rfft(self, fftsize)
        return fourierseries.FourierSeries(fft_ar, self.header.new_header())

    def running_mean(self, window: int = 10001) -> TimeSeries:
        """Filter time series with a running mean.

        Parameters
        ----------
        window : int, optional
            width in bins of running mean filter, by default 10001

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            filtered time series

        Raises
        ------
        ValueError
            If window size < 1

        Notes
        -----
        Window edges will be dealt by reflecting about the edges of the time series.
        For more robust implemetation, use :py:obj:`scipy.ndimage.uniform_filter1d`.
        """
        if window < 1:
            raise ValueError("incorrect window size")
        tim_ar = libcpp.running_mean(self, window, self.size)
        return tim_ar.view(TimeSeries)

    def running_median(self, window: int = 10001) -> TimeSeries:
        """Filter time series with a running median.

        Parameters
        ----------
        window : int, optional
            width in bins of running median filter, by default 10001

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            filtered time series

        Notes
        -----
        Window edges will be dealt with only at the start of the time series.
        """
        tim_ar = libcpp.running_median(self, window, self.size)
        return tim_ar.view(TimeSeries)

    def apply_boxcar(self, width: int) -> TimeSeries:
        """Apply a boxcar filter to the time series.

        Parameters
        ----------
        width : int
            width in bins of filter

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            filtered time series

        Notes
        -----
        Time series returned is of size nsamples-width with width/2
        removed from either end.
        """
        tim_ar = libcpp.run_boxcar(self, width, self.size)
        return tim_ar.view(TimeSeries)

    def downsample(self, factor: int) -> TimeSeries:
        """Downsample the time series.

        Parameters
        ----------
        factor : int
            factor by which time series will be downsampled

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            downsampled time series

        Notes
        -----
        Returned time series is of size nsamples//factor
        """
        if factor == 1:
            return self
        newlen = self.size // factor
        tim_ar = libcpp.downsample_tim(self, factor, newlen)
        return TimeSeries(
            tim_ar, self.header.new_header({"tsamp": self.header.tsamp * factor})
        )

    def pad(self, npad: int) -> TimeSeries:
        """Pad a time series with mean valued data.

        Parameters
        ----------
        npad : int
            number of padding points

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            padded time series
        """
        new_ar = np.hstack((self, self.mean() * np.ones(npad)))
        return TimeSeries(new_ar, self.header.new_header())

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
        TimeSeries
            resampled time series
        """
        if accel > 0:
            new_size = self.size - 1
        else:
            new_size = self.size
        out_ar = np.zeros(new_size, dtype="float32")
        libcpp.resample(self, out_ar, new_size, accel, self.header.tsamp)

        new_header = self.header.new_header({"nsamples": out_ar.size, "accel": accel})
        return TimeSeries(out_ar, new_header)

    def correlate(self, other: Union[TimeSeries, npt.ArrayLike]) -> TimeSeries:
        """Cross correlate with another time series of the same length.

        Parameters
        ----------
        other : Union[TimeSeries, npt.ArrayLike]
            array to correlate with

        Returns
        -------
        TimeSeries
            time series containing the correlation

        Raises
        ------
        IOError
            if input array ``other`` is not array like
        """
        if not isinstance(other, TimeSeries):
            try:
                other = TimeSeries(other, self.header.newHeader())
            except Exception:
                raise IOError("Could not convert input to TimeSeries instance")
        return (self.rfft() * other.rfft()).ifft()

    def to_dat(self, basename: str) -> Tuple[str, str]:
        """Write time series in presto ``.dat`` format.

        Parameters
        ----------
        basename : str
            file basename for output ``.dat`` and ``.inf`` files

        Returns
        -------
        tuple of str
            ``.dat`` file name and ``.inf`` file name

        Notes
        -----
        Method also writes a corresponding .inf file from the header data
        """
        self.header.make_inf(outfile=f"{basename}.inf")
        if self.size % 2 == 0:
            self.tofile(f"{basename}.dat")
        else:
            self[:-1].tofile(f"{basename}.dat")
        return f"{basename}.dat", f"{basename}.inf"

    def to_file(self, filename: Optional[str] = None) -> str:
        """Write time series in sigproc format.

        Parameters
        ----------
        filename : str, optional
            name of file to write to, by default ``basename.tim``

        Returns
        -------
        str
            output file name
        """
        if filename is None:
            filename = f"{self.header.basename}.tim"
        with self.header.prep_outfile(filename, nbits=32) as outfile:
            outfile.cwrite(self)
        return filename

    @classmethod
    def read_dat(cls, datfile: str, inffile: Optional[str] = None) -> TimeSeries:
        """Read a presto format ``.dat`` file.

        Parameters
        ----------
        datfile : str
            the name of the ``.dat`` file to read
        inffile : str, optional
            the name of the corresponding ``.inf`` file, by default None

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
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
            raise IOError("No corresponding .inf file found")
        data = np.fromfile(datfile, dtype=np.float32)
        header = Header.from_inffile(inffile)
        header.filename = datfile
        header.nsamples = data.size
        return cls(data, header)

    @classmethod
    def read_tim(cls, timfile: str) -> TimeSeries:
        """Read a sigproc format ``.tim`` file.

        Parameters
        ----------
        timfile : str
            the name of the ``.tim`` file to read

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            a new TimeSeries object
        """
        header = Header.from_sigproc(timfile)
        data = np.fromfile(timfile, dtype=header.dtype, offset=header.hdrlen)
        data = data.astype(np.float32, copy=False)
        return cls(data, header)
