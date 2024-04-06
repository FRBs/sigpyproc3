from __future__ import annotations
import pathlib
import numpy as np

from numpy import typing as npt

try:
    from pyfftw.interfaces import numpy_fft  # noqa: WPS433
except ModuleNotFoundError:
    from numpy import fft as numpy_fft  # noqa: WPS433

from sigpyproc import foldedcube
from sigpyproc import fourierseries
from sigpyproc.header import Header
from sigpyproc.core import stats, kernels


class TimeSeries(np.ndarray):
    """An array class to handle pulsar/FRB data in time series.

    Parameters
    ----------
    input_array : :py:obj:`~numpy.typing.ArrayLike`
        1 dimensional array of shape (nsamples)
    header : :class:`~sigpyproc.header.Header`
        observational metadata

    Returns
    -------
    :py:obj:`~numpy.ndarray`
        1 dimensional time series array with header metadata

    Notes
    -----
    Data is converted to 32-bit floats regardless of original type.
    """

    def __new__(cls, input_array: npt.ArrayLike, header: Header) -> TimeSeries:
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
            fold_ar, self.header.new_header(), period, self.header.dm, accel
        )

    def rfft(self) -> fourierseries.FourierSeries:
        """Perform 1-D real to complex forward FFT.

        Returns
        -------
        :class:`~sigpyproc.fourierseries.FourierSeries`
            output of One-Dimensional DFTs of Real Data
        """
        if self.size % 2 == 0:
            fftsize = self.size
        else:
            fftsize = self.size - 1
        fft_ar = numpy_fft.rfft(self, fftsize)
        return fourierseries.FourierSeries(fft_ar, self.header.new_header())

    def running_mean(self, window: int = 10001) -> TimeSeries:
        """Filter time series with a running mean.

        Parameters
        ----------
        window : int, optional
            width in bins of running mean filter, by default 10001

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            filtered time series

        Raises
        ------
        ValueError
            If window size < 1

        Notes
        -----
        Window edges is dealt by reflecting about the edges of the time series.
        """
        if window < 1:
            raise ValueError("incorrect window size")
        tim_ar = stats.running_mean(self, window)
        return tim_ar.view(TimeSeries)

    def running_median(self, window: int = 10001) -> TimeSeries:
        """Filter time series with a running median.

        Parameters
        ----------
        window : int, optional
            width in bins of running median filter, by default 10001

        Returns
        -------
        :class:`~sigpyproc.timeseries.TimeSeries`
            filtered time series

        Notes
        -----
        Window edges is dealt by reflecting about the edges of the time series.
        """
        tim_ar = stats.running_median(self, window)
        return tim_ar.view(TimeSeries)

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
            raise ValueError("incorrect boxcar window size")
        mean_ar = stats.running_mean(self, width) * np.sqrt(width)
        ref_bin = -width // 2 + 1 if width % 2 else -width // 2
        boxcar_ar = np.roll(mean_ar, ref_bin)
        return boxcar_ar.view(TimeSeries)

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

        Notes
        -----
        Returned time series is of size nsamples // factor
        """
        if factor == 1:
            return self
        tim_ar = kernels.downsample_1d(self, factor)
        changes = {"tsamp": self.header.tsamp * factor, "nsamples": tim_ar.size}
        return TimeSeries(tim_ar, self.header.new_header(changes))

    def pad(self, npad: int, mode: str = "mean", **pad_kwargs) -> TimeSeries:
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
        tim_ar = np.pad(self, (0, npad), mode=mode, **pad_kwargs)
        return tim_ar.view(TimeSeries)

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
        changes = {"nsamples": tim_ar.size, "accel": accel}
        return TimeSeries(tim_ar, self.header.new_header(changes))

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
            try:
                other = TimeSeries(other, self.header.new_header())
            except Exception:
                raise IOError("Could not convert input to TimeSeries instance")
        corr_ar = self.rfft() * other.rfft()
        return corr_ar.ifft()  # type: ignore

    def to_dat(self, basename: str) -> tuple[str, str]:
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

    def to_file(self, filename: str | None = None) -> str:
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
    def read_dat(cls, datfile: str, inffile: str | None = None) -> TimeSeries:
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
        :class:`~sigpyproc.timeseries.TimeSeries`
            a new TimeSeries object
        """
        header = Header.from_sigproc(timfile)
        data = np.fromfile(
            timfile, dtype=header.dtype, offset=header.stream_info.entries[0].hdrlen
        )
        return cls(data, header)
