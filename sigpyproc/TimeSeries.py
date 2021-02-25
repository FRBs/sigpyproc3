import os
import numpy as np

from sigpyproc import FoldedData
from sigpyproc import FourierSeries
from sigpyproc.Header import Header
from sigpyproc import libSigPyProc as lib


class TimeSeries(np.ndarray):
    """Class for handling pulsar/FRB data in time series.

    Parameters
    ----------
    input_array : :py:obj:`numpy.ndarray`
        1 dimensional array of shape (nsamples)
    header : :class:`~sigpyproc.Header.Header`
        observational metadata

    Returns
    -------
    :py:obj:`numpy.ndarray`
        1 dimensional time series with header
    """

    def __new__(cls, input_array, header):
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, 'header', None)

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
        fold_ar  = np.zeros(nbins * nints, dtype="float64")
        count_ar = np.zeros(nbins * nints, dtype="int32")
        lib.foldTim(
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
        fold_ar  = fold_ar.reshape(nints, 1, nbins)
        return FoldedData.FoldedData(
            fold_ar, self.header.newHeader(), period, self.header.refdm, accel
        )

    def rFFT(self):
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
        return FourierSeries.FourierSeries(fft_ar, self.header.newHeader())

    def runningMean(self, window=10001):
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
        RuntimeError
            If window size < 1

        Notes
        -----
        Window edges will be dealt by reflecting about the edges of the time series.
        For more robust implemetation, use :py:obj:`scipy.ndimage.uniform_filter1d`.
        """
        if window < 1:
            raise RuntimeError('incorrect window size')
        tim_ar = lib.runningMean(self, window, self.size)
        return tim_ar.view(TimeSeries)

    def runningMedian(self, window=10001):
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
        tim_ar = lib.runningMedian(self, window, self.size)
        return tim_ar.view(TimeSeries)

    def applyBoxcar(self, width):
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
        tim_ar = lib.runBoxcar(self, width, self.size)
        return tim_ar.view(TimeSeries)

    def downsample(self, factor):
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
        newLen = self.size // factor
        tim_ar = lib.downsampleTim(self, factor, newLen)
        return TimeSeries(
            tim_ar, self.header.newHeader({'tsamp': self.header.tsamp * factor})
        )

    def pad(self, npad):
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
        return TimeSeries(new_ar, self.header.newHeader())

    def resample(self, accel, jerk=0):
        """Perform time domain resampling to remove acceleration and jerk.

        Parameters
        ----------
        accel : float
            The acceleration to remove from the time series
        jerk : float, optional
            The jerk/jolt to remove from the time series, by default 0

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            resampled time series
        """
        if accel > 0:
            new_size = self.size - 1
        else:
            new_size = self.size
        out_ar = np.zeros(new_size, dtype="float32")
        lib.resample(self, out_ar, new_size, accel, self.header.tsamp)

        new_header = self.header.newHeader({"nsamples": out_ar.size, "accel": accel})
        return TimeSeries(out_ar, new_header)

    def correlate(self, other):
        """Cross correlate with another time series of the same length.

        Parameters
        ----------
        other : array to correlate with
            :class:`numpy.ndarray`

        Returns
        -------
        :class:`sigpyproc.TimeSeries.TimeSeries`
            time series containing the correlation
        """
        if not isinstance(other, TimeSeries):
            try:
                other = TimeSeries(other, self.header.newHeader())
            except Exception:
                raise Exception("Could not convert argument to TimeSeries instance")
        return (self.rFFT() * other.rFFT()).iFFT()

    def toDat(self, basename):
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
        self.header.makeInf(outfile=f"{basename}.inf")
        with open(f"{basename}.dat", "w+") as datfile:
            if self.size % 2 == 0:
                self.tofile(datfile)
            else:
                self[:-1].tofile(datfile)
        return f"{basename}.dat", f"{basename}.inf"

    def toFile(self, filename=None):
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
        outfile = self.header.prepOutfile(filename, nbits=32)
        outfile.cwrite(self)
        outfile.close()
        return outfile.name

    @classmethod
    def readDat(cls, filename, inf=None):
        """Read a presto format ``.dat`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.dat`` file to read
        inf : str, optional
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
        If inf=None, then the associated .inf file must be in the same directory.
        """
        datfile = os.path.realpath(filename)
        basename, ext = os.path.splitext(datfile)
        if inf is None:
            inf = f"{basename}.inf"
        if not os.path.isfile(inf):
            raise IOError("No corresponding .inf file found")
        header = Header.parseInfHeader(inf)
        data = np.fromfile(filename, dtype=np.float32)
        header["basename"] = basename
        header["inf"]      = inf
        header["filename"] = filename
        header["nsamples"] = data.size
        return cls(data, header)

    @classmethod
    def readTim(cls, filename):
        """Read a sigproc format ``.tim`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.tim`` file to read

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            a new TimeSeries object
        """
        header = Header.parseSigprocHeader(filename)
        hdrlen = header["hdrlen"]
        data = np.fromfile(filename, dtype=header["dtype"], offset=hdrlen)
        data = data.astype(np.float32, copy=False)
        return cls(data, header)
