import os
import numpy as np
from sigpyproc.Utils import File
from sigpyproc import FoldedData
from sigpyproc import FourierSeries
from sigpyproc.Readers import parseInfHeader, parseSigprocHeader

import sigpyproc.libSigPyProc as lib


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

        :param period: period in seconds to fold with
        :type period: float

        :param nbins: number of phase bins in output
        :type nbins: int

        :param nints: number of subintegrations in output
        :type nints: int

        :returns: data cube containing the folded data
        :rtype: :class:`~sigpyproc.FoldedData.FoldedData`
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
        """Perform 1-D real to complex forward FFT.

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
        For more robust implemetation, use :py:function:`scipy.ndimage.uniform_filter1d`.
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

        :param npad: number of padding points
        :type nzeros: int

        :return: padded time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
        """
        new_ar = np.hstack((self, self.mean() * np.ones(npad)))
        return TimeSeries(new_ar, self.header.newHeader())

    def resample(self, accel, jerk=0):
        """Perform time domain resampling to remove acceleration and jerk.

        :param accel: The acceleration to remove from the time series
        :type accel: float

        :param jerk: The jerk/jolt to remove from the time series
        :type jerk: float

        :param period: The mimimum period that the resampling
                       will be sensitive to.
        :type period: float

        :return: resampled time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
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

        :param other: array to correlate with
        :type other: :class:`numpy.ndarray`

        :return: time series containing the correlation
        :rtype: :class:`sigpyproc.TimeSeries.TimeSeries`
        """
        if type(self) != type(other):
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
            if self.size % 2 != 0:
                self[:-1].tofile(datfile)
            else:
                self.tofile(datfile)
        return f"{basename}.dat", f"{basename}.inf"

    def toFile(self, filename):
        """Write time series in sigproc format.

        Parameters
        ----------
        filename : str
            output file name

        Returns
        -------
        str
            [output file name]
        """
        outfile = self.header.prepOutfile(filename)
        self.tofile(outfile)
        return outfile.name

    @classmethod
    def readDat(cls, filename, inf=None):
        """Create a new TimeSeries from a presto format ``.dat`` file.

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
        header = parseInfHeader(inf)
        data = np.fromfile(datfile, dtype=np.float32)
        header["basename"] = basename
        header["inf"]      = inf
        header["filename"] = filename
        header["nsamples"] = data.size
        return cls(data, header)

    @classmethod
    def readTim(cls, filename):
        """Create a new TimeSeries from a sigproc format ``.tim`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.tim`` file to read

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            a new TimeSeries object
        """
        header = parseSigprocHeader(filename)
        nbits  = header["nbits"]
        hdrlen = header["hdrlen"]
        with File(filename, "r", nbits=nbits) as f:
            f.seek(hdrlen)
            data = np.fromfile(f, dtype=header["dtype"]).astype(np.float32, copy=False)
        return cls(data, header)
