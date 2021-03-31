import os
import numpy as np

from sigpyproc import FoldedData
from sigpyproc import TimeSeries
from sigpyproc.Header import Header
from sigpyproc import libSigPyProc as lib


class PowerSpectrum(np.ndarray):
    """Class to handle power spectra.

    Parameters
    ----------
    input_array : :py:obj:`numpy.ndarray`
        1 dimensional array of shape (nsamples)
    header : :class:`~sigpyproc.Header.Header`
        observational metadata

    Returns
    -------
    :py:obj:`numpy.ndarray`
        1 dimensional power spectrum with header
    """

    def __new__(cls, input_array, header):
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, 'header', None)

    def bin2freq(self, bin_):
        """Return centre frequency of a given bin.

        Parameters
        ----------
        bin_ : int
            bin number

        Returns
        -------
        float
            frequency of bin
        """
        return (bin_) / (self.header.tobs)

    def bin2period(self, bin_):
        """Return centre period of a given bin.

        Parameters
        ----------
        bin_ : int
            bin number

        Returns
        -------
        float
            period of bin
        """
        return 1 / self.bin2freq(bin_)

    def freq2bin(self, freq):
        """Return nearest bin to a given frequency.

        Parameters
        ----------
        freq : float
            frequency

        Returns
        -------
        float
            nearest bin to frequency
        """
        return int(round(freq * self.header.tobs))

    def period2bin(self, period):
        """Return nearest bin to a given periodicity.

        Parameters
        ----------
        period : float
            periodicity

        Returns
        -------
        float
            nearest bin to period
        """
        return self.freq2bin(1 / period)

    def harmonicFold(self, nfolds=1):
        """Perform Lyne-Ashworth harmonic folding of the power spectrum.

        Parameters
        ----------
        nfolds : int, optional
            number of harmonic folds to perform, by default 1

        Returns
        -------
        list of :class:`~sigpyproc.FourierSeries.PowerSpectrum`
            A list of folded spectra where the i :sup:`th` element
            is the spectrum folded i times.
        """
        sum_ar = self.copy()

        nfold1 = 0  # int(self.header.tsamp*2*self.size/maxperiod)
        folds = []
        for ii in range(nfolds):
            nharm = 2**(ii + 1)
            nfoldi = int(max(1, min(nharm * nfold1 - nharm // 2, self.size)))
            harm_ar = np.array(
                [
                    int(kk * ll / float(nharm))
                    for ll in range(nharm)
                    for kk in range(1, nharm, 2)
                ]
            ).astype("int32")

            facts_ar = np.array(
                [(kk * nfoldi + nharm // 2) / nharm for kk in range(1, nharm, 2)]
            ).astype("int32")

            lib.sumHarms(self, sum_ar, harm_ar, facts_ar, nharm, self.size, nfoldi)

            new_header = self.header.newHeader({"tsamp": self.header.tsamp * nharm})
            folds.append(PowerSpectrum(sum_ar, new_header))
        return folds


class FourierSeries(np.ndarray):
    """Class to handle output of FFT'd time series.

    Parameters
    ----------
    input_array : :py:obj:`numpy.ndarray`
        1 dimensional array of shape (nsamples)
    header : :class:`~sigpyproc.Header.Header`
        observational metadata

    Returns
    -------
    :py:obj:`numpy.ndarray`
        1 dimensional fourier series with header
    """

    def __new__(cls, input_array, header):
        obj = np.asarray(input_array).astype(np.float32, copy=False).view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.header = getattr(obj, 'header', None)

    def __mul__(self, other):
        if isinstance(other, FourierSeries):
            if other.size != self.size:
                raise Exception("Instances must be the same size")
            out_ar = lib.multiply_fs(self, other, self.size)
            return FourierSeries(out_ar, self.header.newHeader())
        return super().__mul__(other)

    def __rmul__(self, other):
        self.__mul__(other)

    def formSpec(self, interpolated=True):
        """Form power spectrum.

        Parameters
        ----------
        interpolated : bool, optional
            flag to set nearest bin interpolation, by default True

        Returns
        -------
        :class:`~sigpyproc.FourierSeries.PowerSpectrum`
            a power spectrum
        """
        specsize = self.size // 2
        spec_ar = lib.formSpec(self, specsize, interpolated=interpolated)
        return PowerSpectrum(spec_ar, self.header.newHeader())

    def iFFT(self):
        """Perform 1-D complex to real inverse FFT using FFTW3.

        Returns
        -------
        :class:`~sigpyproc.TimeSeries.TimeSeries`
            a time series
        """
        fftsize = self.size - 2
        tim_ar = lib.irfft(self, fftsize)
        tim_ar *= 1.0 / fftsize
        return TimeSeries.TimeSeries(tim_ar, self.header.newHeader())

    def rednoise(self, startwidth=6, endwidth=100, endfreq=1.0):
        """Perform rednoise removal via Presto style method.

        Parameters
        ----------
        startwidth : int, optional
            size of initial array for median calculation, by default 6
        endwidth : int, optional
            size of largest array for median calculation, by default 100
        endfreq : float, optional
            remove rednoise up to this frequency, by default 1.0

        Returns
        -------
        :class:`~sigpyproc.FourierSeries.FourierSeries`
            whitened fourier series
        """
        buf_c1 = np.empty(2 * endwidth, dtype="float32")
        buf_c2 = np.empty(2 * endwidth, dtype="float32")
        buf_f1 = np.empty(endwidth, dtype="float32")
        out_ar = lib.rednoise(self,
                              buf_c1,
                              buf_c2,
                              buf_f1,
                              self.size // 2,
                              self.header.tsamp,
                              startwidth,
                              endwidth,
                              endfreq,
                              )
        return FourierSeries(out_ar, self.header.newHeader())

    def conjugate(self):
        """Conjugate the Fourier series.

        Returns
        -------
        :class:`sigpyproc.FourierSeries.FourierSeries`
            conjugated Fourier series.

        Notes
        -----
        Function assumes that the Fourier series is the non-conjugated
        product of a real to complex FFT.
        """
        out_ar = lib.conjugate(self, self.size)
        return FourierSeries(out_ar, self.header.newHeader())

    def reconProf(self, freq, nharms=32):
        """Reconstruct the time domain pulse profile from a signal and its harmonics.

        Parameters
        ----------
        freq : float
            frequency of signal to reconstruct
        nharms : int, optional
            number of harmonics to use in reconstruction, by default 32

        Returns
        -------
        :class:`sigpyproc.FoldedData.Profile`
            a pulse profile
        """
        bin_ = freq * self.header.tobs
        real_ids = np.array([int(round(ii * 2 * bin_))
                            for ii in range(1, nharms + 1)])
        imag_ids = real_ids + 1
        harms = self[real_ids] + 1j * self[imag_ids]
        harm_ar = np.hstack((harms, np.conj(harms[1:][::-1])))
        return FoldedData.Profile(abs(np.fft.ifft(harm_ar)))

    def toFile(self, filename=None):
        """Write spectrum to file in sigpyproc format.

        Parameters
        ----------
        filename : str, optional
            name of file to write to, by default ``basename.spec``

        Returns
        -------
        str
            output file name
        """
        if filename is None:
            filename = f"{self.header.basename}.spec"
        with self.header.prepOutfile(filename, nbits=32) as outfile:
            outfile.cwrite(self)
        return filename

    def toFFTFile(self, basename=None):
        """Write spectrum to file in presto ``.fft`` format.

        Parameters
        ----------
        basename : str, optional
            basename of ``.fft`` and ``.inf`` file to be written, by default None

        Returns
        -------
        tuple of str
            name of files written to
        """
        if basename is None:
            basename = self.header.basename
        self.header.makeInf(outfile=f"{basename}.inf")
        with open(f"{basename}.fft", "w+") as fftfile:
            self.tofile(fftfile)
        return f"{basename}.fft", f"{basename}.inf"

    @classmethod
    def readFFT(cls, filename, inf=None):
        """Read a presto format ``.fft`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.fft`` file to read
        inf : str, optional
            the name of the corresponding ``.inf`` file, by default None

        Returns
        -------
        :class:`~sigpyproc.FourierSeries.FourierSeries`
            an array containing the whole file contents

        Raises
        ------
        IOError
            If no ``.inf`` file found in the same directory of ``.fft`` file.

        Notes
        -----
        If inf=None, then the associated .inf file must be in the same directory.
        """
        fftfile = os.path.realpath(filename)
        basename, ext = os.path.splitext(fftfile)
        if inf is None:
            inf = f"{basename}.inf"
        if not os.path.isfile(inf):
            raise IOError("No corresponding inf file found")
        header = Header.parseInfHeader(inf)
        header["basename"] = basename
        header["inf"]      = inf
        header["filename"] = filename
        data = np.fromfile(filename, dtype="float32")
        return cls(data, header)

    @classmethod
    def readSpec(cls, filename):
        """Read a sigpyproc format ``.spec`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.spec`` file to read

        Returns
        -------
        :class:`~sigpyproc.FourierSeries.FourierSeries`
            an array containing the whole file contents

        Notes
        -----
        This is not setup to handle ``.spec`` files such as are
        created by Sigprocs seek module. To do this would require
        a new header parser for that file format.
        """
        header = Header.parseSigprocHeader(filename)
        hdrlen = header["hdrlen"]
        data = np.fromfile(filename, dtype="complex32", offset=hdrlen)
        return cls(data, header)
