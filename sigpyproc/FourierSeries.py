import numpy as np
from sigpyproc.Utils import File
from sigpyproc import TimeSeries
from sigpyproc import FoldedData

import sigpyproc.libSigPyProcSpec as libSpec

class PowerSpectrum(np.ndarray):
    """Class to handle power spectra.

    :param input_array: 1 dimensional array of shape (nsamples)
    :type input_array: :class:`numpy.ndarray`

    :param header: observational metadata
    :type header: :class:`~sigpyproc.Header.Header`
    """

    def __new__(cls, input_array, header):
        obj = input_array.astype("float32").view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self,obj):
        if obj is None: return
        if hasattr(obj,"header"):
            self.header = obj.header
            
    def bin2freq(self, bin_):
        """Return centre frequency of a given bin.

        :param bin_: bin number
        :type bin_: int

        :return: frequency of bin
        :rtype: float
        """
        return (bin_)/(self.header.tobs)
            
    def bin2period(self, bin_):
        """Return centre period of a given bin.

        :param bin_: bin number
        :type bin_: int

        :return: period of bin
        :rtype: float 
        """
        return 1/self.bin2freq(bin_)

    def freq2bin(self, freq):
        """Return nearest bin to a given frequency.

        :param freq: frequency
        :type freq: float

        :return: nearest bin to frequency
        :rtype: float
        """
        return int(round(freq*self.header.tobs))

    def period2bin(self, period):
        """Return nearest bin to a given periodicity.

        :param period: periodicity
        :type period: float

        :return: nearest bin to period
        :rtype: float
        """
        return self.freq2bin(1/period)
        
    def harmonicFold(self, nfolds=1):
        """Perform Lyne-Ashworth harmonic folding of the power spectrum.

        :param nfolds: number of harmonic folds to perform (def=1)
        :type nfolds: int

        :return: A list of folded spectra where the i :sup:`th` element is the spectrum folded i times.
        :rtype: :func:`list` of :class:`~sigpyproc.FourierSeries.PowerSpectrum`
        """

        sum_ar    = self.copy()
        
        nfold1 = 0 #int(self.header.tsamp*2*self.size/maxperiod)
        folds = [] 
        for ii in range(nfolds):
            nharm   = 2**(ii+1)
            nfoldi  = int(max(1, min(nharm*nfold1-nharm//2, self.size)))
            harm_ar = np.array([int(kk*ll/float(nharm)) 
                                for ll in range(nharm) 
                                for kk in range(1, nharm, 2)]).astype("int32")

            facts_ar = np.array([(kk*nfoldi+nharm//2)/nharm 
                                for kk in range(1, nharm, 2)]).astype("int32")

            libSpec.sumHarms(self,
                         sum_ar,
                         harm_ar,
                         facts_ar,
                         nharm,
                         self.size,
                         nfoldi)
            
            new_header = self.header.newHeader({"tsamp":self.header.tsamp*nharm})
            folds.append(PowerSpectrum(sum_ar, new_header))
        return folds
                         

class FourierSeries(np.ndarray):
    """Class to handle output of FFT'd time series.

    :param input_array: 1 dimensional array of shape (nsamples)                                                          
    :type input_array: :class:`numpy.ndarray`
    
    :param header: observational metadata                                                                                              
    :type header: :class:`~sigpyproc.Header.Header`    
    """
    def __new__(cls, input_array, header):
        obj = input_array.astype("float32").view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj,"header"):
            self.header = obj.header

    def __mul__(self, other):
        if type(other) == type(self):
            if other.size != self.size:
                raise Exception("Instances must be the same size")
            else:
                out_ar = np.empty_like(self)
                libSpec.multiply_fs(self,
                                other,
                                out_ar,
                                self.size)
                return FourierSeries(out_ar, self.header.newHeader())
        else:
            return super().__mul__(other)
            
    def __rmul__(self, other):
        self.__mul__(other)
    
    def formSpec(self, interpolated=True):
        """Form power spectrum.
        
        :param interpolated: flag to set nearest bin interpolation (def=True)
        :type interpolated: bool
        
        :return: a power spectrum
        :rtype: :class:`~sigpyproc.FourierSeries.PowerSpectrum`
        """
        spec_ar = np.empty(self.size//2, dtype="float32")
        if interpolated:
            libSpec.formSpecInterpolated(self,
                                     spec_ar,
                                     self.size//2)
        else:
            libSpec.formSpec(self,
                         spec_ar,
                         self.size)
        
        return PowerSpectrum(spec_ar, self.header.newHeader())

    def iFFT(self):
        """Perform 1-D complex to real inverse FFT using FFTW3.

        :return: a time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
        """
        tim_ar = np.empty(self.size-2, dtype="float32")
        libSpec.ifft(self,
                 tim_ar,
                 self.size-2)
        return TimeSeries.TimeSeries(tim_ar, self.header.newHeader())

    def rednoise(self, startwidth=6, endwidth=100, endfreq=1.0):
        """Perform rednoise removal via Presto style method.

        :param startwidth: size of initial array for median calculation
        :type startwidth: int

        :param endwidth: size of largest array for median calculation
        :type endwidth: int

        :param endfreq: remove rednoise up to this frequency
        :type endfreq: float

        :return: whitened fourier series
        :rtype: :class:`~sigpyproc.FourierSeries.FourierSeries`

        """
        out_ar = np.empty_like(self)
        buf_c1 = np.empty(2*endwidth, dtype="float32")
        buf_c2 = np.empty(2*endwidth, dtype="float32")
        buf_f1 = np.empty(endwidth, dtype="float32")
        libSpec.rednoise(self,
                     out_ar,
                     buf_c1,
                     buf_c2,
                     buf_f1,
                     self.size//2,
                     self.header.tsamp,
                     startwidth,
                     endwidth,
                     endfreq)
        return FourierSeries(out_ar, self.header.newHeader())

    def conjugate(self):
        """Conjugate the Fourier series.

        :return: conjugated Fourier series.
        :rtype: :class:`sigpyproc.FourierSeries.FourierSeries`

        .. note::
        
           Function assumes that the Fourier series is the non-conjugated
           product of a real to complex FFT.
        """
        out_ar = np.empty(2*self.size-2, dtype="float32")
        libSpec.conjugate(self,
                      out_ar,
                      self.size)
        return FourierSeries(out_ar, self.header.newHeader())

   
    def reconProf(self, freq, nharms=32):
        """Reconstruct the time domain pulse profile from a signal and its harmonics.

        :param freq: frequency of signal to reconstruct
        :type freq: float

        :param nharms: number of harmonics to use in reconstruction (def=32)
        :type nharms: int
        
        :return: a pulse profile
        :rtype: :class:`sigpyproc.FoldedData.Profile`
        """
        bin_ = freq*self.header.tobs
        real_ids = np.array([int(round(ii*2*bin_)) for ii in range(1, nharms+1)])
        imag_ids = real_ids+1
        harms   = self[real_ids] + 1j*self[imag_ids]
        harm_ar = np.hstack((harms, np.conj(harms[1:][::-1])))
        return FoldedData.Profile(abs(np.fft.ifft(harm_ar)))
        
    def toFile(self, filename=None):
        """Write spectrum to file in sigpyproc format.

        :param filename: name of file to write to (def=``basename.spec``)
        :type filename: str
        
        :return: name of file written to
        :rtype: :func:`str`
        """
        if filename is None:
            filename = f"{self.header.basename}.spec"
        outfile = self.header.prepOutfile(filename, nbits=32)
        self.tofile(outfile)
        return outfile.name
        
    def toFFTFile(self, basename=None):
        """Write spectrum to file in sigpyproc format.

        :param basename: basename of .fft and .inf file to be written
        :type filename: str

        :return: name of files written to
        :rtype: :func:`tuple` of :func:`str`
        """
        if basename is None: basename = self.header.basename
        self.header.makeInf(outfile=f"{basename}.inf")
        fftfile = File(f"{basename}.fft", "w+")
        self.tofile(fftfile)
        return f"{basename}.fft", f"{basename}.inf"
