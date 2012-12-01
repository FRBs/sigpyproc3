from numpy.ctypeslib import as_ctypes as as_c
import numpy as np
import ctypes as C
lib = C.CDLL("libSigPyProcTim.so")

class TimeSeries(np.ndarray):
    """Class for handling pulsar data in time series.

    :param input_array: 1 dimensional array of shape (nsamples)
    :type input_array: :class:`numpy.ndarray`

    :param header: observational metadata
    :type header: :class:`~sigpyproc.Header.Header`
    """
    def __new__(cls,input_array,header):
        if getattr(input_array,"dtype",False) == np.dtype("float32"):
            obj = input_array.view(cls)
        else:
            obj = input_array.astype("float32").view(cls)
        obj.header = header
        return obj

    def __array_finalize__(self,obj):
        if obj is None: return
        if hasattr(obj,"header"):
            self.header = obj.header
            
    def fold(self,period,accel=0,nbins=50,nints=32):
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
        if self.size/(nbins*nints) < 10: 
            raise ValueError,"nbins x nints is too large for length of data"
        fold_ar  = np.zeros(nbins*nints,dtype="float64")
        count_ar = np.zeros(nbins*nints,dtype="int32")
        lib.foldTim(as_c(self),
                    as_c(fold_ar),
                    as_c(count_ar),
                    C.c_double(self.header.tsamp),
                    C.c_double(period),
                    C.c_double(accel),
                    C.c_int(self.size),
                    C.c_int(nbins),
                    C.c_int(nints))
        fold_ar/=count_ar
        fold_ar = fold_ar.reshape(nints,1,nbins)
        return FoldedData(fold_ar,
                          self.header.newHeader(),
                          period,
                          self.header.refdm,
                          accel)

    def rFFT(self):
        """Perform 1-D real to complex forward FFT.
        
        :return: output of FFTW3
        :rtype: :class:`~sigpyproc.FourierSeries.FourierSeries`
        """
        if self.size%2 ==0:
            fftsize = self.size
        else:
            fftsize = self.size-1
        fft_ar = np.empty(fftsize+2,dtype="float32")
        lib.rfft(as_c(self),
                 as_c(fft_ar),
                 fftsize)
        return FourierSeries(fft_ar,self.header.newHeader())

    def runningMean(self,window=10001):
        """Filter time series with a running mean. 
        
        :param window: width in bins of running mean filter
        :type window: int
        
        :return: filtered time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`

        .. note::
        
                Window edges will be dealt with only at the start of the time series.

        """
        tim_ar  = np.empty_like(self)
        lib.runningMean(as_c(self),
                        as_c(tim_ar),
                        C.c_int(window),
                        C.c_int(self.size))
        return tim_ar.view(TimeSeries)

    def runningMedian(self,window=10001):
        """Filter time series with a running median.
        
        :param window: width in bins of running median filter
        :type window: int 
        
        :returns: filtered time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`

        .. note::
        
                Window edges will be dealt with only at the start of the time series.
        """
        tim_ar  = np.empty_like(self)
        lib.runningMedian(as_c(self),
                          as_c(tim_ar),
                          C.c_int(window),
                          C.c_int(self.size))
        return tim_ar.view(TimeSeries)

    def applyBoxcar(self,width):
        """Apply a boxcar filter to the time series.

        :param width: width in bins of filter
        :type width: int
        
        :return: filtered time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`

        .. note::
        
                Time series returned is of size nsamples-width with width/2 removed removed from either end.
        """
        tim_ar  = np.empty_like(self)
        lib.runBoxcar(as_c(self),
                      as_c(tim_ar),
                      C.c_int(width),
                      C.c_int(self.size))
        return tim_ar.view(TimeSeries)

    def downsample(self,factor):
        """Downsample the time series.

        :param factor: factor by which time series will be downsampled
        :type factor: int
        
        :return: downsampled time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
 
        .. note::
        
                Returned time series is of size nsamples//factor
        """
        if factor == 1: return self
        newLen = self.size//factor
        tim_ar  = np.empty(newLen,dtype="float32")
        lib.downsampleTim(as_c(self),
                          as_c(tim_ar),
                          C.c_int(factor),
                          C.c_int(newLen))
        return TimeSeries(tim_ar,self.newHeader({'tsamp':self.tsamp*factor}))
           
    def toDat(self,basename):
        """Write time series in presto ``.dat`` format.

        :param basename: file basename for output ``.dat`` and ``.inf`` files
        :type basename: string
        
        :return: ``.dat`` file name and ``.inf`` file name
        :rtype: :func:`tuple` of :func:`str`
        
        .. note::
        
                Method also writes a corresponding .inf file from the header data
        """
        self.header.makeInf(outfile="%s.inf"%(basename))
        datfile = open("%s.dat"%(basename),"w+")
        if self.size%2 != 0:
            self[:-1].tofile(datfile)
        else:
            self.tofile(datfile)
        return "%s.dat"%(basename),"%s.inf"%(basename) 

    def toFile(self,filename):
        """Write time series in sigproc format.

        :param filename: output file name
        :type filename: str
        
        :return: output file name
        :rtype: :func:`str`
        """
        outfile = self.header.prepOutfile(filename,nbits=32)
        self.tofile(outfile)
        return outfile.name
        
    def pad(self,npad):
        """Pad a time series with mean valued data.

        :param npad: number of padding points
        :type nzeros: int
       
        :return: padded time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
        """        
        new_ar = np.hstack((self,self.mean()*np.ones(npad)))
        return TimeSeries(new_ar,self.header.newHeader())

    def resample(self,accel,jerk=0):
        """Perform time domain resampling to remove acceleration and jerk.

        :param accel: The acceleration to remove from the time series
        :type accel: float
        
        :param jerk: The jerk/jolt to remove from the time series
        :type jerk: float

        :param period: The mimimum period that the resampling will be sensitive to.
        :type period: float

        :return: resampled time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
        """
        speed_of_light = 299792458.0
        total_drift = (accel*self.header.tsamp * self.size**2) / (2*299792458.0)
        new_size = self.size+int(total_drift - np.sign(accel)) #factor to handle rounding
        out_ar = np.zeros(new_size,dtype="float32")
        lib.resample(as_c(self),
                     as_c(out_ar),
                     C.c_int(self.size),
                     C.c_float(accel),
                     C.c_float(self.header.tsamp))
        
        new_header = self.header.newHeader({"nsamples":out_ar.size,
                                            "accel":accel})
        return TimeSeries(out_ar,new_header)
    
    def resample2(self,accel,jerk=0):
        speed_of_light = 299792458.0
        total_drift = (accel*self.header.tsamp * (self.size/2)**2) / (2*299792458.0)
        new_size = self.size+int(total_drift - np.sign(accel)) #factor to handle rounding                                   
        out_ar = np.zeros(new_size,dtype="float32")
        lib.resample2(as_c(self),
                     as_c(out_ar),
                     C.c_int(self.size),
                     C.c_float(accel),
                     C.c_float(self.header.tsamp))

        new_header = self.header.newHeader({"nsamples":out_ar.size,
                                            "accel":accel})
        return TimeSeries(out_ar,new_header)


from sigpyproc.FoldedData import FoldedData
from sigpyproc.FourierSeries import FourierSeries
