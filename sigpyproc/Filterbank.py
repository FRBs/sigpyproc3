from numpy.ctypeslib import as_ctypes as as_c
from sigpyproc.Utils import rollArray
from sigpyproc.FoldedData import FoldedData
import ctypes as C
import numpy as np

class Filterbank(object):
    """Class exporting methods for the manipulation of frequency-major
    order pulsar data.

    .. note::

       The Filterbank class should never be instantiated directly. Instead it 
       should be inherited by data reading classes.
    """
    def __init__(self):
        if self.header.nbits == 32:
            self.lib = C.CDLL("libSigPyProc32.so") #if 32-bit data select 32-bit library
        else:
            self.lib = C.CDLL("libSigPyProc8.so") #if 8-bit data select 8-bit library
        self.chan_means  = None
        self.chan_stdevs = None
        self.chan_maxima = None
        self.chan_minima = None


    def setNthreads(self,nthreads=None):
        """Set the number of threads available to OpenMP.
        
        :param nthreads: number of threads to use (def = 4)
        :type nthreads: int
        """
        if nthreads is None:
            nthreads=4
        self.lib.omp_set_num_threads(nthreads)

    def collapse(self,gulp=512,start=0,nsamps=None):
        """Sum across all frequencies for each time sample.

        :param gulp: number of samples in each read
        :type gulp: int

        :return: A zero-DM time series 
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
        """
        if nsamps is None:
            size = self.header.nsamples-start
        else:
            size = nsamps
        timar = np.zeros(size,dtype="float32")
        timar_c = as_c(timar)
        for nsamps,ii,data in self.readPlan(gulp,start=start,nsamps=nsamps):
            self.lib.getTim(as_c(data),
                            timar_c,
                            C.c_int(self.header.nchans),
                            C.c_int(nsamps),
                            C.c_int(ii*gulp))
        return TimeSeries(timar,self.header.newHeader({"nchans":1,"refdm":0.0}))

    def invertFreq(self,gulp=512,start=0,nsamps=None,filename=None,back_compatible=True):
        """Invert the frequency ordering of the data and write new data to a new file.
        
        :param gulp: number of samples in each read                                                                                      
        :type gulp: int
        
        :param start: start sample                                                                       
        :type start: int                                                                                     
        
        :param nsamps: number of samples in split                                                
        :type nsamps: int

        :param filename: name of output file (defaults to ``basename_inverted.fil``)
        :type filename: string

        :param back_compatible: sigproc compatibility flag (legacy code)
        :type back_compatible: bool
        
        :return: name of output file
        :return type: :func:`str`
        """
        if filename is None:
            filename = "%s_inverted.fil"%(self.header.basename)

        if nsamps is None:
            size = self.header.nsamples-start
        else:
            size = nsamps
        out_ar = np.empty(size*self.header.nchans,dtype=self.header.dtype)

        if self.header.foff >= 0.0: 
            sign = 1.0
        else:
            sign = -1.0
        changes    = {"fch1":self.header.fch1+sign*(self.header.nchans-1)*self.header.foff,
                      "foff":self.header.foff*sign*(-1.0)}
        #NB bandwidth is +ive by default
        out_file = self.header.prepOutfile(filename, changes,
                                          nbits=self.header.nbits,
                                          back_compatible=back_compatible)
        for nsamps,ii,data in self.readPlan(gulp,start=start,nsamps=nsamps):
            self.lib.invertFreq(as_c(data),
                                as_c(out_ar), 
                                C.c_int(self.header.nchans),
                                C.c_int(nsamps))
            out_file.cwrite(out_ar[:nsamps*self.header.nchans])
        out_file.close()
        return out_file.name

    def bandpass(self,gulp=512):
        """Sum across each time sample for all frequencies.

        :param gulp: number of samples in each read
        :type gulp: int

        :return: the bandpass of the data
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
        """
        bpass_ar = np.zeros(self.header.nchans,dtype="float32")
        bpass_ar_c = as_c(bpass_ar)
        for nsamps,ii,data in self.readPlan(gulp):
            self.lib.getBpass(as_c(data),
                              bpass_ar_c,
                              C.c_int(self.header.nchans),
                              C.c_int(nsamps))
        return TimeSeries(bpass_ar,self.header.newHeader({"nchans":1}))

    def dedisperse(self,dm,gulp=10000):
        """Dedisperse the data to a time series.

        :param dm: dispersion measure to dedisperse to 
        :type dm: float

        :param gulp: number of samples in each read
        :type gulp: int

        :return: a dedispersed time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`

        .. note::
        
               If gulp < maximum dispersion delay, gulp is taken to be twice the maximum dispersion delay.
              
        """
        chan_delays   = self.header.getDMdelays(dm)
        chan_delays_c = as_c(chan_delays)
        max_delay     = int(chan_delays.max())
        gulp          = max(2*max_delay,gulp)
        tim_len       = self.header.nsamples-max_delay
        tim_ar        = np.zeros(tim_len,dtype="float32")
        tim_ar_c      = as_c(tim_ar) 
        for nsamps,ii,data in self.readPlan(gulp,skipback=max_delay):
            self.lib.dedisperse(as_c(data),
                                tim_ar_c,
                                chan_delays_c,
                                C.c_int(max_delay),
                                C.c_int(self.header.nchans), 
                                C.c_int(nsamps), 
                                C.c_int(ii*(gulp-max_delay)))
        return TimeSeries(tim_ar,self.header.newHeader({"nchans":1,"refdm":dm}))

    def subband(self,dm,nsub,filename=None,gulp=10000):
        """Produce a set of dedispersed subbands from the data.

        :param dm: the DM of the subbands
        :type dm: float
        
        :param nsub: the number of subbands to produce
        :type nsub: int

        :param filename: output file name of subbands (def=basename_DM.subbands)
        :type filename: :func:`str`
        
        :param gulp: number of samples in each read
        :type gulp: int

        :return: name of output subbands file
        :rtype: :func:`str`
        """
        
        subfactor     = self.header.nchans/nsub
        chan_delays   = self.header.getDMdelays(dm)
        chan_delays_c = as_c(chan_delays)
        max_delay     = int(chan_delays.max())
        gulp          = max(2*max_delay,gulp)
        out_ar        = np.empty((gulp-max_delay)*nsub,dtype="float32") #must be memset to zero in c code
        out_ar_c      = as_c(out_ar)
        new_foff      = self.header.foff*self.header.nchans/nsub
        new_fch1      = self.header.ftop-new_foff/2.
        chan_to_sub   = np.arange(self.header.nchans,dtype="int32")/subfactor
        chan_to_sub_c = as_c(chan_to_sub)
        changes       = {"fch1"  :new_fch1,
                         "foff"  :new_foff,
                         "refdm" :dm,
                         "nchans":nsub,
                         "nbits" :32}
        if filename is None:
            filename = "%s_DM%06.2f.subbands"%(self.header.basename,dm)
        out_file = self.header.prepOutfile(filename, changes, nbits=32,
                                           back_compatible=True)
        
        for nsamps,ii,data in self.readPlan(gulp,skipback=max_delay):
            self.lib.subband(as_c(data),
                             out_ar_c,
                             chan_delays_c,
                             chan_to_sub_c,
                             C.c_int(max_delay),
                             C.c_int(self.header.nchans),
                             C.c_int(nsub),
                             C.c_int(nsamps))
            out_file.cwrite(out_ar[:(nsamps-max_delay)*nsub])
        return filename

    def upTo8bit(self,filename=None,gulp=512,back_compatible=True):
        """Convert 1-,2- or 4-bit data to 8-bit data and write to file.

        :param filename: name of file to write to (defaults to ``basename_8bit.fil`` )
        :type filename: str

        :param gulp: number of samples in each read 
        :type gulp: int

        :param back_compatible: sigproc compatibility flag
        :type back_compatible: bool

        :return: name of output file
        :rtype: :func:`str`
        
        """
        if filename is None:
            filename = "%s_8bit.fil"%(self.header.basename)
            
        out_file = self.header.prepOutfile(filename,{"nbits":8},nbits=8,back_compatible=back_compatible)
        for nsamps,ii,data in self.readPlan(gulp):
            out_file.cwrite(data)
        return out_file.name


    def downsample(self,tfactor=1,ffactor=1,gulp=512,filename=None,back_compatible=True):
        """Downsample data in time and/or frequency and write to file.

        :param tfactor: factor by which to downsample in time
        :type tfactor: int

        :param ffactor: factor by which to downsample in frequency
        :type ffactor: int

        :param gulp: number of samples in each read
        :type gulp: int

        :param filename: name of file to write to (defaults to ``basename_tfactor_ffactor.fil``)
        :type filename: str

        :param back_compatible: sigproc compatibility flag (legacy code)
        :type back_compatible: bool

        :return: output file name
        :rtype: :func:`str`
        """
        if filename is None:
            filename = "%s_f%d_t%d.fil"%(self.header.basename,ffactor,tfactor)
        if not self.header.nchans%ffactor == 0:
            raise ValueError,"Bad frequency factor given"
        if not gulp%tfactor == 0:
            raise ValueError,"Gulp must be a multiple of tfactor"
        out_file = self.header.prepOutfile(filename,
                                   {"tsamp":self.header.tsamp*tfactor,
                                    "nchans":self.header.nchans/ffactor,
                                    "foff":self.header.foff*ffactor},
                                   back_compatible=back_compatible)

        write_ar = np.zeros(gulp*self.header.nchans/ffactor/tfactor,dtype=self.header.dtype)
        write_ar_c = as_c(write_ar)
        for nsamps,ii,data in self.readPlan(gulp):
            self.lib.downsample(as_c(data),
                                write_ar_c,
                                C.c_int(tfactor),
                                C.c_int(ffactor),
                                C.c_int(self.header.nchans),
                                C.c_int(nsamps))
            out_file.cwrite(write_ar)
        return out_file.name

    def fold(self,period,dm,accel=0,nbins=50,nints=32,nbands=32,gulp=10000):
        """Fold data into discrete phase, subintegration and subband bins. 
        
        :param period: period in seconds to fold with
        :type period: float

        :param dm: dispersion measure to dedisperse to
        :type dm: float

        :param accel: acceleration in m/s/s to fold with
        :type accel: float

        :param nbins: number of phase bins in output
        :type nbins: int

        :param nints: number of subintegrations in output
        :type nints: int

        :param nbands: number of subbands in output
        :type nbands: int

        :param gulp: number of samples in each read
        :type gulp: int

        :return: 3 dimensional data cube
        :rtype: :class:`~sigpyproc.FoldedData.FoldedData`
        
        .. note::

                If gulp < maximum dispersion delay, gulp is taken to be twice the maximum dispersion delay.      

        """
        if np.modf(period/self.header.tsamp)[0]<0.001:
            print "WARNING: Foldng interval is an integer multiple of the sampling time"
        if nbins > period/self.header.tsamp:
            print "WARNING: Number of phase bins is greater than period/sampling time"
        if (self.header.nsamples*self.header.nchans)/(nbands*nints*nbins) < 10:
            raise ValueError,"nbands x nints x nbins is too large."
        nbands        = min(nbands,self.header.nchans)
        chan_delays   = self.header.getDMdelays(dm)
        chan_delays_c = as_c(chan_delays)
        max_delay     = int(chan_delays.max())
        gulp          = max(2*max_delay,gulp)
        fold_ar       = np.zeros(nbins*nints*nbands,dtype="float32")
        fold_ar_c     = as_c(fold_ar)
        count_ar      = np.zeros(nbins*nints*nbands,dtype="int32")
        count_ar_c    = as_c(count_ar)
        for nsamps,ii,data in self.readPlan(gulp,skipback=max_delay):
            self.lib.foldFil(as_c(data),
                             fold_ar_c,
                             count_ar_c,
                             chan_delays_c,
                             C.c_int(max_delay),
                             C.c_double(self.header.tsamp),
                             C.c_double(period),
                             C.c_double(accel),
                             C.c_int(self.header.nsamples), 
                             C.c_int(nsamps), 
                             C.c_int(self.header.nchans), 
                             C.c_int(nbins),
                             C.c_int(nints), 
                             C.c_int(nbands), 
                             C.c_int(ii*(gulp-max_delay)))
        fold_ar/=count_ar
        fold_ar = fold_ar.reshape(nints,nbands,nbins)
        return FoldedData(fold_ar,self.header.newHeader(),period,dm,accel)


    def getChan(self,chan,gulp=512):
        """Retrieve a single frequency channel from the data.

        :param chan: channel to retrieve (0 is the highest frequency channel)
        :type chan: int

        :param gulp: number of samples in each read
        :type gulp: int

        :return: selected channel as a time series
        :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
        """
        if chan >= self.header.nchans or chan < 0:
            raise ValueError,"Selected channel out of range."
        tim_ar   = np.empty(self.header.nsamples,dtype="float32")
        tim_ar_c = as_c(tim_ar)
        for nsamps,ii,data in self.readPlan(gulp):
            self.lib.getChan(as_c(data),
                             tim_ar_c,
                             C.c_int(chan),
                             C.c_int(self.header.nchans),
                             C.c_int(nsamps),
                             C.c_int(ii*gulp))
        return TimeSeries(tim_ar,self.header.newHeader({"channel":chan,"refdm":0.0,"nchans":1}))

    def split(self,start,nsamps,filename=None,gulp=1024,back_compatible=True):
        """Split data in time.

        :param start: start sample of split
        :type start: int

        :param nsamps: number of samples in split
        :type nsamps: int

        :param filename: name of output file
        :type filename: :func:`str` 

        :param gulp: number of samples in each read
        :type gulp: int

        :param back_compatible: sigproc compatibility flag (legacy code)
        :type back_compatible: bool

        :return: name of new file
        :rtype: :func:`str`
        """
        if filename is None:
            filename = "%s_%d_%d.fil"%(self.header.basename,start,start+nsamps)
        out_file = self.header.prepOutfile(filename, nbits=self.header.nbits)
        
        for count, ii, data in self.readPlan(gulp,start=start,nsamps=nsamps):
            out_file.cwrite(data)
        out_file.close()
        return out_file.name
       
    def splitToChans(self,gulp=1024,back_compatible=True):
        """Split the data into component channels and write each to file.

        :param gulp: number of samples in each read
        :type gulp: int 

        :param back_compatible: sigproc compatibility flag (legacy code)
        :type back_compatible: bool

        :return: names of all files written to disk
        :rtype: :func:`list` of :func:`str`
        
        .. note::
        
                Time series are written to disk with names based on channel number.

        """
        tim_ar   = np.empty([self.header.nchans,gulp],dtype="float32")
        tim_ar_c = as_c(tim_ar)
        out_files = [self.header.prepOutfile("%s_chan%04d.tim"%(self.header.basename,ii),
                                             {"nchans":1,"nbits":32,"data_type":2},
                                             back_compatible=back_compatible,nbits=32)
            
                     for ii in xrange(self.header.nchans)]
        for nsamps,ii,data in self.readPlan(gulp):
            self.lib.splitToChans(as_c(data),
                                  tim_ar_c,
                                  C.c_int(self.header.nchans),
                                  C.c_int(nsamps),
                                  C.c_int(gulp))
            for ii,f in enumerate(out_files):
                f.cwrite(tim_ar[ii][:nsamps])
                
        for f in out_files:
            f.close()
            
        return [f.name for f in out_files]
        
    def getStats(self,gulp=512):
        """Retrieve channelwise statistics of data.

        :param gulp: number of samples in each read
        :type gulp: int
        
        Function creates four instance attributes:
        
           * :attr:`chan_means`: the mean value of each channel
           * :attr:`chan_stdevs`: the standard deviation of each channel
           * :attr:`chan_max`: the maximum value of each channel
           * :attr:`chan_min`: the minimum value of each channel
        """
        maxima_ar = np.zeros(self.header.nchans,dtype="float32")
        minima_ar = np.zeros(self.header.nchans,dtype="float32")
        means_ar  = np.zeros(self.header.nchans,dtype="float32")
        stdev_ar  = np.zeros(self.header.nchans,dtype="float32")
        maxima_ar_c = as_c(maxima_ar)
        minima_ar_c = as_c(minima_ar)
        means_ar_c  = as_c(means_ar)
        stdev_ar_c  = as_c(stdev_ar)
        for nsamps,ii,data in self.readPlan(gulp):
            self.lib.getStats(as_c(data),
                              means_ar_c,
                              stdev_ar_c,
                              maxima_ar_c,
                              minima_ar_c,
                              C.c_int(self.header.nchans),
                              C.c_int(nsamps),
                              C.c_int(ii))

        means_ar /= self.header.nsamples
        stdev_ar = np.sqrt((stdev_ar/self.header.nsamples)-means_ar**2)
        stdev_ar[np.where(np.isnan(stdev_ar))] = 0        
        self.chan_means = means_ar
        self.chan_stdevs = stdev_ar
        self.chan_maxima = maxima_ar
        self.chan_minima = minima_ar

class FilterbankBlock(np.ndarray):
    """Class to handle a discrete block of data in time-major order.

    :param input_array: 2 dimensional array of shape (nchans,nsamples)
    :type input_array: :class:`numpy.ndarray`

    :param header: observational metadata
    :type header: :class:`~sigpyproc.Header.Header`

    .. note::

            Data is converted to 32 bits regardless of original type. 
    """
    def __new__(cls,input_array,header):
        obj = input_array.astype("float32").view(cls)
        obj.header = header
        obj.lib = C.CDLL("libSigPyProc32.so")
        obj.dm = 0.0
        return obj
    
    def __array_finalize__(self,obj):
        if obj is None: return
        if hasattr(obj,"header"):
            self.header = obj.header
        self.lib = C.CDLL("libSigPyProc32.so")
        self.dm = getattr(obj,"dm",0.0)
        
    def downsample(self,tfactor=1,ffactor=1):
        """Downsample data block in frequency and/or time.

        :param tfactor: factor by which to downsample in time       
        :type tfactor: int                                              

        :param ffactor: factor by which to downsample in frequency                   
        :type ffactor: int

        :return: 2 dimensional array of downsampled data
        :rtype: :class:`~sigpyproc.Filterbank.FilterbankBlock`

        .. note::

                ffactor must be a factor of nchans.

        """ 
        if not self.shape[0]%ffactor == 0:
            raise ValueError,"Bad frequency factor given"
        newnsamps = self.shape[1] - self.shape[0]%tfactor
        new_ar = np.empty(newnsamps*self.shape[0]/ffactor/tfactor,dtype="float32")
        ar = self.transpose().ravel().copy()
        self.lib.downsample(as_c(ar),
                            as_c(new_ar),
                            C.c_int(tfactor),
                            C.c_int(ffactor),
                            C.c_int(self.shape[0]),
                            C.c_int(newnsamps))
        new_ar = new_ar.reshape(newnsamps//tfactor,self.shape[0]//ffactor).transpose()
        new_tsamp = self.header.tsamp/tfactor
        new_nchans = self.header.nchans//ffactor
        new_header = self.header.newHeader({"tsamp":new_tsamp,"nchans":new_nchans})
        return FilterbankBlock(new_ar,new_header)
    
    def toFile(self,filename=None,back_compatible=True):
        """Write the data to file.

        :param filename: name of the output file (defaults to ``basename_split_start_to_end.fil``)  
        :type filename: str 

        :param back_compatible: sigproc compatibility flag (legacy code)
        :type back_compatible: bool

        :return: name of output file
        :rtype: :func:`str`
        """
        if filename is None:
            filename = "%s_%d_to_%d.fil"%(self.header.basename,self.header.tstart,self.header.mjdAfterNsamps(self.shape[1]))
        out_file = self.prepOutfile(filename,back_compatible=back_compatible)
        out_file.cwrite(self)
        return filename

    def normalise(self):
        """Divide each frequency channel by its average.
        
        :return: normalised version of the data
        :rtype: :class:`~sigpyproc.Filterbank.FilterbankBlock`
        """
        return self/self.mean(axis=1).reshape(self.shape[0],1)

    def get_tim(self):
        return self.sum(axis=0)
        
    def get_bandpass(self):
        return self.sum(axis=1)
    
    def dedisperse(self,dm):
        """Dedisperse the block.

        :param dm: dm to dedisperse to
        :type dm: float

        :return: a dedispersed version of the block
        :rtype: :class:`~sigpyproc.Filterbank.FilterbankBlock`
        
        .. note::
        
                Frequency dependent delays are applied as rotations to each
                channel in the block.
        """
        new_ar = self.copy()
        delays = self.header.getDMdelays(dm)
        for ii in range(self.shape[0]):
            new_ar[ii] = rollArray(self[ii],delays[ii]%self.shape[1],0)
        new_ar.dm = dm
        return new_ar
        
from sigpyproc.TimeSeries import TimeSeries

