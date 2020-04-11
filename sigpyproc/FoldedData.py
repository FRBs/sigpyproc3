import numpy as np
import ctypes as C
from sigpyproc.Utils import rollArray
from os import popen

from .ctype_helper import load_lib
lib  = load_lib("libSigPyProc.so")

class Profile(np.ndarray):
    """Class to handle a 1-D pulse profile.

    :param input_array: a pulse profile in array form
    :type input_array: :class:`numpy.ndarray`
    """
     
    def __new__(cls,input_array): 
        obj = input_array.astype("float32").view(cls)
        return obj
     
    def __array_finalize__(self,obj):
        if obj is None: return
         
    def _getWidth(self):
        self-=np.median(self)
        trial_widths = np.arange(1,self.size)
        convmaxs = np.array([np.convolve(np.ones(ii),self,mode="same").max()
                             /np.sqrt(ii) for ii in trial_widths])
        return trial_widths[convmaxs.argmax()]
     
    def _getPosition(self,width):
        return np.convolve(np.ones(width),self,mode="same").argmax()
     
    def _getBaseline(self,width):
        pos = self._getPosition(width)
        wing = np.ceil(width/2.)
        return np.hstack((self[:pos-wing],self[pos+wing+1:]))
     
    def SN(self):
        """Return a rudimentary signal-to-noise measure for the profile.
        
        .. note::
          
           This is a bare-bones, quick-n'-dirty algorithm that should not be used for 
           high quality signal-to-noise measurements.
        """   
        tmp_ar = self.copy()
        width= self._getWidth()
        baseline = self._getBaseline(width)
        tmp_ar-=baseline.mean()
        tmp_ar/=baseline.std()
        return float(tmp_ar.sum()/np.sqrt(width))

    def retroProf(self,height=0.7,width=0.7):
        """Display the profile in ASCII formay in the terminal window.

        :param height: fraction of terminal rows to use
        :type height: float

        :param width: fraction of terminal columns to use
        :param width:

        .. note::
           
           This function requires a system call to the Linux/Unix ``stty`` command.
        """
             
        rows, columns = popen('stty size', 'r').read().split()
        rows = int(int(rows)*height)
        columns = int(int(columns)*width)
        bins = np.linspace(0,self.size-1,columns)
        new_prof = np.interp(bins,np.arange(self.size),self)
        new_prof -= new_prof.min()
        new_prof /= new_prof.max()
        new_prof *= rows
        for ii in np.arange(rows)[::-1]:
            print("".join([("#" if val >= ii else " ") for val in new_prof]))

          
class FoldSlice(np.ndarray):
    """Class to handle a 2-D slice of a :class:`~sigpyproc.FoldedData.FoldedData` instance.
    
    :param input_array: a 2-D array with phase in x axis.
    :type input_array: :class:`numpy.ndarray`
    """
    def __new__(cls,input_array):
        obj = input_array.astype("float32").view(cls)
        return obj
     
    def __array_finalize__(self,obj):
        if obj is None: return

    def normalise(self):
        """Normalise the slice by dividing each row by its mean.
        
        :return: normalised version of slice
        :rtype: :class:`~sigpyproc.FoldedData.FoldSlice`
        """
        return self/self.mean(axis=1).reshape(self.shape[0],1)
          
    def getProfile(self):
        """Return the pulse profile from the slice.
        
        :return: a pulse profile
        :rtype: :class:`~sigpyproc.FoldedData.Profile`
        """
        return self.sum(axis=0).view(Profile)
          
class FoldedData(np.ndarray):
    """Class to handle a data cube produced by any of the sigpyproc folding methods.

    :param input_array: 3-D array of folded data
    :type input_array: :class:`numpy.ndarray`
     
    :param header: observational metadata
    :type header: :class:`~sigpyproc.Header.Header`

    :param period: period that data was folded with
    :type period: float

    :param dm: DM that data was folded with
    :type dm: float
     
    :param accel: accleration that data was folded with (def=0)
    :type accel: float

    .. note::

       Data cube should have the shape:
       (number of subintegrations, number of subbands, number of profile bins)
    """

    def __new__(cls,input_array,header,period,dm,accel=0):
        obj = input_array.astype("float32").view(cls)
        obj.header = header
        obj.period = period
        obj.dm = dm
        obj.accel = accel
        obj._setDefaults()
        return obj

    def __array_finalize__(self,obj):
        if obj is None: return
        self.header = getattr(obj,"header",None)
        self.period = getattr(obj,"period",None)
        self.dm = getattr(obj,"dm",None)
        self.accel = getattr(obj,"accel",None)

    def _setDefaults(self):
        self.nints   = self.shape[0]
        self.nbands  = self.shape[1]
        self.nbins   = self.shape[2]
        self._orig   = self.copy()
        self._period = self.period
        self._dm     = self.dm
        self._tph_shifts = np.zeros(self.nints,dtype="int32")
        self._fph_shifts = np.zeros(self.nbands,dtype="int32")
          
    def getSubint(self,n):
        """Return a single subintegration from the data cube.
        
        :param n: subintegration number (n=0 is first subintegration)
        :type n: int
        
        :return: a 2-D array containing the subintegration
        :rtype: :class:`~sigpyproc.FoldedData.FoldSlice`
        """
        return self[n].view(FoldSlice)

    def getSubband(self,n):
        """Return a single subband from the data cube.

        :param n: subband number (n=0 is first subband)
        :type n: int

        :return: a 2-D array containing the subband
        :rtype: :class:`~sigpyproc.FoldedData.FoldSlice`
        """
        return self[:,n].view(FoldSlice)
         
    def getProfile(self):
        """Return a the data cube summed in time and frequency.

        :return: a 1-D array containing the power as a function of phase (pulse profile)
        :rtype: :class:`~sigpyproc.FoldedData.Profile`
        """
        return self.sum(axis=0).sum(axis=0).view(Profile)
     
    def getTimePhase(self):
        """Return the data cube collapsed in frequency.
        
        :return: a 2-D array containing the time vs. phase plane
        :rtype: :class:`~sigpyproc.FoldedData.FoldSlice`
        """
        return self.sum(axis=1).view(FoldSlice)

    def getFreqPhase(self):
        """Return the data cube collapsed in time.
        
        :return: a 2-D array containing the frequency vs. phase plane
        :rtype: :class:`~sigpyproc.FoldedData.FoldSlice`
        """
        return self.sum(axis=0).view(FoldSlice)

    def centre(self):
        """Try and roll the data cube to center the pulse."""
        p = self.getProfile()
        pos = p._getPosition(p._getWidth())
        self = rollArray(self,(pos-self.nbins/2),2)
               
    def _replaceNan(self):
        bad_ids = np.where(np.isnan(self))
        good_ids = np.where(np.isfinite(self))
        med = np.median(self[good_ids])
        self[bad_ids] =med
          
    def _normalise(self):
        self.freqPhase /= self.freqPhase.mean(axis=1).reshape(self.nbands,1)
        self.timePhase /= self.timePhase.mean(axis=1).reshape(self.nints,1)

    def _getDMdelays(self,dm):
        delta_dm = dm-self._dm
        if delta_dm == 0:
            drifts = -1*self._fph_shifts
            self._fph_shifts[:] = 0
            return drifts
        else:
            chan_width = self.header.foff*self.header.nchans/self.nbands
            freqs  = (np.arange(self.nbands)*chan_width)+self.header.fch1
            fact   = delta_dm * 4.148808e3  
            drifts = (fact * ((freqs**-2)-(self.header.fch1**-2))/((self.period/self.nbins)))
            drifts = drifts.round().astype("int32")
            bin_drifts = drifts - self._fph_shifts
            self._fph_shifts = drifts
            return bin_drifts
        
    def _getPdelays(self,p):
        dbins = (p/self._period-1)*self.header.tobs*self.nbins/self._period
        if dbins == 0:
            drifts = -1*self._tph_shifts
            self._tph_shifts[:] = 0
            return drifts
        else:
            drifts = np.round(np.arange(float(self.nints))/(self.nints/dbins)
                              ).astype("int32")
            bin_drifts = drifts-self._tph_shifts
            self._tph_shifts = drifts
            return bin_drifts

    def updateParams(self,dm=None,period=None):
        """Install a new folding period and/or DM in the data cube.
        
        :param dm: the new DM to dedisperse to
        :type dm: float

        :param period: the new period to fold with
        :type period: float
        """
          
        if dm is None: 
            dm = self.dm
        if period is None: 
            period = self.period
          
        dmdelays = self._getDMdelays(dm)
        pdelays = self._getPdelays(period)
        for ii in range(self.nints):
            for jj in range(self.nbands):
                if dmdelays is not None:
                    self[ii][jj] = rollArray(self[ii][jj],dmdelays[jj],0)
                if pdelays is not None:
                    self[ii][jj] = rollArray(self[ii][jj],pdelays[ii],0)
        self.dm = dm
        self.period = period
                    
