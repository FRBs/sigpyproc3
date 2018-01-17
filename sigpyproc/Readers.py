import os
import time
import sigpyproc.HeaderParams as conf
import numpy as np
from inspect import stack as istack
from struct import unpack
from sys import stdout
from sigpyproc.Utils import File
from sigpyproc.Header import Header
from sigpyproc.Filterbank import Filterbank,FilterbankBlock
from sigpyproc.TimeSeries import TimeSeries
from sigpyproc.FourierSeries import FourierSeries

class FilReader(Filterbank):
    """Class to handle the reading of sigproc format filterbank files
    
    :param filename: name of filterbank file
    :type filename: :func:`str`
    
    .. note::
    
       To be considered as a Sigproc format filterbank file the header must only 
       contain keywords found in the ``HeaderParams.header_keys`` dictionary. 
    """
    def __init__(self,filename):
        self.filename = filename
        self.header = parseSigprocHeader(self.filename)
        self._file = File(filename,"r",self.header.nbits)
        self.itemsize = np.dtype(self.header.dtype).itemsize
        if self.header.nbits in [1,2,4]:
            self.bitfact = 8/self.header.nbits
        else:
            self.bitfact = 1
        self.sampsize = self.header.nchans*self.itemsize/self.bitfact
        super(FilReader,self).__init__()

    def readBlock(self,start,nsamps):
        """Read a block of filterbank data.
        
        :param start: first time sample of the block to be read
        :type start: int

        :param nsamps: number of samples in the block (i.e. block will be nsamps*nchans in size)
        :type nsamps: int

        :return: 2-D array of filterbank data
        :rtype: :class:`~sigpyproc.Filterbank.FilterbankBlock`
        """
        self._file.seek(self.header.hdrlen+start*self.sampsize)
        data = self._file.cread(self.header.nchans*nsamps)
        data = data.reshape(nsamps,self.header.nchans).transpose()
        start_mjd = self.header.mjdAfterNsamps(start)
        new_header = self.header.newHeader({'tstart':start_mjd})
        return FilterbankBlock(data,new_header)
                
    def readPlan(self,gulp,skipback=0,start=0,nsamps=None,verbose=True):
        """A generator used to perform filterbank reading.
 
        :param gulp: number of samples in each read
        :type gulp: int

        :param skipback: number of samples to skip back after each read (def=0)
        :type skipback: int

        :param start: first sample to read from filterbank (def=start of file)
        :type start: int

        :param nsamps: total number samples to read (def=end of file)
        :type nsamps: int

        :param verbose: flag for display of reading plan information (def=True)
        :type verbose: bool

        :return: An generator that can read through the file.
        :rtype: generator object
        
        .. note::

           For each read, the generator yields a tuple ``x``, where:
           
              * ``x[0]`` is the number of samples read
              * ``x[1]`` is the index of the read (i.e. ``x[1]=0`` is the first read)
              * ``x[2]`` is a 1-D numpy array containing the data that was read

           The normal calling syntax for this is function is:

           .. code-block:: python
           
              for nsamps, ii, data in self.readPlan(*args,**kwargs):
                  # do something

           where data always has contains ``nchans*nsamps`` points. 

        """

        if nsamps is None:
            nsamps = self.header.nsamples-start
        if nsamps<gulp:
            gulp = nsamps
        tstart = time.time()
        skipback = abs(skipback)
        if skipback >= gulp:
            raise ValueError,"readsamps must be > skipback value"
        self._file.seek(self.header.hdrlen+start*self.sampsize)
        nreads = nsamps//(gulp-skipback)
        lastread = nsamps-(nreads*(gulp-skipback))
        if lastread<skipback:
            nreads -= 1
            lastread = nsamps-(nreads*(gulp-skipback))
        blocks = [(ii,gulp*self.header.nchans,-skipback*self.header.nchans) for ii in range(nreads)]
        blocks.append((nreads,lastread*self.header.nchans,0))
        
        if verbose:
            print
            print "Filterbank reading plan:"
            print "------------------------"
            print "Called on file:       ",self.filename      
            print "Called by:            ",istack()[1][3]
            print "Number of samps:      ",nsamps 
            print "Number of reads:      ",nreads
            print "Nsamps per read:      ",blocks[0][1]/self.header.nchans
            print "Nsamps of final read: ",blocks[-1][1]/self.header.nchans
            print "Nsamps to skip back:  ",-1*blocks[0][2]/self.header.nchans
            print
        
        for ii,block,skip in blocks:
            if verbose:
                stdout.write("Percentage complete: %d%%\r"%(100*ii/nreads))
                stdout.flush()
            data = self._file.cread(block)
            self._file.seek(skip*self.itemsize/self.bitfact,os.SEEK_CUR)
            yield int(block/self.header.nchans),int(ii),data
        if verbose:
            print "Execution time: %f seconds     \n"%(time.time()-tstart)



def readDat(filename,inf=None):
    """Read a presto format .dat file.

    :param filename: the name of the file to read
    :type filename: :func:`str`
    
    :params inf: the name of the corresponding .inf file (def=None)
    :type inf: :func:`str`

    :return: an array containing the whole dat file contents
    :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
    
    .. note::

       If inf=None, the function will look for a corresponding file with 
       the same basename which has the .inf file extension.
    """   

    basename = os.path.splitext(filename)[0]
    if inf is None:
        inf = "%s.inf"%(basename)
    if not os.path.isfile(inf):
        raise IOError,"No corresponding inf file found"
    header = parseInfHeader(inf)
    f = File(filename,"r",nbits=32)
    data = np.fromfile(f,dtype="float32")
    header["basename"] = basename
    header["inf"] = inf
    header["filename"] = filename
    header["nsamples"] = data.size
    return TimeSeries(data,header)

def readTim(filename):
    """Read a sigproc format time series from file.

    :param filename: the name of the file to read
    :type filename: :func:`str`
    
    :return: an array containing the whole file contents
    :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
    """
    header = parseSigprocHeader(filename)
    nbits    = header["nbits"]
    hdrlen   = header["hdrlen"]
    f = File(filename,"r",nbits=nbits)
    f.seek(hdrlen)
    data = np.fromfile(f,dtype=header["dtype"]).astype("float32")
    return TimeSeries(data,header)

def readFFT(filename,inf=None):
    """Read a presto .fft format file.

    :param filename: the name of the file to read
    :type filename: :func:`str`
    
    :params inf: the name of the corresponding .inf file (def=None)
    :type inf: :func:`str`
    
    :return: an array containing the whole file contents
    :rtype: :class:`~sigpyproc.FourierSeries.FourierSeries`

    .. note::

       If inf=None, the function will look for a corresponding file with 
       the same basename which has the .inf file extension.
    """
    basename = os.path.splitext(filename)[0]
    if inf is None:
        inf = "%s.inf"%(basename)
    if not os.path.isfile(inf):
        raise IOError,"No corresponding inf file found"
    header = parseInfHeader(inf)
    f = File(filename,"r",nbits=32)
    data = np.fromfile(f,dtype="float32")
    header["basename"] = basename
    header["inf"] = inf
    header["filename"] = filename
    return FourierSeries(data,header)

def readSpec(filename):
    """Read a sigpyproc format spec file.

    :param filename: the name of the file to read
    :type filename: :func:`str`
    
    :return: an array containing the whole file contents
    :rtype: :class:`~sigpyproc.FourierSeries.FourierSeries`

    .. note::

       This is not setup to handle ``.spec`` files such as are
       created by Sigprocs seek module. To do this would require 
       a new header parser for that file format.
    """
    header = parseSigprocHeader(filename)
    hdrlen   = header["hdrlen"]
    f = File(filename,"r",nbits=32)
    f.seek(hdrlen)
    data = np.fromfile(f,dtype="complex32")
    return FourierSeries(data,header)

def parseInfHeader(filename):
    """Parse the metadata from a presto ``.inf`` file.

    :param filename: file containing the header
    :type filename: :func:`str`

    :return: observational metadata
    :rtype: :class:`~sigpyproc.Header.Header`
    """
    f = open(filename,"r")
    header = {}
    lines = f.readlines()
    f.close()
    for line in lines:
        key = line.split("=")[0].strip()
        val = line.split("=")[-1].strip()
        if not key in conf.inf_to_header.keys():
            continue
        else:
            key,keytype = conf.inf_to_header[key]
            header[key] = keytype(val)

    header["src_raj"]      = float("".join(header["src_raj"].split(":")))
    header["src_dej"]      = float("".join(header["src_dej"].split(":")))
    header["telescope_id"] = conf.telescope_ids.get(header["telescope_id"],10)
    header["machine_id"]   = conf.machine_ids.get(header["machine_id"],9)
    header["data_type"]    = 2
    header["nchans"]       = 1
    header["nbits"]        = 32
    header["hdrlen"]       = 0
    header["nsamples"]     = 0
    return Header(header)

def parseSigprocHeader(filename):
    """Parse the metadata from a Sigproc-style file header.

    :param filename: file containing the header
    :type filename: :func:`str`
    
    :return: observational metadata
    :rtype: :class:`~sigpyproc.Header.Header`
    """
    f = open(filename,"r")
    header = {}
    try:
        keylen = unpack("I",f.read(4))[0]
    except struct.error:
        raise IOError,"File Header is not in sigproc format... Is file empty?"
    key = f.read(keylen)
    if key != "HEADER_START":
        raise IOError,"File Header is not in sigproc format"
    while True:
        keylen = unpack("I",f.read(4))[0]
        key = f.read(keylen)
        if not key in conf.header_keys:
            print "'%s' not recognised header key"%(key)
            return None

        if conf.header_keys[key] == "str":
            header[key] = _read_string(f)
        elif conf.header_keys[key] == "I":
            header[key] = _read_int(f)
        elif conf.header_keys[key] == "b":
            header[key] = _read_char(f)
        elif conf.header_keys[key] == "d":
            header[key] = _read_double(f)
        if key == "HEADER_END":
            break

    header["hdrlen"] = f.tell()
    f.seek(0,2)
    header["filelen"]  = f.tell()
    header["nbytes"] =  header["filelen"]-header["hdrlen"]
    header["nsamples"] = 8*header["nbytes"]/header["nbits"]/header["nchans"]
    f.seek(0)
    header["filename"] = filename
    header["basename"] = os.path.splitext(filename)[0]
    f.close()
    return Header(header) 
        
def _read_char(f):
    return unpack("b",f.read(1))[0]

def _read_string(f):
    strlen = unpack("I",f.read(4))[0]
    return f.read(strlen)

def _read_int(f):
    return unpack("I",f.read(4))[0]

def _read_double(f):
    return unpack("d",f.read(8))[0]

