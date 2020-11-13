import os
import numpy as np
import inspect
import struct
from tqdm import tqdm

import sigpyproc.HeaderParams as conf
from sigpyproc.Utils import File
from sigpyproc.Header import Header
from sigpyproc.Filterbank import Filterbank, FilterbankBlock
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
    def __init__(self, filename):
        self.filename = filename
        self.header   = parseSigprocHeader(self.filename)
        self._file    = File(filename, "r", self.header.nbits)
        self.itemsize = np.dtype(self.header.dtype).itemsize
        if self.header.nbits in [1, 2, 4]:
            self.bitfact = 8 // self.header.nbits
        else:
            self.bitfact = 1
        self.sampsize = self.header.nchans * self.itemsize // self.bitfact
        super().__init__()

    def readBlock(self, start, nsamps, as_filterbankBlock=True):
        """Read a block of filterbank data.

        :param start: first time sample of the block to be read
        :type start: int

        :param nsamps: number of samples in the block (i.e. block will be nsamps*nchans in size)
        :type nsamps: int

        :param as_filterbankBlock: whether to read data as filterbankBlock or numpy array
        :type as_filterbankBlock: bool

        :return: 2-D array of filterbank data
        :rtype: :class:`~sigpyproc.Filterbank.FilterbankBlock`
        """
        self._file.seek(self.header.hdrlen + start * self.sampsize)
        data = self._file.cread(self.header.nchans * nsamps)
        nsamps_read = data.size // self.header.nchans
        data = data.reshape(nsamps_read, self.header.nchans).transpose()
        start_mjd  = self.header.mjdAfterNsamps(start)
        new_header = self.header.newHeader({'tstart': start_mjd})
        if as_filterbankBlock:
            return FilterbankBlock(data, new_header)
        else:
            return data

    def readDedispersedBlock(self, start, nsamps, dm, as_filterbankBlock=True, small_reads=True):
        """Read a block of dedispersed filterbank data, best used in cases where
            I/O time dominates reading a block of data.

        :param start: first time sample of the block to be read
        :type start: int

        :param nsamps: number of samples in the block (i.e. block will be nsamps*nchans in size)
        :type nsamps: int

        :param dm: dispersion measure to dedisperse at
        :type dm: float

        :param as_filterbankBlock: whether to read data as filterbankBlock or numpy array
        :type as_filterbankBlock: bool

        :param small_reads: if the datum size is greater than 1 byte, only read the data needed
            instead of every frequency of every sample
        :type small_reads: bool

        :return: 2-D array of filterbank data
        :rtype: :class:`~sigpyproc.Filterbank.FilterbankBlock`
        """
        data = np.zeros((self.header.nchans, nsamps), dtype=self._file.dtype)
        min_sample = start + self.header.getDMdelays(dm)
        max_sample = min_sample + nsamps
        curr_sample = np.zeros(self.header.nchans, dtype=int)

        start_mjd  = self.header.mjdAfterNsamps(start)
        new_header = self.header.newHeader({'tstart': start_mjd})

        lowest_chan, highest_chan, sample_offset = (0, 0, start)
        with tqdm(total=nsamps * self.header.nchans) as progress:
            while curr_sample[-1] < nsamps:
                relevant_channels = np.argwhere(np.logical_and(max_sample > sample_offset,
                                                min_sample <= sample_offset)).flatten()
                lowest_chan = np.min(relevant_channels)
                highest_chan = np.max(relevant_channels)
                sampled_chans = np.arange(lowest_chan, highest_chan + 1, dtype=int)
                read_length = sampled_chans.size

                if self.bitfact == 1 and small_reads:
                    next_offset = sample_offset * self.sampsize + lowest_chan * self.itemsize
                    self._file.seek(self.header.hdrlen + next_offset)

                    data[sampled_chans, curr_sample[sampled_chans]] = self._file.cread(read_length)

                else:
                    next_offset = sample_offset * self.sampsize
                    self._file.seek(self.header.hdrlen + next_offset)

                    sample = self._file.cread(self.sampsize)
                    data[sampled_chans, curr_sample[sampled_chans]] = sample[sampled_chans]

                curr_sample[sampled_chans] += 1

                if curr_sample[highest_chan] > nsamps:
                    sample_offset = min_sample[highest_chan + 1]
                else:
                    sample_offset += 1

                progress.update(read_length)

        if as_filterbankBlock:
            data = FilterbankBlock(data, new_header)
            data.dm = dm
            return data
        else:
            return data

    def readPlan(self, gulp, skipback=0, start=0, nsamps=None, tqdm_desc=None, verbose=True):
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
            nsamps = self.header.nsamples - start
        gulp     = min(nsamps, gulp)
        skipback = abs(skipback)
        if skipback >= gulp:
            raise ValueError("readsamps must be > skipback value")
        self._file.seek(self.header.hdrlen + start * self.sampsize)
        nreads   = nsamps // (gulp - skipback)
        lastread = nsamps - (nreads * (gulp - skipback))
        if lastread < skipback:
            nreads  -= 1
            lastread = nsamps - (nreads * (gulp - skipback))
        blocks = [(ii, gulp * self.header.nchans, -skipback * self.header.nchans)
                  for ii in range(nreads)]
        if lastread != 0:
            blocks.append((nreads, lastread * self.header.nchans, 0))

        if verbose:
            print("\nFilterbank reading plan:")
            print("------------------------")
            print(f"Called on file:       {self.filename}")
            print(f"Called by:            {inspect.stack()[1][3]}")
            print(f"Number of samps:      {nsamps}")
            print(f"Number of reads:      {nreads}")
            print(f"Nsamps per read:      {blocks[0][1]//self.header.nchans}")
            print(f"Nsamps of final read: {blocks[-1][1]//self.header.nchans}")
            print(f"Nsamps to skip back:  {-1*blocks[0][2]//self.header.nchans}\n")

        if tqdm_desc is None:
            tqdm_desc = f'{inspect.stack()[1][3]} : '
        for ii, block, skip in tqdm(blocks, desc=tqdm_desc):
            data = self._file.cread(block)
            self._file.seek(skip * self.itemsize // self.bitfact, os.SEEK_CUR)
            yield int(block // self.header.nchans), int(ii), data


def readDat(filename, inf=None):
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
        inf = f"{basename}.inf"
    if not os.path.isfile(inf):
        raise IOError("No corresponding inf file found")
    header = parseInfHeader(inf)
    f = File(filename, "r", nbits=32)
    data = np.fromfile(f, dtype="float32")
    header["basename"] = basename
    header["inf"]      = inf
    header["filename"] = filename
    header["nsamples"] = data.size
    return TimeSeries(data, header)


def readTim(filename):
    """Read a sigproc format time series from file.

    :param filename: the name of the file to read
    :type filename: :func:`str`

    :return: an array containing the whole file contents
    :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
    """
    header = parseSigprocHeader(filename)
    nbits  = header["nbits"]
    hdrlen = header["hdrlen"]
    f = File(filename, "r", nbits=nbits)
    f.seek(hdrlen)
    data = np.fromfile(f, dtype=header["dtype"]).astype("float32")
    return TimeSeries(data, header)


def readFFT(filename, inf=None):
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
        inf = f"{basename}.inf"
    if not os.path.isfile(inf):
        raise IOError("No corresponding inf file found")
    header = parseInfHeader(inf)
    f = File(filename, "r", nbits=32)
    data = np.fromfile(f, dtype="float32")
    header["basename"] = basename
    header["inf"]      = inf
    header["filename"] = filename
    return FourierSeries(data, header)


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
    hdrlen = header["hdrlen"]
    f = File(filename, "r", nbits=32)
    f.seek(hdrlen)
    data = np.fromfile(f, dtype="complex32")
    return FourierSeries(data, header)


def parseInfHeader(filename):
    """Parse the metadata from a presto ``.inf`` file.

    :param filename: file containing the header
    :type filename: :func:`str`

    :return: observational metadata
    :rtype: :class:`~sigpyproc.Header.Header`
    """
    header = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        key = line.split("=")[0].strip()
        val = line.split("=")[-1].strip()
        if key not in list(conf.inf_to_header.keys()):
            continue
        else:
            key, keytype = conf.inf_to_header[key]
            header[key] = keytype(val)

    header["src_raj"]      = float("".join(header["src_raj"].split(":")))
    header["src_dej"]      = float("".join(header["src_dej"].split(":")))
    header["telescope_id"] = conf.telescope_ids.get(header["telescope_id"], 10)
    header["machine_id"]   = conf.machine_ids.get(header["machine_id"], 9)
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
    f = open(filename, "rb")
    header = {}
    try:
        keylen = struct.unpack("I", f.read(4))[0]
    except struct.error:
        raise IOError("File Header is not in sigproc format... Is file empty?")
    key = f.read(keylen)
    if key != b"HEADER_START":
        raise IOError("File Header is not in sigproc format")
    while True:
        keylen = struct.unpack("I", f.read(4))[0]
        key = f.read(keylen)

        # convert bytestring to unicode (Python 3)
        try:
            key = key.decode()
        except UnicodeDecodeError as e:
            print(f"Could not convert to unicode: {str(e)}")

        if key not in list(conf.header_keys.keys()):
            print(f"'{key}' not recognised header key")
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

    header["hdrlen"]   = f.tell()
    f.seek(0, 2)
    header["filelen"]  = f.tell()
    header["nbytes"]   = header["filelen"] - header["hdrlen"]
    header["nsamples"] = 8 * header["nbytes"] // header["nbits"] // header["nchans"]
    f.seek(0)
    header["filename"] = filename
    header["basename"] = os.path.splitext(filename)[0]
    f.close()
    return Header(header)


def _read_char(f):
    return struct.unpack("b", f.read(1))[0]


def _read_string(f):
    strlen = struct.unpack("I", f.read(4))[0]
    return f.read(strlen).decode()


def _read_int(f):
    return struct.unpack("I", f.read(4))[0]


def _read_double(f):
    return struct.unpack("d", f.read(8))[0]
