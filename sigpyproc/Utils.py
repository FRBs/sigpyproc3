import io
import numpy as np
import warnings
import logging

from sigpyproc import libSigPyProc as lib
from sigpyproc.HeaderParams import nbits_to_dtype


class File(io.FileIO):
    """A class to handle writing of arbitrary bit size data to file.

    Parameters
    ----------
    filename : str
        name of file to open
    mode : str
        file access mode, can be either "r", "r+", "w" or "a".
    nbits : int, optional
        the bit size of units to be read from or written to file, by default 8

    Notes
    -----
    The File class handles all packing and unpacking of sub-byte size data
    under the hood, so all calls can be made requesting numbers of units
    rather than numbers of bits or bytes.
    """

    def __init__(self, filename, mode, nbits=8):
        super().__init__(filename, mode)
        self.nbits = nbits
        self.dtype = nbits_to_dtype[self.nbits]
        if nbits in {1, 2, 4}:
            self.bitfact = nbits / 8.0
            self.unpack  = True
        else:
            self.bitfact = 1
            self.unpack  = False

    def cread(self, nunits):
        """Read nunits of data from the file.

        Parameters
        ----------
        nunits : int
            number of units to be read from file

        Returns
        -------
        :py:obj:`numpy.ndarray`
            an array containing the read data
        """
        count = int(nunits * self.bitfact)
        data  = np.fromfile(self, count=count, dtype=self.dtype)
        if self.unpack:
            return lib.unpack(data, self.nbits)
        return data

    def cwrite(self, ar):
        """Write an array to file.

        Parameters
        ----------
        ar : :py:obj:`numpy.ndarray`
            a numpy array

        Notes
        -----
        Regardless of the dtype of the array argument, the data will be packed
        with a bitsize determined by the nbits attribute of the File instance.
        To change this attribute, use the _setNbits methods.
        It is the responsibility of the user to ensure that values in the array
        do not go beyond the maximum and minimum values allowed by the nbits
        attribute.
        """
        if self.dtype != ar.dtype:
            warnings.warn(
                f"Given data (dtype={ar.dtype}) will be unsafely cast to the"
                f"requested dtype={self.dtype} before being written out to file",
                stacklevel=2
            )
            ar = ar.astype(self.dtype, casting="unsafe")

        # The lib.pack function has an assumption that the given array has 8-bit
        # data. If the given array was, say 32-bit floats and the requested nbits
        # is, say 2-bit, then the output will be garbage, hence the casting above is
        # necessary.
        if self.unpack:
            packed = lib.pack(ar, self.nbits)
            packed.tofile(self)
        else:
            ar.tofile(self)

    def __del__(self):
        self.close()


def rollArray(y, shift, axis):
    """Roll the elements in the array by 'shift' positions along the
    given axis.

    Parameters
    ----------
    y : :py:obj:`numpy.ndarray`
        array to roll
    shift : int
        number of bins to shift by
    axis : int
        axis to roll along

    Returns
    -------
    :py:obj:`numpy.ndarray`
        shifted numpy array
    """
    y = np.asanyarray(y)
    n = y.shape[axis]
    shift %= n
    return y.take(np.concatenate((np.arange(shift, n), np.arange(shift))), axis)


def _flattenList(n):
    new = []
    repack = lambda x: [new.append(int(y)) for y in x]
    for elem in n:
        if hasattr(elem, "__iter__"):
            repack(elem)
        else:
            new.append(int(elem))
    return new


def stackRecarrays(arrays):
    """Wrapper for stacking :py:obj:`numpy.recarrays`
    """
    return arrays[0].__array_wrap__(np.hstack(arrays))


def nearestFactor(n, val):
    """Find nearest factor.

    :param n: number that we wish to factor
    :type n: int

    :param val: number that we wish to find nearest factor to
    :type val: int

    :return: nearest factor
    :rtype: int
    """
    fact  = [1, n]
    check = 2
    rootn = np.sqrt(n)
    while check < rootn:
        if n % check == 0:
            fact.append(check)
            fact.append(n // check)
        check += 1
    if rootn == check:
        fact.append(check)
    fact.sort()
    return fact[np.abs(np.array(fact) - val).argmin()]


def editInplace(inst, key, value):
    """Edit a sigproc style header in place

    :param inst: a header instance with a ``.filename`` attribute
    :type inst: :class:`~sigpyproc.Header.Header`

    :param key: name of parameter to change (must be a valid sigproc key)
    :type key: :func:`str`

    :param value: new value to enter into header

    .. note::

       It is up to the user to be responsible with this function, as it will directly
       change the file on which it is being operated. The only fail contition of
       editInplace comes when the new header to be written to file is longer or shorter than the
       header that was previously in the file.
    """
    temp = File(inst.header.filename, "r+")
    if key == "source_name":
        oldlen = len(inst.header.source_name)
        value  = value[:oldlen] + " " * (oldlen - len(value))
    inst.header[key] = value
    new_header = inst.header.SPPHeader(back_compatible=True)
    if inst.header.hdrlen == len(new_header):
        temp.seek(0)
        temp.write(new_header)
    else:
        raise ValueError("New header is too long/short for file")


class CustomHandler(logging.StreamHandler):
    def emit(self, record):
        messages = record.msg.splitlines()
        for message in messages:
            record.msg = message
            super().emit(record)


def get_logger(name, formatter=None, level=logging.INFO, logfile=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if formatter is None:
        logformat = "%(asctime)s.%(msecs)03d - %(name)s - %(message)s"
        formatter = logging.Formatter(fmt=logformat, datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.hasHandlers():
        if logfile is None:
            handler = CustomHandler()
        else:
            handler = logging.FileHandler(logfile)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
