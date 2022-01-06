import io
import os
import warnings
import numpy as np

from typing import List

from sigpyproc.io.bits import BitsInfo, unpack, pack
from sigpyproc.utils import get_logger


class _FileBase(object):
    """File I/O base class."""

    def __init__(self, files, mode, opener=None):
        self.files = files
        self.mode = mode
        self.opener = io.FileIO if opener is None else opener
        # TODO cHECK IF THE FILE EXISTS, otherwise raise OSError('ran out of files.')
        self.file_obj = None
        self.ifile_cur = None
        self._open(ifile=0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close_current()

    def _open(self, ifile):
        if ifile != self.ifile_cur:
            file_obj = self.opener(self.files[ifile], mode=self.mode)
            self._close_current()
            self.file_obj = file_obj
            self.ifile_cur = ifile

    def _close_current(self):
        """Close the currently open local file, and therewith the set."""
        if self.ifile_cur is not None:
            self.file_obj.close()


class FileReader(_FileBase):
    def __init__(
        self,
        files: List[str],
        hdrlens: List[int],
        datalens: List[int],
        mode: str = "r",
        nbits: int = 8,
    ) -> None:
        self.hdrlens = hdrlens
        self.datalens = datalens
        self.nbits = nbits
        self.bitsinfo = BitsInfo(nbits)
        self._configure_logger()

        super().__init__(files, mode)

    @property
    def cur_data_pos(self) -> int:
        return self.file_obj.tell() - self.hdrlens[self.ifile_cur]

    def cread(self, nunits: int) -> np.ndarray:
        """Read nunits (nbytes) of data from the file.

        Parameters
        ----------
        nunits : int
            number of units (nbytes) to be read from the file

        Returns
        -------
        :py:obj:`~numpy.ndarray`
            an 1-D array containing the read data

        Raises
        ------
        IOError
            if file is closed.
        """
        if self.file_obj.closed:
            raise IOError("Cannot read closed file.")

        count = nunits // self.bitsinfo.bitfact
        data = []
        while count >= 0:
            count_read = min(self.datalens[self.ifile_cur], count)
            data_read = np.fromfile(
                self.file_obj, count=count_read, dtype=self.bitsinfo.dtype
            )
            count -= data_read.shape[0]
            data.append(data_read)

            if count == 0:
                break
            self._seek2hdr(self.ifile_cur + 1)

        data_ar = np.concatenate(np.asarray(data))
        if self.bitsinfo.unpack:
            return unpack(data_ar, self.nbits)
        return data_ar

    def seek(self, offset: int, whence: int = 0) -> None:
        """Change the multifile stream position to the given data offset.

        offset is always interpreted for a headerless file and is
        relative to start of the first file.

        Parameters
        ----------
        offset : int
            Absolute position to seek in the data stream
        whence : int
            0 (SEEK_SET) 1 (SEEK_CUR)

        Raises
        ------
        ValueError
            if whence is not 0 or 1.
        """
        if self.file_obj.closed:
            raise ValueError("Cannot read closed file.")

        if whence == 0:
            self._seek_set(offset)
        elif whence == 1:
            self._seek_cur(offset)
        else:
            raise ValueError("whence should be either 0 (SEEK_SET) or 1 (SEEK_CUR)")

    def _seek2hdr(self, fileid):
        self._open(fileid)
        self.file_obj.seek(self.hdrlens[fileid])

    def _seek_set(self, offset):
        if offset < 0:
            raise ValueError("offset should be zero or positive when SEEK_SET")

        cumsum_data_bytes = np.cumsum(self.datalens)
        fileid = np.where(offset < cumsum_data_bytes)[0][0]
        self._seek2hdr(fileid)

        if fileid == 0:
            self.file_obj.seek(offset, os.SEEK_CUR)
        else:
            file_offset = offset - cumsum_data_bytes[fileid - 1]
            self.file_obj.seek(file_offset, os.SEEK_CUR)

    def _seek_cur(self, offset):
        cumsum_data_bytes = np.cumsum(self.datalens) - self.cur_data_pos
        fileid = np.where(offset <= cumsum_data_bytes)[0][0]

        if fileid == self.ifile_cur:
            self.file_obj.seek(offset, os.SEEK_CUR)
        else:
            self._seek2hdr(fileid)
            file_offset = offset - cumsum_data_bytes[fileid - 1]
            self.file_obj.seek(file_offset, os.SEEK_CUR)

    def _configure_logger(self, **kwargs):
        logger_name = "FileReader"
        self.logger = get_logger(logger_name, **kwargs)


class FileWriter(_FileBase):
    def __init__(
        self,
        file,
        mode="w",
        nbits=8,
        quantize=False,
        tsamp=None,
        nchans=None,
        interval_seconds=10,
        constant_offset_scale=False,
        **kwargs,
    ):
        super().__init__([file], mode)
        self.name = file
        self.nbits = nbits
        self.bitsinfo = BitsInfo(nbits)
        self.quantize = quantize

        if self.quantize:
            if self.nbits == 32:
                raise ValueError("Output nbits can not be 32 while quantizing")
            digi = self.bitsinfo.properties()
            digi.update(
                (key, value) for key, value in kwargs.items() if key in digi.keys()
            )
            self._transform = Transform(
                tsamp,
                nchans,
                digi["digi_mean"],
                digi["digi_scale"],
                digi["digi_min"],
                digi["digi_max"],
                interval_seconds,
                constant_offset_scale,
            )

    def cwrite(self, ar):
        """Write an array to file.

        Parameters
        ----------
        ar : :py:obj:`~numpy.ndarray`
            a 1-D numpy array

        Notes
        -----
        Regardless of the dtype of the array argument, the data will be packed
        with a bitsize determined by the nbits attribute of the File instance.
        To change this attribute, use the _setNbits methods.
        It is the responsibility of the user to ensure that values in the array
        do not go beyond the maximum and minimum values allowed by the nbits
        attribute.
        """
        if self.quantize:
            ar = self._transform.rescale(ar)
            ar = self._transform.quantize(ar)

        if self.bitsinfo.dtype != ar.dtype:
            warnings.warn(
                f"Given data (dtype={ar.dtype}) will be unsafely cast to the"
                + f"requested dtype={self.bitsinfo.dtype} before being written out to file",
                stacklevel=2,
            )
            ar = ar.astype(self.bitsinfo.dtype, casting="unsafe")
        # The lib.pack function has an assumption that the given array has 8-bit
        # data. If the given array was, say 32-bit floats and the requested nbits
        # is, say 2-bit, then the output will be garbage, hence the casting above is
        # necessary.
        if self.bitsinfo.unpack:
            packed = pack(ar, self.nbits)
            packed.tofile(self.file_obj)
        else:
            ar.tofile(self.file_obj)

    def write(self, bo):
        """Write the given bytes-like object, bo to the file stream.

        Wrapper for io.RawIOBase.write().

        Parameters
        ----------
        bo : bytes
            bytes-like object
        """
        self.file_obj.write(bo)

    def close(self):
        """Close the currently open file object."""
        self._close_current()


class Transform(object):
    def __init__(
        self,
        tsamp,
        nchans,
        digi_mean,
        digi_scale,
        digi_min,
        digi_max,
        interval_seconds=10,
        constant_offset_scale=False,
    ):
        self.tsamp = tsamp
        self.nchans = nchans
        self.interval_seconds = interval_seconds
        self.constant_offset_scale = constant_offset_scale

        self.digi_mean = digi_mean
        self.digi_scale = digi_scale
        self.digi_min = digi_min
        self.digi_max = digi_max

        self.scale = None
        self.offset = None
        self.first_call = True

        self._initialize_arr()

    @property
    def interval_samples(self):
        return round(self.interval_seconds / self.tsamp)

    def rescale(self, data):
        data = data.reshape(-1, self.nchans)

        if not self.constant_offset_scale or self.first_call:
            self._compute_stats(data)

        normdata = (data + self.offset) * self.scale
        self.first_call = False
        return normdata.ravel()

    def quantize(self, data):
        ar = (data * self.digi_scale) + self.digi_mean + 0.5
        ar = ar.astype(int)
        return np.clip(ar, self.digi_min, self.digi_max)

    def _compute_stats(self, data):
        self.sum_ar += np.sum(data, axis=0)
        self.sumsq_ar += np.sum(data ** 2, axis=0)
        self.isample += data.shape[0]

        if self.isample >= self.interval_samples or self.first_call:
            mean = self.sum_ar / self.isample
            variance = self.sumsq_ar / self.isample - mean * mean
            self.offset = -mean
            self.scale = np.where(
                np.isclose(variance, 0, atol=1e-5), 1, 1.0 / np.sqrt(variance)
            )
            self._initialize_arr()

    def _initialize_arr(self):
        self.sum_ar = np.zeros(self.nchans, dtype=float)
        self.sumsq_ar = np.zeros(self.nchans, dtype=float)
        self.isample = 0
