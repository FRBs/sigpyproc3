from __future__ import annotations

import io
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Buffer

from sigpyproc.io.bits import BitsInfo, pack, unpack
from sigpyproc.utils import get_logger

if TYPE_CHECKING:
    from typing import Callable

    from typing_extensions import Self

    from sigpyproc.io.sigproc import StreamInfo


def allocate_buffer(allocator: Callable[[int], Buffer], nbytes: int) -> Buffer:
    """Allocate a buffer of the given size safely using the given allocator.

    Parameters
    ----------
    allocator : Callable[[int], Buffer]
        A callable that takes an integer argument and returns a buffer object.
    nbytes : int
        Number of bytes to allocate.

    Returns
    -------
    Buffer
        A buffer object of the given size. collections.abc.Buffer, PEP 688.

    Raises
    ------
    ValueError
        if nbytes is less than or equal to zero.
    RuntimeError
        if the buffer allocation fails.
    TypeError
        if the allocator does not return a buffer object.
    ValueError
        if the allocated buffer is not of the expected size.
    """
    if nbytes <= 0:
        msg = f"Requested buffer size is invalid {nbytes}"
        raise ValueError(msg)
    try:
        buffer = allocator(nbytes)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to allocate buffer of size {nbytes}"
        raise RuntimeError(msg) from exc

    if not isinstance(buffer, Buffer):
        msg = f"Allocator did not return a buffer object {type(buffer)}"
        raise TypeError(msg)

    allocated_nbytes = len(buffer)  # type: ignore [arg-type]
    if allocated_nbytes != nbytes:
        msg = (
            f"Allocated buffer is not the expected size {allocated_nbytes} "
            f"(actual) != {nbytes} (expected)"
        )
        raise ValueError(msg)
    return buffer


class FileBase:
    """File I/O base class."""

    def __init__(self, files: list[str], mode: str) -> None:
        self.files = files
        self.mode = mode
        self.opener = io.FileIO
        self.ifile_cur = -1
        self._open(ifile=0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self._close_current()

    def _open(self, ifile: int) -> None:
        """Open a file from the list of files.

        If a file is already open, it will be closed before opening the new file.

        Parameters
        ----------
        ifile : int
            index of the file to be opened

        Raises
        ------
        ValueError
            if ifile is out of bounds
        """
        if ifile < 0 or ifile >= len(self.files):
            msg = f"ifile should be between 0 and {len(self.files) - 1}, got {ifile}"
            raise ValueError(msg)

        if ifile != self.ifile_cur:
            file_obj = self.opener(self.files[ifile], mode=self.mode)
            self._close_current()
            self.file_obj = file_obj
            self.ifile_cur = ifile

    def eos(self) -> bool:
        """Check if the end of the file stream has been reached."""
        # First check if we are at the end of the current file
        eof = self.file_obj.tell() == os.fstat(self.file_obj.fileno()).st_size
        # Now check if we are at the end of the list of files
        eol = self.ifile_cur == len(self.files) - 1
        return eof & eol

    def _close_current(self) -> None:
        """Close the currently open file object."""
        if self.ifile_cur != -1:
            self.file_obj.close()


class FileReader(FileBase):
    """A file reader class that can read from multiple files.

    Files should have format similar to ``sigproc``.

    Parameters
    ----------
    stream_info : StreamInfo
        stream information object containing header and data lengths
    mode : str, optional
        file opening mode, by default "r"
    nbits : int, optional
        number of bits per sample in the files, by default 8
    """

    def __init__(
        self, stream_info: StreamInfo, mode: str = "r", nbits: int = 8
    ) -> None:
        self.sinfo = stream_info
        self.nbits = nbits
        self.bitsinfo = BitsInfo(nbits)
        self._configure_logger()

        filenames = self.sinfo.get_info_list("filename")
        super().__init__(filenames, mode)

    @property
    def cur_data_pos_file(self) -> int:
        """int: Current data position in the current file."""
        return self.file_obj.tell() - self.sinfo.entries[self.ifile_cur].hdrlen

    @property
    def cur_data_pos_stream(self) -> int:
        """int: Current data position in the data stream."""
        if self.ifile_cur == 0:
            return self.cur_data_pos_file
        return self.cur_data_pos_file + self.sinfo.cumsum_datalens[self.ifile_cur - 1]

    def cread(self, nunits: int) -> np.ndarray:
        """Read nunits data of the given number of bits from the file stream.

        Parameters
        ----------
        nunits : int
            number of units to read

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
            msg = f"Cannot read from closed file {self.files[self.ifile_cur]}"
            raise OSError(msg)

        count = nunits // self.bitsinfo.bitfact
        data = []
        while count >= 0:
            count_read = min(self.sinfo.entries[self.ifile_cur].datalen, count)
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

    def creadinto(
        self,
        read_buffer: Buffer,
        unpack_buffer: Buffer | None = None,
    ) -> int:
        """Read from file stream into a buffer of pre-defined length.

        Parameters
        ----------
        read_buffer : Buffer
            An object exposing the Python Buffer Protocol interface [PEP 3118]

        unpack_buffer : Buffer, optional
            An object exposing the Python Buffer Protocol interface [PEP 3118],
            by default None

        Returns
        -------
        int
            Number of bytes readinto the buffer

        Raises
        ------
        IOError
            if file is closed.

        Notes
        -----
        It is the responsibility of the caller to handle the case than fewer bytes
        than requested are read into the buffer. When at the end of the file stream
        the number of bytes returned will be zero.
        """
        if self.file_obj.closed:
            msg = f"Cannot read from closed file {self.files[self.ifile_cur]}"
            raise OSError(msg)

        nbytes = 0
        read_buffer_view = memoryview(read_buffer)
        while True:
            nbytes_read = self.file_obj.readinto(read_buffer_view[nbytes:])
            if nbytes_read is None:
                if self.eos():
                    # We have reached the end of the stream
                    break
                else:  # noqa: RET508
                    # Might be non-blocking IO, so maybe try again
                    msg = "file might in non-blocking mode"
                    raise OSError(msg)
            else:
                nbytes += nbytes_read
                if nbytes == read_buffer_view.nbytes or self.eos():
                    break
                self._seek2hdr(self.ifile_cur + 1)
        if nbytes < read_buffer_view.nbytes:
            warnings.warn("End of file reached before buffer was filled", stacklevel=2)

        if self.bitsinfo.unpack:
            read_ar = np.frombuffer(read_buffer_view, dtype=np.uint8)
            if unpack_buffer is not None:
                unpack_ar = np.frombuffer(memoryview(unpack_buffer), dtype=np.uint8)
            unpack(read_ar, self.nbits, unpack_ar)
        return nbytes

    def seek(self, offset: int, whence: int = 0) -> None:
        """Change the file stream position to the given offset relative to the whence.

        offset is the number of bytes (in the data stream) to move from the reference
        position given by whence.

        Parameters
        ----------
        offset : int
            number of bytes to move from the reference position
        whence : int
            0 (SEEK_SET) 1 (SEEK_CUR), by default 0

        Raises
        ------
        ValueError
            if whence is not 0 or 1.
        """
        if self.file_obj.closed:
            msg = f"Cannot read from closed file {self.files[self.ifile_cur]}"
            raise OSError(msg)

        if whence == 0:
            self._seek_set(offset)
        elif whence == 1:
            offset_start = offset + self.cur_data_pos_stream
            self._seek_set(offset_start)
        else:
            msg = "whence should be either 0 (SEEK_SET) or 1 (SEEK_CUR)"
            raise ValueError(msg)

    def _seek2hdr(self, fileid: int) -> None:
        """Go to the header end position of the file with the given fileid."""
        self._open(fileid)
        self.file_obj.seek(self.sinfo.entries[fileid].hdrlen)

    def _seek_set(self, offset: int) -> None:
        if offset < 0 or offset >= self.sinfo.get_combined("datalen"):
            msg = f"offset out of bounds: {offset}"
            raise ValueError(msg)

        fileid = np.where(offset < self.sinfo.cumsum_datalens)[0][0]
        self._seek2hdr(fileid)

        if fileid == 0:
            self.file_obj.seek(offset, os.SEEK_CUR)
        else:
            file_offset = offset - self.sinfo.cumsum_datalens[fileid - 1]
            self.file_obj.seek(file_offset, os.SEEK_CUR)

    def _configure_logger(self, **kwargs) -> None:
        logger_name = "FileReader"
        self.logger = get_logger(logger_name, **kwargs)


class FileWriter(FileBase):
    """A file writer class that can write to a ``sigproc`` format file.

    Parameters
    ----------
    file : str
        file to be written to
    mode : str, optional
        file writing mode, by default "w"
    nbits : int, optional
        number of bits per sample in the file, by default 8
    quantize : bool, optional
        whether to quantize the data, by default False
    tsamp : float, optional
        sampling time, by default None
    nchans : int, optional
        number of channels, by default None
    interval_seconds : float, optional
        sample interval used for quantization in seconds, by default 10
    constant_offset_scale : bool, optional
        whether to use constant offset and scale, by default False
    **digi_kwargs : dict
        keyword arguments for the digitizer: ``digi_mean``, ``digi_scale``, ``digi_min``, ``digi_max``

    Raises
    ------
    ValueError
        if quantize is True and outut ``nbits`` is 32.
    """

    def __init__(
        self,
        file: str,
        tsamp: float,
        nchans: int,
        mode: str = "w",
        nbits: int = 8,
        quantize: bool = False,
        interval_seconds: float = 10,
        constant_offset_scale: bool = False,
        **digi_kwargs,
    ):
        super().__init__([file], mode)
        self.name = file
        self.nbits = nbits
        self.bitsinfo = BitsInfo(nbits)
        self.quantize = quantize

        if self.quantize:
            if self.nbits == 32:
                raise ValueError("Output nbits can not be 32 while quantizing")
            digi = self.bitsinfo.to_dict()
            digi.update(
                (key, value) for key, value in digi_kwargs.items() if key in digi.keys()
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

    def cwrite(self, ar: np.ndarray) -> None:
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

    def write(self, bo: bytes) -> None:
        """Write the given bytes-like object, bo to the file stream.

        Wrapper for :py:obj:`io.RawIOBase.write()`.

        Parameters
        ----------
        bo : bytes
            bytes-like object
        """
        self.file_obj.write(bo)

    def close(self) -> None:
        """Close the currently open file object."""
        self._close_current()


class Transform:
    """A class to transform data to the quantized format.

    Parameters
    ----------
    tsamp : float
        sampling time, by default None
    nchans : int
        number of channels, by default None
    digi_mean : float
        mean of the quantized data
    digi_scale : float
        scale of the quantized data
    digi_min : float
        minimum value of the quantized data
    digi_max : float
        maximum value of the quantized data
    interval_seconds : float, optional
        sample interval used for quantization in seconds, by default 10
    constant_offset_scale : bool, optional
        whether to use constant offset and scale, by default False
    """

    def __init__(
        self,
        tsamp: float,
        nchans: int,
        digi_mean: float,
        digi_scale: float,
        digi_min: float,
        digi_max: float,
        interval_seconds: float = 10,
        constant_offset_scale: bool = False,
    ) -> None:
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
    def interval_samples(self) -> int:
        return round(self.interval_seconds / self.tsamp)

    def rescale(self, data: np.ndarray) -> np.ndarray:
        data = data.reshape(-1, self.nchans)

        if not self.constant_offset_scale or self.first_call:
            self._compute_stats(data)

        normdata = (data + self.offset) * self.scale
        self.first_call = False
        return normdata.ravel()

    def quantize(self, data: np.ndarray) -> np.ndarray:
        ar = (data * self.digi_scale) + self.digi_mean + 0.5
        ar = ar.astype(int)
        return np.clip(ar, self.digi_min, self.digi_max)

    def _compute_stats(self, data: np.ndarray) -> None:
        self.sum_ar += np.sum(data, axis=0)
        self.sumsq_ar += np.sum(data**2, axis=0)
        self.isample += data.shape[0]

        if self.isample >= self.interval_samples or self.first_call:
            mean = self.sum_ar / self.isample
            variance = self.sumsq_ar / self.isample - mean * mean
            self.offset = -mean
            self.scale = np.where(
                np.isclose(variance, 0, atol=1e-5), 1, 1.0 / np.sqrt(variance)
            )
            self._initialize_arr()

    def _initialize_arr(self) -> None:
        self.sum_ar = np.zeros(self.nchans, dtype=float)
        self.sumsq_ar = np.zeros(self.nchans, dtype=float)
        self.isample = 0
