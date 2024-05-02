from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Buffer

from sigpyproc.io.bits import BitsInfo, pack, unpack

if TYPE_CHECKING:
    from collections.abc import Callable

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
        A buffer object of the given size. collections.abc.Buffer (PEP 688).

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
        msg = f"Requested buffer size ({nbytes}) is invalid, should be > 0"
        raise ValueError(msg)
    try:
        buffer = allocator(nbytes)
    except Exception as exc:
        msg = f"Failed to allocate buffer of size {nbytes}"
        raise RuntimeError(msg) from exc

    if not isinstance(buffer, Buffer):
        msg = f"Allocator did not return a buffer object, got {type(buffer)}"
        raise TypeError(msg)

    allocated_nbytes = len(memoryview(buffer))
    if allocated_nbytes != nbytes:
        msg = (
            f"Allocated buffer is not the expected size: "
            f"{allocated_nbytes} (actual) != {nbytes} (expected)"
        )
        raise ValueError(msg)
    return buffer


class FileBase:
    """File I/O base class.

    Parameters
    ----------
    files : list[str]
        list of file names to open
    mode : str
        file opening mode
    """

    def __init__(self, files: list[str], mode: str) -> None:
        self.files = files
        self.mode = mode
        self.opener = io.FileIO
        self.ifile_cur: int | None = None
        self._open(ifile=0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self._close_current()

    @property
    def file_cur(self) -> str | None:
        """str: Name of the currently open file."""
        if self.ifile_cur is None:
            return None
        return self.files[self.ifile_cur]

    def close(self) -> None:
        """Close the stream."""
        self._close_current()

    def eos(self) -> bool:
        """Check if the end of the file stream has been reached."""
        # First check if we are at the end of the current file
        eof = self.file_obj.tell() == os.fstat(self.file_obj.fileno()).st_size
        # Now check if we are at the end of the list of files
        eol = self.ifile_cur == len(self.files) - 1
        return eof & eol

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

    def _close_current(self) -> None:
        """Close the currently open file object."""
        if self.ifile_cur is not None:
            self.file_obj.close()
            self.ifile_cur = None


class FileReader(FileBase):
    """A file reader class that can read from multiple files.

    Files should have format similar to ``sigproc``.

    Parameters
    ----------
    sinfo : StreamInfo
        stream information object containing header and data lengths
    mode : str, optional
        file opening mode, by default "r"
    nbits : int, optional
        number of bits per sample in the files, by default 8
    """

    def __init__(
        self,
        sinfo: StreamInfo,
        mode: str = "r",
        nbits: int = 8,
    ) -> None:
        self.sinfo = sinfo
        self.bitsinfo = BitsInfo(nbits)

        filenames = self.sinfo.get_info_list("filename")
        super().__init__(filenames, mode)

    @property
    def cur_data_pos_file(self) -> int | None:
        """int: Current data position in the current file."""
        if self.ifile_cur is None:
            return None
        return self.file_obj.tell() - self.sinfo.entries[self.ifile_cur].hdrlen

    @property
    def cur_data_pos_stream(self) -> int | None:
        """int: Current data position in the data stream."""
        if self.ifile_cur is None:
            return None
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
        OSError
            if no file is open for reading
        """
        if self.ifile_cur is None:
            msg = "No file is open for reading"
            raise OSError(msg)

        count = nunits // self.bitsinfo.bitfact
        data = []
        while count >= 0:
            count_read = min(self.sinfo.entries[self.ifile_cur].datalen, count)
            data_read = np.fromfile(
                self.file_obj,
                count=count_read,
                dtype=self.bitsinfo.dtype,
            )
            count -= len(data_read)
            data.append(data_read)

            if count == 0:
                break
            self._seek2hdr(self.ifile_cur + 1)
        data_ar = np.concatenate(data)
        if self.bitsinfo.unpack:
            return unpack(data_ar, self.bitsinfo.nbits)
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
        OSError
            if no file is open for reading

        Notes
        -----
        It is the responsibility of the caller to handle the case than fewer bytes
        than requested are read into the buffer. When at the end of the file stream
        the number of bytes returned will be zero.
        """
        if self.ifile_cur is None:
            msg = "No file is open for reading"
            raise OSError(msg)

        nbytes = 0
        read_buffer_view = memoryview(read_buffer)
        while True:
            nbytes_read = self.file_obj.readinto(read_buffer_view[nbytes:])
            if nbytes_read is None:
                # Might be non-blocking IO, so maybe try again
                msg = "file might in non-blocking mode"
                raise BlockingIOError(msg)
            nbytes += nbytes_read
            if nbytes == len(read_buffer_view) or self.eos():
                # We have either filled the buffer or reached the end of the stream
                break
            self._seek2hdr(self.ifile_cur + 1)

        if self.bitsinfo.unpack and unpack_buffer is not None:
            read_ar = np.frombuffer(read_buffer_view, dtype=np.uint8)
            unpack_ar = np.frombuffer(memoryview(unpack_buffer), dtype=np.uint8)
            unpack(read_ar, self.bitsinfo.nbits, unpack_ar)
        elif self.bitsinfo.unpack:
            msg = "unpack_buffer should be provided when unpacking"
            raise ValueError(msg)
        return nbytes

    def seek(self, offset: int, whence: int = 0) -> None:
        """Change the file stream position to the given offset relative to the whence.

        Parameters
        ----------
        offset : int
            number of bytes (in the data stream) to move from the reference position
            given by whence
        whence : int
            0 (SEEK_SET) 1 (SEEK_CUR), by default 0

        Raises
        ------
        ValueError
            if whence is not 0 or 1.
        """
        if self.ifile_cur is None:
            msg = "No file is open for reading"
            raise OSError(msg)

        if whence == 0:
            self._seek_set(offset)
        elif whence == 1:
            self._seek_set(offset + self.cur_data_pos_stream)  # type: ignore [operator]
        else:
            msg = "whence should be either 0 (SEEK_SET) or 1 (SEEK_CUR)"
            raise ValueError(msg)

    def _seek2hdr(self, ifile: int) -> None:
        """Go to the header end position of the given file index."""
        self._open(ifile)
        self.file_obj.seek(self.sinfo.entries[ifile].hdrlen)

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


class FileWriter(FileBase):
    """A file writer class that can write to a ``sigproc`` format file.

    Parameters
    ----------
    file : str
        file name to write to
    mode : str, optional
        file writing mode, by default "w"
    nbits : int, optional
        number of bits per sample in the file, by default 8
    scale_fac : float, optional
        Additional scale factor to apply to data, by default 1.0
    rescale : bool, optional
        whether to rescale the data using the nbit-dependent values,
        by default True

    Raises
    ------
    ValueError
        if quantize is True and outut ``nbits`` is 32.
    """

    def __init__(
        self,
        file: str,
        *,
        mode: str = "w",
        nbits: int = 8,
        rescale: bool = False,
    ) -> None:
        super().__init__([file], mode)
        self.bitsinfo = BitsInfo(nbits)
        self.rescale = rescale

    def cwrite(self, arr: np.ndarray) -> None:
        """Write an array to file.

        Parameters
        ----------
        ar : :py:obj:`~numpy.ndarray`
            a 1-D numpy array containing the data

        Notes
        -----
        Input data will be packed with a bitsize determined by the nbits
        """
        if self.bitsinfo.nbits < 32 and self.rescale:
            # arr should be normalized first
            arr = self.bitsinfo.quantize(arr)
        if self.bitsinfo.unpack:
            packed = pack(arr, self.bitsinfo.nbits)
            packed.tofile(self.file_obj)
        else:
            arr.tofile(self.file_obj)

    def write(self, bo: bytes) -> None:
        """Write the given bytes-like object, bo to the file stream.

        Wrapper for :py:obj:`io.RawIOBase.write()`.

        Parameters
        ----------
        bo : bytes
            bytes-like object
        """
        self.file_obj.write(bo)
