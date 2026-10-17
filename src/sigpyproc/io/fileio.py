"""File I/O base class and readers/writers.

This module contains the FileBase class and the FileReader and FileWriter
classes for reading and writing filterbank files.
"""

from __future__ import annotations

import io
import os
from collections.abc import Buffer
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sigpyproc.io.bits import BitsInfo, pack, unpack

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Self

    from sigpyproc.io.sigproc import StreamInfo


def ensure_output_does_not_alias(
    sources: Iterable[str | Path],
    destinations: Iterable[str | Path],
) -> None:
    """Refuse to open a destination that is an existing input file.

    Comparison uses the resolved path and ``(st_dev, st_ino)``, so a symlink or
    hard link to an input is refused as well. A destination that does not yet
    exist is allowed, including when a generator's header uses that path as its
    filename. An existing file that is not one of the sources may still be
    truncated by the writer.

    Parameters
    ----------
    sources : iterable of str or Path
        Files that must remain unchanged.
    destinations : iterable of str or Path
        Files about to be opened for writing.

    Raises
    ------
    ValueError
        If a destination is the same file as a source.
    """
    protected: list[tuple[Path, tuple[int, int]]] = []
    for source in sources:
        src = Path(source)
        try:
            stat = src.stat()
        except OSError:
            continue
        protected.append((src.resolve(), (stat.st_dev, stat.st_ino)))
    if not protected:
        return
    paths = [item[0] for item in protected]
    inodes = [item[1] for item in protected]
    for destination in destinations:
        dest = Path(destination)
        try:
            resolved = dest.resolve()
        except OSError:
            resolved = dest.absolute()
        inode: tuple[int, int] | None = None
        if dest.exists() or dest.is_symlink():
            try:
                stat = dest.stat()
            except OSError:
                stat = None
            if stat is not None:
                inode = (stat.st_dev, stat.st_ino)
        same_path = resolved in paths
        same_inode = inode is not None and inode in inodes
        if same_path or same_inode:
            msg = (
                f"Refusing to write {dest} because it is the same file as an input. "
                "Streaming transforms do not replace their inputs. Choose a different "
                "output path. spp_clean zap-times --in-place writes a temporary file "
                "and replaces the input only after that write succeeds."
            )
            raise ValueError(msg)


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

    allocated_nbytes = memoryview(buffer).nbytes
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
    sinfo : :py:obj:`~sigpyproc.io.sigproc.StreamInfo`
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
            return unpack(data_ar, self.bitsinfo.nbits, bitorder=self.bitsinfo.bitorder)
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
            unpack(
                read_ar,
                self.bitsinfo.nbits,
                unpack_ar,
                bitorder=self.bitsinfo.bitorder,
            )
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
            cur_pos = self.cur_data_pos_stream
            if cur_pos is None:
                msg = "No file is open for reading"
                raise OSError(msg)
            self._seek_set(offset + cur_pos)
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
    signed : bool, optional
        Clip and store 8-bit samples as int8 in ``[-128, 127]``. Ignored unless
        ``nbits`` is 8 and ``rescale`` is False. By default False, which clips
        integer samples to the unsigned range.
    rescale : bool, optional
        whether to rescale the data using the nbit-dependent values,
        by default False. Rescaling writes unsigned levels and ignores ``signed``.

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
        signed: bool = False,
        rescale: bool = False,
    ) -> None:
        super().__init__([file], mode)
        self.bitsinfo = BitsInfo(nbits)
        self.signed = bool(signed) and nbits == 8 and not rescale
        self.rescale = rescale

    def cwrite(self, arr: np.ndarray) -> None:
        """Write an array to file.

        Parameters
        ----------
        ar : :py:obj:`~numpy.ndarray`
            a 1-D numpy array containing the data

        Notes
        -----
        Input data will be packed with a bitsize determined by the nbits.
        Signed 8-bit output is clipped to ``[-128, 127]`` and written as int8.
        Other integer widths are clipped to the unsigned range. 32-bit output is
        float32 and is not clipped. ``rescale=True`` quantises to unsigned levels.
        """
        arr = np.asarray(arr)
        if self.bitsinfo.nbits < 32 and self.rescale:
            # arr should be normalized first
            arr = self.bitsinfo.quantize(arr)
        elif self.signed:
            arr = np.clip(np.rint(arr), -128, 127).astype(np.int8)
        elif self.bitsinfo.nbits < 32:
            arr = np.clip(np.rint(arr), self.bitsinfo.digi_min, self.bitsinfo.digi_max)
            arr = arr.astype(self.bitsinfo.dtype, copy=False)
        else:
            arr = np.asarray(arr, dtype=self.bitsinfo.dtype)
        if self.bitsinfo.unpack:
            packed = pack(
                np.ascontiguousarray(arr, dtype=np.uint8),
                self.bitsinfo.nbits,
                bitorder=self.bitsinfo.bitorder,
            )
            packed.tofile(self.file_obj)
        else:
            arr.tofile(self.file_obj)

    def write(self, bo: bytes) -> None:
        """Write the given bytes-like object, bo to the file stream.

        Wrapper for :py:obj:`io.RawIOBase.write`.

        Parameters
        ----------
        bo : bytes
            bytes-like object
        """
        self.file_obj.write(bo)
