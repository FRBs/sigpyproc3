import os
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from pytest_mock import MockerFixture
from typing_extensions import Buffer

from sigpyproc.io import fileio, sigproc


class TestBufferAllocator:
    def test_allocate_buffer_success(self) -> None:
        def allocator(nbytes: int) -> Buffer:
            return bytearray(nbytes)

        nbytes = 100
        buffer = fileio.allocate_buffer(allocator, nbytes)
        assert isinstance(buffer, bytearray)
        assert len(buffer) == nbytes

    def test_allocate_buffer_custom(self) -> None:
        def custom_allocator(nbytes: int) -> Buffer:
            buffer = np.zeros(nbytes, dtype="ubyte")
            return memoryview(cast("Buffer", buffer))

        nbytes = 100
        buffer = fileio.allocate_buffer(custom_allocator, nbytes)
        assert isinstance(buffer, memoryview)
        assert len(buffer) == nbytes

    def test_allocate_buffer_invalid_size(self) -> None:
        def allocator(nbytes: int) -> Buffer:
            return bytearray(nbytes)

        nbytes = -100
        with pytest.raises(ValueError):
            fileio.allocate_buffer(allocator, nbytes)

    def test_allocate_buffer_invalid_allocator(self) -> None:
        def allocator(nbytes: int) -> Buffer:
            msg = f"Out of memory: {nbytes} bytes requested"
            raise MemoryError(msg)

        nbytes = 100
        with pytest.raises(RuntimeError):
            fileio.allocate_buffer(allocator, nbytes)

    def test_allocate_buffer_invalid_return(self) -> None:
        def allocator(nbytes: int) -> str:
            return str(nbytes)

        nbytes = 100
        with pytest.raises(TypeError):
            fileio.allocate_buffer(allocator, nbytes)  # type: ignore [arg-type]

    def test_allocate_buffer_size_mismatch(self) -> None:
        def allocator(nbytes: int) -> Buffer:
            return bytearray(nbytes + 1)

        nbytes = 100
        with pytest.raises(ValueError):
            fileio.allocate_buffer(allocator, nbytes)


class TestFileBase:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        filfile_1bit: str,
        filfile_2bit: str,
        filfile_4bit: str,
    ) -> Generator[None, Any, None]:
        self.files = [filfile_1bit, filfile_2bit, filfile_4bit]
        self.mode = "r"
        self.file_base = fileio.FileBase(self.files, self.mode)
        yield
        self.file_base.close()

    def test_enter_exit(self) -> None:
        with fileio.FileBase(self.files, self.mode) as file_base:
            assert file_base.ifile_cur == 0
            assert file_base.file_obj.closed is False
        assert file_base.file_obj.closed is True

    def test_open_valid_file(self) -> None:
        self.file_base._open(1)
        assert self.file_base.ifile_cur == 1
        assert self.file_base.file_obj.closed is False
        assert self.file_base.file_cur == self.files[1]

    def test_open_invalid_file(self) -> None:
        with pytest.raises(ValueError):
            self.file_base._open(-1)
        with pytest.raises(ValueError):
            self.file_base._open(len(self.files))

    def test_eos_success(self) -> None:
        self.file_base._open(2)
        self.file_base.file_obj.seek(0, os.SEEK_END)
        assert self.file_base.eos() is True

    def test_eos_failure(self) -> None:
        self.file_base._open(0)
        self.file_base.file_obj.seek(0)
        assert self.file_base.eos() is False

    def test_close_current(self) -> None:
        self.file_base._open(0)
        self.file_base.close()
        assert self.file_base.file_obj.closed is True
        assert self.file_base.ifile_cur is None
        assert self.file_base.file_cur is None


class TestFileReader:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        filfile_8bit_1: str,
        filfile_8bit_2: str,
    ) -> Generator[None, Any, None]:
        self.files = [filfile_8bit_1, filfile_8bit_2]
        self.mode = "r"
        hdr = sigproc.parse_header_multi([Path(f) for f in self.files])
        self.sinfo: sigproc.StreamInfo = hdr["stream_info"]
        self.file_reader = fileio.FileReader(self.sinfo, self.mode, hdr["nbits"])
        yield
        self.file_reader.close()

    def test_initalize(self) -> None:
        self.file_reader._seek2hdr(0)
        assert self.file_reader.ifile_cur == 0
        assert self.file_reader.file_obj.closed is False
        assert self.file_reader.files == self.files
        assert self.file_reader.cur_data_pos_file == 0
        assert self.file_reader.cur_data_pos_stream == 0

    def test_seek_invalid_whence(self) -> None:
        with pytest.raises(ValueError):
            self.file_reader.seek(0, whence=2)

    def test_seek_no_file_open(self) -> None:
        self.file_reader.close()
        assert self.file_reader.file_obj.closed is True
        assert self.file_reader.ifile_cur is None
        assert self.file_reader.cur_data_pos_file is None
        assert self.file_reader.cur_data_pos_stream is None
        with pytest.raises(OSError):
            self.file_reader.seek(0)

    def test_seek_set(self) -> None:
        self.file_reader._seek2hdr(0)
        datalens = self.sinfo.get_info_list("datalen")
        offset = datalens[0] + 1
        self.file_reader.seek(offset, whence=0)
        assert self.file_reader.ifile_cur == 1
        assert self.file_reader.cur_data_pos_file == 1
        assert self.file_reader.cur_data_pos_stream == offset

    def test_seek_cur(self) -> None:
        self.file_reader._seek2hdr(0)
        datalens = self.sinfo.get_info_list("datalen")
        offset = datalens[0] + datalens[1] // 2
        self.file_reader.seek(offset, whence=0)
        self.file_reader.seek(-offset, whence=1)
        assert self.file_reader.ifile_cur == 0
        assert self.file_reader.cur_data_pos_file == 0
        assert self.file_reader.cur_data_pos_stream == 0

    def test_seek_set_out_of_bounds(self) -> None:
        self.file_reader._seek2hdr(0)
        with pytest.raises(ValueError):
            self.file_reader.seek(-1, whence=0)
        datalen_stream = self.sinfo.get_combined("datalen")
        with pytest.raises(ValueError):
            self.file_reader.seek(datalen_stream + 1, whence=0)

    def test_cread(self) -> None:
        self.file_reader._seek2hdr(0)
        datalens = self.sinfo.get_info_list("datalen")
        nunits = 100
        data = self.file_reader.cread(nunits)
        assert len(data) == nunits
        assert data.dtype == self.file_reader.bitsinfo.dtype
        nunits_multi = datalens[0] + datalens[1] // 2
        data = self.file_reader.cread(nunits_multi)
        assert len(data) == nunits_multi
        assert data.dtype == self.file_reader.bitsinfo.dtype
        self.file_reader.close()
        with pytest.raises(OSError):
            self.file_reader.cread(nunits)

    def test_creadinto(self) -> None:
        self.file_reader._seek2hdr(0)
        datalens = self.sinfo.get_info_list("datalen")
        nunits = 100
        read_buffer = fileio.allocate_buffer(bytearray, nunits)
        nread = self.file_reader.creadinto(read_buffer)
        assert nread == nunits
        nunits_multi = datalens[0] + datalens[1] // 2
        read_buffer = fileio.allocate_buffer(bytearray, nunits_multi)
        nread = self.file_reader.creadinto(read_buffer)
        assert nread == nunits_multi
        self.file_reader.close()
        with pytest.raises(OSError):
            self.file_reader.creadinto(read_buffer)

    def test_creadinto_blocking_io(self, mocker: MockerFixture) -> None:
        mock_file_obj = mocker.patch.object(self.file_reader, "file_obj", autospec=True)
        mock_file_obj.readinto.return_value = None
        nunits = 100
        read_buffer = fileio.allocate_buffer(bytearray, nunits)
        with pytest.raises(BlockingIOError) as exc_info:
            self.file_reader.creadinto(read_buffer)

        assert str(exc_info.value) == "file might in non-blocking mode"
        mock_file_obj.readinto.assert_called_once_with(memoryview(read_buffer))


class TestFileReaderUnpack:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        filfile_2bit: str,
    ) -> Generator[None, Any, None]:
        self.files = [filfile_2bit]
        self.mode = "r"
        hdr = sigproc.parse_header_multi(self.files)
        self.sinfo: sigproc.StreamInfo = hdr["stream_info"]
        self.file_reader = fileio.FileReader(self.sinfo, self.mode, hdr["nbits"])
        yield
        self.file_reader.close()

    def test_cread(self) -> None:
        self.file_reader._seek2hdr(0)
        nunits = 100
        data = self.file_reader.cread(nunits)
        assert len(data) == nunits
        assert data.dtype == self.file_reader.bitsinfo.dtype

    def test_creadinto(self) -> None:
        self.file_reader._seek2hdr(0)
        nunits = 100
        nbytes = nunits // 4
        read_buffer = fileio.allocate_buffer(bytearray, nbytes)
        unpack_buffer = fileio.allocate_buffer(bytearray, nunits)
        nread = self.file_reader.creadinto(read_buffer, unpack_buffer)
        assert nread == nbytes

    def test_creadinto_no_unpack(self) -> None:
        self.file_reader._seek2hdr(0)
        nunits = 100
        read_buffer = fileio.allocate_buffer(bytearray, nunits)
        with pytest.raises(ValueError):
            self.file_reader.creadinto(read_buffer)


class TestFileWriter:
    def test_write(self, tmpfile: str) -> None:
        with fileio.FileWriter(tmpfile, mode="w") as file_writer:
            bo = bytearray(b"Hello, World!")
            file_writer.write(bo)
        with Path(tmpfile).open("rb") as fp:
            assert fp.read() == bo

    def test_cwrite(self, tmpfile: str) -> None:
        with fileio.FileWriter(tmpfile, mode="w", nbits=8) as file_writer:
            arr = np.arange(100, dtype=np.uint8)
            file_writer.cwrite(arr)
        data = np.fromfile(tmpfile, dtype=np.uint8)
        np.testing.assert_array_equal(data, arr)

    def test_cwrite_pack(self, tmpfile: str) -> None:
        nbits = 1
        with fileio.FileWriter(tmpfile, mode="w", nbits=nbits) as file_writer:
            rng = np.random.default_rng()
            arr = rng.integers((1 << nbits) - 1, size=1024, dtype=np.uint8)
            file_writer.cwrite(arr)
        data = np.fromfile(tmpfile, dtype=np.uint8)
        np.testing.assert_array_equal(data, np.packbits(arr))
