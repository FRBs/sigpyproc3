import pytest
import os
import numpy as np
from typing import cast
from typing_extensions import Buffer

from sigpyproc.io import fileio


class TestBuffer(object):
    def test_allocate_buffer_success(self):
        def allocator(nbytes: int) -> Buffer:
            return bytearray(nbytes)

        nbytes = 100
        buffer = fileio.allocate_buffer(allocator, nbytes)
        assert isinstance(buffer, bytearray)
        assert len(buffer) == nbytes

    def test_allocate_buffer_custom(self):
        def custom_allocator(nbytes: int) -> Buffer:
            buffer = np.zeros(nbytes, dtype="ubyte")
            return memoryview(cast(Buffer, buffer))

        nbytes = 100
        buffer = fileio.allocate_buffer(custom_allocator, nbytes)
        assert isinstance(buffer, memoryview)
        assert len(buffer) == nbytes

    def test_allocate_buffer_invalid_size(self):
        def allocator(nbytes: int) -> Buffer:
            return bytearray(nbytes)

        nbytes = -100
        with pytest.raises(ValueError):
            fileio.allocate_buffer(allocator, nbytes)

    def test_allocate_buffer_invalid_allocator(self):
        def allocator(nbytes: int):
            raise MemoryError("Out of memory")

        nbytes = 100
        with pytest.raises(RuntimeError):
            fileio.allocate_buffer(allocator, nbytes)

    def test_allocate_buffer_invalid_return(self):
        def allocator(nbytes: int) -> str:
            return str(nbytes)

        nbytes = 100
        with pytest.raises(TypeError):
            fileio.allocate_buffer(allocator, nbytes)

    def test_allocate_buffer_size_mismatch(self):
        def allocator(nbytes: int) -> Buffer:
            return bytearray(nbytes + 1)

        nbytes = 100
        with pytest.raises(ValueError):
            fileio.allocate_buffer(allocator, nbytes)


class TestFileBase(object):
    @pytest.fixture(autouse=True)
    def setup(self, filfile_1bit, filfile_2bit, filfile_4bit):
        self.files = [filfile_1bit, filfile_2bit, filfile_4bit]
        self.mode = "r"
        self.file_base = fileio.FileBase(self.files, self.mode)
        yield
        self.file_base._close_current()

    def test_enter_exit(self):
        with fileio.FileBase(self.files, self.mode) as file_base:
            assert file_base.ifile_cur == 0
            assert file_base.file_obj.closed is False
        assert file_base.file_obj.closed is True

    def test_open_valid_file(self):
        self.file_base._open(1)
        assert self.file_base.ifile_cur == 1
        assert self.file_base.file_obj.closed is False

    def test_open_invalid_file(self):
        with pytest.raises(ValueError):
            self.file_base._open(-1)
        with pytest.raises(ValueError):
            self.file_base._open(len(self.files))

    def test_eos_success(self):
        self.file_base._open(2)
        self.file_base.file_obj.seek(0, os.SEEK_END)
        assert self.file_base.eos() is True

    def test_eos_failure(self):
        self.file_base._open(0)
        self.file_base.file_obj.seek(0)
        assert self.file_base.eos() is False

    def test_close_current(self):
        self.file_base._close_current()
        assert self.file_base.file_obj.closed is True
