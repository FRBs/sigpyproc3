import pytest
from sigpyproc.readers import FilReader
from sigpyproc.header import Header
from sigpyproc.block import FilterbankBlock


class TestFilReader(object):
    def test_filterbank_single(self, filfile_4bit):
        fil = FilReader(filfile_4bit)
        assert fil.header.nchans == 832

    def test_filterbank_empty(self):
        with pytest.raises(TypeError):
            FilReader()

    def test_filterbank_otherfile(self, inffile):
        with pytest.raises(OSError):
            FilReader(inffile)

    def test_read_block(self, filfile_8bit):
        fil = FilReader(filfile_8bit)
        block = fil.read_block(0, 100)
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block.header, Header)
        assert block.dm == 0

    def test_read_block_outofrange(self, filfile_8bit):
        fil = FilReader(filfile_8bit)
        with pytest.raises(ValueError):
            fil.read_block(-10, 100)
        with pytest.raises(ValueError):
            fil.read_block(100, fil.header.nsamples + 1)

    def test_read_dedisp_block(self, filfile_8bit):
        fil = FilReader(filfile_8bit)
        dm = 10
        block = fil.read_dedisp_block(0, 100, dm)
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block.header, Header)
        assert block.dm == dm

    def test_read_dedisp_block_outofrange(self, filfile_8bit):
        fil = FilReader(filfile_8bit)
        dm = 10
        with pytest.raises(ValueError):
            fil.read_dedisp_block(-10, 100, dm)
        with pytest.raises(ValueError):
            fil.read_dedisp_block(100, fil.header.nsamples + 1, dm)

    def test_read_dedisp_block_outofrange_dm(self, filfile_8bit):
        fil = FilReader(filfile_8bit)
        with pytest.raises(ValueError):
            fil.read_dedisp_block(0, 100, -10)
        with pytest.raises(ValueError):
            fil.read_dedisp_block(0, 100, 10000)
