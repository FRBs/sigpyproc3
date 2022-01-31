import numpy as np
from sigpyproc.readers import FilReader
from sigpyproc.header import Header
from sigpyproc.block import FilterbankBlock


class TestFilReader(object):
    def test_filterbank_single(self, filfile_4bit):
        fil = FilReader(filfile_4bit)
        assert fil.header.nchans == 832

    def test_filterbank_empty(self):
        with np.testing.assert_raises(TypeError):
            FilReader()

    def test_filterbank_otherfile(self, inffile):
        with np.testing.assert_raises(OSError):
            FilReader(inffile)

    def test_read_block(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(0, 100)
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block.header, Header)
        np.testing.assert_equal(block.dm, 0)

    def test_read_block_outofrange(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        with np.testing.assert_raises(ValueError):
            fil.read_block(-10, 100)
        with np.testing.assert_raises(ValueError):
            fil.read_block(100, fil.header.nsamples + 1)

    def test_read_dedisp_block(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        dm = 10
        block = fil.read_dedisp_block(0, 100, dm)
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block.header, Header)
        np.testing.assert_equal(block.dm, dm)

    def test_read_dedisp_block_outofrange(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        dm = 10
        with np.testing.assert_raises(ValueError):
            fil.read_dedisp_block(-10, 100, dm)
        with np.testing.assert_raises(ValueError):
            fil.read_dedisp_block(100, fil.header.nsamples + 1, dm)

    def test_read_dedisp_block_outofrange_dm(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        with np.testing.assert_raises(ValueError):
            fil.read_dedisp_block(0, 100, 10000)
