import pytest
import numpy as np
from sigpyproc.readers import FilReader, PFITSReader, PulseExtractor
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
            
    def test_read_plan(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        for nsamps, ii, data in fil.read_plan(gulp=512, nsamps=1024):
            assert isinstance(data, np.ndarray)
            
    def test_read_plan(self, filterbank_files):
        for filfile in filterbank_files:
            fil = FilReader(filfile)
            for nsamps, ii, data in fil.read_plan(gulp=512, nsamps=1024):
                assert isinstance(data, np.ndarray)
    
    def test_read_plan_buffered(self, filterbank_files):
        for filfile in filterbank_files:
            fil = FilReader(filfile)
            for nsamps, ii, data in fil.read_plan_buffered(gulp=512, nsamps=1024):
                assert isinstance(data, np.ndarray)
                
    def test_compare_buffered_and_nonbuffered(self, filterbank_files):
        def allocator(nbytes):
            buffer = np.zeros(nbytes, dtype="ubyte")
            return memoryview(buffer)
        
        for filfile in filterbank_files:
            fil0 = FilReader(filfile)
            plan0 = fil0.read_plan(quiet=True)            
            fil1 = FilReader(filfile)
            plan1 = fil1.read_plan_buffered(quiet=True, allocator=allocator)
            for (nsamps0, ii0, data0), (nsamps1, ii1, data1) in zip(plan0, plan1):
                assert nsamps0 == nsamps1   
                assert ii0 == ii1
                assert np.array_equal(data0, data1)
                
    def test_nsamps_eq_skipback(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        with pytest.raises(ValueError):
            for nsamps, ii, data in fil.read_plan_buffered(gulp=512, nsamps=1024, skipback=1024):
                pass

    def test_read_plan_custom_allocator(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        def allocator(nbytes):
            buffer = np.zeros(nbytes, dtype="ubyte")
            return memoryview(buffer)
        for nsamps, ii, data in fil.read_plan_buffered(allocator=allocator):
            pass
            
    def test_read_plan_invalid_custom_allocator(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        def allocator():
            pass
        with pytest.raises(TypeError):
            for nsamps, ii, data in fil.read_plan_buffered(allocator=allocator):
                pass


class TestPFITSReader(object):
    def test_pfits_single(self, fitsfile_4bit):
        fits = PFITSReader(fitsfile_4bit)
        np.testing.assert_equal(fits.header.nchans, 416)
        np.testing.assert_equal(fits.header.nbits, 4)
        np.testing.assert_equal(fits.header.nifs, 1)

    def test_pfits_empty(self):
        with np.testing.assert_raises(TypeError):
            PFITSReader()

    def test_pfits_otherfile(self, inffile):
        with np.testing.assert_raises(OSError):
            PFITSReader(inffile)

    def test_read_block(self, fitsfile_4bit):
        fits = PFITSReader(fitsfile_4bit)
        block = fits.read_block(0, 100)
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block.header, Header)
        np.testing.assert_equal(block.dm, 0)

    def test_read_block_outofrange(self, fitsfile_4bit):
        fits = PFITSReader(fitsfile_4bit)
        with np.testing.assert_raises(ValueError):
            fits.read_block(-10, 100)
        with np.testing.assert_raises(ValueError):
            fits.read_block(100, fits.header.nsamples + 1)

    def test_read_plan(self, fitsfile_4bit):
        fits = PFITSReader(fitsfile_4bit)
        for nsamps, ii, data in fits.read_plan(gulp=512, nsamps=1024):
            assert isinstance(data, np.ndarray)

class TestPulseExtractor(object):
    def test_filterbank_single(self, filfile_4bit):
        pulse = PulseExtractor(filfile_4bit, 1000, 50, 0)
        block = pulse.get_data()
        assert isinstance(block, FilterbankBlock)

    def test_filterbank_dm(self, filfile_4bit):
        pulse = PulseExtractor(filfile_4bit, 1000, 50, 10)
        block = pulse.get_data()
        assert isinstance(block, FilterbankBlock)