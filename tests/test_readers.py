import numpy as np
import pytest

from sigpyproc.block import FilterbankBlock
from sigpyproc.header import Header
from sigpyproc.readers import FilReader, PFITSReader, PulseExtractor


class TestFilReader:
    def test_filterbank_single(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        assert fil.header.nchans == 832
        assert fil.filename == filfile_4bit

    def test_filterbank_otherfile(self, inffile: str) -> None:
        with np.testing.assert_raises(OSError):
            FilReader(inffile)

    def test_read_block(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(0, 100)
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block.header, Header)
        np.testing.assert_equal(block.dm, 0)
        np.testing.assert_equal(block.data.shape, (fil.header.nchans, 100))
        nchans = fil.header.nchans // 2
        block = fil.read_block(0, 100, nchans=nchans)
        np.testing.assert_equal(block.data.shape, (nchans, 100))

    def test_read_block_outofrange(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        with np.testing.assert_raises(ValueError):
            fil.read_block(-10, 100)
        with np.testing.assert_raises(ValueError):
            fil.read_block(100, fil.header.nsamples + 1)
        with np.testing.assert_raises(ValueError):
            fil.read_block(0, 100, nchans=1000)
        with np.testing.assert_raises(ValueError):
            fil.read_block(0, 100, fch1=10000)

    def test_read_dedisp_block(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        dm = 0
        block = fil.read_block(0, fil.header.nsamples)
        block_dd = fil.read_dedisp_block(0, fil.header.nsamples, dm)
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block.header, Header)
        np.testing.assert_equal(block.data, block_dd.data)

    def test_read_dedisp_block_outofrange(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        dm = 10
        with np.testing.assert_raises(ValueError):
            fil.read_dedisp_block(-10, 100, dm)
        with np.testing.assert_raises(ValueError):
            fil.read_dedisp_block(100, fil.header.nsamples + 1, dm)

    def test_read_dedisp_block_outofrange_dm(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        with np.testing.assert_raises(ValueError):
            fil.read_dedisp_block(0, 100, 10000)

    def test_read_plan(self, filfiles: list[str]) -> None:
        for filfile in filfiles:
            fil = FilReader(filfile)
            for _, _, data in fil.read_plan(gulp=512, nsamps=1024):
                assert isinstance(data, np.ndarray)

    def test_read_plan_checks(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        for _, _, data in fil.read_plan(gulp=512):
            assert isinstance(data, np.ndarray)
            assert data.size == 512 * fil.header.nchans
        for _, _, data in fil.read_plan(gulp=400, skipback=200):
            assert isinstance(data, np.ndarray)

    def test_read_plan_corrupted_file(self, filfile_8bit_1: str, tmpfile: str) -> None:
        fil = FilReader(filfile_8bit_1)
        corrupted_file = fil.header.prep_outfile(tmpfile)
        for _, _, data in fil.read_plan(gulp=fil.header.nsamples):
            corrupted_file.cwrite(data[: fil.header.nsamples * fil.header.nchans - 100])
        corrupted_file.close()
        corruted_fil = FilReader(tmpfile)
        with pytest.raises(ValueError):  # noqa: PT012
            for _, _, _ in corruted_fil.read_plan(gulp=512):
                pass

    def test_nsamps_eq_skipback(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        with pytest.raises(ValueError):  # noqa: PT012
            for _, _, _ in fil.read_plan(gulp=512, nsamps=1024, skipback=1024):
                pass


class TestPFITSReader:
    def test_pfits_single(self, fitsfile_4bit: str) -> None:
        fits = PFITSReader(fitsfile_4bit)
        np.testing.assert_equal(fits.header.nchans, 416)
        np.testing.assert_equal(fits.header.nbits, 4)
        np.testing.assert_equal(fits.header.nifs, 1)
        np.testing.assert_equal(fits.filename, fitsfile_4bit)
        np.testing.assert_equal(fits.pri_hdr.telescope, "Parkes")
        np.testing.assert_equal(fits.bitsinfo.nbits, 4)

    def test_pfits_otherfile(self, inffile: str) -> None:
        with np.testing.assert_raises(OSError):
            PFITSReader(inffile)

    def test_read_block(self, fitsfile_4bit: str) -> None:
        fits = PFITSReader(fitsfile_4bit)
        block = fits.read_block(0, 100)
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block.header, Header)
        np.testing.assert_equal(block.dm, 0)
        np.testing.assert_equal(block.data.shape, (fits.header.nchans, 100))
        nchans = fits.header.nchans // 2
        block = fits.read_block(0, 100, nchans=nchans)
        np.testing.assert_equal(block.data.shape, (nchans, 100))

    def test_read_block_outofrange(self, fitsfile_4bit: str) -> None:
        fits = PFITSReader(fitsfile_4bit)
        with np.testing.assert_raises(ValueError):
            fits.read_block(-10, 100)
        with np.testing.assert_raises(ValueError):
            fits.read_block(100, fits.header.nsamples + 1)
        with np.testing.assert_raises(ValueError):
            fits.read_block(0, 100, nchans=1000)
        with np.testing.assert_raises(ValueError):
            fits.read_block(0, 100, fch1=10000)

    def test_read_plan(self, fitsfile_4bit: str) -> None:
        fits = PFITSReader(fitsfile_4bit)
        for _, _, data in fits.read_plan(gulp=512, nsamps=1024):
            assert isinstance(data, np.ndarray)
        with pytest.raises(ValueError): # noqa: PT012
            for _, _, _ in fits.read_plan(gulp=512, skipback=1024):
                pass


class TestPulseExtractor:
    def test_init(self, filfile_8bit_1: str) -> None:
        pulse = PulseExtractor(filfile_8bit_1, 1000, 50, 0, toa_freq=2366.0, nchans=416)
        np.testing.assert_equal(pulse.filfile, filfile_8bit_1)
        np.testing.assert_equal(pulse.pulse_toa, 1000)
        np.testing.assert_equal(pulse.pulse_width, 50)
        np.testing.assert_equal(pulse.pulse_dm, 0)
        np.testing.assert_equal(pulse.toa_freq, 2366.0)
        np.testing.assert_equal(pulse.nchans, 416)
        np.testing.assert_equal(pulse.t_decimate, 25)
        np.testing.assert_equal(pulse.pulse_toa_block, pulse.nsamps // 2)

    def test_extract(self, filfile_8bit_1: str) -> None:
        pulse = PulseExtractor(filfile_8bit_1, 1000, 50, 0, toa_freq=2366.0, nchans=416)
        block = pulse.get_data()
        assert isinstance(block, FilterbankBlock)
        np.testing.assert_equal(block.header.nchans, 416)
        block = pulse.get_data(pad_mode="mean")
        assert isinstance(block, FilterbankBlock)
        np.testing.assert_equal(block.header.nchans, 416)
        with pytest.raises(ValueError):
            pulse.get_data(pad_mode="unknown")

    def test_filterbank_dm(self, filfile_4bit: str) -> None:
        pulse = PulseExtractor(filfile_4bit, 1000, 50, 10)
        block = pulse.get_data()
        assert isinstance(block, FilterbankBlock)
        np.testing.assert_equal(block.header.nchans, 832)
