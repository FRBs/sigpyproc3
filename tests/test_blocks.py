import pytest
import numpy as np
from pathlib import Path

from sigpyproc.header import Header
from sigpyproc.readers import FilReader
from sigpyproc.block import FilterbankBlock
from sigpyproc.timeseries import TimeSeries


class TestFilterbankBlock(object):
    def test_block_noheader(self):
        data = np.random.normal(size=(128, 1024))
        with pytest.raises(TypeError):
            FilterbankBlock(data)

    def test_block_header(self, filfile_8bit_1):
        data = np.random.normal(size=(128, 1024))
        header = Header.from_sigproc(filfile_8bit_1)
        block = FilterbankBlock(data, header)
        assert isinstance(block.header, Header)
        assert block.dm == 0
        assert isinstance(block, FilterbankBlock)
        assert isinstance(block, np.ndarray)

    def test_downsample(self, filfile_8bit_1):
        tfactor = 4
        ffactor = 2
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        new_data = data.downsample(tfactor, ffactor)
        np.testing.assert_equal(
            new_data.shape, (data.shape[0] // ffactor, data.shape[1] // tfactor)
        )
        np.testing.assert_equal(new_data.header.nchans, data.header.nchans // ffactor)
        np.testing.assert_equal(new_data.header.nsamples, data.header.nsamples // tfactor)
        np.testing.assert_equal(new_data.header.foff, data.header.foff * ffactor)
        np.testing.assert_equal(new_data.header.tsamp, data.header.tsamp * tfactor)

    def test_downsample_invalid(self, filfile_8bit_1):
        tfactor = 4
        ffactor = 3
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        with pytest.raises(ValueError):
            data.downsample(tfactor, ffactor)

    def test_normalise_chans(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        norm_data = data.normalise()
        np.testing.assert_equal(norm_data.header.nchans, data.header.nchans)
        np.testing.assert_allclose(norm_data.mean(), 0, atol=0.01)
        np.testing.assert_allclose(norm_data.std(), 1, atol=0.01)

    def test_normalise(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        norm_data = data.normalise(chans=False)
        np.testing.assert_equal(norm_data.header.nchans, data.header.nchans)
        np.testing.assert_allclose(norm_data.mean(), 0, atol=0.01)
        np.testing.assert_allclose(norm_data.std(), 1, atol=0.01)       

    def test_get_tim(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        tim = data.get_tim()
        assert isinstance(tim, TimeSeries)
        np.testing.assert_equal(tim.header.nchans, 1)
        np.testing.assert_equal(tim.header.dm, data.dm)

    def test_get_bandpass(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        bpass = data.get_bandpass()
        np.testing.assert_equal(bpass.size, data.header.nchans)

    def test_dedisperse(self, filfile_8bit_1):
        dm = 50
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        dm_data = data.dedisperse(dm)
        np.testing.assert_equal(data.shape, dm_data.shape)
        np.testing.assert_equal(dm_data.dm, dm)
        np.testing.assert_array_equal(data.mean(axis=1), dm_data.mean(axis=1))

    def test_dedisperse_valid_samples(self, filfile_8bit_1):
        dm = 50
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        dm_data = data.dedisperse(dm, only_valid_samples=True)
        np.testing.assert_equal(dm_data.dm, dm)
        np.testing.assert_equal(dm_data.shape[0], data.shape[0])
        np.testing.assert_equal(
            dm_data.shape[1], data.shape[1] - data.header.get_dmdelays(dm).max()
        )

    def test_dedisperse_valid_samples_fail(self, filfile_8bit_1):
        dm = 10000
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        with pytest.raises(ValueError):
            data.dedisperse(dm, only_valid_samples=True)

    def test_dmt_transform(self, filfile_8bit_1):
        dm = 50
        dmsteps = 256
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        dmt_data = data.dmt_transform(dm, dmsteps)
        np.testing.assert_equal(dmt_data.shape[0], dmsteps)
        np.testing.assert_equal(dmt_data.shape[1], data.shape[1])
        np.testing.assert_equal(dmt_data.dm, dm)

    def test_to_file(self, filfile_8bit_1, tmpfile):
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        outfile = data.to_file(tmpfile)
        new_fil = FilReader(outfile)
        new_data = new_fil.read_block(0, 1024)
        np.testing.assert_equal(new_fil.header.nbits, 32)
        np.testing.assert_array_equal(data, new_data)

    def test_to_file_without_path(self, filfile_8bit_1):
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        outfile = data.to_file()
        outfile_path = Path(outfile)
        assert outfile_path.is_file()

        new_fil = FilReader(outfile)
        new_data = new_fil.read_block(0, 1024)
        np.testing.assert_equal(new_fil.header.nbits, 32)
        np.testing.assert_array_equal(data, new_data)
        outfile_path.unlink()
