from pathlib import Path

import numpy as np
import pytest

from sigpyproc.block import DMTBlock, FilterbankBlock
from sigpyproc.header import Header
from sigpyproc.readers import FilReader
from sigpyproc.timeseries import TimeSeries


class TestFilterbankBlock:
    def test_block_fails(self, filfile_8bit_1: str) -> None:
        rng = np.random.default_rng()
        data = rng.normal(size=(128, 1024))
        header = Header.from_sigproc(filfile_8bit_1)
        with pytest.raises(TypeError):
            FilterbankBlock(data, data)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            FilterbankBlock(data[0], header.new_header())
        with pytest.raises(ValueError):
            FilterbankBlock(data, header.new_header())

    def test_block_header(self, filfile_8bit_1: str) -> None:
        rng = np.random.default_rng()
        data = rng.normal(size=(128, 1024))
        header = Header.from_sigproc(filfile_8bit_1)
        block = FilterbankBlock(
            data,
            header.new_header({"nchans": 128, "nsamples": 1024}),
        )
        assert isinstance(block.header, Header)
        assert block.dm == 0
        assert isinstance(block, FilterbankBlock)

    def test_downsample(self, filfile_8bit_1: str) -> None:
        tfactor = 4
        ffactor = 2
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_down = block.downsample(tfactor, ffactor)
        np.testing.assert_equal(
            block_down.data.shape,
            (block.nchans // ffactor, block.nsamples // tfactor),
        )
        np.testing.assert_equal(
            block_down.header.nchans,
            block.header.nchans // ffactor,
        )
        np.testing.assert_equal(
            block_down.header.nsamples,
            block.header.nsamples // tfactor,
        )
        np.testing.assert_equal(block_down.header.foff, block.header.foff * ffactor)
        np.testing.assert_equal(block_down.header.tsamp, block.header.tsamp * tfactor)

    @pytest.mark.parametrize(("tfactor", "ffactor"), [(4, 3), (3, 4), (7, 7)])
    def test_downsample_invalid(
        self,
        filfile_8bit_1: str,
        tfactor: int,
        ffactor: int,
    ) -> None:
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        with pytest.raises(ValueError):
            data.downsample(tfactor, ffactor)

    def test_normalise_chans(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_norm = block.normalise(axis=1)
        np.testing.assert_equal(block_norm.header.nchans, block.header.nchans)
        np.testing.assert_allclose(block_norm.data.mean(), 0, atol=0.01)
        np.testing.assert_allclose(block_norm.data.std(), 1, atol=0.01)

    def test_normalise(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_norm = block.normalise(axis=None)
        np.testing.assert_equal(block_norm.header.nchans, block.header.nchans)
        np.testing.assert_allclose(block_norm.data.mean(), 0, atol=0.01)
        np.testing.assert_allclose(block_norm.data.std(), 1, atol=0.01)
        with pytest.raises(ValueError):
            block.normalise(loc_method="invalid") # type: ignore[arg-type]

    def test_pad_samples(self, filfile_8bit_1: str) -> None:
        nsamps_final = 2048
        offset = 512
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_pad = block.pad_samples(nsamps_final, offset)
        np.testing.assert_equal(block_pad.data.shape[1], 2048)
        np.testing.assert_equal(block_pad.header.nsamples, 2048)
        with pytest.raises(ValueError):
            block.pad_samples(nsamps_final, offset, pad_mode="invalid") # type: ignore[arg-type]

    def test_get_tim(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        tim = data.get_tim()
        assert isinstance(tim, TimeSeries)
        np.testing.assert_equal(tim.header.nchans, 1)
        np.testing.assert_equal(tim.header.dm, data.dm)

    def test_get_bandpass(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        bpass = data.get_bandpass()
        np.testing.assert_equal(bpass.size, data.header.nchans)

    def test_dedisperse(self, filfile_8bit_1: str) -> None:
        dm = 50
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_dedisp = block.dedisperse(dm)
        np.testing.assert_equal(block.data.shape, block_dedisp.data.shape)
        np.testing.assert_equal(block_dedisp.dm, dm)
        np.testing.assert_array_equal(
            block.data.mean(axis=1),
            block_dedisp.data.mean(axis=1),
        )

    def test_dedisperse_valid_samples(self, filfile_8bit_1: str) -> None:
        dm = 50
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_dedisp = block.dedisperse(dm, only_valid_samples=True)
        np.testing.assert_equal(block_dedisp.dm, dm)
        np.testing.assert_equal(block_dedisp.data.shape[0], block.data.shape[0])
        np.testing.assert_equal(
            block_dedisp.nsamples,
            block.nsamples - block.header.get_dmdelays(dm).max(),
        )

    def test_dedisperse_valid_samples_fail(self, filfile_8bit_1: str) -> None:
        dm = 10000
        fil = FilReader(filfile_8bit_1)
        data = fil.read_block(100, 1024)
        with pytest.raises(ValueError):
            data.dedisperse(dm, only_valid_samples=True)

    def test_dmt_transform(self, filfile_8bit_1: str) -> None:
        dm = 50
        dmsteps = 256
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_dmt = block.dmt_transform(dm, dmsteps)
        np.testing.assert_equal(block_dmt.ndms, dmsteps)
        np.testing.assert_equal(block_dmt.nsamples, block.nsamples)

    def test_to_file(self, filfile_8bit_1: str, tmpfile: str) -> None:
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        outfile = block.to_file(tmpfile)
        new_fil = FilReader(outfile)
        block_new = new_fil.read_block(0, 1024)
        np.testing.assert_equal(new_fil.header.nbits, 32)
        np.testing.assert_array_equal(block_new.data, block.data)

    def test_to_file_without_path(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        outfile = block.to_file()
        outfile_path = Path(outfile)
        assert outfile_path.is_file()

        new_fil = FilReader(outfile)
        block_new = new_fil.read_block(0, 1024)
        np.testing.assert_equal(new_fil.header.nbits, 32)
        np.testing.assert_array_equal(block_new.data, block.data)
        outfile_path.unlink()


class TestDMTBlock:
    def test_block_fails(self, filfile_8bit_1: str) -> None:
        rng = np.random.default_rng()
        data = rng.normal(size=(128, 1024))
        dms = np.linspace(0, 100, 128)
        header = Header.from_sigproc(filfile_8bit_1)
        with pytest.raises(TypeError):
            DMTBlock(data, data, dms)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            DMTBlock(data[0], header.new_header(), dms)
        with pytest.raises(ValueError):
            DMTBlock(data, header.new_header(), dms)
        with pytest.raises(ValueError):
            DMTBlock(data, header.new_header({"nchans": 1, "nsamples": 1024}), dms[:-1])

    def test_block_header(self, filfile_8bit_1: str) -> None:
        rng = np.random.default_rng()
        data = rng.normal(size=(128, 1024))
        dms = np.linspace(0, 100, 128, dtype=np.float32)
        header = Header.from_sigproc(filfile_8bit_1)
        block = DMTBlock(data, header.new_header({"nchans": 1, "nsamples": 1024}), dms)
        assert isinstance(block.header, Header)
        assert isinstance(block, DMTBlock)
        np.testing.assert_array_equal(block.dms, dms)
