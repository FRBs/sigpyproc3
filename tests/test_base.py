from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigpyproc.core import rfi, stats
from sigpyproc.foldedcube import FoldedData
from sigpyproc.readers import FilReader
from sigpyproc.timeseries import TimeSeries


class TestFilterbank:
    def test_compute_stats(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        fil.compute_stats()
        assert isinstance(fil.chan_stats, stats.ChannelStats)

    def test_compute_stats_basic(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        fil.compute_stats_basic()
        assert isinstance(fil.chan_stats, stats.ChannelStats)

    def test_collapse(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        tim = fil.collapse()
        np.testing.assert_equal(tim.data.dtype, np.float32)
        np.testing.assert_equal(tim.data.size, fil.header.nsamples)
        np.testing.assert_equal(tim.header.nchans, 1)

    def test_bandpass(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        tim = fil.bandpass()
        np.testing.assert_equal(tim.data.dtype, np.float32)
        np.testing.assert_equal(tim.data.size, fil.header.nchans)
        np.testing.assert_equal(tim.header.nchans, 1)

    def test_dedisperse(self, filfile_4bit: str) -> None:
        dm = 100
        fil = FilReader(filfile_4bit)
        tim = fil.dedisperse(100)
        np.testing.assert_equal(tim.data.dtype, np.float32)
        np.testing.assert_equal(tim.header.nchans, 1)
        np.testing.assert_equal(tim.header.dm, dm)

    def test_read_chan(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        tim = fil.read_chan(5)
        np.testing.assert_equal(tim.data.dtype, np.float32)
        np.testing.assert_equal(tim.data.size, fil.header.nsamples)
        np.testing.assert_equal(tim.header.nchans, 1)
        with pytest.raises(ValueError):
            fil.read_chan(50000)
        with pytest.raises(ValueError):
            fil.read_chan(-1)

    def test_invert_freq(self, filfile_4bit: str, tmpfile: str) -> None:
        fil = FilReader(filfile_4bit)
        block = fil.read_block(0, 100)
        outfile = fil.invert_freq(outfile_name=tmpfile)
        new_fil = FilReader(outfile)
        block_new = new_fil.read_block(0, 100)
        np.testing.assert_equal(new_fil.header.dtype, fil.header.dtype)
        np.testing.assert_equal(new_fil.header.nsamples, fil.header.nsamples)
        np.testing.assert_equal(new_fil.header.foff, -1 * fil.header.foff)
        np.testing.assert_array_equal(block.data, np.flip(block_new.data, axis=0))

    def test_apply_channel_mask(self, filfile_4bit: str, tmpfile: str) -> None:
        fil = FilReader(filfile_4bit)
        rng = np.random.default_rng()
        chanmask = rng.integers(2, size=fil.header.nchans)
        outfile = fil.apply_channel_mask(chanmask=chanmask, outfile_name=tmpfile)
        new_fil = FilReader(outfile)
        block_new = new_fil.read_block(0, 100)
        np.testing.assert_equal(new_fil.header.dtype, fil.header.dtype)
        np.testing.assert_equal(new_fil.header.nsamples, fil.header.nsamples)
        np.testing.assert_array_equal(
            np.where(~block_new.data.any(axis=1))[0],
            np.where(chanmask == 1)[0],
        )

    def test_downsample(self, filfile_4bit: str, tmpfile: str) -> None:
        tfactor = 4
        ffactor = 2
        fil = FilReader(filfile_4bit)
        outfile = fil.downsample(tfactor=tfactor, ffactor=ffactor, outfile_name=tmpfile)
        new_fil = FilReader(outfile)
        np.testing.assert_equal(new_fil.header.dtype, fil.header.dtype)
        np.testing.assert_equal(new_fil.header.tsamp, fil.header.tsamp * tfactor)
        np.testing.assert_equal(new_fil.header.nsamples, fil.header.nsamples // tfactor)
        np.testing.assert_equal(new_fil.header.foff, fil.header.foff * ffactor)
        np.testing.assert_equal(new_fil.header.nchans, fil.header.nchans // ffactor)
        with pytest.raises(ValueError):
            fil.downsample(ffactor=3)

    def test_extract_samps(self, filfile_4bit: str, tmpfile: str) -> None:
        nsamps = 1024
        fil = FilReader(filfile_4bit)
        outfile = fil.extract_samps(start=100, nsamps=nsamps, outfile_name=tmpfile)
        new_fil = FilReader(outfile)
        np.testing.assert_equal(new_fil.header.dtype, fil.header.dtype)
        np.testing.assert_equal(new_fil.header.nsamples, nsamps)
        with pytest.raises(ValueError):
            fil.extract_samps(start=100, nsamps=fil.header.nsamples + 1)
        with pytest.raises(ValueError):
            fil.extract_samps(start=-10, nsamps=nsamps)

    @pytest.mark.parametrize("chans", [None, np.array([1, 2, 3, 4, 5])])
    def test_extract_chans(self, filfile_4bit: str, chans: np.ndarray | None) -> None:
        fil = FilReader(filfile_4bit)
        timfiles = fil.extract_chans(chans)
        for timfile in timfiles:
            timfile_path = Path(timfile)
            assert timfile_path.is_file()
            tim = TimeSeries.from_tim(timfile)
            np.testing.assert_equal(tim.header.nbits, 32)
            np.testing.assert_equal(tim.header.nchans, 1)
            timfile_path.unlink()
        with pytest.raises(ValueError):
            fil.extract_chans(np.array([5000]))

    @pytest.mark.parametrize("chanpersub", [None, 32])
    def test_extract_bands(self, filfile_4bit: str, chanpersub: int | None) -> None:
        chanstart = 10
        nchans = 96
        fil = FilReader(filfile_4bit)
        subfiles = fil.extract_bands(
            chanstart=chanstart,
            nchans=nchans,
            chanpersub=chanpersub,
        )
        nchans_out = nchans if chanpersub is None else chanpersub
        for subfile in subfiles:
            subfile_path = Path(subfile)
            assert subfile_path.is_file()
            new_fil = FilReader(subfile)
            np.testing.assert_equal(new_fil.header.nbits, fil.header.nbits)
            np.testing.assert_equal(new_fil.header.dtype, fil.header.dtype)
            np.testing.assert_equal(new_fil.header.nsamples, fil.header.nsamples)
            np.testing.assert_equal(new_fil.header.nchans, nchans_out)
            subfile_path.unlink()

        with pytest.raises(ValueError):
            fil.extract_bands(chanstart=chanstart, nchans=nchans, chanpersub=-1)
        with pytest.raises(ValueError):
            fil.extract_bands(chanstart=-1, nchans=nchans, chanpersub=chanpersub)
        if chanpersub is not None:
            with pytest.raises(ValueError):
                fil.extract_bands(
                    chanstart=chanstart,
                    nchans=nchans + 1,
                    chanpersub=chanpersub,
                )

    def test_remove_zerodm(self, filfile_4bit: str, tmpfile: str) -> None:
        fil = FilReader(filfile_4bit)
        outfile = fil.remove_zerodm(outfile_name=tmpfile)
        new_fil = FilReader(outfile)
        np.testing.assert_equal(new_fil.header.nchans, fil.header.nchans)
        np.testing.assert_equal(new_fil.header.nsamples, fil.header.nsamples)

    def test_subband(self, filfile_4bit: str, tmpfile: str) -> None:
        fil = FilReader(filfile_4bit)
        outfile = fil.subband(dm=0, nsub=fil.header.nchans // 16, outfile_name=tmpfile)
        new_fil = FilReader(outfile)
        np.testing.assert_equal(new_fil.header.nchans, fil.header.nchans // 16)
        np.testing.assert_equal(new_fil.header.nsamples, fil.header.nsamples)

    def test_fold(self, filfile_4bit: str) -> None:
        fil = FilReader(filfile_4bit)
        cube = fil.fold(period=1, dm=10, nints=16, nbins=50)
        assert isinstance(cube, FoldedData)
        np.testing.assert_equal(cube.header.nchans, fil.header.nchans)
        np.testing.assert_equal(cube.nsubints, 16)
        np.testing.assert_equal(cube.nbins, 50)

    def test_clean_rfi(self, filfile_4bit: str, tmpfile: str) -> None:
        fil = FilReader(filfile_4bit)
        out_file, rfimask = fil.clean_rfi(outfile_name=tmpfile)
        new_fil = FilReader(out_file)
        np.testing.assert_equal(new_fil.header.nchans, fil.header.nchans)
        np.testing.assert_equal(new_fil.header.nsamples, fil.header.nsamples)
        with pytest.raises(ValueError):
            fil.clean_rfi(method="invalid")

    def test_clean_rfi_mask(self, filfile_4bit: str, tmpfile: str) -> None:
        fil = FilReader(filfile_4bit)
        out_file, rfimask = fil.clean_rfi(outfile_name=tmpfile)
        assert isinstance(rfimask, rfi.RFIMask)
        mask_file = Path(rfimask.to_file())
        assert mask_file.is_file()
        mask_file.unlink()
