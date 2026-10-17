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

    def test_normalise_and_pad_keep_dm(self, filfile_8bit_1: str) -> None:
        header = Header.from_sigproc(filfile_8bit_1).new_header(
            {"nchans": 4, "nsamples": 16},
        )
        data = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
        block = FilterbankBlock(data, header, dm=12.5)
        assert block.normalise().dm == pytest.approx(12.5)
        padded = block.pad_samples(20, 3)
        assert padded.dm == pytest.approx(12.5)
        assert padded.header.nsamples == 20
        assert padded.header.tstart == pytest.approx(header.mjd_after_nsamps(-3))
        np.testing.assert_array_equal(padded.data[:, 3:19], data)

    def test_dmt_normalise_keeps_the_grid(self, filfile_8bit_1: str) -> None:
        header = Header.from_sigproc(filfile_8bit_1).new_header({"nsamples": 8})
        dms = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        data = np.arange(24, dtype=np.float32).reshape(3, 8)
        block = DMTBlock(data, header, dms)
        normalised = block.normalise()
        assert isinstance(normalised, DMTBlock)
        np.testing.assert_array_equal(normalised.dms, dms)

    def test_downsample(self, filfile_8bit_1: str) -> None:
        tfactor = 4
        ffactor = 2
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_down = block.downsample(ffactor, tfactor)
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
            block.normalise(loc_method="invalid")  # type: ignore[arg-type]

    def test_pad_samples(self, filfile_8bit_1: str) -> None:
        nsamps_final = 2048
        offset = 512
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(100, 1024)
        block_pad = block.pad_samples(nsamps_final, offset)
        np.testing.assert_equal(block_pad.data.shape[1], 2048)
        np.testing.assert_equal(block_pad.header.nsamples, 2048)
        with pytest.raises(ValueError):
            block.pad_samples(nsamps_final, offset, pad_mode="invalid")  # type: ignore[arg-type]

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
        data = np.zeros((fil.header.nchans, fil.header.nsamples), dtype=np.float32)
        block = FilterbankBlock(data, fil.header)
        block.data[:, block.nsamples // 2] = 1.0
        block_dedisp = block.dedisperse(dm)
        np.testing.assert_equal(block_dedisp.data.shape, block.data.shape)
        np.testing.assert_equal(block_dedisp.dm, dm)
        np.testing.assert_array_equal(
            block_dedisp.data.mean(axis=1),
            block.data.mean(axis=1),
        )
        # check the direction of the dedispersion
        assert block_dedisp.data[:, block.nsamples // 2 + 1 :].sum() == 0
        block_dedisp = block.dedisperse(-dm)
        assert block_dedisp.data[:, : block.nsamples // 2].sum() == 0

    def test_dedisperse_valid_samples(self, filfile_8bit_1: str) -> None:
        dm = 50
        fil = FilReader(filfile_8bit_1)
        data = np.zeros((fil.header.nchans, fil.header.nsamples), dtype=np.float32)
        block = FilterbankBlock(data, fil.header)
        block.data[:, block.nsamples // 2] = 1.0
        block_dedisp = block.dedisperse(dm, only_valid_samples=True)
        np.testing.assert_equal(block_dedisp.dm, dm)
        np.testing.assert_equal(block_dedisp.data.shape[0], block.data.shape[0])
        np.testing.assert_equal(
            block_dedisp.nsamples,
            block.nsamples - block.header.get_dmdelays(dm).max(),
        )
        # check the direction of the dedispersion
        assert block_dedisp.data[:, block.nsamples // 2 + 1 :].sum() == 0

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
        block = fil.read_block(100, 256)
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

    def test_plot(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(0, 512)
        fig = block.plot()
        assert fig is not None
        fig_no_panels = block.plot(show_ts=False, show_bp=False)
        assert fig_no_panels is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
        plt.close(fig_no_panels)


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

    def test_plot(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        block = fil.read_block(0, 512)
        dmt = block.dmt_transform(dm=10.0, dmsteps=32)
        fig = dmt.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dmt_peaks_at_the_injected_dm(self) -> None:
        from sigpyproc.simulation.pulsar import make_header

        header = make_header(
            nchans=8,
            nbits=32,
            tsamp=1e-4,
            tobs=0.05,
            fch1=1500.0,
            foff=-5.0,
        )
        dm = 20.0
        delays = np.asarray(header.get_dmdelays(dm), dtype=int)
        assert int(delays.max()) < 80
        data = np.zeros((header.nchans, header.nsamples), dtype=np.float32)
        origin = 100
        for ichan, delay in enumerate(delays):
            data[ichan, origin + int(delay)] = 1.0
        block = FilterbankBlock(data, header)
        dmt = block.dmt_transform(dm, dmsteps=5, only_valid_samples=True)
        trial = int(np.argmin(np.abs(np.asarray(dmt.dms) - dm)))
        assert dmt.header.nsamples == dmt.data.shape[1]
        assert dmt.data[trial].max() > dmt.data[0].max()
