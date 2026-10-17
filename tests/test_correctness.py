"""Value-level checks for decimation, dedispersion, and partial reads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigpyproc.readers import FilReader, PFITSReader, _dedisp_read_mode
from sigpyproc.simulation.pulsar import make_header


def _write_fil(
    path: str,
    data: np.ndarray,
    *,
    fch1: float = 1400.0,
    foff: float = -10.0,
    tsamp: float = 1e-3,
) -> None:
    nsamps, nchans = data.shape
    header = make_header(
        filename=path,
        nchans=nchans,
        nbits=32,
        tsamp=tsamp,
        tobs=nsamps * tsamp,
        tstart=50000.0,
        fch1=fch1,
        foff=foff,
    )
    with header.prep_outfile(path, nbits=32, sources=()) as outfile:
        outfile.cwrite(np.asarray(data, dtype=np.float32).ravel())


def test_remove_zerodm_keeps_a_constant_baseline(tmpdir: str) -> None:
    nsamps, nchans = 128, 8
    data = np.full((nsamps, nchans), 5, dtype=np.float32)
    path = str(Path(tmpdir) / "const.fil")
    out = str(Path(tmpdir) / "const_zdm.fil")
    _write_fil(path, data, foff=-1.0)
    FilReader(path).remove_zerodm(outfile_name=out, gulp=32)
    got = FilReader(out).read_block(0, nsamps).data
    np.testing.assert_allclose(got, 5)


def test_remove_zerodm_suppresses_a_common_spike(tmpdir: str) -> None:
    nsamps, nchans = 16, 4
    data = np.zeros((nsamps, nchans), dtype=np.float32)
    data[4] = 8
    path = str(Path(tmpdir) / "spike.fil")
    out = str(Path(tmpdir) / "spike_zdm.fil")
    _write_fil(path, data, foff=-1.0)
    FilReader(path).remove_zerodm(outfile_name=out)
    got = FilReader(out).read_block(0, nsamps).data
    # The common spike is removed and the mean bandpass, which includes it, is restored.
    np.testing.assert_allclose(got, 0.5)


def test_channel_stats_use_the_window_length(tmpdir: str) -> None:
    nsamps = 128
    data = np.column_stack(
        [
            np.arange(nsamps, dtype=np.float32),
            np.full(nsamps, 3, dtype=np.float32),
        ],
    )
    path = str(Path(tmpdir) / "ramp.fil")
    _write_fil(path, data, foff=-1.0)
    fil = FilReader(path)
    window = 32
    fil.compute_stats(nsamps=window, gulp=8)
    expected = np.var(data[:window, 0])
    np.testing.assert_allclose(fil.chan_stats.var[0], expected)
    assert fil.chan_stats.nsamps == window


class TestDownsample:
    def test_matches_block_decimation(self, tmpdir: str) -> None:
        nsamps, nchans = 8, 6
        data = np.arange(nsamps * nchans, dtype=np.float32).reshape(nsamps, nchans)
        path = str(Path(tmpdir) / "ramp.fil")
        out = str(Path(tmpdir) / "ramp_dec.fil")
        _write_fil(path, data, foff=-1.0)
        fil = FilReader(path)
        fil.downsample(tfactor=2, ffactor=3, outfile_name=out)
        got = FilReader(out).read_block(0, nsamps // 2)
        expected = fil.read_block(0, nsamps).downsample(3, 2)
        np.testing.assert_allclose(got.data, expected.data)
        assert got.header.fch1 == expected.header.fch1
        assert got.header.nsamples == nsamps // 2
        assert got.header.nchans == nchans // 3


class TestSubband:
    def test_constant_band_does_not_accumulate(self, tmpdir: str) -> None:
        nsamps, nchans, nsub = 6, 4, 2
        data = np.ones((nsamps, nchans), dtype=np.float32)
        path = str(Path(tmpdir) / "ones.fil")
        out = str(Path(tmpdir) / "ones.sub")
        _write_fil(path, data, foff=-1.0)
        fil = FilReader(path)
        fil.subband(dm=0.0, nsub=nsub, outfile_name=out, gulp=3)
        got = FilReader(out).read_block(0, nsamps)
        np.testing.assert_allclose(got.data, nchans / nsub)
        subfactor = nchans // nsub
        expected_fch1 = fil.header.fch1 + fil.header.foff * (subfactor - 1) / 2
        assert got.header.fch1 == expected_fch1
        assert got.header.foff == fil.header.foff * (nchans // nsub)


def _dispersed_impulse(
    path: str,
    *,
    fch1: float,
    foff: float,
    dm: float,
    ref_freq: str,
) -> tuple[FilReader, int]:
    nsamps, nchans = 4000, 4
    _write_fil(
        path,
        np.zeros((nsamps, nchans), dtype=np.float32),
        fch1=fch1,
        foff=foff,
        tsamp=1e-4,
    )
    fil = FilReader(path)
    delays = np.asarray(fil.header.get_dmdelays(dm, ref_freq=ref_freq), dtype=int)
    lo = int(delays.min())
    hi = int(delays.max())
    origin = -lo + 5
    if origin + hi >= nsamps:
        msg = f"delay span {lo}..{hi} does not fit in {nsamps} samples"
        raise RuntimeError(msg)
    data = np.zeros((nsamps, nchans), dtype=np.float32)
    for ichan, delay in enumerate(delays):
        data[origin + int(delay), ichan] = 1.0
    _write_fil(path, data, fch1=fch1, foff=foff, tsamp=1e-4)
    return FilReader(path), nchans


class TestDedisperse:
    def test_fcenter_stacks_a_dispersed_impulse(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "imp.fil")
        fil, nchans = _dispersed_impulse(
            path,
            fch1=1500.0,
            foff=-50.0,
            dm=50.0,
            ref_freq="fcenter",
        )
        delays = fil.header.get_dmdelays(50.0, ref_freq="fcenter")
        assert int(np.min(delays)) < 0
        series = fil.dedisperse(50.0, ref_freq="fcenter", gulp=64)
        assert series.data.max() == nchans
        assert int(np.count_nonzero(series.data)) == 1

    def test_positive_foff_stacks(self, tmpdir: str) -> None:
        path = str(Path(tmpdir) / "pos.fil")
        fil, nchans = _dispersed_impulse(
            path,
            fch1=1000.0,
            foff=50.0,
            dm=40.0,
            ref_freq="fch1",
        )
        delays = fil.header.get_dmdelays(40.0)
        assert int(np.min(delays)) < 0
        series = fil.dedisperse(40.0, gulp=64)
        assert series.data.max() == nchans
        assert int(np.count_nonzero(series.data)) == 1


class TestPartialReads:
    def test_collapse_ragged_gulp(self, tmpdir: str) -> None:
        nsamps, nchans = 100, 4
        data = np.arange(nsamps, dtype=np.float32)[:, None] * np.ones(
            nchans,
            dtype=np.float32,
        )
        path = str(Path(tmpdir) / "idx.fil")
        _write_fil(path, data, foff=-1.0)
        series = FilReader(path).collapse(gulp=30)
        np.testing.assert_allclose(series.data, np.arange(nsamps) * nchans)

    def test_read_chan_ragged_gulp(self, tmpdir: str) -> None:
        nsamps, nchans = 50, 3
        data = np.arange(nsamps * nchans, dtype=np.float32).reshape(nsamps, nchans)
        path = str(Path(tmpdir) / "chan.fil")
        _write_fil(path, data, foff=-1.0)
        series = FilReader(path).read_chan(1, gulp=16)
        np.testing.assert_allclose(series.data, data[:, 1])
        assert series.nsamples == nsamps

    def test_read_plan_stops_at_requested_samples(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        counts = [nsamp for nsamp, _, _ in fil.read_plan(gulp=1000, nsamps=2500)]
        assert counts == [1000, 1000, 500]

    def test_read_dedisp_block_follows_the_delay(self, tmpdir: str) -> None:
        nsamps, nchans = 40, 4
        data = np.zeros((nsamps, nchans), dtype=np.float32)
        path = str(Path(tmpdir) / "dd.fil")
        _write_fil(path, data, fch1=1400.0, foff=-20.0, tsamp=1e-3)
        fil = FilReader(path)
        delays = np.asarray(fil.header.get_dmdelays(20.0), dtype=int)
        assert int(delays.max()) > 0
        origin = 5
        for ichan, delay in enumerate(delays):
            data[origin + int(delay), ichan] = 7.0
        _write_fil(path, data, fch1=1400.0, foff=-20.0, tsamp=1e-3)
        fil = FilReader(path)
        direct = fil.read_dedisp_block(origin, 8, 20.0, small_reads=False)
        sought = fil.read_dedisp_block(origin, 8, 20.0, small_reads=True)
        np.testing.assert_allclose(direct.data[:, 0], 7.0)
        np.testing.assert_array_equal(direct.data, sought.data)
        assert direct.header.nsamples == 8


class TestDedispReadMode:
    def test_small_window_is_one_read(self) -> None:
        assert _dedisp_read_mode(1000, 800, 1024, 8, small_reads=None) == "bulk"

    def test_wide_high_dm_seeks_channels(self) -> None:
        # 20k channels, 4k output samples, 160k-sample delay: a few GiB.
        nsamps = 4_000
        n_in = 160_000
        stride = 20_000
        assert _dedisp_read_mode(n_in, nsamps, stride, 8, small_reads=None) == "partial"

    def test_wide_low_dm_streams(self) -> None:
        nsamps = 80_000
        n_in = 100_000
        stride = 4_000
        assert _dedisp_read_mode(n_in, nsamps, stride, 8, small_reads=None) == "stream"

    def test_subbyte_cannot_force_channel_seeks(self) -> None:
        with pytest.raises(ValueError, match="8 bits"):
            _dedisp_read_mode(10_000, 100, 416, 4, small_reads=True)

    def test_modes_match_on_8bit(self, filfile_8bit_1: str) -> None:
        fil = FilReader(filfile_8bit_1)
        dm = 50.0
        nsamps = 1500
        bulk = fil.read_dedisp_block(0, nsamps, dm, small_reads=False)
        sought = fil.read_dedisp_block(0, nsamps, dm, small_reads=True)
        np.testing.assert_array_equal(bulk.data, sought.data)
        assert bulk.data.shape == (fil.header.nchans, nsamps)
        fil.close()


class TestPFITSReadBlock:
    def test_read_across_subint(self, fitsfile_4bit: str) -> None:
        fits = PFITSReader(fitsfile_4bit)
        sub = fits.sub_hdr.subint_samples
        nsamps = 30
        block = fits.read_block(sub - 10, nsamps)
        direct = fits._read_samples(sub - 10, nsamps)
        assert block.data.shape == (fits.header.nchans, nsamps)
        np.testing.assert_allclose(block.data, direct.T)
        fits.close()
