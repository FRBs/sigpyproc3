import logging
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

from sigpyproc.io import pfits
from sigpyproc.io.bits import BitsInfo, pack
from sigpyproc.readers import PFITSReader
from sigpyproc.utils import FrequencyChannels


class TestPrimaryHdr:
    def test_read(self, fitsfile_4bit: str) -> None:
        hdr = pfits.PrimaryHdr(fitsfile_4bit)
        assert isinstance(hdr.header, fits.Header)
        assert isinstance(hdr.location, EarthLocation)
        assert isinstance(hdr.coord, SkyCoord)
        assert isinstance(hdr.freqs, FrequencyChannels)
        assert isinstance(hdr.tstart, Time)
        np.testing.assert_equal(hdr.project_id, "P958")
        np.testing.assert_equal(hdr.telescope, "Parkes")


class TestSubintHdr:
    def test_read(self, fitsfile_4bit: str) -> None:
        hdr = pfits.SubintHdr(fitsfile_4bit)
        assert isinstance(hdr.header, fits.Header)
        assert isinstance(hdr.azimuth, Angle)
        assert isinstance(hdr.zenith, Angle)
        assert isinstance(hdr.freqs, FrequencyChannels)
        np.testing.assert_equal(hdr.poln_type, "AABBCRCI")
        np.testing.assert_equal(hdr.poln_state, "Coherence")
        np.testing.assert_equal(hdr.npol, 4)


class TestPFITSFile:
    def test_read(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            assert isinstance(fitsfile.bitsinfo, BitsInfo)

    def test_read_freqs(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            freqs = fitsfile.read_freqs(0)
            assert isinstance(freqs, FrequencyChannels)
            np.testing.assert_equal(freqs.nchans, fitsfile.sub_hdr.nchans)

    def test_read_weights(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            weights = fitsfile.read_weights(0)
            np.testing.assert_equal(weights.size, fitsfile.sub_hdr.nchans)

    def test_read_scales(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            scales = fitsfile.read_scales(0)
            np.testing.assert_equal(
                scales.shape,
                ((fitsfile.sub_hdr.npol, fitsfile.sub_hdr.nchans)),
            )

    def test_read_offsets(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            scales = fitsfile.read_offsets(0)
            np.testing.assert_equal(
                scales.shape,
                ((fitsfile.sub_hdr.npol, fitsfile.sub_hdr.nchans)),
            )

    def test_read_subint(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            data = fitsfile.read_subint(0)
            np.testing.assert_equal(data.shape, fitsfile.sub_hdr.subint_shape)

    def test_read_subint_pol(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            data = fitsfile.read_subint_pol(0, poln_select=1)
            np.testing.assert_equal(
                data.shape,
                (fitsfile.sub_hdr.subint_samples, fitsfile.sub_hdr.nchans),
            )

    def test_read_subints(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            nsub = 2
            data = fitsfile.read_subints(0, nsub)
            np.testing.assert_equal(
                data.shape,
                (fitsfile.sub_hdr.subint_samples * nsub, fitsfile.sub_hdr.nchans),
            )

    def test_read_subint_pol_ppqq(
        self,
        fitsfile_4bit: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            monkeypatch.setattr(
                type(fitsfile.sub_hdr),
                "poln_state",
                property(lambda _self: "PPQQ"),
            )
            raw = fitsfile.read_subint(0)
            data = fitsfile.read_subint_pol(0, poln_select=1)
            np.testing.assert_equal(
                data.shape,
                (fitsfile.sub_hdr.subint_samples, fitsfile.sub_hdr.nchans),
            )
            np.testing.assert_allclose(data, raw[:, 0, :] + raw[:, 1, :])

    def test_read_subint_pol_ppqq_vs_coherence(
        self,
        fitsfile_4bit: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            coherence = fitsfile.read_subint_pol(0, poln_select=1)
            monkeypatch.setattr(
                type(fitsfile.sub_hdr),
                "poln_state",
                property(lambda _self: "PPQQ"),
            )
            ppqq = fitsfile.read_subint_pol(0, poln_select=1)
            np.testing.assert_allclose(ppqq, coherence, rtol=1e-6)

    def test_read_subint_pol_unsupported_state(
        self,
        fitsfile_4bit: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as fitsfile:
            monkeypatch.setattr(
                type(fitsfile.sub_hdr),
                "poln_state",
                property(lambda _self: "Mystery"),
            )
            with pytest.raises(NotImplementedError, match="Mystery"):
                fitsfile.read_subint_pol(0)

    def test_read_subint_pol_invalid_select(self, fitsfile_4bit: str) -> None:
        with (
            pfits.PFITSFile(fitsfile_4bit) as fitsfile,
            pytest.raises(ValueError, match="poln_select=99"),
        ):
            fitsfile.read_subint_pol(0, poln_select=99)


class TestPFITSReaderReadPlan:
    def test_blocks_cover_file(self, fitsfile_4bit: str) -> None:
        reader = PFITSReader(fitsfile_4bit)
        nsamps = reader.header.nsamples
        blocks = list(reader.read_plan(gulp=300, quiet=True))
        assert sum(block for block, _, _ in blocks) == nsamps
        assert all(block <= 300 for block, _, _ in blocks)
        data = np.concatenate([d for _, _, d in blocks])
        expected = reader.read_block(0, nsamps).data.T.ravel()
        np.testing.assert_allclose(data, expected, rtol=1e-6)


def _mhz(value: float) -> float:
    if hasattr(value, "to_value"):
        return float(value.to_value("MHz"))
    return float(value)


def _primary_cards(*, obs_bw: float, nchan: int, fcenter: float) -> fits.Header:
    header = fits.Header()
    header["FITSTYPE"] = "PSRFITS"
    header["OBS_MODE"] = "SEARCH"
    header["OBSERVER"] = "test"
    header["PROJID"] = "test"
    header["TELESCOP"] = "Parkes"
    header["ANT_X"] = -4554231.5
    header["ANT_Y"] = 2816759.1
    header["ANT_Z"] = -3454036.3
    header["FRONTEND"] = "TEST"
    header["NRCVR"] = 2
    header["FD_POLN"] = "LIN"
    header["FD_HAND"] = 1
    header["FD_SANG"] = 0.0
    header["FD_XYPH"] = 0.0
    header["FD_MODE"] = "FA"
    header["FA_REQ"] = 0.0
    header["BACKEND"] = "TEST"
    header["BE_PHASE"] = 0.0
    header["BE_DCC"] = 0
    header["BE_DELAY"] = 0.0
    header["TCYCLE"] = 0.0
    header["BECONFIG"] = "none"
    header["DATE-OBS"] = "2020-01-01T00:00:00.0"
    header["OBSFREQ"] = fcenter
    header["OBSBW"] = obs_bw
    header["OBSNCHAN"] = nchan
    header["SRC_NAME"] = "J0000+0000"
    header["RA"] = "00:00:00.000"
    header["DEC"] = "+00:00:00.000"
    header["STT_IMJD"] = 58849
    header["STT_SMJD"] = 0
    header["STT_OFFS"] = 0.0
    return header


def _data_rows(
    samples: np.ndarray,
    nbits: int,
    permute: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Pack ``(nsub, nsblk, npol, nchan)`` into standard-or-permuted DATA bytes."""
    _nsub, nsblk, npol, nchan = samples.shape
    bitfact = 8 // nbits if nbits < 8 else 1
    packed_time = nsblk // bitfact if nbits < 8 else nsblk
    rows = [
        _pack_row(samples[isub], nbits, packed_time, permute)
        for isub in range(len(samples))
    ]
    return np.stack(rows), (nchan, npol, packed_time)


def _pack_row(
    cube: np.ndarray,
    nbits: int,
    packed_time: int,
    permute: tuple[int, int, int],
) -> np.ndarray:
    _nsblk, npol, nchan = cube.shape
    if nbits < 8:
        packed = pack(
            np.ascontiguousarray(cube, dtype=np.uint8).reshape(-1),
            nbits,
            bitorder="big",
        )
        elements = packed.reshape(packed_time, npol, nchan)
    elif nbits == 16:
        elements = np.ascontiguousarray(cube.astype(np.int16))
    else:
        elements = np.ascontiguousarray(cube.astype(np.uint8))
    ordered = np.ascontiguousarray(np.transpose(elements, permute))
    if nbits == 16:
        return np.ascontiguousarray(ordered.astype(">i2")).view(np.uint8).ravel()
    return ordered.ravel()


def _write_search(
    path: Path,
    samples: np.ndarray,
    *,
    nbits: int,
    pol_type: str,
    chan0: float,
    chan_bw: float,
    obs_bw: float = 0.0,
    signint: int = 0,
    zero_off: float = 0.0,
    scales: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    tdim: tuple[int, int, int] | None = None,
    permute: tuple[int, int, int] = (0, 1, 2),
) -> None:
    """Write ``samples`` of shape ``(nsub, nsblk, npol, nchan)``.

    ``permute`` reorders the C-order byte cube ``(time, pol, chan)`` before it
    is stored, and ``tdim`` must describe that layout. The default is the
    PSRFITS Fortran order ``(nchan, npol, packed_time)``.
    """
    nsub, nsblk, npol, nchan = samples.shape
    data, default_tdim = _data_rows(samples, nbits, permute)
    if tdim is None:
        tdim = default_tdim
    if scales is None:
        scales = np.ones((npol, nchan), dtype=np.float32)
    if offsets is None:
        offsets = np.zeros((npol, nchan), dtype=np.float32)
    if weights is None:
        weights = np.ones(nchan, dtype=np.float32)
    freqs = chan0 + np.arange(nchan) * chan_bw
    freq_rows = np.broadcast_to(freqs, (nsub, nchan)).copy()
    scale_rows = np.broadcast_to(scales.ravel(), (nsub, npol * nchan)).copy()
    offset_rows = np.broadcast_to(offsets.ravel(), (nsub, npol * nchan)).copy()
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    weight_rows = np.broadcast_to(weights, (nsub, weights.size)).copy()
    # Row centres of a contiguous observation. TBIN below is the same 1 ms.
    tbin = 1e-3
    duration = nsblk * tbin
    tsub = np.full(nsub, duration)
    offs = (np.arange(nsub, dtype=np.float64) + 0.5) * duration
    dim = str(tdim).replace(" ", "")
    if nbits == 16:
        ordered = np.stack(
            [
                np.transpose(samples[irow].astype(np.int16), permute)
                for irow in range(nsub)
            ],
        )
        data_column = fits.Column(
            name="DATA",
            format=f"{nsblk * npol * nchan}I",
            dim=dim,
            array=ordered,
        )
    else:
        data_column = fits.Column(
            name="DATA",
            format=f"{data.shape[1]}B",
            dim=dim,
            array=data,
        )
    columns = [
        fits.Column(name="TSUBINT", format="1D", array=tsub),
        fits.Column(name="OFFS_SUB", format="1D", array=offs),
        fits.Column(name="DAT_FREQ", format=f"{nchan}D", array=freq_rows),
        fits.Column(name="DAT_WTS", format=f"{weights.size}E", array=weight_rows),
        fits.Column(name="DAT_OFFS", format=f"{npol * nchan}E", array=offset_rows),
        fits.Column(name="DAT_SCL", format=f"{npol * nchan}E", array=scale_rows),
        data_column,
    ]
    table = fits.BinTableHDU.from_columns(columns, name="SUBINT")
    table.header["POL_TYPE"] = pol_type
    table.header["NPOL"] = npol
    table.header["NBITS"] = nbits
    table.header["NCHAN"] = nchan
    table.header["NSBLK"] = nsblk
    table.header["TBIN"] = tbin
    table.header["CHAN_BW"] = chan_bw
    table.header["SIGNINT"] = signint
    table.header["ZERO_OFF"] = zero_off
    table.header["NSUBOFFS"] = 0
    table.header["NSTOT"] = nsub * nsblk
    primary = fits.PrimaryHDU(
        header=_primary_cards(
            obs_bw=obs_bw,
            nchan=nchan,
            fcenter=chan0 + 0.5 * chan_bw * (nchan - 1),
        )
    )
    fits.HDUList([primary, table]).writeto(path, overwrite=True)


class TestSearchLayout:
    def test_open_does_not_load_the_table(self, fitsfile_4bit: str) -> None:
        with pfits.PFITSFile(fitsfile_4bit) as src:
            assert src._table_loaded is False
            assert src.pri_hdr.freqs.fch1.value == pytest.approx(4028.0)
            assert src.sub_hdr.freqs.foff.value == pytest.approx(-8.0)

    def test_positive_band_keeps_dat_freq_order(self, tmp_path: Path) -> None:
        nsblk, npol, nchan, nsub = 8, 4, 4, 2
        samples = np.arange(nsub * nsblk * npol * nchan, dtype=np.float32)
        samples = samples.reshape(nsub, nsblk, npol, nchan)
        path = tmp_path / "up.sf"
        _write_search(
            path,
            samples,
            nbits=8,
            pol_type="AABBCRCI",
            chan0=100.0,
            chan_bw=0.5,
            obs_bw=0.0,
            zero_off=0.0,
        )
        with pfits.PFITSFile(str(path)) as src:
            assert src._table_loaded is False
            got = src.read_subint(0, scloffs=False, weights=False)
            np.testing.assert_array_equal(got, samples[0])
            intensity = src.read_subint_pol(0, scloffs=False, weights=False)
            np.testing.assert_array_equal(
                intensity,
                samples[0, :, 0] + samples[0, :, 1],
            )
        reader = PFITSReader(str(path))
        block = reader.read_block(0, nsblk)
        assert _mhz(block.header.fch1) == pytest.approx(100.0)
        assert _mhz(block.header.foff) == pytest.approx(0.5)
        np.testing.assert_array_equal(
            block.data[:, 0],
            samples[0, 0, 0] + samples[0, 0, 1],
        )
        header = reader.header
        assert _mhz(header.fch1) == pytest.approx(100.0)
        assert header.nchans == nchan

    def test_transposed_tdim_and_calibration(self, tmp_path: Path) -> None:
        nsblk, npol, nchan = 8, 2, 4
        samples = np.arange(nsblk * npol * nchan, dtype=np.uint8).reshape(
            1,
            nsblk,
            npol,
            nchan,
        )
        scales = np.array([[2.0, 3, 4, 5], [1, 1, 1, 1]], dtype=np.float32)
        offsets = np.array([[10, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
        weights = np.array([1, 1, 0, 1], dtype=np.float32)
        path = tmp_path / "ppqq.sf"
        _write_search(
            path,
            samples,
            nbits=8,
            pol_type="AABB",
            chan0=200.0,
            chan_bw=-1.0,
            zero_off=1.0,
            scales=scales,
            offsets=offsets,
            weights=weights,
            permute=(1, 2, 0),
            tdim=(nsblk, nchan, npol),
        )
        with pfits.PFITSFile(str(path)) as src:
            raw = src.read_subint(0, scloffs=False, weights=False)
            np.testing.assert_array_equal(raw, samples[0])
            cal = src.read_subint_pol(0)
            expected = ((samples[0, :, 0] - 1.0) * scales[0] + offsets[0]) + (
                (samples[0, :, 1] - 1.0) * scales[1] + offsets[1]
            )
            expected *= weights
            np.testing.assert_allclose(cal, expected.astype(np.float32))

    def test_packed_bits_and_signed_samples(self, tmp_path: Path) -> None:
        nsblk, npol, nchan = 8, 2, 4
        bits4 = (np.arange(nsblk * npol * nchan) % 16).reshape(1, nsblk, npol, nchan)
        path4 = tmp_path / "four.sf"
        _write_search(
            path4,
            bits4,
            nbits=4,
            pol_type="AABB",
            chan0=10.0,
            chan_bw=1.0,
            zero_off=7.5,
        )
        with pfits.PFITSFile(str(path4)) as src:
            raw = src.read_subint(0, scloffs=False, weights=False)
            np.testing.assert_array_equal(raw, bits4[0])
            shifted = src.read_subint(0, scloffs=True, weights=False)
            np.testing.assert_allclose(shifted, bits4[0].astype(np.float32) - 7.5)

        bits2 = (np.arange(nsblk * npol * nchan) % 4).reshape(1, nsblk, npol, nchan)
        path2 = tmp_path / "two.sf"
        _write_search(
            path2,
            bits2,
            nbits=2,
            pol_type="AABB",
            chan0=10.0,
            chan_bw=1.0,
            zero_off=1.5,
        )
        lookup = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
        with pfits.PFITSFile(str(path2)) as src:
            got = src.read_subint(0, scloffs=True, weights=False)
            np.testing.assert_array_equal(got, lookup[bits2[0]])

        levels = np.zeros((1, 8, 1, 2), dtype=np.uint8)
        levels[0, :, 0, 0] = np.array([0, 1, 127, 128, 255, 0, 0, 0])
        path8 = tmp_path / "signed.sf"
        _write_search(
            path8,
            levels,
            nbits=8,
            pol_type="INTEN",
            chan0=1400.0,
            chan_bw=1.0,
            signint=1,
            zero_off=0.0,
        )
        with pfits.PFITSFile(str(path8)) as src:
            got = src.read_subint_pol(0, scloffs=False, weights=False)[:, 0]
            expect = np.array([0, 1, 127, -128, -1, 0, 0, 0], dtype=np.float32)
            np.testing.assert_array_equal(got, expect)

        wide = np.arange(4 * 2 * 2, dtype=np.int16).reshape(1, 4, 2, 2) - 3
        path16 = tmp_path / "wide.sf"
        _write_search(
            path16,
            wide,
            nbits=16,
            pol_type="AABB",
            chan0=800.0,
            chan_bw=-0.5,
            zero_off=99.0,
        )
        with pfits.PFITSFile(str(path16)) as src:
            got = src.read_subint(0, scloffs=True, weights=False)
            np.testing.assert_array_equal(got, wide[0])

    def test_unknown_pol_type_and_stokes(self, tmp_path: Path) -> None:
        samples = np.zeros((1, 4, 4, 2), dtype=np.uint8)
        samples[0, :, 0, 0] = 5
        samples[0, :, 1, 0] = 9
        path = tmp_path / "iquv.sf"
        _write_search(
            path,
            samples,
            nbits=8,
            pol_type="IQUV",
            chan0=1000.0,
            chan_bw=1.0,
        )
        with pfits.PFITSFile(str(path)) as src:
            assert src.sub_hdr.poln_state == "Stokes"
            got = src.read_subint_pol(0, scloffs=False, weights=False)
            np.testing.assert_array_equal(got[:, 0], 5)
        bad = tmp_path / "bad.sf"
        _write_search(
            bad,
            samples[:, :, :1, :],
            nbits=8,
            pol_type="NOPE",
            chan0=1000.0,
            chan_bw=1.0,
        )
        with pytest.raises(ValueError, match="NOPE"):
            pfits.PFITSFile(str(bad))

    def test_one_channel_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "one.sf"
        _write_search(
            path,
            np.zeros((1, 4, 1, 1), dtype=np.uint8),
            nbits=8,
            pol_type="INTEN",
            chan0=1400.0,
            chan_bw=-0.25,
        )
        with pytest.raises(ValueError, match="at least two channels"):
            pfits.PFITSFile(str(path))

    def test_short_weights_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "short.sf"
        _write_search(
            path,
            np.zeros((1, 4, 1, 4), dtype=np.uint8),
            nbits=8,
            pol_type="INTEN",
            chan0=100.0,
            chan_bw=1.0,
            weights=np.ones(2, dtype=np.float32),
        )
        with pfits.PFITSFile(str(path)) as src:
            with pytest.raises(ValueError, match="DAT_WTS"):
                src.read_weights(0)
            with pytest.raises(ValueError, match="DAT_WTS"):
                src.read_subint_pol(0)

    def test_irregular_frequencies_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "irregular.sf"
        _write_search(
            path,
            np.zeros((1, 4, 1, 4), dtype=np.uint8),
            nbits=8,
            pol_type="INTEN",
            chan0=100.0,
            chan_bw=1.0,
        )
        _assign_column(path, "DAT_FREQ", np.array([[100.0, 101.0, 104.0, 103.0]]))
        with pytest.raises(ValueError, match="not uniform"):
            pfits.PFITSFile(str(path))

    def test_row_gap_is_rejected(self, tmp_path: Path) -> None:
        nsblk = 4
        path = tmp_path / "gap.sf"
        _write_search(
            path,
            np.zeros((2, nsblk, 1, 2), dtype=np.uint8),
            nbits=8,
            pol_type="INTEN",
            chan0=100.0,
            chan_bw=1.0,
        )
        duration = nsblk * 1e-3
        centres = np.array([0.5 * duration, 0.5 * duration + 10.0])
        _assign_column(path, "OFFS_SUB", centres)
        with pytest.raises(ValueError, match="not contiguous"):
            pfits.PFITSFile(str(path))

    def test_fold_mode_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "fold.sf"
        _write_search(
            path,
            np.zeros((1, 4, 1, 2), dtype=np.uint8),
            nbits=8,
            pol_type="INTEN",
            chan0=100.0,
            chan_bw=1.0,
        )
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["OBS_MODE"] = "PSR"
        with pytest.raises(ValueError, match="SEARCH"):
            pfits.PFITSFile(str(path))

    def test_nstot_past_the_rows_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "long.sf"
        _write_search(
            path,
            np.zeros((1, 4, 1, 2), dtype=np.uint8),
            nbits=8,
            pol_type="INTEN",
            chan0=100.0,
            chan_bw=1.0,
        )
        with fits.open(path, mode="update") as hdul:
            hdul["SUBINT"].header["NSTOT"] = 100
        with pytest.raises(ValueError, match="NSTOT"):
            pfits.PFITSFile(str(path))

    def test_partial_nstot_is_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.sf"
        _write_search(
            path,
            np.zeros((1, 8, 1, 2), dtype=np.uint8),
            nbits=8,
            pol_type="INTEN",
            chan0=100.0,
            chan_bw=1.0,
        )
        with fits.open(path, mode="update") as hdul:
            hdul["SUBINT"].header["NSTOT"] = 3
        with pfits.PFITSFile(str(path)) as src:
            assert src.sub_hdr.nsamples == 3


def _assign_column(path: Path, name: str, values: np.ndarray) -> None:
    """Replace one SUBINT column and write the file again."""
    with fits.open(path) as hdul:
        table = hdul["SUBINT"].data
        table[name][:] = values
        copied = fits.HDUList([hdu.copy() for hdu in hdul])
    copied.writeto(path, overwrite=True)
    copied.close()


@pytest.mark.parametrize("tform", ["not-a-format", "1PJ(4)", "1QJ(4)", "3X"])
def test_invalid_fits_column_formats_are_rejected(tform: str) -> None:
    with pytest.raises(ValueError):
        pfits._parse_tform(tform)


@pytest.mark.parametrize(
    ("pol_type", "npol"), [("UNKNOWN", 2), ("INTEN", 4), ("IQUV", 2)]
)
def test_polarization_layout_must_match_its_product_count(
    pol_type: str, npol: int
) -> None:
    with pytest.raises(ValueError, match="POL_TYPE"):
        pfits.polarisation_state(pol_type, npol)


@pytest.mark.parametrize(
    "changes",
    [
        {"NBITS": 12},
        {"NCHAN": 1},
        {"NPOL": 0},
        {"NSBLK": 0},
        {"NSTOT": -1},
        {"POL_TYPE": "UNKNOWN"},
    ],
)
def test_malformed_search_headers_fail_before_sample_reads(
    tmp_path: Path, changes: dict
) -> None:
    path = tmp_path / "invalid.sf"
    _write_search(
        path,
        np.zeros((1, 4, 1, 4), dtype=np.uint8),
        nbits=8,
        pol_type="INTEN",
        chan0=100.0,
        chan_bw=1.0,
    )
    with fits.open(path, mode="update") as hdul:
        for key, value in changes.items():
            hdul["SUBINT"].header[key] = value
    with pytest.raises(ValueError):
        pfits.PFITSFile(str(path))


@pytest.mark.parametrize("frequencies", [[100.0, 101.0, 103.0, 104.0], [np.nan] * 4])
def test_nonuniform_or_missing_frequency_coordinates_are_rejected(
    tmp_path: Path, frequencies: list[float]
) -> None:
    path = tmp_path / "coordinates.sf"
    _write_search(
        path,
        np.zeros((1, 4, 1, 4), dtype=np.uint8),
        nbits=8,
        pol_type="INTEN",
        chan0=100.0,
        chan_bw=1.0,
    )
    _assign_column(path, "DAT_FREQ", np.asarray([frequencies]))
    with pytest.raises(ValueError, match="DAT_FREQ"):
        pfits.PFITSFile(str(path))


@pytest.mark.parametrize("duration", [0.0, -1.0, np.nan, np.inf])
def test_invalid_subintegration_durations_are_rejected(
    tmp_path: Path, duration: float
) -> None:
    path = tmp_path / "duration.sf"
    _write_search(
        path,
        np.zeros((2, 4, 1, 4), dtype=np.uint8),
        nbits=8,
        pol_type="INTEN",
        chan0=100.0,
        chan_bw=1.0,
    )
    _assign_column(path, "TSUBINT", np.full(2, duration))
    with pytest.raises(ValueError, match="TSUBINT"):
        pfits.PFITSFile(str(path))


def test_row_reader_rejects_invalid_indices_and_closed_files(tmp_path: Path) -> None:
    path = tmp_path / "rows.sf"
    samples = np.arange(16, dtype=np.uint8).reshape(1, 4, 1, 4)
    _write_search(path, samples, nbits=8, pol_type="INTEN", chan0=100.0, chan_bw=1.0)
    reader = pfits.PFITSFile(str(path))
    for row in [-1, 1]:
        with pytest.raises(IndexError):
            reader._table.row_slice(row, "DATA")
    for pol in [-2, 1]:
        with pytest.raises(ValueError, match="polarisation"):
            reader.read_detected(0, pol=pol)
    with pytest.raises(ValueError, match="NPOL"):
        reader.read_detected(0, pol=-1)
    reader.close()
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader._table.row_slice(0, "DATA")


@pytest.mark.parametrize("small_reads", [None, False, True])
def test_pfits_dedispersion_matches_channel_sample_gather(
    tmp_path: Path,
    small_reads: bool | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sigpyproc import readers

    path = tmp_path / "dispersed.sf"
    samples = (np.arange(2 * 32 * 4) % 200).astype(np.uint8).reshape(2, 32, 1, 4)
    _write_search(path, samples, nbits=8, pol_type="INTEN", chan0=100.0, chan_bw=10.0)
    monkeypatch.setattr(readers, "_BULK_BYTES", 1)
    monkeypatch.setattr(readers, "_STREAM_GULP_BYTES", 16)
    with PFITSReader(str(path)) as reader:
        delays = reader.header.get_dmdelays(0.02)
        block = reader.read_dedisp_block(10, 16, 0.02, small_reads=small_reads)
        raw = samples.reshape(64, 4)
        expected = np.stack(
            [raw[10 + delay + np.arange(16), chan] for chan, delay in enumerate(delays)]
        )
        np.testing.assert_array_equal(block.data, expected)
        assert block.header.tstart == reader.header.mjd_after_nsamps(10)
        assert block.dm == 0.02


def test_pfits_overlapping_plan_preserves_samples_and_indices(tmp_path: Path) -> None:
    path = tmp_path / "plan.sf"
    samples = np.arange(64, dtype=np.uint8).reshape(2, 8, 1, 4)
    _write_search(path, samples, nbits=8, pol_type="INTEN", chan0=100.0, chan_bw=1.0)
    with PFITSReader(str(path)) as reader:
        blocks = list(
            reader.read_plan(gulp=6, start=2, nsamps=11, skipback=2, quiet=True)
        )
        for index, (count, _block_id, values) in enumerate(blocks):
            np.testing.assert_array_equal(
                values.reshape(count, 4),
                samples.reshape(16, 4)[2 + index * 4 : 2 + index * 4 + count],
            )
        for kwargs in [{"gulp": 0}, {"start": -1}, {"gulp": 4, "skipback": 4}]:
            with pytest.raises(ValueError):
                list(reader.read_plan(**kwargs, quiet=True))
        with pytest.raises(ValueError):
            reader.read_block(0, 4, fch1=1000.0, nchans=1)


def _subint_card(path: Path, key: str, value: object | None) -> None:
    with fits.open(path, mode="update") as hdul:
        header = hdul["SUBINT"].header
        if value is None:
            del header[key]
        else:
            header[key] = value


def _data_tdim_key(path: Path) -> str:
    with fits.open(path) as hdul:
        header = hdul["SUBINT"].header
        for index in range(1, int(header["TFIELDS"]) + 1):
            if str(header[f"TTYPE{index}"]).strip() == "DATA":
                return f"TDIM{index}"
    msg = "DATA column is missing"
    raise AssertionError(msg)


def _drop_column(path: Path, name: str) -> None:
    kept_cards = (
        "POL_TYPE",
        "NPOL",
        "NBITS",
        "NCHAN",
        "NSBLK",
        "TBIN",
        "CHAN_BW",
        "SIGNINT",
        "ZERO_OFF",
        "NSUBOFFS",
        "NSTOT",
    )
    with fits.open(path) as hdul:
        table = hdul["SUBINT"]
        kept = [column for column in table.columns if column.name != name]
        replacement = fits.BinTableHDU.from_columns(kept, name="SUBINT")
        for key in kept_cards:
            if key in table.header:
                replacement.header[key] = table.header[key]
        copied = fits.HDUList([hdul[0].copy(), replacement])
    copied.writeto(path, overwrite=True)
    copied.close()


def _tiny_search(path: Path, *, nsub: int = 1) -> None:
    _write_search(
        path,
        np.zeros((nsub, 4, 1, 4), dtype=np.uint8),
        nbits=8,
        pol_type="INTEN",
        chan0=100.0,
        chan_bw=1.0,
    )


def test_search_header_reports_recorded_metadata(tmp_path: Path) -> None:
    path = tmp_path / "meta.sf"
    _tiny_search(path)
    with pfits.PFITSFile(str(path)) as src:
        assert src.filename == str(path)
        assert src.pri_hdr.observer == "test"
        assert src.pri_hdr.project_id == "test"
        assert src.pri_hdr.telescope == "Parkes"
        assert src.pri_hdr.obs_mode == "SEARCH"
        assert src.pri_hdr.chan_dm == 0
        assert src.sub_hdr.zero_off == 0
        assert src.sub_hdr.signint == 0
        assert src.sub_hdr.subint_offset == 0
        assert src.sub_hdr.nchans == 4
        assert src.sub_hdr.chan_bw == pytest.approx(1.0)
        assert src.sub_hdr.channel_offset == 0
        assert src.sub_hdr.tsubint == pytest.approx(0.004)
        assert src.sub_hdr.offs_sub == pytest.approx(0.002)
        empty = src.read_subints(0, 0)
        assert empty.shape == (0, 4)
        with pytest.raises(IndexError, match="subint range"):
            src.read_subints(1, 1)


def test_data_geometry_rejections() -> None:
    raw = np.zeros(8, dtype=np.uint8)
    with pytest.raises(ValueError, match="axes"):
        pfits.arrange_data_bytes(raw, (2, 4), nbits=8, nsblk=2, npol=1, nchan=2)
    with pytest.raises(ValueError, match="does not match"):
        pfits.arrange_data_bytes(
            np.zeros(8, dtype=np.uint8),
            (2, 2, 2),
            nbits=8,
            nsblk=4,
            npol=2,
            nchan=2,
        )
    with pytest.raises(ValueError, match="TDIM"):
        pfits.arrange_data_bytes(
            np.zeros(3, dtype=np.uint8),
            (2, 2, 2),
            nbits=8,
            nsblk=2,
            npol=1,
            nchan=2,
        )
    with pytest.raises(ValueError, match="element size"):
        pfits.arrange_data_bytes(
            np.zeros(4, dtype=np.uint8),
            (2, 1, 2),
            nbits=16,
            nsblk=2,
            npol=1,
            nchan=2,
        )
    with pytest.raises(ValueError, match="Empty TDIM"):
        pfits._parse_tdim("()")
    with pytest.raises(ValueError, match="Cannot parse TFORM"):
        pfits._parse_tform("123")


def _expect_open_error(path: Path, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pfits.PFITSFile(str(path))


def test_unrecognised_fits_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "type.sf"
    _tiny_search(path)
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["FITSTYPE"] = "OTHER"
    _expect_open_error(path, "not a PSRFITS")


def test_column_width_must_match_naxis1(tmp_path: Path) -> None:
    path = tmp_path / "naxis.sf"
    _tiny_search(path)
    _subint_card(path, "NAXIS1", 1)
    _expect_open_error(path, "NAXIS1")


def test_empty_data_dimensions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tdim.sf"
    _tiny_search(path)
    _subint_card(path, _data_tdim_key(path), "()")
    _expect_open_error(path, "Empty TDIM")


def test_packed_channels_must_fill_whole_bytes(tmp_path: Path) -> None:
    path = tmp_path / "pack.sf"
    _tiny_search(path)
    _subint_card(path, "NBITS", 2)
    _subint_card(path, "NCHAN", 2)
    _expect_open_error(path, "whole number")


def test_missing_data_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nodata.sf"
    _tiny_search(path)
    _drop_column(path, "DATA")
    _expect_open_error(path, "no DATA")


def test_missing_weights_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "noweights.sf"
    _tiny_search(path)
    _drop_column(path, "DAT_WTS")
    with pfits.PFITSFile(str(path)) as src, pytest.raises(ValueError, match="DAT_WTS"):
        src.read_weights(0)


def _replace_column(path: Path, column: fits.Column) -> None:
    kept_cards = (
        "POL_TYPE",
        "NPOL",
        "NBITS",
        "NCHAN",
        "NSBLK",
        "TBIN",
        "CHAN_BW",
        "SIGNINT",
        "ZERO_OFF",
        "NSUBOFFS",
        "NSTOT",
    )
    with fits.open(path) as hdul:
        table = hdul["SUBINT"]
        kept = [old if old.name != column.name else column for old in table.columns]
        replacement = fits.BinTableHDU.from_columns(kept, name="SUBINT")
        for key in kept_cards:
            if key in table.header:
                replacement.header[key] = table.header[key]
        copied = fits.HDUList([hdul[0].copy(), replacement])
    copied.writeto(path, overwrite=True)
    copied.close()


@pytest.mark.parametrize(
    ("name", "match", "reader_name"),
    [
        ("DAT_SCL", "DAT_SCL", "read_scales"),
        ("DAT_OFFS", "DAT_OFFS", "read_offsets"),
    ],
)
def test_short_calibration_columns_are_rejected(
    tmp_path: Path, name: str, match: str, reader_name: str
) -> None:
    path = tmp_path / f"{name}.sf"
    _tiny_search(path)
    _replace_column(
        path,
        fits.Column(name=name, format="1E", array=np.ones(1, dtype=np.float32)),
    )
    with pfits.PFITSFile(str(path)) as src, pytest.raises(ValueError, match=match):
        getattr(src, reader_name)(0)


def test_split_files_without_row_centres_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "centres.sf"
    _tiny_search(path, nsub=2)
    _drop_column(path, "OFFS_SUB")
    _expect_open_error(path, "OFFS_SUB")


def test_missing_or_nonpositive_sample_interval_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sf"
    _tiny_search(missing, nsub=2)
    _subint_card(missing, "TBIN", None)
    _expect_open_error(missing, "no TBIN")
    zero = tmp_path / "zero.sf"
    _tiny_search(zero, nsub=2)
    _subint_card(zero, "TBIN", 0.0)
    _expect_open_error(zero, "TBIN")


def test_missing_sample_count_uses_the_table_capacity(tmp_path: Path) -> None:
    path = tmp_path / "nstot.sf"
    _tiny_search(path)
    _subint_card(path, "NSTOT", None)
    with pfits.PFITSFile(str(path)) as src:
        assert src.sub_hdr.nsamples == src.sub_hdr.subint_samples * src.sub_hdr.nsubint


def test_frequency_table_must_agree_across_rows_and_with_the_header(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    shifted = tmp_path / "shifted.sf"
    _tiny_search(shifted, nsub=2)
    _assign_column(
        shifted,
        "DAT_FREQ",
        np.array([[100.0, 101.0, 102.0, 103.0], [110.0, 111.0, 112.0, 113.0]]),
    )
    _expect_open_error(shifted, "differs from subint 0")

    short = tmp_path / "shortfreq.sf"
    _tiny_search(short)
    _subint_card(short, "NCHAN", 8)
    _expect_open_error(short, "DAT_FREQ")

    implicit = tmp_path / "notdim.sf"
    _tiny_search(implicit)
    _subint_card(implicit, _data_tdim_key(implicit), None)
    with pfits.PFITSFile(str(implicit)) as src:
        assert src.sub_hdr.nchans == 4

    no_bw = tmp_path / "nobw.sf"
    _tiny_search(no_bw)
    _subint_card(no_bw, "CHAN_BW", None)
    with pfits.PFITSFile(str(no_bw)) as src:
        assert src.sub_hdr.nchans == 4

    disagree = tmp_path / "disagree.sf"
    _tiny_search(disagree)
    _subint_card(disagree, "CHAN_BW", 50.0)
    with (
        caplog.at_level(logging.WARNING, logger="sigpyproc.io.pfits"),
        pfits.PFITSFile(str(disagree)) as src,
    ):
        assert src.sub_hdr.chan_bw == pytest.approx(50.0)
    assert "CHAN_BW" in caplog.text


def test_nonfinite_row_centres_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "centres.sf"
    _tiny_search(path, nsub=2)
    _assign_column(path, "OFFS_SUB", np.array([np.nan, 0.002]))
    _expect_open_error(path, "OFFS_SUB")


def test_file_without_a_subint_table_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "primary.sf"
    primary = fits.PrimaryHDU(header=_primary_cards(obs_bw=4.0, nchan=4, fcenter=101.5))
    fits.HDUList([primary]).writeto(path)
    with pytest.raises(ValueError, match="SUBINT"):
        pfits.PFITSFile(str(path))
