from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sigpyproc.apps.spp_digifil import main as digifil_cli
from sigpyproc.io import pfits, sigproc
from sigpyproc.io.digifil import (
    MEGABYTE,
    DigifilConfig,
    Rescaler,
    _detected_pol,
    _rechunk,
    _validate_pfits_polarisation,
    default_gulp,
    digifil,
    dspsr_block_samples,
)
from sigpyproc.io.fileio import FileWriter
from sigpyproc.kernels.digitizer import (
    apply_scale_offset,
    fscrunch,
    pscrunch_intensity,
    quantise,
    quantise_packed,
    tscrunch,
)
from sigpyproc.readers import FilReader

NCHAN_8BIT = 832
NSAMP_8BIT = 4096
TSAMP_8BIT = 0.000512


def _reference_rescale(
    blocks: list[np.ndarray],
    nsample: int,
    *,
    constant: bool,
) -> np.ndarray:
    """Sample-by-sample transcription of dspsr Rescale over block boundaries."""
    nchan = blocks[0].shape[1]
    first = True
    isample = 0
    sums = np.zeros(nchan, dtype=np.float64)
    sumsq = np.zeros(nchan, dtype=np.float64)
    offset = np.zeros(nchan, dtype=np.float32)
    scale = np.ones(nchan, dtype=np.float32)
    outs: list[np.ndarray] = []

    def apply(sample: np.ndarray) -> np.ndarray:
        return ((sample + offset) * scale).astype(np.float32)

    for block in blocks:
        pending: list[np.ndarray] = []
        for j, sample in enumerate(block):
            if constant and not first:
                outs.append(apply(sample))
                continue
            if first or not constant:
                sums += sample.astype(np.float64)
                sumsq += sample.astype(np.float64) ** 2
                isample += 1
            pending.append(sample)
            if isample == nsample or (first and j == block.shape[0] - 1):
                nval_recip = 1.0 / isample
                mean = sums * nval_recip
                variance = sumsq * nval_recip - mean * mean
                positive = variance > 0
                offset = (-mean).astype(np.float32)
                scale = np.where(
                    positive,
                    1.0 / np.sqrt(np.where(positive, variance, 1.0)),
                    1.0,
                ).astype(np.float32)
                outs.extend(apply(p) for p in pending)
                pending = []
                if isample == nsample:
                    first = False
                    isample = 0
                    sums[:] = 0
                    sumsq[:] = 0
        outs.extend(apply(p) for p in pending)
    return np.array(outs, dtype=np.float32)


def _reference_digitise(
    data: np.ndarray,
    mean: float,
    scale: float,
    nbits: int,
) -> np.ndarray:
    """Transcription of SigProcDigitizer's quantisation with a fused multiply-add.

    dspsr computes ``x * scale + mean`` with a single rounding to float32 (the
    multiply-add is contracted by the compiler), then adds 0.5 in double.
    """
    fused = (data.astype(np.float64) * np.float64(scale) + np.float64(mean)).astype(
        np.float32,
    )
    value = np.trunc(fused.astype(np.float64) + 0.5)
    return np.clip(value, 0, 2**nbits - 1).astype(np.int64)


def _raw_8bit(filename: str) -> np.ndarray:
    reader = FilReader(filename)
    return reader.read_block(0, reader.header.nsamples).data.T.astype(np.float32)


def _out(tmpdir: str, name: str) -> str:
    return str(Path(tmpdir) / name)


class TestKernels:
    def test_fscrunch(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(size=(5, 2, 12)).astype(np.float32)
        out = fscrunch(data, 3)
        ref = np.empty((5, 2, 4), dtype=np.float32)
        for c in range(4):
            ref[:, :, c] = data[:, :, 3 * c] + data[:, :, 3 * c + 1]
            ref[:, :, c] = (ref[:, :, c] + data[:, :, 3 * c + 2]).astype(np.float32)
        np.testing.assert_array_equal(out, ref)

    def test_tscrunch(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.normal(size=(9, 1, 3)).astype(np.float32)
        out = tscrunch(data, 4)
        assert out.shape == (2, 1, 3)
        ref = np.stack([data[0:4].sum(axis=0), data[4:8].sum(axis=0)])
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_pscrunch_intensity(self) -> None:
        data = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        np.testing.assert_array_equal(
            pscrunch_intensity(data),
            data[:, 0, :] + data[:, 1, :],
        )

    def test_quantise_matches_reference(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.normal(0, 3, size=(64, 7)).astype(np.float32)
        out = np.empty(data.shape, dtype=np.uint8)
        quantise(
            data,
            False,
            np.full(data.shape[0], 127.5, dtype=np.float32),
            np.float32(21.25),
            255,
            out,
        )
        np.testing.assert_array_equal(
            out,
            _reference_digitise(data, 127.5, 21.25, 8),
        )

    def test_quantise_flip(self) -> None:
        data = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        out = np.empty(data.shape, dtype=np.uint8)
        quantise(data, True, np.zeros(1, np.float32), np.float32(1.0), 255, out)
        np.testing.assert_array_equal(out, [[3, 2, 1, 0]])

    def test_quantise_truncates_towards_zero_and_clips(self) -> None:
        data = np.array([[-0.7, -0.4, 0.49, 0.5, 1e9]], dtype=np.float32)
        out = np.empty(data.shape, dtype=np.uint8)
        quantise(data, False, np.zeros(1, np.float32), np.float32(1.0), 3, out)
        np.testing.assert_array_equal(out, [[0, 0, 0, 1, 3]])

    def test_quantise_packed_layout(self) -> None:
        data = np.array([[1.0, 2.0, 3.0, 0.0], [3.0, 3.0, 0.0, 1.0]], dtype=np.float32)
        packed = np.zeros(2, dtype=np.uint8)
        quantise_packed(
            data,
            False,
            np.zeros(2, np.float32),
            np.float32(1.0),
            3,
            2,
            packed,
        )
        expected = [
            1 | (2 << 2) | (3 << 4) | (0 << 6),
            3 | (3 << 2) | (0 << 4) | (1 << 6),
        ]
        np.testing.assert_array_equal(packed, expected)

    def test_quantise_packed_flip(self) -> None:
        data = np.array([[1.0, 2.0, 3.0, 0.0]], dtype=np.float32)
        packed = np.zeros(1, dtype=np.uint8)
        quantise_packed(
            data,
            True,
            np.zeros(1, np.float32),
            np.float32(1.0),
            3,
            2,
            packed,
        )
        assert packed[0] == 0 | (3 << 2) | (2 << 4) | (1 << 6)

    def test_apply_scale_offset(self) -> None:
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        out = np.empty_like(data)
        apply_scale_offset(
            data,
            np.array([-1.0, 0.5], dtype=np.float32),
            np.array([2.0, 4.0], dtype=np.float32),
            out,
        )
        np.testing.assert_array_equal(out, [[0.0, 10.0]])


class TestRescaler:
    def test_single_block_partial_window(self) -> None:
        rng = np.random.default_rng(4)
        data = rng.normal(5, 2, size=(500, 4)).astype(np.float32)
        rescaler = Rescaler(4, nsample=10_000, constant=False)
        out = rescaler.process(data)
        mean = data.astype(np.float64).mean(axis=0)
        std = data.astype(np.float64).std(axis=0)
        np.testing.assert_allclose(out, (data - mean) / std, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("constant", [False, True])
    def test_multi_block_matches_reference(self, constant: bool) -> None:
        rng = np.random.default_rng(5)
        sizes = [300, 50, 700, 1000, 20, 650]
        blocks = [rng.normal(10, 3, size=(n, 6)).astype(np.float32) for n in sizes]
        rescaler = Rescaler(6, nsample=900, constant=constant)
        got = np.concatenate([rescaler.process(b) for b in blocks])
        ref = _reference_rescale(blocks, 900, constant=constant)
        np.testing.assert_array_equal(got, ref)

    def test_rechunk(self) -> None:
        chunks = [np.full((n, 2, 1), i, dtype=np.float32) for i, n in enumerate([5, 7])]
        sizes = [b.shape[0] for b in _rechunk(chunks, 4)]
        assert sizes == [4, 4, 4]


class TestDigifilDriver:
    def test_redigitise_default_matches_reference(
        self,
        filfile_8bit_1: str,
        tmpdir: str,
    ) -> None:
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "default.fil"),
            DigifilConfig(nbits=8),
        )
        got = FilReader(out).read_block(0, NSAMP_8BIT).data.T
        raw = _raw_8bit(filfile_8bit_1)
        nsample = int(10.0 * (1.0 / TSAMP_8BIT))
        norm = _reference_rescale([raw], nsample, constant=False)
        expected = _reference_digitise(norm, 127.5, 127.5 / 6, 8)
        np.testing.assert_array_equal(got, expected.astype(np.uint8))

    def test_no_rescale_scale_fac(self, filfile_8bit_1: str, tmpdir: str) -> None:
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "i0.fil"),
            DigifilConfig(nbits=8, rescale_seconds=0, scale_fac=2.0),
        )
        got = FilReader(out).read_block(0, NSAMP_8BIT).data.T
        raw = _raw_8bit(filfile_8bit_1)
        expected = _reference_digitise(raw, 0.0, 0.5, 8)
        np.testing.assert_array_equal(got, expected.astype(np.uint8))

    def test_float_output(self, filfile_8bit_1: str, tmpdir: str) -> None:
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "float.fil"),
            DigifilConfig(nbits=-32),
        )
        reader = FilReader(out)
        assert reader.header.nbits == 32
        got = reader.read_block(0, NSAMP_8BIT).data.T
        raw = _raw_8bit(filfile_8bit_1)
        nsample = int(10.0 * (1.0 / TSAMP_8BIT))
        norm = _reference_rescale([raw], nsample, constant=False)
        np.testing.assert_array_equal(got, norm)

    def test_scrunch_header_and_data(self, filfile_8bit_1: str, tmpdir: str) -> None:
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "scrunch.fil"),
            DigifilConfig(
                nbits=8,
                rescale_seconds=0,
                fscrunch_factor=2,
                tscrunch_factor=2,
            ),
        )
        src = FilReader(filfile_8bit_1).header
        hdr = FilReader(out).header
        assert hdr.nchans == src.nchans // 2
        assert hdr.foff == pytest.approx(2 * src.foff)
        assert hdr.tsamp == pytest.approx(2 * src.tsamp)
        raw = _raw_8bit(filfile_8bit_1)
        summed = raw.reshape(NSAMP_8BIT, -1, 2).sum(axis=-1)
        summed = summed.reshape(NSAMP_8BIT // 2, 2, -1).sum(axis=1)
        expected = _reference_digitise(summed, 0.0, 0.25, 8)
        got = FilReader(out).read_block(0, hdr.nsamples).data.T
        np.testing.assert_array_equal(got, expected.astype(np.uint8))

    def test_output_header_adds_no_optional_keys(
        self,
        filfile_8bit_1: str,
        tmpdir: str,
    ) -> None:
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "meta.fil"),
            DigifilConfig(nbits=8, rescale_seconds=0),
        )
        optional = {
            "barycentric",
            "pulsarcentric",
            "signed",
            "refdm",
            "period",
            "az_start",
            "za_start",
            "ibeam",
            "nbeams",
            "rawdatafile",
        }
        src_keys = set(sigproc.parse_header(filfile_8bit_1)) & optional
        out_keys = set(sigproc.parse_header(out)) & optional
        assert out_keys <= src_keys

    def test_output_keeps_existing_rawdatafile(
        self,
        filfile_8bit_1: str,
        tmpdir: str,
    ) -> None:
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "rawname.fil"),
            DigifilConfig(nbits=8, rescale_seconds=0),
        )
        src_name = sigproc.parse_header(filfile_8bit_1).get("rawdatafile")
        assert sigproc.parse_header(out).get("rawdatafile") == src_name

    def test_positive_foff_is_flipped(self, filfile_8bit_1: str, tmpdir: str) -> None:
        src = FilReader(filfile_8bit_1).header
        raw = FilReader(filfile_8bit_1).read_block(0, src.nsamples).data
        flipped_path = _out(tmpdir, "flipped_in.fil")
        flipped_hdr = src.new_header(
            {
                "foff": -src.foff,
                "fch1": src.fch1 + src.foff * (src.nchans - 1),
            },
        )
        writer = FileWriter(flipped_path, mode="w+", nbits=8)
        writer.write(sigproc.encode_header(flipped_hdr.to_sigproc()))
        writer.write(np.ascontiguousarray(raw[::-1].T).astype(np.uint8).tobytes())
        writer.close()

        cfg = DigifilConfig(nbits=8, rescale_seconds=0)
        ref = digifil(filfile_8bit_1, _out(tmpdir, "ref.fil"), cfg)
        flip = digifil(flipped_path, _out(tmpdir, "flip.fil"), cfg)
        np.testing.assert_array_equal(
            FilReader(ref).read_block(0, NSAMP_8BIT).data,
            FilReader(flip).read_block(0, NSAMP_8BIT).data,
        )
        assert FilReader(flip).header.foff < 0

    def test_subbyte_output_is_lsb_first(
        self,
        filfile_8bit_1: str,
        tmpdir: str,
    ) -> None:
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "two_bit.fil"),
            DigifilConfig(nbits=2, rescale_seconds=0),
        )
        raw = _raw_8bit(filfile_8bit_1)
        expected = _reference_digitise(raw, 0.0, 1.0, 2).ravel()
        spb = 4
        packed = np.zeros(expected.size // spb, dtype=np.uint8)
        for k, val in enumerate(expected):
            packed[k // spb] |= np.uint8(val << ((k % spb) * 2))
        data = np.fromfile(out, dtype=np.uint8)[-packed.size :]
        np.testing.assert_array_equal(data, packed)

    def test_multi_block_rescale_matches_reference(
        self,
        filfile_8bit_1: str,
        tmpdir: str,
    ) -> None:
        """Rescale window spans dspsr-style blocks of 1000 samples."""
        rescale_seconds = 0.7685
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "blocks.fil"),
            DigifilConfig(
                nbits=8,
                gulp=1000,
                rescale_seconds=rescale_seconds,
            ),
        )
        got = FilReader(out).read_block(0, NSAMP_8BIT).data.T
        raw = _raw_8bit(filfile_8bit_1)
        blocks = [raw[i : i + 1000] for i in range(0, raw.shape[0], 1000)]
        nsample = int(rescale_seconds * (1.0 / TSAMP_8BIT))
        norm = _reference_rescale(blocks, nsample, constant=False)
        expected = _reference_digitise(norm, 127.5, 127.5 / 6, 8)
        np.testing.assert_array_equal(got, expected.astype(np.uint8))

    def test_tscrunch_carry_across_blocks(
        self,
        filfile_8bit_1: str,
        tmpdir: str,
    ) -> None:
        """Samples that do not fill a time-scrunch group carry to the next block."""
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "carry.fil"),
            DigifilConfig(
                nbits=-32,
                gulp=1000,
                rescale_seconds=0,
                tscrunch_factor=3,
            ),
        )
        raw = _raw_8bit(filfile_8bit_1)
        nout = NSAMP_8BIT // 3
        summed = raw[: nout * 3].reshape(nout, 3, NCHAN_8BIT).sum(axis=1) / 3.0
        got = FilReader(out).read_block(0, nout).data.T
        np.testing.assert_allclose(got, summed.astype(np.float32), rtol=1e-6)

    def test_psrfits_input(self, fitsfile_4bit: str, tmpdir: str) -> None:
        out = digifil(
            fitsfile_4bit,
            _out(tmpdir, "pfits.fil"),
            DigifilConfig(nbits=8, rescale_seconds=0),
        )
        hdr = FilReader(out).header
        assert hdr.nbits == 8
        assert hdr.foff < 0

    def test_npol_other_than_stokes_i_is_rejected(self) -> None:
        for npol in (2, 3, 4):
            with pytest.raises(NotImplementedError, match=f"-d {npol}"):
                DigifilConfig(npol=npol)

    def test_parkes_float_matches_dspsr(self, fitsfile_4bit: str, tmpdir: str) -> None:
        dspsr = shutil.which("digifil")
        if dspsr is None:
            pytest.skip("dspsr digifil is not installed")
        ref = Path(tmpdir) / "dspsr_ref"
        subprocess.run(  # noqa: S603
            [
                dspsr,
                "-threads",
                "1",
                "-q",
                "-b-32",
                "-I",
                "0",
                "-d",
                "1",
                "-o",
                str(ref),
                fitsfile_4bit,
            ],
            check=True,
        )
        out = digifil(
            fitsfile_4bit,
            _out(tmpdir, "spp_ref.fil"),
            DigifilConfig(nbits=-32, rescale_seconds=0),
        )
        ref_reader = FilReader(str(ref))
        nsamples = ref_reader.header.nsamples
        got = FilReader(out).read_block(0, nsamples).data
        np.testing.assert_array_equal(got, ref_reader.read_block(0, nsamples).data)

    def test_uwl_8bit_tscrunch_matches_dspsr_reference(self, tmpdir: str) -> None:
        uwl = Path("tmp_files/uwl_210405_055326_60.sf")
        ref_path = Path("tmp_files/dspsr_ref.fil")
        if not uwl.is_file() or not ref_path.is_file():
            pytest.skip("UWL reference files are not in tmp_files")
        out = digifil(
            str(uwl),
            _out(tmpdir, "uwl.fil"),
            DigifilConfig(nbits=8, rescale_seconds=0, tscrunch_factor=4),
        )
        ref = FilReader(str(ref_path))
        got = FilReader(out)
        assert got.header.nchans == ref.header.nchans
        assert got.header.nbits == ref.header.nbits
        assert got.header.nsamples == ref.header.nsamples
        assert got.header.tsamp == pytest.approx(ref.header.tsamp)
        assert got.header.foff == pytest.approx(ref.header.foff)
        np.testing.assert_array_equal(
            got.read_block(0, ref.header.nsamples).data,
            ref.read_block(0, ref.header.nsamples).data,
        )

    def test_psrfits_rejects_unknown_state(
        self,
        fitsfile_4bit: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sub = pfits.SubintHdr(fitsfile_4bit)
        monkeypatch.setattr(type(sub), "poln_state", property(lambda _self: "Mystery"))
        assert _validate_pfits_polarisation(sub) == "Mystery"
        with pytest.raises(NotImplementedError, match="Mystery"):
            _detected_pol(sub, DigifilConfig(nbits=8, rescale_seconds=0))

    def test_psrfits_digifil_rejects_unknown_state(
        self,
        fitsfile_4bit: str,
        tmpdir: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            pfits.SubintHdr,
            "poln_state",
            property(lambda _self: "Mystery"),
        )
        with pytest.raises(NotImplementedError, match="Mystery"):
            digifil(
                fitsfile_4bit,
                _out(tmpdir, "bad.fil"),
                DigifilConfig(nbits=8, rescale_seconds=0),
            )

    def test_reader_decodes_lsb_subbyte_output(
        self,
        filfile_8bit_1: str,
        tmpdir: str,
    ) -> None:
        """FilReader reads the LSB-first 2-bit output written by digifil (as dspsr)."""
        out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "lsb_read.fil"),
            DigifilConfig(nbits=2, rescale_seconds=0),
        )
        raw = _raw_8bit(filfile_8bit_1)
        expected = _reference_digitise(raw, 0.0, 1.0, 2)
        got = FilReader(out).read_block(0, NSAMP_8BIT).data.T
        np.testing.assert_array_equal(got, expected)

    def test_requantize_matches_digifil(self, filfile_8bit_1: str, tmpdir: str) -> None:
        fil = FilReader(filfile_8bit_1)
        req = fil.requantize(nbits_out=4, outfile_name=_out(tmpdir, "req4.fil"))
        dig = digifil(
            filfile_8bit_1,
            _out(tmpdir, "dig4.fil"),
            DigifilConfig(nbits=4),
        )
        np.testing.assert_array_equal(
            FilReader(req).read_block(0, NSAMP_8BIT).data,
            FilReader(dig).read_block(0, NSAMP_8BIT).data,
        )

    def test_invalid_config(self, filfile_8bit_1: str, tmpdir: str) -> None:
        with pytest.raises(ValueError, match="nbits"):
            digifil(
                filfile_8bit_1,
                _out(tmpdir, "bad.fil"),
                DigifilConfig(nbits=32),
            )
        with pytest.raises(ValueError, match="gulp"):
            digifil(
                filfile_8bit_1,
                _out(tmpdir, "tiny.fil"),
                DigifilConfig(gulp=0),
            )


class TestDspsrBlocks:
    @pytest.mark.parametrize(
        ("ram_mb", "expected"),
        [
            (1, 32),
            (2, 64),
            (4, 128),
            (8, 252),
            (16, 504),
            (64, 2016),
            (256, 8068),
            (512, 16132),
            (1024, 32264),
        ],
    )
    def test_sigproc_block_follows_dspsr_u_sweep(
        self, ram_mb: int, expected: int
    ) -> None:
        """Values observed from dspsr ``-U`` for an 8-bit, 6656-channel filterbank."""
        block = dspsr_block_samples(
            8,
            1,
            6656,
            copies=1,
            resolution=4,
            max_ram_bytes=ram_mb * MEGABYTE,
        )
        assert block == expected

    def test_psrfits_block_rounds_to_subint(self) -> None:
        block = dspsr_block_samples(
            8,
            4,
            6656,
            copies=2,
            resolution=8192,
            max_ram_bytes=256 * MEGABYTE,
        )
        assert block == 8192

    def test_default_gulp_is_dspsr_block(
        self, filfile_8bit_1: str, tmpdir: str
    ) -> None:
        nchan = NCHAN_8BIT
        gulp = default_gulp(filfile_8bit_1)
        expected = dspsr_block_samples(
            8,
            1,
            nchan,
            copies=1,
            resolution=4,
            max_ram_bytes=256 * MEGABYTE,
        )
        assert gulp == expected
        assert default_gulp(filfile_8bit_1, max_ram_mb=1) < gulp
        default_out = digifil(filfile_8bit_1, _out(tmpdir, "d.fil"), DigifilConfig())
        gulp_out = digifil(
            filfile_8bit_1,
            _out(tmpdir, "g.fil"),
            DigifilConfig(gulp=gulp),
        )
        np.testing.assert_array_equal(
            FilReader(default_out).read_block(0, NSAMP_8BIT).data,
            FilReader(gulp_out).read_block(0, NSAMP_8BIT).data,
        )


class TestDigifilCli:
    def test_cli(self, filfile_8bit_1: str, tmpdir: str) -> None:
        outfile = _out(tmpdir, "cli.fil")
        runner = CliRunner()
        result = runner.invoke(
            digifil_cli,
            [filfile_8bit_1, "-b", "4", "-I", "0", "-t", "2", "-o", outfile],
        )
        assert result.exit_code == 0, result.output
        assert Path(outfile).exists()
        assert FilReader(outfile).header.nbits == 4

    def test_cli_bad_nbits(self, filfile_8bit_1: str) -> None:
        runner = CliRunner()
        result = runner.invoke(digifil_cli, [filfile_8bit_1, "-b", "3"])
        assert result.exit_code != 0
