"""Output alias refusal and signed 8-bit round trips."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from click.testing import CliRunner

from sigpyproc.apps import spp_clean
from sigpyproc.header import Header
from sigpyproc.io.sigproc import FileInfo, StreamInfo
from sigpyproc.readers import FilReader
from sigpyproc.simulation.furby import FurbyGenerator, PulseParams
from sigpyproc.simulation.inject import Injector, inject_file

if TYPE_CHECKING:
    from pathlib import Path


def _header(path: Path, *, signed: bool, nsamples: int, nchans: int = 1) -> Header:
    return Header(
        filename=str(path),
        data_type="filterbank",
        nchans=nchans,
        foff=-1.0,
        fch1=1400.0,
        nbits=8,
        signed=signed,
        tsamp=1e-3,
        tstart=50000.0,
        nsamples=nsamples,
    )


def _write_signed(path: Path, samples: np.ndarray) -> None:
    header = _header(path, signed=True, nsamples=samples.size)
    with header.prep_outfile(str(path), sources=()) as writer:
        writer.cwrite(samples)


def test_signed_extract_preserves_negatives(tmp_path: Path) -> None:
    samples = np.array([-100, -1, -128, 0, 127, -50], dtype=np.int8)
    src = tmp_path / "signed.fil"
    _write_signed(src, samples)
    copied = tmp_path / "copy.fil"
    with FilReader(src) as fil:
        fil.extract_samps(0, samples.size, outfile_name=str(copied))
    with FilReader(copied) as out:
        assert out.header.signed is True
        assert out.header.nbits == 8
        got = out.read_block(0, samples.size).data.ravel()
    np.testing.assert_array_equal(got, samples.astype(np.float32))


def test_signed_clip_and_unsigned_clip(tmp_path: Path) -> None:
    signed_path = tmp_path / "clip_s.fil"
    values = np.array([-200.0, 200.0, -128.0, 127.0], dtype=np.float32)
    with _header(signed_path, signed=True, nsamples=4).prep_outfile(
        str(signed_path),
        sources=(),
    ) as writer:
        writer.cwrite(values)
    with FilReader(signed_path) as fil:
        got = fil.read_block(0, 4).data.ravel()
    np.testing.assert_array_equal(got, [-128, 127, -128, 127])

    unsigned_path = tmp_path / "clip_u.fil"
    with _header(unsigned_path, signed=False, nsamples=3).prep_outfile(
        str(unsigned_path),
        sources=(),
    ) as writer:
        writer.cwrite(np.array([-1.0, 128.0, 300.0]))
    with FilReader(unsigned_path) as fil:
        assert fil.header.signed is False
        got_u = fil.read_block(0, 3).data.ravel()
    np.testing.assert_array_equal(got_u, [0, 128, 255])


def test_wider_output_clears_signed_flag(tmp_path: Path) -> None:
    src = tmp_path / "s8.fil"
    _write_signed(src, np.array([1, -2, 3, -4], dtype=np.int8))
    out = tmp_path / "f32.fil"
    with (
        FilReader(src) as fil,
        fil.header.prep_outfile(str(out), nbits=32) as writer,
    ):
        writer.cwrite(np.array([1.5, -2.5, 3.5, -4.5], dtype=np.float32))
    with FilReader(out) as fil:
        assert fil.header.nbits == 32
        assert fil.header.signed is False
        np.testing.assert_allclose(
            fil.read_block(0, 4).data.ravel(),
            [1.5, -2.5, 3.5, -4.5],
        )


def test_prep_outfile_does_not_mutate_updates(tmp_path: Path) -> None:
    path = tmp_path / "out.fil"
    updates = {"nsamples": 2}
    header = _header(path, signed=False, nsamples=2)
    with header.prep_outfile(str(path), updates=updates, sources=()):
        pass
    assert updates == {"nsamples": 2}


def test_identical_output_does_not_truncate(tmp_path: Path) -> None:
    src = tmp_path / "in.fil"
    _write_signed(src, np.array([-5, 6, -7, 8], dtype=np.int8))
    before = src.read_bytes()
    with FilReader(src) as fil, pytest.raises(ValueError, match="same file"):
        fil.extract_samps(0, 4, outfile_name=str(src))
    assert src.read_bytes() == before


def test_symlink_and_hardlink_are_refused(tmp_path: Path) -> None:
    src = tmp_path / "in.fil"
    _write_signed(src, np.array([1, -1, 2, -2], dtype=np.int8))
    before = src.read_bytes()
    link = tmp_path / "link.fil"
    link.symlink_to(src)
    hard = tmp_path / "hard.fil"
    os.link(src, hard)
    with FilReader(src) as fil:
        with pytest.raises(ValueError, match="same file"):
            fil.extract_samps(0, 4, outfile_name=str(link))
        with pytest.raises(ValueError, match="same file"):
            fil.extract_samps(0, 4, outfile_name=str(hard))
    assert src.read_bytes() == before


def test_stream_member_is_protected(tmp_path: Path) -> None:
    first = tmp_path / "a.fil"
    second = tmp_path / "b.fil"
    first.write_bytes(b"first-file")
    second.write_bytes(b"second-file-bytes")
    info = StreamInfo(
        entries=[
            FileInfo(
                filename=str(first),
                hdrlen=1,
                datalen=1,
                nsamples=1,
                tstart=50000.0,
                tsamp=1e-3,
            ),
            FileInfo(
                filename=str(second),
                hdrlen=1,
                datalen=1,
                nsamples=1,
                tstart=50000.0,
                tsamp=1e-3,
            ),
        ],
    )
    header = Header(
        filename=str(first),
        data_type="filterbank",
        nchans=1,
        foff=-1.0,
        fch1=1400.0,
        nbits=8,
        tsamp=1e-3,
        tstart=50000.0,
        nsamples=1,
        stream_info=info,
    )
    before = second.read_bytes()
    with pytest.raises(ValueError, match="same file"):
        header.prep_outfile(str(second))
    assert second.read_bytes() == before


def test_unrelated_existing_output_is_replaced(tmp_path: Path) -> None:
    src = tmp_path / "in.fil"
    _write_signed(src, np.array([-3, 4, -5, 6], dtype=np.int8))
    other = tmp_path / "other.fil"
    other.write_bytes(b"sentinel-not-a-filterbank")
    with FilReader(src) as fil:
        fil.extract_samps(0, 4, outfile_name=str(other))
    assert other.read_bytes() != b"sentinel-not-a-filterbank"
    with FilReader(other) as out:
        np.testing.assert_array_equal(out.read_block(0, 4).data.ravel(), [-3, 4, -5, 6])


def test_inject_refuses_same_path(tmp_path: Path) -> None:
    src = tmp_path / "in.fil"
    header = Header(
        filename=str(src),
        data_type="filterbank",
        nchans=8,
        foff=-1.0,
        fch1=1400.0,
        nbits=8,
        signed=True,
        tsamp=1e-3,
        tstart=50000.0,
        nsamples=64,
    )
    with header.prep_outfile(str(src), sources=()) as writer:
        writer.cwrite(np.zeros(64 * 8, dtype=np.float32))
    before = src.read_bytes()
    pulse = FurbyGenerator(
        header,
        PulseParams(dm=0.0, snr=5.0, width=1e-3, dmsmear=False),
        seed=1,
    ).generate()
    injector = Injector.from_furbies([pulse], [4])
    with pytest.raises(ValueError, match="same file"):
        inject_file(src, src, injector)
    assert src.read_bytes() == before


def test_zap_in_place_replaces_after_success_and_keeps_input_on_error(
    tmp_path: Path,
) -> None:
    src = tmp_path / "zap.fil"
    with _header(src, signed=False, nsamples=8, nchans=2).prep_outfile(
        str(src),
        sources=(),
    ) as writer:
        writer.cwrite(np.full(16, 9, dtype=np.float32))
    before = src.read_bytes()
    runner = CliRunner()
    failed = runner.invoke(
        spp_clean.main,
        ["zap-times", str(src), "-s", "4", "-e", "1", "--in-place"],
    )
    assert failed.exit_code != 0
    assert src.read_bytes() == before
    assert not src.with_name(src.name + ".zap.tmp").exists()

    ok = runner.invoke(
        spp_clean.main,
        ["zap-times", str(src), "-s", "0", "-e", "2", "--in-place"],
    )
    assert ok.exit_code == 0, ok.output
    assert not src.with_name(src.name + ".zap.tmp").exists()
    with FilReader(src) as fil:
        data = fil.read_block(0, 8).data
    np.testing.assert_array_equal(data[:, :2], 0)
    np.testing.assert_array_equal(data[:, 2:], 9)
