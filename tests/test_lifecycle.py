"""File lifecycle, sample-count axes, and packing frame checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.io import sigproc
from sigpyproc.io.bits import pack
from sigpyproc.io.fileio import allocate_buffer
from sigpyproc.io.sigproc import FileInfo
from sigpyproc.readers import FilReader
from sigpyproc.viz.waterfall import WaterfallViewer

if TYPE_CHECKING:
    from pathlib import Path


def _header(**extra: object) -> Header:
    values: dict[str, object] = {
        "filename": "mem.fil",
        "data_type": "filterbank",
        "nchans": 4,
        "foff": -1.0,
        "fch1": 1400.0,
        "nbits": 8,
        "tsamp": 1e-3,
        "tstart": 50000.0,
        "nsamples": 8,
    }
    values.update(extra)
    return Header(**values)  # type: ignore[arg-type]


def test_headers_do_not_share_stream_info() -> None:
    first = _header()
    second = _header()
    first.stream_info.add_entry(
        FileInfo(
            filename="a.fil",
            hdrlen=1,
            datalen=1,
            nsamples=1,
            tstart=50000.0,
            tsamp=1e-3,
        ),
    )
    assert second.stream_info.entries == []


def test_time_series_may_have_zero_channel_width() -> None:
    header = _header(data_type="time series", nchans=1, foff=0.0)
    assert header.foff == 0.0


def test_header_rejects_non_positive_geometry() -> None:
    with pytest.raises(ValueError, match="nchans"):
        _header(nchans=0)


def test_nsamples_counts_every_if(tmp_path: Path) -> None:
    path = tmp_path / "two_if.fil"
    fields = {
        "data_type": 1,
        "telescope_id": 0,
        "machine_id": 0,
        "tstart": 50000.0,
        "tsamp": 1e-3,
        "nbits": 8,
        "nsamples": 0,
        "fch1": 1400.0,
        "foff": -1.0,
        "nchans": 1,
        "nifs": 2,
    }
    path.write_bytes(sigproc.encode_header(fields) + bytes(8))
    header = Header.from_sigproc(path)
    assert header.nifs == 2
    assert header.nsamples == 4


def test_partial_trailing_frame_is_not_a_sample(tmp_path: Path) -> None:
    path = tmp_path / "partial.fil"
    fields = {
        "data_type": 1,
        "telescope_id": 0,
        "machine_id": 0,
        "tstart": 50000.0,
        "tsamp": 1e-3,
        "nbits": 8,
        "nsamples": 0,
        "fch1": 1400.0,
        "foff": -1.0,
        "nchans": 2,
        "nifs": 1,
    }
    # Three bytes is one 2-channel sample plus a trailing byte.
    path.write_bytes(sigproc.encode_header(fields) + bytes(3))
    header = Header.from_sigproc(path)
    assert header.nsamples == 1


def test_pack_rejects_a_partial_group() -> None:
    with pytest.raises(ValueError, match="multiple of 2"):
        pack(np.array([1, 2, 3], dtype=np.uint8), 4)


def test_allocate_buffer_uses_byte_length() -> None:
    def allocator(nbytes: int) -> np.ndarray:
        return np.empty(nbytes // 2, dtype=np.uint16)

    buffer = allocate_buffer(allocator, 8)
    assert memoryview(buffer).nbytes == 8


def test_bad_channel_mask_creates_no_file(filfile_8bit_1: str, tmp_path: Path) -> None:
    out = tmp_path / "masked.fil"
    with FilReader(filfile_8bit_1) as fil, pytest.raises(ValueError, match="chan_mask"):
        fil.apply_channel_mask(np.array([True]), outfile_name=str(out))
    assert not out.exists()


def test_gulp_must_be_positive(filfile_8bit_1: str) -> None:
    with FilReader(filfile_8bit_1) as fil, pytest.raises(ValueError, match="gulp"):
        list(fil.read_plan(gulp=0))


def test_viewer_close_keeps_a_caller_owned_reader(filfile_8bit_1: str) -> None:
    with FilReader(filfile_8bit_1) as fil:
        viewer = WaterfallViewer(fil, batch=True, gui=False)
        viewer.close()
        assert fil.read_block(0, 1).data.shape[1] == 1


def test_viewer_close_closes_a_file_it_opened(filfile_8bit_1: str) -> None:
    viewer = WaterfallViewer(filfile_8bit_1, batch=True, gui=False)
    viewer.close()
    assert viewer.fil._file.file_obj.closed
