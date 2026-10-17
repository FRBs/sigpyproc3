"""Frequency and time combining rules."""

from pathlib import Path

import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.io.combine import merge_filterbanks, splice_filterbanks
from sigpyproc.readers import FilReader


def _write_band(
    path: Path,
    fch1: float,
    *,
    signed: bool = False,
    tstart: float = 50000.0,
    nsamples: int = 8,
) -> None:
    nchan = 4
    header = Header(
        filename="source.fil",
        data_type="filterbank",
        nchans=nchan,
        foff=-1.0,
        fch1=fch1,
        nbits=8,
        tsamp=0.001,
        tstart=tstart,
        nsamples=nsamples,
        signed=signed,
        source="J0000+0000",
    )
    data = np.arange(nsamples * nchan, dtype=np.uint8)
    with header.prep_outfile(str(path), nbits=8, sources=()) as outfile:
        outfile.cwrite(data)


def test_adjoining_bands_splice(tmp_path: Path) -> None:
    high = tmp_path / "high.fil"
    low = tmp_path / "low.fil"
    _write_band(high, 100.0)
    _write_band(low, 96.0)
    out = tmp_path / "spliced.fil"
    splice_filterbanks([low, high], str(out))
    loaded = FilReader(out)
    assert loaded.header.nchans == 8
    assert loaded.header.fch1 == pytest.approx(100.0)
    assert loaded.header.foff == pytest.approx(-1.0)


def test_gap_and_signed_mismatch_leave_no_output(tmp_path: Path) -> None:
    high = tmp_path / "high.fil"
    gapped = tmp_path / "gap.fil"
    signed = tmp_path / "signed.fil"
    _write_band(high, 100.0)
    _write_band(gapped, 90.0)
    _write_band(signed, 96.0, signed=True)
    out = tmp_path / "out.fil"
    with pytest.raises(ValueError, match="do not adjoin"):
        splice_filterbanks([high, gapped], str(out))
    assert not out.exists()
    with pytest.raises(ValueError, match="signed"):
        splice_filterbanks([high, signed], str(out))
    assert not out.exists()


def test_combine_rejects_too_few_files_before_writing(tmp_path: Path) -> None:
    only = tmp_path / "only.fil"
    _write_band(only, 100.0)
    out = tmp_path / "out.fil"
    out.write_bytes(b"keep")
    with pytest.raises(ValueError, match="at least two"):
        merge_filterbanks([only], str(out))
    with pytest.raises(ValueError, match="at least two"):
        splice_filterbanks([only], str(out))
    with pytest.raises(ValueError, match="gulp"):
        merge_filterbanks([only, only], str(out), gulp=0)
    assert out.read_bytes() == b"keep"


def test_auto_axis_follows_the_shared_coordinate(tmp_path: Path) -> None:
    nsamples = 8
    tsamp = 0.001
    first = tmp_path / "first.fil"
    second = tmp_path / "second.fil"
    _write_band(first, 100.0, nsamples=nsamples)
    _write_band(
        second,
        100.0,
        nsamples=nsamples,
        tstart=50000.0 + nsamples * tsamp / 86400.0,
    )
    merged = tmp_path / "time.fil"
    merge_filterbanks([first, second], str(merged))
    with FilReader(merged) as reader:
        assert reader.header.nchans == 4
        assert reader.header.nsamples == 2 * nsamples

    high = tmp_path / "high.fil"
    low = tmp_path / "low.fil"
    _write_band(high, 100.0)
    _write_band(low, 96.0)
    spliced = tmp_path / "freq.fil"
    merge_filterbanks([low, high], str(spliced))
    with FilReader(spliced) as reader:
        assert reader.header.nchans == 8
        assert reader.header.fch1 == pytest.approx(100.0)
        assert reader.header.nsamples == 8


def test_bad_axis_and_gulp_are_rejected_before_writing(tmp_path: Path) -> None:
    high = tmp_path / "high.fil"
    low = tmp_path / "low.fil"
    _write_band(high, 100.0)
    _write_band(low, 96.0)
    out = tmp_path / "out.fil"
    with pytest.raises(ValueError, match="axis"):
        merge_filterbanks([high, low], str(out), axis="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gulp"):
        splice_filterbanks([high, low], str(out), gulp=0)
    assert not out.exists()


def _example_series_header(n: int = 256) -> Header:
    return Header(
        filename="example.tim",
        data_type="time series",
        nchans=1,
        fch1=1400.0,
        foff=1.0,
        tsamp=0.001,
        nbits=32,
        tstart=50000.0,
        nsamples=n,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"tsamp": 0.002},
        {"nbits": 16},
        {"nsamples": 7},
        {"foff": -2.0},
        {"dm": 20.0},
        {"frame": "barycentric"},
        {"source": "Other"},
        {"tstart": 50001.0},
    ],
)
def test_splice_mismatches_leave_existing_output_untouched(
    updates: dict,
    tmp_path: Path,
) -> None:
    inputs = []
    for name, freq, changes in [("high", 1420.0, {}), ("low", 1416.0, updates)]:
        path = tmp_path / f"{name}.fil"
        header = _example_series_header(8).new_header(
            {
                "filename": str(path),
                "data_type": "filterbank",
                "nbits": 8,
                "nchans": 4,
                "fch1": freq,
                "foff": -1.0,
                **changes,
            }
        )
        with header.prep_outfile(str(path), sources=()) as writer:
            writer.cwrite(np.arange(header.nsamples * 4, dtype=np.float32))
        inputs.append(path)
    destination = tmp_path / "existing.fil"
    destination.write_bytes(b"keep this output on validation failure")
    with pytest.raises(ValueError, match="match"):
        splice_filterbanks(inputs, str(destination))
    assert destination.read_bytes() == b"keep this output on validation failure"


def test_ascending_splice_sorts_samples_and_channel_coordinates(tmp_path: Path) -> None:
    inputs = []
    arrays = []
    for band in range(2):
        path = tmp_path / f"band{band}.fil"
        data = (np.arange(32) + band * 100).astype(np.uint8).reshape(8, 4)
        header = _example_series_header(8).new_header(
            {
                "filename": str(path),
                "data_type": "filterbank",
                "nbits": 8,
                "nchans": 4,
                "foff": 1.0,
                "fch1": 1400.0 + band * 4,
            }
        )
        with header.prep_outfile(str(path), sources=()) as writer:
            writer.cwrite(data.ravel())
        inputs.append(path)
        arrays.append(data)
    out = tmp_path / "joined.fil"
    splice_filterbanks(inputs[::-1], str(out), gulp=3)
    with FilReader(out) as reader:
        np.testing.assert_allclose(reader.header.chan_freqs, np.arange(1400, 1408))
        np.testing.assert_array_equal(
            reader.read_block(0, 8).data.T, np.concatenate(arrays, axis=1)
        )
