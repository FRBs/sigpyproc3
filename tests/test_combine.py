"""Frequency and time combining rules."""

from pathlib import Path

import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.io.combine import merge_filterbanks, splice_filterbanks
from sigpyproc.readers import FilReader


def _write_band(path: Path, fch1: float, *, signed: bool = False) -> None:
    nchan = 4
    header = Header(
        filename="source.fil",
        data_type="filterbank",
        nchans=nchan,
        foff=-1.0,
        fch1=fch1,
        nbits=8,
        tsamp=0.001,
        tstart=50000.0,
        nsamples=8,
        signed=signed,
        source="J0000+0000",
    )
    data = np.arange(8 * nchan, dtype=np.uint8)
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
