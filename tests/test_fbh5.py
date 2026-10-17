"""Header-only Sigproc HDF5 parsing."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from sigpyproc.header import Header
from sigpyproc.io import fbh5


def _write_fbh5(
    path: Path,
    *,
    klass: bytes | str | None = b"FILTERBANK",
    labels: list[bytes] | None = None,
    shape: tuple[int, ...] = (8, 1, 4),
    write_labels: bool = True,
) -> None:
    if labels is None:
        labels = [b"time", b"feed_id", b"frequency"]
    nchans = 4
    if write_labels and b"frequency" in labels:
        axis = labels.index(b"frequency")
        if axis < len(shape):
            nchans = int(shape[axis])
    with h5py.File(path, "w") as handle:
        if klass is not None:
            handle.attrs["CLASS"] = klass
        data = handle.create_dataset("data", shape=shape, dtype=np.uint8)
        if write_labels:
            data.attrs["DIMENSION_LABELS"] = np.asarray(labels)
        data.attrs["nchans"] = nchans
        data.attrs["nbits"] = 8
        data.attrs["tsamp"] = 6.4e-5
        data.attrs["tstart"] = 59000.0
        data.attrs["fch1"] = 1500.0
        data.attrs["foff"] = -0.5
        data.attrs["nifs"] = 1
        data.attrs["data_type"] = 1


def test_parse_header_reads_named_axes(tmp_path: Path) -> None:
    path = tmp_path / "band.h5"
    _write_fbh5(path)
    header = fbh5.parse_header(path)
    assert header["nsamples"] == 8
    assert header["nbeams"] == 1
    loaded = Header.from_fbh5(str(path))
    assert loaded.nsamples == 8
    assert loaded.nchans == 4
    assert loaded.fch1 == pytest.approx(1500.0)
    assert loaded.foff == pytest.approx(-0.5)

    permuted = tmp_path / "permuted.h5"
    _write_fbh5(
        permuted,
        shape=(4, 8, 1),
        labels=[b"frequency", b"time", b"feed_id"],
    )
    parsed = fbh5.parse_header(permuted)
    assert parsed["nsamples"] == 8
    assert parsed["nbeams"] == 1


def test_missing_file_and_unknown_labels_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        fbh5.parse_header(tmp_path / "absent.h5")
    assert fbh5.map_dimensions(["time", "feed_id", "frequency"])["frequency"] == 2
    with pytest.raises(ValueError, match="polarisation"):
        fbh5.map_dimensions(["time", "feed_id", "polarisation"])
    with pytest.raises(TypeError, match="Unexpected dimension label"):
        fbh5.map_dimensions(["time", "feed_id", 1])
    with pytest.raises(ValueError, match="Missing expected dimension"):
        fbh5.map_dimensions(["time", "feed_id"])


def test_missing_class_is_an_oserror(tmp_path: Path) -> None:
    path = tmp_path / "noclass.h5"
    _write_fbh5(path, klass=None)
    with pytest.raises(OSError, match="CLASS"):
        fbh5.parse_header(path)


def test_wrong_class_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "image.h5"
    _write_fbh5(path, klass=b"IMAGE")
    with pytest.raises(OSError, match="sigproc"):
        fbh5.parse_header(path)


def test_duplicate_labels_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.h5"
    _write_fbh5(path, labels=[b"time", b"time", b"frequency"])
    with pytest.raises(ValueError, match="Duplicate dimension label: time"):
        fbh5.parse_header(path)


def test_missing_labels_and_rank_are_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "nolabel.h5"
    _write_fbh5(missing, write_labels=False)
    with pytest.raises(OSError, match="DIMENSION_LABELS"):
        fbh5.parse_header(missing)

    flat = tmp_path / "flat.h5"
    _write_fbh5(flat, shape=(8, 4))
    with pytest.raises(OSError, match="rank 2"):
        fbh5.parse_header(flat)

    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as handle:
        handle.attrs["CLASS"] = b"FILTERBANK"
    with pytest.raises(OSError, match="data"):
        fbh5.parse_header(empty)

    text = tmp_path / "not.h5"
    text.write_text("not hdf5")
    with pytest.raises(OSError, match="not an HDF5"):
        fbh5.parse_header(text)
