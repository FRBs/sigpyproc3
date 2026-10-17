"""Sigproc-style HDF5 header parsing.

Only the filterbank header is read. There is no HDF5 sample reader or writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py

if TYPE_CHECKING:
    import numpy as np


def parse_header(filename: str | Path) -> dict:
    """
    Parse the metadata from a single Sigproc-style HDF5 file.

    Parameters
    ----------
    filename : str | Path
        Path to the sigproc HDF5 file containing the metadata

    Returns
    -------
    dict
        Observational metadata

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    OSError
        If the file is not HDF5, has no ``CLASS`` or ``DIMENSION_LABELS``,
        is not a filterbank header, or the data cube is not three-dimensional.
    ValueError
        If a dimension label is missing, duplicated, or not one of
        ``time``, ``feed_id`` and ``frequency``.
    """
    filename = Path(filename)
    if not filename.is_file():
        msg = f"File {filename} not found"
        raise FileNotFoundError(msg)
    if not h5py.is_hdf5(filename):
        msg = f"File {filename} is not an HDF5 file"
        raise OSError(msg)
    with h5py.File(filename, "r") as h5file:
        try:
            class_atr = h5file.attrs["CLASS"]
        except KeyError as exc:
            msg = f"File {filename} has no CLASS attribute"
            raise OSError(msg) from exc
        if not isinstance(class_atr, str | bytes) or class_atr.lower() not in {
            "filterbank",
            b"filterbank",
        }:
            msg = "File header is not in sigproc format"
            raise OSError(msg)
        data = h5file.get("data")
        if not isinstance(data, h5py.Dataset):
            msg = "No data block found in file"
            raise OSError(msg)
        if data.ndim != 3:
            msg = (
                f"File {filename} data has rank {data.ndim}; "
                "a filterbank header needs time, feed_id and frequency axes"
            )
            raise OSError(msg)
        header: dict[str, Any] = {}
        for key, val in data.attrs.items():
            header[key] = val.decode() if isinstance(val, bytes) else val
        if "DIMENSION_LABELS" not in header:
            msg = f"File {filename} data has no DIMENSION_LABELS attribute"
            raise OSError(msg)
        axis_map = map_dimensions(header["DIMENSION_LABELS"])

        header["filelen"] = filename.stat().st_size
        header["nsamples"] = data.shape[axis_map["time"]]
        header["nbeams"] = data.shape[axis_map["feed_id"]]
        header["filename"] = filename.as_posix()
    return header


def map_dimensions(dimension_labels: np.ndarray | list[str]) -> dict[str, int]:
    """
    Map HDF5 dimension labels to their corresponding axis numbers.

    Parameters
    ----------
    dimension_labels : np.ndarray or List[str]
        Array or list of dimension labels from HDF5 file.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping dimension names to axis numbers.

    Raises
    ------
    TypeError
        If a dimension label is not text.
    ValueError
        If a dimension label is unexpected or duplicated.
    """
    expected_labels = {"time", "feed_id", "frequency"}
    dimension_map = {}

    for i, label in enumerate(dimension_labels):
        if isinstance(label, str):
            label_str = label.strip().lower()
        elif isinstance(label, bytes):
            label_str = label.decode().strip().lower()
        else:
            msg = f"Unexpected dimension label: {label!r}"
            raise TypeError(msg)
        if label_str not in expected_labels:
            msg = f"Unexpected dimension label: {label_str}"
            raise ValueError(msg)
        if label_str in dimension_map:
            msg = f"Duplicate dimension label: {label_str}"
            raise ValueError(msg)
        dimension_map[label_str] = i

    # Ensure all expected labels are present
    for label in expected_labels:
        if label not in dimension_map:
            msg = f"Missing expected dimension: {label}"
            raise ValueError(msg)

    return dimension_map
