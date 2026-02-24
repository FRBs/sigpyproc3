"""Sigproc-style HDF5 header parsing.

This module contains functions for parsing Sigproc-style HDF5 headers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
        If the file does not exist
    OSError
        If file header is not in sigproc HDF5 format or no data block found in file
    """
    filename = Path(filename)
    if not filename.is_file():
        msg = f"File {filename} not found"
        raise FileNotFoundError(msg)
    if not h5py.is_hdf5(filename):
        msg = f"File {filename} is not an HDF5 file"
        raise OSError(msg)
    with h5py.File(filename, "r") as h5file:
        class_atr = h5file.attrs["CLASS"]
        if not isinstance(class_atr, str | bytes) or class_atr.lower() not in [
            "filterbank",
            b"filterbank",
        ]:
            msg = "File header is not in sigproc format"
            raise OSError(msg)
        if "data" not in h5file:
            msg = "No data block found in file"
            raise OSError(msg)
        header = {}
        for key, val in h5file["data"].attrs.items():
            header[key] = val.decode() if isinstance(val, bytes) else val
        axis_map = map_dimensions(header["DIMENSION_LABELS"])

        header["filelen"] = filename.stat().st_size
        header["nsamples"] = h5file["data"].shape[axis_map["time"]]
        header["nbeams"] = h5file["data"].shape[axis_map["feed_id"]]
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
    ValueError
        If an unexpected dimension label is encountered.
    """
    expected_labels = {"time", "feed_id", "frequency"}
    dimension_map = {}

    for i, label in enumerate(dimension_labels):
        label_str = label.lower() if isinstance(label, str) else label.decode().lower()
        if label_str not in expected_labels:
            msg = f"Unexpected dimension label: {label_str}"
            raise ValueError(msg)
        dimension_map[label_str] = i

    # Ensure all expected labels are present
    for label in expected_labels:
        if label not in dimension_map:
            msg = f"Missing expected dimension: {label}"
            raise ValueError(msg)

    return dimension_map
