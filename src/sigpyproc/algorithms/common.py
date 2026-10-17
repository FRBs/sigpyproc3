"""Common functions for pulse and period search algorithms.

This module contains common functions for pulse and period search algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sigpyproc.core.candidates import CandidateTable

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from sigpyproc.core.custom_types import WidthStyle

# Width preset from PRESTO single_pulse_search.py by Scott Ransom (2015).
PRESTO_WIDTHS = (1, 2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 100, 150, 220, 300)


def _positive_int(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer >= {minimum}"
        raise ValueError(msg)  # noqa: TRY004 - consistent search validation
    if value < minimum:
        msg = f"{name} must be >= {minimum}"
        raise ValueError(msg)


def _finite_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)


def _samples(data: ArrayLike) -> np.ndarray:
    raw = np.asarray(data)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        msg = "data must be a one-dimensional real numeric array"
        raise ValueError(msg)
    if not np.all(np.isfinite(raw)):
        msg = "data must contain only finite samples"
        raise ValueError(msg)
    with np.errstate(over="ignore"):
        arr = np.asarray(raw, dtype=np.float32, order="C")
    if not np.all(np.isfinite(arr)):
        msg = "data must be representable as finite float32 samples"
        raise ValueError(msg)
    return arr


def _validate_detections(detections: PulseDetections) -> None:
    bins, snrs, widths = detections.bins, detections.snrs, detections.widths
    if (
        bins.ndim != 1
        or snrs.shape != bins.shape
        or widths.shape != bins.shape
        or not np.issubdtype(bins.dtype, np.integer)
        or not np.issubdtype(widths.dtype, np.integer)
        or np.any(bins < 0)
        or np.any(widths < 1)
        or not np.all(np.isfinite(snrs))
    ):
        msg = (
            "detections require equal-length finite SNRs, nonnegative bins, "
            "and positive integer widths"
        )
        raise ValueError(msg)


def make_width_bank(
    max_width: int = 256,
    *,
    style: WidthStyle = "presto",
    spacing_factor: float = 2.0,
    widths: ArrayLike | None = None,
) -> np.ndarray:
    """Build sorted unique integer widths bounded by ``max_width``.

    Explicit ``widths`` override ``style``. The PRESTO preset stops at 300;
    use geometric or explicit widths for wider pulses. Geometric generation
    follows ``max(previous + 1, int(previous * spacing_factor))``.

    Parameters
    ----------
    max_width : int, optional
        Width ceiling in samples, by default 256.
    style : {"presto", "geometric", "powers_of_two"}, optional
        Preset bank, by default "presto".
    spacing_factor : float, optional
        Geometric spacing (> 1), by default 2.0.
    widths : array_like, optional
        Explicit integer bank, overriding style.

    Returns
    -------
    ndarray
        Sorted, unique int64 widths.
    """
    _positive_int(max_width, "max_width")
    if style not in {"presto", "geometric", "powers_of_two"}:
        msg = f"unknown width style: {style}"
        raise ValueError(msg)
    if not np.isfinite(spacing_factor) or spacing_factor <= 1:
        msg = "spacing_factor must be finite and greater than 1"
        raise ValueError(msg)
    if widths is not None:
        values = np.asarray(widths)
        if (
            values.ndim != 1
            or values.size == 0
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 1)
            or np.any(values > max_width)
        ):
            msg = "widths must be a nonempty array of integers in [1, max_width]"
            raise ValueError(msg)
        return np.unique(values).astype(np.int64)
    if style == "presto":
        return np.array([w for w in PRESTO_WIDTHS if w <= max_width], dtype=np.int64)
    if style == "powers_of_two":
        spacing_factor = 2.0
    bank = [1]
    while bank[-1] < max_width:
        # Compare before multiplying to avoid overflow for very large spacing.
        if spacing_factor > max_width / bank[-1]:
            break
        width = max(bank[-1] + 1, int(bank[-1] * spacing_factor))
        if width > max_width:
            break
        bank.append(width)
    return np.array(bank, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class PulseDetections:
    """Parallel arrays of sample bins, SNRs, and boxcar widths in samples."""

    bins: np.ndarray
    snrs: np.ndarray
    widths: np.ndarray

    @classmethod
    def empty(cls) -> PulseDetections:
        """Construct correctly typed empty detection arrays."""
        return cls(
            np.empty(0, np.int64), np.empty(0, np.float64), np.empty(0, np.int64)
        )


def pulse_candidates(
    detections: PulseDetections,
    tsamp: float,
    *,
    dm: float = 0.0,
    accel: float = 0.0,
    bin_origin: str = "center",
) -> CandidateTable:
    """Convert detection arrays to the shared pulse candidate table schema.

    Parameters
    ----------
    detections : PulseDetections
        Finite detections, with widths in samples.
    tsamp : float
        Positive sample interval in seconds.
    dm : float, optional
        Dispersion measure attached to each row, by default 0.0.
    accel : float, optional
        Acceleration attached to each row, by default 0.0.

    Returns
    -------
    CandidateTable
        Pulse rows in SNR priority order, with widths converted to seconds.
    """
    _finite_positive(tsamp, "tsamp")
    _validate_detections(detections)
    rows = [
        {
            "period_s": np.nan,
            "freq_hz": np.nan,
            "snr": float(snr),
            "dm": dm,
            "accel": accel,
            "nharm": 1,
            "bin": int(bin_),
            "width_s": int(width) * tsamp,
            "kind": "pulse",
            "bin_origin": bin_origin,
        }
        for bin_, snr, width in zip(
            detections.bins,
            detections.snrs,
            detections.widths,
            strict=True,
        )
    ]
    # Sort in float64 before CandidateTable converts SNRs to float32.
    order = np.lexsort((detections.bins, detections.widths, -detections.snrs))
    return CandidateTable.from_rows([rows[i] for i in order])
