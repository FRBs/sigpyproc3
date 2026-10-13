"""Composable PRESTO-inspired single-pulse search stages.

Adapted from PRESTO single_pulse_search.py by Scott Ransom (copyright 2015):
blockwise IQR normalization, valid-window filtering, and deterministic greedy
pruning are shared by running-sum and FFT scoring engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sigpyproc import kernels
from sigpyproc.algorithms.common import (
    PulseDetections,
    _finite_positive,
    _positive_int,
    _samples,
    _validate_detections,
    make_width_bank,
    pulse_candidates,
)
from sigpyproc.core import stats
from sigpyproc.core.candidates import CandidateTable
from sigpyproc.kernels import search as search_kernels

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import ArrayLike

    from sigpyproc.core.custom_types import (
        BoxcarEngine,
        PulseDetrend,
    )


@dataclass(frozen=True, slots=True)
class PreparedPulseData:
    """Normalized samples, validity mask, block bounds, and noise estimates.

    ``block_ends`` are exclusive. Rejected samples are zero in ``data`` and
    false in ``valid``; scorers reject any window containing such a sample.
    """

    data: np.ndarray
    valid: np.ndarray
    block_starts: np.ndarray
    block_ends: np.ndarray
    noise: np.ndarray


@dataclass(frozen=True, slots=True)
class BoxcarScores:
    """SNRs for consecutive center bins starting at ``offset`` at one width.

    Invalid or incomplete windows have SNR ``-inf``. The score array is owned
    by this object and remains usable after the iterator advances.
    """

    offset: int
    width: int
    snrs: np.ndarray


def preprocess_pulse_data(
    data: ArrayLike,
    *,
    block_size: int = 8000,
    detrend: PulseDetrend = "linear",
    mask_bad_blocks: bool = False,
) -> PreparedPulseData:
    """Detrend and IQR-normalize blocks without dropping trailing samples.

    A remainder is merged into the last full block; short arrays form one
    block. Zero IQR falls back to standard deviation. Zero-noise blocks are
    always invalid. Optional noise-outlier rejection uses median +/- four
    Gaussian-normalized MADs, with at least ten usable blocks and nonzero MAD.

    Parameters
    ----------
    data : array_like
        Finite, real, one-dimensional input samples. Not modified.
    block_size : int, optional
        Nominal block length in samples (>= 2), by default 8000.
    detrend : {"linear", "median"}, optional
        Baseline removal, by default "linear".
    mask_bad_blocks : bool, optional
        Reject noise outliers, by default False. Bright pulses can skew a
        linear fit and trigger rejection; compare median removal as needed.

    Returns
    -------
    PreparedPulseData
        Float32 normalized samples, boolean validity, block bounds, and
        float64 noise estimates before normalization.
    """
    _positive_int(block_size, "block_size", 2)
    if detrend not in {"linear", "median"}:
        msg = f"unknown detrend method: {detrend}"
        raise ValueError(msg)
    arr = _samples(data).copy()
    nblocks = max(1, arr.size // block_size) if arr.size else 0
    starts = np.arange(nblocks, dtype=np.int64) * block_size
    ends = (
        np.append(starts[1:], arr.size).astype(np.int64) if nblocks else starts.copy()
    )
    noise = np.zeros(nblocks, dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
        block = arr[lo:hi]
        if detrend == "linear":
            arr[lo:hi] = kernels.detrend_1d(block)
        else:
            arr[lo:hi] = block.astype(np.float64) - np.median(block.astype(np.float64))
        noise[i] = float(stats.estimate_scale(arr[lo:hi], "iqr"))
        if noise[i] == 0:
            noise[i] = np.std(arr[lo:hi], dtype=np.float64)
    usable = np.isfinite(noise) & (noise > 0)
    if mask_bad_blocks and np.count_nonzero(usable) >= 10:
        scales = noise[usable]
        median = stats.estimate_loc(scales, "median")
        spread = stats.estimate_scale(scales, "mad")
        if spread > 0:
            usable &= np.abs(noise - median) <= 4 * spread
    valid = np.zeros(arr.size, dtype=np.bool_)
    for i, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
        if usable[i]:
            arr[lo:hi] /= noise[i]
            valid[lo:hi] = True
        else:
            arr[lo:hi] = 0
    return PreparedPulseData(arr, valid, starts, ends, noise)


def iter_boxcar_scores(
    prepared: PreparedPulseData,
    widths: ArrayLike,
    *,
    engine: BoxcarEngine = "running_sum",
    chunk_size: int = 8192,
) -> Iterator[BoxcarScores]:
    """Yield one chunk/width of centered scores, with halo and validity checks.

    Both engines return sum(data) / sqrt(width). A window starting at ``s``
    is reported at ``s + width // 2``. FFT chunks share one data transform;
    kernel transforms are reused across chunks with the same transform size.

    Parameters
    ----------
    prepared : PreparedPulseData
        Normalized samples and a boolean validity mask. Block statistics do
        not affect scoring; they are retained for inspection.
    widths : array_like
        Positive integer widths that fit the observation.
    engine : {"running_sum", "fft"}, optional
        Numba scorer, by default "running_sum".
    chunk_size : int, optional
        Number of output center bins per chunk, by default 8192. Halos are
        added internally; chunk size need not exceed the widest boxcar.

    Yields
    ------
    BoxcarScores
        Owned float64 scores for one chunk and width. Invalid windows have
        score -inf. Iteration order is chunk, then ascending width.
    """
    _positive_int(chunk_size, "chunk_size")
    if engine not in {"running_sum", "fft"}:
        msg = f"unknown boxcar engine: {engine}"
        raise ValueError(msg)
    arr = _samples(prepared.data)
    valid = np.asarray(prepared.valid)
    if valid.shape != arr.shape or valid.dtype != np.bool_:
        msg = "valid must be a boolean mask with the same shape as data"
        raise ValueError(msg)
    values = np.asarray(widths)
    if values.size == 0:
        msg = "widths must not be empty"
        raise ValueError(msg)
    bank = make_width_bank(max(1, arr.size), widths=values)
    largest = int(bank[-1])
    # Keep at most one chunk's kernel responses, even for a short final chunk.
    cached_size = 0
    responses: dict[int, np.ndarray] = {}
    for offset in range(0, arr.size, chunk_size):
        count = min(chunk_size, arr.size - offset)
        lo = max(0, offset - largest // 2)
        hi = min(arr.size, offset + count + (largest - 1) // 2)
        data_chunk = arr[lo:hi]
        valid_chunk = valid[lo:hi]
        if engine == "fft":
            nfft = kernels.nb_fft_good_size(data_chunk.size + largest - 1, real=True)
            data_fft = kernels.nb_rfft(data_chunk.astype(np.float64), nfft)
            invalid_prefix = np.empty(data_chunk.size + 1, np.int64)
            invalid_prefix[0] = 0
            np.cumsum(~valid_chunk, out=invalid_prefix[1:])
            if nfft != cached_size:
                responses.clear()
                cached_size = nfft
        for width in bank:
            w = int(width)
            if engine == "fft":
                if w not in responses:
                    responses[w] = search_kernels.boxcar_fft_kernel(w, nfft)
                scores = search_kernels.boxcar_fft_scores(
                    data_fft,
                    invalid_prefix,
                    responses[w],
                    nfft,
                    w,
                    offset - lo,
                    count,
                )
            else:
                scores = search_kernels.boxcar_running_scores(
                    data_chunk,
                    valid_chunk,
                    w,
                    offset - lo,
                    count,
                )
            yield BoxcarScores(offset, w, scores)


def detect_boxcar_peaks(scores: BoxcarScores, snr_min: float = 6.0) -> PulseDetections:
    """Collect threshold crossings; pruning is a separate stage.

    Parameters
    ----------
    scores : BoxcarScores
        One chunk/width of centered scores.
    snr_min : float, optional
        Positive threshold (inclusive), by default 6.0.

    Returns
    -------
    PulseDetections
        Compact arrays of finite threshold crossings in increasing bin order.
    """
    _finite_positive(snr_min, "snr_min")
    _positive_int(scores.width, "width")
    _positive_int(scores.offset, "offset", 0)
    values = np.asarray(scores.snrs, dtype=np.float64)
    if values.ndim != 1:
        msg = "scores must be one-dimensional"
        raise ValueError(msg)
    bins, snrs = search_kernels.threshold_boxcar(values, scores.offset, snr_min)
    return PulseDetections(bins, snrs, np.full(bins.size, scores.width, np.int64))


def prune_pulse_detections(
    detections: PulseDetections,
    *,
    deadtime: int = 0,
    across_widths: bool = True,
) -> PulseDetections:
    """Keep strongest detections using width-dependent greedy suppression.

    Set ``across_widths=False`` to prune each width independently. Within a
    width the radius is width // 2; across widths it is the larger half-width,
    with a minimum of one sample. ``deadtime`` supplies an additional minimum
    radius. Ties prefer narrower widths, then earlier bins. Output is in this
    same SNR priority order; rejected candidates do not suppress others.

    Parameters
    ----------
    detections : PulseDetections
        Equal-length arrays, in any order. Not modified.
    deadtime : int, optional
        Additional minimum exclusion radius in samples, by default 0.
    across_widths : bool, optional
        Compare different widths, by default True. The end-to-end pipeline
        first calls this with False, then with True on the surviving rows.

    Returns
    -------
    PulseDetections
        Surviving detections sorted by decreasing SNR, narrower width, then bin.
    """
    _positive_int(deadtime, "deadtime", 0)
    _validate_detections(detections)
    if detections.bins.size == 0:
        return PulseDetections.empty()
    order = np.argsort(detections.bins, kind="stable")
    bins = np.ascontiguousarray(detections.bins[order], dtype=np.int64)
    snrs = np.ascontiguousarray(detections.snrs[order], dtype=np.float64)
    widths = np.ascontiguousarray(detections.widths[order], dtype=np.int64)
    priority = np.lexsort((bins, widths, -snrs))
    if across_widths:
        keep = search_kernels.prune_boxcar_candidates(
            bins, widths, priority, deadtime, 1
        )
    else:
        keep = np.zeros(bins.size, np.bool_)
        for width in np.unique(widths):
            indices = np.flatnonzero(widths == width)
            local_priority = np.lexsort((bins[indices], -snrs[indices]))
            keep[indices] = search_kernels.prune_boxcar_candidates(
                bins[indices],
                widths[indices],
                local_priority,
                deadtime,
                0,
            )
    selected = priority[keep[priority]]
    return PulseDetections(bins[selected], snrs[selected], widths[selected])


def presto_pulse_search(
    data: ArrayLike,
    tsamp: float,
    *,
    widths: ArrayLike,
    engine: BoxcarEngine = "running_sum",
    snr_min: float = 6.0,
    deadtime: int = 0,
    block_size: int = 8000,
    detrend: PulseDetrend = "linear",
    mask_bad_blocks: bool = False,
    chunk_size: int = 8192,
    dm: float = 0.0,
    accel: float = 0.0,
) -> CandidateTable:
    """Run PRESTO-inspired preprocessing, filtering, and suppression.

    The public frontend in ``sigpyproc.algorithms.box`` selects and validates
    the method and width bank. These stages can also be composed independently.
    """
    prepared = preprocess_pulse_data(
        data,
        block_size=block_size,
        detrend=detrend,
        mask_bad_blocks=mask_bad_blocks,
    )
    parts: list[PulseDetections] = []
    for scores in iter_boxcar_scores(
        prepared, widths, engine=engine, chunk_size=chunk_size
    ):
        detected = detect_boxcar_peaks(scores, snr_min)
        if detected.bins.size:
            parts.append(detected)
    if not parts:
        return CandidateTable.empty()
    detections = PulseDetections(
        np.concatenate([part.bins for part in parts]),
        np.concatenate([part.snrs for part in parts]),
        np.concatenate([part.widths for part in parts]),
    )
    per_width = prune_pulse_detections(detections, across_widths=False)
    pruned = prune_pulse_detections(per_width, deadtime=deadtime)
    return pulse_candidates(pruned, tsamp, dm=dm, accel=accel)
