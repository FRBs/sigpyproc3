"""Numba kernels for single-pulse search (PRESTO boxcars and Heimdall stages).

PRESTO-style boxcar scoring and pruning follow ``single_pulse_search.py``
(Ransom et al.). Heimdall baseline, filtering, and clustering follow the
single-series CPU model of the Heimdall pipeline (Barsdell et al.).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from sigpyproc.kernels.fourier import nb_irfft, nb_rfft


@njit(cache=True)
def boxcar_running_scores(
    data: np.ndarray,
    valid: np.ndarray,
    width: int,
    first: int,
    count: int,
) -> np.ndarray:
    """Score ``count`` center bins starting at ``first`` within a halo buffer."""
    out = np.full(count, -np.inf, dtype=np.float64)
    start = first - width // 2
    total = 0.0
    invalid = 0
    for j in range(start, start + width):
        if 0 <= j < data.size:
            total += np.float64(data[j])
            invalid += not valid[j]
        else:
            invalid += 1
    norm = np.sqrt(width)
    for i in range(count):
        if invalid == 0:
            out[i] = total / norm
        leave = start + i
        enter = leave + width
        if 0 <= leave < data.size:
            total -= np.float64(data[leave])
            invalid -= not valid[leave]
        else:
            invalid -= 1
        if 0 <= enter < data.size:
            total += np.float64(data[enter])
            invalid += not valid[enter]
        else:
            invalid += 1
    return out


@njit(cache=True)
def boxcar_fft_scores(
    data_fft: np.ndarray,
    invalid_prefix: np.ndarray,
    kernel_fft: np.ndarray,
    nfft: int,
    width: int,
    first: int,
    count: int,
) -> np.ndarray:
    """Extract valid centered scores from a zero-padded linear convolution."""
    conv = nb_irfft(data_fft * kernel_fft, nfft)
    out = np.full(count, -np.inf, dtype=np.float64)
    nsamp = invalid_prefix.size - 1
    for i in range(count):
        start = first + i - width // 2
        end = start + width
        if start >= 0 and end <= nsamp and invalid_prefix[end] == invalid_prefix[start]:
            out[i] = conv[end - 1]
    return out


@njit(cache=True)
def boxcar_fft_kernel(width: int, nfft: int) -> np.ndarray:
    """Frequency response of a square-normalized boxcar."""
    kernel = np.zeros(nfft, dtype=np.float64)
    kernel[:width] = 1.0 / np.sqrt(width)
    return nb_rfft(kernel)


@njit(cache=True)
def threshold_boxcar(
    scores: np.ndarray,
    offset: int,
    snr_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect threshold crossings without Python per-sample work."""
    bins = np.empty(scores.size, dtype=np.int64)
    snrs = np.empty(scores.size, dtype=np.float64)
    n = 0
    for i in range(scores.size):
        if np.isfinite(scores[i]) and scores[i] >= snr_min:
            bins[n] = offset + i
            snrs[n] = scores[i]
            n += 1
    # Compact results so sparse crossings do not retain full chunk buffers.
    return bins[:n].copy(), snrs[:n].copy()


@njit(cache=True)
def prune_boxcar_candidates(
    bins: np.ndarray,
    widths: np.ndarray,
    priority: np.ndarray,
    deadtime: int,
    minimum_radius: int,
) -> np.ndarray:
    """Greedy suppression in priority order over time-sorted candidate arrays."""
    active = np.ones(bins.size, dtype=np.bool_)
    keep = np.zeros(bins.size, dtype=np.bool_)
    if bins.size == 0:
        return keep
    max_radius = max(deadtime, minimum_radius, np.max(widths) // 2)
    for idx in priority:
        if not active[idx]:
            continue
        keep[idx] = True
        lo = np.searchsorted(bins, bins[idx] - max_radius)
        hi = np.searchsorted(bins, bins[idx] + max_radius, side="right")
        for j in range(lo, hi):
            radius = max(deadtime, minimum_radius, widths[idx] // 2, widths[j] // 2)
            if abs(bins[j] - bins[idx]) <= radius:
                active[j] = False
    return keep


@njit(cache=True)
def heimdall_median_scrunch(data: np.ndarray) -> np.ndarray:
    """Median each group of five; discard its remainder, as upstream does.

    For fewer than five samples return their ordinary median. Use insertion
    sorting and a reusable five-value buffer rather than upstream's network.
    """
    n = data.size
    if n == 0:
        return np.empty(0, np.float64)
    group_size = min(5, n)
    out = np.empty(max(1, n // 5), np.float64)
    work = np.empty(group_size, np.float64)
    for group in range(out.size):
        for j in range(group_size):
            value = np.float64(data[group * 5 + j])
            k = j
            while k > 0 and work[k - 1] > value:
                work[k] = work[k - 1]
                k -= 1
            work[k] = value
        middle = group_size // 2
        if group_size % 2:
            out[group] = work[middle]
        else:
            out[group] = (work[middle - 1] + work[middle]) / 2
    return out


@njit(cache=True)
def heimdall_rms(data: np.ndarray) -> float:
    """Upstream's approximate zero-centered MAD, including its 1.4862 factor."""
    if data.size == 0:
        return 0.0
    reduced = np.abs(data).astype(np.float64)
    while reduced.size > 1:
        reduced = heimdall_median_scrunch(reduced)
    return 1.4862 * reduced[0]


@njit(cache=True)
def heimdall_linear_stretch(data: np.ndarray, size: int) -> np.ndarray:
    """Linear interpolation/extrapolation with upstream's sample-phase shift."""
    out = np.empty(size, np.float64)
    if data.size == 1:
        out[:] = data[0]
        return out
    if size == 1:
        out[0] = data[0]
        return out
    step = (size - 1) / (data.size - 1)
    phase = int(step / 2) / step
    for j in range(size):
        coordinate = j / step - phase
        left = min(max(int(coordinate), 0), data.size - 2)
        out[j] = data[left] + (coordinate - left) * (data[left + 1] - data[left])
    return out


@njit(cache=True)
def heimdall_baseline(data: np.ndarray, radius: int) -> np.ndarray:
    """Resample, median-scrunch, extrapolate, and stretch a robust baseline."""
    n = data.size
    baseline = np.zeros(n, np.float64)
    if n < 2:
        return baseline
    nknots = int(n / radius + 0.5)
    if nknots <= 1:
        # Upstream leaves the data unchanged when smoothing has too few knots.
        return baseline
    factor = 1
    while factor <= (n // nknots) // 5:
        factor *= 5
    reduced = heimdall_linear_stretch(data, nknots * factor)
    while reduced.size > nknots:
        reduced = heimdall_median_scrunch(reduced)
    expanded = heimdall_linear_stretch(reduced, 2 * nknots)
    knots = np.empty(expanded.size + 2, np.float64)
    knots[1:-1] = expanded
    knots[0] = 2 * expanded[0] - expanded[1]
    knots[-1] = 2 * expanded[-1] - expanded[-2]
    return heimdall_linear_stretch(knots, n)


@njit(cache=True)
def heimdall_filter(
    prefix: np.ndarray,
    max_width: int,
    width: int,
    stride: int,
) -> np.ndarray:
    """Boxcar sums on the common max-width valid span, optionally strided."""
    count = prefix.size - max_width
    nout = (count + stride - 1) // stride
    out = np.empty(nout, np.float64)
    first = max_width // 2
    for j in range(nout):
        center = first + j * stride
        lo = center - width // 2
        out[j] = prefix[lo + width] - prefix[lo]
    return out


@njit(cache=True)
def heimdall_giants(
    scores: np.ndarray,
    threshold: float,
    merge_distance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segment strictly-above-threshold samples by gaps; keep first tied peak."""
    peaks = np.empty(scores.size, np.int64)
    values = np.empty(scores.size, np.float64)
    begins = np.empty(scores.size, np.int64)
    ends = np.empty(scores.size, np.int64)
    count = 0
    previous = -1
    for j in range(scores.size):
        if not np.isfinite(scores[j]) or scores[j] <= threshold:
            continue
        if count == 0 or j - previous > merge_distance:
            peaks[count] = j
            values[count] = scores[j]
            begins[count] = j
            ends[count] = j + 1
            count += 1
        else:
            ends[count - 1] = j + 1
            if scores[j] > values[count - 1]:
                peaks[count - 1] = j
                values[count - 1] = scores[j]
        previous = j
    return (
        peaks[:count].copy(),
        values[:count].copy(),
        begins[:count].copy(),
        ends[:count].copy(),
    )


@njit(cache=True)
def _root(parents: np.ndarray, i: int) -> int:
    while parents[i] != i:
        parents[i] = parents[parents[i]]
        i = parents[i]
    return i


@njit(cache=True)
def heimdall_cluster_labels(
    bins: np.ndarray,
    widths: np.ndarray,
    time_tolerance: int,
    filter_tolerance: int,
) -> np.ndarray:
    """Label connected components of time-sorted giants from one DM and gulp."""
    n = bins.size
    parents = np.arange(n, dtype=np.int64)
    levels = np.empty(n, np.int64)
    for i in range(n):
        value = widths[i]
        level = 0
        while value > 1:
            value //= 2
            level += 1
        levels[i] = level
    if n == 0:
        return parents
    largest_radius = time_tolerance * np.max(widths)
    for i in range(n):
        stop = np.searchsorted(bins, bins[i] + largest_radius, side="right")
        for j in range(i + 1, stop):
            if abs(levels[j] - levels[i]) > filter_tolerance:
                continue
            radius = time_tolerance * max(widths[i], widths[j])
            if bins[j] - bins[i] <= radius:
                a, b = _root(parents, i), _root(parents, j)
                parents[max(a, b)] = min(a, b)
    for i in range(n):
        parents[i] = _root(parents, i)
    return parents
