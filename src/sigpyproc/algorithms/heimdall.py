"""Python implementation of single-dispersion-measure (single-DM) search stages.

This module provides a standalone, inspectable reference implementation of the
time-series search stages (boxcar convolution, peak finding, and clustering)
based on the methodology described for the Heimdall pipeline (Barsdell et al.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sigpyproc.algorithms.common import (
    PulseDetections,
    _finite_positive,
    _positive_int,
    _samples,
    make_width_bank,
    pulse_candidates,
)
from sigpyproc.kernels import search as search_kernels

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import ArrayLike

    from sigpyproc.core.candidates import CandidateTable


@dataclass(frozen=True, slots=True, kw_only=True)
class HeimdallConfig:
    """Controls for the CPU single-series Heimdall pipeline.

    Parameters
    ----------
    gulp_size : int, optional
        Fresh samples per gulp, by default 262144. Must exceed max_width.
    baseline_length : float, optional
        Smoothing length in seconds, by default 2.0; its radius is
        floor(baseline_length / (2 * tsamp)).
    min_tscrunch_width : int, optional
        Power-of-two width at which filter output starts striding, by default
        4096. Stride is max(1, 2 * width // min_tscrunch_width).
    cand_sep_time : int, optional
        Giant gap and cluster time tolerance in units of filter width, default 3.
    cand_sep_filter : int, optional
        Maximum filter-index difference for clustering, by default 3.
    boxcar_renorm : bool, optional
        Re-estimate noise after filtering, by default False. Otherwise rescale
        boxcar sums by sqrt(width), as upstream's default does.
    """

    gulp_size: int = 262144
    baseline_length: float = 2.0
    min_tscrunch_width: int = 4096
    cand_sep_time: int = 3
    cand_sep_filter: int = 3
    boxcar_renorm: bool = False

    def __post_init__(self) -> None:
        _positive_int(self.gulp_size, "gulp_size", 2)
        _finite_positive(self.baseline_length, "baseline_length")
        _positive_int(self.min_tscrunch_width, "min_tscrunch_width", 2)
        _positive_int(self.cand_sep_time, "cand_sep_time", 0)
        _positive_int(self.cand_sep_filter, "cand_sep_filter", 0)
        if not isinstance(self.boxcar_renorm, (bool, np.bool_)):
            msg = "boxcar_renorm must be a boolean"
            raise ValueError(msg)  # noqa: TRY004 - consistent search validation
        if self.min_tscrunch_width & (self.min_tscrunch_width - 1):
            msg = "min_tscrunch_width must be a power of two"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class HeimdallPrepared:
    """A gulp's normalized samples, removed baseline, and approximate RMS."""

    data: np.ndarray
    baseline: np.ndarray
    noise: float


@dataclass(frozen=True, slots=True)
class HeimdallScores:
    """One width's SNRs on a strided grid of actual window-center sample bins."""

    offset: int
    width: int
    stride: int
    snrs: np.ndarray


@dataclass(frozen=True, slots=True)
class HeimdallGiants:
    """Giants or clusters, with exclusive ends and strongest-member bounds.

    ``members`` counts contributing width-giants, not threshold samples.
    ``begins`` and ``ends`` describe the threshold grid, not boxcar support.
    Arrays use observation-relative bins after applying the gulp offset.
    """

    bins: np.ndarray
    snrs: np.ndarray
    widths: np.ndarray
    begins: np.ndarray
    ends: np.ndarray
    members: np.ndarray

    @classmethod
    def empty(cls) -> HeimdallGiants:
        """Construct correctly typed empty giant arrays."""
        ints = np.empty(0, np.int64)
        return cls(
            ints,
            np.empty(0, np.float64),
            ints.copy(),
            ints.copy(),
            ints.copy(),
            ints.copy(),
        )


def _power_widths(widths: ArrayLike, max_width: int) -> np.ndarray:
    bank = make_width_bank(max_width, widths=widths)
    if np.any(bank & (bank - 1)):
        msg = "Heimdall trial widths must be powers of two"
        raise ValueError(msg)
    return bank


def preprocess_heimdall(
    data: ArrayLike,
    tsamp: float,
    *,
    baseline_length: float = 2.0,
) -> HeimdallPrepared:
    """Remove a median-of-five baseline and use Heimdall's approximate MAD.

    Parameters
    ----------
    data : array_like
        One finite, real gulp. Input is not modified.
    tsamp : float
        Positive sampling interval in seconds.
    baseline_length : float, optional
        Smoothing length in seconds, by default 2.0. Must span >= 2 samples.

    Returns
    -------
    HeimdallPrepared
        Float64 normalized data and baseline. Empty, constant, or zero-scale
        gulps have zero data/noise and produce no detections. A smoothing
        length with too few baseline knots leaves the baseline zero, as upstream.
    """
    _finite_positive(tsamp, "tsamp")
    _finite_positive(baseline_length, "baseline_length")
    radius = int(baseline_length / (2 * tsamp))
    _positive_int(radius, "baseline smoothing radius")
    arr = _samples(data).astype(np.float64)
    baseline = search_kernels.heimdall_baseline(arr, radius)
    centered = arr - baseline
    if arr.size == 0 or np.max(arr) == np.min(arr):
        noise = 0.0
    else:
        noise = float(search_kernels.heimdall_rms(centered))
    if noise > 0 and np.isfinite(noise):
        centered /= noise
    else:
        noise = 0.0
        centered[:] = 0
    return HeimdallPrepared(centered, baseline, noise)


def iter_heimdall_scores(
    prepared: HeimdallPrepared,
    widths: ArrayLike,
    *,
    max_width: int,
    min_tscrunch_width: int = 4096,
    boxcar_renorm: bool = False,
) -> Iterator[HeimdallScores]:
    """Score power-of-two boxcars over Heimdall's common max-width valid span.

    Parameters
    ----------
    prepared : HeimdallPrepared
        Normalized gulp. A zero-noise gulp yields no scores.
    widths : array_like
        Power-of-two trial widths <= max_width.
    max_width : int
        Common valid-span guard, even when not itself a trial width.
    min_tscrunch_width : int, optional
        Striding threshold, by default 4096 (power of two, >= 2).
    boxcar_renorm : bool, optional
        Normalize each filtered series with approximate MAD, default False.

    Yields
    ------
    HeimdallScores
        Owned float64 SNRs. First center is max_width // 2; all widths share
        the span of len(data) + 1 - max_width unstrided centers. Output stride
        is max(1, 2 * width // min_tscrunch_width).
    """
    _positive_int(max_width, "max_width")
    config = HeimdallConfig(min_tscrunch_width=min_tscrunch_width)
    bank = _power_widths(widths, max_width)
    arr = np.asarray(prepared.data, dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        msg = "prepared data must be a finite one-dimensional array"
        raise ValueError(msg)
    if max_width > arr.size:
        msg = "max_width exceeds the Heimdall gulp length"
        raise ValueError(msg)
    if not np.isfinite(prepared.noise) or prepared.noise < 0:
        msg = "prepared noise must be finite and nonnegative"
        raise ValueError(msg)
    if prepared.noise == 0:
        return
    prefix = np.empty(arr.size + 1, np.float64)
    prefix[0] = 0
    np.cumsum(arr, dtype=np.float64, out=prefix[1:])
    for width in bank:
        w = int(width)
        stride = max(1, 2 * w // config.min_tscrunch_width)
        filtered = search_kernels.heimdall_filter(prefix, max_width, w, stride)
        scale = (
            float(search_kernels.heimdall_rms(filtered))
            if boxcar_renorm
            else np.sqrt(w)
        )
        if scale > 0 and np.isfinite(scale):
            filtered /= scale
        else:
            filtered[:] = 0
        yield HeimdallScores(max_width // 2, w, stride, filtered)


def find_heimdall_giants(
    scores: HeimdallScores,
    *,
    snr_min: float = 6.0,
    cand_sep_time: int = 3,
) -> HeimdallGiants:
    """Group above-threshold samples connected by width-dependent gaps.

    Parameters
    ----------
    scores : HeimdallScores
        One width's scores. Offset and stride identify actual center bins.
    snr_min : float, optional
        Strict threshold (SNR > snr_min), by default 6.0.
    cand_sep_time : int, optional
        Gap tolerance in relative width units, by default 3. In score-grid
        samples the tolerance is cand_sep_time * (width // stride).

    Returns
    -------
    HeimdallGiants
        One first-maximum representative per connected giant, with exclusive
        grid bounds and one contributing member each.
    """
    _finite_positive(snr_min, "snr_min")
    _positive_int(cand_sep_time, "cand_sep_time", 0)
    _positive_int(scores.stride, "stride")
    _positive_int(scores.offset, "offset", 0)
    _power_widths([scores.width], scores.width)
    arr = np.asarray(scores.snrs, dtype=np.float64)
    if arr.ndim != 1:
        msg = "scores must be one-dimensional"
        raise ValueError(msg)
    gap = cand_sep_time * (scores.width // scores.stride)
    peaks, snrs, begins, ends = search_kernels.heimdall_giants(arr, snr_min, gap)
    return HeimdallGiants(
        scores.offset + peaks * scores.stride,
        snrs,
        np.full(peaks.size, scores.width, np.int64),
        scores.offset + begins * scores.stride,
        scores.offset + ends * scores.stride,
        np.ones(peaks.size, np.int64),
    )


def _validate_giants(giants: HeimdallGiants) -> None:
    arrays = (giants.bins, giants.widths, giants.begins, giants.ends, giants.members)
    if any(
        a.ndim != 1
        or a.shape != giants.snrs.shape
        or not np.issubdtype(a.dtype, np.integer)
        for a in arrays
    ):
        msg = "giants require equal-length integer arrays and SNRs"
        raise ValueError(msg)
    if (
        not np.all(np.isfinite(giants.snrs))
        or np.any(giants.bins < 0)
        or np.any(giants.begins < 0)
        or np.any(giants.begins > giants.bins)
        or np.any(giants.ends <= giants.bins)
        or np.any(giants.members < 1)
    ):
        msg = "giants require finite SNRs, valid exclusive bounds, and positive members"
        raise ValueError(msg)
    if giants.widths.size:
        _power_widths(giants.widths, int(np.max(giants.widths)))


def cluster_heimdall_giants(
    giants: HeimdallGiants,
    *,
    cand_sep_time: int = 3,
    cand_sep_filter: int = 3,
) -> HeimdallGiants:
    """Merge connected components in time/filter space for a single DM gulp.

    Parameters
    ----------
    giants : HeimdallGiants
        Raw giants, in any order; inputs are not modified.
    cand_sep_time : int, optional
        Connection radius multiplier: tolerance * max(widths), by default 3.
    cand_sep_filter : int, optional
        Maximum difference in log2(width), by default 3.

    Returns
    -------
    HeimdallGiants
        Strongest member per transitive component, with its original bounds
        and summed member counts. Ties prefer earlier bin, then narrower width.
    """
    _positive_int(cand_sep_time, "cand_sep_time", 0)
    _positive_int(cand_sep_filter, "cand_sep_filter", 0)
    _validate_giants(giants)
    if giants.bins.size == 0:
        return HeimdallGiants.empty()
    order = np.argsort(giants.bins, kind="stable")
    bins, widths = giants.bins[order], giants.widths[order]
    labels = search_kernels.heimdall_cluster_labels(
        bins, widths, cand_sep_time, cand_sep_filter
    )
    snrs = giants.snrs[order]
    priority = np.lexsort((widths, bins, -snrs))
    # One pass in deterministic priority order selects each component's winner.
    winners: dict[int, int] = {}
    for i in priority:
        winners.setdefault(int(labels[i]), int(i))
    selected = np.array(list(winners.values()), np.int64)
    member_sums = np.zeros(labels.size, np.int64)
    np.add.at(member_sums, labels, giants.members[order])
    original = order[selected]
    return HeimdallGiants(
        giants.bins[original],
        giants.snrs[original],
        giants.widths[original],
        giants.begins[original],
        giants.ends[original],
        member_sums[labels[selected]],
    )


def _concatenate_giants(parts: list[HeimdallGiants]) -> HeimdallGiants:
    if not parts:
        return HeimdallGiants.empty()
    return HeimdallGiants(
        *(
            np.concatenate([getattr(p, name) for p in parts])
            for name in ("bins", "snrs", "widths", "begins", "ends", "members")
        )
    )


def iter_heimdall_gulps(
    data: ArrayLike,
    *,
    gulp_size: int = 262144,
    max_width: int = 256,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield observation offsets and raw gulps with max-width overlap.

    The first gulp contains gulp_size samples; subsequent gulps retain the
    preceding max_width samples and read up to gulp_size fresh samples. The
    final partial gulp is searched if it has enough data. No padding is added.
    Yielded data are read-only views; normalization never modifies them.
    """
    _positive_int(gulp_size, "gulp_size", 2)
    _positive_int(max_width, "max_width")
    if gulp_size <= max_width:
        msg = "gulp_size must exceed max_width"
        raise ValueError(msg)
    arr = _samples(data)
    if arr.size and arr.size < max_width:
        msg = "max_width exceeds the time series length"
        raise ValueError(msg)
    start, end = 0, min(gulp_size, arr.size)
    while end - start >= max_width:
        view = arr[start:end].view()
        view.setflags(write=False)
        yield start, view
        if end == arr.size:
            break
        start, end = end - max_width, min(arr.size, end + gulp_size)


def heimdall_candidates(
    giants: HeimdallGiants,
    tsamp: float,
    *,
    dm: float = 0.0,
    accel: float = 0.0,
) -> CandidateTable:
    """Convert giants to a pulse table, retaining filter, bounds, and members.

    Parameters
    ----------
    giants : HeimdallGiants
        Clustered or raw giants with physical-sample indices and widths.
    tsamp : float
        Positive sample interval in seconds.
    dm : float, optional
        Dispersion measure attached to each row, by default 0.0.
    accel : float, optional
        Acceleration attached to each row, by default 0.0.

    Returns
    -------
    CandidateTable
        SNR-sorted pulse rows with additional integer columns filter_index,
        begin_bin, end_bin, and members. Bounds belong to the strongest member.
    """
    _validate_giants(giants)
    order = np.lexsort((giants.bins, giants.widths, -giants.snrs))
    result = pulse_candidates(
        PulseDetections(giants.bins[order], giants.snrs[order], giants.widths[order]),
        tsamp,
        dm=dm,
        accel=accel,
    )
    result.table["filter_index"] = np.array(
        [int(w).bit_length() - 1 for w in giants.widths[order]], np.int64
    )
    result.table["begin_bin"] = giants.begins[order]
    result.table["end_bin"] = giants.ends[order]
    result.table["members"] = giants.members[order]
    return result


def heimdall_pulse_search(
    data: ArrayLike,
    tsamp: float,
    *,
    snr_min: float = 6.0,
    max_width: int = 256,
    widths: ArrayLike | None = None,
    config: HeimdallConfig | None = None,
    dm: float = 0.0,
    accel: float = 0.0,
) -> CandidateTable:
    """Search an already dedispersed single series with CPU Heimdall stages.

    GPU dedispersion, RFI excision, multi-DM/beam coincidence, and candidate-rate
    aborts are outside this adaptation. Clusters are formed within each gulp.
    The common max-width guard intentionally excludes observation-edge centers
    even for narrow filters. The final valid remainder is searched safely.
    """
    if config is not None and not isinstance(config, HeimdallConfig):
        msg = "config must be a HeimdallConfig"
        raise ValueError(msg)
    config = HeimdallConfig() if config is None else config
    _finite_positive(tsamp, "tsamp")
    _finite_positive(snr_min, "snr_min")
    _positive_int(max_width, "max_width")
    bank = (
        make_width_bank(max_width, style="powers_of_two")
        if widths is None
        else _power_widths(widths, max_width)
    )
    _positive_int(
        int(config.baseline_length / (2 * tsamp)), "baseline smoothing radius"
    )
    if config.gulp_size <= max_width:
        msg = "gulp_size must exceed max_width"
        raise ValueError(msg)
    arr = _samples(data)
    groups: list[HeimdallGiants] = []
    for offset, gulp in iter_heimdall_gulps(
        arr, gulp_size=config.gulp_size, max_width=max_width
    ):
        prepared = preprocess_heimdall(
            gulp, tsamp, baseline_length=config.baseline_length
        )
        parts: list[HeimdallGiants] = []
        for scores in iter_heimdall_scores(
            prepared,
            bank,
            max_width=max_width,
            min_tscrunch_width=config.min_tscrunch_width,
            boxcar_renorm=config.boxcar_renorm,
        ):
            snrs = scores.snrs
            # Adjacent valid spans share one center. Assign it to the next gulp
            # to avoid reporting the identical grid point twice.
            if (
                offset + gulp.size < arr.size
                and (gulp.size - max_width) % scores.stride == 0
            ):
                snrs = snrs[:-1]
            global_scores = HeimdallScores(
                offset + scores.offset, scores.width, scores.stride, snrs
            )
            found = find_heimdall_giants(
                global_scores, snr_min=snr_min, cand_sep_time=config.cand_sep_time
            )
            if found.bins.size:
                parts.append(found)
        if parts:
            groups.append(
                cluster_heimdall_giants(
                    _concatenate_giants(parts),
                    cand_sep_time=config.cand_sep_time,
                    cand_sep_filter=config.cand_sep_filter,
                )
            )
    return heimdall_candidates(_concatenate_giants(groups), tsamp, dm=dm, accel=accel)
