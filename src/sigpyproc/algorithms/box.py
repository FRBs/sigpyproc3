"""Boxcar pulse search algorithms on plain 1D arrays.

This module contains functions for finding multiple pulses with a boxcar bank.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import typed

from sigpyproc import kernels
from sigpyproc.algorithms.common import (
    _finite_positive,
    _positive_int,
    _samples,
    make_width_bank,
)
from sigpyproc.algorithms.heimdall import heimdall_pulse_search
from sigpyproc.algorithms.sps import presto_pulse_search
from sigpyproc.core import stats
from sigpyproc.core.candidates import CandidateTable
from sigpyproc.core.filters import MatchedFilter, Template

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from sigpyproc.algorithms.heimdall import HeimdallConfig
    from sigpyproc.core.custom_types import (
        BoxcarEngine,
        PulseDetrend,
        PulseMethod,
        WidthStyle,
    )


def boxcar_pulse_search(
    data: ArrayLike,
    tsamp: float,
    *,
    snr_min: float,
    max_width: int,
    spacing_factor: float,
    deadtime: int,
    dm: float,
    accel: float,
    widths: np.ndarray | None = None,
) -> CandidateTable:
    """Find multiple pulses with a boxcar bank and a greedy dead-time.

    The data are robustly normalised, convolved with boxcars whose widths
    are geometric from 1 to ``max_width``, and the best width per sample is
    kept. Peaks are accepted in decreasing SNR; ``deadtime`` bins on either
    side of each accepted peak are then excluded.

    Parameters
    ----------
    data : array_like
        Time series samples.
    tsamp : float
        Sampling interval in seconds, used to convert widths to seconds.
    snr_min : float
        Minimum matched-filter SNR.
    max_width : int
        Widest boxcar in bins.
    spacing_factor : float
        Geometric spacing of the width bank.
    deadtime : int
        Bins excluded on either side of an accepted peak.
    dm : float
        Dispersion measure stored on each row.
    accel : float
        Acceleration stored on each row.
    widths : ndarray, optional
        Explicit trial widths, otherwise generated geometrically.

    Returns
    -------
    CandidateTable
        Pulse candidates (``kind="pulse"``), sorted by SNR.

    Raises
    ------
    ValueError
        If the widest boxcar is longer than the data.
    """
    arr = np.asarray(data)
    zscores = stats.estimate_zscore(arr, loc_method="median", scale_method="iqr")
    if widths is None:
        widths = MatchedFilter.get_box_width_spacing(max_width, spacing_factor)
    templates = [Template.gen_boxcar(int(width)) for width in widths]
    if any(temp.data.size > arr.size for temp in templates):
        msg = "max_width is larger than the time series"
        raise ValueError(msg)
    convs = kernels.convolve_templates(
        zscores.data,
        typed.List([temp.data for temp in templates]),
        typed.List([temp.ref_bin for temp in templates]),
    )
    best_idx = np.argmax(convs, axis=0)
    best_snr = np.max(convs, axis=0).astype(np.float64, copy=True)

    rows: list[dict[str, object]] = []
    nsamp = best_snr.size
    while True:
        peak = int(np.argmax(best_snr))
        peak_snr = float(best_snr[peak])
        if peak_snr < snr_min or not np.isfinite(peak_snr):
            break
        width_bins = int(widths[best_idx[peak]])
        rows.append(
            {
                "period_s": np.nan,
                "freq_hz": np.nan,
                "snr": peak_snr,
                "dm": dm,
                "accel": accel,
                "nharm": 1,
                "bin": peak,
                "width_s": width_bins * tsamp,
                "kind": "pulse",
                "bin_origin": "start",
            },
        )
        lo = max(0, peak - deadtime)
        hi = min(nsamp, peak + deadtime + 1)
        best_snr[lo:hi] = -np.inf
    return CandidateTable.from_rows(rows).sort_by_snr()


def pulse_search(  # noqa: PLR0913 - common parameters plus PRESTO controls
    data: ArrayLike,
    tsamp: float,
    *,
    method: PulseMethod = "presto",
    engine: BoxcarEngine = "running_sum",
    snr_min: float = 6.0,
    max_width: int = 256,
    width_style: WidthStyle | None = None,
    widths: ArrayLike | None = None,
    spacing_factor: float = 2.0,
    deadtime: int | None = None,
    block_size: int = 8000,
    detrend: PulseDetrend = "linear",
    mask_bad_blocks: bool = False,
    chunk_size: int = 8192,
    dm: float = 0.0,
    accel: float = 0.0,
    heimdall_config: HeimdallConfig | None = None,
) -> CandidateTable:
    """Search one time series using selectable boxcar pipelines.

    PRESTO uses blockwise robust normalization, centered full-resolution
    boxcars, and width-dependent suppression. ``engine``, ``block_size``,
    ``detrend``, ``mask_bad_blocks``, and ``chunk_size`` control that method.
    Matched filter preserves the original global normalization, circular
    convolution, start-bin alignment, and fixed deadtime (8 by default).

    ``max_width`` bounds trial widths, not the preset's final entry. Explicit
    widths must be positive integers <= max_width and override width_style.
    Empty input returns an empty table. For nonempty input every selected
    width must fit the observation. Nonfinite data and invalid options raise
    ValueError. Inputs are never modified.

    Parameters
    ----------
    data : array_like
        Finite, real, one-dimensional time series.
    tsamp : float
        Positive sampling interval in seconds.
    method : {"presto", "matched_filter", "heimdall"}, optional
        Pipeline, by default "presto".
    engine : {"running_sum", "fft"}, optional
        PRESTO scorer, by default "running_sum".
    snr_min : float, optional
        Positive minimum SNR, by default 6.0.
    max_width : int, optional
        Trial width ceiling in samples, by default 256.
    width_style : {"presto", "geometric", "powers_of_two"}, optional
        Defaults to the PRESTO bank for PRESTO, geometric for matched filter,
        and powers of two for Heimdall.
    widths : array_like, optional
        Explicit integer widths, overriding width_style.
    spacing_factor : float, optional
        Geometric spacing (> 1), by default 2.0.
    deadtime : int, optional
        Minimum exclusion radius in samples; default 0 for PRESTO, 8 for
        matched filter. PRESTO also suppresses by width.
    block_size : int, optional
        PRESTO preprocessing block length in samples, by default 8000.
    detrend : {"linear", "median"}, optional
        PRESTO baseline removal, by default "linear".
    mask_bad_blocks : bool, optional
        Reject PRESTO noise-outlier blocks, by default False.
    chunk_size : int, optional
        PRESTO scoring chunk length in samples, by default 8192.
    dm : float, optional
        Dispersion measure attached to each row, by default 0.0.
    accel : float, optional
        Acceleration attached to each row, by default 0.0.
    heimdall_config : HeimdallConfig, optional
        CPU Heimdall gulp, baseline, striding, and clustering controls. Only
        valid with method="heimdall". Its bank defaults to powers of two.
        PRESTO-specific controls and fixed deadtime do not apply to this method.

    Returns
    -------
    CandidateTable
        Pulse candidates in SNR priority order. PRESTO and CPU Heimdall report
        actual window-center bins; matched filter reports start bins.

    Raises
    ------
    ValueError
        If samples or options are invalid, or a selected width cannot fit.
    """
    _check_search_choice(
        method, ("presto", "matched_filter", "heimdall"), "pulse search method"
    )
    _check_search_choice(engine, ("running_sum", "fft"), "boxcar engine")
    _finite_positive(tsamp, "tsamp")
    _finite_positive(snr_min, "snr_min")
    _positive_int(block_size, "block_size", 2)
    _positive_int(chunk_size, "chunk_size")
    if detrend not in {"linear", "median"}:
        msg = f"unknown detrend method: {detrend}"
        raise ValueError(msg)
    exclusion = (
        (8 if method == "matched_filter" else 0) if deadtime is None else deadtime
    )
    _positive_int(exclusion, "deadtime", 0)
    default_styles: dict[PulseMethod, WidthStyle] = {
        "presto": "presto",
        "matched_filter": "geometric",
        "heimdall": "powers_of_two",
    }
    style = default_styles[method] if width_style is None else width_style
    bank = make_width_bank(
        max_width, style=style, spacing_factor=spacing_factor, widths=widths
    )
    if method == "heimdall":
        _check_heimdall_controls(
            engine=engine,
            deadtime=deadtime,
            block_size=block_size,
            detrend=detrend,
            mask_bad_blocks=mask_bad_blocks,
            chunk_size=chunk_size,
        )
        return heimdall_pulse_search(
            data,
            tsamp,
            snr_min=snr_min,
            max_width=max_width,
            widths=bank,
            config=heimdall_config,
            dm=dm,
            accel=accel,
        )
    if heimdall_config is not None:
        msg = "heimdall_config is only valid for method='heimdall'"
        raise ValueError(msg)
    arr = _samples(data)
    if arr.size == 0:
        return CandidateTable.empty()
    if bank[-1] > arr.size:
        msg = (
            "selected widths exceed the time series length; "
            "reduce max_width or supply widths"
        )
        raise ValueError(msg)
    if method == "matched_filter":
        return boxcar_pulse_search(
            arr,
            tsamp,
            snr_min=snr_min,
            max_width=max_width,
            spacing_factor=spacing_factor,
            deadtime=exclusion,
            dm=dm,
            accel=accel,
            widths=bank,
        )
    return presto_pulse_search(
        arr,
        tsamp,
        widths=bank,
        engine=engine,
        snr_min=snr_min,
        deadtime=exclusion,
        block_size=block_size,
        detrend=detrend,
        mask_bad_blocks=mask_bad_blocks,
        chunk_size=chunk_size,
        dm=dm,
        accel=accel,
    )


def _check_search_choice(value: str, choices: tuple[str, ...], name: str) -> None:
    if value not in choices:
        msg = f"unknown {name}: {value}"
        raise ValueError(msg)


def _check_heimdall_controls(
    *,
    engine: BoxcarEngine,
    deadtime: int | None,
    block_size: int,
    detrend: PulseDetrend,
    mask_bad_blocks: bool,
    chunk_size: int,
) -> None:
    if engine != "running_sum" or deadtime is not None:
        msg = (
            "Heimdall uses strided running sums and clustering; "
            "engine='fft' and deadtime do not apply"
        )
        raise ValueError(msg)
    if (
        block_size != 8000
        or detrend != "linear"
        or mask_bad_blocks
        or chunk_size != 8192
    ):
        msg = "use heimdall_config for Heimdall preprocessing and gulp controls"
        raise ValueError(msg)
