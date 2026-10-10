"""Period and pulse search algorithms on plain arrays.

The public search methods live on the data objects:
:meth:`~sigpyproc.timeseries.TimeSeries.search`,
:meth:`~sigpyproc.timeseries.TimeSeries.find_pulses`,
:meth:`~sigpyproc.timeseries.TimeSeries.ffa`, and
:meth:`~sigpyproc.fourierseries.PowerSpectrum.find_peaks`. They validate
their arguments and do the object-level work (downsampling, resampling,
folding), then call the functions here. Each function returns a
:class:`~sigpyproc.candidates.CandidateTable`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import typed

from sigpyproc.candidates import CandidateTable
from sigpyproc.core import kernels, stats
from sigpyproc.core.filters import MatchedFilter, Template

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

FFA_MAX_NSAMP = 1 << 24
FFA_WIDTH_REBINS = (1, 2, 4, 8, 16)


def pick_spectral_peaks(
    spec: ArrayLike,
    tobs: float,
    *,
    npeaks: int,
    snr_min: float,
    pmax: float,
    dm: float,
    accel: float,
    nharm: int,
) -> CandidateTable:
    """Pick local maxima of a whitened spectrum.

    SNR in bin ``i`` is ``(spec[i] - med) / (1.4826 * MAD)`` over a running
    window of ``max(128, nfreq // 64)`` bins. The DC bin and every bin with
    period ``> pmax`` are skipped. A local maximum is strictly greater than
    both neighbours.

    Parameters
    ----------
    spec : array_like
        Spectrum with bin ``r`` at frequency ``r / tobs``.
    tobs : float
        Observation length in seconds that maps bins to frequencies.
    npeaks : int
        Maximum number of peaks to return.
    snr_min : float
        Minimum SNR.
    pmax : float
        Longest period to keep, in seconds.
    dm : float
        Dispersion measure stored on each row.
    accel : float
        Acceleration stored on each row.
    nharm : int
        Harmonic fold that produced ``spec`` (1 = unfolded).

    Returns
    -------
    CandidateTable
        Up to ``npeaks`` rows with ``kind="fft"``, sorted by SNR.
    """
    spec = np.asarray(spec, dtype=np.float64)
    nfreq = spec.size
    window = max(128, nfreq // 64)
    if window >= nfreq:
        window = nfreq if nfreq % 2 == 1 else max(nfreq - 1, 1)
    if window < 3:
        med = np.full(nfreq, np.median(spec))
        mad = np.full(nfreq, np.median(np.abs(spec - med[0])))
    else:
        med = np.asarray(
            stats.running_filter(spec, window, "median"),
            dtype=np.float64,
        )
        mad = np.asarray(
            stats.running_filter(np.abs(spec - med), window, "median"),
            dtype=np.float64,
        )
    scale = np.where(mad > 0.0, 1.4826 * mad, 1.0)
    snr = (spec - med) / scale

    r_min = max(1, int(np.ceil(tobs / pmax)))
    r_max = nfreq - 1
    if r_min >= r_max:
        return CandidateTable.empty()

    local = (snr[1:-1] > snr[:-2]) & (snr[1:-1] > snr[2:]) & (snr[1:-1] >= snr_min)
    peak_bins = np.flatnonzero(local) + 1
    peak_bins = peak_bins[(peak_bins >= r_min) & (peak_bins < r_max)]
    if peak_bins.size == 0:
        return CandidateTable.empty()

    peak_snr = snr[peak_bins]
    order = np.argsort(-peak_snr)[:npeaks]
    peak_bins = peak_bins[order]
    peak_snr = peak_snr[order]
    freq = peak_bins / tobs
    rows = [
        {
            "period_s": 1.0 / f_i,
            "freq_hz": f_i,
            "snr": float(s_i),
            "dm": dm,
            "accel": accel,
            "nharm": nharm,
            "bin": int(r_i),
            "width_s": np.nan,
            "kind": "fft",
        }
        for r_i, s_i, f_i in zip(peak_bins, peak_snr, freq, strict=True)
    ]
    return CandidateTable.from_rows(rows).sort_by_snr()


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
            },
        )
        lo = max(0, peak - deadtime)
        hi = min(nsamp, peak + deadtime + 1)
        best_snr[lo:hi] = -np.inf
    return CandidateTable.from_rows(rows).sort_by_snr()


def ffa_search(
    data: ArrayLike,
    tsamp: float,
    pmin: float,
    pmax: float,
    *,
    snr_min: float,
    zero_pad: bool,
    dm: float,
    accel: float,
) -> CandidateTable:
    """Fast Folding Analysis of one time series (Staelin 1969 / Mueller).

    For every integer trial period ``P`` (in bins) between ``pmin`` and
    ``pmax``, the series is reshaped to ``(nsamp // P, P)`` and folded with
    the FFA tree in :func:`~sigpyproc.core.kernels.ffa_stage`. The robust
    SNR of each output profile is computed and the best one is kept.

    The input is capped at ``FFA_MAX_NSAMP`` samples. One candidate is kept
    per integer trial period. The fractional FFA offset is used only when it
    clearly outranks the unshifted fold (by more than 10 %): wide pulses make
    every tree shift look equally loud, so the integer period is then the
    honest answer.

    Parameters
    ----------
    data : array_like
        Time series samples, already rebinned if required.
    tsamp : float
        Sampling interval in seconds of ``data``.
    pmin : float
        Shortest trial period in seconds.
    pmax : float
        Longest trial period in seconds.
    snr_min : float
        Minimum profile SNR.
    zero_pad : bool
        Pad the series with zeros to the next power of two (after the cap).
    dm : float
        Dispersion measure stored on each row.
    accel : float
        Acceleration stored on each row.

    Returns
    -------
    CandidateTable
        FFA candidates (``kind="ffa"``), one per integer trial period.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.size > FFA_MAX_NSAMP:
        arr = arr[:FFA_MAX_NSAMP]
    if zero_pad:
        n_pow2 = 1 << (arr.size - 1).bit_length()
        if n_pow2 > arr.size:
            arr = np.pad(arr, (0, n_pow2 - arr.size))

    pmin_bins = max(2, round(pmin / tsamp))
    pmax_bins = min(arr.size // 2, round(pmax / tsamp))
    if pmax_bins < pmin_bins:
        return CandidateTable.empty()

    rows: list[dict[str, object]] = []
    for p_bins in range(pmin_bins, pmax_bins + 1):
        n_rows = arr.size // p_bins
        if n_rows < 2:
            continue
        block = np.ascontiguousarray(arr[: n_rows * p_bins].reshape(n_rows, p_bins))
        profiles = kernels.ffa_stage(block)
        snrs = kernels.ffa_profile_snr(profiles)
        j_best = int(np.argmax(snrs))
        snr_best = float(snrs[j_best])
        if snr_best < snr_min:
            continue
        n_pad = profiles.shape[0]
        if snr_best <= 1.1 * float(snrs[0]):
            p_eff = p_bins * tsamp
        else:
            p_eff = (p_bins + j_best / n_pad) * tsamp
        rows.append(
            {
                "period_s": p_eff,
                "freq_hz": 1.0 / p_eff,
                "snr": snr_best,
                "dm": dm,
                "accel": accel,
                "nharm": 1,
                "bin": p_bins,
                "width_s": tsamp,
                "kind": "ffa",
            },
        )
    return CandidateTable.from_rows(rows).sort_by_snr()
