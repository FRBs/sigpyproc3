"""Period search algorithms on plain 1D arrays.

This module contains functions for finding periodic signals in 1D arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sigpyproc import kernels
from sigpyproc.core import stats
from sigpyproc.core.candidates import CandidateTable

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

FFA_MAX_NSAMP = 1 << 24
FFA_WIDTH_REBINS = (1, 2, 4, 8, 16)


def _spectral_running_window(nfreq: int) -> int:
    window = max(128, nfreq // 64)
    if window >= nfreq:
        return nfreq if nfreq % 2 == 1 else max(nfreq - 1, 1)
    return window


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

    SNR in bin ``i`` uses :func:`sigpyproc.core.stats.running_robust_standardize`
    over a running window of ``max(128, nfreq // 64)`` bins. The DC bin and every
    bin with period ``> pmax`` are skipped. A local maximum is strictly greater than
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
    if nfreq < 3:
        return CandidateTable.empty()
    snr = stats.running_robust_standardize(spec, _spectral_running_window(nfreq)).data

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
    the FFA tree in ``sigpyproc.kernels.folding.ffa_stage``. The robust
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
