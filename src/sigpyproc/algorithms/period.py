"""Period search algorithms on plain 1D arrays.

This module contains functions for finding periodic signals in 1D arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sigpyproc.core import stats
from sigpyproc.core.candidates import CandidateTable

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


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
