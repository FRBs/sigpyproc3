"""Numba kernels for interstellar medium (ISM) propagation simulations."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from sigpyproc.kernels.fourier import fftconvolve


@njit(cache=True, fastmath=True, parallel=True)
def simulate_ism(
    signal: np.ndarray,
    spectrum: np.ndarray,
    dm_smear: np.ndarray,
    tau_nus: np.ndarray,
    over_sampling: int = 1,
) -> np.ndarray:
    """Convolve the input signal with the ISM effects."""
    nsamps_pure = len(signal)
    nchans = len(spectrum)
    nsamps_smear = int(max(over_sampling, np.ceil(dm_smear.max())))
    nsamps_scat = int(max(over_sampling, np.ceil(6.0 * float(tau_nus.max()))))
    max_len = nsamps_pure + nsamps_smear + nsamps_scat - 2
    final_arr = np.zeros((nchans, max_len), dtype=signal.dtype)

    x_scat = np.arange(nsamps_scat, dtype=signal.dtype)
    do_smear = dm_smear.max() > 0
    do_scatter = tau_nus.max() > 0

    for ichan in prange(nchans):
        chan_data = signal * spectrum[ichan]
        if do_smear:
            dm_smear_samps = int(max(1, np.ceil(dm_smear[ichan])))
            dm_smear_prof = np.zeros(nsamps_smear, dtype=signal.dtype)
            dm_smear_prof[:dm_smear_samps] = 1
            dm_smear_prof /= dm_smear_prof.sum()
            chan_data = fftconvolve(chan_data, dm_smear_prof)
        if do_scatter:
            scat_prof = np.exp(-x_scat / tau_nus[ichan])
            scat_prof /= scat_prof.sum()
            chan_data = fftconvolve(chan_data, scat_prof)
        nwrite = min(max_len, chan_data.size)
        final_arr[ichan, :nwrite] = chan_data[:nwrite]
    return final_arr
