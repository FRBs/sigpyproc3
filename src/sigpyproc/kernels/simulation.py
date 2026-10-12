"""Numba kernels for pulse profiles, ISM broadening, and injection.

Smearing is a per-channel boxcar and scattering is a per-channel truncated
exponential. Both are applied in the time domain. Dispersion is not applied
here: it is a shift, done by :func:`inject_pulses` or
:func:`sigpyproc.kernels.dedispersion.disperse_block`.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _smear_extra(smear: float) -> int:
    """Extra full-convolution samples from a boxcar of width ``smear``."""
    if smear <= 1.0:
        return 0
    length = int(np.ceil(smear))
    if length <= 1:
        return 0
    return length - 1


@njit(cache=True, fastmath=True)
def _scatter_extra(tau: float) -> int:
    """Extra samples from a 6 e-folding exponential of timescale ``tau``."""
    if tau <= 0.5:
        return 0
    length = int(np.ceil(6.0 * tau))
    if length <= 1:
        return 0
    return length - 1


@njit(cache=True, fastmath=True)
def smear_lead(dm_smear: np.ndarray) -> int:
    """Maximum fine samples a centered smearing boxcar shifts to the left."""
    lead = 0
    for ichan in range(len(dm_smear)):
        extra = _smear_extra(dm_smear[ichan])
        lead = max(lead, extra // 2)
    return lead


@njit(cache=True, fastmath=True)
def _boxcar_centered(
    src: np.ndarray,
    n_cur: int,
    length: int,
    lead_offset: int,
    dst: np.ndarray,
) -> int:
    """Convolve ``src[:n_cur]`` with a normalised rect of ``length`` samples, centered.

    The rect spreads ``lead_offset = (length - 1) // 2`` samples to the left and
    ``length - 1 - lead_offset`` samples to the right.
    """
    prefix = np.zeros(n_cur + 1, dtype=np.float64)
    for i in range(n_cur):
        prefix[i + 1] = prefix[i] + src[i]
    n_new = n_cur + length - 1
    inv = 1.0 / length
    for i in range(n_new):
        out_idx = i - lead_offset
        if out_idx < 0:
            continue
        left = max(i + 1 - length, 0)
        right = min(i + 1, n_cur)
        dst[out_idx] = np.float32((prefix[right] - prefix[left]) * inv)
    return n_new - lead_offset


@njit(cache=True, fastmath=True)
def _scatter_full(src: np.ndarray, n_cur: int, tau: float, dst: np.ndarray) -> int:
    """Convolve ``src[:n_cur]`` with a causal truncated exponential.

    The kernel is ``exp(-t / tau)`` for ``t = 0 .. T-1`` with ``T = ceil(6 tau)``,
    normalised by its sum. An IIR plus a delayed correction matches that
    truncated convolution.
    """
    length = max(int(np.ceil(6.0 * tau)), 1)
    n_new = n_cur + length - 1
    decay = np.exp(-1.0 / tau)
    tail = np.exp(-length / tau)
    denom = 1.0 - decay
    norm = float(length) if denom < 1e-12 else (1.0 - tail) / denom
    inv = 1.0 / norm
    iir = np.zeros(n_new, dtype=np.float64)
    previous = 0.0
    for i in range(n_new):
        sample = src[i] if i < n_cur else np.float32(0.0)
        previous = sample + decay * previous
        iir[i] = previous
    for i in range(n_new):
        corrected = iir[i]
        if i >= length:
            corrected = corrected - tail * iir[i - length]
        dst[i] = np.float32(corrected * inv)
    return n_new


@njit(cache=True, fastmath=True, parallel=True)
def simulate_ism(
    signal: np.ndarray,
    spectrum: np.ndarray,
    dm_smear: np.ndarray,
    tau_nus: np.ndarray,
    over_sampling: int = 1,
) -> np.ndarray:
    """Broaden one intrinsic pulse by per-channel smearing and scattering.

    Smearing is centered around zero delay so channel arrival times retain their
    nominal cold-plasma dispersion delay. Scattering is a causal truncated
    exponential tail.

    Parameters
    ----------
    signal : ndarray
        Intrinsic pulse, shape ``(nsamps,)``.
    spectrum : ndarray
        Multiplicative gain of each channel, shape ``(nchans,)``.
    dm_smear : ndarray
        Intra-channel dispersion smearing in samples, shape ``(nchans,)``.
        Widths of 1 sample or less are left unchanged.
    tau_nus : ndarray
        Scattering timescale in samples, shape ``(nchans,)``. Values of half
        a sample or less are left unchanged.
    over_sampling : int, optional
        Accepted for compatibility. Kernel lengths come from ``dm_smear`` and
        ``tau_nus``.

    Returns
    -------
    ndarray
        Float32 array of shape ``(nchans, n_out)``. Sample 0 is the start of
        the earliest broadened support.
    """
    if signal.ndim != 1 or spectrum.ndim != 1:
        msg = "signal and spectrum must be 1D"
        raise ValueError(msg)
    if dm_smear.ndim != 1 or tau_nus.ndim != 1:
        msg = "dm_smear and tau_nus must be 1D"
        raise ValueError(msg)
    nchans = spectrum.shape[0]
    if dm_smear.shape[0] != nchans or tau_nus.shape[0] != nchans:
        msg = "spectrum, dm_smear, and tau_nus must have the same length"
        raise ValueError(msg)
    sig = signal.astype(np.float32)
    spec = spectrum.astype(np.float32)
    smear = dm_smear.astype(np.float64)
    taus = tau_nus.astype(np.float64)
    n_sig = sig.shape[0]
    if n_sig == 0 or nchans == 0:
        return np.zeros((nchans, 0), dtype=np.float32)

    lead = smear_lead(smear)
    extra_right = 0
    for ichan in range(nchans):
        sm_extra = _smear_extra(smear[ichan])
        sm_right = sm_extra - (sm_extra // 2)
        chan_extra = sm_right + _scatter_extra(taus[ichan])
        extra_right = max(extra_right, chan_extra)
    n_out = lead + n_sig + extra_right
    out = np.zeros((nchans, n_out), dtype=np.float32)

    for ichan in prange(nchans):
        work = np.zeros(n_out, dtype=np.float32)
        held = np.zeros(n_out, dtype=np.float32)
        gain = spec[ichan]
        for isamp in range(n_sig):
            work[lead + isamp] = sig[isamp] * gain
        n_cur = lead + n_sig
        smear_len = _smear_extra(smear[ichan]) + 1
        if smear_len > 1:
            lead_offset = (smear_len - 1) // 2
            n_cur = _boxcar_centered(work, n_cur, smear_len, lead_offset, held)
            for isamp in range(n_cur):
                work[isamp] = held[isamp]
        if taus[ichan] > 0.5:
            n_cur = _scatter_full(work, n_cur, taus[ichan], held)
            for isamp in range(n_cur):
                work[isamp] = held[isamp]
        for isamp in range(n_cur):
            out[ichan, isamp] = work[isamp]
    return out


@njit(cache=True, fastmath=True, parallel=True)
def downsample_profiles(fine: np.ndarray, os_fact: int) -> np.ndarray:
    """Average groups of ``os_fact`` samples along the last axis.

    Parameters
    ----------
    fine : ndarray
        Profile array of shape ``(nchans, n_fine)``.
    os_fact : int
        Number of fine samples per output sample. Must be at least 1.

    Returns
    -------
    ndarray
        Float32 array of shape ``(nchans, n_fine // os_fact)``. Remainder
        samples at the end of each row are dropped.
    """
    if fine.ndim != 2:
        msg = "fine profiles must be 2D"
        raise ValueError(msg)
    if os_fact < 1:
        msg = "os_fact must be at least 1"
        raise ValueError(msg)
    nchans, n_fine = fine.shape
    n_out = n_fine // os_fact
    out = np.empty((nchans, n_out), dtype=np.float32)
    inv = 1.0 / os_fact
    for ichan in prange(nchans):
        for isamp in range(n_out):
            base = isamp * os_fact
            acc = 0.0
            for k in range(os_fact):
                acc += fine[ichan, base + k]
            out[ichan, isamp] = np.float32(acc * inv)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def inject_pulse_2d(
    block: np.ndarray,
    profile: np.ndarray,
    delays: np.ndarray,
    start: int,
    scale: float = 1.0,
) -> None:
    """Add one dedispersed profile into a time-major block, in place.

    Parameters
    ----------
    block : ndarray
        Float32 array of shape ``(nsamp, nchan)``, modified in place.
    profile : ndarray
        Float32 array of shape ``(nchan, nprof)``.
    delays : ndarray
        Float64 array of shape ``(nchan,)``. A positive delay moves
        that channel later. The fractional part is a two-tap interpolation.
    start : int
        Native sample, relative to ``block`` row 0, where profile index 0 sits
        when the delay is 0.
    scale : float, optional
        Amplitude multiplier, by default 1.0.
    """
    if block.ndim != 2 or profile.ndim != 2:
        msg = "block and profile must be 2D"
        raise ValueError(msg)
    nsamp, nchan = block.shape
    if profile.shape[0] != nchan or delays.shape[0] != nchan:
        msg = "profile and delay channel counts must match the block"
        raise ValueError(msg)
    length = profile.shape[1]
    if length <= 0:
        return
    for ichan in prange(nchan):
        origin = start + delays[ichan]
        i_lo = max(int(np.floor(origin - 1.0)), 0)
        i_hi = min(int(np.floor(origin + length)) + 1, nsamp)
        for isamp in range(i_lo, i_hi):
            coord = isamp - origin
            i0 = int(np.floor(coord))
            frac = coord - i0
            value = 0.0
            if 0 <= i0 < length:
                value += (1.0 - frac) * profile[ichan, i0]
            nxt = i0 + 1
            if 0 <= nxt < length:
                value += frac * profile[ichan, nxt]
            block[isamp, ichan] = np.float32(block[isamp, ichan] + scale * value)


@njit(cache=True, fastmath=True, parallel=True)
def inject_pulses(
    block: np.ndarray,
    profiles: np.ndarray,
    delays: np.ndarray,
    starts: np.ndarray,
    scales: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """Add dedispersed pulses into a time-major block, in place.

    Parameters
    ----------
    block : ndarray
        Float32 array of shape ``(nsamp, nchan)``, modified in place.
    profiles : ndarray
        Float32 array of shape ``(npulse, nchan, nprof)``.
    delays : ndarray
        Float64 array of shape ``(npulse, nchan)``. A positive delay moves
        that channel later. The fractional part is a two-tap interpolation.
    starts : ndarray
        Int64 array of shape ``(npulse,)``. Native sample, relative to
        ``block`` row 0, where profile index 0 sits when the delay is 0.
    scales : ndarray
        Float32 array of shape ``(npulse,)``.
    lengths : ndarray
        Int64 array of shape ``(npulse,)``. Valid samples in each profile.
    """
    if block.ndim != 2 or profiles.ndim != 3:
        msg = "block must be 2D and profiles 3D"
        raise ValueError(msg)
    nsamp, nchan = block.shape
    npulse = profiles.shape[0]
    if profiles.shape[1] != nchan or delays.shape[0] != npulse:
        msg = "profile and delay channel counts must match the block"
        raise ValueError(msg)
    for ichan in prange(nchan):
        for ipulse in range(npulse):
            length = lengths[ipulse]
            if length <= 0:
                continue
            origin = starts[ipulse] + delays[ipulse, ichan]
            scale = scales[ipulse]
            i_lo = max(int(np.floor(origin - 1.0)), 0)
            i_hi = min(int(np.floor(origin + length)) + 1, nsamp)
            for isamp in range(i_lo, i_hi):
                coord = isamp - origin
                i0 = int(np.floor(coord))
                frac = coord - i0
                value = 0.0
                if 0 <= i0 < length:
                    value += (1.0 - frac) * profiles[ipulse, ichan, i0]
                nxt = i0 + 1
                if 0 <= nxt < length:
                    value += frac * profiles[ipulse, ichan, nxt]
                block[isamp, ichan] = np.float32(block[isamp, ichan] + scale * value)
