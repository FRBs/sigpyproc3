"""Dispersion related parameters, delays, and trial planning.

This module contains ISM dispersion delay and smearing calculations, dispersion
constants, and trial dispersion measure tables (DMPlan).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
from astropy import constants, units

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from sigpyproc.header import Header

DM_CONSTANT_LK = 4.148808e3  # L&K Handbook of Pulsar Astronomy
DM_CONSTANT_MT = 1 / 0.000241  # TEMPO2 Manchester & Taylor (1972)
DM_CONSTANT_SI = (
    (constants.e.esu**2 / (2 * np.pi * constants.m_e * constants.c)).to(  # ty: ignore[unresolved-attribute]
        units.s * units.MHz**2 * units.cm**3 / units.pc,
    )
).value  # Precise SI constants

K_SIGPROC = 8.3
"""Dispersion smearing constant used by SIGPROC ``getDMtable.C`` (µs, MHz, GHz)."""


def _delay_inputs(
    freqs: ArrayLike,
    dm: float | ArrayLike,
    tsamp: float,
    ref_freq: float,
    index: float,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Return checked frequencies, DM rows, whether DM was scalar, and the reference."""
    if not np.isfinite(index) or index == 0.0:
        msg = f"dispersion index must be finite and non-zero, got {index}"
        raise ValueError(msg)
    if not np.isfinite(tsamp) or tsamp <= 0:
        msg = f"tsamp must be positive and finite, got {tsamp}"
        raise ValueError(msg)
    ref = float(ref_freq)
    if not np.isfinite(ref) or ref <= 0:
        msg = f"ref_freq must be positive and finite, got {ref_freq}"
        raise ValueError(msg)
    freqs_arr = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    if freqs_arr.ndim != 1 or freqs_arr.size == 0:
        msg = "freqs must be a non-empty 1-D array"
        raise ValueError(msg)
    if not np.all(np.isfinite(freqs_arr)) or np.any(freqs_arr <= 0):
        msg = "freqs must be positive and finite"
        raise ValueError(msg)
    dm_input = np.asarray(dm, dtype=np.float64)
    if dm_input.ndim > 1:
        msg = "dm must be a scalar or a 1-D array"
        raise ValueError(msg)
    dm_arr = np.atleast_1d(dm_input)
    if not np.all(np.isfinite(dm_arr)):
        msg = "dm must be finite"
        raise ValueError(msg)
    return freqs_arr, dm_arr, dm_input.ndim == 0, ref


def compute_dmdelays(
    freqs: ArrayLike,
    dm: float | ArrayLike,
    tsamp: float,
    ref_freq: float,
    *,
    in_samples: bool = True,
    index: float = -2.0,
) -> NDArray[np.int32 | np.float32]:
    """Compute ISM dispersion delays for given DM value(s).

    Parameters
    ----------
    freqs : ArrayLike
        Frequencies of each channel.
    dm : float | ArrayLike
        Dispersion measure(s) to calculate delays for.
    tsamp : float
        Sampling time of the data.
    ref_freq : float
        Reference frequency to calculate delays from.
    in_samples : bool, optional
        Flag to return delays as numbers of samples, by default True.
    index : float, optional
        Exponent of frequency in the delay law, by default -2. ``-2`` is the
        cold-plasma law and matches the historical formula exactly.

    Returns
    -------
    NDArray[int32 | float32]
        Dispersion delays at middle of each channel with respect to reference frequency.

        If dm is a scalar, returns a 1D array of delays. If dm is an array,
        returns a 2D array with shape ``(len(dm), len(freqs))``.
    """
    freqs_arr, dm_arr, scalar_dm, ref = _delay_inputs(
        freqs,
        dm,
        tsamp,
        ref_freq,
        index,
    )
    if index == -2.0:
        phase = (freqs_arr**-2) - (ref**-2)
    else:
        phase = (-2.0 / index) * (freqs_arr**index - (ref**index))
    delays = dm_arr[:, np.newaxis] * DM_CONSTANT_LK * phase
    if in_samples:
        samples = delays / tsamp
        if not np.all(np.isfinite(samples)):
            msg = "dispersion delay is not finite"
            raise ValueError(msg)
        rounded = np.rint(samples)
        limit = np.iinfo(np.int32).max
        if np.any(np.abs(rounded) > limit):
            msg = "dispersion delay does not fit in a 32-bit sample count"
            raise ValueError(msg)
        out = rounded.astype(np.int32)
    else:
        out = delays.astype(np.float32)
    if scalar_dm:
        return out.reshape(freqs_arr.size)
    return out.reshape(dm_arr.size, freqs_arr.size)


def align_delays(delays: ArrayLike) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Shift delays so the smallest value on the frequency axis is zero.

    Parameters
    ----------
    delays : ArrayLike
        Dispersion delays in samples. The last axis is frequency.

    Returns
    -------
    aligned : NDArray[int32]
        ``delays - min(delays)`` along the frequency axis.
    lead : NDArray[int32]
        ``-min(delays)`` for each dispersion-measure row. Output sample 0
        corresponds to this many input samples after the start of the read.
        A 1-D input returns a length-1 array.
    """
    arr = np.asarray(delays, dtype=np.int64)
    lo = np.min(arr, axis=-1, keepdims=True)
    aligned = np.ascontiguousarray(arr - lo, dtype=np.int32)
    lead = np.ascontiguousarray((-lo).reshape(-1), dtype=np.int32)
    return aligned, lead


def compute_dmsmearing(
    freqs: ArrayLike,
    dm: float | ArrayLike,
    tsamp: float,
    *,
    in_samples: bool = True,
    index: float = -2.0,
) -> NDArray[np.int32 | np.float32]:
    """Compute ISM smearing due to finite bandwidth for given DM value(s).

    Parameters
    ----------
    freqs : ArrayLike
        Frequencies of each channel.
    dm : float | ArrayLike
        Dispersion measure(s) to calculate smearing for.
    tsamp : float
        Sampling time of the data.
    in_samples : bool, optional
        Flag to return smearing as numbers of samples, by default True.
    index : float, optional
        Exponent of frequency in the delay law, by default -2. Smearing is the
        absolute derivative of that delay across one channel.

    Returns
    -------
    NDArray[int32 | float32]
        DM smearing in the frequency channels due to finite bandwidth.

        If dm is a scalar, returns a 1D array of smearing. If dm is an array,
        returns a 2D array with shape ``(len(dm), len(freqs))``.
    """
    if index == 0.0:
        msg = "dispersion index must be non-zero"
        raise ValueError(msg)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    dm = np.atleast_1d(dm)[:, np.newaxis].astype(np.float64)
    foff = float(np.abs(freqs[1] - freqs[0])) if freqs.size > 1 else 0.0
    if index == -2.0:
        smearing = 2 * DM_CONSTANT_LK * foff * dm / freqs**3
    else:
        smearing = np.abs(2 * DM_CONSTANT_LK * foff * dm * freqs ** (index - 1.0))
    if in_samples:
        smearing = (smearing / tsamp).round().astype(np.int32)
    else:
        smearing = smearing.astype(np.float32)
    return smearing.squeeze()


@attrs.define(auto_attribs=True, slots=True)
class DMPlan:
    """A table of trial dispersion measures.

    Parameters
    ----------
    dms : ndarray
        Trial DMs in pc cm^-3.
    """

    dms: np.ndarray

    def __len__(self) -> int:
        return int(self.dms.size)

    @classmethod
    def from_sigproc_units(
        cls,
        dm_min: float,
        dm_max: float,
        tsamp_us: float,
        ti_us: float,
        bw_mhz: float,
        cfreq_ghz: float,
        nchan: int,
        tol: float = 1.25,
    ) -> DMPlan:
        """Build a DM table using SIGPROC mixed units (verbatim ``getDMtable.C``)."""
        if dm_max < dm_min:
            msg = f"dm_max ({dm_max}) must be >= dm_min ({dm_min})"
            raise ValueError(msg)
        if nchan <= 0:
            msg = "nchan must be positive"
            raise ValueError(msg)
        bw = abs(float(bw_mhz))
        dms: list[float] = []
        dm = float(dm_min)
        k = K_SIGPROC
        while dm <= dm_max + 1e-12:
            dms.append(dm)
            t00 = np.sqrt(tsamp_us**2 + ti_us**2 + (k * bw * dm / cfreq_ghz**3) ** 2)
            a = k * bw / cfreq_ghz**3
            b = k * nchan * bw / 4.0 / cfreq_ghz**3
            c = tol * tol * t00 * t00 - tsamp_us * tsamp_us - ti_us * ti_us
            disc = -a * a * b * b * dm * dm + a * a * c + b * b * c
            if disc < 0 or (a * a + b * b) == 0:
                msg = "DM plan recurrence failed (negative discriminant)"
                raise ValueError(msg)
            new_dm = (b * b * dm + np.sqrt(disc)) / (a * a + b * b)
            if new_dm <= dm:
                msg = f"DM plan recurrence stalled at DM={dm}"
                raise ValueError(msg)
            dm = float(new_dm)
            if len(dms) > 1_000_000:
                msg = "DM plan produced too many trials"
                raise ValueError(msg)
        return cls(dms=np.asarray(dms, dtype=np.float64))

    @classmethod
    def from_header(
        cls,
        header: Header,
        dm_max: float,
        dm_min: float = 0.0,
        *,
        ti: float = 40e-6,
        tol: float = 1.25,
        nsub: int = 1,
    ) -> DMPlan:
        """Build a DM table from a header.

        ``ti`` is in seconds. Internally converted to the mixed units used by
        SIGPROC ``getDMtable.C``.
        """
        if nsub < 1:
            msg = "nsub must be >= 1"
            raise ValueError(msg)
        tsamp_us = header.tsamp * 1e6
        ti_us = ti * 1e6
        bw_mhz = abs(header.foff)
        cfreq_ghz = (header.fch1 + (header.nchans / 2 - 0.5) * header.foff) / 1000.0
        nchan = header.nchans // nsub
        return cls.from_sigproc_units(
            dm_min,
            dm_max,
            tsamp_us,
            ti_us,
            bw_mhz,
            cfreq_ghz,
            nchan,
            tol,
        )
