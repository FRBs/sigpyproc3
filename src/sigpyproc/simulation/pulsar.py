"""Generate fake filterbank data with noise, pulsars, or single pulses.

A modern counterpart of SIGPROC ``fake``: Gaussian noise, an optional
rectangular periodic pulse, or a single dispersed pulse. Signal is added
to unit Gaussian noise and then quantized with
:meth:`~sigpyproc.io.bits.BitsInfo.quantize` (no running rescale), so
injected SNR is not squashed.

Binary orbits, red noise, even/odd channel tests, and headerless output
are not implemented. Use :mod:`~sigpyproc.simulation.furby` for FRBs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attrs
import numpy as np

from sigpyproc.header import Header
from sigpyproc.io.bits import BitsInfo

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SPEED_OF_LIGHT = 299792458.0
_DEFAULT_DUTY = 0.04


@attrs.frozen(auto_attribs=True, kw_only=True)
class PulsarParams:
    """Signal parameters for a fake pulsar or single pulse.

    Parameters
    ----------
    period : float, optional
        Spin period in seconds. Mutually exclusive with ``pulse_time``.
    pulse_time : float, optional
        Single-pulse arrival time at ``fch1`` in seconds from the start.
    width : float, optional
        Intrinsic pulse width in seconds. Defaults to 4% of ``period``
        for a periodic pulsar; required for a single pulse.
    dm : float, optional
        Dispersion measure, by default 0.
    snr : float, optional
        Single-pulse peak SNR before dividing by ``sqrt(nchans)``,
        by default 1.0 (SIGPROC ``-snrpeak``).
    pdot : float, optional
        Period derivative (s/s), by default 0.
    accel : float, optional
        Line-of-sight acceleration in m/s^2, by default 0. If set,
        it takes precedence over ``pdot`` and ``period`` is the value
        at mid-observation.
    smear : bool, optional
        Broaden the width by DM smearing and sampling, by default True.
    """

    period: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    pulse_time: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0.0)),
    )
    width: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    dm: float = 0.0
    snr: float = 1.0
    pdot: float = 0.0
    accel: float = 0.0
    smear: bool = True

    def __attrs_post_init__(self) -> None:
        if self.period is not None and self.pulse_time is not None:
            msg = "period and pulse_time are mutually exclusive"
            raise ValueError(msg)

    def resolved_width(self) -> float:
        """Intrinsic width in seconds after applying the pulsar default."""
        if self.period is None and self.pulse_time is None:
            return 0.0
        if self.width is not None:
            return self.width
        if self.period is not None:
            return _DEFAULT_DUTY * self.period
        msg = "width is required for a single pulse"
        raise ValueError(msg)


def make_header(
    *,
    filename: str = "fake.fil",
    nchans: int = 128,
    nbits: int = 4,
    tsamp: float = 80e-6,
    tobs: float = 10.0,
    tstart: float = 50000.0,
    fch1: float = 433.968,
    foff: float = -0.062,
    source: str = "Fake",
    dm: float = 0.0,
    period: float = 0.0,
) -> Header:
    """Build a SIGPROC-like filterbank header (defaults match ``fake``)."""
    if tsamp <= 0.0:
        msg = "tsamp must be positive"
        raise ValueError(msg)
    if tobs <= 0.0:
        msg = "tobs must be positive"
        raise ValueError(msg)
    if nchans < 1:
        msg = "nchans must be at least 1"
        raise ValueError(msg)
    nsamples = round(tobs / tsamp)
    if nsamples < 1:
        msg = "tobs / tsamp must yield at least one sample"
        raise ValueError(msg)
    data_type = "time series" if nchans == 1 else "filterbank"
    return Header(
        filename=filename,
        data_type=data_type,
        nchans=nchans,
        foff=foff,
        fch1=fch1,
        nbits=nbits,
        tsamp=tsamp,
        tstart=tstart,
        nsamples=nsamples,
        source=source,
        rawdatafile=filename,
        dm=dm,
        period=period,
    )


def fake_filterbank(
    header: Header,
    params: PulsarParams | None = None,
    outfile: str | None = None,
    *,
    seed: int | None = None,
    gulp: int = 16384,
) -> str:
    """Write a fake filterbank of Gaussian noise plus an optional signal.

    Parameters
    ----------
    header : Header
        Observing metadata (``nsamples``, ``nchans``, ``nbits``, band).
    params : PulsarParams, optional
        Signal parameters. ``None`` writes noise only.
    outfile : str, optional
        Output path, by default ``header.filename``.
    seed : int, optional
        ``numpy`` Generator seed. ``None`` uses the OS entropy.
    gulp : int, optional
        Samples written per block, by default 16384.

    Returns
    -------
    str
        Path of the written filterbank.
    """
    if params is None:
        params = PulsarParams()
    bitsinfo = _check_fake(header, params, gulp)
    width = params.resolved_width()

    if outfile is None:
        outfile = header.filename
    updates: dict[str, Any] = {"filename": outfile, "rawdatafile": outfile}
    if params.dm:
        updates["dm"] = params.dm
    if params.period:
        updates["period"] = params.period
    header = header.new_header(updates)
    rng = np.random.default_rng(seed)
    delays = np.atleast_1d(
        header.get_dmdelays(params.dm, in_samples=False),
    ).astype(np.float64)
    width_eff = _effective_width(header, params, width)
    snr_chan = params.snr / np.sqrt(header.nchans)

    with header.prep_outfile(outfile, rescale=False) as out:
        start = 0
        while start < header.nsamples:
            nsamp = min(gulp, header.nsamples - start)
            block = rng.standard_normal((nsamp, header.nchans), dtype=np.float32)
            if width_eff > 0.0 and (
                params.period is not None or params.pulse_time is not None
            ):
                times = (start + np.arange(nsamp, dtype=np.float64)) * header.tsamp
                block += _signal_block(
                    times,
                    delays,
                    params,
                    width_eff,
                    snr_chan,
                    header.tobs,
                )
            if header.nbits < 32:
                block = bitsinfo.quantize(block)
            out.cwrite(np.ravel(block))
            start += nsamp
    return outfile


def _check_fake(header: Header, params: PulsarParams, gulp: int) -> BitsInfo:
    if gulp < 1:
        msg = "gulp must be at least 1"
        raise ValueError(msg)
    if header.nsamples < 1:
        msg = "header.nsamples must be at least 1"
        raise ValueError(msg)
    bitsinfo = BitsInfo(header.nbits)
    if bitsinfo.unpack and (header.nchans * header.nbits) % 8 != 0:
        msg = (
            f"nchans * nbits must be a multiple of 8 for {header.nbits}-bit data, "
            f"got nchans={header.nchans}"
        )
        raise ValueError(msg)
    if (
        params.accel != 0.0
        and (1.0 + params.accel * header.tobs / _SPEED_OF_LIGHT) <= 0.0
    ):
        msg = "acceleration would drive the period through zero"
        raise ValueError(msg)
    if (
        params.period is not None
        and params.pdot != 0.0
        and (params.period + params.pdot * header.tobs) <= 0.0
    ):
        msg = "pdot would drive the period through zero"
        raise ValueError(msg)
    return bitsinfo


def _effective_width(header: Header, params: PulsarParams, width: float) -> float:
    if width <= 0.0:
        return 0.0
    if not params.smear:
        return width
    if header.nchans == 1:
        tdm = 0.0
    else:
        tdm = float(np.max(header.get_dmsmearing(params.dm, in_samples=False)))
    return float(np.hypot(np.hypot(tdm, header.tsamp), width))


def _signal_block(
    times: NDArray[np.float64],
    delays: NDArray[np.float64],
    params: PulsarParams,
    width_eff: float,
    snr_chan: float,
    tobs: float,
) -> NDArray[np.float32]:
    t_eff = times[:, np.newaxis] + delays[np.newaxis, :]
    if params.period is not None:
        period0 = params.period
        if params.accel != 0.0:
            period0 = params.period / (
                1.0 + params.accel * tobs / (2.0 * _SPEED_OF_LIGHT)
            )
        phase = np.mod(_pulse_phase(t_eff, period0, params.pdot, params.accel), 1.0)
        duty = min(width_eff / params.period, 1.0)
        rising = 0.5 - 0.5 * duty
        trailing = 0.5 + 0.5 * duty
        mask = (phase >= rising) & (phase <= trailing)
    else:
        pulse_time = params.pulse_time
        if pulse_time is None:
            msg = "pulse_time is required for a single pulse"
            raise ValueError(msg)
        arrival = pulse_time + delays
        mask = np.abs(t_eff - arrival) <= (0.5 * width_eff)
    return mask.astype(np.float32) * np.float32(snr_chan)


def _pulse_phase(
    t_eff: NDArray[np.float64],
    period: float,
    pdot: float,
    accel: float,
) -> NDArray[np.float64]:
    if accel != 0.0:
        return np.log1p(accel * t_eff / _SPEED_OF_LIGHT) * (
            _SPEED_OF_LIGHT / (period * accel)
        )
    if pdot != 0.0:
        return np.log1p((pdot / period) * t_eff) / pdot
    return t_eff / period
