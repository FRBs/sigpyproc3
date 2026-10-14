"""Stamp compact furby profiles into filterbank blocks and files.

Pulses are stored dedispersed. Dispersion delays are applied when adding each
profile into a time-major block via 2D profile streaming. Integer files are
converted with either the fixed digitiser (matching fake_filterbank output) or
an empirical per-channel bandpass estimator (preserving original telescope
baseline and scale).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np

from sigpyproc.io.digifil import digitiser_params, quantise_rows
from sigpyproc.kernels.bitpacking import pack_array
from sigpyproc.kernels.simulation import inject_pulse_2d
from sigpyproc.readers import FilReader, open_filterbank
from sigpyproc.simulation.furby import Furby, FurbyGenerator, PulseParams

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sigpyproc.header import Header
    from sigpyproc.io.fileio import FileWriter


@attrs.frozen(auto_attribs=True, kw_only=True)
class InjectedPulse:
    """One dedispersed profile and where its first sample belongs.

    Parameters
    ----------
    profile : ndarray
        Channel-major float32 array, shape ``(nchans, nprof)``.
    delays : ndarray
        Delay of each channel in native samples. Positive arrives later.
    start : int
        Native sample of profile index 0 when the delay is zero.
    scale : float, optional
        Extra multiplier, by default 1. The furby profile is already scaled.
    """

    profile: np.ndarray = attrs.field(eq=False)
    delays: np.ndarray = attrs.field(eq=False)
    start: int
    scale: float = 1.0

    def __attrs_post_init__(self) -> None:
        if self.profile.ndim != 2:
            msg = "profile must have shape (nchans, nprof)"
            raise ValueError(msg)
        if self.delays.shape != (self.profile.shape[0],):
            msg = "delays must have one entry per channel"
            raise ValueError(msg)


class Injector:
    """Add one or more pulses into successive filterbank gulps.

    Parameters
    ----------
    pulses : sequence of InjectedPulse
        Pulses to add. An empty sequence adds nothing.
    """

    def __init__(self, pulses: Sequence[InjectedPulse]) -> None:
        self._pulses = tuple(pulses)
        if not pulses:
            self._nchan = 0
            self._min_delays = np.zeros(0, dtype=np.float64)
            self._max_delays = np.zeros(0, dtype=np.float64)
            return
        self._nchan = int(pulses[0].profile.shape[0])
        min_delays = []
        max_delays = []
        for pulse in pulses:
            if pulse.profile.shape[0] != self._nchan:
                msg = "all pulses must have the same number of channels"
                raise ValueError(msg)
            min_delays.append(float(np.min(pulse.delays)))
            max_delays.append(float(np.max(pulse.delays)))
        self._min_delays = np.asarray(min_delays, dtype=np.float64)
        self._max_delays = np.asarray(max_delays, dtype=np.float64)

    @property
    def pulses(self) -> tuple[InjectedPulse, ...]:
        """Sequence of injected pulses."""
        return self._pulses

    @classmethod
    def from_furbies(
        cls,
        furbies: Sequence[Furby],
        arrival_samples: Sequence[int],
    ) -> Injector:
        """Build an injector from compact furbies.

        Parameters
        ----------
        furbies : sequence of Furby
            Dedispersed profiles. Each profile already includes its SNR scale.
        arrival_samples : sequence of int
            Native sample of each reference-frequency arrival.

        Returns
        -------
        Injector
            Injector that places ``arrival + furby.sample0`` at profile index 0.
        """
        if len(furbies) != len(arrival_samples):
            msg = "each furby needs one arrival sample"
            raise ValueError(msg)
        pulses = [
            InjectedPulse(
                profile=np.ascontiguousarray(pulse.block.data, dtype=np.float32),
                delays=np.asarray(pulse.delays, dtype=np.float64),
                start=int(arrival) + int(pulse.sample0),
                scale=1.0,
            )
            for pulse, arrival in zip(furbies, arrival_samples, strict=True)
        ]
        return cls(pulses)

    def overlaps(self, nsamples: int) -> bool:
        """Return whether any pulse intersects samples ``[0, nsamples)``."""
        for pulse, min_d, max_d in zip(
            self._pulses, self._min_delays, self._max_delays, strict=True
        ):
            begin = pulse.start + int(np.floor(min_d)) - 1
            end = pulse.start + int(np.ceil(max_d)) + int(pulse.profile.shape[1]) + 1
            if end > 0 and begin < nsamples:
                return True
        return False

    def add_into(self, block: np.ndarray, start_sample: int) -> None:
        """Add every overlapping pulse into a time-major block.

        Parameters
        ----------
        block : ndarray
            Float32 array of shape ``(nsamp, nchans)``, modified in place.
            Values are in units of the per-channel noise rms.
        start_sample : int
            Native sample index of ``block`` row 0 in the observation.
        """
        if not self._pulses:
            return
        if block.ndim != 2 or block.shape[1] != self._nchan:
            msg = f"block must have shape (nsamp, {self._nchan})"
            raise ValueError(msg)
        nsamp = block.shape[0]
        for pulse, min_d, max_d in zip(
            self._pulses, self._min_delays, self._max_delays, strict=True
        ):
            length = pulse.profile.shape[1]
            if length <= 0:
                continue
            rel_start = pulse.start - int(start_sample)
            p_lo = rel_start + min_d - 1.0
            p_hi = rel_start + max_d + length + 1.0
            if p_hi <= 0.0 or p_lo >= nsamp:
                continue
            inject_pulse_2d(
                block,
                pulse.profile,
                pulse.delays,
                rel_start,
                pulse.scale,
            )


def read_pulse_catalogue(
    path: str | Path,
) -> list[tuple[Any, ...]]:
    """Read pulse parameter rows from a text or CSV file.

    Blank lines and ``#`` comments are ignored. Each row must have at least
    4 numbers: ``time dm width snr``, optionally followed by ``spec_idx``,
    ``tau0``, and ``spec_kind``.

    Parameters
    ----------
    path : str | Path
        Text or CSV file.

    Returns
    -------
    list of tuple
        Tuple of floats for each row: ``(time_s, dm, width_s, snr, ...)``.

    Raises
    ------
    ValueError
        If a row does not have at least four numbers, or the file has no pulses.
    """
    rows: list[tuple[Any, ...]] = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cleaned = line.replace(",", " ")
        parts = cleaned.split()
        if len(parts) < 4:
            msg = f"{path}:{lineno} expected at least 'time dm width snr'"
            raise ValueError(msg)
        n_num = min(len(parts), 6)
        try:
            num_values = [float(part) for part in parts[:n_num]]
        except ValueError as exc:
            msg = f"{path}:{lineno} invalid numeric value: {exc}"
            raise ValueError(msg) from exc
        if len(parts) > 6:
            values: tuple[Any, ...] = (*num_values, parts[6])
        else:
            values = tuple(num_values)
        rows.append(values)
    if not rows:
        msg = f"{path} contains no pulses"
        raise ValueError(msg)
    return rows


def injector_from_rows(
    header: Header,
    rows: Sequence[tuple[Any, ...]],
    template: PulseParams,
    *,
    ref_freq: str = "fch1",
    seed: int | None = None,
) -> Injector:
    """Generate one furby per catalogue row and pack them into an injector.

    Parameters
    ----------
    header : Header
        Band and sampling time used for every pulse.
    rows : sequence of tuple
        ``(time, dm, width, snr, [spec_idx], [tau0], [spec_kind])``.
    template : PulseParams
        Shared shape, spectrum, scattering, and oversampling.
    ref_freq : str, optional
        Arrival reference frequency, by default ``"fch1"``.
    seed : int, optional
        Spectral-structure seed. The same seed is used for every row.

    Returns
    -------
    Injector
        Pulses placed at ``round(time / tsamp)``.
    """
    furbies: list[Furby] = []
    arrivals: list[int] = []
    for row in rows:
        time_s, dm, width, snr = row[0], row[1], row[2], row[3]
        extra_kwargs: dict[str, Any] = {}
        if len(row) > 4:
            extra_kwargs["spec_idx"] = float(row[4])
        if len(row) > 5:
            extra_kwargs["tau0"] = float(row[5])
        if len(row) > 6:
            extra_kwargs["spec_kind"] = str(row[6])
        params = attrs.evolve(template, dm=dm, width=width, snr=snr, **extra_kwargs)
        generated = FurbyGenerator(
            header,
            params,
            ref_freq=ref_freq,
            seed=seed,
        ).generate()
        furbies.append(generated)
        arrivals.append(int(np.rint(time_s / header.tsamp)))
    return Injector.from_furbies(furbies, arrivals)


def write_sigma_block(writer: FileWriter, block: np.ndarray, nbits: int) -> None:
    """Digitise one time-major block of unit-variance samples and write it.

    ``nbits`` of 32 writes float32. Narrower samples use the dspsr digitiser
    mean and scale for unit-variance input, with no running rescale. 1-, 2-,
    and 4-bit samples are packed least-significant-bit first.

    Parameters
    ----------
    writer : FileWriter
        Open sigproc writer. The header is already written.
    block : ndarray
        Float32 array of shape ``(nsamp, nchans)``.
    nbits : int
        Output bits per sample.
    """
    rows = np.ascontiguousarray(block, dtype=np.float32)
    if nbits >= 32:
        writer.cwrite(rows.ravel())
        return
    mean, scale = digitiser_params(
        nbits,
        rescale_on=True,
        scale_in=1.0,
        scale_fac=1.0,
    )
    means = np.full(rows.shape[0], mean, dtype=np.float32)
    packed = quantise_rows(rows, nbits, flip=False, means=means, scale=scale)
    writer.write(np.ascontiguousarray(packed).tobytes())


def estimate_bandpass(
    reader: FilReader,
    nsamples: int = 16384,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-channel median and standard deviation from the data."""
    nsamp = min(reader.header.nsamples, nsamples)
    block = reader.read_block(0, nsamp).data.astype(np.float32)
    means = np.median(block, axis=1)
    diff = np.abs(block - means[:, None])
    stds = np.median(diff, axis=1) * np.float32(1.4826)
    stds = np.where(stds > 1e-6, stds, np.float32(1.0))
    return means.astype(np.float32), stds.astype(np.float32)


def write_bandpass_block(
    writer: FileWriter,
    block: np.ndarray,
    nbits: int,
    means: np.ndarray,
    stds: np.ndarray,
    *,
    signed: bool = False,
) -> None:
    """Scale sigma block by channel bandpass, quantize and write."""
    rows = np.ascontiguousarray(block, dtype=np.float32)
    reconstructed = rows * stds + means
    if nbits >= 32:
        writer.cwrite(reconstructed.ravel())
        return
    if nbits == 8:
        if signed:
            clipped_s8 = np.clip(np.rint(reconstructed), -128, 127).astype(np.int8)
            writer.write(clipped_s8.tobytes())
        else:
            clipped_u8 = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)
            writer.write(clipped_u8.tobytes())
        return
    if nbits == 16:
        if signed:
            clipped_s16 = np.clip(np.rint(reconstructed), -32768, 32767).astype("<i2")
            writer.write(clipped_s16.tobytes())
        else:
            clipped_u16 = np.clip(np.rint(reconstructed), 0, 65535).astype("<u2")
            writer.write(clipped_u16.tobytes())
        return
    digi_max = (1 << nbits) - 1
    clipped = np.clip(np.rint(reconstructed), 0, digi_max).astype(np.uint8)
    if nbits in {1, 2, 4}:
        packed = np.zeros(clipped.size // (8 // nbits), dtype=np.uint8)
        pack_array(clipped.ravel(), nbits, packed, "little")
        writer.write(packed.tobytes())


def to_sigma(block: np.ndarray, nbits: int) -> np.ndarray:
    """Invert the fixed digitiser, returning unit-variance float32 samples.

    32-bit input is cast to float32. Narrower input is treated as unsigned
    levels from :func:`~sigpyproc.io.digifil.digitiser_params` with rescaling on.

    Parameters
    ----------
    block : ndarray
        Time-major samples.
    nbits : int
        Bits per sample in ``block``.

    Returns
    -------
    ndarray
        Float32 array of the same shape.
    """
    values = np.asarray(block, dtype=np.float32)
    if nbits >= 32:
        return np.ascontiguousarray(values)
    mean, scale = digitiser_params(
        nbits,
        rescale_on=True,
        scale_in=1.0,
        scale_fac=1.0,
    )
    return np.ascontiguousarray((values - mean) / np.float32(scale))


def inject_file(
    infile: str | Path,
    outfile: str | Path,
    injector: Injector,
    *,
    gulp: int = 16384,
    dm: float | None = None,
    preserve_bandpass: bool = False,
    estimate_samples: int = 16384,
) -> str:
    """Add pulses to a sigproc filterbank in one streaming pass.

    When ``preserve_bandpass`` is False, integer samples are converted to sigma
    with the fixed digitiser (mean and scale for unit-variance noise, matching
    files written by :func:`~sigpyproc.simulation.pulsar.fake_filterbank`).
    When ``preserve_bandpass`` is True, per-channel median and noise scale are
    measured from the input data, so pulses can be added to real telescope
    filterbanks without corrupting the original bandpass.

    Parameters
    ----------
    infile : str | Path
        Input sigproc filterbank.
    outfile : str | Path
        Output path.
    injector : Injector
        Pulses to add, in sigma units.
    gulp : int, optional
        Samples per read, by default 16384.
    dm : float, optional
        Dispersion measure to store in the output header.
    preserve_bandpass : bool, optional
        Estimate and preserve channel-by-channel bandpass, by default False.
    estimate_samples : int, optional
        Number of samples used to estimate the channel bandpass, by default 16384.

    Returns
    -------
    str
        Output path.

    Raises
    ------
    TypeError
        If ``infile`` is not a sigproc filterbank.
    """
    reader = open_filterbank(infile)
    try:
        if not isinstance(reader, FilReader):
            msg = "inject reads sigproc filterbanks"
            raise TypeError(msg)
        return _inject_open(
            reader,
            outfile,
            injector,
            gulp=gulp,
            dm=dm,
            preserve_bandpass=preserve_bandpass,
            estimate_samples=estimate_samples,
        )
    finally:
        reader.close()


def _inject_open(
    reader: FilReader,
    outfile: str | Path,
    injector: Injector,
    *,
    gulp: int,
    dm: float | None,
    preserve_bandpass: bool,
    estimate_samples: int,
) -> str:
    header = reader.header
    updates: dict[str, object] = {
        "filename": str(outfile),
        "rawdatafile": str(outfile),
    }
    if dm is not None:
        updates["dm"] = dm
    nchan = header.nchans
    start = 0

    means: np.ndarray | None = None
    stds: np.ndarray | None = None
    if preserve_bandpass:
        means, stds = estimate_bandpass(reader, nsamples=estimate_samples)

    is_signed = bool(getattr(header, "signed", False))
    with header.prep_outfile(
        str(outfile),
        updates=updates,
        nbits=header.nbits,
        rescale=False,
    ) as writer:
        for nsamps_r, _index, data in reader.read_plan(gulp=gulp, quiet=True):
            count = nsamps_r * nchan
            if preserve_bandpass and means is not None and stds is not None:
                raw = (
                    np.array(data[:count], copy=True)
                    .reshape(nsamps_r, nchan)
                    .astype(np.float32)
                )
                block = (raw - means) / stds
                injector.add_into(block, start)
                write_bandpass_block(
                    writer,
                    block,
                    header.nbits,
                    means,
                    stds,
                    signed=is_signed,
                )
            else:
                block = to_sigma(np.array(data[:count], copy=True), header.nbits)
                block = block.reshape(nsamps_r, nchan)
                injector.add_into(block, start)
                write_sigma_block(writer, block, header.nbits)
            start += nsamps_r
    return str(outfile)
