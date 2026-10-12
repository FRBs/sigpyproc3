from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.readers import FilReader, open_filterbank
from sigpyproc.simulation.furby import SPEC_KINDS, PulseParams
from sigpyproc.simulation.inject import (
    inject_file,
    injector_from_rows,
    read_pulse_catalogue,
)
from sigpyproc.simulation.population import BurstDistribution, RandomBurstPopulation
from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank, make_header

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sigpyproc.core.custom_types import SpecSimulMethods
    from sigpyproc.header import Header
    from sigpyproc.simulation.inject import Injector


def _obs_options(func: Callable[..., Any]) -> Callable[..., Any]:
    options = [
        click.option(
            "-o",
            "--outfile",
            type=click.Path(exists=False),
            default="fake.fil",
            help="Output filterbank",
        ),
        click.option("--nchans", type=int, default=128, help="Number of channels"),
        click.option("--nbits", type=int, default=4, help="Bits per sample"),
        click.option("--tsamp", type=float, default=80e-6, help="Sampling time (s)"),
        click.option("--tobs", type=float, default=10.0, help="Observation length (s)"),
        click.option("--tstart", type=float, default=50000.0, help="Start MJD"),
        click.option(
            "--fch1",
            type=float,
            default=433.968,
            help="First-channel frequency (MHz)",
        ),
        click.option(
            "--foff",
            type=float,
            default=-0.062,
            help="Channel bandwidth (MHz)",
        ),
        click.option("--source", type=str, default=None, help="Source name in header"),
        click.option(
            "--seed",
            type=int,
            default=None,
            help="RNG seed (printed to stderr if omitted)",
        ),
        click.option(
            "-g",
            "--gulp",
            type=int,
            default=16384,
            help="Samples written per block",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        seed = int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        click.echo(f"seed = {seed}", err=True)
    return seed


def _write(
    outfile: str,
    nchans: int,
    nbits: int,
    tsamp: float,
    tobs: float,
    tstart: float,
    fch1: float,
    foff: float,
    source: str | None,
    seed: int | None,
    gulp: int,
    params: PulsarParams,
    default_source: str,
) -> None:
    seed = _resolve_seed(seed)
    header = make_header(
        filename=outfile,
        nchans=nchans,
        nbits=nbits,
        tsamp=tsamp,
        tobs=tobs,
        tstart=tstart,
        fch1=fch1,
        foff=foff,
        source=source if source is not None else default_source,
        dm=params.dm,
        period=params.period or 0.0,
    )
    path = fake_filterbank(header, params, outfile, seed=seed, gulp=gulp)
    click.echo(path)


def _pulse_options(func: Callable[..., Any]) -> Callable[..., Any]:
    options = [
        click.option(
            "-t",
            "--time",
            "times",
            type=float,
            multiple=True,
            help="Arrival time at the reference frequency (s). Repeatable.",
        ),
        click.option("-d", "--dm", type=float, default=0.0, help="Dispersion measure"),
        click.option(
            "-w",
            "--width",
            type=float,
            default=None,
            help="Intrinsic width in seconds (FWHM for a Gaussian)",
        ),
        click.option(
            "-s",
            "--snr",
            type=float,
            default=10.0,
            help="Matched-filter SNR of the dedispersed sum",
        ),
        click.option(
            "--shape",
            type=click.Choice(["gaussian", "boxcar"]),
            default="gaussian",
            help="Intrinsic pulse shape",
        ),
        click.option(
            "--spec-kind",
            type=click.Choice(list(SPEC_KINDS)),
            default="flat",
            help="Spectral shape. flat ignores --spec-index.",
        ),
        click.option(
            "--spec-index",
            type=float,
            default=0.0,
            help="Spectral index for power_law and other structured shapes",
        ),
        click.option(
            "--tau",
            type=float,
            default=0.0,
            help="Scattering time at --tau-ref-freq (s). 0 turns scattering off.",
        ),
        click.option(
            "--tau-ref-freq",
            type=float,
            default=1000.0,
            help="Reference frequency for the scattering time (MHz, default: 1000.0)",
        ),
        click.option(
            "--scatt-index",
            type=float,
            default=-4.4,
            help="Frequency exponent of the scattering time",
        ),
        click.option(
            "--disp-index",
            type=float,
            default=-2.0,
            help="Frequency exponent of the dispersion delay",
        ),
        click.option(
            "--os-fact",
            type=int,
            default=10,
            help="Time oversampling used while building each pulse",
        ),
        click.option(
            "--nosmear",
            is_flag=True,
            help="Do not apply intra-channel dispersion smearing",
        ),
        click.option(
            "--ref-freq",
            type=click.Choice(["fch1", "fcenter", "fmin", "fmax"]),
            default="fch1",
            help="Frequency the arrival time refers to",
        ),
        click.option(
            "--catalogue",
            type=click.Path(exists=True, dir_okay=False),
            default=None,
            help="Text file of 'time dm width snr' rows. # comments are ignored.",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def _rows_from_cli(
    times: tuple[float, ...],
    dm: float,
    width: float | None,
    snr: float,
    catalogue: str | None,
) -> list[tuple[Any, ...]]:
    if catalogue is not None and times:
        msg = "--time and --catalogue are mutually exclusive"
        raise click.UsageError(msg)
    if catalogue is not None:
        try:
            return read_pulse_catalogue(catalogue)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
    if not times:
        msg = "give at least one --time, or a --catalogue"
        raise click.UsageError(msg)
    if width is None:
        msg = "--width is required when --catalogue is not set"
        raise click.UsageError(msg)
    return [(time, dm, width, snr) for time in times]


def _shared_dm(rows: Sequence[tuple[Any, ...]]) -> float | None:
    """Return the DM when every row shares one, otherwise None."""
    dms = {float(row[1]) for row in rows}
    if len(dms) == 1:
        return dms.pop()
    return None


def _pulse_template(
    *,
    shape: str,
    spec_kind: str,
    spec_index: float,
    tau: float,
    scatt_index: float,
    tau_ref_freq: float = 1000.0,
    disp_index: float,
    os_fact: int,
    nosmear: bool,
) -> PulseParams:
    return PulseParams(
        dm=0.0,
        snr=1.0,
        width=1.0,
        shape=shape,
        dmsmear=not nosmear,
        disp_ind=disp_index,
        scatt_idx=scatt_index,
        tau0=tau,
        tau_ref_freq=tau_ref_freq,
        spec_kind=cast("SpecSimulMethods", spec_kind),
        spec_idx=spec_index,
        os_fact=os_fact,
    )


def _injector_for(
    header: Header,
    rows: Sequence[tuple[Any, ...]],
    template: PulseParams,
    *,
    ref_freq: str,
    seed: int,
) -> Injector:
    try:
        return injector_from_rows(
            header,
            rows,
            template,
            ref_freq=ref_freq,
            seed=seed,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


@logging_options
@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main(verbose: int, *, quiet: bool) -> None:
    """Generate fake filterbanks, or inject furby pulses into one."""
    setup_cli_logging(verbose, quiet=quiet)


@main.command()
@_obs_options
def noise(
    outfile: str,
    nchans: int,
    nbits: int,
    tsamp: float,
    tobs: float,
    tstart: float,
    fch1: float,
    foff: float,
    source: str | None,
    seed: int | None,
    gulp: int,
) -> None:
    """Gaussian noise filterbank."""
    _write(
        outfile,
        nchans,
        nbits,
        tsamp,
        tobs,
        tstart,
        fch1,
        foff,
        source,
        seed,
        gulp,
        PulsarParams(),
        "Fake",
    )


@main.command()
@click.option("-p", "--period", type=float, required=True, help="Spin period (s)")
@click.option(
    "-w",
    "--width",
    type=float,
    default=None,
    help="Intrinsic pulse width (s, default 4% of period)",
)
@click.option("-d", "--dm", type=float, default=0.0, help="Dispersion measure")
@click.option(
    "-s",
    "--snr",
    type=float,
    default=1.0,
    help="Single-pulse peak SNR (then /sqrt(nchans))",
)
@click.option("--pdot", type=float, default=0.0, help="Period derivative (s/s)")
@click.option("--accel", type=float, default=0.0, help="Acceleration (m/s^2)")
@click.option("--nosmear", is_flag=True, help="Do not add DM/sampling smearing")
@_obs_options
def pulsar(
    period: float,
    width: float | None,
    dm: float,
    snr: float,
    pdot: float,
    accel: float,
    outfile: str,
    nchans: int,
    nbits: int,
    tsamp: float,
    tobs: float,
    tstart: float,
    fch1: float,
    foff: float,
    source: str | None,
    seed: int | None,
    gulp: int,
    *,
    nosmear: bool,
) -> None:
    """Periodic rectangular-pulse pulsar."""
    params = PulsarParams(
        period=period,
        width=width,
        dm=dm,
        snr=snr,
        pdot=pdot,
        accel=accel,
        smear=not nosmear,
    )
    _write(
        outfile,
        nchans,
        nbits,
        tsamp,
        tobs,
        tstart,
        fch1,
        foff,
        source,
        seed,
        gulp,
        params,
        f"P: {period} s, DM: {dm}",
    )


@main.command()
@click.option(
    "-t",
    "--time",
    "pulse_time",
    type=float,
    required=True,
    help="Arrival time at fch1 (s from start)",
)
@click.option(
    "-w",
    "--width",
    type=float,
    required=True,
    help="Intrinsic pulse width (s)",
)
@click.option("-d", "--dm", type=float, default=0.0, help="Dispersion measure")
@click.option(
    "-s",
    "--snr",
    type=float,
    default=1.0,
    help="Peak SNR (then /sqrt(nchans))",
)
@click.option("--nosmear", is_flag=True, help="Do not add DM/sampling smearing")
@_obs_options
def pulse(
    pulse_time: float,
    width: float,
    dm: float,
    snr: float,
    outfile: str,
    nchans: int,
    nbits: int,
    tsamp: float,
    tobs: float,
    tstart: float,
    fch1: float,
    foff: float,
    source: str | None,
    seed: int | None,
    gulp: int,
    *,
    nosmear: bool,
) -> None:
    """Single dispersed pulse."""
    params = PulsarParams(
        pulse_time=pulse_time,
        width=width,
        dm=dm,
        snr=snr,
        smear=not nosmear,
    )
    _write(
        outfile,
        nchans,
        nbits,
        tsamp,
        tobs,
        tstart,
        fch1,
        foff,
        source,
        seed,
        gulp,
        params,
        f"pulse @ {pulse_time} s, DM: {dm}",
    )


@main.command()
@_pulse_options
@_obs_options
def furby(
    times: tuple[float, ...],
    dm: float,
    width: float | None,
    snr: float,
    shape: str,
    spec_kind: str,
    spec_index: float,
    tau: float,
    tau_ref_freq: float,
    scatt_index: float,
    disp_index: float,
    os_fact: int,
    ref_freq: str,
    catalogue: str | None,
    outfile: str,
    nchans: int,
    nbits: int,
    tsamp: float,
    tobs: float,
    tstart: float,
    fch1: float,
    foff: float,
    source: str | None,
    seed: int | None,
    gulp: int,
    *,
    nosmear: bool,
) -> None:
    """Mock FRB filterbank. ``--snr`` is the dedispersed matched-filter SNR."""
    rows = _rows_from_cli(times, dm, width, snr, catalogue)
    seed = _resolve_seed(seed)
    shared = _shared_dm(rows)
    header_dm = 0.0 if shared is None else shared
    if source is not None:
        source_name = source
    elif header_dm:
        source_name = f"DM {header_dm:g}"
    else:
        source_name = "Furby"
    header = make_header(
        filename=outfile,
        nchans=nchans,
        nbits=nbits,
        tsamp=tsamp,
        tobs=tobs,
        tstart=tstart,
        fch1=fch1,
        foff=foff,
        source=source_name,
        dm=header_dm,
    )
    injector = _injector_for(
        header,
        rows,
        _pulse_template(
            shape=shape,
            spec_kind=spec_kind,
            spec_index=spec_index,
            tau=tau,
            scatt_index=scatt_index,
            tau_ref_freq=tau_ref_freq,
            disp_index=disp_index,
            os_fact=os_fact,
            nosmear=nosmear,
        ),
        ref_freq=ref_freq,
        seed=seed,
    )
    if not injector.overlaps(header.nsamples):
        click.echo("warning: no pulse overlaps the observation", err=True)
    path = fake_filterbank(
        header,
        PulsarParams(),
        outfile,
        seed=seed,
        gulp=gulp,
        injector=injector,
    )
    click.echo(path)


@main.command(name="inject")
@click.argument("infile", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    required=True,
    help="Output filterbank",
)
@click.option(
    "-g",
    "--gulp",
    type=int,
    default=16384,
    help="Samples read per block",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Spectral-structure seed (printed to stderr if omitted)",
)
@click.option(
    "--preserve-bandpass",
    is_flag=True,
    help="Preserve measured per-channel mean and scale from input data",
)
@click.option(
    "--estimate-samples",
    type=int,
    default=16384,
    help="Samples used to estimate channel bandpass (default: 16384)",
)
@_pulse_options
def inject_filterbank(
    infile: str,
    outfile: str,
    gulp: int,
    seed: int | None,
    times: tuple[float, ...],
    dm: float,
    width: float | None,
    snr: float,
    shape: str,
    spec_kind: str,
    spec_index: float,
    tau: float,
    tau_ref_freq: float,
    scatt_index: float,
    disp_index: float,
    os_fact: int,
    ref_freq: str,
    catalogue: str | None,
    *,
    nosmear: bool,
    preserve_bandpass: bool,
    estimate_samples: int,
) -> None:
    """Add furby pulses to a sigproc filterbank.

    Integer samples are inverted with the fixed digitiser by default, or
    using empirical channel baseline and standard deviation when
    ``--preserve-bandpass`` is set. PSRFITS files are rejected.
    """
    if Path(infile).resolve() == Path(outfile).resolve():
        msg = "outfile must be a different file from the input"
        raise click.UsageError(msg)
    rows = _rows_from_cli(times, dm, width, snr, catalogue)
    seed = _resolve_seed(seed)
    reader = open_filterbank(infile)
    try:
        if not isinstance(reader, FilReader):
            msg = "inject reads sigproc filterbanks written with the fixed digitiser"
            raise click.UsageError(msg)
        header = reader.header
    finally:
        reader.close()
    injector = _injector_for(
        header,
        rows,
        _pulse_template(
            shape=shape,
            spec_kind=spec_kind,
            spec_index=spec_index,
            tau=tau,
            scatt_index=scatt_index,
            tau_ref_freq=tau_ref_freq,
            disp_index=disp_index,
            os_fact=os_fact,
            nosmear=nosmear,
        ),
        ref_freq=ref_freq,
        seed=seed,
    )
    if not injector.overlaps(header.nsamples):
        click.echo("warning: no pulse overlaps the observation", err=True)
    try:
        path = inject_file(
            infile,
            outfile,
            injector,
            gulp=gulp,
            dm=_shared_dm(rows),
            preserve_bandpass=preserve_bandpass,
            estimate_samples=estimate_samples,
        )
    except (TypeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(path)


@main.command()
@click.option(
    "-n",
    "--count",
    type=int,
    default=10,
    help="Number of bursts to generate",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(),
    required=True,
    help="Output catalogue path (.txt)",
)
@click.option("--tobs", type=float, default=10.0, help="Observation duration (s)")
@click.option("--dm-min", type=float, default=10.0, help="Minimum DM")
@click.option("--dm-max", type=float, default=1000.0, help="Maximum DM")
@click.option("--width-min", type=float, default=1e-3, help="Minimum width (s)")
@click.option("--width-max", type=float, default=1e-2, help="Maximum width (s)")
@click.option("--snr-min", type=float, default=10.0, help="Minimum SNR")
@click.option("--snr-max", type=float, default=50.0, help="Maximum SNR")
@click.option(
    "--snr-law",
    type=click.Choice(["uniform", "euclidean"]),
    default="uniform",
    help="SNR distribution law",
)
@click.option(
    "--tau-min",
    type=float,
    default=0.0,
    help="Minimum scattering time at reference frequency (s)",
)
@click.option(
    "--tau-max",
    type=float,
    default=0.0,
    help="Maximum scattering time at reference frequency (s)",
)
@click.option(
    "--tau-ref-freq",
    type=float,
    default=1000.0,
    help="Reference frequency for the scattering timescale (MHz, default: 1000.0)",
)
@click.option("--seed", type=int, default=None, help="Random seed")
def population(
    count: int,
    outfile: str,
    tobs: float,
    dm_min: float,
    dm_max: float,
    width_min: float,
    width_max: float,
    snr_min: float,
    snr_max: float,
    snr_law: str,
    tau_min: float,
    tau_max: float,
    tau_ref_freq: float,
    seed: int | None,
) -> None:
    """Generate a mock FRB population catalogue."""
    dist = BurstDistribution(
        dm_range=(dm_min, dm_max),
        width_range=(width_min, width_max),
        snr_range=(snr_min, snr_max),
        snr_law=cast("Literal['uniform', 'euclidean']", snr_law),
        tau_range=(tau_min, tau_max),
        tau_ref_freq=tau_ref_freq,
    )
    pop = RandomBurstPopulation(dist, seed=seed)
    path = pop.write_catalogue(outfile, count, tobs)
    click.echo(path)


if __name__ == "__main__":
    main()
