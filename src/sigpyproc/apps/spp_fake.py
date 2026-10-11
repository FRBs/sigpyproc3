from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.simulation.pulsar import PulsarParams, fake_filterbank, make_header

if TYPE_CHECKING:
    from collections.abc import Callable


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


@logging_options
@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main(verbose: int, quiet: bool) -> None:
    """Generate fake filterbank data (noise | pulsar | pulse)."""
    setup_cli_logging(verbose, quiet)


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


if __name__ == "__main__":
    main()
