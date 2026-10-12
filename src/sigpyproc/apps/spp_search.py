from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Literal

import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.core.candidates import CandidateTable
from sigpyproc.timeseries import TimeSeries


@logging_options
@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main(verbose: int, *, quiet: bool) -> None:
    """Search a time series (fft | pulses | ffa | cluster)."""
    setup_cli_logging(verbose, quiet=quiet)


def _write_cands(
    cands: CandidateTable,
    outfile: str | None,
    source: str,
    tag: str,
) -> None:
    if outfile is None:
        outfile = f"{Path(source).stem}_{tag}.csv"
    if outfile == "-":
        buf = StringIO()
        cands.to_csv(buf)
        click.echo(buf.getvalue(), nl=False)
        return
    cands.to_csv(outfile)
    click.echo(outfile)


@main.command()
@click.argument("timfile", type=click.Path(exists=True))
@click.option(
    "--nfolds",
    type=int,
    default=4,
    help=(
        "Number of harmonic folds (2, 4, …, 2**N) in addition to the unfolded spectrum."
    ),
)
@click.option("--pmax", type=float, default=10.0, help="Maximum period (s)")
@click.option("--accel", type=float, default=0.0, help="Acceleration (m/s^2)")
@click.option("--snr", type=float, default=6.0, help="Minimum peak SNR")
@click.option("--npeaks", type=int, default=100, help="Maximum peaks per spectrum")
@click.option(
    "--method",
    type=click.Choice(["lyne", "simple", "presto", "greedy"]),
    default="lyne",
    help="Harmonic sum: lyne (SIGPROC), simple, presto, or greedy.",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False, allow_dash=True),
    default=None,
    help="Output CSV (default: <stem>_fft.csv; '-' for stdout)",
)
def fft(
    timfile: str,
    nfolds: int,
    pmax: float,
    accel: float,
    snr: float,
    npeaks: int,
    method: Literal["lyne", "simple", "presto", "greedy"],
    outfile: str | None,
) -> None:
    """FFT search a SIGPROC time series (.tim)."""
    cands = TimeSeries.from_tim(timfile).search(
        nfolds=nfolds,
        pmax=pmax,
        accel=accel,
        snr_min=snr,
        npeaks=npeaks,
        method=method,  # type: ignore[arg-type]
    )
    _write_cands(cands, outfile, timfile, "fft")


@main.command()
@click.argument("timfile", type=click.Path(exists=True))
@click.option("--snr", type=float, default=6.0, help="Minimum pulse SNR")
@click.option("--max-width", type=int, default=256, help="Widest boxcar (bins)")
@click.option(
    "--spacing-factor",
    type=float,
    default=2.0,
    help="Geometric spacing of the boxcar bank",
)
@click.option(
    "--deadtime",
    type=int,
    default=8,
    help="Exclusion window either side of a peak",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False, allow_dash=True),
    default=None,
    help="Output CSV (default: <stem>_pulses.csv; '-' for stdout)",
)
def pulses(
    timfile: str,
    snr: float,
    max_width: int,
    spacing_factor: float,
    deadtime: int,
    outfile: str | None,
) -> None:
    """Single-pulse search a SIGPROC time series (.tim)."""
    cands = TimeSeries.from_tim(timfile).find_pulses(
        snr_min=snr,
        max_width=max_width,
        spacing_factor=spacing_factor,
        deadtime=deadtime,
    )
    _write_cands(cands, outfile, timfile, "pulses")


@main.command()
@click.argument("timfile", type=click.Path(exists=True))
@click.option(
    "--pmin-ms",
    type=float,
    required=True,
    help="Shortest period (ms); converted to seconds immediately",
)
@click.option(
    "--pmax-ms",
    type=float,
    required=True,
    help="Longest period (ms); converted to seconds immediately",
)
@click.option("--rebin", type=int, default=1, help="Mean-downsample factor")
@click.option(
    "--search-width",
    is_flag=True,
    default=False,
    help="Loop rebin in {1, 2, 4, 8, 16} (exclusive with --rebin != 1)",
)
@click.option(
    "--zero-pad",
    is_flag=True,
    default=False,
    help="Pad to the next power of two",
)
@click.option("--snr", type=float, default=8.0, help="Minimum profile SNR")
@click.option("--npeaks", type=int, default=100, help="Maximum candidates to keep")
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False, allow_dash=True),
    default=None,
    help="Output CSV (default: <stem>_ffa.csv; '-' for stdout)",
)
def ffa(
    timfile: str,
    pmin_ms: float,
    pmax_ms: float,
    rebin: int,
    snr: float,
    npeaks: int,
    outfile: str | None,
    *,
    search_width: bool,
    zero_pad: bool,
) -> None:
    """Fast Folding Analysis of a SIGPROC time series (.tim)."""
    cands = TimeSeries.from_tim(timfile).ffa(
        pmin_ms / 1000.0,
        pmax_ms / 1000.0,
        nrebin=rebin,
        search_width=search_width,
        snr_min=snr,
        npeaks=npeaks,
        zero_pad=zero_pad,
    )
    _write_cands(cands, outfile, timfile, "ffa")


@main.command()
@click.argument("candfile", type=click.Path(exists=True))
@click.option("--snr", type=float, default=8.0, help="Minimum SNR")
@click.option(
    "--period-tol",
    type=float,
    default=1e-4,
    help="Relative period match for harmonic families",
)
@click.option("--harm-max", type=int, default=16, help="Largest k, n in the k/n grid")
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False, allow_dash=True),
    default=None,
    help="Output CSV (default: <stem>_cluster.csv; '-' for stdout)",
)
def cluster(
    candfile: str,
    snr: float,
    period_tol: float,
    harm_max: int,
    outfile: str | None,
) -> None:
    """Keep the loudest member of each harmonic family."""
    cands = CandidateTable.from_csv(candfile).cluster(
        period_tol=period_tol,
        harm_max=harm_max,
        snr_min=snr,
    )
    _write_cands(cands, outfile, candfile, "cluster")


if __name__ == "__main__":
    main()
