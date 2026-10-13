from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import rich_click as click

from sigpyproc.algorithms.heimdall import HeimdallConfig
from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.core.candidates import CandidateTable
from sigpyproc.timeseries import TimeSeries

if TYPE_CHECKING:
    from sigpyproc.core.custom_types import (
        BoxcarEngine,
        PulseDetrend,
        PulseMethod,
        WidthStyle,
    )


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
    *,
    pulse_tsamp: float | None = None,
) -> None:
    if outfile is None:
        outfile = f"{Path(source).stem}_{tag}.csv"
    if outfile == "-":
        buf = StringIO()
        if pulse_tsamp is None:
            cands.to_csv(buf)
        else:
            cands.to_pulse_csv(buf, pulse_tsamp)
        click.echo(buf.getvalue(), nl=False)
        return
    if pulse_tsamp is None:
        cands.to_csv(outfile)
    else:
        cands.to_pulse_csv(outfile, pulse_tsamp)
    click.echo(outfile)


def _load_pulse_series(filename: str) -> TimeSeries:
    suffix = Path(filename).suffix.lower()
    if suffix == ".dat":
        return TimeSeries.from_dat(filename)
    if suffix == ".tim":
        return TimeSeries.from_tim(filename)
    msg = "pulse input must be a .tim or .dat file (with a matching .inf)"
    raise ValueError(msg)


def _heimdall_cli_config(
    method: PulseMethod,
    options: dict[str, int | float | bool],
) -> HeimdallConfig | None:
    if method != "heimdall":
        if options:
            msg = "Heimdall options require --method heimdall"
            raise ValueError(msg)
        return None
    # Click has validated individual option types; the config validates their
    # relationships and applies defaults for omitted options.
    defaults = HeimdallConfig()
    return HeimdallConfig(
        gulp_size=int(options.get("gulp_size", defaults.gulp_size)),
        baseline_length=float(options.get("baseline_length", defaults.baseline_length)),
        min_tscrunch_width=int(
            options.get("min_tscrunch_width", defaults.min_tscrunch_width)
        ),
        cand_sep_time=int(options.get("cand_sep_time", defaults.cand_sep_time)),
        cand_sep_filter=int(options.get("cand_sep_filter", defaults.cand_sep_filter)),
        boxcar_renorm=bool(options.get("boxcar_renorm", defaults.boxcar_renorm)),
    )


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
@click.option(
    "--max-width",
    type=click.IntRange(min=1),
    default=256,
    help="Trial width ceiling (bins)",
)
@click.option(
    "--spacing-factor",
    type=float,
    default=2.0,
    help="Geometric spacing of the boxcar bank",
)
@click.option(
    "--deadtime",
    type=click.IntRange(min=0),
    default=None,
    help="Minimum exclusion radius (bins; default: PRESTO 0, matched filter 8)",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False, allow_dash=True),
    default=None,
    help="Output CSV (default: <stem>_pulses.csv; '-' for stdout)",
)
@click.option(
    "--method",
    type=click.Choice(["presto", "matched_filter", "heimdall"]),
    default="presto",
)
@click.option(
    "--engine",
    type=click.Choice(["running_sum", "fft"]),
    default="running_sum",
    help="PRESTO boxcar engine",
)
@click.option(
    "--width-style",
    type=click.Choice(["presto", "geometric", "powers_of_two"]),
    default=None,
    help="Width bank (default follows method)",
)
@click.option(
    "--width",
    "widths",
    type=click.IntRange(min=1),
    multiple=True,
    help="Explicit width in bins; repeat to build a bank",
)
@click.option(
    "--block-size",
    type=click.IntRange(min=2),
    default=8000,
    help="PRESTO preprocessing block size (bins)",
)
@click.option(
    "--detrend",
    type=click.Choice(["linear", "median"]),
    default="linear",
    help="PRESTO baseline removal",
)
@click.option(
    "--mask-bad-blocks/--no-mask-bad-blocks",
    default=False,
    help="Reject PRESTO noise-outlier blocks; may reject bright broad pulses",
)
@click.option(
    "--chunk-size",
    type=click.IntRange(min=1),
    default=8192,
    help="PRESTO scoring chunk size (bins)",
)
@click.option(
    "--gulp-size",
    type=click.IntRange(min=2),
    default=None,
    help="Heimdall fresh samples per gulp (default: 262144)",
)
@click.option(
    "--baseline-length",
    type=float,
    default=None,
    help="Heimdall baseline smoothing length in seconds (default: 2)",
)
@click.option(
    "--min-tscrunch-width",
    type=click.IntRange(min=2),
    default=None,
    help="Heimdall filter-striding threshold (default: 4096)",
)
@click.option(
    "--cand-sep-time",
    type=click.IntRange(min=0),
    default=None,
    help="Heimdall giant/cluster separation in filter widths (default: 3)",
)
@click.option(
    "--cand-sep-filter",
    type=click.IntRange(min=0),
    default=None,
    help="Heimdall cluster separation in filter indices (default: 3)",
)
@click.option(
    "--boxcar-renorm/--no-boxcar-renorm",
    default=None,
    help="Heimdall: re-estimate noise after each boxcar (default: disabled)",
)
def pulses(  # noqa: PLR0913 - method-specific Click options
    timfile: str,
    snr: float,
    max_width: int,
    spacing_factor: float,
    deadtime: int | None,
    outfile: str | None,
    method: PulseMethod,
    engine: BoxcarEngine,
    width_style: WidthStyle | None,
    widths: tuple[int, ...],
    block_size: int,
    detrend: PulseDetrend,
    chunk_size: int,
    *,
    gulp_size: int | None,
    baseline_length: float | None,
    min_tscrunch_width: int | None,
    cand_sep_time: int | None,
    cand_sep_filter: int | None,
    mask_bad_blocks: bool,
    boxcar_renorm: bool | None,
) -> None:
    """Single-pulse search a .tim or PRESTO .dat/.inf time series."""
    try:
        series = _load_pulse_series(timfile)
        hoptions = {
            "gulp_size": gulp_size,
            "baseline_length": baseline_length,
            "min_tscrunch_width": min_tscrunch_width,
            "cand_sep_time": cand_sep_time,
            "cand_sep_filter": cand_sep_filter,
            "boxcar_renorm": boxcar_renorm,
        }
        supplied = {
            name: value for name, value in hoptions.items() if value is not None
        }
        config = _heimdall_cli_config(method, supplied)
        cands = series.find_pulses(
            heimdall_config=config,
            snr_min=snr,
            max_width=max_width,
            spacing_factor=spacing_factor,
            deadtime=deadtime,
            method=method,
            engine=engine,
            width_style=width_style,
            widths=widths or None,
            block_size=block_size,
            detrend=detrend,
            mask_bad_blocks=mask_bad_blocks,
            chunk_size=chunk_size,
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    _write_cands(cands, outfile, timfile, "pulses", pulse_tsamp=series.header.tsamp)


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
