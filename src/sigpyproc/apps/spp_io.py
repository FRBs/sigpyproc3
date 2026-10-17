from __future__ import annotations

from typing import Literal

import numpy as np
import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.io.combine import merge_filterbanks, splice_filterbanks
from sigpyproc.readers import open_filterbank


@logging_options
@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main(verbose: int, *, quiet: bool) -> None:
    """Combine filterbanks or extract samples, channels and bands."""
    setup_cli_logging(verbose, quiet=quiet)


@main.group()
def extract() -> None:
    """Extract samples, channels, or bands from a filterbank."""


@extract.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-s", "--start", type=int, default=0, help="Start time sample")
@click.option(
    "-n",
    "--nsamps",
    type=int,
    default=None,
    help="Number of time samples to extract",
)
@click.option("--start-sec", type=float, default=None, help="Start time in seconds")
@click.option("--duration", type=float, default=None, help="Duration in seconds")
@click.option(
    "-g",
    "--gulp",
    type=int,
    default=16384,
    help="Number of samples to read at once",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    default=None,
    help="Output filename",
)
def samples(
    filfile: str,
    start: int,
    nsamps: int | None,
    start_sec: float | None,
    duration: float | None,
    gulp: int,
    outfile: str | None,
) -> None:
    """Extract time samples from filterbank data."""
    with open_filterbank(filfile) as fil:
        if start_sec is not None:
            start = round(start_sec / fil.header.tsamp)
        if duration is not None:
            nsamps = round(duration / fil.header.tsamp)
        if nsamps is None:
            msg = "Provide -n/--nsamps or --duration"
            raise click.UsageError(msg)
        path = fil.extract_samps(
            start=start,
            nsamps=nsamps,
            outfile_name=outfile,
            gulp=gulp,
        )
        click.echo(path)


@extract.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-c",
    "--chans",
    type=click.IntRange(0, None),
    multiple=True,
    help="Channels to extract (default: all)",
)
@click.option("--chan-start", type=int, default=None, help="First channel of a range")
@click.option(
    "--nchans",
    type=int,
    default=None,
    help="Number of channels in the range",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    default=None,
    help="Output file basename",
)
def channels(
    filfile: str,
    chans: tuple[int, ...],
    chan_start: int | None,
    nchans: int | None,
    outfile: str | None,
) -> None:
    """Extract frequency channels as time series (one ``.tim`` per channel)."""
    with open_filterbank(filfile) as fil:
        if chans:
            chan_arr = np.asarray(chans)
        elif chan_start is not None:
            n_out = fil.header.nchans - chan_start if nchans is None else nchans
            chan_arr = np.arange(chan_start, chan_start + n_out)
        else:
            chan_arr = None
        for path in fil.extract_chans(chans=chan_arr, outfile_base=outfile):
            click.echo(path)


@extract.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-s", "--chanstart", type=int, required=True, help="Start channel")
@click.option(
    "-n",
    "--nchans",
    type=int,
    required=True,
    help="Number of channels to extract",
)
@click.option(
    "-c",
    "--chanpersub",
    type=int,
    help="Number of channels in each sub-band",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    default=None,
    help="Output file basename",
)
def bands(
    filfile: str,
    chanstart: int,
    nchans: int,
    chanpersub: int | None,
    outfile: str | None,
) -> None:
    """Extract frequency bands from filterbank data."""
    with open_filterbank(filfile) as fil:
        for path in fil.extract_bands(
            chanstart=chanstart,
            nchans=nchans,
            chanpersub=chanpersub,
            outfile_base=outfile,
        ):
            click.echo(path)


@main.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfiles", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "-a",
    "--axis",
    type=click.Choice(["auto", "time", "freq"]),
    default="auto",
    help="Concatenate in time, frequency, or pick automatically",
)
@click.option(
    "-g",
    "--gulp",
    type=int,
    default=16384,
    help="Number of samples to read at once",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    required=True,
    help="Output filename",
)
def merge(
    filfiles: tuple[str, ...],
    axis: Literal["auto", "time", "freq"],
    gulp: int,
    outfile: str,
) -> None:
    """Merge filterbanks in time or frequency."""
    path = merge_filterbanks(filfiles, outfile, axis=axis, gulp=gulp)
    click.echo(path)


@main.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfiles", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "-g",
    "--gulp",
    type=int,
    default=16384,
    help="Number of samples to read at once",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    required=True,
    help="Output filename",
)
def splice(filfiles: tuple[str, ...], gulp: int, outfile: str) -> None:
    """Splice simultaneous filterbanks along frequency (linear fch1/foff)."""
    path = splice_filterbanks(filfiles, outfile, gulp=gulp)
    click.echo(path)


if __name__ == "__main__":
    main()
