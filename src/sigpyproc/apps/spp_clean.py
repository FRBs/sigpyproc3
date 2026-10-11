from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.readers import FilReader

if TYPE_CHECKING:
    from sigpyproc.core.custom_types import MaskMethods


@logging_options
@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main(verbose: int, quiet: bool) -> None:
    """Clean RFI, mask channels, or samples in filterbank data."""
    setup_cli_logging(verbose, quiet)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-m",
    "--method",
    type=click.Choice(["mad", "iqrm"]),
    default="mad",
    help="RFI cleaning method to use.",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=3.0,
    help="Sigma threshold for RFI cleaning.",
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
    default=None,
    help="Output masked filterbank file",
)
@click.option(
    "-s",
    "--maskfile",
    type=click.Path(exists=False),
    default=None,
    help="Output mask file",
)
def rfi(
    filfile: str,
    method: MaskMethods,
    threshold: float,
    outfile: str | None,
    maskfile: str | None,
    gulp: int,
) -> None:
    """Clean RFI from filterbank data."""
    fil = FilReader(filfile)
    _out_file, rfimask = fil.clean_rfi(
        method=method,
        threshold=threshold,
        outfile_name=outfile,
        gulp=gulp,
    )
    if maskfile is not None:
        rfimask.to_file(filename=maskfile)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-c",
    "--chans",
    type=click.IntRange(0, None),
    multiple=True,
    help="Channels to zero",
)
@click.option(
    "--mask-file",
    type=click.Path(exists=True),
    default=None,
    help="Text file of channel indices to zero (one per line)",
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
    default=None,
    help="Output filterbank file",
)
def zap_channels(
    filfile: str,
    chans: tuple[int, ...],
    mask_file: str | None,
    gulp: int,
    outfile: str | None,
) -> None:
    """Zero selected channels and write a new filterbank."""
    if mask_file is not None:
        zap = np.loadtxt(mask_file, dtype=int, ndmin=1)
    else:
        zap = np.asarray(chans, dtype=int)
    if zap.size == 0:
        msg = "Provide -c/--chans or --mask-file"
        raise click.UsageError(msg)
    fil = FilReader(filfile)
    chan_mask = np.zeros(fil.header.nchans, dtype=bool)
    chan_mask[zap] = True
    path = fil.apply_channel_mask(chan_mask, outfile_name=outfile, gulp=gulp)
    click.echo(path)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-s", "--start", type=int, required=True, help="First sample to replace")
@click.option("-e", "--end", type=int, required=True, help="End sample (exclusive)")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["zero", "gaussian", "samples"]),
    default="zero",
    help="Replacement mode",
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
    default=None,
    help="Output filename (ignored with --in-place)",
)
@click.option(
    "--in-place",
    is_flag=True,
    default=False,
    help="Replace the input file via a temporary write",
)
def zap_times(
    filfile: str,
    start: int,
    end: int,
    mode: str,
    gulp: int,
    outfile: str | None,
    *,
    in_place: bool,
) -> None:
    """Replace a time-sample range in a filterbank."""
    fil = FilReader(filfile)
    if in_place:
        tmp_path = Path(filfile).with_suffix(Path(filfile).suffix + ".zap.tmp")
        fil.zap_times([(start, end)], mode=mode, outfile_name=str(tmp_path), gulp=gulp)
        tmp_path.replace(filfile)
        click.echo(filfile)
        return
    path = fil.zap_times([(start, end)], mode=mode, outfile_name=outfile, gulp=gulp)
    click.echo(path)


if __name__ == "__main__":
    main()
