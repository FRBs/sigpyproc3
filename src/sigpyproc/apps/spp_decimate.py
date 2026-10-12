from __future__ import annotations

import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.readers import open_filterbank


@logging_options
@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-t", "--tfactor", type=int, default=1, help="Time downsample factor")
@click.option(
    "-c", "--ffactor", type=int, default=1, help="Frequency downsample factor"
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
    help="Output filename",
)
def main(
    verbose: int,
    filfile: str,
    tfactor: int,
    ffactor: int,
    gulp: int,
    outfile: str | None,
    *,
    quiet: bool,
) -> None:
    """Downsample a filterbank in time and/or frequency."""
    setup_cli_logging(verbose, quiet=quiet)
    fil = open_filterbank(filfile)
    path = fil.downsample(tfactor, ffactor, outfile_name=outfile, gulp=gulp)
    click.echo(path)


if __name__ == "__main__":
    main()
