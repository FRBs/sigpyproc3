from __future__ import annotations

import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.io.combine import splice_filterbanks


@logging_options
@click.command(
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
def main(
    verbose: int, filfiles: tuple[str, ...], gulp: int, outfile: str, *, quiet: bool
) -> None:
    """Splice simultaneous filterbanks along frequency (linear fch1/foff)."""
    setup_cli_logging(verbose, quiet=quiet)
    path = splice_filterbanks(filfiles, outfile, gulp=gulp)
    click.echo(path)


if __name__ == "__main__":
    main()
