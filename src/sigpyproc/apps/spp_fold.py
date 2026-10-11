from __future__ import annotations

import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.readers import FilReader


@logging_options
@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-p", "--period", type=float, required=True, help="Fold period in seconds"
)
@click.option("-d", "--dm", type=float, default=0.0, help="Dispersion measure")
@click.option("-a", "--accel", type=float, default=0.0, help="Acceleration (m/s^2)")
@click.option("-n", "--nbins", type=int, default=50, help="Phase bins")
@click.option("--nints", type=int, default=32, help="Subintegrations")
@click.option("--nbands", type=int, default=32, help="Subbands")
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
    help="Output basename (writes .npz and .txt)",
)
def main(
    verbose: int,
    quiet: bool,
    filfile: str,
    period: float,
    dm: float,
    accel: float,
    nbins: int,
    nints: int,
    nbands: int,
    gulp: int,
    outfile: str | None,
) -> None:
    """Fold a filterbank at a period and DM."""
    setup_cli_logging(verbose, quiet)
    fil = FilReader(filfile)
    cube = fil.fold(
        period,
        dm,
        accel=accel,
        nbins=nbins,
        nints=nints,
        nbands=nbands,
        gulp=gulp,
    )
    stem = outfile if outfile is not None else fil.header.basename
    cube.to_npz(f"{stem}.npz")
    cube.to_ascii(f"{stem}.txt")
    click.echo(stem)


if __name__ == "__main__":
    main()
