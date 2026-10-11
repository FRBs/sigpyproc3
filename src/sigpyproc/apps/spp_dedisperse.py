from __future__ import annotations

import rich_click as click

from sigpyproc.core.dmplan import DMPlan
from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.readers import FilReader


@logging_options
@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main(verbose: int, quiet: bool) -> None:
    """Dedisperse filterbank data (tim | subband | tree | plan)."""
    setup_cli_logging(verbose, quiet)


def _parse_ref_freq(value: str) -> str | float:
    if value in {"fch1", "fmax", "fmin", "fcenter"}:
        return value
    return float(value)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-d", "--dm", type=float, required=True, help="Dispersion measure")
@click.option(
    "-f",
    "--ref-freq",
    default="fch1",
    help="Reference frequency (fch1/fmax/fmin/fcenter or MHz)",
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
    help="Output .tim file",
)
def tim(
    filfile: str,
    dm: float,
    ref_freq: str,
    gulp: int,
    outfile: str | None,
) -> None:
    """Incoherent dedispersion to a time series."""
    fil = FilReader(filfile)
    series = fil.dedisperse(dm, _parse_ref_freq(ref_freq), gulp=gulp)
    click.echo(series.to_tim(outfile))


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-d", "--dm", type=float, required=True, help="Dispersion measure")
@click.option("-b", "--nsub", type=int, required=True, help="Number of subbands")
@click.option(
    "-f",
    "--ref-freq",
    default="fch1",
    help="Reference frequency (fch1/fmax/fmin/fcenter or MHz)",
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
    help="Output filterbank",
)
def subband(
    filfile: str,
    dm: float,
    nsub: int,
    ref_freq: str,
    gulp: int,
    outfile: str | None,
) -> None:
    """Dedisperse into frequency subbands."""
    fil = FilReader(filfile)
    path = fil.subband(dm, nsub, _parse_ref_freq(ref_freq), outfile, gulp=gulp)
    click.echo(path)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-l", "--dm-lo", type=int, default=0, help="Lowest tree DM index")
@click.option("-u", "--dm-hi", type=int, default=None, help="Highest tree DM index")
@click.option("--chunk", type=int, default=4096, help="Valid samples per gulp")
@click.option("--no-flip", is_flag=True, help="Do not reverse channel order")
@click.option("--max-gb", type=float, default=2.0, help="Gulp-buffer memory cap")
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    default=None,
    help="Output file stem",
)
def tree(
    filfile: str,
    dm_lo: int,
    dm_hi: int | None,
    chunk: int,
    max_gb: float,
    outfile: str | None,
    *,
    no_flip: bool,
) -> None:
    """Taylor-tree dedispersion (one .tim per DM index)."""
    fil = FilReader(filfile)
    flip = False if no_flip else None
    paths = fil.tree_dedisperse(
        dm_lo,
        dm_hi,
        flip=flip,
        chunk=chunk,
        outfile_base=outfile,
        max_gb=max_gb,
    )
    for path in paths:
        click.echo(path)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("--dm-max", type=float, required=True, help="Maximum trial DM")
@click.option("--dm-min", type=float, default=0.0, help="Minimum trial DM")
@click.option("--tol", type=float, default=1.25, help="Smear tolerance")
@click.option("--ti", type=float, default=40e-6, help="Intrinsic width in seconds")
@click.option(
    "-f",
    "--ref-freq",
    default="fch1",
    help="Reference frequency (fch1/fmax/fmin/fcenter or MHz)",
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
    help="Output file stem",
)
def plan(
    filfile: str,
    dm_max: float,
    dm_min: float,
    tol: float,
    ti: float,
    ref_freq: str,
    gulp: int,
    outfile: str | None,
) -> None:
    """Dedisperse a Levin/Cordes DM plan to many .tim files."""
    fil = FilReader(filfile)
    dmplan = DMPlan.from_header(fil.header, dm_max, dm_min, ti=ti, tol=tol)
    paths = fil.dedisperse_plan(
        dmplan,
        gulp=gulp,
        ref_freq=_parse_ref_freq(ref_freq),
        outfile_base=outfile,
    )
    for path in paths:
        click.echo(path)


if __name__ == "__main__":
    main()
