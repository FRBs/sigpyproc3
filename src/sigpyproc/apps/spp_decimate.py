from __future__ import annotations

import rich_click as click

from sigpyproc.readers import FilReader


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
    filfile: str,
    tfactor: int,
    ffactor: int,
    gulp: int,
    outfile: str | None,
) -> None:
    """Downsample a filterbank in time and/or frequency."""
    fil = FilReader(filfile)
    path = fil.downsample(tfactor, ffactor, outfile_name=outfile, gulp=gulp)
    click.echo(path)


if __name__ == "__main__":
    main()
