import click

from sigpyproc.readers import FilReader


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-t",
    "--tfactor",
    type=int,
    default=1,
    help="Number of time samples to add",
)
@click.option(
    "-c",
    "--ffactor",
    type=int,
    default=1,
    help="Number of frequency channels to add",
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
def main(filfile: str, tfactor: int, ffactor: int, gulp: int, outfile: str) -> None:
    """Reduce time and/or frequency resolution of filterbank data."""
    fil = FilReader(filfile)
    fil.downsample(tfactor=tfactor, ffactor=ffactor, gulp=gulp, filename=outfile)


if __name__ == "__main__":
    main()
