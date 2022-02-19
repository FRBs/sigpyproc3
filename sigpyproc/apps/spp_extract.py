import click

from sigpyproc.readers import FilReader


@click.command(
    context_settings=dict(help_option_names=["-h", "--help"], show_default=True)
)
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-s", "--start", type=int, default=0, help="Start time sample"
)
@click.option(
    "-n", "--nsamps", type=int, help="Number of time samples to extract"
)
@click.option(
    "-g", "--gulp", type=int, default=16384, help="Number of samples to read at once"
)
@click.option(
    "-o", "--outfile", type=click.Path(exists=False), default=None, help="Output filename"
)
def main(filfile: str, start: int, nsamps: int, gulp: int, outfile: str) -> None:
    """Extract time samples from filterbank data."""
    fil = FilReader(filfile)
    fil.extract_samps(start=start, nsamps=nsamps, outfile=outfile, gulp=gulp)


if __name__ == "__main__":
    main()
