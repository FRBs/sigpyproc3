import click
from sigpyproc.readers import FilReader


@click.group(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
def main() -> None:
    pass


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-s", "--start", type=int, required=True, help="Start time sample")
@click.option(
    "-n", "--nsamps", type=int, required=True, help="Number of time samples to extract"
)
@click.option(
    "-g", "--gulp", type=int, default=16384, help="Number of samples to read at once"
)
@click.option(
    "-o", "--outfile", type=click.Path(exists=False), default=None, help="Output filename"
)
def samples(filfile: str, start: int, nsamps: int, gulp: int, outfile: str) -> None:
    """Extract time samples from filterbank data."""
    fil = FilReader(filfile)
    fil.extract_samps(start=start, nsamps=nsamps, filename=outfile, gulp=gulp)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-c",
    "--chans",
    type=click.IntRange(0, None),
    multiple=True,
    help="Channels to extract",
)
def channels(filfile: str, chans: int) -> None:
    """Extract frequency channels from filterbank data."""
    fil = FilReader(filfile)
    fil.extract_chans(chans=chans)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-s", "--chanstart", type=int, required=True, help="Start channel")
@click.option(
    "-n", "--nchans", type=int, required=True, help="Number of channels to extract"
)
@click.option("-c", "--chanpersub", type=int, help="Number of channels in each sub-band")
def bands(filfile: str, chanstart: int, nchans: int, chanpersub: int) -> None:
    """Extract frequency bands from filterbank data."""
    fil = FilReader(filfile)
    fil.extract_bands(chanstart=chanstart, nchans=nchans, chanpersub=chanpersub)


if __name__ == "__main__":
    main()
