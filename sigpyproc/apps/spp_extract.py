import click
import numpy as np

from sigpyproc.readers import FilReader


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main() -> None:
    pass


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option("-s", "--start", type=int, required=True, help="Start time sample")
@click.option(
    "-n",
    "--nsamps",
    type=int,
    required=True,
    help="Number of time samples to extract",
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
def samples(filfile: str, start: int, nsamps: int, gulp: int, outfile: str) -> None:
    """Extract time samples from filterbank data."""
    fil = FilReader(filfile)
    fil.extract_samps(start=start, nsamps=nsamps, outfile_name=outfile, gulp=gulp)


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-c",
    "--chans",
    required=True,
    type=click.IntRange(0, None),
    multiple=True,
    help="Channels to extract",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    default=None,
    help="Output file basename",
)
def channels(filfile: str, chans: np.ndarray, outfile: str) -> None:
    """Extract frequency channels from filterbank data."""
    fil = FilReader(filfile)
    fil.extract_chans(chans=chans, outfile_base=outfile)


@main.command()
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
    chanpersub: int,
    outfile: str,
) -> None:
    """Extract frequency bands from filterbank data."""
    fil = FilReader(filfile)
    fil.extract_bands(
        chanstart=chanstart,
        nchans=nchans,
        chanpersub=chanpersub,
        outfile_base=outfile,
    )


if __name__ == "__main__":
    main()
