import click

from sigpyproc.readers import FilReader
from sigpyproc.utils import get_logger


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "--cont",
    is_flag=True,
    help="Input files are contiguous (disable check)",
)
@click.option(
    "-b",
    "--nbits",
    type=int,
    default=8,
    help="Number of bits per output sample",
)
@click.option(
    "-B",
    "--block_size",
    type=int,
    default=128,
    help="block size in megabytes",
)
@click.option(
    "-c",
    "--rescale_constant",
    is_flag=True,
    help="Keep offset and scale constant",
)
@click.option(
    "-t",
    "--tscrunch_factor",
    type=int,
    default=1,
    help="Decimate in time",
)
@click.option(
    "-f",
    "--fscrunch_factor",
    type=int,
    default=1,
    help="Decimate in frequency",
)
@click.option(
    "-I",
    "--rescale_seconds",
    type=float,
    default=10.0,
    help="Rescale interval in seconds (0 -> disable rescaling)",
)
@click.option(
    "-s",
    "--scale_fac",
    type=float,
    default=1.0,
    help="Data scale factor to apply",
)
@click.option(
    "-scloffs",
    "--apply_FITS_scale_and_offset",
    is_flag=True,
    help="Denormalize using DAT_SCL and DAT_OFFS [PSRFITS]",
)
@click.option(
    "-o",
    "--output_filename",
    type=click.Path(),
    help="Output filename",
)
def main(
    filfile: str,
    cont: bool,
    nbits: int,
    block_size: int,
    rescale_constant: bool,
    tscrunch_factor: int,
    fscrunch_factor: int,
    rescale_seconds: float,
    scale_fac: float,
    apply_FITS_scale_and_offset: bool,
) -> None:
    """Convert to sigproc output digifil style."""
    msg = "This function is not implemented yet."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
