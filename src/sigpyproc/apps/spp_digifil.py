import attrs
import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.io.digifil import DigifilConfig, default_gulp, digifil


@logging_options
@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-b",
    "--nbits",
    type=int,
    default=8,
    help="Number of bits per output sample (1, 2, 4, 8, 16, or -32 for float)",
)
@click.option(
    "-B",
    "--block_size",
    type=float,
    default=None,
    help="Accepted for dspsr compatibility; ignored (dspsr uses -U for block size)",
)
@click.option(
    "-U",
    "--max_ram",
    type=float,
    default=None,
    help="RAM limit in MB (dspsr default 256); sets the input block as dspsr does",
)
@click.option(
    "--gulp",
    type=int,
    default=None,
    help="Input samples per block (overrides -U); default is dspsr's block for 256 MB",
)
@click.option(
    "-c",
    "--rescale_constant",
    is_flag=True,
    help="Keep offset and scale constant after the first estimate",
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
    "--scloffs",
    is_flag=True,
    help="Denormalise using DAT_SCL and DAT_OFFS [PSRFITS]",
)
@click.option(
    "-d",
    "--npol",
    type=click.IntRange(1, 4),
    default=1,
    help="1=PP+QQ intensity, 2 or 4=keep all input polarisations",
)
@click.option(
    "-P",
    "--poln_select",
    type=int,
    default=-1,
    help="Output only this input polarisation index (-1 keeps all)",
)
@click.option(
    "-o",
    "--output_filename",
    type=click.Path(),
    help="Output filename (default: <input stem>.fil)",
)
def main(
    verbose: int,
    filfile: str,
    nbits: int,
    block_size: float | None,  # noqa: ARG001
    max_ram: float | None,
    gulp: int | None,
    *,
    quiet: bool,
    rescale_constant: bool,
    tscrunch_factor: int,
    fscrunch_factor: int,
    rescale_seconds: float,
    scale_fac: float,
    scloffs: bool,
    npol: int,
    poln_select: int,
    output_filename: str | None,
) -> None:
    """Convert to Sigproc filterbank format, following dspsr digifil."""
    setup_cli_logging(verbose, quiet=quiet)
    if gulp is not None and max_ram is not None:
        msg = "give either --gulp or -U, not both"
        raise click.UsageError(msg)
    config = DigifilConfig(
        nbits=nbits,
        gulp=gulp,
        rescale_seconds=rescale_seconds,
        rescale_constant=rescale_constant,
        tscrunch_factor=tscrunch_factor,
        fscrunch_factor=fscrunch_factor,
        scale_fac=scale_fac,
        scloffs=scloffs,
        npol=npol,
        poln_select=poln_select,
    )
    try:
        if max_ram is not None:
            config = attrs.evolve(
                config,
                gulp=default_gulp(filfile, config, max_ram_mb=max_ram),
            )
        outfile = digifil(filfile, output_filename, config)
    except (ValueError, NotImplementedError) as err:
        raise click.ClickException(str(err)) from err
    click.echo(outfile)


if __name__ == "__main__":
    main()
