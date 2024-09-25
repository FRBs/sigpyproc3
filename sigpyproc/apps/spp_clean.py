from __future__ import annotations

import rich_click as click

from sigpyproc.readers import FilReader


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-m",
    "--method",
    type=click.Choice(["mad", "iqrm"]),
    default="mad",
    help="RFI cleaning method to use.",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=3.0,
    help="Sigma threshold for RFI cleaning.",
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
    help="Output masked filterbank file",
)
@click.option(
    "-s",
    "--maskfile",
    type=click.Path(exists=False),
    default=None,
    help="Output mask file",
)
def main(
    filfile: str,
    method: str,
    threshold: float,
    outfile: str | None,
    maskfile: str | None,
    gulp: int,
) -> None:
    """Clean RFI from filterbank data."""
    fil = FilReader(filfile)
    _out_file, rfimask = fil.clean_rfi(
        method=method,
        threshold=threshold,
        outfile_name=outfile,
        gulp=gulp,
    )
    if maskfile is not None:
        rfimask.to_file(filename=maskfile)


if __name__ == "__main__":
    main()
