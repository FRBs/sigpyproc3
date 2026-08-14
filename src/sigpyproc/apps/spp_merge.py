from __future__ import annotations

from typing import Literal

import rich_click as click

from sigpyproc.combine import merge_filterbanks


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfiles", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "-a",
    "--axis",
    type=click.Choice(["auto", "time", "freq"]),
    default="auto",
    help="Concatenate in time, frequency, or pick automatically",
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
    required=True,
    help="Output filename",
)
def main(
    filfiles: tuple[str, ...],
    axis: Literal["auto", "time", "freq"],
    gulp: int,
    outfile: str,
) -> None:
    """Merge filterbanks in time or frequency."""
    path = merge_filterbanks(filfiles, outfile, axis=axis, gulp=gulp)
    click.echo(path)


if __name__ == "__main__":
    main()
