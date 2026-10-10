from __future__ import annotations

import rich_click as click

from sigpyproc.combine import splice_filterbanks


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfiles", nargs=-1, type=click.Path(exists=True), required=True)
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
def main(filfiles: tuple[str, ...], gulp: int, outfile: str) -> None:
    """Splice simultaneous filterbanks along frequency (linear fch1/foff)."""
    path = splice_filterbanks(filfiles, outfile, gulp=gulp)
    click.echo(path)


if __name__ == "__main__":
    main()
