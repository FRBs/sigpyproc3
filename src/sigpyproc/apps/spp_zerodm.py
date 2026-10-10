from __future__ import annotations

import rich_click as click

from sigpyproc.readers import FilReader


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("filfile", type=click.Path(exists=True))
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
def main(filfile: str, gulp: int, outfile: str | None) -> None:
    """Subtract the channel-weighted zero-DM (Eatough / PRESTO)."""
    fil = FilReader(filfile)
    path = fil.remove_zerodm(outfile_name=outfile, gulp=gulp)
    click.echo(path)


if __name__ == "__main__":
    main()
