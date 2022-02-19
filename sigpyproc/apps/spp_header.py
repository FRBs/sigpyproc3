from __future__ import annotations
import click

from sigpyproc.header import Header
from sigpyproc.io.sigproc import edit_header


@click.group(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.argument("filfile", type=click.Path(exists=True))
def main(filfile: str) -> None:
    """Print or edit the header of a filterbank file."""
    header = Header.from_sigproc(filfile)
    click.echo(header.to_string())


@main.command()
@click.option(
    "-k", "--key", type=str, help="A header key to read (e.g. telescope, fch1, nsamples)"
)
def get(filfile: str, key: str) -> None:
    header = Header.from_sigproc(filfile)
    click.echo(f"{key} = {getattr(header, key)}")


@main.command()
@click.option(
    "-i",
    "--item",
    nargs=2,
    type=click.Tuple([str, str]),
    help="(key, value) to update in header",
)
def update(filfile: str, item: tuple[str, str]) -> None:
    key, value = item
    edit_header(filfile, key, value)


if __name__ == "__main__":
    main()
