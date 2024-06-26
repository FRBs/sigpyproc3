from __future__ import annotations

import click

from sigpyproc.header import Header
from sigpyproc.io import sigproc


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main() -> None:
    pass


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
def print(filfile: str) -> None:  # noqa: A001
    """Print the header information."""
    header = Header.from_sigproc(filfile)
    click.echo(header.to_string())


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-k",
    "--key",
    required=True,
    type=str,
    help="A header key to read (e.g. telescope, fch1, nsamples)",
)
def get(filfile: str, key: str) -> None:
    """Get the value of a header key."""
    header = Header.from_sigproc(filfile)
    try:
        value = getattr(header, key)
    except AttributeError:
        hdr = sigproc.parse_header(filfile)
        value = hdr[key]
    click.echo(f"{key} = {value}")


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-i",
    "--item",
    required=True,
    nargs=2,
    type=click.Tuple([str, str]),
    help="(key, value) to update in header",
)
def update(filfile: str, item: tuple[str, str]) -> None:
    """Update a header key."""
    key, value = item
    sigproc.edit_header(filfile, key, value)


if __name__ == "__main__":
    main()
