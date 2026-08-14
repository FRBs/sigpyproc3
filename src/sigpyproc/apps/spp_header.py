from __future__ import annotations

from typing import TYPE_CHECKING

import rich_click as click
from rich import print as rprint
from rich.pretty import pprint

from sigpyproc.header import Header
from sigpyproc.io import sigproc
from sigpyproc.utils import detect_file_type

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

file_readers: dict[str, Callable] = {
    "sigproc": Header.from_sigproc,
    "pfits": Header.from_pfits,
    "fbh5": Header.from_fbh5,
}


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main() -> None:
    """Print or edit filterbank headers."""


@main.command("print")
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "-f",
    "--file_type",
    type=click.Choice(["auto", "sigproc", "pfits", "fbh5"]),
    default="auto",
    help="File type",
)
@click.option(
    "--long",
    is_flag=True,
    default=False,
    help="Print the full Header information",
)
def print_header(filename: str | Path, file_type: str, *, long: bool = False) -> None:
    """Print the header information."""
    if file_type == "auto":
        file_type = detect_file_type(filename)
    reader = file_readers.get(file_type)
    if reader is None:
        msg = f"Unsupported file format: {file_type}"
        raise ValueError(msg)
    header = reader(filename)
    if long:
        pprint(header)
    else:
        rprint(header.to_string())


@main.command()
@click.argument("filfile", type=click.Path(exists=True), required=False)
@click.option(
    "-k",
    "--key",
    type=str,
    default=None,
    help="A header key to read (e.g. telescope, fch1, nsamples)",
)
@click.option(
    "--keys-help",
    "keys_help",
    is_flag=True,
    default=False,
    help="Print recognised SIGPROC header keys and exit",
)
def get(filfile: str | None, key: str | None, *, keys_help: bool = False) -> None:
    """Get the value of a header key."""
    if keys_help:
        for name, field in sigproc.SIGPROC_SCHEMA.items():
            click.echo(f"{name:<15}- {field.doc}")
        return
    if filfile is None or key is None:
        msg = "Provide FILFILE and -k/--key (or --keys-help)"
        raise click.UsageError(msg)
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
    """Update a SIGPROC header key in place."""
    key, value = item
    sigproc.edit_header(filfile, key, value)


if __name__ == "__main__":
    main()
