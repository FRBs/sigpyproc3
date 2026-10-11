from __future__ import annotations

from typing import TYPE_CHECKING

import rich_click as click
from rich import print as rprint
from rich.pretty import pprint
from rich.table import Table

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
    try:
        if file_type == "auto":
            file_type = detect_file_type(filename)
        reader = file_readers.get(file_type)
        if reader is None:
            msg = f"Unsupported file format: {file_type}"
            raise click.UsageError(msg)
        header = reader(filename)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise click.ClickException(str(exc)) from exc
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
    try:
        header = Header.from_sigproc(filfile)
        values = header.to_dict()
        if key not in values:
            values = {**sigproc.parse_header(filfile), **values}
    except (OSError, ValueError, NotImplementedError) as exc:
        raise click.ClickException(str(exc)) from exc
    if key not in values:
        valid = ", ".join(sorted(values))
        msg = f"Unknown header key '{key}'. Valid keys: {valid}"
        raise click.UsageError(msg)
    click.echo(f"{key} = {values[key]}")


@main.command()
@click.argument("filfile", type=click.Path(exists=True))
@click.option(
    "-i",
    "--item",
    required=True,
    multiple=True,
    nargs=2,
    type=click.Tuple([str, str]),
    help="(key, value) to update in header; repeat for more keys",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show before/after values without writing",
)
def update(filfile: str, item: tuple[tuple[str, str], ...], *, dry_run: bool) -> None:
    """Update SIGPROC header keys in place.

    All values are checked first. If any check fails, the file is not changed.
    Header size never changes: a shorter string is padded with spaces, a longer one
    is refused, and keys absent from the file cannot be added.
    """
    updates: dict[str, str] = {}
    for key, value in item:
        if key not in sigproc.SIGPROC_SCHEMA:
            msg = (
                f"'{key}' is not a SIGPROC header key. See: spp_header get --keys-help"
            )
            raise click.BadParameter(msg, param_hint="--item")
        if key in updates:
            msg = f"'{key}' is given more than once."
            raise click.BadParameter(msg, param_hint="--item")
        updates[key] = value
    try:
        if dry_run:
            before, after = sigproc.prepare_header_edits(filfile, updates)
            _print_edit_preview(before, after, list(updates))
        else:
            sigproc.edit_headers(filfile, updates)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise click.ClickException(str(exc)) from exc


def _print_edit_preview(
    before: dict[str, object],
    after: dict[str, object],
    updated: list[str],
) -> None:
    table = Table(title="Header update preview")
    table.add_column("Key", overflow="fold")
    table.add_column("Before", overflow="fold")
    table.add_column("After", overflow="fold")
    keys = [k for k in sigproc.SIGPROC_SCHEMA if k in before]
    keys += [k for k in updated if k not in keys]
    for name in keys:
        style = "bold" if name in updated else None
        table.add_row(
            name,
            str(before.get(name, "")),
            str(after.get(name, "")),
            style=style,
        )
    rprint(table)


if __name__ == "__main__":
    main()
