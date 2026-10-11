from __future__ import annotations

from pathlib import Path

import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging
from sigpyproc.readers import FilReader


@logging_options
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
    help="Write CSV (freq_mhz, mean)",
)
def main(
    verbose: int, quiet: bool, filfile: str, gulp: int, outfile: str | None
) -> None:
    """Compute the mean bandpass of a filterbank."""
    setup_cli_logging(verbose, quiet)
    fil = FilReader(filfile)
    bpass = fil.bandpass(method="mean", gulp=gulp)
    freqs = fil.header.chan_freqs
    if outfile is not None:
        lines = ["freq_mhz,mean"]
        lines.extend(
            f"{freq:.6f},{val:.6f}" for freq, val in zip(freqs, bpass.data, strict=True)
        )
        Path(outfile).write_text("\n".join(lines) + "\n")
        click.echo(outfile)
        return
    click.echo("freq_mhz\tmean")
    for freq, val in zip(freqs, bpass.data, strict=True):
        click.echo(f"{freq:.6f}\t{val:.6f}")


if __name__ == "__main__":
    main()
