from __future__ import annotations

import csv
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
    help="Write CSV (chan, freq_mhz, mean, std, skew, kurtosis, min, max)",
)
def main(
    verbose: int, quiet: bool, filfile: str, gulp: int, outfile: str | None
) -> None:
    """Print per-channel statistics for a filterbank."""
    setup_cli_logging(verbose, quiet)
    fil = FilReader(filfile)
    fil.compute_stats(gulp=gulp)
    stats = fil.chan_stats
    if stats is None:
        msg = "Failed to compute channel statistics"
        raise RuntimeError(msg)

    rows = [
        {
            "chan": ichan,
            "freq_mhz": float(fil.header.chan_freqs[ichan]),
            "mean": float(stats.mean[ichan]),
            "std": float(stats.std[ichan]),
            "skew": float(stats.skew[ichan]),
            "kurtosis": float(stats.kurtosis[ichan]),
            "min": float(stats.minima[ichan]),
            "max": float(stats.maxima[ichan]),
        }
        for ichan in range(fil.header.nchans)
    ]
    fieldnames = list(rows[0])
    if outfile is not None:
        with Path(outfile).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        click.echo(outfile)
        return

    click.echo("\t".join(fieldnames))
    for row in rows:
        click.echo(
            f"{row['chan']}\t{row['freq_mhz']:.6f}\t{row['mean']:.6f}\t"
            f"{row['std']:.6f}\t{row['skew']:.6f}\t{row['kurtosis']:.6f}\t"
            f"{row['min']:.6f}\t{row['max']:.6f}",
        )


if __name__ == "__main__":
    main()
