"""Unified interactive viewer CLI for sigpyproc data files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging


@logging_options
@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-s", "--start", type=int, default=0, help="Start sample index")
@click.option("-g", "--gulp", type=int, default=4096, help="Gulp size in samples")
@click.option("-d", "--dm", type=float, default=0.0, help="Initial DM (pc cm^-3)")
@click.option("-t", "--tfactor", type=int, default=1, help="Time downsample factor")
@click.option("-f", "--ffactor", type=int, default=1, help="Freq downsample factor")
@click.option(
    "-m",
    "--mask",
    type=click.Path(exists=True),
    default=None,
    help="RFIMask .h5 file",
)
@click.option(
    "--scale",
    type=click.Choice(["zscore", "minmax", "zscale", "percentile", "clip"]),
    default="zscore",
    help="Data scaling method",
)
@click.option("--cmap", type=str, default="magma", help="Colormap name")
@click.option(
    "--theme",
    type=click.Choice(["light", "dark"]),
    default="light",
    help="Theme",
)
@click.option(
    "--zerodm",
    is_flag=True,
    default=False,
    help="Subtract zero-DM mean",
)
@click.option(
    "--flatten",
    is_flag=True,
    default=False,
    help="Divide by bandpass median",
)
@click.option(
    "-b",
    "--batch",
    is_flag=True,
    default=False,
    help="Batch non-interactive mode",
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(),
    default=".",
    help="Output directory for batch",
)
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["png", "pdf", "pdf-multi"]),
    default="png",
    help="Batch output format",
)
@click.option("--stop", type=int, default=None, help="Batch stop sample index")
@click.option("-j", "--jobs", type=int, default=1, help="Parallel worker jobs")
def main(
    verbose: int,
    files: tuple[str, ...],
    start: int,
    gulp: int,
    dm: float,
    tfactor: int,
    ffactor: int,
    mask: str | None,
    scale: str,
    cmap: str,
    theme: str,
    *,
    zerodm: bool,
    flatten: bool,
    batch: bool,
    outdir: str,
    out_format: Literal["png", "pdf", "pdf-multi"],
    stop: int | None,
    jobs: int,
    quiet: bool,
) -> None:
    """Visualize pulsar/FRB filterbanks, time series, and power spectra."""
    setup_cli_logging(verbose, quiet=quiet)

    first_ext = Path(files[0]).suffix.lower()

    if first_ext in (".tim", ".dat"):
        from sigpyproc.timeseries import TimeSeries
        from sigpyproc.viz.series import SeriesViewer

        tim = (
            TimeSeries.from_tim(files[0])
            if first_ext == ".tim"
            else TimeSeries.from_dat(files[0])
        )
        viewer = SeriesViewer(tim)
        viewer.show()
        return

    if first_ext in (".fft", ".spec"):
        from sigpyproc.fourierseries import FourierSeries, PowerSpectrum
        from sigpyproc.viz.series import SpectrumViewer

        spec = (
            FourierSeries.from_fft(files[0]).form_spec()
            if first_ext == ".fft"
            else PowerSpectrum.from_spec(files[0])
        )
        viewer_spec = SpectrumViewer(spec)
        viewer_spec.show()
        return

    # Filterbank / PSRFITS / FBH5 files
    if batch:
        from sigpyproc.readers import open_filterbank
        from sigpyproc.viz.batch import render_waterfall_batch

        fil = open_filterbank(files[0])
        render_waterfall_batch(
            fil,
            outdir,
            start=start,
            stop=stop,
            gulp=gulp,
            dm=dm,
            tfactor=tfactor,
            ffactor=ffactor,
            mask=mask,
            scaler=scale,
            cmap=cmap,
            theme=theme,
            zerodm=zerodm,
            flatten=flatten,
            out_format=out_format,
            jobs=jobs,
        )
        return

    from sigpyproc.viz.waterfall import WaterfallViewer

    viewer = WaterfallViewer(
        files,
        start=start,
        gulp=gulp,
        dm=dm,
        tfactor=tfactor,
        ffactor=ffactor,
        mask=mask,
        scaler=scale,
        cmap=cmap,
        theme=theme,
        zerodm=zerodm,
        flatten=flatten,
    )
    viewer.show()


if __name__ == "__main__":
    main()
