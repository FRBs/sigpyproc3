"""Unified interactive viewer CLI for sigpyproc data files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import rich_click as click

from sigpyproc.apps._logging import logging_options, setup_cli_logging

if TYPE_CHECKING:
    from sigpyproc.fourierseries import PowerSpectrum


def _read_spectrum(filename: str, suffix: str) -> PowerSpectrum:
    """Read amplitude, complex, or legacy complex spectra on their real grid."""
    from sigpyproc.fourierseries import FourierSeries, PowerSpectrum
    from sigpyproc.header import Header

    if suffix == ".fft":
        return FourierSeries.from_fft(filename).form_spec()
    header = Header.from_sigproc(filename)
    if header.data_type == "amplitude spectrum":
        return PowerSpectrum.from_spec(filename)
    return FourierSeries.from_spec(filename).form_spec()


def _echo_paths(paths: list[str]) -> None:
    for path in paths:
        click.echo(path)


def _batch_series(
    files: tuple[str, ...],
    suffix: str,
    *,
    theme: str,
    start: int,
    gulp: int,
    outdir: str,
) -> list[str]:
    from sigpyproc.timeseries import TimeSeries
    from sigpyproc.viz.series import SeriesViewer

    Path(outdir).mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name in files:
        tim = (
            TimeSeries.from_tim(name) if suffix == ".tim" else TimeSeries.from_dat(name)
        )
        viewer = SeriesViewer(tim, theme=theme, batch=True)
        viewer.start_samp = max(0, start)
        viewer.end_samp = min(len(tim.data), viewer.start_samp + max(gulp, 1))
        out = Path(outdir) / f"{tim.header.basename}_timeseries.png"
        written.append(viewer.save_png(str(out)))
    return written


def _batch_spectra(
    files: tuple[str, ...],
    suffix: str,
    *,
    theme: str,
    outdir: str,
) -> list[str]:
    from sigpyproc.viz.series import SpectrumViewer

    Path(outdir).mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name in files:
        spec = _read_spectrum(name, suffix)
        viewer = SpectrumViewer(spec, theme=theme, batch=True)
        out = Path(outdir) / f"{spec.header.basename}_spec.png"
        written.append(viewer.save_png(str(out)))
    return written


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

    exts = {Path(name).suffix.lower() for name in files}
    if len(exts) != 1:
        msg = f"batch inputs must share one suffix, got {sorted(exts)}"
        raise click.UsageError(msg)
    first_ext = exts.pop()
    if not batch and first_ext in {".tim", ".dat", ".fft", ".spec"} and len(files) > 1:
        msg = "pass one time series or spectrum, or use --batch to render every file"
        raise click.UsageError(msg)

    if first_ext in (".tim", ".dat"):
        if batch:
            _echo_paths(
                _batch_series(
                    files, first_ext, theme=theme, start=start, gulp=gulp, outdir=outdir
                ),
            )
            return
        from sigpyproc.timeseries import TimeSeries
        from sigpyproc.viz.series import SeriesViewer

        tim = (
            TimeSeries.from_tim(files[0])
            if first_ext == ".tim"
            else TimeSeries.from_dat(files[0])
        )
        SeriesViewer(tim, theme=theme).show()
        return

    if first_ext in (".fft", ".spec"):
        if batch:
            _echo_paths(_batch_spectra(files, first_ext, theme=theme, outdir=outdir))
            return
        from sigpyproc.viz.series import SpectrumViewer

        spec = _read_spectrum(files[0], first_ext)
        SpectrumViewer(spec, theme=theme).show()
        return

    # Filterbank / PSRFITS / FBH5 files
    if batch:
        from sigpyproc.readers import open_filterbank
        from sigpyproc.viz.batch import render_waterfall_batch

        written = []
        for name in files:
            fil = open_filterbank(name)
            try:
                written.extend(
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
                    ),
                )
            finally:
                fil.close()
        for path in written:
            click.echo(path)
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
