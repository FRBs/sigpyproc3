"""Headless and parallel batch renderer for filterbank data."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from rich.progress import Progress

from sigpyproc.core.logging import progress_disabled
from sigpyproc.viz._backend import paint_theme, viewer_rc
from sigpyproc.viz.scaling import SCALERS
from sigpyproc.viz.waterfall import WaterfallModel, WaterfallState

if TYPE_CHECKING:
    from sigpyproc.base import Filterbank
    from sigpyproc.core.rfi import RFIMask


def _load_mask(mask: str | Path | RFIMask | None) -> RFIMask | None:
    if mask is None or not isinstance(mask, (str, Path)):
        return mask
    from sigpyproc.core.rfi import RFIMask

    return RFIMask.from_file(str(mask))


def _render_single_block_fig(
    fil: Filterbank,
    state: WaterfallState,
    title: str | None = None,
    *,
    mask: RFIMask | None = None,
    theme: str = "light",
) -> Figure:
    """Render one gulp onto an Agg figure."""
    model = WaterfallModel(fil, mask=mask)
    state = model.clamp(state)
    products = model.products(state, screen_size=(640, 1100))

    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[4, 1],
        height_ratios=[1, 3],
        left=0.08,
        right=0.96,
        top=0.90,
        bottom=0.10,
        hspace=0.12,
        wspace=0.08,
    )

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_im = fig.add_subplot(gs[1, 0], sharex=ax_ts)
    ax_bp = fig.add_subplot(gs[1, 1], sharey=ax_im)
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    ax_ts.plot(products.ts_time, products.ts, color="#1f77b4", lw=1.0)
    ax_ts.set_xlim(products.extent[0], products.extent[1])
    ts_valid = products.ts[np.isfinite(products.ts)]
    if ts_valid.size > 0:
        ax_ts.set_ylim(float(np.min(ts_valid)), float(np.max(ts_valid)) * 1.05)
    ax_ts.set_ylabel("Power", fontsize=9)
    ax_ts.tick_params(labelbottom=False, labelsize=8)

    ax_im.imshow(
        products.image,
        aspect="auto",
        origin="lower",
        cmap=state.cmap,
        extent=products.extent,
        vmin=products.clim[0],
        vmax=products.clim[1],
        interpolation="nearest",
    )
    ax_im.set_xlabel("Time (s)", fontsize=9)
    ax_im.set_ylabel("Frequency (MHz)", fontsize=9)
    ax_im.tick_params(labelsize=8)

    order = np.argsort(products.bp_freqs)
    bp_freqs = products.bp_freqs[order]
    bp_mean = products.bp_mean[order]
    bp_std = np.where(np.isfinite(products.bp_std[order]), products.bp_std[order], 0.0)
    ax_bp.plot(bp_mean, bp_freqs, color="#1f77b4", lw=1.0)
    ax_bp.fill_betweenx(
        bp_freqs,
        bp_mean - bp_std,
        bp_mean + bp_std,
        color="#1f77b4",
        alpha=0.25,
        lw=0,
    )
    ax_bp.set_ylim(products.extent[2], products.extent[3])
    bp_valid = products.bp_mean[np.isfinite(products.bp_mean)]
    if bp_valid.size > 0:
        ax_bp.set_xlim(float(np.min(bp_valid)), float(np.max(bp_valid)) * 1.05)
    ax_bp.set_xlabel("Mean", fontsize=9)
    ax_bp.tick_params(labelleft=False, labelsize=8)

    scaler_name = SCALERS[state.scaler_idx % len(SCALERS)].name
    stats = products.stats
    info_lines = [
        f"File: {fil.header.basename}",
        f"Start: {state.start} ({products.extent[0]:.2f}s)",
        f"Gulp: {state.nsamps} | DM: {state.dm:.2f}",
        f"Tscrunch: {state.tfactor}x | Fscrunch: {state.ffactor}x",
        f"Scaler: {scaler_name}",
        f"Zero-DM: {state.zerodm} | Flat: {state.flatten}",
        f"mean {stats['mean']:.3g}  std {stats['std']:.3g}",
    ]
    info = ax_info.text(
        0.05,
        0.95,
        "\n".join(info_lines),
        transform=ax_info.transAxes,
        va="top",
        fontsize=8,
        family="monospace",
    )
    header_title = title or f"{fil.header.basename} - t={products.extent[0]:.2f}s"
    fig.suptitle(header_title, fontsize=10, fontweight="bold")
    paint_theme(fig, [ax_ts, ax_im, ax_bp, ax_info], theme, texts=[info])
    # Keep the info axes the same colour as the figure so the panel reads as text.
    figure_face = viewer_rc(theme)["figure.facecolor"]
    if isinstance(figure_face, str):
        ax_info.set_facecolor(figure_face)
    return fig


def _stream_filenames(fil: Filterbank) -> list[str]:
    """Every file in a multi-file stream, not only ``header.filename``."""
    entries = getattr(fil.header, "stream_info", None)
    names = [entry.filename for entry in getattr(entries, "entries", [])]
    names = [name for name in names if name]
    if names:
        return names
    return [str(fil.header.filename)]


def _batch_one(task: dict[str, Any]) -> str:
    """Render one gulp in this process. Safe to call from a spawn pool."""
    # Local import: viz is loaded while readers is still initialising.
    from sigpyproc.readers import FilReader, open_filterbank

    names = task["filenames"]
    fil = FilReader(names) if len(names) > 1 else open_filterbank(names[0])
    fig = None
    try:
        raw_mask = task["mask"]
        mask = _load_mask(raw_mask) if isinstance(raw_mask, (str, Path)) else raw_mask
        state = WaterfallState(**task["state"])
        fig = _render_single_block_fig(
            fil,
            state,
            mask=mask,
            theme=task["theme"],
        )
        out = str(task["out"])
        fig.savefig(out, dpi=120)
        return out
    finally:
        fil.close()
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)


def _run_tasks(tasks: list[dict[str, Any]], jobs: int) -> list[str]:
    if not tasks:
        return []
    if jobs <= 1:
        outs: list[str] = []
        with Progress(disable=progress_disabled()) as progress:
            bar = progress.add_task("spp_view", total=len(tasks))
            for task in tasks:
                outs.append(_batch_one(task))
                progress.advance(bar)
        return outs

    ctx = mp.get_context("spawn")
    with Progress(disable=progress_disabled()) as progress, ctx.Pool(jobs) as pool:
        bar = progress.add_task("spp_view", total=len(tasks))
        outs = []
        for out in pool.imap(_batch_one, tasks):
            outs.append(out)
            progress.advance(bar)
    return outs


def _stitch_pdf(pngs: list[str], pdf_path: Path) -> None:
    import matplotlib.image as mpimg

    with PdfPages(pdf_path) as pdf:
        for png in pngs:
            fig = Figure(figsize=(10, 6))
            FigureCanvasAgg(fig)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.axis("off")
            ax.imshow(mpimg.imread(png))
            pdf.savefig(fig)
            Path(png).unlink(missing_ok=True)


def render_waterfall_batch(
    fil: Filterbank,
    outdir: str | Path | None = None,
    *,
    start: int = 0,
    stop: int | None = None,
    gulp: int = 4096,
    dm: float = 0.0,
    tfactor: int = 1,
    ffactor: int = 1,
    mask: str | Path | RFIMask | None = None,
    scaler: str = "zscore",
    cmap: str = "magma",
    theme: str = "light",
    zerodm: bool = False,
    flatten: bool = False,
    out_format: Literal["png", "pdf", "pdf-multi"] = "png",
    jobs: int = 1,
) -> list[str]:
    """Render sequential gulps of a filterbank to disk.

    ``jobs`` above 1 uses a spawn process pool. Each worker builds its own Agg
    figure. A multi-page PDF with one job is written directly. With several
    jobs the pages are rendered to PNG and then stitched, so order is preserved.

    Parameters
    ----------
    fil : Filterbank
        Input filterbank instance.
    outdir : str or Path or None, optional
        Output directory. If None, the current directory is used.
    start, stop, gulp, dm, tfactor, ffactor
        Window, dispersion measure, and scrunch passed to each gulp.
    mask : str or Path or RFIMask or None, optional
        Channel mask applied to every gulp.
    scaler, cmap, theme, zerodm, flatten
        Display settings.
    out_format : {"png", "pdf", "pdf-multi"}, optional
        One file per gulp, or a single multi-page PDF.
    jobs : int, optional
        Worker processes, by default 1.

    Returns
    -------
    list of str
        Paths of the files left on disk. Temporary page images are removed.
    """
    out_dir_path = Path(outdir) if outdir is not None else Path.cwd()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    total_nsamps = fil.header.nsamples
    end_samp = total_nsamps if stop is None else min(stop, total_nsamps)
    starts = list(range(start, end_samp, gulp))
    mask_path = str(mask) if isinstance(mask, (str, Path)) else None
    if jobs > 1 and mask is not None and mask_path is None:
        msg = "batch jobs>1 needs the mask as a file path"
        raise TypeError(msg)
    # A live RFIMask stays in-process. A path is what a spawn worker can load.
    mask_for_task = mask_path if jobs > 1 else mask

    scaler_names = [s.name.lower() for s in SCALERS]
    scaler_idx = (
        scaler_names.index(scaler.lower()) if scaler.lower() in scaler_names else 0
    )

    def _state(sample: int) -> dict[str, Any]:
        return {
            "start": sample,
            "nsamps": gulp,
            "dm": dm,
            "tfactor": tfactor,
            "ffactor": ffactor,
            "scaler_idx": scaler_idx,
            "cmap": cmap,
            "zerodm": zerodm,
            "flatten": flatten,
        }

    if out_format == "pdf-multi" and jobs <= 1:
        multi_pdf_path = out_dir_path / f"{fil.header.basename}_dm{dm:.2f}.pdf"
        loaded = _load_mask(mask)
        with (
            PdfPages(multi_pdf_path) as pdf,
            Progress(disable=progress_disabled()) as progress,
        ):
            bar = progress.add_task("spp_view", total=len(starts))
            for sample in starts:
                fig = _render_single_block_fig(
                    fil,
                    WaterfallState(**_state(sample)),
                    mask=loaded,
                    theme=theme,
                )
                pdf.savefig(fig)
                progress.advance(bar)
        return [str(multi_pdf_path)]

    ext = ".pdf" if out_format == "pdf" else ".png"
    tasks: list[dict[str, Any]] = []
    for sample in starts:
        if out_format == "pdf-multi":
            out_file = out_dir_path / f".{fil.header.basename}_t{sample:08d}.png"
        else:
            out_file = (
                out_dir_path / f"{fil.header.basename}_t{sample:08d}_dm{dm:.2f}{ext}"
            )
        tasks.append(
            {
                "filenames": _stream_filenames(fil),
                "mask": mask_for_task,
                "theme": theme,
                "state": _state(sample),
                "out": str(out_file),
            }
        )

    written = _run_tasks(tasks, jobs)
    if out_format == "pdf-multi":
        multi_pdf_path = out_dir_path / f"{fil.header.basename}_dm{dm:.2f}.pdf"
        _stitch_pdf(written, multi_pdf_path)
        return [str(multi_pdf_path)]
    return written
