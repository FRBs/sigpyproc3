"""Interactive filterbank waterfall viewer and its headless model."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np

from sigpyproc import kernels
from sigpyproc.block import FilterbankBlock
from sigpyproc.core.dispersion import DM_CONSTANT_LK
from sigpyproc.viz._backend import (
    ensure_interactive_backend,
    paint_theme,
    use_headless_backend,
    viewer_context,
)
from sigpyproc.viz.keymap import BlitManager, KeyMap
from sigpyproc.viz.scaling import SCALERS, frequency_oriented

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from matplotlib.backend_bases import KeyEvent, MouseEvent
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    from sigpyproc.base import Filterbank
    from sigpyproc.core.rfi import RFIMask


@contextmanager
def _quiet_nan() -> Generator[None, None, None]:
    """Hide empty-slice warnings from channels a mask or zap has set to NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            yield


_CMAPS: tuple[str, ...] = (
    "magma",
    "viridis",
    "plasma",
    "cividis",
    "gray",
    "gray_r",
    "RdBu_r",
    "cubehelix",
    "jet",
)
_DISPLAY_POINTS = 2000
_SCREEN_CAP = 1024

_LIGHT_BUTTONS = {
    "nav": "#d6e6f5",
    "dm": "#ffe8d2",
    "disp": "#ececf1",
    "save": "#d9f2e3",
    "file": "#e4dff5",
}
_DARK_BUTTONS = {
    "nav": "#243044",
    "dm": "#3a2a22",
    "disp": "#2a2a32",
    "save": "#1d3328",
    "file": "#2c2840",
}


def snap_divisor(n: int, factor: int) -> int:
    """Largest divisor of ``n`` that is at most ``factor``."""
    n = max(1, int(n))
    factor = max(1, min(int(factor), n))
    if n % factor == 0:
        return factor
    for candidate in range(factor, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def step_divisor(n: int, current: int, *, larger: bool) -> int:
    """Next divisor of ``n`` above or below ``current``."""
    n = max(1, int(n))
    current = snap_divisor(n, current)
    if larger:
        for candidate in range(current + 1, n + 1):
            if n % candidate == 0:
                return candidate
        return current
    for candidate in range(current - 1, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


@attrs.define
class WaterfallState:
    """State of the waterfall viewer."""

    start: int = 0
    nsamps: int = 4096
    dm: float = 0.0
    tfactor: int = 1
    ffactor: int = 1
    scaler_idx: int = 0
    zerodm: bool = False
    flatten: bool = False
    apply_mask: bool = True
    lod_mode: int = 0
    cmap: str = "magma"
    invert: bool = False
    theme: str = "light"
    hires: bool = False
    period: float | None = None
    phase: float = 0.0
    view_xlim: tuple[float, float] | None = None
    zapped_channels: set[int] = attrs.field(factory=set)

    def clone(self) -> WaterfallState:
        """Return a detached copy, including the zap set."""
        return attrs.evolve(self, zapped_channels=set(self.zapped_channels))


class History:
    """Undo/redo navigation history stack."""

    def __init__(self, maxlen: int = 100) -> None:
        self.maxlen = maxlen
        self._past: list[WaterfallState] = []
        self._future: list[WaterfallState] = []

    def push(self, state: WaterfallState) -> None:
        """Remember ``state`` and drop the redo branch."""
        self._past.append(state.clone())
        if len(self._past) > self.maxlen:
            self._past.pop(0)
        self._future.clear()

    def back(self, current: WaterfallState) -> WaterfallState | None:
        """Pop one state, parking ``current`` on the redo stack."""
        if not self._past:
            return None
        self._future.append(current.clone())
        return self._past.pop()

    def forward(self, current: WaterfallState) -> WaterfallState | None:
        """Re-apply one redo state."""
        if not self._future:
            return None
        self._past.append(current.clone())
        return self._future.pop()


@attrs.frozen
class WaterfallProducts:
    """Renderable products computed from a block."""

    image: np.ndarray
    extent: tuple[float, float, float, float]
    ts_time: np.ndarray
    ts: np.ndarray
    bp_freqs: np.ndarray
    bp_mean: np.ndarray
    bp_std: np.ndarray
    clim: tuple[float, float]
    stats: dict[str, Any]
    padded_end: bool = False


class WaterfallModel:
    """Read, dedisperse, mask, and decimate filterbank data for display."""

    def __init__(
        self,
        fil: Filterbank,
        *,
        mask: RFIMask | None = None,
        cache_blocks: int = 8,
        max_block_bytes: int = 512 << 20,
    ) -> None:
        self.fil = fil
        self.mask = mask
        self.max_block_bytes = max_block_bytes
        self.cache_blocks = cache_blocks
        self._cache: dict[tuple[Any, ...], FilterbankBlock] = {}
        self._cache_keys: list[tuple[Any, ...]] = []

    def clamp(self, state: WaterfallState) -> WaterfallState:
        """Clamp the window and snap scrunch factors to exact divisors."""
        header = self.fil.header
        total_samps = header.nsamples

        bytes_per_sample = header.nchans * 4
        max_nsamps = max(64, self.max_block_bytes // max(1, bytes_per_sample))
        nsamps = max(64, min(state.nsamps, total_samps, max_nsamps))
        nsamps = min(nsamps, total_samps)

        start = max(0, min(state.start, max(0, total_samps - 1)))
        if start + nsamps > total_samps:
            start = max(0, total_samps - nsamps)

        cloned = state.clone()
        cloned.start = start
        cloned.nsamps = nsamps
        cloned.tfactor = snap_divisor(nsamps, state.tfactor)
        cloned.ffactor = snap_divisor(header.nchans, state.ffactor)
        cloned.dm = max(0.0, float(state.dm))
        return cloned

    def _read_raw_block(self, start: int, nsamps: int, dm: float) -> FilterbankBlock:
        """Read and, when requested, dedisperse one block from disk."""
        if dm != 0:
            try:
                return self.fil.read_dedisp_block(start, nsamps, dm)
            except (ValueError, RuntimeError):
                blk = self.fil.read_block(start, nsamps)
                return blk.dedisperse(dm)
        return self.fil.read_block(start, nsamps)

    def _window(self, state: WaterfallState) -> tuple[int, int]:
        total = self.fil.header.nsamples
        nsamps = max(1, min(int(state.nsamps), total))
        start = max(0, min(int(state.start), max(0, total - 1)))
        if start + nsamps > total:
            nsamps = max(1, total - start)
        return start, nsamps

    def _touch(self, key: tuple[Any, ...]) -> None:
        self._cache_keys.remove(key)
        self._cache_keys.append(key)

    def _store(self, key: tuple[Any, ...], block: FilterbankBlock) -> None:
        if key in self._cache:
            self._touch(key)
        else:
            self._cache_keys.append(key)
        self._cache[key] = block
        while len(self._cache_keys) > self.cache_blocks:
            oldest = self._cache_keys.pop(0)
            self._cache.pop(oldest, None)

    def _raw_block(self, state: WaterfallState) -> FilterbankBlock:
        """Dedispersed block for this window, without mask, zap, or scrunch."""
        start, nsamps = self._window(state)
        key = (start, nsamps, float(state.dm))
        cached = self._cache.get(key)
        if cached is not None:
            self._touch(key)
            return cached
        blk = self._read_raw_block(start, nsamps, state.dm)
        # Readers may hand back a view of a reused buffer. Own the bytes.
        owned = FilterbankBlock(
            np.array(blk.data, dtype=np.float32, copy=True),
            blk.header,
            dm=state.dm,
        )
        self._store(key, owned)
        return owned

    def _apply_filters(
        self, data: np.ndarray, state: WaterfallState, nchans: int
    ) -> None:
        """Apply channel masking, zapping, zero-DM, and flattening in place."""
        if self.mask is not None and state.apply_mask:
            mask_arr = np.asarray(self.mask.chan_mask)
            if mask_arr.shape == (nchans,):
                data[mask_arr, :] = np.nan

        if state.zapped_channels:
            zapped = [ch for ch in state.zapped_channels if 0 <= ch < nchans]
            if zapped:
                data[np.fromiter(zapped, dtype=np.intp, count=len(zapped)), :] = np.nan

        if state.zerodm:
            with _quiet_nan():
                spectrum = np.nanmean(data, axis=0)
            data -= spectrum[None, :]

        if state.flatten:
            with _quiet_nan():
                band = np.nanmedian(data, axis=1)
            band = np.where((band == 0) | ~np.isfinite(band), 1.0, band)
            data /= band[:, None]

    def _downsample(
        self, block: FilterbankBlock, state: WaterfallState
    ) -> FilterbankBlock:
        ffactor = max(1, state.ffactor)
        tfactor = max(1, state.tfactor)
        if ffactor == 1 and tfactor == 1:
            return block
        if not np.isnan(block.data).any():
            reduced = block.downsample(ffactor=ffactor, tfactor=tfactor)
            return FilterbankBlock(reduced.data, reduced.header, dm=state.dm)

        data = block.data
        new_nc = data.shape[0] // ffactor
        new_ns = data.shape[1] // tfactor
        cropped = data[: new_nc * ffactor, : new_ns * tfactor]
        with _quiet_nan():
            reduced_data = np.nanmean(
                cropped.reshape(new_nc, ffactor, new_ns, tfactor),
                axis=(1, 3),
            ).astype(np.float32, copy=False)
        changes = {
            "tsamp": block.header.tsamp * tfactor,
            "foff": block.header.foff * ffactor,
            "fch1": block.header.fch1 + block.header.foff * (ffactor - 1) / 2,
            "nsamples": new_ns,
            "nchans": new_nc,
        }
        return FilterbankBlock(
            reduced_data,
            block.header.new_header(changes),
            dm=state.dm,
        )

    def get_block(self, state: WaterfallState) -> tuple[FilterbankBlock, bool]:
        """Return the filtered, scrunched block for ``state``.

        The disk cache ignores zaps and scalers, so a channel click does not
        read the file again.
        """
        raw = self._raw_block(state)
        data = np.array(raw.data, dtype=np.float32, copy=True)
        self._apply_filters(data, state, data.shape[0])
        processed = self._downsample(
            FilterbankBlock(data, raw.header, dm=state.dm),
            state,
        )
        return processed, False

    def channel_stats(
        self, state: WaterfallState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mean, variance, max, and min of the cached dedispersed gulp."""
        raw = self._raw_block(state)
        with _quiet_nan():
            mean = np.nanmean(raw.data, axis=1)
            var = np.nanvar(raw.data, axis=1)
            maxima = np.nanmax(raw.data, axis=1)
            minima = np.nanmin(raw.data, axis=1)
        return mean, var, maxima, minima

    def products(
        self,
        state: WaterfallState,
        screen_size: tuple[int, int] = (300, 600),
    ) -> WaterfallProducts:
        """Build the screen image, time series, bandpass, and contrast limits."""
        blk, padded_end = self.get_block(state)
        data = blk.data
        nchans, nsamps = data.shape

        hdr = blk.header
        t0 = state.start * self.fil.header.tsamp
        t1 = t0 + (nsamps * hdr.tsamp)
        freqs = np.asarray(hdr.chan_freqs)

        with _quiet_nan():
            ts = np.nanmean(data, axis=0)
            bp_mean = np.nanmean(data, axis=1)
            bp_std = np.nanstd(data, axis=1)

        ts_time = np.linspace(t0, t1, nsamps, endpoint=False)
        scaler = SCALERS[state.scaler_idx % len(SCALERS)]
        prepared = scaler.prepare(data)
        clim = scaler.clim(prepared)

        if state.hires:
            decimated = prepared
        else:
            out_rows = min(max(1, screen_size[0]), nchans)
            out_cols = min(max(1, screen_size[1]), nsamps)
            if out_rows == nchans and out_cols == nsamps:
                decimated = prepared
            else:
                decimated = kernels.decimate_2d(
                    np.ascontiguousarray(prepared, dtype=np.float32),
                    out_rows,
                    out_cols,
                    mode=state.lod_mode,
                )

        image, f_lo, f_hi = frequency_oriented(decimated, freqs, foff=hdr.foff)
        extent = (t0, t1, f_lo, f_hi)

        def _finite(value: float, fallback: float = 0.0) -> float:
            return float(value) if np.isfinite(value) else fallback

        with _quiet_nan():
            stats = {
                "mean": _finite(float(np.nanmean(data))),
                "std": _finite(float(np.nanstd(data))),
                "min": _finite(float(np.nanmin(data))),
                "max": _finite(float(np.nanmax(data))),
                "ts_max": _finite(float(np.nanmax(ts))),
                "ts_mean": _finite(float(np.nanmean(ts))),
            }

        return WaterfallProducts(
            image=image,
            extent=extent,
            ts_time=ts_time,
            ts=ts,
            bp_freqs=freqs,
            bp_mean=bp_mean,
            bp_std=bp_std,
            clim=clim,
            stats=stats,
            padded_end=padded_end,
        )

    @staticmethod
    def fit_dm(t1: float, f1: float, t2: float, f2: float) -> float:
        """Dispersion measure that connects two ``(time_s, freq_MHz)`` points."""
        if f1 <= 0 or f2 <= 0 or f1 == f2:
            return 0.0
        dt = t2 - t1
        df_inv = (f2**-2) - (f1**-2)
        if df_inv == 0:
            return 0.0
        return float(dt / (DM_CONSTANT_LK * df_inv))


class WaterfallViewer:
    """Interactive GUI viewer for filterbank dynamic spectra and time series."""

    def __init__(
        self,
        files: str | Path | Sequence[str | Path] | Filterbank,
        *,
        start: int = 0,
        gulp: int = 4096,
        dm: float = 0.0,
        tfactor: int = 1,
        ffactor: int = 1,
        mask: str | Path | RFIMask | None = None,
        cmap: str = "magma",
        scaler: str = "zscore",
        theme: str = "light",
        zerodm: bool = False,
        flatten: bool = False,
        batch: bool = False,
        gui: bool | None = None,
    ) -> None:
        if batch:
            use_headless_backend()
        else:
            ensure_interactive_backend()
        # Batch mode is the headless path. Building the widget figure reads a
        # gulp and draws it, which dominates viewer tests that only check state.
        self._gui = (not batch) if gui is None else gui

        from sigpyproc.base import Filterbank
        from sigpyproc.readers import open_filterbank

        if isinstance(files, Filterbank):
            self.file_list = [str(files.header.filename)]
            self.fil = files
            self._owns_reader = False
        elif isinstance(files, (str, Path)):
            self.file_list = [str(files)]
            self.fil = open_filterbank(self.file_list[0])
            self._owns_reader = True
        else:
            self.file_list = [str(f) for f in files]
            self.fil = open_filterbank(self.file_list[0])
            self._owns_reader = True

        self.file_idx = 0
        self.rfimask = self._load_mask(mask)
        scaler_idx = 0
        for i, scaler_obj in enumerate(SCALERS):
            if scaler_obj.name == scaler:
                scaler_idx = i
                break

        self.state = WaterfallState(
            start=start,
            nsamps=gulp,
            dm=dm,
            tfactor=tfactor,
            ffactor=ffactor,
            scaler_idx=scaler_idx,
            cmap=cmap,
            theme=theme,
            zerodm=zerodm,
            flatten=flatten,
        )
        self.history = History()
        self.model = WaterfallModel(self.fil, mask=self.rfimask)
        self.keymap = KeyMap()
        self._init_keymap()

        self._fit_points: list[tuple[float, float]] = []
        self._dm_fit_armed = False
        self._bp_pressed = False
        self._bp_drag_start: float | None = None
        self._sweep_on = False
        self._note = ""
        self._products: WaterfallProducts | None = None
        self._data_key_last: tuple[Any, ...] | None = None
        self._in_update = False
        self._ts_fill = None
        self._bp_fill = None
        self._zap_spans: list[Any] = []
        self._buttons: list[Any] = []
        self._button_kinds: list[str] = []
        self._file_buttons: list[Any] = []
        self._textboxes: list[Any] = []

        if self._gui:
            self._build_figure()
            self.update_view()

    @staticmethod
    def _load_mask(mask: str | Path | RFIMask | None) -> RFIMask | None:
        if mask is None:
            return None
        if isinstance(mask, (str, Path)):
            from sigpyproc.core.rfi import RFIMask

            return RFIMask.from_file(str(mask))
        return mask

    def _init_keymap(self) -> None:
        self.keymap.bind(["right", "l"], self.next_gulp, "Next gulp", "Navigation")
        self.keymap.bind(["left", "h"], self.prev_gulp, "Previous gulp", "Navigation")
        self.keymap.bind(
            ["shift+right", "L"],
            lambda: self.step_gulp(0.5),
            "Forward half a gulp",
            "Navigation",
        )
        self.keymap.bind(
            ["shift+left", "H"],
            lambda: self.step_gulp(-0.5),
            "Back half a gulp",
            "Navigation",
        )
        self.keymap.bind(["home"], self.goto_start, "Start of file", "Navigation")
        self.keymap.bind(["end"], self.goto_end, "End of file", "Navigation")
        self.keymap.bind(["g"], self.focus_start, "Edit start sample", "Navigation")
        self.keymap.bind(["backspace"], self.history_back, "Undo", "Navigation")
        self.keymap.bind(["N"], self.next_file, "Next file", "Navigation")
        self.keymap.bind(["P"], self.prev_file, "Previous file", "Navigation")

        self.keymap.bind(["up", "k"], lambda: self.step_dm(1.0), "DM +1", "DM")
        self.keymap.bind(["down", "j"], lambda: self.step_dm(-1.0), "DM -1", "DM")
        self.keymap.bind(["shift+up", "K"], lambda: self.step_dm(10.0), "DM +10", "DM")
        self.keymap.bind(
            ["shift+down", "J"], lambda: self.step_dm(-10.0), "DM -10", "DM"
        )
        self.keymap.bind(["d"], self.toggle_dm, "Toggle DM 0 / last", "DM")
        self.keymap.bind(["D"], self.start_dm_fit, "Arm two-click DM fit", "DM")
        self.keymap.bind(["w"], self.toggle_sweep, "DM sweep guide", "DM")

        self.keymap.bind(["c", "C"], self.cycle_cmap, "Cycle colormap", "Display")
        self.keymap.bind(["i"], self.toggle_invert, "Invert colormap", "Display")
        self.keymap.bind(["n"], self.cycle_scaler, "Cycle scaling", "Display")
        self.keymap.bind(["z"], self.toggle_zerodm, "Toggle zero-DM", "Display")
        self.keymap.bind(
            ["f"], self.toggle_flatten, "Toggle bandpass flatten", "Display"
        )
        self.keymap.bind(["T"], self.toggle_theme, "Toggle light/dark", "Display")
        self.keymap.bind(
            ["R"], self.toggle_hires, "Toggle screen / full-res", "Display"
        )

        self.keymap.bind(
            ["]"],
            lambda: self.step_time_downsample(larger=True),
            "Finer time scrunch",
            "Resolution",
        )
        self.keymap.bind(
            ["["],
            lambda: self.step_time_downsample(larger=False),
            "Coarser time scrunch",
            "Resolution",
        )
        self.keymap.bind(
            ["}"],
            lambda: self.step_freq_downsample(larger=True),
            "Finer freq scrunch",
            "Resolution",
        )
        self.keymap.bind(
            ["{"],
            lambda: self.step_freq_downsample(larger=False),
            "Coarser freq scrunch",
            "Resolution",
        )

        self.keymap.bind(["s"], self.save_png, "Save PNG", "Export")
        self.keymap.bind(["S"], self.save_npz, "Save block .npz", "Export")
        self.keymap.bind(["e"], self.export_fil, "Write block .fil", "Export")
        self.keymap.bind(["Z"], self._export_mask_key, "Export RFIMask .h5", "Export")
        self.keymap.bind(["?"], self.show_help, "Toggle help", "Help")
        self.keymap.bind(["q"], self.close, "Quit", "Help")

    def _build_figure(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, TextBox

        with viewer_context(self.state.theme):
            self.fig: Figure = plt.figure(figsize=(13.2, 8.8))

        gs = self.fig.add_gridspec(
            2,
            3,
            width_ratios=[4.4, 0.12, 1.25],
            height_ratios=[1.05, 4.0],
            left=0.07,
            right=0.985,
            top=0.93,
            bottom=0.24,
            hspace=0.14,
            wspace=0.08,
        )
        self.ax_ts = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1:])
        self.ax_im = self.fig.add_subplot(gs[1, 0], sharex=self.ax_ts)
        self.ax_cbar = self.fig.add_subplot(gs[1, 1])
        self.ax_bp = self.fig.add_subplot(gs[1, 2], sharey=self.ax_im)
        self.ax_info.axis("off")

        (self.line_ts,) = self.ax_ts.plot([], [], color="#1f77b4", lw=1.1)
        (self.line_bp,) = self.ax_bp.plot([], [], color="#1f77b4", lw=1.1)
        self.im_artist = self.ax_im.imshow(
            np.zeros((10, 10)),
            aspect="auto",
            origin="lower",
            cmap=self.state.cmap,
            interpolation="nearest",
        )
        self.cbar = self.fig.colorbar(self.im_artist, cax=self.ax_cbar)

        self.crosshair_v = self.ax_im.axvline(0, color="#ffe14a", lw=0.8, visible=False)
        self.crosshair_h = self.ax_im.axhline(0, color="#ffe14a", lw=0.8, visible=False)
        self.cursor_text = self.ax_im.text(
            0.01,
            0.02,
            "",
            transform=self.ax_im.transAxes,
            color="#ffe14a",
            fontsize=8,
            backgroundcolor="black",
            visible=False,
            zorder=7,
        )
        (self.sweep_line,) = self.ax_im.plot(
            [],
            [],
            color="#7dcea0",
            lw=1.1,
            visible=False,
        )
        self.fit_marks = self.ax_im.scatter(
            [],
            [],
            s=36,
            c="white",
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )
        (self.fit_curve,) = self.ax_im.plot(
            [],
            [],
            color="white",
            lw=1.15,
            ls="--",
            zorder=5,
        )
        self.help_text = self.ax_im.text(
            0.02,
            0.98,
            "",
            transform=self.ax_im.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            family="monospace",
            visible=False,
            zorder=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.94, "lw": 0.4},
        )
        self.info_text = self.ax_info.text(
            0.02,
            0.98,
            "",
            transform=self.ax_info.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
        )
        self.blit_manager = BlitManager(
            self.fig,
            [self.crosshair_v, self.crosshair_h, self.cursor_text, self.sweep_line],
            axes=self.ax_im,
        )

        self.ax_ts.set_ylabel("Power")
        self.ax_ts.tick_params(labelbottom=False)
        self.ax_im.set_xlabel("Time (s)")
        self.ax_im.set_ylabel("Frequency (MHz)")
        self.ax_bp.set_xlabel("Mean")
        self.ax_bp.tick_params(labelleft=False)
        self.fig.suptitle("sigpyproc viewer", fontsize=12, fontweight="bold")

        ax_slider = self.fig.add_axes((0.12, 0.175, 0.84, 0.028))
        n_total = max(1, self.fil.header.nsamples - 1)
        self.slider = Slider(
            ax_slider,
            "sample",
            0,
            n_total,
            valinit=min(self.state.start, n_total),
            valstep=max(1, self.state.nsamps),
            valfmt="%d",
        )
        self.slider.on_changed(self._on_slider)

        nav_axes = self._axes_row(0.112, 7, 0.035, 0.60)
        nav_specs = [
            ("Start", self.goto_start, "nav"),
            ("Prev", self.prev_gulp, "nav"),
            ("Next", self.next_gulp, "nav"),
            ("End", self.goto_end, "nav"),
            ("Fit DM", self.start_dm_fit, "dm"),
            ("File -", self.prev_file, "file"),
            ("File +", self.next_file, "file"),
        ]
        for ax, (label, callback, kind) in zip(nav_axes, nav_specs, strict=True):
            button = Button(ax, label, color=_LIGHT_BUTTONS[kind], hovercolor="#c5daf0")
            button.label.set_fontsize(8)
            button.on_clicked(lambda _ev, cb=callback: cb())
            self._buttons.append(button)
            self._button_kinds.append(kind)
            if kind == "file":
                self._file_buttons.append(button)

        box_axes = self._axes_row(0.112, 3, 0.63, 0.985)
        self.dm_box = TextBox(box_axes[0], "DM", initial=f"{self.state.dm:.2f}")
        self.gulp_box = TextBox(box_axes[1], "Gulp", initial=str(self.state.nsamps))
        self.start_box = TextBox(box_axes[2], "Start", initial=str(self.state.start))
        self.dm_box.on_submit(self._submit_dm)
        self.gulp_box.on_submit(self._submit_gulp)
        self.start_box.on_submit(self._submit_start)
        self._textboxes = [self.dm_box, self.gulp_box, self.start_box]

        disp_axes = self._axes_row(0.048, 8, 0.035, 0.985)
        disp_specs = [
            ("Colormap", self.cycle_cmap, "disp"),
            ("Scale", self.cycle_scaler, "disp"),
            ("Zero-DM", self.toggle_zerodm, "disp"),
            ("Flat", self.toggle_flatten, "disp"),
            ("Theme", self.toggle_theme, "disp"),
            ("Save PNG", self.save_png, "save"),
            ("Save mask", self._export_mask_key, "save"),
            ("Help", self.show_help, "disp"),
        ]
        for ax, (label, callback, kind) in zip(disp_axes, disp_specs, strict=True):
            button = Button(ax, label, color=_LIGHT_BUTTONS[kind], hovercolor="#c5daf0")
            button.label.set_fontsize(8)
            button.on_clicked(lambda _ev, cb=callback: cb())
            self._buttons.append(button)
            self._button_kinds.append(kind)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        self._sync_file_buttons()

    def _axes_row(
        self,
        bottom: float,
        n: int,
        left: float,
        right: float,
        height: float = 0.04,
        gap: float = 0.006,
    ) -> list[Any]:
        span = right - left
        width = (span - gap * (n - 1)) / n
        return [
            self.fig.add_axes((left + i * (width + gap), bottom, width, height))
            for i in range(n)
        ]

    def _screen_size(self) -> tuple[int, int]:
        pos = self.ax_im.get_position()
        dpi = float(self.fig.dpi)
        cols = int(pos.width * self.fig.get_figwidth() * dpi)
        rows = int(pos.height * self.fig.get_figheight() * dpi)
        return (
            int(np.clip(rows, 32, _SCREEN_CAP)),
            int(np.clip(cols, 32, _SCREEN_CAP)),
        )

    def _data_key(self, screen: tuple[int, int]) -> tuple[Any, ...]:
        st = self.state
        return (
            self.file_idx,
            st.start,
            st.nsamps,
            st.dm,
            st.tfactor,
            st.ffactor,
            st.scaler_idx,
            st.zerodm,
            st.flatten,
            st.apply_mask,
            st.lod_mode,
            st.hires,
            tuple(sorted(st.zapped_channels)),
            screen,
        )

    def _swap_poly(
        self, old: PolyCollection | None, new: PolyCollection
    ) -> PolyCollection:
        if old is not None:
            old.remove()
        return new

    def _draw_time(self, products: WaterfallProducts) -> None:
        y = np.asarray(products.ts, dtype=np.float32)
        x = np.asarray(products.ts_time)
        if y.size > _DISPLAY_POINTS:
            nbins = _DISPLAY_POINTS
            mins, maxs = kernels.minmax_envelope(y, 0, int(y.size), nbins)
            t_axis = np.linspace(float(x[0]), float(x[-1]), nbins)
            self.line_ts.set_visible(False)
            self._ts_fill = self._swap_poly(
                self._ts_fill,
                self.ax_ts.fill_between(
                    t_axis, mins, maxs, color="#1f77b4", alpha=0.85, lw=0
                ),
            )
            with np.errstate(all="ignore"):
                y0 = float(np.nanmin(mins))
                y1 = float(np.nanmax(maxs))
        else:
            if self._ts_fill is not None:
                self._ts_fill.set_visible(False)
            self.line_ts.set_visible(True)
            self.line_ts.set_data(x, y)
            finite = y[np.isfinite(y)]
            if finite.size == 0:
                y0, y1 = 0.0, 1.0
            else:
                y0 = float(np.min(finite))
                y1 = float(np.max(finite))
        self.ax_ts.set_xlim(products.extent[0], products.extent[1])
        if np.isfinite(y0) and np.isfinite(y1):
            pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
            self.ax_ts.set_ylim(y0 - pad, y1 + pad)

    def _draw_bandpass(self, products: WaterfallProducts) -> None:
        freqs = np.asarray(products.bp_freqs, dtype=np.float64)
        mean = np.asarray(products.bp_mean, dtype=np.float64)
        std = np.asarray(products.bp_std, dtype=np.float64)
        if freqs.size == 0:
            return
        order = np.argsort(freqs)
        freqs = freqs[order]
        mean = mean[order]
        std = np.where(np.isfinite(std[order]), std[order], 0.0)
        if mean.size > _DISPLAY_POINTS:
            nbins = 1000
            mean32 = mean.astype(np.float32)
            std32 = std.astype(np.float32)
            mins, maxs = kernels.minmax_envelope(mean32, 0, mean.size, nbins)
            _std_lo, smax = kernels.minmax_envelope(std32, 0, std.size, nbins)
            f_axis = np.linspace(float(freqs[0]), float(freqs[-1]), nbins)
            low = mins - np.where(np.isfinite(smax), smax, 0.0)
            high = maxs + np.where(np.isfinite(smax), smax, 0.0)
            self.line_bp.set_visible(False)
            self._bp_fill = self._swap_poly(
                self._bp_fill,
                self.ax_bp.fill_betweenx(
                    f_axis, low, high, color="#1f77b4", alpha=0.3, lw=0
                ),
            )
            valid = high[np.isfinite(high)]
            x0 = float(np.nanmin(low)) if np.isfinite(low).any() else 0.0
            x1 = float(np.max(valid)) if valid.size else 1.0
        else:
            self.line_bp.set_visible(True)
            self.line_bp.set_data(mean, freqs)
            low = mean - std
            high = mean + std
            self._bp_fill = self._swap_poly(
                self._bp_fill,
                self.ax_bp.fill_betweenx(
                    freqs, low, high, color="#1f77b4", alpha=0.25, lw=0
                ),
            )
            finite = mean[np.isfinite(mean)]
            if finite.size == 0:
                x0, x1 = 0.0, 1.0
            else:
                x0 = float(np.min(finite))
                x1 = float(np.max(finite))
        self.ax_bp.set_ylim(products.extent[2], products.extent[3])
        if np.isfinite(x0) and np.isfinite(x1):
            pad = 0.05 * (x1 - x0) if x1 > x0 else 1.0
            self.ax_bp.set_xlim(x0 - pad, x1 + pad)
        self._draw_zaps()

    def _draw_zaps(self) -> None:
        for span in self._zap_spans:
            span.remove()
        self._zap_spans = []
        zapped = sorted(ch for ch in self.state.zapped_channels if ch >= 0)
        if not zapped:
            return
        freqs = np.asarray(self.fil.header.chan_freqs)
        half = 0.5 * abs(float(self.fil.header.foff))
        run = [zapped[0]]
        spans: list[tuple[int, int]] = []
        for ch in zapped[1:]:
            if ch == run[-1] + 1:
                run.append(ch)
            else:
                spans.append((run[0], run[-1]))
                run = [ch]
        spans.append((run[0], run[-1]))
        for lo, hi in spans:
            if hi >= freqs.size:
                continue
            f0 = float(np.min(freqs[lo : hi + 1]) - half)
            f1 = float(np.max(freqs[lo : hi + 1]) + half)
            self._zap_spans.append(
                self.ax_bp.axhspan(f0, f1, color="#d62728", alpha=0.28, lw=0, zorder=0)
            )

    def _write_info(self, products: WaterfallProducts) -> None:
        st = self.state
        scaler_name = SCALERS[st.scaler_idx % len(SCALERS)].name
        res = "full-res" if st.hires else "screen"
        stats = products.stats
        lines = [
            f"{self.fil.header.basename}  ({self.file_idx + 1}/{len(self.file_list)})",
            (f"samples {st.start}-{st.start + st.nsamps} / {self.fil.header.nsamples}"),
            f"time {products.extent[0]:.3f}-{products.extent[1]:.3f} s",
            f"DM {st.dm:.2f}   scrunch t={st.tfactor} f={st.ffactor}  {res}",
            f"scale {scaler_name}   cmap {st.cmap}",
            f"mean {stats['mean']:.4g}   std {stats['std']:.4g}",
            f"min {stats['min']:.4g}   max {stats['max']:.4g}",
            f"peak {stats['ts_max']:.4g}   zapped {len(st.zapped_channels)}",
        ]
        if self._dm_fit_armed:
            lines.append("DM fit: click two points on the sweep")
        if self._note:
            lines.append(self._note)
        self.info_text.set_text("\n".join(lines))
        self.fig.suptitle(
            f"sigpyproc viewer — {self.fil.header.basename}",
            fontsize=12,
            fontweight="bold",
        )

    def _apply_widget_colors(self) -> None:
        dark = self.state.theme == "dark"
        palette = _DARK_BUTTONS if dark else _LIGHT_BUTTONS
        label = "#f2f2f2" if dark else "#1a1a1a"
        hover = "#3d4f6a" if dark else "#c5daf0"
        for button, kind in zip(self._buttons, self._button_kinds, strict=True):
            color = palette[kind]
            button.color = color
            button.hovercolor = hover
            button.ax.set_facecolor(color)
            button.label.set_color(label)
        box_face = "#2a2a32" if dark else "#ffffff"
        box_ink = "#f2f2f2" if dark else "#1a1a1a"
        for box in self._textboxes:
            box.color = box_face
            box.ax.set_facecolor(box_face)
            box.text_disp.set_color(box_ink)
            box.label.set_color(label)
        if dark:
            self.slider.track.set_facecolor("#3a3a3a")
            self.slider.label.set_color(label)
            self.slider.valtext.set_color(label)
        else:
            self.slider.track.set_facecolor("#d9d9d9")
            self.slider.label.set_color("#1a1a1a")
            self.slider.valtext.set_color("#1a1a1a")
        help_face = "#202020" if dark else "#ffffff"
        self.help_text.set_color(label)
        self.help_text.set_bbox(
            {"boxstyle": "round", "facecolor": help_face, "alpha": 0.94, "lw": 0.4}
        )

    def _sync_widgets(self) -> None:
        self._in_update = True
        try:
            n_total = max(1, int(self.fil.header.nsamples) - 1)
            self.slider.valmin = 0
            self.slider.valmax = n_total
            self.slider.valstep = max(1, int(self.state.nsamps))
            self.slider.ax.set_xlim(0, n_total)
            self.slider.eventson = False
            # valstep snaps to gulps. Drop it so a half-gulp step stays put.
            stepped = self.slider.valstep
            self.slider.valstep = None
            self.slider.set_val(min(self.state.start, n_total))
            self.slider.valstep = stepped
            self.slider.eventson = True
            values = {
                self.dm_box: f"{self.state.dm:.2f}",
                self.gulp_box: str(self.state.nsamps),
                self.start_box: str(self.state.start),
            }
            for box, text in values.items():
                box.eventson = False
                box.set_val(text)
                box.eventson = True
            self._sync_file_buttons()
        finally:
            self._in_update = False

    def _sync_file_buttons(self) -> None:
        show = len(self.file_list) > 1
        for button in self._file_buttons:
            button.ax.set_visible(show)

    def _plot_axes(self) -> list[Any]:
        return [self.ax_ts, self.ax_im, self.ax_bp, self.ax_info, self.ax_cbar]

    def update_view(self) -> None:
        """Refresh artists from the current state."""
        self.state = self.model.clamp(self.state)
        if not self._gui:
            return
        screen = (0, 0) if self.state.hires else self._screen_size()
        key = self._data_key(screen)
        if self._products is None or key != self._data_key_last:
            self._products = self.model.products(
                self.state,
                screen_size=screen if screen != (0, 0) else (300, 600),
            )
            self._data_key_last = key
            fresh = True
        else:
            fresh = False
        products = self._products

        if fresh:
            self._draw_time(products)
            self.im_artist.set_data(products.image)
            self.im_artist.set_extent(products.extent)
            self.im_artist.set_clim(products.clim)
            self._draw_bandpass(products)
        self.im_artist.set_cmap(self.state.cmap)
        self.cbar.update_normal(self.im_artist)
        self._write_info(products)
        self._apply_widget_colors()
        paint_theme(
            self.fig,
            self._plot_axes(),
            self.state.theme,
            texts=[self.info_text],
        )
        self._sync_widgets()
        self.fig.canvas.draw_idle()

    def _on_resize(self, _event: object) -> None:
        if self._in_update or self._products is None:
            return
        self.update_view()

    def on_key_press(self, event: KeyEvent) -> None:
        """Dispatch a key, unless a text box is capturing it."""
        if any(getattr(box, "capturekeystrokes", False) for box in self._textboxes):
            if event.key == "escape":
                for box in self._textboxes:
                    if getattr(box, "capturekeystrokes", False):
                        box.stop_typing()
            return
        self.keymap.handle(event)

    def on_mouse_move(self, event: MouseEvent) -> None:
        """Update the crosshair, cursor value, and optional DM sweep."""
        if (
            event.inaxes != self.ax_im
            or event.xdata is None
            or event.ydata is None
            or self._products is None
        ):
            if self.crosshair_v.get_visible() or self.sweep_line.get_visible():
                self.crosshair_v.set_visible(False)
                self.crosshair_h.set_visible(False)
                self.cursor_text.set_visible(False)
                self.sweep_line.set_visible(False)
                self.blit_manager.update()
            return

        t = float(event.xdata)
        f = float(event.ydata)
        self.crosshair_v.set_xdata([t, t])
        self.crosshair_h.set_ydata([f, f])
        self.crosshair_v.set_visible(True)
        self.crosshair_h.set_visible(True)
        value = self._sample_value(t, f)
        mjd = self.fil.header.tstart + (t / 86400.0)
        text = f"t={t:.3f}s  MJD={mjd:.6f}  f={f:.2f} MHz"
        if value is not None:
            text = f"{text}  v={value:.3g}"
        self.cursor_text.set_text(text)
        self.cursor_text.set_visible(True)
        if self._sweep_on:
            self._update_sweep(t, f)
        else:
            self.sweep_line.set_visible(False)
        self.blit_manager.update()

    def _sample_value(self, t: float, f: float) -> float | None:
        if self._products is None:
            return None
        image = self._products.image
        t0, t1, f0, f1 = self._products.extent
        if t1 == t0 or f1 == f0 or image.size == 0:
            return None
        col = int((t - t0) / (t1 - t0) * image.shape[1])
        row = int((f - f0) / (f1 - f0) * image.shape[0])
        if not (0 <= row < image.shape[0] and 0 <= col < image.shape[1]):
            return None
        value = float(image[row, col])
        if not np.isfinite(value):
            return None
        return value

    def _update_sweep(self, t: float, f: float) -> None:
        if self._products is None or f <= 0:
            self.sweep_line.set_visible(False)
            return
        f_lo, f_hi = self._products.extent[2], self._products.extent[3]
        freqs = np.linspace(f_lo, f_hi, 256)
        delay = self.state.dm * DM_CONSTANT_LK * (freqs**-2 - f**-2)
        self.sweep_line.set_data(t + delay, freqs)
        self.sweep_line.set_visible(True)

    def on_button_press(self, event: MouseEvent) -> None:
        """Arm-gated DM fit clicks, or the start of a bandpass zap drag."""
        if (
            self._dm_fit_armed
            and event.inaxes == self.ax_im
            and event.xdata is not None
            and event.ydata is not None
            and len(self._fit_points) < 2
        ):
            self._fit_points.append((float(event.xdata), float(event.ydata)))
            self.fit_marks.set_offsets(np.asarray(self._fit_points, dtype=float))
            if len(self._fit_points) == 2:
                self._finish_dm_fit()
            else:
                self.fig.canvas.draw_idle()
            return

        if event.inaxes == self.ax_bp and event.ydata is not None:
            self._bp_pressed = True
            self._bp_drag_start = float(event.ydata)

    def _finish_dm_fit(self) -> None:
        p1, p2 = self._fit_points
        residual = WaterfallModel.fit_dm(p1[0], p1[1], p2[0], p2[1])
        new_dm = max(0.0, float(np.round(self.state.dm + residual, 2)))
        if self._products is not None:
            f_lo, f_hi = self._products.extent[2], self._products.extent[3]
            freqs = np.linspace(f_lo, f_hi, 256)
            times = p1[0] + residual * DM_CONSTANT_LK * (freqs**-2 - p1[1] ** -2)
            self.fit_curve.set_data(times, freqs)
        self._note = f"DM fit {residual:+.2f} → {new_dm:.2f}"
        print(self._note)
        self._dm_fit_armed = False
        self.history.push(self.state)
        self.state.dm = new_dm
        self.update_view()

    def on_button_release(self, event: MouseEvent) -> None:
        """Zap or restore every original channel in the dragged band."""
        if not self._bp_pressed or self._bp_drag_start is None:
            self._bp_pressed = False
            return
        self._bp_pressed = False
        if event.inaxes != self.ax_bp or event.ydata is None:
            return
        f_lo, f_hi = sorted((self._bp_drag_start, float(event.ydata)))
        freqs = np.asarray(self.fil.header.chan_freqs)
        chans = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
        if chans.size == 0:
            chans = np.array([int(np.argmin(np.abs(freqs - f_lo)))], dtype=int)
        selected = [int(ch) for ch in chans]
        if all(ch in self.state.zapped_channels for ch in selected):
            for ch in selected:
                self.state.zapped_channels.discard(ch)
        else:
            self.state.zapped_channels.update(selected)
        self.update_view()

    def _clear_fit(self) -> None:
        self._fit_points.clear()
        self._dm_fit_armed = False
        if not self._gui:
            return
        self.fit_marks.set_offsets(np.empty((0, 2)))
        self.fit_curve.set_data([], [])

    def next_gulp(self) -> None:
        """Page forward one gulp."""
        self._clear_fit()
        self._note = ""
        self.history.push(self.state)
        self.state.start += self.state.nsamps
        self.update_view()

    def prev_gulp(self) -> None:
        """Page back one gulp."""
        self._clear_fit()
        self._note = ""
        self.history.push(self.state)
        self.state.start = max(0, self.state.start - self.state.nsamps)
        self.update_view()

    def step_gulp(self, factor: float) -> None:
        """Move by ``factor`` times the current gulp."""
        self._clear_fit()
        self.history.push(self.state)
        step = int(self.state.nsamps * factor)
        self.state.start = max(0, self.state.start + step)
        self.update_view()

    def goto_start(self) -> None:
        """Jump to the first sample."""
        self._clear_fit()
        self.history.push(self.state)
        self.state.start = 0
        self.update_view()

    def goto_end(self) -> None:
        """Jump so the last gulp fills the window."""
        self._clear_fit()
        self.history.push(self.state)
        self.state.start = max(0, self.fil.header.nsamples - self.state.nsamps)
        self.update_view()

    def focus_start(self) -> None:
        """Type a new start sample. A trailing ``s`` means seconds."""
        self.start_box.begin_typing()

    def history_back(self) -> None:
        """Restore the previous viewer state."""
        prev = self.history.back(self.state)
        if prev is not None:
            self._clear_fit()
            self.state = prev
            self.update_view()

    def step_dm(self, delta: float) -> None:
        """Add ``delta`` to the DM, floored at zero."""
        self._clear_fit()
        self.history.push(self.state)
        self.state.dm = float(np.round(max(0.0, self.state.dm + delta), 2))
        self.update_view()

    def toggle_dm(self) -> None:
        """Swap between DM 0 and the last non-zero DM."""
        self._clear_fit()
        self.history.push(self.state)
        if self.state.dm != 0.0:
            self._last_dm = self.state.dm
            self.state.dm = 0.0
        else:
            self.state.dm = getattr(self, "_last_dm", 10.0)
        self.update_view()

    def start_dm_fit(self) -> None:
        """Arm the next two waterfall clicks as a DM measurement."""
        self._clear_fit()
        self._dm_fit_armed = True
        self._note = ""
        print("Two-click DM fit: click two points on the dispersed sweep.")
        self.update_view()

    def toggle_sweep(self) -> None:
        """Show or hide the dispersion-delay curve under the cursor."""
        self._sweep_on = not self._sweep_on
        self.sweep_line.set_visible(False)
        self._note = "DM sweep guide on" if self._sweep_on else ""
        if self._products is not None:
            self._write_info(self._products)
        self.fig.canvas.draw_idle()

    def cycle_cmap(self) -> None:
        """Advance to the next colormap in the curated list."""
        name = self.state.cmap
        if name in _CMAPS:
            idx = _CMAPS.index(name)
        elif name.endswith("_r") and name[:-2] in _CMAPS:
            idx = _CMAPS.index(name[:-2])
        else:
            idx = -1
        self.state.cmap = _CMAPS[(idx + 1) % len(_CMAPS)]
        self.update_view()

    def toggle_invert(self) -> None:
        """Append or remove a reversed-colormap suffix."""
        name = self.state.cmap
        self.state.cmap = name[:-2] if name.endswith("_r") else f"{name}_r"
        self.update_view()

    def cycle_scaler(self) -> None:
        """Advance to the next contrast scaler."""
        self.state.scaler_idx = (self.state.scaler_idx + 1) % len(SCALERS)
        self.update_view()

    def toggle_zerodm(self) -> None:
        """Subtract the per-sample channel mean."""
        self.state.zerodm = not self.state.zerodm
        self.update_view()

    def toggle_flatten(self) -> None:
        """Divide each channel by its median."""
        self.state.flatten = not self.state.flatten
        self.update_view()

    def toggle_theme(self) -> None:
        """Switch the open window between the light and dark palettes."""
        self.state.theme = "dark" if self.state.theme == "light" else "light"
        self.update_view()

    def toggle_hires(self) -> None:
        """Bypass screen decimation and send the full block to the image."""
        self.state.hires = not self.state.hires
        self.update_view()

    def step_time_downsample(self, *, larger: bool) -> None:
        """Step the time scrunch through divisors of the gulp."""
        self.state.tfactor = step_divisor(
            self.state.nsamps, self.state.tfactor, larger=larger
        )
        self.update_view()

    def step_freq_downsample(self, *, larger: bool) -> None:
        """Step the frequency scrunch through divisors of the channel count."""
        self.state.ffactor = step_divisor(
            self.fil.header.nchans, self.state.ffactor, larger=larger
        )
        self.update_view()

    def _open_index(self, idx: int) -> None:
        from sigpyproc.readers import open_filterbank

        if self._owns_reader:
            self.fil.close()
        self.file_idx = idx % len(self.file_list)
        self.fil = open_filterbank(self.file_list[self.file_idx])
        self._owns_reader = True
        self.model = WaterfallModel(self.fil, mask=self.rfimask)
        self.state.start = 0
        self.state.zapped_channels = set()
        self._products = None
        self._data_key_last = None
        self._clear_fit()
        self._note = ""
        self.update_view()

    def next_file(self) -> None:
        """Open the next file passed on the command line."""
        if len(self.file_list) <= 1:
            return
        self._open_index(self.file_idx + 1)

    def prev_file(self) -> None:
        """Open the previous file passed on the command line."""
        if len(self.file_list) <= 1:
            return
        self._open_index(self.file_idx - 1)

    def _on_slider(self, val: float) -> None:
        if self._in_update:
            return
        sample = int(val)
        if sample == self.state.start:
            return
        self._clear_fit()
        self.state.start = sample
        self.update_view()

    def _submit_dm(self, text: str) -> None:
        if self._in_update:
            return
        try:
            dm = float(text)
        except ValueError:
            self._sync_widgets()
            return
        self._clear_fit()
        self.history.push(self.state)
        self.state.dm = max(0.0, dm)
        self.update_view()

    def _submit_gulp(self, text: str) -> None:
        if self._in_update:
            return
        try:
            gulp = int(float(text))
        except ValueError:
            self._sync_widgets()
            return
        self.history.push(self.state)
        self.state.nsamps = max(1, gulp)
        self.update_view()

    def _submit_start(self, text: str) -> None:
        if self._in_update:
            return
        raw = text.strip().lower()
        try:
            if raw.endswith("s"):
                sample = round(float(raw[:-1]) / self.fil.header.tsamp)
            else:
                sample = int(float(raw))
        except ValueError:
            self._sync_widgets()
            return
        self._clear_fit()
        self.history.push(self.state)
        self.state.start = max(0, sample)
        self.update_view()

    def save_png(self) -> None:
        """Save the current window to PNG."""
        out = f"{self.fil.header.basename}_t{self.state.start}_dm{self.state.dm}.png"
        self.fig.savefig(out, dpi=150)
        print(f"Saved {out}")

    def save_npz(self, filename: str | None = None) -> str:
        """Save the current processed block to a pickle-free ``.npz``.

        The arrays are the block that is on screen, including its epoch,
        sampling interval, frequency grid, and DM. They are not the original
        file header.
        """
        blk, _ = self.model.get_block(self.state)
        out = filename or (
            f"{self.fil.header.basename}_t{self.state.start}_dm{self.state.dm}.npz"
        )
        hdr = blk.header
        np.savez(
            out,
            data=np.asarray(blk.data, dtype=np.float32),
            tstart=np.float64(hdr.tstart),
            tsamp=np.float64(hdr.tsamp),
            fch1=np.float64(hdr.fch1),
            foff=np.float64(hdr.foff),
            nchans=np.int64(hdr.nchans),
            nsamples=np.int64(blk.data.shape[1]),
            dm=np.float64(blk.dm),
            source=np.asarray(hdr.source),
        )
        if filename is None:
            print(f"Saved {out}")
        return out

    def export_fil(self) -> None:
        """Write the current processed block as a 32-bit filterbank."""
        blk, _ = self.model.get_block(self.state)
        out = (
            f"{self.fil.header.basename}_t{self.state.start}_dm{self.state.dm:.2f}.fil"
        )
        path = blk.to_file(out)
        print(f"Wrote {path}")

    def _export_mask_key(self) -> None:
        """Export the session mask from the keyboard or the Save mask button."""
        self.export_rfimask()

    def export_rfimask(self, filename: str | None = None) -> str:
        """Write session zaps unioned with the loaded mask.

        Channel mean and variance come from the current dedispersed gulp.
        Higher moments are left at zero.
        """
        from sigpyproc.core.rfi import RFIMask

        nchans = self.fil.header.nchans
        mask_arr = np.zeros(nchans, dtype=bool)
        if self.rfimask is not None and len(self.rfimask.chan_mask) == nchans:
            mask_arr |= np.asarray(self.rfimask.chan_mask, dtype=bool)
        for ch in self.state.zapped_channels:
            if 0 <= ch < nchans:
                mask_arr[ch] = True

        mean, var, maxima, minima = self.model.channel_stats(self.state)
        zeros = np.zeros(nchans, dtype=np.float64)

        def _clean(values: np.ndarray) -> np.ndarray:
            out = np.asarray(values, dtype=np.float64)
            return np.where(np.isfinite(out), out, 0.0)

        mask_obj = RFIMask(
            threshold=3.0,
            header=self.fil.header,
            chan_mean=_clean(mean),
            chan_var=_clean(var),
            chan_skew=zeros,
            chan_kurt=zeros,
            chan_maxima=_clean(maxima),
            chan_minima=_clean(minima),
            chan_mask=mask_arr,
        )
        out = filename or f"{self.fil.header.basename}_mask.h5"
        path = mask_obj.to_file(out)
        print(f"Exported RFIMask to {path}")
        return path

    def show_help(self) -> None:
        """Toggle the on-figure shortcut list and print it."""
        text = self.keymap.format_help()
        print(text)
        visible = not self.help_text.get_visible()
        self.help_text.set_text(text)
        self.help_text.set_visible(visible)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Enter the GUI event loop."""
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        """Close the window and any file this viewer opened."""
        if self._owns_reader:
            self.fil.close()
            self._owns_reader = False
        if not self._gui:
            return
        import matplotlib.pyplot as plt

        plt.close(self.fig)
