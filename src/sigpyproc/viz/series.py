"""Interactive Matplotlib TimeSeries Viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.widgets import SpanSelector

from sigpyproc import kernels
from sigpyproc.viz._backend import (
    ensure_interactive_backend,
    use_headless_backend,
    viewer_context,
)
from sigpyproc.viz.keymap import BlitManager, KeyMap

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure

    from sigpyproc.fourierseries import PowerSpectrum
    from sigpyproc.timeseries import TimeSeries


class SeriesViewer:
    """Interactive overview + detail viewer for 1D TimeSeries."""

    def __init__(
        self,
        tim: TimeSeries,
        *,
        theme: str = "light",
        batch: bool = False,
        gui: bool | None = None,
    ) -> None:
        if batch:
            use_headless_backend()
        else:
            ensure_interactive_backend()
        self._gui = (not batch) if gui is None else gui

        self.tim = tim
        self.theme = theme
        self.keymap = KeyMap()
        self._init_keymap()

        self.start_samp = 0
        self.end_samp = min(len(self.tim.data), 16384)
        self.total_samps = len(self.tim.data)

        if self._gui:
            self._build_figure()
            self.update_detail()

    def _init_keymap(self) -> None:
        self.keymap.bind(["right", "l"], self.pan_right, "Pan right", "Navigation")
        self.keymap.bind(["left", "h"], self.pan_left, "Pan left", "Navigation")
        self.keymap.bind(["+", "="], self.zoom_in, "Zoom in", "Zoom")
        self.keymap.bind(["-", "_"], self.zoom_out, "Zoom out", "Zoom")
        self.keymap.bind(["0"], self.reset_view, "Reset view", "Zoom")
        self.keymap.bind(["s"], self.save_png, "Save PNG", "Export")
        self.keymap.bind(
            ["?"], lambda: print(self.keymap.format_help()), "Help", "Help"
        )
        self.keymap.bind(["q"], self.close, "Quit", "Help")

    def _build_figure(self) -> None:
        import matplotlib.pyplot as plt

        with viewer_context(self.theme):
            self.fig: Figure = plt.figure(figsize=(12, 6))

        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.25)
        self.ax_overview = self.fig.add_subplot(gs[0])
        self.ax_detail = self.fig.add_subplot(gs[1])

        # Overview envelope
        nbins = 1000
        mins, maxs = kernels.minmax_envelope(self.tim.data, 0, self.total_samps, nbins)
        t_over = np.linspace(0, self.total_samps * self.tim.header.tsamp, nbins)
        self.ax_overview.fill_between(t_over, mins, maxs, color="#7f7f7f", alpha=0.6)
        self.ax_overview.set_title(f"Overview: {self.tim.header.basename}", fontsize=10)
        self.ax_overview.set_ylabel("Power")
        self.ax_overview.tick_params(labelbottom=False)

        # Detail line. The envelope artist is created on demand and never
        # replaces this line, so zooming back in still has an artist to update.
        (self.line_detail,) = self.ax_detail.plot([], [], color="#1f77b4", lw=1.0)
        self._detail_fill = None
        self.ax_detail.set_xlabel("Time (s)")
        self.ax_detail.set_ylabel("Power")

        # Span selector on overview
        self.span = SpanSelector(
            self.ax_overview,
            self.on_span_select,
            "horizontal",
            useblit=True,
            props={"alpha": 0.3, "facecolor": "red"},
            interactive=True,
            drag_from_anywhere=True,
        )

        self.fig.canvas.mpl_connect("key_press_event", self.keymap.handle)

    def on_span_select(self, xmin: float, xmax: float) -> None:
        s0 = max(0, int(xmin / self.tim.header.tsamp))
        s1 = min(self.total_samps, int(xmax / self.tim.header.tsamp))
        if s1 - s0 > 8:
            self.start_samp = s0
            self.end_samp = s1
            self.update_detail()

    def update_detail(self) -> None:
        if not self._gui:
            return
        nsamps = self.end_samp - self.start_samp
        if nsamps < 2:
            return
        t0 = self.start_samp * self.tim.header.tsamp
        t1 = self.end_samp * self.tim.header.tsamp

        if nsamps > 4000:
            nbins = 2000
            mins, maxs = kernels.minmax_envelope(
                self.tim.data, self.start_samp, self.end_samp, nbins
            )
            t_axis = np.linspace(t0, t1, nbins)
            self.line_detail.set_visible(False)
            if self._detail_fill is not None:
                self._detail_fill.remove()
            self._detail_fill = self.ax_detail.fill_between(
                t_axis, mins, maxs, color="#1f77b4", alpha=0.85, lw=0
            )
            self.ax_detail.set_xlim(t0, t1)
            finite = np.isfinite(mins)
            if finite.any():
                self.ax_detail.set_ylim(
                    float(np.min(mins[finite])),
                    float(np.max(maxs[finite])) * 1.05,
                )
        else:
            if self._detail_fill is not None:
                self._detail_fill.set_visible(False)
            sub = self.tim.data[self.start_samp : self.end_samp]
            t_axis = np.linspace(t0, t1, nsamps, endpoint=False)
            self.line_detail.set_visible(True)
            self.line_detail.set_data(t_axis, sub)
            self.ax_detail.set_xlim(t0, t1)
            valid = sub[np.isfinite(sub)]
            if valid.size > 0:
                self.ax_detail.set_ylim(
                    float(np.min(valid)),
                    float(np.max(valid)) * 1.05,
                )

        self.fig.canvas.draw_idle()

    def pan_right(self) -> None:
        step = (self.end_samp - self.start_samp) // 2
        if self.end_samp + step <= self.total_samps:
            self.start_samp += step
            self.end_samp += step
            self.update_detail()

    def pan_left(self) -> None:
        step = (self.end_samp - self.start_samp) // 2
        if self.start_samp - step >= 0:
            self.start_samp -= step
            self.end_samp -= step
            self.update_detail()

    def zoom_in(self) -> None:
        mid = (self.start_samp + self.end_samp) // 2
        span = max(8, (self.end_samp - self.start_samp) // 4)
        self.start_samp = max(0, mid - span)
        self.end_samp = min(self.total_samps, mid + span)
        self.update_detail()

    def zoom_out(self) -> None:
        mid = (self.start_samp + self.end_samp) // 2
        span = self.end_samp - self.start_samp
        self.start_samp = max(0, mid - span)
        self.end_samp = min(self.total_samps, mid + span)
        self.update_detail()

    def reset_view(self) -> None:
        self.start_samp = 0
        self.end_samp = min(self.total_samps, 16384)
        self.update_detail()

    def save_png(self, filename: str | None = None) -> str:
        """Write the current window to PNG and return the path.

        Batch viewers have no widget figure. They draw this window with Agg.
        """
        out = filename or f"{self.tim.header.basename}_timeseries.png"
        if self._gui:
            self.fig.savefig(out, dpi=150)
        else:
            _save_series_png(self, out)
        if filename is None:
            print(f"Saved {out}")
        return out

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        if not self._gui:
            return
        import matplotlib.pyplot as plt

        plt.close(self.fig)


class SpectrumViewer:
    """Interactive PowerSpectrum viewer with harmonic markers and peak detection."""

    def __init__(
        self,
        spec: PowerSpectrum,
        *,
        theme: str = "light",
        batch: bool = False,
        gui: bool | None = None,
    ) -> None:
        if batch:
            use_headless_backend()
        else:
            ensure_interactive_backend()
        self._gui = (not batch) if gui is None else gui

        self.spec = spec
        self.theme = theme
        self.keymap = KeyMap()
        self._init_keymap()

        self.data = np.asarray(self.spec.data, dtype=np.float32)
        self.total_bins = len(self.data)
        # Bin width is the inverse of the real transform length, not twice
        # the number of stored rfft bins (that is one bin too long).
        length = self.spec.header.nsamples * self.spec.header.tsamp
        self.df = 1.0 / length if length else np.nan
        self.freqs = np.linspace(
            0, self.total_bins * self.df, self.total_bins, endpoint=False
        )

        self.logy = False
        if self._gui:
            self._build_figure()
            self.update_plot()

    def _init_keymap(self) -> None:
        self.keymap.bind(["l"], self.toggle_logy, "Toggle Log-Y", "Display")
        self.keymap.bind(["s"], self.save_png, "Save PNG", "Export")
        self.keymap.bind(
            ["?"], lambda: print(self.keymap.format_help()), "Help", "Help"
        )
        self.keymap.bind(["q"], self.close, "Quit", "Help")

    def _build_figure(self) -> None:
        import matplotlib.pyplot as plt

        with viewer_context(self.theme):
            self.fig: Figure = plt.figure(figsize=(12, 6))

        self.ax = self.fig.add_subplot(1, 1, 1)
        (self.line,) = self.ax.plot([], [], color="#d62728", lw=0.9)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Power")
        self.ax.set_title(f"Power Spectrum: {self.spec.header.basename}", fontsize=11)
        self._spec_fill = None

        # Harmonics overlay lines
        self.marker_lines = [
            self.ax.axvline(0, color="blue", ls="--", lw=0.8, visible=False)
            for _ in range(8)
        ]
        self.blit_manager = BlitManager(self.fig, self.marker_lines)

        self.fig.canvas.mpl_connect("key_press_event", self.keymap.handle)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event: MouseEvent) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return

        f0 = float(event.xdata)
        for i, line in enumerate(self.marker_lines):
            fn = f0 * (i + 1)
            line.set_xdata([fn, fn])
            line.set_visible(True)

        self.blit_manager.update()

    def update_plot(self) -> None:
        """Draw the spectrum, using a min/max envelope above 10000 bins."""
        if not self._gui:
            return
        if self.total_bins > 10000:
            nbins = 4000
            mins, maxs = kernels.minmax_envelope(self.data, 0, self.total_bins, nbins)
            f_axis = np.linspace(0, float(np.max(self.freqs)), nbins)
            self.line.set_visible(False)
            if self._spec_fill is not None:
                self._spec_fill.remove()
            self._spec_fill = self.ax.fill_between(
                f_axis, mins, maxs, color="#d62728", alpha=0.7, lw=0
            )
            self.ax.set_xlim(float(f_axis[0]), float(f_axis[-1]))
            finite = np.isfinite(mins)
            if finite.any():
                self.ax.set_ylim(
                    float(np.min(mins[finite])),
                    float(np.max(maxs[finite])) * 1.05,
                )
        else:
            if self._spec_fill is not None:
                self._spec_fill.set_visible(False)
            self.line.set_visible(True)
            self.line.set_data(self.freqs, self.data)
            self.ax.set_xlim(float(self.freqs[0]), float(self.freqs[-1]))
            finite = self.data[np.isfinite(self.data)]
            if finite.size > 0:
                self.ax.set_ylim(float(np.min(finite)), float(np.max(finite)) * 1.05)

        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Power")
        self.ax.set_yscale("log" if self.logy else "linear")
        self.fig.canvas.draw_idle()

    def toggle_logy(self) -> None:
        self.logy = not self.logy
        if not self._gui:
            return
        self.ax.set_yscale("log" if self.logy else "linear")
        self.fig.canvas.draw_idle()

    def save_png(self, filename: str | None = None) -> str:
        """Write the spectrum to PNG and return the path."""
        out = filename or f"{self.spec.header.basename}_spec.png"
        if self._gui:
            self.fig.savefig(out, dpi=150)
        else:
            _save_spectrum_png(self, out)
        if filename is None:
            print(f"Saved {out}")
        return out

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        if not self._gui:
            return
        import matplotlib.pyplot as plt

        plt.close(self.fig)


def _save_series_png(viewer: SeriesViewer, out: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from sigpyproc.viz._backend import paint_theme

    fig = Figure(figsize=(10, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    start = max(0, viewer.start_samp)
    stop = viewer.end_samp if viewer.end_samp > start else viewer.total_samps
    y = np.asarray(viewer.tim.data[start:stop])
    t = (start + np.arange(y.size)) * viewer.tim.header.tsamp
    ax.plot(t, y, color="#1f77b4", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power")
    ax.set_title(viewer.tim.header.basename)
    paint_theme(fig, [ax], viewer.theme)
    fig.savefig(out, dpi=120)


def _save_spectrum_png(viewer: SpectrumViewer, out: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from sigpyproc.viz._backend import paint_theme

    fig = Figure(figsize=(10, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(viewer.freqs, viewer.data, color="#d62728", lw=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(viewer.spec.header.basename)
    paint_theme(fig, [ax], viewer.theme)
    fig.savefig(out, dpi=120)
