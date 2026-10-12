"""Backend management and Matplotlib configuration for sigpyproc viewers."""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Generator


class ViewerError(RuntimeError):
    """Exception raised when interactive viewer cannot initialize or display."""


def ensure_interactive_backend(preferred: str | None = None) -> str:
    """Ensure a valid interactive backend is active, or raise ViewerError.

    Parameters
    ----------
    preferred : str | None, optional
        Preferred backend name, by default None.

    Returns
    -------
    str
        Active matplotlib backend name.

    Raises
    ------
    ViewerError
        If no display or interactive backend is available.
    """
    if preferred is not None:
        try:
            mpl.use(preferred, force=True)
            return mpl.get_backend()
        except (ValueError, ImportError) as e:
            msg = f"Could not set preferred backend '{preferred}': {e}"
            raise ViewerError(msg) from e

    current = mpl.get_backend().lower()
    if current not in ("agg", "pdf", "svg", "ps", "cairo"):
        return mpl.get_backend()

    has_display = bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or (
            os.uname().sysname == "Darwin"
            and os.environ.get("TERM_PROGRAM") != "Apple_Terminal_Headless"
        )
    )

    candidates = ["macosx", "TkAgg", "QtAgg", "Qt5Agg"]
    for candidate in candidates:
        with suppress(ValueError, ImportError):
            mpl.use(candidate, force=True)
            return mpl.get_backend()

    if not has_display:
        msg = (
            "No display detected (DISPLAY/WAYLAND_DISPLAY not set). "
            "Use --batch for non-interactive rendering."
        )
        raise ViewerError(msg)

    msg = (
        "Could not initialize an interactive matplotlib GUI backend. "
        "Ensure Tkinter or PyQt is available, or use --batch."
    )
    raise ViewerError(msg)


def viewer_rc(theme: str = "light") -> dict[str, object]:
    """Get customized rcParams dict for sigpyproc interactive viewers.

    Parameters
    ----------
    theme : {"light", "dark"}, optional
        Color theme name, by default "light".

    Returns
    -------
    dict[str, object]
        Dictionary of rcParams settings.
    """
    rc: dict[str, object] = {
        "image.interpolation": "nearest",
        "image.resample": False,
        "path.simplify": True,
        "path.simplify_threshold": 1.0,
        "agg.path.chunksize": 10000,
        "toolbar": "toolbar2",
        "keymap.back": [],
        "keymap.forward": [],
        "keymap.pan": [],
        "keymap.zoom": [],
        "keymap.save": [],
        "keymap.grid": [],
        "keymap.grid_minor": [],
        "keymap.home": [],
        "keymap.xscale": [],
        "keymap.yscale": [],
        "keymap.copy": [],
        "keymap.fullscreen": ["ctrl+f"],
        "keymap.quit": ["ctrl+w", "cmd+w"],
    }

    if theme == "dark":
        rc.update(
            {
                "figure.facecolor": "#181818",
                "figure.edgecolor": "#181818",
                "axes.facecolor": "#202020",
                "axes.edgecolor": "#555555",
                "axes.labelcolor": "#dddddd",
                "xtick.color": "#cccccc",
                "ytick.color": "#cccccc",
                "text.color": "#eeeeee",
                "grid.color": "#333333",
            }
        )
    else:
        rc.update(
            {
                "figure.facecolor": "#f8f9fa",
                "figure.edgecolor": "#f8f9fa",
                "axes.facecolor": "#ffffff",
                "axes.edgecolor": "#333333",
                "axes.labelcolor": "#222222",
                "xtick.color": "#222222",
                "ytick.color": "#222222",
                "text.color": "#111111",
                "grid.color": "#e0e0e0",
            }
        )

    return rc


@contextmanager
def viewer_context(theme: str = "light") -> Generator[None]:
    """Context manager applying viewer_rc cleanly."""
    with plt.rc_context(cast("Any", viewer_rc(theme))):
        yield


def paint_theme(
    fig: plt.Figure,
    axes: list[Any],
    theme: str = "light",
    texts: list[Any] | None = None,
) -> None:
    """Apply the viewer palette to an already-open figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to recolor.
    axes : list of Axes
        Plot axes. Widget axes are left alone so button colours stay put.
    theme : {"light", "dark"}, optional
        Palette name, by default "light".
    texts : list of Text or None, optional
        Extra text artists, such as the stats panel, to recolor.
    """
    rc = viewer_rc(theme)
    fig.set_facecolor(cast("Any", rc["figure.facecolor"]))
    fig.set_edgecolor(cast("Any", rc["figure.edgecolor"]))
    label = cast("Any", rc["axes.labelcolor"])
    tick = cast("Any", rc["xtick.color"])
    edge = cast("Any", rc["axes.edgecolor"])
    face = cast("Any", rc["axes.facecolor"])
    text_c = cast("Any", rc["text.color"])
    for ax in axes:
        ax.set_facecolor(face)
        ax.tick_params(colors=tick, labelcolor=tick)
        ax.xaxis.label.set_color(label)
        ax.yaxis.label.set_color(label)
        ax.title.set_color(text_c)
        for spine in ax.spines.values():
            spine.set_color(edge)
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_color(text_c)
    for text in texts or []:
        text.set_color(text_c)
