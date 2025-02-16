from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import attrs
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib import pyplot as plt


def set_seaborn(font_scale: float = 1, **rc_kwargs) -> None:  # noqa: ANN003
    """Set seaborn style with custom rc parameters."""
    rc = {
        # Fontsizes
        "font.size": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.top": False,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.right": False,
        # Set line widths
        "axes.axisbelow": "line",
        "axes.linewidth": 1,
        "lines.linewidth": 1.5,
        "lines.markersize": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        # Use LaTeX for math formatting
        "text.usetex": False,  # Set to True if available
        "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
    }
    rc.update(rc_kwargs)
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font_scale=font_scale,
        rc=rc,
    )


@attrs.frozen(auto_attribs=True)
class TableEntry:
    """Single entry in the PlotTable with optional unit."""

    name: str = attrs.field(converter=str)
    value: str = attrs.field(converter=str)
    unit: str = attrs.field(converter=str, default="")
    color: str = attrs.field(default="black")


class PlotTable:
    """A class to plot text tables in matplotlib using relative positions.

    Parameters
    ----------
    col_offsets : dict[str, float] | None, optional
        Dictionary of column names to x-positions (0-1), by default None.
    top_margin : float | None, optional
        Top margin as a fraction of the axis height, by default None.
    line_height : float | None, optional
        Height of each line as a fraction of the axis height, by default None.
    font_size : int | None, optional
        Font size, by default None.
    font_family : str | None, optional
        Font family, by default monospace.
    """

    DEFAULTS: ClassVar[dict] = {
        "col_offsets": {
            "name": 0.4,
            "value": 0.8,
            "unit": 0.85,
        },
        "top_margin": 0.05,
        "line_height": 0.05,
        "font_size": 12,
        "font_family": "monospace",
    }

    def __init__(
        self,
        col_offsets: dict[str, float] | None = None,
        top_margin: float | None = None,
        line_height: float | None = None,
        font_size: int | None = None,
        font_family: str | None = None,
    ) -> None:
        self.col_offsets = col_offsets or self.DEFAULTS["col_offsets"]
        self.top_margin = top_margin or self.DEFAULTS["top_margin"]
        self.line_height = line_height or self.DEFAULTS["line_height"]
        self.font_size = font_size or self.DEFAULTS["font_size"]
        self.font_family = font_family or self.DEFAULTS["font_family"]

        self.entries: list[TableEntry | None] = []

    def add_entry(
        self,
        name: str,
        value: str | float,
        unit: str = "",
        color: str = "black",
    ) -> None:
        """Add an entry to the table."""
        self.entries.append(TableEntry(name, str(value), unit, color))

    def skip_line(self) -> None:
        """Add a blank line to the table."""
        self.entries.append(None)

    def plot(self, ax: plt.Axes) -> None:
        """Plot the table on the given axis."""
        ax.axis("off")
        y = 1.0 - self.top_margin
        for entry in self.entries:
            if entry is not None:
                for col, x in self.col_offsets.items():
                    ax.text(
                        x,
                        y,
                        getattr(entry, col),
                        ha="right" if col != "unit" else "left",
                        va="center",
                        transform=ax.transAxes,
                        family=self.font_family,
                        size=self.font_size,
                        color=entry.color,
                    )
            y -= self.line_height
