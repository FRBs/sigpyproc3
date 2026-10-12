"""Keymap manager and help generator for interactive viewers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import KeyEvent
    from matplotlib.figure import Figure
    from matplotlib.transforms import BboxBase


class KeyMap:
    """Registry of viewer keybindings and generator for help text."""

    def __init__(self) -> None:
        self._bindings: dict[str, tuple[Callable[[], None], str, str]] = {}
        self._entries: list[tuple[str, list[str], str]] = []  # (group, keys, desc)

    def bind(
        self,
        keys: str | Sequence[str],
        action: Callable[[], None],
        desc: str,
        group: str = "General",
    ) -> None:
        """Bind one or more keys to an action."""
        key_list = [keys] if isinstance(keys, str) else list(keys)
        claimed = set(key_list)
        for k in key_list:
            self._bindings[k] = (action, desc, group)
        # A later bind wins. Drop the key from older help rows so the overlay
        # matches what handle() will actually call.
        kept_entries: list[tuple[str, list[str], str]] = []
        for old_group, old_keys, old_desc in self._entries:
            remaining = [k for k in old_keys if k not in claimed]
            if remaining:
                kept_entries.append((old_group, remaining, old_desc))
        kept_entries.append((group, key_list, desc))
        self._entries = kept_entries

    def handle(self, event: KeyEvent) -> bool:
        """Handle a matplotlib key event. Returns True if handled."""
        key = event.key
        if key in self._bindings:
            func, _, _ = self._bindings[key]
            func()
            return True
        return False

    def help_table(self) -> list[tuple[str, str, str]]:
        """Return list of (group, keys_str, desc)."""
        table = []
        for group, keys, desc in self._entries:
            table.append((group, ", ".join(keys), desc))
        return table

    def format_help(self) -> str:
        """Format a human-readable help string grouped by category."""
        groups: dict[str, list[tuple[str, str]]] = {}
        for group, keys, desc in self._entries:
            groups.setdefault(group, []).append((", ".join(keys), desc))

        lines = ["=== sigpyproc viewer shortcuts ==="]
        for grp, items in groups.items():
            lines.append(f"\n[{grp}]")
            for k_str, desc in items:
                lines.append(f"  {k_str:<18} : {desc}")
        return "\n".join(lines)


class BlitManager:
    """Manages blit updates for interactive animations in matplotlib."""

    def __init__(
        self,
        fig: Figure,
        animated_artists: Iterable[Artist] = (),
        axes: Axes | None = None,
    ) -> None:
        self.fig = fig
        self.axes = axes
        self.canvas: Any = cast("Any", fig.canvas)
        self._artists: list[Artist] = []
        for a in animated_artists:
            self.add_artist(a)
        self._bg = None
        self._cid = self.canvas.mpl_connect("draw_event", self.on_draw)

    def _bbox(self) -> BboxBase:
        if self.axes is not None:
            return self.axes.bbox
        return self.fig.bbox

    def on_draw(self, event: object) -> None:
        """Cache the clean background on full canvas draw."""
        if event is not None and getattr(event, "canvas", None) != self.canvas:
            return
        self._bg = self.canvas.copy_from_bbox(self._bbox())
        self._draw_animated()

    def add_artist(self, art: Artist) -> None:
        """Add an artist to be managed and drawn during blits."""
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self) -> None:
        """Draw all managed artists."""
        drawer = self.axes if self.axes is not None else self.fig
        for a in self._artists:
            drawer.draw_artist(a)

    def update(self) -> None:
        """Restore background, redraw animated artists, and blit to screen."""
        if self._bg is None:
            self.on_draw(None)
            return
        self.canvas.restore_region(self._bg)
        self._draw_animated()
        self.canvas.blit(self._bbox())
