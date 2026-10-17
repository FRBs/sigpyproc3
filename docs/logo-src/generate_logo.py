"""Waterfall mark for sigpyproc, set in IBM Plex Mono.

This file is the logo source. It is not a documentation page.
Text is outlined, so the SVGs do not need the font installed.

Writes the files the README and the docs reference into docs/_static/logo/:

- ``mark.svg``: the dispersed pulse, for the docs landing hero
- ``logo.svg``: mark plus the name, for the docs sidebar
- ``logo-tagline.svg``: mark, name, and one line, for the README
- ``favicon.svg``: the mark, following the browser light/dark preference

Each of the first three also has a ``-dark`` file. Run::

    python docs/logo-src/generate_logo.py

IBM Plex Mono v1.1.0 is downloaded once into ``.cache/`` and checked by sha256.
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "_static" / "logo"
BUILD = HERE / "build"
CACHE = HERE / ".cache"

FONT_URL = (
    "https://cdn.jsdelivr.net/npm/@ibm/plex-mono@1.1.0"
    "/fonts/complete/woff/IBMPlexMono-{}.woff"
)
FONT_SHA256 = {
    "Medium": "15d0fcad70465530211f78116d13df0fc1a930361b613f4418238d212bafebf1",
    "Regular": "1d5732f53287cbe58936deabda52a60d810b84ac4a3fbdefa8e7632a2c38c39d",
}

# Name and tagline. "proc" takes the sweep colour; the tagline stays quiet.
WORD = "sigpyproc"
ACCENT_AT = 5
TAGLINE = "PULSAR AND FRB I/O"


def fmt(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def palette(mode: str) -> dict[str, str]:
    if mode == "dark":
        return {
            "ink": "#EDF0F4",
            "accent": "#38bdf8",
            "muted": "#8A94A3",
            "off": "#232c3a",
            "halo": "#1e5f84",
            "core": "#38bdf8",
            "fill": "#664614",
            "trace": "#f59e0b",
        }
    return {
        "ink": "#12151B",
        "accent": "#0284c7",
        "muted": "#5C6573",
        "off": "#e3e9f0",
        "halo": "#9fd8f5",
        "core": "#0284c7",
        "fill": "#fde3b8",
        "trace": "#d97706",
    }


MARK_ROLES = ("off", "halo", "core", "fill", "trace")


def _paint(role: str, color: dict[str, str], *, adaptive: bool) -> str:
    if adaptive:
        return f'class="{role}"'
    return f'fill="{color[role]}"'


def _cells(
    color: dict[str, str], *, adaptive: bool, band_h: float, simple: bool = False
) -> list[str]:
    """Dynamic spectrum: 4 channels by 7 samples, high frequency on top.

    The pulse arrives later at lower frequency, so the bright cells slide
    down and to the right, one cell per channel, with a dim cell either side.
    """
    n_chan, n_samples = (3, 5) if simple else (4, 7)
    span = n_samples - 2.6
    gap = 3.0 if simple else 2.2
    ch_h = (band_h - (n_chan - 1) * gap) / n_chan
    cell_w = (100.0 - (n_samples - 1) * gap) / n_samples
    rx = min(cell_w, ch_h) * 0.28
    parts: list[str] = []
    for ch in range(n_chan):
        center = 0.8 + ch * (span / (n_chan - 1))
        for col in range(n_samples):
            dist = abs(col - center)
            role = "core" if dist < 0.6 else "halo" if dist < 1.3 else "off"
            if simple and role != "core":
                continue
            parts.append(
                f'<rect x="{fmt(col * (cell_w + gap))}" y="{fmt(ch * (ch_h + gap))}" '
                f'width="{fmt(cell_w)}" height="{fmt(ch_h)}" rx="{fmt(rx)}" '
                f"{_paint(role, color, adaptive=adaptive)}/>"
            )
    return parts


def _profile(
    color: dict[str, str], *, adaptive: bool, height: float, stroke: float
) -> list[str]:
    """Pulse profile along the bottom edge of the 100 by 100 box."""
    peak_x, sigma, steps = 50.0, 10.0, 60
    curve = []
    for i in range(steps + 1):
        px = i * 100.0 / steps
        pulse = math.exp(-0.5 * ((px - peak_x) / sigma) ** 2)
        curve.append(f"{fmt(px)},{fmt(100.0 - pulse * height)}")
    area = ["0,100", *curve, "100,100"]
    fill = _paint("fill", color, adaptive=adaptive)
    trace = 'class="trace"' if adaptive else f'stroke="{color["trace"]}"'
    return [
        f'<polygon points="{" ".join(area)}" {fill}/>',
        (
            f'<polyline points="{" ".join(curve)}" fill="none" {trace} '
            f'stroke-width="{fmt(stroke)}" stroke-linecap="round" stroke-linejoin="round"/>'
        ),
    ]


def render_mark(mode: str, *, adaptive: bool = False, simple: bool = False) -> str:
    """Dispersed pulse in a 100 by 100 box, profile along the bottom.

    ``simple`` keeps three bright cells on a coarser grid and thickens the
    line, for sizes where the full grid would turn to noise.
    """
    color = palette("light" if adaptive else mode)
    height = 30.0
    stroke = 5.5 if simple else 3.6
    band_h = 100.0 - height - 8.0
    return "".join(
        _cells(color, adaptive=adaptive, band_h=band_h, simple=simple)
        + _profile(color, adaptive=adaptive, height=height - stroke / 2, stroke=stroke)
    )


def svg(viewbox: tuple[float, float, float, float], body: str, label: str) -> str:
    box = " ".join(fmt(value) for value in viewbox)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}" role="img" '
        f'aria-label="{label}"><title>{label}</title>{body}</svg>\n'
    )


def font_path(weight: str) -> Path:
    path = CACHE / f"IBMPlexMono-{weight}.woff"
    digest = FONT_SHA256[weight]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        CACHE.mkdir(parents=True, exist_ok=True)
        url = FONT_URL.format(weight)
        data = urllib.request.urlopen(url, timeout=30).read()  # noqa: S310
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            msg = f"IBMPlexMono-{weight}: unexpected sha256 {got}"
            raise SystemExit(msg)
        path.write_bytes(data)
    return path


class Face:
    def __init__(self, weight: str) -> None:
        self.font = TTFont(font_path(weight))
        self.glyphs = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm
        self.hmtx = self.font["hmtx"]

    def bounds(self, char: str) -> tuple[float, float, float, float]:
        pen = BoundsPen(self.glyphs)
        self.glyphs[self.cmap[ord(char)]].draw(pen)
        if pen.bounds is None:
            msg = f"glyph {char!r} has no outline"
            raise SystemExit(msg)
        return pen.bounds

    def outline(
        self, text: str, size: float, x0: float, baseline: float, tracking: float = 0.0
    ) -> tuple[str, float]:
        """Path data for ``text`` and its advance. ``tracking`` is in em."""
        scale = size / self.upm
        x, parts = x0, []
        for i, char in enumerate(text):
            name = self.cmap[ord(char)]
            pen = SVGPathPen(self.glyphs, ntos=fmt)
            self.glyphs[name].draw(
                TransformPen(pen, (scale, 0, 0, -scale, x, baseline))
            )
            parts.append(pen.getCommands())
            x += self.hmtx[name][0] * scale
            if i < len(text) - 1:
                x += tracking * size
        return "".join(parts), x - x0


def _extent(face: Face, text: str) -> tuple[float, float]:
    """Lowest and highest outline point of ``text`` in em (descender, top)."""
    boxes = [face.bounds(c) for c in text if c != " "]
    return (
        min(b[1] for b in boxes) / face.upm,
        max(b[3] for b in boxes) / face.upm,
    )


def lockup_svg(med: Face, reg: Face, mode: str, *, tagline: bool) -> str:
    color = palette(mode)
    mark_h = 100.0
    # Size the name from the x-height so the lowercase word matches the mark.
    x_height = 40.0
    name_size = x_height / (med.bounds("s")[3] / med.upm)
    x_text = mark_h + 18.0
    low, top = _extent(med, WORD)
    word_top, word_bot = top * name_size, -low * name_size
    if tagline:
        sub_size, sub_gap = 21.0, 8.0
        sub_low, sub_top = _extent(med, TAGLINE)
        sub_h = (sub_top - sub_low) * sub_size
        block = word_top + word_bot + sub_gap + sub_h
        baseline = (mark_h - block) / 2 + word_top
        sub_base = baseline + word_bot + sub_gap + sub_top * sub_size
    else:
        baseline = (mark_h + word_top - word_bot) / 2
    stem, rest = WORD[:ACCENT_AT], WORD[ACCENT_AT:]
    stem_d, stem_w = med.outline(stem, name_size, x_text, baseline)
    rest_d, rest_w = med.outline(rest, name_size, x_text + stem_w, baseline)
    lsb = med.bounds("s")[0] / med.upm * name_size
    body = (
        f'<g transform="translate({fmt(-lsb)} 0)">'
        f'<path d="{stem_d}" fill="{color["ink"]}"/>'
        f'<path d="{rest_d}" fill="{color["accent"]}"/>'
    )
    word_w = stem_w + rest_w
    right = x_text + word_w - lsb
    if tagline:
        sub_d, sub_w = med.outline(TAGLINE, sub_size, x_text, sub_base, tracking=0.14)
        sub_lsb = med.bounds("P")[0] / med.upm * sub_size
        body += (
            f'<path d="{sub_d}" transform="translate({fmt(lsb - sub_lsb)} 0)" '
            f'fill="{color["muted"]}"/>'
        )
        right = max(right, x_text + sub_w - sub_lsb)
    body += "</g>" + render_mark(mode)
    pad = 2.0
    label = "sigpyproc: pulsar and FRB I/O" if tagline else "sigpyproc"
    return svg((-pad, -pad, right + 2 * pad, mark_h + 2 * pad), body, label)


def mark_svg(mode: str) -> str:
    pad = 2.0
    return svg(
        (-pad, -pad, 100 + 2 * pad, 100 + 2 * pad), render_mark(mode), "sigpyproc"
    )


def favicon_svg() -> str:
    light, dark = palette("light"), palette("dark")
    fills = {
        "off": "fill",
        "halo": "fill",
        "core": "fill",
        "fill": "fill",
        "trace": "stroke",
    }

    def rules(pal: dict[str, str]) -> str:
        return "".join(f".{r}{{{fills[r]}:{pal[r]}}}" for r in MARK_ROLES)

    style = (
        f"<style>{rules(light)}"
        f"@media (prefers-color-scheme:dark){{{rules(dark)}}}</style>"
    )
    body = style + render_mark("light", adaptive=True, simple=True)
    return svg((-4, -4, 108, 108), body, "sigpyproc")


def social_preview_svg(med: Face, reg: Face) -> str:
    """1280 by 640 card for GitHub social preview."""
    color = palette("dark")
    mark_h = 100.0
    x_height = 40.0
    name_size = x_height / (med.bounds("s")[3] / med.upm)
    x_text = mark_h + 18.0
    low, top = _extent(med, WORD)
    word_top, word_bot = top * name_size, -low * name_size
    sub_size, sub_gap = 21.0, 8.0
    sub_low, sub_top = _extent(med, TAGLINE)
    sub_h = (sub_top - sub_low) * sub_size
    block = word_top + word_bot + sub_gap + sub_h
    baseline = (mark_h - block) / 2 + word_top
    sub_base = baseline + word_bot + sub_gap + sub_top * sub_size

    stem, rest = WORD[:ACCENT_AT], WORD[ACCENT_AT:]
    stem_d, stem_w = med.outline(stem, name_size, x_text, baseline)
    rest_d, rest_w = med.outline(rest, name_size, x_text + stem_w, baseline)
    lsb = med.bounds("s")[0] / med.upm * name_size

    sub_d, sub_w = med.outline(TAGLINE, sub_size, x_text, sub_base, tracking=0.14)
    sub_lsb = med.bounds("P")[0] / med.upm * sub_size

    lockup = (
        f'<g transform="translate({fmt(-lsb)} 0)">'
        f'<path d="{stem_d}" fill="{color["ink"]}"/>'
        f'<path d="{rest_d}" fill="{color["accent"]}"/>'
        f'<path d="{sub_d}" transform="translate({fmt(lsb - sub_lsb)} 0)" '
        f'fill="{color["muted"]}"/>'
        f"</g>{render_mark('dark')}"
    )
    lockup_w = max(x_text + stem_w + rest_w - lsb, x_text + sub_w - sub_lsb)
    lockup_h = 100.0

    scale = 1.95
    tx = (1280.0 - lockup_w * scale) / 2.0
    ty = (640.0 - lockup_h * scale) / 2.0

    card = (
        '<rect width="1280" height="640" fill="#0B0F19"/>'
        '<rect x="1" y="1" width="1278" height="638" rx="0" fill="none" stroke="#1E293B" stroke-width="2"/>'
        f'<g transform="translate({fmt(tx)} {fmt(ty)}) scale({fmt(scale)})">'
        f"{lockup}"
        "</g>"
    )
    return svg((0, 0, 1280, 640), card, "sigpyproc: pulsar and FRB I/O")


def main() -> None:
    med, reg = Face("Medium"), Face("Regular")
    STATIC.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    files = {
        "mark.svg": mark_svg("light"),
        "mark-dark.svg": mark_svg("dark"),
        "logo.svg": lockup_svg(med, reg, "light", tagline=False),
        "logo-dark.svg": lockup_svg(med, reg, "dark", tagline=False),
        "logo-tagline.svg": lockup_svg(med, reg, "light", tagline=True),
        "logo-tagline-dark.svg": lockup_svg(med, reg, "dark", tagline=True),
        "favicon.svg": favicon_svg(),
    }
    for name, content in files.items():
        (STATIC / name).write_text(content, encoding="utf-8")

    # GitHub social preview. Kept out of the docs tree, same as dmt's build/.
    social = social_preview_svg(med, reg)
    (BUILD / "social-preview.svg").write_text(social, encoding="utf-8")
    try:
        import cairosvg
    except ImportError:
        return
    cairosvg.svg2png(
        bytestring=social.encode("utf-8"),
        write_to=str(BUILD / "social-preview.png"),
        output_width=1280,
        output_height=640,
    )


if __name__ == "__main__":
    main()
