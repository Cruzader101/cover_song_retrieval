"""Plot styling: one palette, two modes, no per-figure colour decisions.

READMEs are read on both light and dark backgrounds, so every figure is rendered
twice from the same code and selected by the browser. Colour is assigned by the
job it does -- grey for the baselines that exist only as a floor, one accent for
the methods that actually use audio content, a single-hue ramp for magnitude --
rather than by which line happens to be drawn first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIGURES = Path("docs/figures")

#: Blue sequential ramp, lightest to darkest.
_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab",
         "#184f95", "#0d366b"]


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text: str
    secondary: str
    muted: str
    grid: str
    #: Baselines. Deliberately recessive: they are context, not findings.
    baseline: str
    #: Categorical slots 1-3, in fixed order.
    series: tuple[str, str, str]

    @property
    def ramp(self) -> LinearSegmentedColormap:
        """Magnitude ramp running away from the surface, so low always recedes."""
        stops = _BLUE if self.name == "light" else list(reversed(_BLUE))
        return LinearSegmentedColormap.from_list(f"csr_{self.name}", stops)


LIGHT = Theme(
    name="light", surface="#fcfcfb", text="#0b0b0b", secondary="#52514e",
    muted="#8a8983", grid="#e2e1dc", baseline="#b0afa8",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
)
DARK = Theme(
    name="dark", surface="#1a1a19", text="#ffffff", secondary="#c3c2b7",
    muted="#8a8983", grid="#33332f", baseline="#6a6963",
    series=("#3987e5", "#d95926", "#199e70"),
)
THEMES = (LIGHT, DARK)


def apply(theme: Theme) -> None:
    """Set the rcParams a figure inherits, so no plot restates them."""
    mpl.rcParams.update({
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.text,
        "axes.labelcolor": theme.secondary,
        "axes.edgecolor": theme.grid,
        "xtick.color": theme.secondary,
        "ytick.color": theme.secondary,
        "grid.color": theme.grid,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, stem: str, theme: Theme, out: Path = FIGURES) -> Path:
    """Write <stem>-<mode>.png and close the figure."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}-{theme.name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def picture_tag(stem: str, alt: str, out: Path = FIGURES) -> str:
    """The <picture> block a README embeds, so the figure follows the reader.

    as_posix() because this string is a URL: on Windows str(Path) hands back a
    backslash path, which renders as a broken image.
    """
    base = Path(out).as_posix()
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" '
        f'srcset="{base}/{stem}-dark.png">\n'
        f'  <img alt="{alt}" src="{base}/{stem}-light.png">\n'
        "</picture>"
    )
