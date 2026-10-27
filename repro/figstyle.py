"""Shared design of the figures (step 30): palette, type, rcParams, panel titles and the TeX check.

One brand blue for the key reference (the count detector); navy and slate greys for everything else, with two muted
accents (teal: the identity-aware detectors, coral: the wiring statistics); heatmap cells purple/ochre. Text in Inter, math in Libertinus, as in
the paper.
"""
from __future__ import annotations

import shutil
import subprocess

TITLE, BODY, TICK, SMALL = 9.5, 9.0, 8.0, 8.0     # panel titles; axis labels; ticks; legend, annotations, cells

# palette -------------------------------------------------------------------------------------------------
BLUE, NAVY, INK, GREY = "#4D6BFE", "#1B2559", "#23272F", "#6B7280"   # brand blue, navy, ink, grey (paper)
TINT, RULE = "#F3F5FF", "#C9D1F5"                                    # light blue tint, blue-grey rule (paper)
CORAL, TEAL, DTEAL = "#E2725B", "#1E9E94", "#0D5C57"                 # the two muted accents (+ dark teal)
SLATE, MIDGREY, LIGHTGREY = "#3F4A6B", "#858D9E", "#A3AAB7"          # navy-grey ramp for neutral entities
TICKTXT, AXIS, GRID, REF = "#4B5563", "#9AA1AD", "#E6E8EC", "#B8BEC8"  # tick text, spines, grid, ref lines
FAINT = "#9CA3AF"                                                    # n/a and other quiet notes
# heatmap cells: purple above the centre, ochre below; hues used for no row family, so a colour has one meaning
DIVERGING = ((0.0, "#9A6B12"), (0.5, "#FFFFFF"), (0.86, "#7B61C4"), (1.0, "#4E3796"))

_PREAMBLE = (r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage[scale=0.88]{inter}"
             r"\usepackage{libertinus-type1}\usepackage{libertinust1math}"
             r"\renewcommand{\sfdefault}{Inter-TLF}\renewcommand{\familydefault}{\sfdefault}"
             r"\usepackage[kerning=true]{microtype}"   # Inter's Type 1 hyphen: close its side bearings
             r"\SetExtraKerning[unit=1em]{encoding=T1,family=Inter-TLF}{-={-90,-90}}")


def usetex_available() -> bool:
    """pdflatex, cm-super and the paper's fonts (Inter, Libertinus) are installed."""
    if shutil.which("pdflatex") is None:
        return False
    try:
        r = [subprocess.run(["kpsewhich", f], capture_output=True, text=True, timeout=20, check=False)
             for f in ("type1ec.sty", "inter.sty", "libertinust1math.sty")]
    except (OSError, subprocess.SubprocessError):
        return False
    return all(x.returncode == 0 and x.stdout.strip() != "" for x in r)


def rc(usetex: bool) -> dict:
    """rcParams of the design (apply after the tueplots bundle)."""
    cfg = {"text.usetex": usetex, "font.family": "sans-serif", "font.size": BODY, "axes.labelsize": BODY,
           "axes.titlesize": TITLE, "legend.fontsize": SMALL, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
           "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
           "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": TICKTXT, "ytick.labelcolor": TICKTXT,
           "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
           "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3.0, "lines.solid_capstyle": "round",
           "legend.handletextpad": 0.4, "legend.frameon": False}
    if usetex:
        cfg["text.latex.preamble"] = _PREAMBLE
    else:  # fallback without LaTeX: the closest installed sans; math font then differs from the paper
        cfg.update({"font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                    "mathtext.fontset": "dejavusans", "axes.unicode_minus": True, "pdf.fonttype": 42})
    return cfg


def text_on(rgb) -> str:
    """Ink or white, whichever has the higher WCAG contrast on the fill rgb (0-1)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    ink = 0.2126 * 0.0168 + 0.7152 * 0.0203 + 0.0722 * 0.0284          # luminance of INK #23272F
    return "white" if 1.05 / (lum + 0.05) > (lum + 0.05) / (ink + 0.05) else INK


def minus(usetex: bool) -> str:
    """Minus sign in the sans font: Inter's en dash under T1 (\\textminus would take a TS1 glyph from CM sans)."""
    return r"\textendash{}" if usetex else "−"   # not "--": the hyphen kerning would break the ligature


def bold(s: str, usetex: bool) -> str:
    return rf"\textbf{{{s}}}" if usetex else s


def panel_title(ax, tag: str, title: str, usetex: bool, x: float = -0.02, y: float = 1.04, **kw):
    """Bold Inter title: the tag '(a)' in brand blue, the title in navy (as the paper's numbered headings)."""
    w = {} if usetex else {"fontweight": "bold"}
    t = ax.text(x, y, bold(f"({tag})", usetex), transform=ax.transAxes, ha="left", va="baseline",
                fontsize=TITLE, color=BLUE, **w, **kw)
    if title:                                       # same baseline, 3.2 pt after the measured tag
        fig = ax.figure
        bb = t.get_window_extent(fig.canvas.get_renderer())
        x1 = ax.transAxes.inverted().transform((bb.x1 + 3.2 * fig.dpi / 72, bb.y0))[0]
        ax.text(x1, y, bold(title, usetex), transform=ax.transAxes, ha="left", va="baseline", fontsize=TITLE,
                color=NAVY, **w, **kw)
    return t


def frame(ax, grid: str | None = "y") -> None:
    """Thin left/bottom axes, light grid on one axis only."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
