"""Shared look for the post's figures (figures/*.png).

LessWrong shows an image at the width of the text column, about 700 px. Every figure is therefore drawn at the same physical width
(WIDTH = 7 in) with the same point sizes, so its text lands at the same size on the page: 1 pt is about 1.4 px there, so tick labels
(8 pt) come out near 11 px and titles (9.5 pt) near 13 px. Saved at 300 dpi (2,100 px wide) so the image stays sharp on high-density screens.

usage (in a plotting script):
    from spurious_probes import figstyle as S
    S.setup()
    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, 2.4), layout="constrained")
    S.panel_title(ax, "A", "How many questions work")
    S.save(fig, "figures/x.png")
"""
from __future__ import annotations

import matplotlib.pyplot as plt

WIDTH, DPI = 7.0, 300
# ink and structure
INK, MUTED, FAINT, GRID, SPINE = "#1f2328", "#6b7280", "#9aa0a6", "#ededed", "#c9ccd1"
# meaning that is shared across figures: evaluation vs real use, and "everything else"
EVAL, REAL, OTHER = "#d9622b", "#2f6fce", "#d9dbdf"
# the toy model (a simulated counterpart of a real curve): at the g fitted to a screen (TOY) and at the g measured on activations (MEAS);
# the null (labels shuffled / g = 0). Each also differs by line style or marker, so no pair relies on colour alone.
TOY, MEAS, NULL = "#7a4fb0", "#15857a", "#a3a8b0"
FS = {"tick": 8.0, "label": 8.5, "legend": 8.0, "note": 7.5, "title": 9.5}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Helvetica Neue", "mathtext.it": "Helvetica Neue:italic", "mathtext.bf": "Helvetica Neue:bold",
        "font.size": FS["label"], "axes.labelsize": FS["label"], "axes.titlesize": FS["title"], "legend.fontsize": FS["legend"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.labelpad": 3, "axes.titlepad": 6, "axes.titlelocation": "left",
        "lines.linewidth": 1.5, "lines.markersize": 4, "patch.linewidth": 0,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.constrained_layout.w_pad": 0.04, "figure.constrained_layout.h_pad": 0.04,
    })


def panel_title(ax, letter: str | None, text: str, **kw) -> None:
    """Panel title, left-aligned: a bold letter and a plain-weight phrase, so the titles label the panels without shouting."""
    t = ax.set_title(("$\\bf{" + letter + "}$   " if letter else "") + text, loc="left", fontsize=FS["title"], color=INK, **kw)
    return t


def save(fig, out: str, pad: float = 0.04) -> None:
    """Save at 300 dpi with a tight white margin, and say how far the saved width is from the column width (text size scales with it)."""
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=pad)
    from PIL import Image
    w, h = Image.open(out).size
    print(f"wrote {out}: {w}x{h} px = {w / DPI:.2f} x {h / DPI:.2f} in (text scale on the page {WIDTH * DPI / w:.2f}x)")
