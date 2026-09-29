"""The toy-model illustration (figures/toy_schematic.png), three panes: (1) the regime offsets the activations, (2) each answer's
logit reads a random projection of the offset, (3) so the answer shares shift with the regime.

All numbers come from one drawn geometry (answer directions at fixed angles to the regime offset), and the shares are simulated from it.
Answer colours follow figure 1.

usage: uv run scripts/figures/toy_schematic.py
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from spurious_probes import figstyle as S

INK, MUTED, EVAL, REAL = "#1f2328", "#6b7280", "#d9622b", "#2f6fce"
ANS = [("frog", "#5aa845", 18), ("axolotl", "#e27aa8", 148), ("salamander", "#b58cc4", 92)]   # name, colour, angle to the regime shift (deg)
BETA, SHIFT, SIGMA = 1.2, 1.8, 0.45   # frog's head start; distance between the two clouds; cloud spread (= score noise)


def arrow(ax, p, q, color, lw=2.0, ms=14, z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def shares(nudge, rng):
    base = np.array([BETA, 0.0, 0.0]); out = {}
    for lab, sign in (("evaluation", +1), ("real use", -1)):
        n = 200000; lg = base + sign * nudge / 2 + rng.normal(0, SIGMA * 2, (n, 3)) + rng.gumbel(size=(n, 3))
        out[lab] = np.bincount(lg.argmax(1), minlength=3) / n
    return out


def three(out: str, rng) -> None:
    """Three panes: 1. the regime shifts the state; 2. each answer reads it along a random direction (nudge = projection);
    3. the answers' shares shift between regimes. Drawn at the post's column width with the shared type sizes (src/spurious_probes/figstyle.py);
    panes 1 and 2 share one scale (so both offset arrows have the same length per unit), and the panes are laid out in inches."""
    S.setup()
    ans = [("frog", "#5aa845", 30), ("axolotl", "#e27aa8", 148), ("salamander", "#b58cc4", -103)]   # salamander nearly perpendicular: a small nudge
    us = [np.array([np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t))]) for _, _, t in ans]; nudge = np.array([SHIFT * u[0] for u in us])
    sh = shares(nudge, rng); ba = 0.5 * (sh["evaluation"][0] + 1 - sh["real use"][0])
    FS, WT = S.FS["label"], "medium"            # in-figure labels: label size, medium weight (calmer than bold)
    # layout in inches: panes 1 and 2 at one shared data scale, pane 3 a fixed width, equal gaps; two-line titles above
    lim1, lim2, ylim12 = (-2.42, 2.42), (-2.05, 3.15), (-1.85, 1.45)
    w3, gap, title_h, FW = 2.05, 0.3, 0.4, S.WIDTH
    scale = (FW - w3 - 2 * gap) / ((lim1[1] - lim1[0]) + (lim2[1] - lim2[0]))      # inches per data unit
    w1, w2, H = (lim1[1] - lim1[0]) * scale, (lim2[1] - lim2[0]) * scale, (ylim12[1] - ylim12[0]) * scale
    FH = H + title_h
    fig = plt.figure(figsize=(FW, FH))
    x0s = [0.0, w1 + gap, w1 + w2 + 2 * gap]
    ax1, ax2, ax3 = (fig.add_axes([x / FW, 0.0, w / FW, H / FH]) for x, w in zip(x0s, (w1, w2, w3)))
    # 1. regime offset
    c = 1.1
    for sign, col, lab in ((-1, S.REAL, "real use"), (+1, S.EVAL, "evaluation")):
        pts = np.clip(rng.normal(0, 0.48, (90, 2)), -1.2, 1.2) + [sign * c, 0]
        ax1.scatter(pts[:, 0], pts[:, 1], s=6, color=col, alpha=0.5, lw=0, zorder=2)
        ax1.text(sign * c, -1.42, lab, ha="center", va="top", fontsize=FS, color=col, fontweight=WT)
    arrow(ax1, (-c, 0), (c, 0), S.INK, lw=1.5, ms=9, z=4)
    ax1.text(0, 0.16, "regime offset", ha="center", va="bottom", fontsize=FS, color=S.INK, fontweight=WT, zorder=5,
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    # 2. random projection
    k_draw = 0.68; Sv = np.array([SHIFT * k_draw, 0.0]); L = 1.55; o = np.array([0.0, 0.0])
    for (name, col, _), u, dn in zip(ans, us, nudge):
        arrow(ax2, tuple(o), tuple(o + L * u), col, lw=1.3, ms=8, z=2)
        foot = o + k_draw * dn * u
        if dn < 0: ax2.plot([o[0], foot[0]], [o[1], foot[1]], color=col, lw=1.1, ls=(0, (1, 1.4)), zorder=1)
        ax2.plot([o[0] + Sv[0], foot[0]], [o[1], foot[1]], color=S.MUTED, lw=0.6, ls=(0, (3, 2.5)), zorder=1)
        ax2.plot(*foot, "o", color=col, ms=4.4, zorder=5, mec="white", mew=0.6)
        tip = o + L * u; lab = f"{name} {'+' if dn > 0 else '−'}{abs(dn):.1f}"
        if name == "salamander":
            ax2.text(tip[0] + 0.14, tip[1] + 0.02, lab, ha="left", va="baseline", fontsize=FS, color=col, fontweight=WT)
        else: ax2.text(*(tip + u * 0.1 + [0, 0.12]), lab, ha="center", va="bottom", fontsize=FS, color=col, fontweight=WT)
    arrow(ax2, tuple(o), tuple(o + Sv), S.INK, lw=1.5, ms=9, z=4)
    ax2.text(o[0] + Sv[0] + 0.12, 0.0, "regime offset", ha="left", va="center", fontsize=FS, color=S.INK, fontweight=WT, zorder=5)
    # 3. preference shift
    y = np.array([1.2, 0.0, -1.2])
    for k, (name, col, _) in enumerate(ans):
        r, e = 100 * sh["real use"][k], 100 * sh["evaluation"][k]
        ax3.plot([0, 100], [y[k]] * 2, color="#eeeff1", lw=4, solid_capstyle="round", zorder=0)
        arrow(ax3, (r, y[k]), (e + (-2.6 if e > r else 2.6), y[k]), col, lw=1.4, ms=8, z=2)
        ax3.plot(r, y[k], "o", color=S.REAL, ms=6, zorder=3); ax3.plot(e, y[k], "o", color=S.EVAL, ms=6, zorder=3)
        close = abs(e - r) < 16
        for v, cc, dy in ((r, S.REAL, -0.24 if close else 0.24), (e, S.EVAL, 0.24)):
            t_ = ax3.text(v, y[k] + dy, f"{v:.0f}%", ha="center", va="bottom" if dy > 0 else "top", fontsize=S.FS["tick"], color=cc, fontweight=WT)
            if dy < 0: low_label = t_   # the lowest label in pane 3; the note in pane 2 is bottom-aligned with it
        ax3.text(-5, y[k], name, ha="right", va="center", fontsize=FS, color=col, fontweight=WT)
    ax3.legend(handles=[plt.Line2D([], [], color=S.REAL, marker="o", ls="", ms=5, label="after real use"), plt.Line2D([], [], color=S.EVAL, marker="o", ls="", ms=5, label="after evaluation")],
               loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=S.FS["legend"], handletextpad=0.1, columnspacing=1.0, borderaxespad=0, borderpad=0)
    for a_ in (ax1, ax2, ax3): a_.axis("off")
    ax1.set_xlim(*lim1); ax1.set_ylim(*ylim12); ax2.set_xlim(*lim2); ax2.set_ylim(*ylim12); ax3.set_xlim(-52, 106); ax3.set_ylim(-1.75, 2.4)
    heads = [(ax1, "1", "The regime offsets\nthe activations"), (ax2, "2", "Logits move by\nrandom projections"), (ax3, "3", "So the preferences\nreflect the regime")]
    for ax, num, t1 in heads:   # number bold, phrase plain weight with a hanging indent (the titles need two lines at the shared size)
        x = ax.get_position().x0
        fig.text(x, 1.0, num, fontsize=S.FS["title"], fontweight="bold", color=S.INK, va="top")
        fig.text(x + 0.17 / FW, 1.0, t1, fontsize=S.FS["title"], color=S.INK, va="top", linespacing=1.15)
    fig.canvas.draw(); yb = fig.transFigure.inverted().transform(low_label.get_window_extent(fig.canvas.get_renderer()))[0, 1]
    fig.text(ax2.get_position().x1, yb, "(Illustrative)", ha="right", va="bottom", fontsize=S.FS["note"], color=S.MUTED, style="italic")
    S.save(fig, out)
    print({k: np.round(v, 3).tolist() for k, v in sh.items()}, "BA", round(ba, 3), "nudges", nudge.round(2).tolist(),
          "panes (in)", [round(w, 2) for w in (w1, w2, w3)], "H", round(H, 2), "FH", round(FH, 2))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures/toy_schematic.png")
    a = ap.parse_args(); rng = np.random.default_rng(3)
    three(a.out, rng)


if __name__ == "__main__":
    main()
