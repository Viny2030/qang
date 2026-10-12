"""
From the unit circle to qang: one figure for the README.

Left: the X-Z great circle of the Bloch sphere drawn as the unit circle. A
state at polar angle theta (measured from +Z) projects to qg_Z = cos(theta)
on the vertical axis and qg_X = sin(theta) on the horizontal one. Right: the
three quantities read from the same angle, qg_Z = cos(theta), the outcome
probability p0 = cos^2(theta/2) = (1 + qg_Z)/2 (the half angle of Born's rule)
and the entropic qang qg_S = H(p0).

python examples/unit_circle_qg.py   ->   examples/unit_circle_qg.png
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
THETA0 = math.pi / 3


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main(out=os.path.join(HERE, "unit_circle_qg.png")):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)

    # left: unit circle in the X-Z plane, theta measured from +Z
    t = np.linspace(0, 2 * np.pi, 400)
    a.plot(np.sin(t), np.cos(t), color=MUTED, lw=1.2)
    a.axhline(0, color=GRID, lw=1)
    a.axvline(0, color=GRID, lw=1)
    x0, z0 = math.sin(THETA0), math.cos(THETA0)
    a.plot([0, x0], [0, z0], color=INK, lw=2)
    a.plot([x0], [z0], "o", color=INK, ms=8, mec=SURFACE, mew=2)
    a.plot([x0, 0], [z0, z0], color=BLUE, lw=2, ls="--")
    a.plot([x0, x0], [z0, 0], color=ORANGE, lw=2, ls=":")
    arc = np.linspace(0, THETA0, 50)
    a.plot(0.28 * np.sin(arc), 0.28 * np.cos(arc), color=INK, lw=1.2)
    a.text(0.14, 0.36, "θ", fontsize=13, color=INK)
    a.text(-0.06, z0, f"qg_Z = cos θ = {z0:.2f}", ha="right", va="center", fontsize=10, color=INK)
    a.text(x0, -0.1, f"qg_X = sin θ = {x0:.2f}", ha="center", va="top", fontsize=10, color=INK)
    a.text(0, 1.1, "|0⟩  (+Z)", ha="center", fontsize=10, color=MUTED)
    a.text(0, -1.17, "|1⟩  (−Z)", ha="center", fontsize=10, color=MUTED)
    a.set_xlim(-1.35, 1.35)
    a.set_ylim(-1.3, 1.3)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("The angle on the circle", fontsize=12, color=INK, loc="left")

    # right: what qang reads from the same angle
    th = np.linspace(0, np.pi, 400)
    p0 = np.cos(th / 2) ** 2
    curves = [(np.cos(th), BLUE, "-", "qg_Z = cos θ"),
              (p0, ORANGE, "--", "p₀ = cos²(θ/2)"),
              (h2(p0), AQUA, "-.", "qg_S = H(p₀)")]
    for y, c, ls, lab in curves:
        b.plot(th, y, color=c, lw=2, ls=ls, label=lab)
    b.axvline(THETA0, color=GRID, lw=1)
    for y, c, _, _ in curves:
        b.plot([THETA0], [np.interp(THETA0, th, y)], "o", color=c, ms=8, mec=SURFACE, mew=2)
    # direct labels at the right end
    b.text(np.pi + 0.05, -1.0, "qg_Z", color=INK, va="center", fontsize=10)
    b.text(np.pi + 0.05, 0.0, "p₀", color=INK, va="center", fontsize=10)
    b.text(np.pi + 0.05, 0.12, "qg_S", color=INK, va="center", fontsize=10)
    b.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], ["0", "π/4", "π/2", "3π/4", "π"])
    b.set_xlim(0, np.pi + 0.45)
    b.set_ylim(-1.08, 1.08)
    b.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        b.spines[s].set_color(MUTED)
    b.tick_params(colors=MUTED)
    b.set_xlabel("polar angle θ", color=INK)
    b.legend(loc="lower left", frameon=False, fontsize=10)
    b.set_title("What qang reads from it (half angle in p₀)", fontsize=12, color=INK, loc="left")

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


if __name__ == "__main__":
    print(main())
