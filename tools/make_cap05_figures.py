"""Figures of chapter 5 of the book (the local Bloch radius), Spanish and English, on the base of chapter 1.

    python tools/make_cap05_figures.py

writes book/es/figuras/cap05_*.png and book/en/figures/cap05_*.png. Same style as make_cap01_figures.py; the
curves come from qang.formulation and qang.complementarity.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE, _num  # noqa: E402
from qang import complementarity as C  # noqa: E402
from qang import formulation as F  # noqa: E402

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "dir_x": r"dirección verdadera $\theta$ (grados)",
        "dir_y": "ángulo leído (grados)",
        "dir_sep": r"separado del radio, $\arccos(qg_Z/r)$, cualquier $r$",
        "dir_naive": r"$\arccos(qg_Z)$ con $r$ = {r}",
        "def_x": "lectura $m$ de un qubit de peso definido (m-qg)",
        "def_y": r"déficit $1-r^2 = 1-(m/1000)^2$",
        "def_p": "$p_0$",
        "def_a1": "990 m-qg:\ndéficit 0,02",
        "def_a2": "10 % de disparos mal ubicados\n(800 m-qg): déficit 0,36",
        "dual_x": r"visibilidad $V=\sqrt{qg_X^2+qg_Y^2}$",
        "dual_y": r"predictibilidad $P=|qg_Z|$",
        "dual_r": "$r$ = {r}",
        "dual_pt": "$\\theta=60^\\circ$, $r=0{,}9$:\n$P^2+V^2=0{,}81$",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "dir_x": r"true direction $\theta$ (degrees)",
        "dir_y": "angle read (degrees)",
        "dir_sep": r"separated from the radius, $\arccos(qg_Z/r)$, any $r$",
        "dir_naive": r"$\arccos(qg_Z)$ with $r$ = {r}",
        "def_x": "reading $m$ of a definite-weight qubit (m-qg)",
        "def_y": r"deficit $1-r^2 = 1-(m/1000)^2$",
        "def_p": "$p_0$",
        "def_a1": "990 m-qg:\ndeficit 0.02",
        "def_a2": "10 % of shots misplaced\n(800 m-qg): deficit 0.36",
        "dual_x": r"visibility $V=\sqrt{qg_X^2+qg_Y^2}$",
        "dual_y": r"predictability $P=|qg_Z|$",
        "dual_r": "$r$ = {r}",
        "dual_pt": "$\\theta=60^\\circ$, $r=0.9$:\n$P^2+V^2=0.81$",
    },
}


def fig_direction(t, lang):
    deg = np.linspace(0, 180, 361)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(deg, deg, color=INK, lw=2.5, label=t["dir_sep"])
    for r, col in ((0.8, BLUE), (0.5, ORANGE), (0.2, GREEN)):
        ax.plot(deg, [math.degrees(math.acos(r * math.cos(math.radians(d)))) for d in deg], color=col, lw=2.2,
                ls="--", label=t["dir_naive"].format(r=_num(r, lang)))
    ax.axhline(90, color="#cccccc", lw=1)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_yticks(range(0, 181, 30))
    ax.set_xlabel(t["dir_x"])
    ax.set_ylabel(t["dir_y"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def fig_deficit(t, lang):
    m = np.linspace(-1000, 1000, 801)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(m, F.weight_sector_radius(m / 1000), color=BLUE, lw=2.5)
    for mm, key, xy in ((990, "def_a1", (560, 0.05)), (800, "def_a2", (80, 0.42))):
        d = F.weight_sector_radius([mm / 1000])[0]
        ax.plot([mm], [d], "o", color=ORANGE, ms=8)
        ax.annotate(t[key], xy=(mm, d), xytext=xy, arrowprops=dict(arrowstyle="->", color=ORANGE), fontsize=9)
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(t["def_x"])
    ax.set_ylabel(t["def_y"])
    top = ax.secondary_xaxis("top", functions=(lambda x: (1000 + x) / 2000, lambda p: 2000 * p - 1000))
    top.set_xlabel(t["def_p"], color=GREY)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_duality(t, lang):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    phi = np.linspace(0, math.pi / 2, 200)
    for r, col in ((1.0, INK), (0.9, BLUE), (0.5, GREY)):
        ax.plot(r * np.cos(phi), r * np.sin(phi), color=col, lw=2 if r == 1 else 1.5,
                label=t["dual_r"].format(r=_num(r, lang)))
    q, x = 0.9 * math.cos(math.radians(60)), 0.9 * math.sin(math.radians(60))
    p, v = C.predictability(q), C.visibility(x)
    ax.plot([v], [p], "o", color=ORANGE, ms=9)
    ax.annotate(t["dual_pt"], xy=(v, p), xytext=(0.12, 0.2), arrowprops=dict(arrowstyle="->", color=ORANGE),
                fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(t["dual_x"])
    ax.set_ylabel(t["dual_y"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


FIGURES = {"cap05_direccion": fig_direction, "cap05_deficit": fig_deficit, "cap05_dualidad": fig_duality}


def main():
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        for name, make in FIGURES.items():
            fig = make(t, lang)
            path = os.path.join(t["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print(path)


if __name__ == "__main__":
    main()
