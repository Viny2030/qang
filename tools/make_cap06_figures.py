"""Figures of chapter 6 of the book (polarization in qg units), Spanish and English, on the base of chapter 1.

    python tools/make_cap06_figures.py

writes book/es/figuras/cap06_*.png and book/en/figures/cap06_*.png. Same style as make_cap01_figures.py; the
curves come from qang.polarization. The degree-of-polarization figure is a Monte Carlo with a fixed seed.
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
from qang import polarization as PZ  # noqa: E402

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "malus_x": r"ángulo del analizador $\vartheta$ (grados)",
        "malus_y": r"transmisión $p_0 = I/I_0$",
        "malus_m": "lectura (m-qg)",
        "H": "luz H: $\\cos^2\\vartheta$",
        "D": "luz D: $\\cos^2(\\vartheta-45^\\circ)$",
        "part": "parcialmente polarizada, $P=0{,}67$",
        "unpol": "no polarizada",
        "ang_title": "En la esfera de Poincaré los ángulos se duplican",
        "ang_note": "polarización lineal a $\\alpha$ = punto del ecuador a $2\\alpha$",
        "deg_x": "fotones por base, $N$",
        "deg_y": r"grado de polarización estimado $\hat P$ (promedio)",
        "plug": "directo, $P$ = {p}",
        "unb": "con cuadrados insesgados, $P$ = {p}",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "malus_x": r"analyser angle $\vartheta$ (degrees)",
        "malus_y": r"transmission $p_0 = I/I_0$",
        "malus_m": "reading (m-qg)",
        "H": "H light: $\\cos^2\\vartheta$",
        "D": "D light: $\\cos^2(\\vartheta-45^\\circ)$",
        "part": "partially polarized, $P=0.67$",
        "unpol": "unpolarized",
        "ang_title": "On the Poincaré sphere angles double",
        "ang_note": "linear polarization at $\\alpha$ = equator point at $2\\alpha$",
        "deg_x": "photons per basis, $N$",
        "deg_y": r"estimated degree of polarization $\hat P$ (mean)",
        "plug": "plug-in, $P$ = {p}",
        "unb": "unbiased squares, $P$ = {p}",
    },
}

PARTIAL = {"Z": 0.6, "X": 0.3, "Y": 0.0}


def fig_malus(t, lang):
    deg = np.linspace(0, 180, 361)
    th = np.radians(deg)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for qg, col, key in (({"Z": 1.0}, INK, "H"), ({"X": 1.0}, BLUE, "D"), (PARTIAL, ORANGE, "part"), ({}, GREY, "unpol")):
        ax.plot(deg, [PZ.malus_intensity(qg, x) for x in th], color=col, lw=2.5, label=t[key])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xlabel(t["malus_x"])
    ax.set_ylabel(t["malus_y"])
    sec = ax.secondary_yaxis("right", functions=(lambda p: 2000 * p - 1000, lambda m: (m + 1000) / 2000))
    sec.set_ylabel(t["malus_m"], color=GREY)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


def fig_angles(t, lang):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")
    phi = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(phi), np.sin(phi), color="#555555", lw=2)
    ax.plot([-1.3, 1.3], [0, 0], color="#dddddd", lw=1)
    ax.plot([0, 0], [-1.3, 1.3], color="#dddddd", lw=1)
    labels = {0: "H", 45: "D", 90: "V", 135: "A"}
    for alpha in range(0, 180, 15):
        a2 = math.radians(2 * alpha)
        x, y = math.cos(a2), math.sin(a2)
        big = alpha in labels
        ax.plot([x], [y], "o", color=BLUE if big else GREY, ms=7 if big else 4)
        # glyph: the physical polarization direction alpha, drawn just outside the circle
        gx, gy = 1.22 * x, 1.22 * y
        dx, dy = 0.09 * math.cos(math.radians(alpha)), 0.09 * math.sin(math.radians(alpha))
        ax.plot([gx - dx, gx + dx], [gy - dy, gy + dy], color=ORANGE, lw=2.2)
        if big:
            ax.text(1.5 * x, 1.45 * y, f"{labels[alpha]} ({alpha}°)", ha="center", va="center", fontsize=10)
    ax.text(0.5, 0.06, r"$qg_Z=S_1/S_0$", fontsize=10, color=GREY)
    ax.text(0.05, 0.6, r"$qg_X=S_2/S_0$", fontsize=10, color=GREY)
    ax.set_title(t["ang_title"], fontsize=12)
    ax.text(0, -1.62, t["ang_note"], ha="center", fontsize=10)
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.75, 1.6)
    return fig


def _mean_p(P, n, unbiased, rng, reps=4000):
    vals = np.empty(reps)
    for i in range(reps):
        s = 0.0
        for q in (P, 0.0, 0.0):
            x = 2 * rng.binomial(n, (1 + q) / 2) / n - 1
            s += (n * x * x - 1) / (n - 1) if unbiased else x * x
        vals[i] = math.sqrt(max(0.0, s))
    return vals.mean()


def fig_degree(t, lang):
    rng = np.random.default_rng(83)
    ns = np.unique(np.round(np.logspace(1, 3.5, 12)).astype(int))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for P, col in ((0.0, ORANGE), (0.5, BLUE)):
        ax.plot(ns, [_mean_p(P, int(n), False, rng) for n in ns], color=col, lw=2.2, marker="o", ms=4,
                label=t["plug"].format(p=_num(P, lang)))
        ax.plot(ns, [_mean_p(P, int(n), True, rng) for n in ns], color=col, lw=2.2, ls="--", marker="s", ms=4,
                label=t["unb"].format(p=_num(P, lang)))
        ax.axhline(P, color=col, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(t["deg_x"])
    ax.set_ylabel(t["deg_y"])
    ax.set_ylim(0, 0.75)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


FIGURES = {"cap06_malus": fig_malus, "cap06_angulos": fig_angles, "cap06_grado": fig_degree}


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
