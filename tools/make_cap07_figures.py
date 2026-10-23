"""Figures of chapter 7 of the book (geometric phases), Spanish and English, on the base of chapter 1.

    python tools/make_cap07_figures.py

writes book/es/figuras/cap07_*.png and book/en/figures/cap07_*.png. Same style as make_cap01_figures.py; the
phases come from qang.geometric, the interferometer is a binomial simulation with a fixed seed.
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
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402
from qang import geometric as G  # noqa: E402

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "cone_x": "altura del bucle $m$ (m-qg)",
        "cone_y": "fase geométrica (vueltas, mód. 1)",
        "law": r"$\varphi = p_0 = (1000+m)/2000$",
        "panch": "producto de solapamientos",
        "solid": "ángulo sólido",
        "flux": "flujo de curvatura (Stokes)",
        "int_x": r"fase geométrica verdadera $\gamma$ (grados)",
        "int_y": r"$\hat\gamma = 2\arccos\sqrt{\hat p_0}$ (grados)",
        "int_l": "{n} disparos por punto, media ± desvío (200 repeticiones)",
        "sd_y": r"desvío de $\hat\gamma$ (mrad)",
        "sd_l": r"$1/\sqrt{N}$",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "cone_x": "height of the loop $m$ (m-qg)",
        "cone_y": "geometric phase (turns, mod 1)",
        "law": r"$\varphi = p_0 = (1000+m)/2000$",
        "panch": "overlap product",
        "solid": "solid angle",
        "flux": "curvature flux (Stokes)",
        "int_x": r"true geometric phase $\gamma$ (degrees)",
        "int_y": r"$\hat\gamma = 2\arccos\sqrt{\hat p_0}$ (degrees)",
        "int_l": "{n} shots per point, mean ± sd (200 repetitions)",
        "sd_y": r"sd of $\hat\gamma$ (mrad)",
        "sd_l": r"$1/\sqrt{N}$",
    },
}


def cone(q, n=400):
    s = math.sqrt(1 - q * q)
    return [np.array([s * math.cos(a), s * math.sin(a), q]) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def fig_cone(t, lang):
    m = np.linspace(-1000, 1000, 201)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(m, (1000 + m) / 2000, color=INK, lw=2.5, label=t["law"])
    pts = np.arange(-900, 901, 150)
    pan = [(G.pancharatnam_phase([G.state_from_bloch(r) for r in cone(x / 1000)]) / (2 * math.pi)) % 1.0 for x in pts]
    sol = [(G.berry_phase_from_solid_angle(G.solid_angle(cone(x / 1000), reference=[0, 0, 1])) / (2 * math.pi)) % 1.0
           for x in pts]
    flx = [(-G.curvature_flux(x / 1000) / (2 * math.pi)) % 1.0 for x in pts]
    ax.plot(pts, pan, "o", color=BLUE, ms=8, label=t["panch"])
    ax.plot(pts, sol, "s", color=ORANGE, ms=5, label=t["solid"])
    ax.plot(pts, flx, "x", color=GREEN, ms=8, mew=2, label=t["flux"])
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(0, 1)
    ax.set_xlabel(t["cone_x"])
    ax.set_ylabel(t["cone_y"])
    top = ax.secondary_xaxis("top", functions=(lambda x: (1000 + x) / 2000, lambda p: 2000 * p - 1000))
    top.set_xlabel("$p_0$", color=GREY)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def fig_interferometer(t, lang, n=1000, reps=200):
    rng = np.random.default_rng(84)
    g = np.radians(np.arange(0, 181, 10))
    mean, sd = [], []
    for x in g:
        k = rng.binomial(n, math.cos(x / 2) ** 2, size=reps)
        est = 2 * np.arccos(np.sqrt(k / n))
        mean.append(est.mean())
        sd.append(est.std(ddof=1))
    mean, sd = np.array(mean), np.array(sd)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot([0, 180], [0, 180], color=GREY, lw=1)
    a1.errorbar(np.degrees(g), np.degrees(mean), yerr=np.degrees(sd), fmt="o", color=BLUE, ms=5, capsize=3,
                label=t["int_l"].format(n=n))
    a1.set_xlim(-5, 185)
    a1.set_ylim(-5, 185)
    a1.set_xticks(range(0, 181, 30))
    a1.set_yticks(range(0, 181, 30))
    a1.set_xlabel(t["int_x"])
    a1.set_ylabel(t["int_y"])
    a1.legend(frameon=False, loc="upper left", fontsize=9)
    a1.grid(alpha=0.25)
    a2.plot(np.degrees(g), 1000 * sd, "o-", color=BLUE, ms=5)
    a2.axhline(1000 / math.sqrt(n), color=ORANGE, lw=2, ls="--", label=t["sd_l"])
    a2.set_xlim(-5, 185)
    a2.set_ylim(0, None)
    a2.set_xticks(range(0, 181, 30))
    a2.set_xlabel(t["int_x"])
    a2.set_ylabel(t["sd_y"])
    a2.legend(frameon=False, loc="lower center")
    a2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


FIGURES = {"cap07_cono": fig_cone, "cap07_interferometro": fig_interferometer}


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
