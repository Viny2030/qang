"""Figures of chapter 8 of the book (Fisher information, clocks, metrology), Spanish and English, on the chapter 1 base.

    python tools/make_cap08_figures.py

writes book/es/figuras/cap08_*.png and book/en/figures/cap08_*.png. Same style as make_cap01_figures.py. The Fisher
curves are the closed forms of §167 and §36; the clock ellipse is a binomial simulation with a fixed seed (§126); the
GHZ gains are the §58 rates, maximized over the interrogation time on a grid.
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

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "fi_x": r"lectura $qg_Z$ (o $m/1000$)",
        "fi_y": "información de Fisher por disparo",
        "fi_theta": r"en el ángulo $\theta$: 1 (plana)",
        "fi_qg": r"en $qg_Z$: $1/(1-qg_Z^2)$ (curva)",
        "fi_ramsey": r"Ramsey, fase, visibilidad $V$: $(V^2-qg^2)/(1-qg^2)$",
        "el_x": r"$qg_{Z,1}$ (reloj de abajo)",
        "el_y": r"$qg_{Z,2}$ (reloj de arriba)",
        "el_pts": "corridas simuladas, 1000 disparos cada una",
        "el_ideal": r"elipse ideal, $2\delta = 0{,}294$ rad (Sr, 1 m, 1 s)",
        "el_none": r"sin diferencia ($\delta = 0$): la diagonal",
        "g_x": "número de qubits $N$",
        "g_y": "ganancia GHZ / independientes (tasa de Fisher)",
        "g_markov": "desfase markoviano",
        "g_gauss": "desfase gaussiano (lento)",
        "g_e": "error de lectura $e$ = {e}",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "fi_x": r"reading $qg_Z$ (or $m/1000$)",
        "fi_y": "Fisher information per shot",
        "fi_theta": r"in the angle $\theta$: 1 (flat)",
        "fi_qg": r"in $qg_Z$: $1/(1-qg_Z^2)$ (curved)",
        "fi_ramsey": r"Ramsey phase, visibility $V$: $(V^2-qg^2)/(1-qg^2)$",
        "el_x": r"$qg_{Z,1}$ (lower clock)",
        "el_y": r"$qg_{Z,2}$ (upper clock)",
        "el_pts": "simulated runs, 1000 shots each",
        "el_ideal": r"ideal ellipse, $2\delta = 0.294$ rad (Sr, 1 m, 1 s)",
        "el_none": r"no difference ($\delta = 0$): the diagonal",
        "g_x": "number of qubits $N$",
        "g_y": "GHZ / independent gain (Fisher rate)",
        "g_markov": "Markovian dephasing",
        "g_gauss": "Gaussian (slow) dephasing",
        "g_e": "readout error $e$ = {e}",
    },
}


def fig_fisher(t, lang):
    q = np.linspace(-0.995, 0.995, 600)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(q, np.ones_like(q), color=BLUE, lw=2.2, label=t["fi_theta"])
    ax.plot(q, 1 / (1 - q ** 2), color=ORANGE, lw=2.2, label=t["fi_qg"])
    for V, ls in ((0.95, "-"), (0.8, "--")):
        qq = q[np.abs(q) < V]
        ax.plot(qq, (V ** 2 - qq ** 2) / (1 - qq ** 2), color=GREEN, lw=1.8, ls=ls,
                label=t["fi_ramsey"] + f", V = {V}".replace(".", "," if lang == "es" else "."))
    ax.set_ylim(0, 4)
    ax.set_xlabel(t["fi_x"], color=INK)
    ax.set_ylabel(t["fi_y"], color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper center")
    fig.tight_layout()
    return fig


def fig_clocks(t, lang, n=1000, runs=200, delta=0.147):
    rng = np.random.default_rng(126)
    phi = rng.uniform(0, 2 * math.pi, runs)
    p1, p2 = (1 + np.cos(phi + delta)) / 2, (1 + np.cos(phi - delta)) / 2
    x = 2 * rng.binomial(n, p1) / n - 1
    y = 2 * rng.binomial(n, p2) / n - 1
    s = np.linspace(0, 2 * math.pi, 400)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([-1, 1], [-1, 1], color=GREY, lw=1, ls=":", label=t["el_none"])
    ax.plot(np.cos(s + delta), np.cos(s - delta), color=BLUE, lw=2, label=t["el_ideal"])
    ax.scatter(x, y, s=9, color=ORANGE, alpha=0.75, label=t["el_pts"])
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel(t["el_x"], color=INK)
    ax.set_ylabel(t["el_y"], color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def ghz_gain(N, e, kind):
    ts = np.linspace(1e-3, 3, 6000)
    if kind == "markov":
        prod = N * ts * np.exp(-2 * ts) * (1 - 2 * e) ** 2
        ghz = N ** 2 * ts * np.exp(-2 * N * ts) * (1 - 2 * e) ** (2 * N)
    else:
        prod = N * ts * np.exp(-2 * ts ** 2) * (1 - 2 * e) ** 2
        ghz = N ** 2 * ts * np.exp(-2 * N * ts ** 2) * (1 - 2 * e) ** (2 * N)
    return float(ghz.max() / prod.max())


def fig_ghz(t, lang):
    Ns = np.unique(np.round(np.logspace(0.3, 2.5, 40)).astype(int))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for kind, color in (("markov", GREY), ("gauss", BLUE)):
        for e, ls in ((0.0, "-"), (0.002, "--"), (0.01, ":")):
            g = [ghz_gain(N, e, kind) for N in Ns]
            label = (t["g_markov"] if kind == "markov" else t["g_gauss"]) + ", " + t["g_e"].format(e=e)
            ax.loglog(Ns, g, color=color, ls=ls, lw=2 if e == 0 else 1.6, label=label.replace(".", ",")
                      if lang == "es" else label)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_ylim(1e-2, 30)
    ax.set_xlabel(t["g_x"], color=INK)
    ax.set_ylabel(t["g_y"], color=INK)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


def main():
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        for name, fn in (("fisher", fig_fisher), ("relojes" if lang == "es" else "clocks", fig_clocks),
                         ("ghz", fig_ghz)):
            fig = fn(t, lang)
            path = os.path.join(t["dir"], f"cap08_{name}.png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", path)
    # the §58 table, recomputed, for the text
    for N in (2, 10, 30, 100):
        print(N, [round(ghz_gain(N, e, k), 3) for k in ("markov", "gauss") for e in (0, 0.002, 0.01)])


if __name__ == "__main__":
    main()
