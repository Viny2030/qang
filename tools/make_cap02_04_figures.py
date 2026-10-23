"""Figures of chapters 2-4 of the book, Spanish and English, on the base of chapter 1 (probability equation, m-qg).

    python tools/make_cap02_04_figures.py

writes book/es/figuras/cap0{2,3,4}_*.png and book/en/figures/cap0{2,3,4}_*.png. Reuses the style of
make_cap01_figures.py; every curve comes from qang.probability, qang.statistics or a closed form of the text.
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
from qang import probability as PB  # noqa: E402
from qang import statistics as ST  # noqa: E402

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "area_x": r"ángulo pedido de $R_x(\theta)$ (grados)",
        "area_y": r"corrimiento de $qg_Z$ (m-qg), $\sigma$ = 0,05",
        "area_naive": r"ángulo tal como se pide, $\theta$",
        "area_min": r"representante de área mínima, $\theta-2\pi$ si $\theta>\pi$",
        "grover_x": "iteraciones de Grover, $k$",
        "grover_y": r"sesgo de la bandera $qg_k$ (m-qg)",
        "grover_a": "fracción buscada $a$ = {a}",
        "grover_ok": "éxito: $qg_k$ cerca de −1000 m-qg",
        "haar_m": "lectura $m$ (m-qg)",
        "haar_t": r"ángulo qang $\theta$ (grados)",
        "haar_y": "densidad",
        "haar_title_m": "en mili-qang: uniforme",
        "haar_title_t": r"en ángulo: $\frac{1}{2}\sin\theta$",
        "int_x": "estimación $m$ (m-qg), $N$ = 50 disparos",
        "int_y": "ancho del intervalo de 95 % (m-qg)",
        "delta": "método delta (Wald)",
        "wilson": "Wilson",
        "bayes": "Bayes–Haar",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "area_x": r"requested angle of $R_x(\theta)$ (degrees)",
        "area_y": r"shift of $qg_Z$ (m-qg), $\sigma$ = 0.05",
        "area_naive": r"angle as requested, $\theta$",
        "area_min": r"minimal-area representative, $\theta-2\pi$ if $\theta>\pi$",
        "grover_x": "Grover iterations, $k$",
        "grover_y": r"flag bias $qg_k$ (m-qg)",
        "grover_a": "marked fraction $a$ = {a}",
        "grover_ok": "success: $qg_k$ near −1000 m-qg",
        "haar_m": "reading $m$ (m-qg)",
        "haar_t": r"qang angle $\theta$ (degrees)",
        "haar_y": "density",
        "haar_title_m": "in milli-qang: uniform",
        "haar_title_t": r"in angle: $\frac{1}{2}\sin\theta$",
        "int_x": "estimate $m$ (m-qg), $N$ = 50 shots",
        "int_y": "width of the 95 % interval (m-qg)",
        "delta": "delta method (Wald)",
        "wilson": "Wilson",
        "bayes": "Bayes–Haar",
    },
}


def fig_area(t, lang):
    deg = np.linspace(0, 360, 721)
    naive = [PB.pulse_error_milliqang(math.radians(d), 0.05) for d in deg]
    red = [PB.pulse_error_milliqang(math.radians(d) - (2 * math.pi if d > 180 else 0.0), 0.05) for d in deg]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(deg, naive, color=ORANGE, lw=2.5, label=t["area_naive"])
    ax.plot(deg, red, color=BLUE, lw=2.5, ls="--", label=t["area_min"])
    ax.axhline(0, color="#cccccc", lw=1)
    ax.axvline(180, color=GREY, lw=1, ls=":")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel(t["area_x"])
    ax.set_ylabel(t["area_y"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def fig_grover(t, lang):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhspan(-1000, -900, color=GREEN, alpha=0.12)
    ax.text(4.95, -955, t["grover_ok"], ha="right", va="center", fontsize=9, color=GREEN)
    for a, col in ((0.1, BLUE), (0.3, ORANGE)):
        tt = math.asin(math.sqrt(a))
        k = np.linspace(0, 5, 400)
        ax.plot(k, 1000 * np.cos((2 * k + 1) * 2 * tt), color=col, lw=1.2, alpha=0.6)
        ks = np.arange(0, 6)
        ax.plot(ks, 1000 * np.cos((2 * ks + 1) * 2 * tt), "o", color=col, ms=8,
                label=t["grover_a"].format(a=_num(a, lang)))
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-1050, 1050)
    ax.set_xlabel(t["grover_x"])
    ax.set_ylabel(t["grover_y"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def fig_haar(t, lang):
    rng = np.random.default_rng(15)
    v = rng.normal(size=(200_000, 2)) + 1j * rng.normal(size=(200_000, 2))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p0 = np.abs(v[:, 0]) ** 2
    m = 2000 * p0 - 1000
    th = np.degrees(2 * np.arccos(np.sqrt(p0)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.hist(m, bins=40, density=True, color=BLUE, alpha=0.75)
    a1.axhline(1 / 2000, color=INK, lw=1.5, ls="--")
    a1.set_xlabel(t["haar_m"])
    a1.set_ylabel(t["haar_y"])
    a1.set_title(t["haar_title_m"], fontsize=11)
    a1.set_xlim(-1000, 1000)
    a2.hist(th, bins=40, density=True, color=ORANGE, alpha=0.75)
    x = np.linspace(0, 180, 300)
    a2.plot(x, 0.5 * np.sin(np.radians(x)) * math.pi / 180, color=INK, lw=1.5, ls="--")
    a2.set_xlabel(t["haar_t"])
    a2.set_title(t["haar_title_t"], fontsize=11)
    a2.set_xlim(0, 180)
    a2.set_xticks(range(0, 181, 30))
    for a in (a1, a2):
        a.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_intervals(t, lang):
    n = 50
    ks = np.arange(0, n + 1)
    m = 1000 * (2 * ks / n - 1)
    rows = {
        "delta": [ST.delta_qg_estimate(int(k), n) for k in ks],
        "wilson": [ST.wilson_qg_estimate(int(k), n) for k in ks],
        "bayes": [ST.bayes_qg_estimate(int(k), n) for k in ks],
    }
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for key, col, ls in (("delta", ORANGE, "-"), ("wilson", BLUE, "-"), ("bayes", GREEN, "--")):
        ax.plot(m, [1000 * (e.high - e.low) for e in rows[key]], color=col, lw=2.5, ls=ls, marker="o", ms=3,
                label=t[key])
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(0, None)
    ax.set_xlabel(t["int_x"])
    ax.set_ylabel(t["int_y"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower center")
    fig.tight_layout()
    return fig


FIGURES = {
    "cap02_area": fig_area,
    "cap03_grover": fig_grover,
    "cap04_haar": fig_haar,
    "cap04_intervalos": fig_intervals,
}


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
