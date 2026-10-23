"""Figures of chapter 1 of the book (the qang probability equation and the milli-qang), Spanish and English.

Every curve is computed with qang.probability, so the figures and the text cite the same numbers.

    python tools/make_cap01_figures.py

writes book/es/figuras/cap01_*.png and book/en/figures/cap01_*.png.
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
from qang import probability as PB  # noqa: E402

BLUE, ORANGE, GREEN, GREY, INK = "#2b78d0", "#e8693a", "#1f9d74", "#9a9a9a", "#1e1e1e"

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "angle_box": "ángulo a partir de la probabilidad medida",
        "z_box": "lectura lineal e insesgada",
        "s_box": "entropía del resultado",
        "m_box": "la misma lectura en mili-qang",
        "readings_x": "probabilidad del resultado 0, $p_0$",
        "readings_y": "lectura (qg)",
        "readings_m": "mili-qang (m-qg)",
        "theta_lbl": r"$\theta/\pi$ (ángulo qang)",
        "ruler_title": "La regla del mili-qang: lineal en $p_0$, no lineal en el ángulo",
        "ruler_theta": r"$\theta$ (grados)",
        "noise_x": r"ángulo qang $\theta$ (grados)",
        "noise_y": "desviación estándar",
        "noise_m": r"$\sigma_m$ (m-qg), $N$ = {n}",
        "noise_t": r"$\sigma_\theta$ (mrad), $N$ = {n}",
        "pulse_x": r"ángulo de la compuerta $R_x(\theta)$ (grados)",
        "pulse_y": r"corrimiento de $qg_Z$ (m-qg)",
        "pulse_l": r"$\sigma$ = {s}",
        "pulse_x_gate": "compuerta X, σ = 0,05:\n12,3 m-qg",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "angle_box": "angle from a measured probability",
        "z_box": "linear, unbiased read-out",
        "s_box": "outcome entropy",
        "m_box": "the same read-out in milli-qang",
        "readings_x": "probability of outcome 0, $p_0$",
        "readings_y": "reading (qg)",
        "readings_m": "milli-qang (m-qg)",
        "theta_lbl": r"$\theta/\pi$ (qang angle)",
        "ruler_title": "The milli-qang ruler: linear in $p_0$, not linear in the angle",
        "ruler_theta": r"$\theta$ (degrees)",
        "noise_x": r"qang angle $\theta$ (degrees)",
        "noise_y": "standard deviation",
        "noise_m": r"$\sigma_m$ (m-qg), $N$ = {n}",
        "noise_t": r"$\sigma_\theta$ (mrad), $N$ = {n}",
        "pulse_x": r"angle of the gate $R_x(\theta)$ (degrees)",
        "pulse_y": r"shift of $qg_Z$ (m-qg)",
        "pulse_l": r"$\sigma$ = {s}",
        "pulse_x_gate": "X gate, σ = 0.05:\n12.3 m-qg",
    },
}


def _num(x, lang, fmt="{:g}"):
    s = fmt.format(x)
    return s.replace(".", ",") if lang == "es" else s


def fig_circle(t, lang):
    p0 = 0.75
    r = PB.readings_from_p0(p0)
    th = r["theta"]
    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_axes([0.0, 0.0, 0.48, 1.0])
    ax.set_aspect("equal")
    ax.axis("off")
    phi = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(phi), np.sin(phi), color="#555555", lw=2)
    ax.plot([-1.25, 1.25], [0, 0], color="#dddddd", lw=1)
    ax.plot([0, 0], [-1.25, 1.25], color="#dddddd", lw=1)
    x, y = math.sin(th), math.cos(th)
    ax.plot([0, x], [0, y], color=INK, lw=3)
    ax.plot([x], [y], "o", color=BLUE, ms=10)
    ax.plot([0, x], [y, y], "--", color=BLUE, lw=2)
    arc = np.linspace(0, th, 60)
    ax.plot(0.25 * np.sin(arc), 0.25 * np.cos(arc), color=INK, lw=1.5)
    ax.text(0.19, 0.3, r"$\theta$", fontsize=16)
    ax.text(0, 1.12, r"$|0\rangle$", ha="center", fontsize=14)
    ax.text(0, -1.22, r"$|1\rangle$", ha="center", fontsize=14)
    ax.text(-0.06, y, r"$qg_Z=\cos\theta$", ha="right", va="center", fontsize=12, weight="bold")
    # milli-qang and probability scales on the vertical axis
    here = round(r["milliqang"])
    for m in (-1000, -500, 0, 500, 1000):
        yy = m / 1000
        col, w = (BLUE, "bold") if m == here else (INK, "normal")
        ax.plot([1.38, 1.44], [yy, yy], color=INK, lw=1)
        ax.text(1.48, yy, f"{m:+d}" if m else "0", va="center", fontsize=10, color=col, weight=w)
        ax.text(1.98, yy, _num(PB.p0_from_milliqang(m), lang), va="center", fontsize=9,
                color=BLUE if m == here else GREY, weight=w)
    ax.plot([1.41, 1.41], [-1, 1], color=INK, lw=1)
    ax.text(1.41, 1.17, "m-qg", ha="left", fontsize=10, weight="bold")
    ax.text(1.98, 1.17, "$p_0$", ha="left", fontsize=10, color=GREY)
    ax.annotate("", xy=(1.40, y), xytext=(x + 0.03, y), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.set_xlim(-1.35, 2.3)
    ax.set_ylim(-1.35, 1.35)

    boxes = [
        (r"$\theta = 2\,\arccos\sqrt{p_0}$", t["angle_box"], BLUE),
        (r"$qg_Z = 2p_0 - 1$", t["z_box"], ORANGE),
        (r"$qg_S = H(p_0)$", t["s_box"], GREEN),
        (r"$m = 2000\,p_0 - 1000$", t["m_box"], INK),
    ]
    for i, (eq, sub, col) in enumerate(boxes):
        bx = fig.add_axes([0.53, 0.77 - i * 0.235, 0.42, 0.19])
        bx.set_xticks([])
        bx.set_yticks([])
        for s in bx.spines.values():
            s.set_edgecolor(col)
            s.set_linewidth(2)
        bx.text(0.04, 0.62, eq, fontsize=17, va="center")
        bx.text(0.04, 0.22, sub, fontsize=11, color="#555555", va="center")
    return fig


def fig_readings(t, lang):
    p = np.linspace(0, 1, 801)
    rd = [PB.readings_from_p0(float(x)) for x in p]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(p, [r["theta"] / math.pi for r in rd], color=BLUE, lw=2.5, label=t["theta_lbl"])
    ax.plot(p, [r["qg_z"] for r in rd], color=ORANGE, lw=2.5, label=r"$qg_Z = 2p_0-1$")
    ax.plot(p, [r["qg_s"] for r in rd], color=GREEN, lw=2.5, label=r"$qg_S = H(p_0)$")
    ax.axhline(0, color="#cccccc", lw=1)
    ax.axvline(0.5, color="#cccccc", lw=1, ls=":")
    ax.set_xlabel(t["readings_x"])
    ax.set_ylabel(t["readings_y"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 1.05)
    sec = ax.secondary_yaxis("right", functions=(lambda q: 1000 * q, lambda m: m / 1000))
    sec.set_ylabel(t["readings_m"], color=ORANGE)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_ruler(t, lang):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    rows = {"m": 2.0, "p": 1.0, "t": 0.0}
    for yy in rows.values():
        ax.plot([-1000, 1000], [yy, yy], color=INK, lw=1.5)
    for m in range(-1000, 1001, 250):
        ax.plot([m, m], [rows["m"] - 0.08, rows["m"] + 0.08], color=INK)
        ax.text(m, rows["m"] + 0.18, f"{m:+d}" if m else "0", ha="center", fontsize=9)
        p0 = PB.p0_from_milliqang(m)
        ax.plot([m, m], [rows["p"] - 0.08, rows["p"] + 0.08], color=GREY)
        ax.text(m, rows["p"] + 0.18, _num(p0, lang), ha="center", fontsize=9, color="#555555")
    for deg in range(180, -1, -15):
        m = PB.milliqang_from_theta(math.radians(deg))
        big = deg % 30 == 0
        ax.plot([m, m], [rows["t"] - (0.1 if big else 0.05), rows["t"] + (0.1 if big else 0.05)], color=BLUE)
        if big:
            ax.text(m, rows["t"] - 0.32, f"{deg}°", ha="center", fontsize=9, color=BLUE)
    ax.text(-1080, rows["m"], "m-qg", ha="right", va="center", weight="bold")
    ax.text(-1080, rows["p"], "$p_0$", ha="right", va="center", color="#555555")
    ax.text(-1080, rows["t"], t["ruler_theta"], ha="right", va="center", color=BLUE)
    ax.set_title(t["ruler_title"], fontsize=12)
    ax.set_xlim(-1300, 1060)
    ax.set_ylim(-0.6, 2.5)
    fig.tight_layout()
    return fig


def fig_noise(t, lang):
    deg = np.linspace(0.5, 179.5, 500)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for n, col in zip((100, 10_000, 1_000_000), (ORANGE, BLUE, GREEN)):
        ax.plot(deg, [PB.shot_noise_milliqang(math.radians(d), n) for d in deg], color=col, lw=2.5,
                label=t["noise_m"].format(n=_num(n, lang, "{:,}").replace(",", ".") if lang == "es" else f"{n:,}"))
        ax.axhline(1000 / math.sqrt(n), color=col, lw=1.2, ls="--",
                   label=t["noise_t"].format(n=_num(n, lang, "{:,}").replace(",", ".") if lang == "es" else f"{n:,}"))
    ax.set_yscale("log")
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xlabel(t["noise_x"])
    ax.set_ylabel(t["noise_y"])
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center")
    ax.set_ylim(0.01, 200)
    fig.tight_layout()
    return fig


def fig_pulse(t, lang):
    deg = np.linspace(0, 180, 361)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for s, col in zip((0.01, 0.02, 0.05, 0.1), (GREEN, BLUE, ORANGE, INK)):
        ax.plot(deg, [PB.pulse_error_milliqang(math.radians(d), s) for d in deg], color=col, lw=2.5,
                label=t["pulse_l"].format(s=_num(s, lang)))
    e = PB.pulse_error_milliqang(math.pi, 0.05)
    ax.plot([180], [e], "o", color=ORANGE, ms=8)
    ax.annotate(t["pulse_x_gate"], xy=(180, e), xytext=(25, 22), arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.axhline(0, color="#cccccc", lw=1)
    ax.set_xlim(0, 182)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xlabel(t["pulse_x"])
    ax.set_ylabel(t["pulse_y"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


FIGURES = {
    "cap01_circulo": fig_circle,
    "cap01_lecturas": fig_readings,
    "cap01_regla": fig_ruler,
    "cap01_ruido": fig_noise,
    "cap01_pulso": fig_pulse,
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
