"""Figures of chapter 9 of the book (flavour oscillation in vacuum), Spanish and English, on the chapter 1 base.

    python tools/make_cap09_figures.py

writes book/es/figuras/cap09_*.png and book/en/figures/cap09_*.png. Same style as make_cap01_figures.py. The curves
are qang.neutrino in closed form (§151); the hardware points are the saved IQM Garnet run of §165
(examples/data/iqm_neutrino_qg_run.json), not a new job.
"""

import json
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
from qang import neutrino as nu  # noqa: E402

L_KM, DELTA_CP = 1300.0, -math.pi / 2
COLORS = {"e": BLUE, "mu": ORANGE, "tau": GREEN}
NAMES = {"e": r"$\nu_e$", "mu": r"$\nu_\mu$", "tau": r"$\nu_\tau$"}

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "c_x": r"fase de oscilación $\Delta/\pi$",
        "c_m": r"lectura del qubit de sabor, $m = 1000\,qg_Z$ (m-qg)",
        "c_a": r"ángulo qang $\vartheta=\arccos qg_Z$ (rad)",
        "c_band": r"$\pm\sigma_m$ con $N = 1000$",
        "c_4t": r"$4\theta$ en el máximo",
        "s_x": "energía $E$ (GeV), $L$ = 1300 km",
        "s_y": r"$P(\nu_\mu\to\nu_\beta)$",
        "s_m": r"lectura del qubit $\beta$, $m = 1000(1-2P)$ (m-qg)",
        "s_anti": "antineutrino (trazos)",
        "s_max": "primer máximo, 2,63 GeV",
        "g_x": "energía $E$ (GeV), $L$ = 1300 km",
        "g_y": r"$P(\nu_\mu\to\nu_\beta)$",
        "g_exact": "exacta",
        "g_raw": "Garnet, crudo",
        "g_filt": "Garnet, con filtro qg",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "c_x": r"oscillation phase $\Delta/\pi$",
        "c_m": r"flavour-qubit reading, $m = 1000\,qg_Z$ (m-qg)",
        "c_a": r"qang angle $\vartheta=\arccos qg_Z$ (rad)",
        "c_band": r"$\pm\sigma_m$ with $N = 1000$",
        "c_4t": r"$4\theta$ at the maximum",
        "s_x": "energy $E$ (GeV), $L$ = 1300 km",
        "s_y": r"$P(\nu_\mu\to\nu_\beta)$",
        "s_m": r"reading of qubit $\beta$, $m = 1000(1-2P)$ (m-qg)",
        "s_anti": "antineutrino (dashed)",
        "s_max": "first maximum, 2.63 GeV",
        "g_x": "energy $E$ (GeV), $L$ = 1300 km",
        "g_y": r"$P(\nu_\mu\to\nu_\beta)$",
        "g_exact": "exact",
        "g_raw": "Garnet, raw",
        "g_filt": "Garnet, qg filter",
    },
}


def dec(s, lang):
    return s.replace(".", ",") if lang == "es" else s


def fig_cone(t, lang, n=1000):
    d = np.linspace(0, 2 * math.pi, 600)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for theta, color in ((0.1, BLUE), (0.3, ORANGE), (math.pi / 4, GREEN)):
        q = np.cos(2 * theta) ** 2 + np.sin(2 * theta) ** 2 * np.cos(d)
        m, s = 1000 * q, 1000 * np.sqrt(1 - q ** 2) / math.sqrt(n)
        lbl = r"$\theta$ = " + (r"$\pi/4$" if theta == math.pi / 4 else dec(f"{theta}", lang))
        a1.plot(d / math.pi, m, color=color, lw=2, label=lbl)
        a1.fill_between(d / math.pi, m - s, m + s, color=color, alpha=0.18, lw=0)
        a2.plot(d / math.pi, np.arccos(np.clip(q, -1, 1)), color=color, lw=2, label=lbl)
        a2.plot([1], [4 * theta], "o", color=color, ms=6)
    a1.fill_between([], [], [], color=GREY, alpha=0.3, label=t["c_band"])
    a2.plot([], [], "o", color=INK, ms=6, label=t["c_4t"])
    for ax, yl in ((a1, t["c_m"]), (a2, t["c_a"])):
        ax.set_xlabel(t["c_x"], color=INK)
        ax.set_ylabel(yl, color=INK)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, loc="center" if ax is a1 else "upper left")
    a1.set_ylim(-1050, 1050)
    a2.set_ylim(0, math.pi + 0.1)
    fig.tight_layout()
    return fig


def fig_spectrum(t, lang):
    E = np.linspace(0.7, 6, 500)
    P = np.array([nu.three_flavour_probabilities(L_KM, e, DELTA_CP)[1] for e in E])
    Pb = np.array([nu.three_flavour_probabilities(L_KM, e, DELTA_CP, True)[1] for e in E])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for k, f in enumerate(nu.FLAVOURS):
        ax.plot(E, P[:, k], color=COLORS[f], lw=2, label=NAMES[f])
        ax.plot(E, Pb[:, k], color=COLORS[f], lw=1.4, ls="--")
    ax.plot([], [], color=GREY, ls="--", label=t["s_anti"])
    emax = nu.first_maximum_energy(L_KM)
    ax.axvline(emax, color=GREY, lw=1, ls=":", label=t["s_max"])
    ax.set_ylim(0, 1)
    ax.set_xlabel(t["s_x"], color=INK)
    ax.set_ylabel(t["s_y"], color=INK)
    sec = ax.secondary_yaxis("right", functions=(lambda p: 1000 * (1 - 2 * p), lambda m: (1 - m / 1000) / 2))
    sec.set_ylabel(t["s_m"], color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    return fig


def fig_garnet(t, lang):
    with open(os.path.join(ROOT, "examples", "data", "iqm_neutrino_qg_run.json")) as fh:
        run = json.load(fh)
    En = np.array(run["energies"])
    raw, filt = np.array(run["raw"]), np.array(run["filtered"])
    E = np.linspace(0.6, 5.4, 500)
    P = np.array([nu.three_flavour_probabilities(L_KM, e, DELTA_CP)[1] for e in E])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for k, f in enumerate(nu.FLAVOURS):
        ax.plot(E, P[:, k], color=COLORS[f], lw=1.6, alpha=0.8, label=NAMES[f] + ", " + t["g_exact"])
        ax.plot(En, raw[:, k], "o", mfc="white", mec=COLORS[f], mew=1.4, ms=6)
        ax.plot(En, filt[:, k], "o", color=COLORS[f], ms=6)
    ax.plot([], [], "o", mfc="white", mec=INK, mew=1.4, label=t["g_raw"])
    ax.plot([], [], "o", color=INK, label=t["g_filt"])
    ax.set_ylim(0, 1)
    ax.set_xlabel(t["g_x"], color=INK)
    ax.set_ylabel(t["g_y"], color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def main():
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        for name, fn in (("cono" if lang == "es" else "cone", fig_cone),
                         ("espectro" if lang == "es" else "spectrum", fig_spectrum), ("garnet", fig_garnet)):
            fig = fn(t, lang)
            path = os.path.join(t["dir"], f"cap09_{name}.png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", path)
    # the numbers quoted in the text
    th = 0.1
    q = math.cos(4 * th)
    print("theta 0.1 at max: m =", round(1000 * q, 1), "sigma_m(1000) =", round(1000 * math.sin(4 * th) / math.sqrt(1000), 1),
          "sigma_theta =", round(nu.mixing_angle_std(1000), 4))
    for E in (0.8, 2.6):
        print(E, nu.three_flavour_probabilities(L_KM, E, DELTA_CP)[1].round(4),
              nu.three_flavour_probabilities(L_KM, E, DELTA_CP, True)[1].round(4))
    print("A_CP(2.6) =", round(nu.cp_asymmetry(L_KM, 2.6, DELTA_CP), 4), "first max", round(nu.first_maximum_energy(L_KM), 3))


if __name__ == "__main__":
    main()
