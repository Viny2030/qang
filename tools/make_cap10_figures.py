"""Figures of chapter 10 of the book (matter, the Sun, collective oscillations), Spanish and English, on the chapter 1
base.

    python tools/make_cap10_figures.py

writes book/es/figuras/cap10_*.png and book/en/figures/cap10_*.png. Same style as make_cap01_figures.py. The curves
are qang.neutrino in closed form (§152, §156, §163) or exact evolution (§154); the device points are the saved runs
of examples/data (neutrino_solar_qg.json, neutrino_collective_qg_posthoc.json), not new jobs.
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

L_KM, RHO, DELTA_CP = 1300.0, 2.85, -math.pi / 2
DATA = os.path.join(ROOT, "examples", "data")
DEVICES = ("FakeBrisbane", "FakeTorino", "FakeFez")

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "m_x": "energía $E$ (GeV), $L$ = 1300 km",
        "m_y": r"$P(\nu_\mu\to\nu_e)$",
        "m_m": r"lectura del qubit $\nu_e$ (m-qg)",
        "m_no": "ordenamiento normal",
        "m_io": "ordenamiento invertido",
        "m_nu": r"$\nu$, materia",
        "m_anu": r"$\bar\nu$, materia",
        "m_vac": "vacío (trazos)",
        "s_x": "energía del neutrino (MeV)",
        "s_y": r"$P_{ee}$ en la Tierra",
        "s_m": r"lectura del qubit $\nu_2$, $m = 1000(1-2p_2)$ (m-qg)",
        "s_ad": "adiabática, núcleo solar",
        "s_low": "límite de vacío (energía baja)",
        "s_high": r"límite $\sin^2\theta_{12}$ (energía alta)",
        "s_dev": "circuito de 8 capas, FakeFez, con filtro",
        "s_ex": "Sol escalado, exacta (referencia del circuito)",
        "c_x": r"tiempo $t$ (con $\omega_i$ = 0,25–1)",
        "c_y": "número total esperado",
        "c_mass": r"número de $\nu_2$ (conservado)",
        "c_flav": r"número de $\nu_x$ (sabor)",
        "c_big": r", $\theta$ = 0,6",
        "c_small": r", $\theta$ = 0,195 (experimento)",
        "b_y": "error medio (post hoc, frente al circuito sin ruido)",
        "b_a": "base de masa, filtro",
        "b_b": "sabor inicial, repesado",
        "b_c": "base de sabor, sin filtro",
        "b_raw": "crudo (claro)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "m_x": "energy $E$ (GeV), $L$ = 1300 km",
        "m_y": r"$P(\nu_\mu\to\nu_e)$",
        "m_m": r"$\nu_e$-qubit reading (m-qg)",
        "m_no": "normal ordering",
        "m_io": "inverted ordering",
        "m_nu": r"$\nu$, matter",
        "m_anu": r"$\bar\nu$, matter",
        "m_vac": "vacuum (dashed)",
        "s_x": "neutrino energy (MeV)",
        "s_y": r"$P_{ee}$ at the Earth",
        "s_m": r"$\nu_2$-qubit reading, $m = 1000(1-2p_2)$ (m-qg)",
        "s_ad": "adiabatic, solar core",
        "s_low": "vacuum limit (low energy)",
        "s_high": r"$\sin^2\theta_{12}$ limit (high energy)",
        "s_dev": "8-slab circuit, FakeFez, filtered",
        "s_ex": "scaled Sun, exact (circuit reference)",
        "c_x": r"time $t$ (with $\omega_i$ = 0.25–1)",
        "c_y": "expected total number",
        "c_mass": r"number of $\nu_2$ (conserved)",
        "c_flav": r"number of $\nu_x$ (flavour)",
        "c_big": r", $\theta$ = 0.6",
        "c_small": r", $\theta$ = 0.195 (experiment)",
        "b_y": "mean error (post hoc, against the noiseless circuit)",
        "b_a": "mass basis, filter",
        "b_b": "flavour start, reweighted",
        "b_c": "flavour basis, no filter",
        "b_raw": "raw (light)",
    },
}


def fig_matter(t, lang):
    E = np.linspace(0.6, 6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    for ax, params, title in ((axes[0], nu.NUFIT_NO, t["m_no"]), (axes[1], nu.NUFIT_IO, t["m_io"])):
        for anti, color, lbl in ((False, BLUE, t["m_nu"]), (True, ORANGE, t["m_anu"])):
            mat = [nu.matter_probabilities(L_KM, e, DELTA_CP, RHO, antineutrino=anti, params=params)[1, 0] for e in E]
            vac = [nu.three_flavour_probabilities(L_KM, e, DELTA_CP, anti, params)[1, 0] for e in E]
            ax.plot(E, mat, color=color, lw=2, label=lbl)
            ax.plot(E, vac, color=color, lw=1.3, ls="--")
        ax.plot([], [], color=GREY, ls="--", label=t["m_vac"])
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel(t["m_x"], color=INK)
        ax.set_ylim(0, 0.16)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper right")
    axes[0].set_ylabel(t["m_y"], color=INK)
    sec = axes[1].secondary_yaxis("right", functions=(lambda p: 1000 * (1 - 2 * p), lambda m: (1 - m / 1000) / 2))
    sec.set_ylabel(t["m_m"], color=INK)
    fig.tight_layout()
    return fig


def fig_solar(t, lang):
    E = np.logspace(-1, math.log10(20), 300)  # MeV
    pee = [nu.solar_pee_adiabatic(e / 1000) for e in E]
    t12, t13 = nu.NUFIT_NO["theta12"], nu.NUFIT_NO["theta13"]
    low = math.cos(t13) ** 4 * (1 - 0.5 * math.sin(2 * t12) ** 2) + math.sin(t13) ** 4
    high = math.cos(t13) ** 4 * math.sin(t12) ** 2 + math.sin(t13) ** 4
    with open(os.path.join(DATA, "neutrino_solar_qg.json")) as fh:
        run = json.load(fh)
    fez = next(r for r in run["rows"] if r["backend"] == "FakeFez")
    En = np.array([0.5, 1, 2, 4, 6, 10, 15])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.semilogx(E, pee, color=BLUE, lw=2.2, label=t["s_ad"])
    ax.axhline(low, color=GREY, lw=1.2, ls="--", label=t["s_low"])
    ax.axhline(high, color=GREY, lw=1.2, ls=":", label=t["s_high"])
    ax.plot(En, run["exact"], "s", mfc="white", mec=INK, ms=6, label=t["s_ex"])
    ax.plot(En, fez["estimate"]["slab8"]["filtered_readout"], "o", color=ORANGE, ms=6, label=t["s_dev"])
    ax.set_ylim(0.25, 0.6)
    ax.set_xlabel(t["s_x"], color=INK)
    ax.set_ylabel(t["s_y"], color=INK)

    # P_ee is linear in p2, so the nu_2-qubit reading is a second linear axis
    a = math.cos(t13) ** 4 * math.cos(t12) ** 2 + math.sin(t13) ** 4
    b = math.cos(t13) ** 4 * (math.sin(t12) ** 2 - math.cos(t12) ** 2)
    sec = ax.secondary_yaxis("right", functions=(lambda p: 1000 * (1 - 2 * (p - a) / b),
                                                 lambda m: a + b * (1 - m / 1000) / 2))
    sec.set_ylabel(t["s_m"], color=INK)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    return fig


def fig_collective(t, lang):
    omegas, mu, init = (0.25, 0.5, 0.75, 1.0), 4.0, "eexx"
    ts = np.linspace(0, 6, 121)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for theta, ls, tag in ((0.195, "-", t["c_small"]), (0.6, "--", t["c_big"])):
        m = [nu.collective_exact(omegas, mu, theta, init, x, measure_basis="mass").sum() for x in ts]
        f = [nu.collective_exact(omegas, mu, theta, init, x, measure_basis="flavour").sum() for x in ts]
        a1.plot(ts, m, color=BLUE, lw=2, ls=ls, label=t["c_mass"] + tag)
        a1.plot(ts, f, color=ORANGE, lw=2, ls=ls, label=t["c_flav"] + tag)
    a1.set_xlabel(t["c_x"], color=INK)
    a1.set_ylabel(t["c_y"], color=INK)
    a1.set_ylim(1.7, 2.3)
    a1.grid(alpha=0.25)
    a1.legend(fontsize=7.5, loc="upper left")

    with open(os.path.join(DATA, "neutrino_collective_qg_posthoc.json")) as fh:
        run = json.load(fh)
    x = np.arange(len(DEVICES))
    w = 0.26
    for k, (part, key, color, lbl) in enumerate((("A", "filtered_readout", BLUE, t["b_a"]),
                                                  ("B", "reweighted_readout", GREEN, t["b_b"]),
                                                  ("C", "readout", ORANGE, t["b_c"]))):
        raw = [run[d]["error"][part]["raw"] for d in DEVICES]
        best = [run[d]["error"][part][key] for d in DEVICES]
        a2.bar(x + (k - 1) * w, raw, w * 0.9, color=color, alpha=0.3)
        a2.bar(x + (k - 1) * w, best, w * 0.9, color=color, label=lbl)
    a2.bar([0], [0], color=GREY, alpha=0.3, label=t["b_raw"])
    a2.set_xticks(x, DEVICES)
    a2.set_ylabel(t["b_y"], color=INK)
    a2.grid(alpha=0.25, axis="y")
    a2.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return fig


def main():
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        for name, fn in (("materia" if lang == "es" else "matter", fig_matter),
                         ("sol" if lang == "es" else "sun", fig_solar),
                         ("colectivas" if lang == "es" else "collective", fig_collective)):
            fig = fn(t, lang)
            path = os.path.join(t["dir"], f"cap10_{name}.png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", path)
    # the numbers quoted in the text
    print("A(2.85 g/cm3, 2.6 GeV) =", nu.matter_potential(RHO, 2.6), "eV^2")
    for name, p in (("NO", nu.NUFIT_NO), ("IO", nu.NUFIT_IO)):
        for anti in (False, True):
            m = nu.matter_probabilities(L_KM, 2.6, DELTA_CP, RHO, antineutrino=anti, params=p)[1, 0]
            v = nu.three_flavour_probabilities(L_KM, 2.6, DELTA_CP, anti, p)[1, 0]
            print(name, "anti" if anti else "nu", "matter", round(m, 4), "vacuum", round(v, 4))
    print("A_CP matter, delta 0:", round(nu.cp_asymmetry(L_KM, 2.6, 0.0, rho=RHO), 4),
          "vacuum, delta 0:", round(nu.cp_asymmetry(L_KM, 2.6, 0.0), 4))
    for E in (0.5, 1, 2, 5, 10, 15):
        print("P_ee", E, "MeV", round(nu.solar_pee_adiabatic(E / 1000), 3))
    for theta in (0.195, 0.6):
        f = [nu.collective_exact((0.25, 0.5, 0.75, 1.0), 4.0, theta, "eexx", x, measure_basis="flavour").sum()
             for x in np.linspace(0, 6, 121)]
        print("theta", theta, "flavour number range", round(min(f), 3), round(max(f), 3),
              "mass sectors", nu.mass_sector_probabilities("eexx", theta).round(3))


if __name__ == "__main__":
    main()
