"""Figures of chapter 11 of the book (Majorana phases, neutrinoless double beta decay, a sterile neutrino, the four
quantum numbers), Spanish and English, on the chapter 1 base.

    python tools/make_cap11_figures.py

writes book/es/figuras/cap11_*.png and book/en/figures/cap11_*.png. Same style as make_cap01_figures.py. The m_bb
bands scan the two Majorana phases on a grid (qang.neutrino.m_bb, §163); the sterile points are the saved run of
examples/data/neutrino_ordering_majorana_sterile_qg.json, not a new job; the angles are qang.angular (§155).
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
from qang import angular as AN  # noqa: E402
from qang import neutrino as nu  # noqa: E402

DEVICES = ("FakeBrisbane", "FakeTorino", "FakeFez")
MARKERS = {"FakeBrisbane": "o", "FakeTorino": "s", "FakeFez": "^"}
STER = {"theta14": 0.15, "theta24": 0.15, "theta34": 0.0}

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "b_x": "masa del neutrino más liviano (eV)",
        "b_y": r"masa efectiva $m_{\beta\beta}$ (eV)",
        "b_no": "ordenamiento normal",
        "b_io": "ordenamiento invertido",
        "b_zero": r"fases nulas, $\alpha_{21}=\alpha_{31}=0$",
        "s_x": "energía $E$ (GeV), $L$ = 0,5 km",
        "s_y": r"$P(\nu_\mu\to\nu_\mu)$",
        "s_m": r"lectura del qubit $\nu_\mu$ (m-qg)",
        "s_ex": r"3+1 exacta, $\theta_{14}=\theta_{24}=0{,}15$, $\Delta m^2_{41}=1$ eV²",
        "s_raw": "crudo (vacío)",
        "s_filt": "con filtro (lleno)",
        "a_x": r"número cuántico magnético $m$",
        "a_y": "ángulo polar (grados)",
        "a_qang": r"ángulo qang $\arccos(m/l)$",
        "a_vec": r"modelo vectorial $\arccos(m/\sqrt{l(l+1)})$",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "b_x": "mass of the lightest neutrino (eV)",
        "b_y": r"effective mass $m_{\beta\beta}$ (eV)",
        "b_no": "normal ordering",
        "b_io": "inverted ordering",
        "b_zero": r"zero phases, $\alpha_{21}=\alpha_{31}=0$",
        "s_x": "energy $E$ (GeV), $L$ = 0.5 km",
        "s_y": r"$P(\nu_\mu\to\nu_\mu)$",
        "s_m": r"$\nu_\mu$-qubit reading (m-qg)",
        "s_ex": r"3+1 exact, $\theta_{14}=\theta_{24}=0.15$, $\Delta m^2_{41}=1$ eV²",
        "s_raw": "raw (open)",
        "s_filt": "filtered (filled)",
        "a_x": r"magnetic quantum number $m$",
        "a_y": "polar angle (degrees)",
        "a_qang": r"qang angle $\arccos(m/l)$",
        "a_vec": r"vector model $\arccos(m/\sqrt{l(l+1)})$",
    },
}


def mbb_band(m_light, ordering, n=37):
    ph = np.linspace(0, 2 * math.pi, n)
    v = [nu.m_bb(m_light, a, b, ordering) for a in ph for b in ph]
    return min(v), max(v)


def fig_mbb(t, lang):
    ms = np.unique(np.concatenate([np.logspace(-4, 0, 70), np.logspace(-3, -2, 80)]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for o, color, lbl in (("NO", BLUE, t["b_no"]), ("IO", ORANGE, t["b_io"])):
        lo, hi = np.array([mbb_band(m, o, 61) for m in ms]).T
        ax.fill_between(ms, np.maximum(lo, 1e-4), hi, color=color, alpha=0.35, lw=0, label=lbl)
        ax.plot(ms, [nu.m_bb(m, 0, 0, o) for m in ms], color=color, lw=1.4)
    ax.plot([], [], color=INK, lw=1.4, label=t["b_zero"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(t["b_x"], color=INK)
    ax.set_ylabel(t["b_y"], color=INK)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    return fig


def fig_sterile(t, lang):
    with open(os.path.join(ROOT, "examples", "data", "neutrino_ordering_majorana_sterile_qg.json")) as fh:
        run = json.load(fh)
    E = np.linspace(0.15, 1.6, 400)
    P = [nu.sterile_probabilities(0.5, e, 1.0, **STER)[1, 1] for e in E]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(E, P, color=INK, lw=2, label=t["s_ex"])
    colors = {"FakeBrisbane": BLUE, "FakeTorino": ORANGE, "FakeFez": GREEN}
    for row in run["rows"]:
        d = row["backend"]
        Es = sorted(row["S"], key=float)
        x = [float(e) for e in Es]
        ax.plot(x, [row["S"][e]["raw"] for e in Es], MARKERS[d], mfc="white", mec=colors[d], mew=1.4, ms=6)
        ax.plot(x, [row["S"][e]["filtered"] for e in Es], MARKERS[d], color=colors[d], ms=6, label=d)
    ax.plot([], [], "o", mfc="white", mec=GREY, mew=1.4, label=t["s_raw"])
    ax.plot([], [], "o", color=GREY, label=t["s_filt"])
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel(t["s_x"], color=INK)
    ax.set_ylabel(t["s_y"], color=INK)
    sec = ax.secondary_yaxis("right", functions=(lambda p: 1000 * (1 - 2 * p), lambda m: (1 - m / 1000) / 2))
    sec.set_ylabel(t["s_m"], color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def fig_angles(t, lang):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for l, color in ((1, BLUE), (2, ORANGE), (3, GREEN)):
        m = np.arange(-l, l + 1)
        qa = [math.degrees(AN.qang_angle_of_eigenstate(l, k)) for k in m]
        va = [math.degrees(AN.vector_model_angle(l, k)) for k in m]
        ax.plot(m, qa, "o-", color=color, lw=1.8, ms=6, label=f"l = {l}")
        ax.plot(m, va, "s--", color=color, lw=1.2, ms=5, mfc="white")
    ax.plot([], [], "o-", color=GREY, label=t["a_qang"])
    ax.plot([], [], "s--", color=GREY, mfc="white", label=t["a_vec"])
    ax.set_xlabel(t["a_x"], color=INK)
    ax.set_ylabel(t["a_y"], color=INK)
    ax.set_yticks(range(0, 181, 30))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    return fig


def main():
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        for name, fn in (("mbb", fig_mbb), ("esteril" if lang == "es" else "sterile", fig_sterile),
                         ("angulos" if lang == "es" else "angles", fig_angles)):
            fig = fn(t, lang)
            path = os.path.join(t["dir"], f"cap11_{name}.png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", path)
    # the numbers quoted in the text
    for o in ("NO", "IO"):
        print(o, "m_bb(0) band meV", [round(1000 * x, 2) for x in mbb_band(0.0, o, 73)], "masses", nu.masses(0, o))
    print("NO minimum of the lower edge:", min((mbb_band(m, "NO", 73)[0], m) for m in np.logspace(-3, -2, 30)))
    print("Majorana invisible, vacuum / matter:",
          nu.majorana_invisible(1300, 2.6, -math.pi / 2, math.pi / 2, 3 * math.pi / 4),
          nu.majorana_invisible(1300, 2.6, -math.pi / 2, math.pi / 2, 3 * math.pi / 4, rho=2.85))
    U = nu.sterile_mixing(**STER)
    print("|U_mu4|^2 =", round(U[1, 3] ** 2, 4), "amplitude 4|U|^2(1-|U|^2) =",
          round(4 * U[1, 3] ** 2 * (1 - U[1, 3] ** 2), 4), "first max E =", round(2 * nu.K_OSC * 1.0 * 0.5 / math.pi, 3))


if __name__ == "__main__":
    main()
