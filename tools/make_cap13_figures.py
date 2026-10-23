"""Figures of chapter 13 of the book (dual rail, the sector filter and long-distance fidelity; entanglement swapping
read with qg correlators), Spanish and English, on the chapter 1 base.

    python tools/make_cap13_figures.py

writes book/es/figuras/cap13_*.png and book/en/figures/cap13_*.png. Same style as make_cap01_figures.py. The device
points are the saved run of §158 (examples/data/qpon_teleport_dual_rail_qg.json), not new jobs. The false-herald
curves are an illustrative model, not a fit. The swapping section is exact density-matrix algebra computed here
(Bell measurement, Pauli correction, average over the four outcomes); it is new in the book and has no device run.
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
from qang import qpon as Q  # noqa: E402

DEVICES = ("FakeBrisbane", "FakeTorino", "FakeFez")
COLORS = {"FakeBrisbane": BLUE, "FakeTorino": ORANGE, "FakeFez": GREEN}
LINKS = ("0,4", "0,8", "20,4", "20,8")

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "c_x": r"predicha por tres correladores, $(3+qg_{XX}-qg_{YY}+qg_{ZZ})/6$",
        "c_y": "fidelidad teleportada medida",
        "c_non": "sin asistencia (vacío)",
        "c_ass": "asistida (lleno)",
        "c_band": r"$\pm0{,}03$",
        "d_x": r"probabilidad de llegada $\eta = T/N$",
        "d_y": "fidelidad media de teleportación",
        "d_single": r"carril simple sin ruido, óptimo $F^\ast$ (no asistido / asistido)",
        "d_model": "modelo de falsos anuncios, $p_f$ = {p}",
        "d_cl": "límite clásico 2/3",
        "s_x": "número de enlaces de la cadena $n$",
        "s_y": "fidelidad de teleportación del par final",
        "s_dep": "despolarizante, $w$ = {w}",
        "s_deph": r"desfase, $p$ = 0,1 por mitad",
        "s_cl": "límite clásico 2/3",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "c_x": r"predicted by three correlators, $(3+qg_{XX}-qg_{YY}+qg_{ZZ})/6$",
        "c_y": "measured teleported fidelity",
        "c_non": "non-assisted (open)",
        "c_ass": "assisted (filled)",
        "c_band": r"$\pm0.03$",
        "d_x": r"arrival probability $\eta = T/N$",
        "d_y": "average teleportation fidelity",
        "d_single": r"noiseless single rail, optimum $F^\ast$ (non-assisted / assisted)",
        "d_model": "false-herald model, $p_f$ = {p}",
        "d_cl": "classical limit 2/3",
        "s_x": "number of links in the chain $n$",
        "s_y": "teleportation fidelity of the final pair",
        "s_dep": "depolarizing, $w$ = {w}",
        "s_deph": r"dephasing, $p$ = 0.1 per half",
        "s_cl": "classical limit 2/3",
    },
}


def load():
    with open(os.path.join(ROOT, "examples", "data", "qpon_teleport_dual_rail_qg.json")) as fh:
        return {r["backend"]: r for r in json.load(fh)["rows"]}


def fig_correlators(t, lang):
    rows = load()
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    lim = (0.64, 0.99)
    x = np.array(lim)
    ax.plot(x, x, color=INK, lw=0.9)
    ax.fill_between(x, x - 0.03, x + 0.03, color=GREY, alpha=0.2, lw=0, label=t["c_band"])
    for d in DEVICES:
        for link in LINKS:
            for p, filled in (("non", False), ("assisted", True)):
                v = rows[d]["links"][link][p]
                ax.plot(v["F_corr"], v["F_tele"], "o", ms=7, color=COLORS[d], mfc=COLORS[d] if filled else "white",
                        mew=1.4, label=d if (link, p) == ("0,4", "assisted") else None)
    ax.plot([], [], "o", color=GREY, mfc="white", mew=1.4, label=t["c_non"])
    ax.plot([], [], "o", color=GREY, label=t["c_ass"])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel(t["c_x"], color=INK)
    ax.set_ylabel(t["c_y"], color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def false_herald(eta, f_true, p_f):
    """Filtered F when a lost photon (probability 1 - eta) still passes the filter with probability p_f, carrying 1/2."""
    return (eta * f_true + (1 - eta) * p_f * 0.5) / (eta + (1 - eta) * p_f)


def fig_dual(t, lang):
    rows = load()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    e = np.linspace(0.04, 0.26, 200)
    ax.plot(e, [Q.optimal_fidelity(x) for x in e], color=GREY, lw=1.6, label=t["d_single"])
    for d in DEVICES:
        pts = sorted((rows[d]["links"][k]["dual"]["eta"], rows[d]["links"][k]["dual"]["filtered"]) for k in LINKS)
        ax.plot(*zip(*pts), "o-", color=COLORS[d], lw=1.4, ms=7, label=d)
    for k in ("20,4", "20,8"):  # the assisted noiseless optimum at the two 20 km links
        L, N = (int(s) for s in k.split(","))
        eta, kappa = Q.link(L, N, True)
        ax.plot([eta], [Q.optimal_fidelity(eta, kappa)], "D", color=GREY, ms=6, mfc="white", mew=1.4)
    for p, ls in ((0.03, "--"), (0.06, ":")):
        ax.plot(e, false_herald(e, 0.87, p), color=INK, lw=1.1, ls=ls,
                label=t["d_model"].format(p=f"{p}".replace(".", "," if lang == "es" else ".")))
    ax.axhline(Q.CLASSICAL_FIDELITY, color=INK, lw=0.8, ls="-.", label=t["d_cl"])
    ax.set_xlabel(t["d_x"], color=INK)
    ax.set_ylabel(t["d_y"], color=INK)
    ax.set_ylim(0.6, 0.92)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- entanglement swapping, exact

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]], complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
_PHI = np.array([1, 0, 0, 1], complex) / math.sqrt(2)


def bell_diagonal(cx, cy, cz):
    """Two-qubit state (I + cx XX + cy YY + cz ZZ) / 4."""
    return (np.kron(_I, _I) + cx * np.kron(_X, _X) + cy * np.kron(_Y, _Y) + cz * np.kron(_Z, _Z)) / 4


def swap(rho1, rho2):
    """Swap rho1 (A, B) and rho2 (C, D): Bell measurement on B, C, Pauli correction on D; returns rho(A, D)."""
    rho = np.kron(rho1, rho2).reshape([2] * 8)  # indices a b c d, a' b' c' d'
    out = np.zeros((4, 4), complex)
    for P in (_I, _X, _Y, _Z):  # Bell outcome (P on C) |Phi+>, correction P on D
        bell = (np.kron(_I, P) @ _PHI).reshape(2, 2)
        proj = np.einsum("bc,abcdefgh,fg->adeh", bell.conj(), rho, bell)  # <bell|_BC rho |bell>_BC
        proj = proj.reshape(4, 4)
        C = np.kron(_I, P.conj().T)
        out += C @ proj @ C.conj().T
    return out


def correlators(rho):
    ev = lambda O: float(np.real(np.trace(rho @ O)))  # noqa: E731
    return ev(np.kron(_X, _X)), ev(np.kron(_Y, _Y)), ev(np.kron(_Z, _Z))


def chain(c, n):
    rho = bell_diagonal(*c)
    for _ in range(n - 1):
        rho = swap(rho, bell_diagonal(*c))
    return correlators(rho)


def fig_swap(t, lang):
    ns = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for w, color in ((0.95, BLUE), (0.9, ORANGE), (0.8, GREEN)):
        F = [Q.fidelity_from_correlators(*chain((w, -w, w), n)) for n in ns]
        ax.plot(ns, F, "o-", color=color, lw=1.8, ms=5,
                label=t["s_dep"].format(w=f"{w}".replace(".", "," if lang == "es" else ".")))
    F = [Q.fidelity_from_correlators(*chain((0.64, -0.64, 1.0), n)) for n in ns]
    ax.plot(ns, F, "s--", color=GREY, lw=1.4, ms=5, label=t["s_deph"])
    ax.axhline(Q.CLASSICAL_FIDELITY, color=INK, lw=0.9, ls="--", label=t["s_cl"])
    ax.set_xticks(ns)
    ax.set_xlabel(t["s_x"], color=INK)
    ax.set_ylabel(t["s_y"], color=INK)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    return fig


def main():
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        for name, fn in (("correladores" if lang == "es" else "correlators", fig_correlators),
                         ("doble_carril" if lang == "es" else "dual_rail", fig_dual),
                         ("intercambio" if lang == "es" else "swapping", fig_swap)):
            fig = fn(t, lang)
            path = os.path.join(t["dir"], f"cap13_{name}.png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", path)
    # the numbers quoted in the text
    rows = load()
    gaps = [abs(rows[d]["links"][k][p]["F_corr"] - rows[d]["links"][k][p]["F_tele"])
            for d in DEVICES for k in LINKS for p in ("non", "assisted")]
    print("max |F_corr - F_tele| =", round(max(gaps), 4), "over", len(gaps))
    for d in DEVICES:
        print(d, {k: (round(rows[d]["links"][k]["dual"]["filtered"], 4), round(rows[d]["links"][k]["dual"]["raw"], 3))
                  for k in LINKS})
    for c in ((0.9, -0.9, 0.9), (0.64, -0.64, 1.0), (0.9, -0.9, 0.82)):
        print("pair", c, "two links", [round(v, 4) for v in chain(c, 2)], "three", [round(v, 4) for v in chain(c, 3)])
    for w in (0.95, 0.9, 0.8):
        n_max = max(n for n in range(1, 60) if Q.fidelity_from_correlators(*chain((w, -w, w), n)) > 2 / 3 + 1e-12)
        print("w", w, "longest chain above 2/3:", n_max, "closed form ln(1/3)/ln(w) =", round(math.log(1 / 3) / math.log(w), 2))


if __name__ == "__main__":
    main()
