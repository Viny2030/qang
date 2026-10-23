"""Figures of chapter 12 of the book (teleportation through a passive optical network, Bell pairs read in qg),
Spanish and English, on the chapter 1 base.

    python tools/make_cap12_figures.py

writes book/es/figuras/cap12_*.png and book/en/figures/cap12_*.png. Same style as make_cap01_figures.py. The curves
are the closed forms of qang.qpon (§157); the device points are the saved FakeFez run
(examples/data/qpon_teleport_qg_fez.json), not a new job; the Bell-pair signatures are exact density matrices of the
§57 noise channels (p = 0.1 on each half).
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

PURPLE = "#8a5cc2"

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "a_x": r"ángulo del recurso $\alpha$ (rad); $\cos\alpha|00\rangle+\sin\alpha|11\rangle$",
        "a_y": "fidelidad media de teleportación",
        "a_non": "sin asistencia",
        "a_ass": "asistida (filtro de sector)",
        "a_dev": "FakeFez (puntos)",
        "a_bell": r"par de Bell, $\alpha=\pi/4$",
        "a_cl": "límite clásico 2/3",
        "a_title": "enlace de 0 km, divisor 1:8",
        "d_x": "longitud de fibra (km), 0,2 dB/km",
        "d_f": r"fidelidad óptima $F^\ast$",
        "d_q": r"lectura del recurso óptimo, $m^\ast = 1000\,qg_Z^\ast$ (m-qg)",
        "d_n": "N = {n}",
        "d_ass": "asistida (trazos)",
        "b_y": "correlador qg",
        "b_noise": ["desfase", "inversión X", "inversión Y", "despolarizante", "amortiguamiento"],
        "b_ma": r"$m_A=\langle Z_A\rangle$",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "a_x": r"resource angle $\alpha$ (rad); $\cos\alpha|00\rangle+\sin\alpha|11\rangle$",
        "a_y": "average teleportation fidelity",
        "a_non": "non-assisted",
        "a_ass": "assisted (sector filter)",
        "a_dev": "FakeFez (points)",
        "a_bell": r"Bell pair, $\alpha=\pi/4$",
        "a_cl": "classical limit 2/3",
        "a_title": "0 km link, 1:8 splitter",
        "d_x": "fibre length (km), 0.2 dB/km",
        "d_f": r"optimal fidelity $F^\ast$",
        "d_q": r"reading of the optimal resource, $m^\ast = 1000\,qg_Z^\ast$ (m-qg)",
        "d_n": "N = {n}",
        "d_ass": "assisted (dashed)",
        "b_y": "qg correlator",
        "b_noise": ["dephasing", "X flip", "Y flip", "depolarizing", "amplitude damping"],
        "b_ma": r"$m_A=\langle Z_A\rangle$",
    },
}


def fig_alpha(t, lang):
    with open(os.path.join(ROOT, "examples", "data", "qpon_teleport_qg_fez.json")) as fh:
        grid = json.load(fh)["row"]["links"]["0,8"]
    a = np.linspace(0.01, 1.55, 400)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for assisted, color, lbl in ((False, BLUE, t["a_non"]), (True, ORANGE, t["a_ass"])):
        eta, kappa = Q.link(0, 8, assisted)
        ax.plot(a, [Q.average_fidelity(x, eta, kappa) for x in a], color=color, lw=2, label=lbl)
        astar = Q.optimal_angle(eta, kappa)
        ax.plot([astar], [Q.optimal_fidelity(eta, kappa)], "*", color=color, ms=13, mec=INK, mew=0.6)
        pts = sorted((float(k), v["assisted" if assisted else "non"]["fidelity"]) for k, v in grid.items())
        ax.plot(*zip(*pts), "o", color=color, ms=5, mfc="white", mew=1.4)
    ax.plot([], [], "o", color=GREY, mfc="white", mew=1.4, label=t["a_dev"])
    ax.axvline(math.pi / 4, color=GREY, lw=1, ls=":", label=t["a_bell"])
    ax.axhline(Q.CLASSICAL_FIDELITY, color=INK, lw=0.9, ls="--", label=t["a_cl"])
    ax.set_title(t["a_title"], color=INK, fontsize=10)
    ax.set_xlabel(t["a_x"], color=INK)
    ax.set_ylabel(t["a_y"], color=INK)
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    return fig


def fig_design(t, lang):
    L = np.linspace(0, 50, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for n, color in ((4, BLUE), (8, ORANGE), (32, GREEN)):
        for assisted, ls in ((False, "-"), (True, "--")):
            ek = [Q.link(x, n, assisted) for x in L]
            a1.plot(L, [Q.optimal_fidelity(e, k) for e, k in ek], color=color, ls=ls, lw=2,
                    label=t["d_n"].format(n=n) if not assisted else None)
            a2.plot(L, [1000 * Q.optimal_qg_z(e, k) for e, k in ek], color=color, ls=ls, lw=2)
    a1.plot([], [], color=GREY, ls="--", label=t["d_ass"])
    a1.axhline(Q.CLASSICAL_FIDELITY, color=INK, lw=0.9, ls=":")
    a2.axhline(0, color=INK, lw=0.9, ls=":")
    a1.set_ylabel(t["d_f"], color=INK)
    a2.set_ylabel(t["d_q"], color=INK)
    for ax in (a1, a2):
        ax.set_xlabel(t["d_x"], color=INK)
        ax.grid(alpha=0.25)
    a1.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- §57: Bell-pair signatures

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]], complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def _kraus(kind, p):
    if kind == "dephasing":
        return [math.sqrt(1 - p) * _I, math.sqrt(p) * _Z]
    if kind == "xflip":
        return [math.sqrt(1 - p) * _I, math.sqrt(p) * _X]
    if kind == "yflip":
        return [math.sqrt(1 - p) * _I, math.sqrt(p) * _Y]
    if kind == "depolarizing":
        return [math.sqrt(1 - 3 * p / 4) * _I] + [math.sqrt(p / 4) * P for P in (_X, _Y, _Z)]
    g = p  # amplitude damping
    return [np.array([[1, 0], [0, math.sqrt(1 - g)]], complex), np.array([[0, math.sqrt(g)], [0, 0]], complex)]


def bell_signature(kind, p=0.1):
    """(c_x, c_y, c_z, m_A, F) of |Phi+> after the channel on each half (§57)."""
    phi = np.array([1, 0, 0, 1], complex) / math.sqrt(2)
    rho = np.outer(phi, phi.conj())
    K = _kraus(kind, p)
    out = sum(np.kron(A, B) @ rho @ np.kron(A, B).conj().T for A in K for B in K)
    ev = lambda O: float(np.real(np.trace(out @ O)))  # noqa: E731
    cx, cy, cz = ev(np.kron(_X, _X)), ev(np.kron(_Y, _Y)), ev(np.kron(_Z, _Z))
    return cx, cy, cz, ev(np.kron(_Z, _I)), (1 + cx - cy + cz) / 4


KINDS = ("dephasing", "xflip", "yflip", "depolarizing", "amplitude")


def fig_bell(t, lang):
    sig = np.array([bell_signature(k) for k in KINDS])
    x = np.arange(len(KINDS))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for j, (lbl, color) in enumerate(((r"$qg_{XX}$", BLUE), (r"$-qg_{YY}$", ORANGE), (r"$qg_{ZZ}$", GREEN),
                                      (t["b_ma"], PURPLE))):
        vals = -sig[:, 1] if j == 1 else sig[:, j if j < 3 else 3]
        ax.bar(x + (j - 1.5) * w, vals, w * 0.92, color=color, label=lbl)
    ax.axhline(1, color=INK, lw=0.8, ls=":")
    ax.set_xticks(x, t["b_noise"])
    ax.set_ylabel(t["b_y"], color=INK)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8.5, loc="upper center", ncol=4)
    fig.tight_layout()
    return fig


def main():
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        for name, fn in (("alfa" if lang == "es" else "alpha", fig_alpha),
                         ("diseno" if lang == "es" else "design", fig_design),
                         ("bell", fig_bell)):
            fig = fn(t, lang)
            path = os.path.join(t["dir"], f"cap12_{name}.png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", path)
    # the numbers quoted in the text
    for L, n in ((0, 4), (0, 8), (20, 4), (20, 8)):
        for assisted in (False, True):
            e, k = Q.link(L, n, assisted)
            a = Q.optimal_angle(e, k)
            print(L, n, "assisted" if assisted else "non", "eta", round(e, 4), "kappa", round(k, 4), "alpha*", round(a, 3),
                  "F*", round(Q.optimal_fidelity(e, k), 4), "qgZ*", round(Q.optimal_qg_z(e, k), 3),
                  "F(pi/4)", round(Q.average_fidelity(math.pi / 4, e, k), 4), "kept*", round(Q.heralding_probability(a, k), 3))
    for k in KINDS:
        print(k, [round(v, 3) for v in bell_signature(k)])


if __name__ == "__main__":
    main()
