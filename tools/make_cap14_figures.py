"""Figures of chapter 14 of the book (postselected closed timelike curves, P-CTC, against Deutsch's D-CTC), Spanish and
English, on the chapter 1 base.

    python tools/make_cap14_figures.py

writes book/es/figuras/cap14_*.png and book/en/figures/cap14_*.png. Same style as make_cap01_figures.py. The P-CTC
curves are qang.postselection (§164); the device points are the saved run
(examples/data/postselection_ctc_blackhole_qg.json), not new jobs. The D-CTC fixed point is computed here (exact
density matrices, Deutsch's maximum-entropy rule); it is new in the book and has no device run.
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
from qang import postselection as PS  # noqa: E402

DEVICES = ("FakeBrisbane", "FakeTorino", "FakeFez")
COLORS = {"FakeBrisbane": BLUE, "FakeTorino": ORANGE, "FakeFez": GREEN}
MARKERS = {"FakeBrisbane": "o", "FakeTorino": "s", "FakeFez": "^"}

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "phi": r"rotación del qubit que viaja al pasado, $\varphi$ (rad)",
        "p_y": "probabilidad de éxito de la posselección",
        "p_ex": r"exacta, $\cos^2(\varphi/2)$",
        "a_y": r"ángulo qang de la probabilidad, $\arccos(2P-1)$ (rad)",
        "a_ex": r"exacto: el ángulo es $\varphi$",
        "c_y": "lectura del sistema S (m-qg)",
        "c_pz": r"P-CTC, $m_Z$ exacta",
        "c_px": r"P-CTC, $m_X$ exacta",
        "c_dz": r"D-CTC, $m_Z$",
        "c_dx": r"D-CTC, $m_X$",
        "c_dev": "P-CTC, $m_Z$ en dispositivos simulados",
        "k_y": "disparos por disparo útil",
        "k_u": "sin control",
        "k_c": "controlada por S",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "phi": r"rotation of the qubit sent to the past, $\varphi$ (rad)",
        "p_y": "postselection success probability",
        "p_ex": r"exact, $\cos^2(\varphi/2)$",
        "a_y": r"qang angle of the probability, $\arccos(2P-1)$ (rad)",
        "a_ex": r"exact: the angle is $\varphi$",
        "c_y": "reading of the system S (m-qg)",
        "c_pz": r"P-CTC, exact $m_Z$",
        "c_px": r"P-CTC, exact $m_X$",
        "c_dz": r"D-CTC, $m_Z$",
        "c_dx": r"D-CTC, $m_X$",
        "c_dev": "P-CTC, $m_Z$ on simulated devices",
        "k_y": "shots per useful shot",
        "k_u": "uncontrolled",
        "k_c": "controlled by S",
    },
}


def load():
    with open(os.path.join(ROOT, "examples", "data", "postselection_ctc_blackhole_qg.json")) as fh:
        return {r["backend"]: r for r in json.load(fh)["rows"]}


def device_points(row, controlled, field):
    pts = []
    for k, v in row["C"].items():
        phi, c = k.split(",")
        if int(c) == int(controlled):
            pts.append((float(phi), v[field]))
    return sorted(pts)


def fig_success(t, lang):
    rows = load()
    phi = np.linspace(0, math.pi, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    a1.plot(phi, np.cos(phi / 2) ** 2, color=INK, lw=2, label=t["p_ex"])
    a2.plot(phi, phi, color=INK, lw=2, label=t["a_ex"])
    for d in DEVICES:
        pts = device_points(rows[d], False, "success")
        x = [p for p, _ in pts]
        a1.plot(x, [s for _, s in pts], MARKERS[d], color=COLORS[d], ms=7, label=d)
        a2.plot(x, [math.acos(max(-1, min(1, 2 * s - 1))) for _, s in pts], MARKERS[d], color=COLORS[d], ms=7)
    a1.set_ylabel(t["p_y"], color=INK)
    a2.set_ylabel(t["a_y"], color=INK)
    for ax in (a1, a2):
        ax.set_xlabel(t["phi"], color=INK)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper right" if ax is a1 else "upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- D-CTC, exact

_X = np.array([[0, 1], [1, 0]], complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def rx(phi):
    return np.array([[math.cos(phi / 2), -1j * math.sin(phi / 2)], [-1j * math.sin(phi / 2), math.cos(phi / 2)]])


def crx(phi):
    """Controlled RX on (S, C), S the first factor."""
    U = np.eye(4, dtype=complex)
    U[2:, 2:] = rx(phi)
    return U


def ptrace(rho, keep):
    r = rho.reshape(2, 2, 2, 2)
    return np.einsum("ijkj->ik", r) if keep == 0 else np.einsum("ijil->jl", r)


def dctc(phi, iters=400):
    """Deutsch: rho_C = Tr_S[U (rho_S x rho_C) U^+], from the maximally mixed start (maximum-entropy rule); returns
    (rho_C fixed point, output rho_S, residual)."""
    plus = np.array([1, 1], complex) / math.sqrt(2)
    rs = np.outer(plus, plus.conj())
    U = crx(phi)
    rc = np.eye(2, dtype=complex) / 2
    for _ in range(iters):
        rc = ptrace(U @ np.kron(rs, rc) @ U.conj().T, 1)
    out = U @ np.kron(rs, rc) @ U.conj().T
    res = np.abs(ptrace(out, 1) - rc).max()
    return rc, ptrace(out, 0), res


def pctc_state(phi):
    """Postselected S of the controlled P-CTC: proportional to Tr_C(U) |+>, Tr_C(U) = diag(1, cos(phi/2))."""
    c = math.cos(phi / 2)
    v = np.array([1, c], complex)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def readings(rho):
    return float(np.real(np.trace(rho @ _Z))), float(np.real(np.trace(rho @ _X)))


def fig_compare(t, lang):
    rows = load()
    phi = np.linspace(0, math.pi, 200)
    p = np.array([readings(pctc_state(x)) for x in phi])
    d = np.array([readings(dctc(x, 60)[1]) for x in phi])
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(phi, 1000 * p[:, 0], color=BLUE, lw=2.2, label=t["c_pz"])
    ax.plot(phi, 1000 * p[:, 1], color=BLUE, lw=1.6, ls="--", label=t["c_px"])
    ax.plot(phi, 1000 * d[:, 0], color=ORANGE, lw=2.2, label=t["c_dz"])
    ax.plot(phi, 1000 * d[:, 1], color=ORANGE, lw=1.6, ls="--", label=t["c_dx"])
    for dev in DEVICES:
        pts = device_points(rows[dev], True, "qg_z_S")
        ax.plot([q for q, _ in pts], [1000 * v for _, v in pts], MARKERS[dev], color=COLORS[dev], ms=6,
                mfc="white", mew=1.3)
    ax.plot([], [], "o", color=GREY, mfc="white", mew=1.3, label=t["c_dev"])
    ax.set_xlabel(t["phi"], color=INK)
    ax.set_ylabel(t["c_y"], color=INK)
    ax.set_ylim(-50, 1050)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="center left")
    fig.tight_layout()
    return fig


def fig_cost(t, lang):
    phi = np.linspace(0, math.pi * 0.98, 300)
    c2 = np.cos(phi / 2) ** 2
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(phi, 1 / c2, color=BLUE, lw=2, label=t["k_u"])
    ax.semilogy(phi, 2 / (1 + c2), color=ORANGE, lw=2, label=t["k_c"])
    ax.set_xlabel(t["phi"], color=INK)
    ax.set_ylabel(t["k_y"], color=INK)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        for name, fn in (("exito" if lang == "es" else "success", fig_success),
                         ("dctc", fig_compare), ("costo" if lang == "es" else "cost", fig_cost)):
            fig = fn(t, lang)
            path = os.path.join(t["dir"], f"cap14_{name}.png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", path)
    # the numbers quoted in the text
    for phi in (0.0, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi):
        rc, rs, res = dctc(phi)
        pz, px = readings(pctc_state(phi))
        dz, dx = readings(rs)
        print(f"phi {phi:.4f}: P-CTC success {PS.pctc_success(phi):.4f}, controlled success {(1 + math.cos(phi / 2) ** 2) / 2:.4f},"
              f" P-CTC (Z, X) = ({pz:.4f}, {px:.4f}); D-CTC (Z, X) = ({dz:.4f}, {dx:.4f}), C fixed point diag"
              f" {np.real(np.diag(rc)).round(4)}, residual {res:.1e}; check pctc_controlled_qg_z {PS.pctc_controlled_qg_z(phi):.4f}")
    # D-CTC fixed points are not unique: any rho_C commuting with RX(phi) works
    for start in ("plus", "zero"):
        v = np.array([1, 1], complex) / math.sqrt(2) if start == "plus" else np.array([1, 0], complex)
        rc = np.outer(v, v.conj())
        plus = np.array([1, 1], complex) / math.sqrt(2)
        U = crx(math.pi / 2)
        for _ in range(400):
            rc = ptrace(U @ np.kron(np.outer(plus, plus.conj()), rc) @ U.conj().T, 1)
        out = ptrace(U @ np.kron(np.outer(plus, plus.conj()), rc) @ U.conj().T, 0)
        print("D-CTC from", start, "at pi/2: fixed point", rc.round(4).tolist(), "S readings", [round(x, 4) for x in readings(out)])
    for phi in (math.pi / 2, 2 * math.pi / 3, math.pi - 0.1):
        print(f"cost at phi {phi:.3f}: uncontrolled {1 / PS.pctc_success(phi):.1f}x, controlled {2 / (1 + math.cos(phi / 2) ** 2):.2f}x")


if __name__ == "__main__":
    main()
