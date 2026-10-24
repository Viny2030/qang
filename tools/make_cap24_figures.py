"""Figures of chapter 24 of the book (Hamiltonian simulation, quantum walks and VarQTE read in qg), Spanish and English,
on the chapter 1 base.

    python tools/make_cap24_figures.py

writes book/es/figuras/cap24_*.png and book/en/figures/cap24_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. Recomputed here: the noiseless Trotter error and the walk's sigma^2 of §147 (with the
exact quantum walk and the classical random walk on the same chain), the §162 channels as affine maps of
(qg_X, qg_Y, qg_Z), the noiseless fidelity of the stored §162 VarQTE angles, the §60 share of the error the filter
removes for each noise and compilation, the §66 Gauss witnesses and the §29 spin witnesses of the noiseless circuits,
the §34 qg sum bound on the coherence and the §47 bounds of the RBM I-eta plane. The device-model results (§147, §148,
§162) are read from the stored JSON of the notes' pre-registered runs, not re-run. Runs in under a minute.
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples"))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402
from qang import channels as CH  # noqa: E402
from qang import varqte as VQ  # noqa: E402

import hubbard_trotter_qg_filters as HB  # noqa: E402
import ibm_trotter_walk_qg as W  # noqa: E402
import ising_coherence_witness_qg as IS  # noqa: E402
import lattice_gauge_gauss_qg as LG  # noqa: E402
import rbm_mutual_information_qg as RBM  # noqa: E402
import varqte_kraus_qg as VK  # noqa: E402
import xxz_trotter_filter_qg as XXZ  # noqa: E402

DATA = os.path.join(ROOT, "examples", "data")

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "w_x": "tiempo $t$",
        "w_y": r"$\sigma^2(t)$ (sitios$^2$)",
        "w_q": "caminata cuántica exacta",
        "w_c": "caminata clásica",
        "w_raw": "21 dispositivos, crudo (mediana)",
        "w_f": "21 dispositivos, filtro + lectura (mediana)",
        "w_mix": "estado mezclado",
        "w_title": r"Caminata en 7 qubits desde el centro",
        "e_y": r"exponente de $\sigma^2\propto t^a$",
        "e_raw": "crudo",
        "e_f": "filtro + lectura",
        "e_title": "Un punto por dispositivo simulado (notas §147)",
        "x_y": "error que quita el filtro",
        "x_cnot": "3 CNOT",
        "x_nc": "compuertas que conservan N",
        "x_noise": {"dep": "despolarizante\n2 qubits", "ad": "amortiguamiento\nde amplitud (T1)",
                    "deph": "desfase Z", "ro": "lectura"},
        "x_title": "XXZ de 6 qubits, 4 pasos: una fuente de ruido por vez",
        "i_x": "campo transversal $g$",
        "i_y": "fracción de la coherencia que capta la cota qg",
        "i_title": r"Ising de 6 qubits: $C\geq\sum_i[qg_S(qg_{Z,i})-qg_S(|r_i|)]$",
        "beta": r"$\beta$",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "w_x": "time $t$",
        "w_y": r"$\sigma^2(t)$ (sites$^2$)",
        "w_q": "exact quantum walk",
        "w_c": "classical random walk",
        "w_raw": "21 devices, raw (median)",
        "w_f": "21 devices, filter + readout (median)",
        "w_mix": "mixed state",
        "w_title": r"Walk on 7 qubits from the middle",
        "e_y": r"exponent of $\sigma^2\propto t^a$",
        "e_raw": "raw",
        "e_f": "filter + readout",
        "e_title": "One point per simulated device (notes §147)",
        "x_y": "error removed by the filter",
        "x_cnot": "3 CNOT",
        "x_nc": "N-conserving gates",
        "x_noise": {"dep": "2-qubit\ndepolarizing", "ad": "amplitude\ndamping (T1)", "deph": "Z dephasing",
                    "ro": "readout"},
        "x_title": "6-qubit XXZ, 4 steps: one noise source at a time",
        "i_x": "transverse field $g$",
        "i_y": "share of the coherence captured by the qg bound",
        "i_title": r"6-qubit Ising: $C\geq\sum_i[qg_S(qg_{Z,i})-qg_S(|r_i|)]$",
        "beta": r"$\beta$",
    },
}


def chain(n):
    """Weight-1 block of the XY chain with J = 1 and no fields: hopping 1 between neighbours."""
    return np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)


def walks(ts):
    n, mid = W.NB, W.MID
    A = chain(n)
    L = A - np.diag(A.sum(1))
    e = np.zeros(n)
    e[mid] = 1
    q = [W.sigma2(np.abs(expm(-1j * A * t) @ e) ** 2) for t in ts]
    c = [W.sigma2(expm(L * t) @ e) for t in ts]
    return np.array(q), np.array(c)


def walk_numbers():
    trot_err, s2_trot = W.theory()
    q, c = walks(W.TIMES)
    with open(os.path.join(DATA, "ibm_trotter_walk_qg.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    return trot_err, s2_trot, q, c, stored


def channel_check(reps=200, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    cases = {"ad": CH.amplitude_damping(0.3), "pd": CH.phase_damping(0.3), "dep": CH.depolarizing(0.2),
             "th": CH.thermal_relaxation(0.5, 1.0, 0.8, p1=0.1)}
    for k in cases.values():
        M, c = CH.bloch_affine(k)
        for _ in range(reps):
            v = rng.normal(size=3)
            r = v / np.linalg.norm(v) * rng.uniform(0, 1)
            rho = (np.eye(2) + r[0] * np.array([[0, 1], [1, 0]]) + r[1] * np.array([[0, -1j], [1j, 0]])
                   + r[2] * np.diag([1, -1])) / 2
            out = CH.apply_kraus(k, rho)
            r_out = np.array([2 * out[0, 1].real, -2 * out[0, 1].imag, (out[0, 0] - out[1, 1]).real])
            worst = max(worst, float(np.abs(r_out - (M @ r + c)).max()))
    g = 0.3
    Mad, cad = CH.bloch_affine(CH.amplitude_damping(g))
    Mth, cth = CH.bloch_affine(CH.thermal_relaxation(0.5, 1.0, 0.8, p1=0.1))
    return worst, Mad, cad, Mth, cth


def varqte_fidelity():
    with open(os.path.join(DATA, "varqte_kraus_qg.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    h = np.random.default_rng(VK.SEED).uniform(-0.5, 0.5, VK.N)
    H = VQ.xxz_matrix(VK.N, VK.J, VK.V, h)
    psi0 = VQ._state(np.zeros(VQ.n_params(VK.LAYERS)), VK.LAYERS, VK.INIT, VK.N)
    fid = {}
    for t, th in stored["thetas"].items():
        psi = VQ._state(np.array(th), VK.LAYERS, VK.INIT, VK.N)
        fid[t] = float(abs(np.vdot(expm(-1j * H * float(t)) @ psi0, psi)) ** 2)
    return fid, stored["fidelity"]


def xxz_shares():
    _, rows = XXZ.noise_decomposition()
    out = {}
    for noise, comp, raw, filt, kept in rows:
        out[(noise, comp)] = (1 - filt / raw, kept)
    return out


def gauge_and_hubbard():
    g = {s: LG.estimates(LG.probabilities(s)) for s in (1, 2, 4, 6)}
    hb = {s: HB.spin_witnesses(HB.ideal_probabilities(s)) for s in (1, 2, 4, 8)}
    return g, hb


def ising_grid(betas=(0.25, 0.5, 1.5, 3.0), gs=(0.25, 0.5, 1.0, 1.5, 2.0)):
    return {(b, g): IS.exact_row(b, g) for b in betas for g in gs}


def rbm_check():
    etas = np.linspace(0.05, 0.95, 19)
    lb_err, ub_err, min_ok = 0.0, 0.0, True
    for eta in etas:
        lb_err = max(lb_err, abs(RBM.mutual_information(0.0, 0.0, eta) - RBM.lower_bound(eta)))
        m = math.sqrt(1 - eta)
        ub_err = max(ub_err, abs(RBM.mutual_information(m, m, eta) - RBM.upper_bound(eta)))
        grid = np.linspace(-0.3, 0.3, 13)
        vals = [RBM.mutual_information(a, b, eta) for a in grid for b in grid
                if np.min(RBM.pair_distribution(a, b, eta)) > 1e-12]   # only pairs of spins that exist
        min_ok &= min(vals) >= RBM.lower_bound(eta) - 1e-12
    small = []
    for eta in (0.2, 0.4):
        for a in (0.02, 0.05):
            exact = RBM.mutual_information(a, a, eta) - RBM.lower_bound(eta)
            small.append(exact / (eta ** 2 * 2 * a * a / (2 * math.log(2))))
    return lb_err, ub_err, min_ok, small


def fig_walk(t, q_fine, c_fine, ts_fine, stored):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ts_fine, q_fine, color=BLUE, lw=2, label=t["w_q"])
    ax1.plot(ts_fine, c_fine, color=GREY, lw=2, ls="--", label=t["w_c"])
    rows = stored["rows"]
    raw = np.median([r["sigma2"]["raw"] for r in rows], axis=0)
    fil = np.median([r["sigma2"]["filtered_readout"] for r in rows], axis=0)
    ax1.plot(W.TIMES, raw, "o", color=ORANGE, ms=7, label=t["w_raw"])
    ax1.plot(W.TIMES, fil, "s", color=GREEN, ms=7, label=t["w_f"])
    ax1.axhline(4, color=INK, lw=0.8, ls=":")
    ax1.text(0.05, 4.1, t["w_mix"], fontsize=8, color=INK)
    ax1.set_xlabel(t["w_x"])
    ax1.set_ylabel(t["w_y"])
    ax1.set_title(t["w_title"], fontsize=10)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax1.grid(alpha=0.3)
    er = [r["exponent"]["raw"] for r in rows]
    ef = [r["exponent"]["filtered_readout"] for r in rows]
    for a, b in zip(er, ef):
        ax2.plot([0, 1], [a, b], color=GREY, lw=0.6, alpha=0.6)
    ax2.plot(np.zeros(len(er)), er, "o", color=ORANGE, ms=6)
    ax2.plot(np.ones(len(ef)), ef, "s", color=GREEN, ms=6)
    ax2.axhline(2, color=BLUE, lw=0.8, ls="--")
    ax2.axhline(1, color=GREY, lw=0.8, ls="--")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([t["e_raw"], t["e_f"]])
    ax2.set_xlim(-0.4, 1.4)
    ax2.set_ylabel(t["e_y"])
    ax2.set_title(t["e_title"], fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_filters(t, shares, ising):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    keys = (("dep", "2q depolarizing p2 = 0.01"), ("ad", "amplitude damping gamma = 0.01"),
            ("deph", "Z dephasing pz = 0.01"), ("ro", "readout e = 0.02"))
    x = np.arange(len(keys))
    cn = [shares.get((k, "3 CNOT"), (np.nan,))[0] for _, k in keys]
    nc = [shares.get((k, "N-conserving"), (np.nan,))[0] for _, k in keys]
    ax1.bar(x - 0.19, cn, 0.36, color=ORANGE, label=t["x_cnot"])
    ax1.bar(x + 0.19, nc, 0.36, color=BLUE, label=t["x_nc"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([t["x_noise"][a] for a, _ in keys], fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel(t["x_y"])
    ax1.set_title(t["x_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    betas = sorted({b for b, _ in ising})
    gs = sorted({g for _, g in ising})
    for b, c in zip(betas, (GREY, ORANGE, GREEN, BLUE)):
        ax2.plot(gs, [ising[(b, g)]["sum"] / ising[(b, g)]["C"] for g in gs], "o-", color=c,
                 label=t["beta"] + f" = {b:g}".replace(".", "," if t is TEXT["es"] else "."))
    ax2.set_ylim(0.85, 1.01)
    ax2.set_xlabel(t["i_x"])
    ax2.set_ylabel(t["i_y"])
    ax2.set_title(t["i_title"], fontsize=10, pad=12)
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    trot_err, s2_trot, q, c, stored = walk_numbers()
    print("§147 noiseless Trotter error, n = 5, T = 3, r =", W.RS, ":", " ".join(f"{e:.3f}" for e in trot_err))
    print("  walk sigma^2 at t =", W.TIMES)
    print("    exact quantum   ", " ".join(f"{x:.3f}" for x in q), " exponent", f"{W.exponent(q):.2f}")
    print("    Trotterized     ", " ".join(f"{x:.3f}" for x in s2_trot), " exponent", f"{W.exponent(s2_trot):.2f}")
    print("    classical       ", " ".join(f"{x:.3f}" for x in c), " exponent", f"{W.exponent(c):.2f}")
    s = stored["summary"]
    print("  stored device summary:", json.dumps({k: s[k] for k in ("median best error", "median r*", "median exponent",
                                                                    "exponent >= 1.5", "sigma2 within 20%")}))

    worst, Mad, cad, Mth, cth = channel_check()
    print(f"\n§162 channels: r' = M r + c holds to {worst:.1e} on 800 random states")
    print("  amplitude damping 0.3: M diag", np.round(np.diag(Mad), 4), " c", np.round(cad, 4))
    print("  thermal t=0.5, T1=1, T2=0.8, p1=0.1: M diag", np.round(np.diag(Mth), 4), " c", np.round(cth, 4),
          f" (e^-0.5 = {math.exp(-0.5):.4f}, e^-0.625 = {math.exp(-0.625):.4f})")

    fid, fid_stored = varqte_fidelity()
    print("\n§162 VarQTE noiseless fidelity from the stored angles:",
          {k: round(v, 4) for k, v in fid.items()}, " stored:", {k: round(v, 4) for k, v in fid_stored.items()})

    shares = xxz_shares()
    print("\n§60 share of the imbalance error the filter removes (n = 6, 4 steps), kept fraction:")
    for (noise, comp), (sh, kept) in shares.items():
        print(f"  {noise:32s} {comp:14s} {sh:6.1%}  kept {kept:.2f}")

    g, hb = gauge_and_hubbard()
    print("\n§66 noiseless Gauss witnesses (steps 1, 2, 4, 6):",
          [[round(x, 12) for x in g[s]["witness_G"]] for s in g], " kept fractions",
          [round(g[s]["kept_Gauss+N"], 12) for s in g])
    print("§29 noiseless Hubbard spin witnesses (mean qg_Z up, down; kept up, down) at steps 1, 2, 4, 8:",
          [tuple(round(x, 12) for x in hb[s]) for s in hb])

    ising = ising_grid()
    fr = [v["sum"] / v["C"] for v in ising.values()]
    print(f"\n§34 qg sum bound / exact coherence over {len(fr)} (beta, g): {min(fr):.1%}..{max(fr):.1%};"
          f" single-qubit bound {min(v['qubit'] / v['C'] for v in ising.values()):.1%}.."
          f"{max(v['qubit'] / v['C'] for v in ising.values()):.1%}; basis bound min {min(v['basis'] for v in ising.values()):.2f} bits")

    lb_err, ub_err, min_ok, small = rbm_check()
    print(f"\n§47 RBM: I at zero marginals = LB to {lb_err:.1e}; locked pair = UB to {ub_err:.1e};"
          f" LB is the minimum on the grid: {min_ok}; small-bias formula / exact gap: "
          + " ".join(f"{x:.3f}" for x in small))

    ts_fine = np.linspace(0.0, 1.8, 91)
    q_fine, c_fine = walks(ts_fine)
    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap24_caminata", "cap24_filtros"), "en": ("cap24_walk", "cap24_filters")}[lang]
        for name, fig in zip(names, (fig_walk(tx, q_fine, c_fine, ts_fine, stored), fig_filters(tx, shares, ising))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
