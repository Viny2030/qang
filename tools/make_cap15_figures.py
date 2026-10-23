"""Figures of chapter 15 of the book (Hayden-Preskill, the Yoshida-Kitaev decoder, a toy Page curve), Spanish and
English, on the chapter 1 base.

    python tools/make_cap15_figures.py

writes book/es/figuras/cap15_*.png and book/en/figures/cap15_*.png. Same style as make_cap01_figures.py. The 3-qubit
decoder is qang.postselection (§164) and its device points are the saved run
(examples/data/postselection_ctc_blackhole_qg.json), not new jobs. New in the book and computed here: the exact law
P F = 1/d_A^2 over random Cliffords and Haar unitaries, the decoder generalised to n-qubit black holes with k radiated
qubits (exact statevectors, plus gate counts and a noisy simulation on the FakeFez model; no device run), and a toy
Page curve read with Pauli correlators. Takes a few minutes (the noisy 16-qubit simulation).
"""

import itertools
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
        "d_x": r"éxito de la posselección, $m_P$ (m-qg)",
        "d_y": r"fidelidad del par $(R,R')$, $m_F$ (m-qg)",
        "d_law": r"ley exacta $(1000+m_P)(1000+m_F)=10^6$",
        "d_cl": "Cliffords aleatorios (exacto)",
        "d_haar": "unitarias de Haar (exacto)",
        "d_scr": "con scrambler",
        "d_id": "sin scrambling",
        "d_dep": "ruido total",
        "b_y": r"fidelidad decodificada, $m_F$ (m-qg)",
        "b_eq": r"$F=1/2$: el ecuador",
        "b_ex": "exacto",
        "s_x": r"qubits radiados, $k$",
        "s_y": r"fidelidad decodificada, $F$",
        "s_th": r"Haar, $F=1/(4P)$, $P=\frac{1}{4}+4^{-k}-4^{-k-1}$",
        "s_n": r"Haar, $n={n}$ (exacto)",
        "g_x": r"qubits del agujero negro, $n$",
        "g_y": "compuertas de dos qubits (FakeFez)",
        "g_cl": "scrambler Clifford",
        "g_haar": "unitaria de Haar",
        "p_x": r"qubits radiados, $k$ (de $n=8$)",
        "p_y": "entropía (bits)",
        "p_vn": "von Neumann (exacta)",
        "p_s2": r"Rényi-2 desde correladores qg, $-\log_2\frac{1}{2^k}\sum_P qg_P^2$",
        "p_sh": r"Shannon de la lectura en $Z$ (el $qg_S$ conjunto)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "d_x": r"postselection success, $m_P$ (m-qg)",
        "d_y": r"fidelity of the pair $(R,R')$, $m_F$ (m-qg)",
        "d_law": r"exact law $(1000+m_P)(1000+m_F)=10^6$",
        "d_cl": "random Cliffords (exact)",
        "d_haar": "Haar unitaries (exact)",
        "d_scr": "with scrambler",
        "d_id": "no scrambling",
        "d_dep": "full noise",
        "b_y": r"decoded fidelity, $m_F$ (m-qg)",
        "b_eq": r"$F=1/2$: the equator",
        "b_ex": "exact",
        "s_x": r"radiated qubits, $k$",
        "s_y": r"decoded fidelity, $F$",
        "s_th": r"Haar, $F=1/(4P)$, $P=\frac{1}{4}+4^{-k}-4^{-k-1}$",
        "s_n": r"Haar, $n={n}$ (exact)",
        "g_x": r"black-hole qubits, $n$",
        "g_y": "two-qubit gates (FakeFez)",
        "g_cl": "Clifford scrambler",
        "g_haar": "Haar unitary",
        "p_x": r"radiated qubits, $k$ (out of $n=8$)",
        "p_y": "entropy (bits)",
        "p_vn": "von Neumann (exact)",
        "p_s2": r"Rényi-2 from qg correlators, $-\log_2\frac{1}{2^k}\sum_P qg_P^2$",
        "p_sh": r"Shannon of the $Z$ reading (the joint $qg_S$)",
    },
}


def m(p):
    return 1000 * (2 * p - 1)


def load():
    with open(os.path.join(ROOT, "examples", "data", "postselection_ctc_blackhole_qg.json")) as fh:
        return {r["backend"]: r for r in json.load(fh)["rows"]}


# ---------------------------------------------------------------- the decoder for n-qubit black holes

def yk_circuit(U, n, k):
    """2n + 2 qubits: R, A, B (n - 1), B' (n - 1), A', R'. Bell pairs (R, A), (B_i, B'_i), (A', R'); U on (A, B),
    U* on (A', B'); the radiated qubits D are the last k wires of B. For n = 3, k = 1 it is PS.yk_decoder_circuit.
    U is a matrix or a Clifford; a Clifford goes in as its own circuit (and U* as that of its conjugate, equal up to a
    global phase), so a transpiler does not synthesise it as a generic unitary.
    Returns (circuit, the qubits of the (D, D') pairs, (R, R'))."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    R, A, Ap, Rp = 0, 1, 2 * n, 2 * n + 1
    B, Bp = list(range(2, n + 1)), list(range(n + 1, 2 * n))
    qc = QuantumCircuit(2 * n + 2)
    for a, b in [(R, A)] + list(zip(B, Bp)) + [(Ap, Rp)]:
        qc.h(a)
        qc.cx(a, b)
    if hasattr(U, "to_circuit"):
        qc.append(U.to_circuit().to_gate(label="U"), [A] + B)
        qc.append(U.conjugate().to_circuit().to_gate(label="U*"), [Ap] + Bp)
    else:
        qc.append(UnitaryGate(U, label="U"), [A] + B)
        qc.append(UnitaryGate(U.conj(), label="U*"), [Ap] + Bp)
    D = list(zip(B[-k:], Bp[-k:]))
    for a, b in D + [(R, Rp)]:
        qc.cx(a, b)
        qc.h(a)
    return qc, [q for pr in D for q in pr], (R, Rp)


def tally(probs, dq, rr):
    """(P, F) from a {bitstring: weight} dict: P(all (D, D') read 00) and P((R, R') reads 00 | that)."""
    tot = P = F = 0.0
    for key, v in probs.items():
        b = key.replace(" ", "")
        bit = lambda q: b[-1 - q]  # noqa: E731
        tot += v
        if all(bit(q) == "0" for q in dq):
            P += v
            if bit(rr[0]) == "0" and bit(rr[1]) == "0":
                F += v
    return P / tot, F / P


def exact(U, n, k):
    from qiskit.quantum_info import Statevector

    qc, dq, rr = yk_circuit(U, n, k)
    return tally(Statevector(qc).probabilities_dict(), dq, rr)


def rr_correlators(U, n, k):
    """qg_XX, qg_YY, qg_ZZ of (R, R') after the postselection on (D, D') (the Bell measurement of (R, R') left out)."""
    from qiskit.quantum_info import Statevector

    out = []
    for P in "XYZ":
        qc, dq, rr = yk_circuit(U, n, k)
        # undo the Bell rotation of (R, R') so the pair is read directly, then rotate both qubits to the P basis
        qc.h(rr[0])
        qc.cx(rr[0], rr[1])
        for q in rr:
            if P == "X":
                qc.h(q)
            elif P == "Y":
                qc.sdg(q)
                qc.h(q)
        num = den = 0.0
        for key, v in Statevector(qc).probabilities_dict().items():
            bit = lambda q: key[-1 - q]  # noqa: E731
            if all(bit(q) == "0" for q in dq):
                den += v
                num += v * (1 if bit(rr[0]) == bit(rr[1]) else -1)
        out.append(num / den)
    return out


def random_cliffords(n, count, seed):
    from qiskit.quantum_info import random_clifford

    rng = np.random.default_rng(seed)
    return [random_clifford(n, seed=int(rng.integers(1e9))) for _ in range(count)]


def haars(n, count, seed):
    from qiskit.quantum_info import random_unitary

    rng = np.random.default_rng(seed)
    return [random_unitary(2 ** n, seed=int(rng.integers(1e9))).data for _ in range(count)]


def first_perfect(n, k, seed):
    """First random Clifford (from seed) that decodes with fidelity 1 when k qubits are radiated."""
    for U in random_cliffords(n, 200, seed):
        if exact(U, n, k)[1] > 1 - 1e-9:
            return U
    raise RuntimeError("no perfect decoder found")


def haar_p(k):
    """Haar average of the success, large n, d_A = 2: 1/d_A^2 + 1/d_D^2 - 1/(d_A^2 d_D^2), d_D = 2^k."""
    return 0.25 + 4.0 ** -k - 4.0 ** (-k - 1)


def noisy(U, n, k, shots=8000):
    """(two-qubit gates, (P, F)) of the decoder transpiled for FakeFez (optimization_level 1, seed 137) and run on its
    noise model (Aer, seed 15). A simulated device, not hardware."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez

    be = FakeFez()
    qc, dq, rr = yk_circuit(U, n, k)
    qc.measure_all()
    t = transpile(qc, be, optimization_level=1, seed_transpiler=137)
    g2 = sum(v for name, v in t.count_ops().items() if name in ("cz", "cx", "ecr"))
    res = None
    if shots:
        c = AerSimulator.from_backend(be).run(t, shots=shots, seed_simulator=15).result().get_counts()
        res = tally(c, dq, rr)
    return g2, res


# ---------------------------------------------------------------- figures

def fig_decoder(t, lang):
    rows = load()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    mp = np.linspace(-520, 1000, 300)
    a1.plot(mp, 1e6 / (1000 + mp) - 1000, color=INK, lw=2, label=t["d_law"])
    cl = [exact(U, 3, 1) for U in random_cliffords(3, 60, 1503)]
    a1.plot([m(p) for p, _ in cl], [m(f) for _, f in cl], "D", color=GREY, ms=9, mfc="none", label=t["d_cl"])
    hr = [exact(U, 3, 1) for U in haars(3, 25, 1504)]
    a1.plot([m(p) for p, _ in hr], [m(f) for _, f in hr], ".", color=GREY, ms=6, label=t["d_haar"])
    for d in DEVICES:
        s, i = rows[d]["H"]["scrambler"], rows[d]["H"]["identity"]
        a1.plot(m(s["success"]), m(s["fidelity"]), MARKERS[d], color=COLORS[d], ms=8, label=f"{d}, {t['d_scr']}")
        a1.plot(m(i["success"]), m(i["fidelity"]), MARKERS[d], color=COLORS[d], ms=8, mfc="white", mew=1.4)
    a1.plot(-500, -500, "x", color=INK, ms=9, mew=2, label=t["d_dep"])
    a1.set_xlim(-560, 1040)
    a1.set_ylim(-560, 1040)
    a1.set_xlabel(t["d_x"], color=INK)
    a1.set_ylabel(t["d_y"], color=INK)
    a1.grid(alpha=0.25)
    a1.legend(fontsize=7.6, loc="upper right")

    labels = [t["b_ex"]] + [d.replace("Fake", "") for d in DEVICES]
    scr = [1000] + [m(rows[d]["H"]["scrambler"]["fidelity"]) for d in DEVICES]
    ide = [-500] + [m(rows[d]["H"]["identity"]["fidelity"]) for d in DEVICES]
    x = np.arange(len(labels))
    a2.bar(x - 0.19, scr, 0.38, color=BLUE, label=t["d_scr"])
    a2.bar(x + 0.19, ide, 0.38, color=ORANGE, label=t["d_id"])
    a2.axhline(0, color=INK, lw=1.4, ls="--")
    a2.text(3.45, 25, t["b_eq"], ha="right", va="bottom", fontsize=8.5, color=INK)
    a2.set_xticks(x)
    a2.set_xticklabels(labels)
    a2.set_ylabel(t["b_y"], color=INK)
    a2.set_ylim(-600, 1080)
    a2.grid(alpha=0.25, axis="y")
    a2.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    return fig


SIZES = os.path.join(ROOT, "examples", "data", "cap15_yk_sizes.json")


def size_data():
    """Exact Haar averages, gate counts and the noisy FakeFez runs of the larger decoders. Several minutes, so the result
    is saved to examples/data/cap15_yk_sizes.json and reused; delete the file to recompute."""
    if os.path.exists(SIZES):
        with open(SIZES) as fh:
            raw = json.load(fh)
        return {part: {tuple(int(x) if x.isdigit() else x for x in key.split(",")): (tuple(v) if isinstance(v, list) else v)
                       for key, v in raw[part].items()} for part in raw}
    out = {"haar": {}, "gates": {}, "noisy": {}}
    for n in (3, 5, 7):
        for k in range(1, min(n, 4)):
            vals = [exact(U, n, k) for U in haars(n, 6, 1500 + 10 * n + k)]
            out["haar"][(n, k)] = (float(np.mean([p for p, _ in vals])), float(np.mean([f for _, f in vals])))
    for n, k, seed in ((3, 1, None), (5, 2, 155), (7, 3, 157)):
        from qiskit.quantum_info import Clifford

        U = Clifford(PS.find_scrambler()) if seed is None else first_perfect(n, k, seed)
        out["noisy"][(n, k)] = noisy(U, n, k)
        out["gates"][("clifford", n)] = out["noisy"][(n, k)][0]
        out["gates"][("haar", n)] = noisy(haars(n, 1, 1600 + n)[0], n, k, shots=0)[0]
    with open(SIZES, "w") as fh:
        json.dump({part: {",".join(map(str, key)): v for key, v in out[part].items()} for part in out}, fh, indent=1)
    return out


def fig_size(t, lang, data):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ks = np.arange(1, 4)
    a1.plot(ks, [1 / (4 * haar_p(k)) for k in ks], color=INK, lw=2, label=t["s_th"])
    for n, col, mk in ((3, BLUE, "o"), (5, ORANGE, "s"), (7, GREEN, "^")):
        pts = sorted((k, v[1]) for (nn, k), v in data["haar"].items() if nn == n)
        a1.plot([k for k, _ in pts], [f for _, f in pts], mk, color=col, ms=8, label=t["s_n"].format(n=n))
    a1.axhline(0.5, color=GREY, lw=1, ls="--")
    a1.set_xticks(ks)
    a1.set_ylim(0.45, 1.02)
    a1.set_xlabel(t["s_x"], color=INK)
    a1.set_ylabel(t["s_y"], color=INK)
    a1.grid(alpha=0.25)
    a1.legend(fontsize=8.2, loc="lower right")

    ns = [3, 5, 7]
    a2.semilogy(ns, [data["gates"][("clifford", n)] for n in ns], "o-", color=BLUE, lw=2, label=t["g_cl"])
    a2.semilogy(ns, [data["gates"][("haar", n)] for n in ns], "s-", color=ORANGE, lw=2, label=t["g_haar"])
    a2.set_xticks(ns)
    a2.set_xlabel(t["g_x"], color=INK)
    a2.set_ylabel(t["g_y"], color=INK)
    a2.grid(alpha=0.25, which="both")
    a2.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def page_data(n=8, seed=15):
    from qiskit.quantum_info import DensityMatrix, Pauli, entropy, partial_trace, random_statevector

    psi = random_statevector(2 ** n, seed=seed)
    rows = [(0, 0.0, 0.0, 0.0, 0)]
    for k in range(1, n + 1):
        rho = partial_trace(psi, list(range(k, n))) if k < n else DensityMatrix(psi)
        s_vn = float(entropy(rho, base=2))
        if k <= 4:  # the purity from the 4^k Pauli correlators of the radiation
            pur = sum(float(np.real(rho.expectation_value(Pauli("".join(lab))))) ** 2
                      for lab in itertools.product("IXYZ", repeat=k)) / 2 ** k
        else:
            pur = float(np.real(np.trace(rho.data @ rho.data)))
        p = np.real(np.diag(rho.data))
        sh = float(-sum(x * math.log2(x) for x in p if x > 1e-15))
        rows.append((k, s_vn, -math.log2(pur), sh, 4 ** k - 1))
    return rows


def fig_page(t, lang, rows):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    k = [r[0] for r in rows]
    ax.plot(k, [r[1] for r in rows], "o-", color=BLUE, lw=2.2, label=t["p_vn"])
    ax.plot(k, [r[2] for r in rows], "s--", color=GREEN, lw=1.8, label=t["p_s2"])
    ax.plot(k, [r[3] for r in rows], "^:", color=ORANGE, lw=1.8, label=t["p_sh"])
    ax.set_xlabel(t["p_x"], color=INK)
    ax.set_ylabel(t["p_y"], color=INK)
    ax.set_xticks(k)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    from qiskit.quantum_info import Clifford, Pauli

    sizes = size_data()
    page = page_data()
    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        figs = (("decodificador" if lang == "es" else "decoder", lambda: fig_decoder(t, lang)),
                ("tamano" if lang == "es" else "size", lambda: fig_size(t, lang, sizes)),
                ("page", lambda: fig_page(t, lang, page)))
        for name, fn in figs:
            fig = fn()
            path = os.path.join(t["dir"], f"cap15_{name}.png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", path)

    # the numbers quoted in the text
    scr = PS.find_scrambler()
    U = Clifford(scr)
    weights = []
    for q in range(3):
        for P in "XYZ":
            lab = ["I"] * 3
            lab[2 - q] = P
            w = Pauli("".join(lab)).evolve(scr, frame="s").to_label().lstrip("-i")
            weights.append(sum(ch != "I" for ch in w))
    print("scrambler: weights of the 9 single-qubit Paulis", weights, "is_scrambler", PS.is_scrambler(U))
    for name, V in (("scrambler", U), ("identity", np.eye(8))):
        p, f = exact(V, 3, 1)
        xx, yy, zz = rr_correlators(V, 3, 1)
        print(f"{name}: P {p:.4f} (m {m(p):.0f}, angle {math.acos(2 * p - 1):.4f}), F {f:.4f} (m {m(f):.0f}),"
              f" P F {p * f:.4f}; (R,R') qg_XX {xx:.3f} qg_YY {yy:.3f} qg_ZZ {zz:.3f} -> (1+XX-YY+ZZ)/4 = {(1 + xx - yy + zz) / 4:.4f}")
    cl = [exact(V, 3, 1) for V in random_cliffords(3, 60, 1503)]
    print("random 3-qubit Cliffords, k=1: perfect decoders", sum(f > 1 - 1e-9 for _, f in cl), "of", len(cl),
          "; (P, F) values", sorted(set((round(p, 4), round(f, 4)) for p, f in cl)),
          "; max |P F - 1/4|", max(abs(p * f - 0.25) for p, f in cl))
    hr = [exact(V, 3, 1) for V in haars(3, 25, 1504)]
    print("Haar n=3 k=1: max |P F - 1/4|", max(abs(p * f - 0.25) for p, f in hr))
    rows = load()
    for d in DEVICES:
        s, i = rows[d]["H"]["scrambler"], rows[d]["H"]["identity"]
        print(f"{d}: scrambler P {s['success']:.4f} (m {m(s['success']):.0f}) F {s['fidelity']:.4f} (m {m(s['fidelity']):.0f})"
              f" 4PF {4 * s['success'] * s['fidelity']:.3f}; identity P {i['success']:.4f} F {i['fidelity']:.4f}"
              f" (m {m(i['fidelity']):.0f}) 4PF {4 * i['success'] * i['fidelity']:.3f}; 2q gates {rows[d]['H_2q']}")
    for (n, k), (p, f) in sorted(sizes["haar"].items()):
        print(f"Haar n={n} k={k}: P {p:.4f} F {f:.4f} (theory P {haar_p(k):.4f}, F {1 / (4 * haar_p(k)):.4f})")
    for (n, k), (g2, pf) in sorted(sizes["noisy"].items()):
        print(f"perfect Clifford decoder n={n} k={k} on FakeFez: {g2} two-qubit gates, P {pf[0]:.4f} F {pf[1]:.4f}"
              f" (m {m(pf[1]):.0f}), 4PF {4 * pf[0] * pf[1]:.3f}; Haar n={n}: {sizes['gates'][('haar', n)]} gates")
    for r in page:
        print(f"Page n=8, k={r[0]}: S_vN {r[1]:.3f}, S_2 {r[2]:.3f}, Z Shannon {r[3]:.3f}, correlators {r[4]}")


if __name__ == "__main__":
    main()
