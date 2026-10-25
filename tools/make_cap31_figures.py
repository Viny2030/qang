"""Figures of chapter 31 of the book (the first real-hardware experiment: IQM Resonance), Spanish and English.

    python tools/make_cap31_figures.py

writes book/es/figuras/cap31_*.png and book/en/figures/cap31_*.png and prints the numbers the chapter quotes.

* Compilation: an RBS gate, an XX+YY gate and the three-flavour neutrino circuit of §151 transpiled to IQM's native
  gates (r, cz) and to IBM's (cz or ecr, with rz, sx, x) on a line of three qubits, at optimization levels 1 and 3.
  The IQM free rehearsal used level 3, the IBM runs of §151 level 1.
* The run of §165 (job 01a15b04, Garnet, 8 x 300 shots) rescored from examples/data/iqm_neutrino_qg_run.json:
  error per flavour, raw and filtered, and where each reading puts the oscillation maximum.
* The credit ledger of the two Garnet runs, from the balances written in examples/iqm_neutrino_qg.py and in
  §168 of the notes (the cost of the neutrino run is not written anywhere: it is the difference of two balances).
iqm-client is not needed: nothing is sent to IQM and the fake backends are not rerun (their numbers are quoted).
"""

import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA = os.path.join(ROOT, "examples", "data")
L, DCP = 1300.0, -np.pi / 2
FLAVOURS = ("e", "mu", "tau")
# (step, balance after it) — iqm_neutrino_qg.py docstring and notes §168
LEDGER = [("start", 30.00), ("nu probe", 29.50), ("nu run", 29.00), ("m-qg probe", 28.50), ("m-qg run", 12.50)]

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "n_x": "energía (GeV)",
        "n_y": "probabilidad",
        "n_title": {"e": r"$P(\nu_\mu\to\nu_e)$", "mu": r"$P(\nu_\mu\to\nu_\mu)$", "tau": r"$P(\nu_\mu\to\nu_\tau)$"},
        "n_exact": "exacta",
        "n_raw": "cruda (sin qang)",
        "n_fil": "filtrada (con qang)",
        "b_y": "créditos",
        "b_steps": ["prueba ν\n100 disparos", "corrida ν\n2400", "prueba m-qg\n100", "corrida m-qg\n104 000", "quedan"],
        "b_title": "Créditos de IQM Resonance: 30 iniciales",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "n_x": "energy (GeV)",
        "n_y": "probability",
        "n_title": {"e": r"$P(\nu_\mu\to\nu_e)$", "mu": r"$P(\nu_\mu\to\nu_\mu)$", "tau": r"$P(\nu_\mu\to\nu_\tau)$"},
        "n_exact": "exact",
        "n_raw": "raw (without qang)",
        "n_fil": "filtered (with qang)",
        "b_y": "credits",
        "b_steps": ["ν probe\n100 shots", "ν run\n2400", "m-qg probe\n100", "m-qg run\n104 000", "left"],
        "b_title": "IQM Resonance credits: 30 to start",
    },
}


def two_qubit(qc):
    return sum(1 for i in qc.data if i.operation.num_qubits == 2)


def compilation():
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.transpiler import CouplingMap

    from qang import ibm
    from qang import neutrino as nu

    rbs = QuantumCircuit(2)
    ibm._rbs(rbs, 0.7, 0, 1)
    xy = QuantumCircuit(2)
    xy.append(XXPlusYYGate(1.1, 0.3), [0, 1])
    neu = nu.three_flavour_circuit("mu", L, 2.6, DCP)
    line = CouplingMap.from_line(3)
    bases = {"IQM (r, cz)": ["r", "cz"], "IBM Heron (cz, rz, sx, x)": ["cz", "rz", "sx", "x"],
             "IBM Eagle (ecr, rz, sx, x)": ["ecr", "rz", "sx", "x"]}
    out = {}
    for name, b in bases.items():
        row = {}
        for lvl in (1, 3):
            kw = dict(basis_gates=b, optimization_level=lvl, seed_transpiler=137)
            row[lvl] = (two_qubit(transpile(rbs, **kw)), two_qubit(transpile(xy, **kw)),
                        two_qubit(transpile(neu, coupling_map=line, **kw)))
        out[name] = row
    return out


def run_rescore():
    from qang import neutrino as nu

    with open(os.path.join(DATA, "iqm_neutrino_qg_run.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    E = np.array(d["energies"])
    raw, fil, ex = (np.array(d[k]) for k in ("raw", "filtered", "exact"))
    check = np.array([nu.three_flavour_probabilities(L, e, DCP)[1] for e in E])
    per = {f: (float(np.abs(raw[:, i] - ex[:, i]).mean()), float(np.abs(fil[:, i] - ex[:, i]).mean()))
           for i, f in enumerate(FLAVOURS)}
    return {"E": E, "raw": raw, "fil": fil, "exact": ex, "exact_check": float(np.abs(check - ex).max()),
            "raw_error": float(np.abs(raw - ex).mean()), "fil_error": float(np.abs(fil - ex).mean()),
            "kept": d["kept"], "per": per, "min_raw": float(E[np.argmin(raw[:, 1])]),
            "min_fil": float(E[np.argmin(fil[:, 1])]), "shots": d["shots"], "job": d["job_id"]}


def fig_neutrinos(t, r):
    from qang import neutrino as nu

    Ed = np.linspace(0.6, 5.2, 300)
    Pd = np.array([nu.three_flavour_probabilities(L, e, DCP)[1] for e in Ed])
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5), sharey=True)
    for i, (f, ax) in enumerate(zip(FLAVOURS, axs)):
        ax.plot(Ed, Pd[:, i], color=INK, lw=1.4, label=t["n_exact"])
        ax.plot(r["E"], r["raw"][:, i], "o", color=GREY, ms=5, label=t["n_raw"])
        ax.plot(r["E"], r["fil"][:, i], "s", color=BLUE, ms=5, label=t["n_fil"])
        ax.set_title(t["n_title"][f], color=INK, fontsize=11)
        ax.set_xlabel(t["n_x"], color=INK)
        ax.grid(alpha=0.25)
    axs[1].axvline(2.63, color=ORANGE, lw=1, ls=":")
    axs[0].set_ylabel(t["n_y"], color=INK)
    axs[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    return fig


def fig_budget(t):
    spent = [LEDGER[i - 1][1] - LEDGER[i][1] for i in range(1, len(LEDGER))] + [LEDGER[-1][1]]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    cols = [GREY, BLUE, GREY, BLUE, GREEN]
    x = np.arange(len(spent))
    ax.bar(x, spent, color=cols)
    for xi, s in zip(x, spent):
        ax.text(xi, s + 0.3, f"{s:.2f}".replace(".", "," if t is TEXT["es"] else "."), ha="center", fontsize=9,
                color=INK)
    ax.set_xticks(x, t["b_steps"], fontsize=8.5)
    ax.set_ylabel(t["b_y"], color=INK)
    ax.set_ylim(0, 18.5)
    ax.set_title(t["b_title"], color=INK, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def main():
    print("two-qubit gates after compilation (RBS, XX+YY, neutrino circuit on a line of 3):")
    for name, row in compilation().items():
        print(f"  {name:28s} level 1 {row[1]}   level 3 {row[3]}")
    r = run_rescore()
    print(f"\n§165 rescored (job {r['job']}, {r['shots']} shots): exact column check {r['exact_check']:.1e}")
    print(f"  mean error raw {r['raw_error']:.4f}, filtered {r['fil_error']:.4f}, ratio "
          f"{r['raw_error'] / r['fil_error']:.2f}, kept {r['kept']:.3f}")
    for f, (a, b) in r["per"].items():
        print(f"  nu_{f:3s} raw {a:.3f} -> filtered {b:.3f}")
    print(f"  P(mu->mu) lowest at {r['min_raw']} GeV raw, {r['min_fil']} GeV filtered (exact maximum 2.63)")
    i26 = int(np.where(r["E"] == 2.6)[0][0])
    print(f"  P(mu->mu) at 2.6 GeV: exact {r['exact'][i26, 1]:.3f}, raw {r['raw'][i26, 1]:.3f}, "
          f"filtered {r['fil'][i26, 1]:.3f}")
    spent = {LEDGER[i][0]: LEDGER[i - 1][1] - LEDGER[i][1] for i in range(1, len(LEDGER))}
    total = LEDGER[0][1]
    print("\ncredits:", spent, "left", LEDGER[-1][1])
    print(f"  shares: probes {(spent['nu probe'] + spent['m-qg probe']) / total:.1%}, runs "
          f"{(spent['nu run'] + spent['m-qg run']) / total:.1%}, left {LEDGER[-1][1] / total:.1%}")
    print(f"  m-qg run: {spent['m-qg run'] / 104:.3f} credits per 1000 shots")
    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap31_neutrinos", "cap31_creditos"), "en": ("cap31_neutrinos", "cap31_credits")}[lang]
        for name, fig in zip(names, (fig_neutrinos(tx, r), fig_budget(tx))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
