"""Figures of chapter 20 of the book (weight-conserving quantum neural networks, WeightQNN, read in qg), Spanish and
English, on the chapter 1 base.

    python tools/make_cap20_figures.py

writes book/es/figuras/cap20_*.png and book/en/figures/cap20_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. What is computed here is structural and exact, on random inputs and parameters (no
datasets, so no scikit-learn): the filter's exactness under equal T1 in the weight-1 and weight-2 sectors, the kept
fraction (1 - gamma)^(k depth), the bias left by unequal T1 and dephasing, the rank of the qg_Z and qg_ZZ readouts,
the classical (quadratic) twin of the weight-1 network, and in PennyLane the gradient of the filtered readout under T1.
The classification accuracies of the chapter are the pre-registered results of the notes (§75-§114), quoted, not
rerun here; the accuracy figure plots the table of §111. Runs in about half a minute.
"""

import math
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402
from qang import qml as QML  # noqa: E402
from qang.sectors import filter_distribution  # noqa: E402

RED = "#8c2d04"
GAMMA = 0.08

# §111 (notes): n, weight, noiseless, equal T1 with qang, without, noise-aware without qang.
SCALING = [(5, 1, 0.958, 0.958, 0.924, 0.961), (5, 2, 0.948, 0.948, 0.861, 0.954),
           (6, 1, 0.969, 0.969, 0.937, 0.965), (6, 2, 0.957, 0.957, 0.878, 0.954),
           (8, 1, 0.977, 0.977, 0.946, 0.968), (8, 2, 0.965, 0.965, 0.705, 0.968)]

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "r_x": "qubit",
        "r_y": r"lectura $m_Z$ de cada qubit (m-qg)",
        "r_clean": "sin ruido",
        "r_raw": r"T1, sin filtro",
        "r_filt": "T1, con el filtro qg",
        "r_title": r"Peso 2, codificación dual, $\gamma=0{,}08$ por subcapa",
        "k_x": "subcapas de RBS",
        "k_y": "fracción de disparos que conserva el filtro",
        "k_law": r"ley $(1-\gamma)^{k\,d}$",
        "k_sim": "simulado",
        "k_title": r"El costo del filtro, fijado antes de correr ($\gamma=0{,}08$)",
        "a_x": "qubits",
        "a_y": "precisión media de prueba",
        "a_clean": "sin ruido = con qang bajo T1",
        "a_raw": "sin qang, entrenada sin ruido",
        "a_aware": "sin qang, entrenada con el ruido",
        "a_w": "peso {k}",
        "a_title": "Notas §111: T1 igual, $\\gamma=0{,}08$, 9 subcapas",
        "g_x": r"ángulo $\theta_1$ (rad)",
        "g_y": r"$\partial\,qg_Z/\partial\theta_1$",
        "g_clean": "sin ruido",
        "g_filt": "T1, filtrada",
        "g_raw": "T1, sin filtro",
        "g_title": "PennyLane: el gradiente filtrado es el de sin ruido",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "r_x": "qubit",
        "r_y": r"$m_Z$ reading of each qubit (m-qg)",
        "r_clean": "noiseless",
        "r_raw": r"T1, no filter",
        "r_filt": "T1, with the qg filter",
        "r_title": r"Weight 2, dual encoding, $\gamma=0.08$ per sublayer",
        "k_x": "RBS sublayers",
        "k_y": "fraction of shots the filter keeps",
        "k_law": r"law $(1-\gamma)^{k\,d}$",
        "k_sim": "simulated",
        "k_title": r"The filter's cost, fixed before running ($\gamma=0.08$)",
        "a_x": "qubits",
        "a_y": "mean test accuracy",
        "a_clean": "noiseless = with qang under T1",
        "a_raw": "without qang, trained noiseless",
        "a_aware": "without qang, trained under the noise",
        "a_w": "weight {k}",
        "a_title": "Notes §111: equal T1, $\\gamma=0.08$, 9 sublayers",
        "g_x": r"angle $\theta_1$ (rad)",
        "g_y": r"$\partial\,qg_Z/\partial\theta_1$",
        "g_clean": "noiseless",
        "g_filt": "T1, filtered",
        "g_raw": "T1, no filter",
        "g_title": "PennyLane: the filtered gradient is the noiseless one",
    },
}


def random_setup(model, n_inputs=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_inputs, model.n - 1))
    theta = rng.uniform(-math.pi, math.pi, size=model.n_theta)
    return model.encode(X), theta


def exactness(model, seed=0):
    """Max |filtered - noiseless| of every readout feature (m-qg), raw error, kept fraction vs the law."""
    psi, theta = random_setup(model, seed=seed)
    clean = model.qg_z(model.probs(theta, psi), qang=False)
    pr = model.probs(theta, psi, gamma=GAMMA)
    filt = model.qg_z(pr, qang=True)
    raw = model.qg_z(pr, qang=False)
    kept = np.array([filter_distribution(p, model.n, model.weight)[1] for p in pr])
    law = QML.kept_fraction(GAMMA, model.weight, model.depth)
    return (1000 * np.abs(filt - clean).max(), 1000 * np.abs(raw - clean).mean(), float(kept.min()),
            float(kept.max()), law)


def residual_bias(model, seed=0):
    """Mean |filtered - noiseless| (m-qg) under unequal T1 (spread 0.5) and under T1 + dephasing 0.01."""
    psi, theta = random_setup(model, seed=seed)
    clean = model.qg_z(model.probs(theta, psi), qang=False)
    u = np.linspace(-1, 1, model.n)
    out = {}
    for name, kw in (("unequal", {"gamma": GAMMA * (1 + 0.5 * u)}), ("dephasing", {"gamma": GAMMA, "dephasing": 0.01})):
        f = model.qg_z(model.probs(theta, psi, **kw), qang=True)
        out[name] = 1000 * float(np.abs(f - clean).mean())
    return out


def readout_rank(model):
    """Rank of the readout as a linear map on the sector distribution (constant term included)."""
    F = model.features[model.idx[model.weight]]
    return int(np.linalg.matrix_rank(np.hstack([F, np.ones((len(F), 1))])))


def quadratic_twin(seed=0):
    """Weight 1: z = sum c_i (1 - 2 u_i^2) + b with u = O v, O the 5x5 orthogonal block (F2, §76)."""
    m = QML.WeightQNN(n_qubits=5, weight=1, layers=3)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(40, 4))
    params = rng.normal(size=m.n_theta + m.n + 1)
    z = m.decision(params, X)
    O = np.eye(m.n)
    for blocks in m.block_unitaries(params[: m.n_theta]):
        O = blocks[1] @ O
    v = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # the weight-1 block is indexed by basis states in increasing index; qubit q is bit (n-1-q)
    states = m.idx[1]
    qubit_of_state = [m.n - 1 - int(math.log2(s)) for s in states]
    v_block = v[:, qubit_of_state]
    u = v_block @ O.T
    p_q = np.zeros_like(u)
    p_q[:, qubit_of_state] = u**2
    c, b = params[m.n_theta:-1], params[-1]
    z_twin = (1 - 2 * p_q) @ c + b
    return float(np.abs(z - z_twin).max()), np.linalg.norm(O @ O.T - np.eye(m.n))


def pennylane_gradients(gamma=GAMMA):
    """3-qubit weight-1 circuit of SingleExcitation (Givens) gates with amplitude damping on default.mixed:
    gradient of qubit 1's qg_Z, noiseless, filtered under T1 and raw under T1 (parameter shift)."""
    import pennylane as qml
    from pennylane import numpy as pnp

    from qang import pennylane_gate as PG

    dev = qml.device("default.mixed", wires=3)

    def make(g):
        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(th):
            qml.PauliX(0)
            for layer in range(2):
                qml.SingleExcitation(th[2 * layer], wires=[0, 1])
                for w in range(3):
                    if g:
                        qml.AmplitudeDamping(g, wires=w)
                qml.SingleExcitation(th[2 * layer + 1], wires=[1, 2])
                for w in range(3):
                    if g:
                        qml.AmplitudeDamping(g, wires=w)
            return qml.probs(wires=range(3))

        return circ

    c0, c1 = make(0.0), make(gamma)
    fns = {"clean": lambda t: PG.qg_z_from_probs(c0(t))[1], "filt": lambda t: PG.filtered_qg_z(c1(t), 1)[1],
           "raw": lambda t: PG.qg_z_from_probs(c1(t))[1]}
    base = [0.3, 1.1, -0.7, 0.5]
    sweep = np.linspace(-math.pi, math.pi, 25)
    curves = {k: [] for k in fns}
    for a in sweep:
        th = pnp.array([a] + base[1:], requires_grad=True)
        for k, f in fns.items():
            curves[k].append(float(qml.grad(f)(th)[0]))
    th = pnp.array(base, requires_grad=True)
    kept = float(PG.kept_fraction(c1(th), 1))
    g = {k: np.array(qml.grad(f)(th)) for k, f in fns.items()}

    # backprop on default.mixed through SingleExcitation (reported, not used)
    @qml.qnode(qml.device("default.mixed", wires=2), diff_method="backprop")
    def one(t):
        qml.PauliX(0)
        qml.SingleExcitation(t, wires=[0, 1])
        return qml.probs(wires=[0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bp = qml.jacobian(one)(pnp.array(0.4, requires_grad=True))
    return sweep, curves, g, kept, bp, qml.__version__


def fig_filter(t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    m = QML.WeightQNN(n_qubits=5, weight=2, layers=3, encoding="dual")
    rng = np.random.default_rng(3)
    psi = m.encode(rng.uniform(-1, 1, size=(1, 4)))
    theta = rng.uniform(-math.pi, math.pi, size=m.n_theta)
    clean = 1000 * m.qg_z(m.probs(theta, psi), qang=False)[0]
    pr = m.probs(theta, psi, gamma=GAMMA)
    raw, filt = 1000 * m.qg_z(pr, qang=False)[0], 1000 * m.qg_z(pr, qang=True)[0]
    q = np.arange(m.n)
    w = 0.27
    ax1.bar(q - w, clean, w, color=GREY, label=t["r_clean"])
    ax1.bar(q, raw, w, color=ORANGE, label=t["r_raw"])
    ax1.bar(q + w, filt, w, color=BLUE, label=t["r_filt"])
    ax1.axhline(0, color=INK, lw=0.6)
    ax1.set_xticks(q)
    ax1.set_xlabel(t["r_x"])
    ax1.set_ylabel(t["r_y"])
    ax1.set_ylim(-1000, 1100)
    ax1.set_title(t["r_title"], fontsize=10)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax1.grid(axis="y", alpha=0.3)
    depths = np.arange(0, 31)
    for k, c in ((1, BLUE), (2, GREEN), (3, ORANGE)):
        ax2.plot(depths, (1 - GAMMA) ** (k * depths), color=c, lw=2, label=f"k = {k}: {t['k_law']}")
    for n, k, mk in ((5, 1, "o"), (5, 2, "s"), (8, 1, "^"), (8, 2, "D")):
        for layers in (1, 2, 3):
            mm = QML.WeightQNN(n_qubits=n, weight=k, layers=layers, encoding="dual" if k == 2 else "pairs")
            ex = exactness(mm)
            ax2.plot(mm.depth, ex[2], mk, color=INK, ms=5, mfc="none",
                     label=f"{t['k_sim']}, n = {n}, k = {k}" if layers == 1 else None)
    ax2.set_xlabel(t["k_x"])
    ax2.set_ylabel(t["k_y"])
    ax2.set_title(t["k_title"], fontsize=10)
    ax2.legend(fontsize=7, frameon=False, ncol=2)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig, (clean, raw, filt)


def fig_results(t, grads):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, ls in ((1, "-"), (2, "--")):
        rows = [r for r in SCALING if r[1] == k]
        ns = [r[0] for r in rows]
        lab = t["a_w"].format(k=k)
        ax1.plot(ns, [r[3] for r in rows], ls, marker="o", color=BLUE, label=f"{lab}: {t['a_clean']}")
        ax1.plot(ns, [r[4] for r in rows], ls, marker="s", color=ORANGE, label=f"{lab}: {t['a_raw']}")
        ax1.plot(ns, [r[5] for r in rows], ls, marker="^", color=GREEN, label=f"{lab}: {t['a_aware']}")
    ax1.set_xticks([5, 6, 8])
    ax1.set_xlabel(t["a_x"])
    ax1.set_ylabel(t["a_y"])
    ax1.set_title(t["a_title"], fontsize=10)
    ax1.legend(fontsize=6.8, frameon=False, loc="lower left")
    ax1.grid(alpha=0.3)
    sweep, curves = grads
    ax2.plot(sweep, curves["clean"], color=GREY, lw=4, label=t["g_clean"])
    ax2.plot(sweep, curves["filt"], color=BLUE, lw=1.5, ls="--", label=t["g_filt"])
    ax2.plot(sweep, curves["raw"], color=ORANGE, lw=1.5, label=t["g_raw"])
    ax2.axhline(0, color=INK, lw=0.6)
    ax2.set_xlabel(t["g_x"])
    ax2.set_ylabel(t["g_y"])
    ax2.set_title(t["g_title"], fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    print(f"Filter exactness under equal T1 (gamma = {GAMMA} per qubit per sublayer), 40 random inputs:")
    print("  model: max |filtered - noiseless| (m-qg) | mean |raw - noiseless| (m-qg) | kept min..max | law")
    for name, kw in (("E, n=5 k=1", dict(n_qubits=5, weight=1)), ("W pairs, n=5 k=2", dict(n_qubits=5, weight=2)),
                     ("W dual zz, n=5 k=2", dict(n_qubits=5, weight=2, encoding="dual", readout="zz")),
                     ("E, n=8 k=1", dict(n_qubits=8, weight=1)), ("W ring, n=8 k=2", dict(n_qubits=8, weight=2, encoding="ring"))):
        m = QML.WeightQNN(layers=3, **kw)
        ex = exactness(m)
        rb = residual_bias(m)
        print(f"  {name:20s} depth {m.depth}: {ex[0]:.1e} | {ex[1]:6.1f} | {ex[2]:.4f}..{ex[3]:.4f} | {ex[4]:.4f}"
              f" | unequal T1 bias {rb['unequal']:.1f} m-qg, dephasing 0.01 bias {rb['dephasing']:.1f} m-qg")

    print("\nReadout rank on the sector distribution (constant included) vs sector size:")
    for k in (1, 2):
        for ro in ("z", "zz"):
            m = QML.WeightQNN(n_qubits=5, weight=k, readout=ro, encoding="dual" if k == 2 else "pairs")
            print(f"  n=5 k={k} readout {ro}: rank {readout_rank(m)} of C(5,{k}) = {len(m.idx[k])}")

    dz, dO = quadratic_twin()
    print(f"\nQuadratic twin of the weight-1 network (F2): max |z_network - z_twin| = {dz:.1e}; |O O^T - I| = {dO:.1e}")

    xs = np.linspace(-1, 1, 11)
    dev = max(abs(math.cos(4 * math.acos(x)) - (8 * x**4 - 8 * x**2 + 1)) for x in xs)
    print(f"qg encoding with 4 loads and identity processing: <Z> = T_4(x), max deviation {dev:.1e}")

    sweep, curves, g, kept, bp, ver = pennylane_gradients()
    print(f"\nPennyLane {ver}, 3 qubits, SingleExcitation + AmplitudeDamping({GAMMA}), parameter shift:")
    for k, v in g.items():
        print(f"  grad qg_Z(qubit 1) {k:5s}: {np.round(v, 6)}")
    print(f"  filtered - noiseless: {np.abs(g['filt'] - g['clean']).max():.1e}; raw / noiseless = "
          f"{np.round(g['raw'] / g['clean'], 6)}; kept fraction {kept:.6f} = (1 - gamma)^4 = {(1 - GAMMA) ** 4:.6f}")
    print(f"  default.mixed backprop through SingleExcitation, jacobian of probs: {bp}")

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        f1, readings = fig_filter(tx)
        if lang == "es":
            print("\nFigure input (m-qg): noiseless", np.round(readings[0]), "raw", np.round(readings[1]),
                  "filtered", np.round(readings[2]))
        names = {"es": ("cap20_filtro", "cap20_resultados"), "en": ("cap20_filter", "cap20_results")}[lang]
        for name, fig in zip(names, (f1, fig_results(tx, (sweep, curves)))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
