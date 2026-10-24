"""Figures of chapter 22 of the book (echo calibration of in-sector errors and the pull effect), Spanish and English, on
the chapter 1 base.

    python tools/make_cap22_figures.py

writes book/es/figuras/cap22_*.png and book/en/figures/cap22_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. Two things are computed here, on random inputs and parameters (no datasets, no device
noise models):

  * the pull effect (F7, §121) on qang.qml.MultiClassQNN: under equal T1 the raw logits are exactly K L + (1 - K) v,
    the decisions it flips grow with a pull index, and a bias-free ("pull-free", §123) readout decides the same with
    and without the filter;
  * echo calibration (§105) on a weight-1 sector with a stochastic in-sector error (after every sublayer each RBS pair
    is swapped with probability p, which keeps the weight and moves the excitation): the transfer matrix M from echo
    circuits, unmixing with M^(1/2) and M through qang.sectors.unmix_sector, and with shots the regime where the echo
    adds more variance than the bias it removes (§108).

The device results of the chapter (IBM fake backends and the IonQ simulator) are the pre-registered results of the notes,
quoted. Runs in about a minute.
"""

import math
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402
from qang import qml as QML  # noqa: E402
from qang import sectors as SEC  # noqa: E402

RED = "#8c2d04"
GAMMA = 0.08
N = 5

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "p_x": "índice de arrastre",
        "p_y": "fracción de decisiones cambiadas sin el filtro",
        "p_std": "lectura con sesgos (estándar)",
        "p_free": "lectura sin arrastre",
        "p_title": r"Arrastre: logits crudos $=K\,L+(1-K)\,v$ (5 clases, $\gamma=0{,}08$)",
        "e_x": r"probabilidad de intercambio por par y subcapa, $p$",
        "e_y": r"error medio de la lectura filtrada (m-qg)",
        "e_f": "filtro",
        "e_half": r"filtro + eco ($M^{1/2}$)",
        "e_full": r"filtro + eco ($M$)",
        "e_exact": "probabilidades exactas",
        "e_shots": "1000 disparos (eco: 4000)",
        "e_title": "Eco dentro del sector de peso 1 (5 qubits, 9 subcapas)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "p_x": "pull index",
        "p_y": "fraction of decisions flipped without the filter",
        "p_std": "readout with biases (standard)",
        "p_free": "pull-free readout",
        "p_title": r"Pull: raw logits $=K\,L+(1-K)\,v$ (5 classes, $\gamma=0.08$)",
        "e_x": r"swap probability per pair and sublayer, $p$",
        "e_y": r"mean error of the filtered reading (m-qg)",
        "e_f": "filter",
        "e_half": r"filter + echo ($M^{1/2}$)",
        "e_full": r"filter + echo ($M$)",
        "e_exact": "exact probabilities",
        "e_shots": "1000 shots (echo: 4000)",
        "e_title": "Echo inside the weight-1 sector (5 qubits, 9 sublayers)",
    },
}


# --- the pull effect (F7) ------------------------------------------------------------------------------------------


def pull_models(n_models=40, n_inputs=300, seed=0):
    """Random 5-class head models with biases; F7 check, flips, pull index; pull-free twin."""
    rng = np.random.default_rng(seed)
    m = QML.MultiClassQNN(n_qubits=N, n_classes=5, readout="head")
    K = QML.kept_fraction(GAMMA, 1, m.depth)
    X = rng.uniform(-1, 1, size=(n_inputs, N - 1))
    psi = m.encode(X)
    rows, f7 = [], 0.0
    for _ in range(n_models):
        th = rng.uniform(-math.pi, math.pi, size=m.n_theta)
        W = rng.normal(0, 1.0, size=(N, 5))
        b = rng.normal(0, 1.0, size=5)
        head = np.concatenate([W.ravel(), b])
        R0 = m.qg_z(m.probs(th, psi), qang=False)
        Rn = m.probs(th, psi, gamma=GAMMA)
        raw, filt = m.qg_z(Rn, qang=False), m.qg_z(Rn, qang=True)
        L, Lraw = m._logits(R0, head), m._logits(raw, head)
        v = m._logits(np.ones((1, N)), head)[0]  # the north pole: every decayed shot reads +1000 m-qg
        f7 = max(f7, float(np.abs(Lraw - (K * L + (1 - K) * v)).max()))
        dec0 = L.argmax(1)
        srt = np.sort(L, axis=1)
        margin = float(np.median(srt[:, -1] - srt[:, -2]))
        pull = (1 - K) / K * (v.max() - v.min()) / margin
        flips = float(np.mean(Lraw.argmax(1) != dec0))
        filt_flips = float(np.mean(m._logits(filt, head).argmax(1) != dec0))
        # pull-free: logits W^T (qg_Z - 1), no bias; decayed shots add zero and only rescale
        Lpf0, Lpf = (R0 - 1) @ W, (raw - 1) @ W
        pf_flips = float(np.mean(Lpf.argmax(1) != Lpf0.argmax(1)))
        pf_scale = float(np.abs(Lpf - K * Lpf0).max())
        rows.append((pull, flips, filt_flips, pf_flips, pf_scale))
    return K, f7, np.array(rows)


# --- echo calibration ----------------------------------------------------------------------------------------------


def weight1_blocks(m, theta):
    return [blocks[1] for blocks in m.block_unitaries(theta)]


def pair_positions(m):
    """For each sublayer, the pairs (a, b) as positions in the weight-1 block."""
    states = list(m.idx[1])
    pos = {m.n - 1 - int(math.log2(s)): i for i, s in enumerate(states)}  # qubit -> block index
    return [[(pos[a], pos[b]) for a, b in sub] for sub in m.sublayers] * m.layers, pos


def run_block(rho, blocks, pairs, p):
    """Weight-1 density matrix through the sublayers; after each, every pair is swapped with probability p."""
    for B, prs in zip(blocks, pairs):
        rho = B @ rho @ B.T
        for a, b in prs:
            S = np.eye(len(rho))
            S[[a, b]] = S[[b, a]]
            rho = (1 - p) * rho + p * (S @ rho @ S.T)
    return rho


def echo_study(ps=(0.0, 0.005, 0.01, 0.02, 0.04, 0.08), n_inputs=60, shots=1000, echo_shots=4000, reps=20, seed=1):
    m = QML.WeightQNN(n_qubits=N, weight=1, layers=3)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-math.pi, math.pi, size=m.n_theta)
    blocks = weight1_blocks(m, theta)
    inv = [B.T for B in reversed(blocks)]
    pairs, pos = pair_positions(m)
    states = SEC.sector_states(N, 1)
    a = rng.normal(size=(n_inputs, N))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    qubit_of = np.array([N - 1 - int(math.log2(s)) for s in states])

    def m_z(pop):  # sector populations (block order) -> per-qubit m_Z
        out = np.zeros(pop.shape[:-1] + (N,))
        out[..., qubit_of] = 1000 * (1 - 2 * pop)
        return out

    def full(pop):
        f = np.zeros(2**N)
        f[states] = pop
        return f

    out = {"p": list(ps), "exact": [], "shots": []}
    for p in ps:
        clean = np.array([np.diag(run_block(np.outer(x, x), blocks, pairs, 0.0)) for x in a])
        noisy = np.array([np.diag(run_block(np.outer(x, x), blocks, pairs, p)) for x in a])
        M = np.zeros((N, N))
        for j in range(N):
            e = np.zeros((N, N))
            e[j, j] = 1.0
            r = run_block(e, blocks, pairs, p)
            r = run_block(r, inv, list(reversed(pairs)), p)
            M[:, j] = np.clip(np.diag(r), 0, None)
        diag = np.diag(M).copy()
        half = np.array([SEC.unmix_sector(full(q), M, N, 1, 0.5)[states] for q in noisy])
        whole = np.array([SEC.unmix_sector(full(q), M, N, 1, 1.0)[states] for q in noisy])
        err = lambda q: float(np.abs(m_z(q) - m_z(clean)).mean())  # noqa: E731
        out["exact"].append((err(noisy), err(half), err(whole), float(diag.min()), float(diag.max())))
        e_f, e_h = [], []
        for _ in range(reps):
            Ms = np.zeros((N, N))
            for j in range(N):
                Ms[:, j] = rng.multinomial(echo_shots, M[:, j] / M[:, j].sum()) / echo_shots
            for q, c in zip(noisy, clean):
                s = rng.multinomial(shots, q / q.sum()) / shots
                e_f.append(np.abs(m_z(s) - m_z(c)).mean())
                e_h.append(np.abs(m_z(SEC.unmix_sector(full(s), Ms, N, 1, 0.5)[states]) - m_z(c)).mean())
        out["shots"].append((float(np.mean(e_f)), float(np.mean(e_h))))
    # shot-noise floor of the filtered reading at p = 0
    return out


def fig_pull(t, rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(rows[:, 0], rows[:, 1], "o", color=ORANGE, ms=5, label=t["p_std"])
    ax.plot(rows[:, 0], rows[:, 3], "s", color=BLUE, ms=5, label=t["p_free"])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([2, 3, 5, 10, 15])
    ax.set_xlabel(t["p_x"])
    ax.set_ylabel(t["p_y"])
    ax.set_title(t["p_title"], fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_echo(t, ec):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    ps = ec["p"]
    for ax, key, title in ((ax1, "exact", t["e_exact"]), (ax2, "shots", t["e_shots"])):
        rows = ec[key]
        ax.plot(ps, [r[0] for r in rows], "o-", color=GREY, label=t["e_f"])
        ax.plot(ps, [r[1] for r in rows], "s-", color=BLUE, label=t["e_half"])
        if key == "exact":
            ax.plot(ps, [r[2] for r in rows], "D--", color=GREEN, label=t["e_full"])
        ax.set_xlabel(t["e_x"])
        ax.set_ylabel(t["e_y"])
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle(t["e_title"], fontsize=10.5)
    fig.tight_layout()
    return fig


def main():
    K, f7, rows = pull_models()
    print(f"Pull effect (F7), 5 qubits, 5 classes, head readout with biases, 40 random models x 300 inputs, K = {K:.4f}:")
    print(f"  max |raw logits - (K L + (1-K) v)| = {f7:.1e}")
    rho, _ = spearmanr(rows[:, 0], rows[:, 1])
    print(f"  decisions flipped without the filter: mean {rows[:, 1].mean():.3f}, range {rows[:, 1].min():.3f}"
          f"..{rows[:, 1].max():.3f}; Spearman(pull index, flips) = {rho:.2f}")
    lo, hi = np.argsort(rows[:, 0])[:13], np.argsort(rows[:, 0])[-13:]
    print(f"  lowest third of pull index: {rows[lo, 1].mean():.3f}; highest third: {rows[hi, 1].mean():.3f}")
    print(f"  with the filter: max flips {rows[:, 2].max():.1e}")
    print(f"  pull-free readout (no biases): flips raw vs noiseless max {rows[:, 3].max():.1e};"
          f" |L_raw - K L_noiseless| max {rows[:, 4].max():.1e}")

    ec = echo_study()
    print("\nEcho calibration, weight-1 sector, 5 qubits, 9 sublayers, random in-sector swaps (60 inputs), m-qg:")
    for p, (f, h, w, d0, d1), (sf, sh) in zip(ec["p"], ec["exact"], ec["shots"]):
        red = "" if f == 0 else f" (M^1/2 removes {1 - h / f:.0%}, M {1 - w / f:.0%})"
        print(f"  p = {p:.3f}: echo diagonal {d0:.3f}..{d1:.3f} | exact: filter {f:6.1f}, M^1/2 {h:6.1f}, M {w:6.1f}{red}"
              f" | 1000 shots: filter {sf:6.1f}, M^1/2 {sh:6.1f}")

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap22_arrastre", "cap22_eco"), "en": ("cap22_pull", "cap22_echo")}[lang]
        for name, fig in zip(names, (fig_pull(tx, rows), fig_echo(tx, ec))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
