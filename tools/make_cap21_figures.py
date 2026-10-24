"""Figures of chapter 21 of the book (the qg symmetry filter and its exactness under T1), Spanish and English, on the
chapter 1 base.

    python tools/make_cap21_figures.py

writes book/es/figuras/cap21_*.png and book/en/figures/cap21_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. Everything here runs on qang.qml.WeightQNN with random sector states and random RBS
angles (no datasets): the exactness theorem in sectors of weight 1 to 5, the no-jump factor, property F5 (only the
ratios of the decay rates matter), property F6 (the filter turns T1 + dephasing into dephasing alone), zero-noise
extrapolation of the filtered readout, and a Monte Carlo of shots against the closed-form MSE rule of §94. The
classifier results of the chapter are the pre-registered results of the notes, quoted. Runs in about a minute.
"""

import math
import os
import sys

import matplotlib
import matplotlib.ticker

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

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "t_x": "peso del sector, $k$ (6 qubits)",
        "t_y": r"$\max|$filtrada − referencia$|$ (m-qg)",
        "t_eq": "T1 igual, contra sin ruido",
        "t_f5": "T1 desigual reescalado, contra T1 desigual (F5)",
        "t_f6": "T1 igual + desfase, contra desfase solo (F6)",
        "t_un": "T1 desigual, contra sin ruido",
        "t_de": "T1 igual + desfase, contra sin ruido",
        "t_title": "Lo que el filtro deja exacto y lo que no",
        "m_x": "disparos por circuito, $S$",
        "m_y": r"error cuadrático medio de las lecturas (m-qg$^2$)",
        "m_qang": "con qang (simulado)",
        "m_raw": "sin qang (simulado)",
        "m_rule_q": r"regla: $(1000^2-m^2)/(KS)$",
        "m_rule_r": r"regla: sesgo$^2+(1000^2-m^2)/S$",
        "m_title": "6 qubits, peso 2, 12 subcapas, $\\gamma=0{,}02$",
        "s_x": "subcapas, $d$",
        "s_y": r"disparos de equilibrio $S^*$",
        "s_title": "Por encima de $S^*$ el filtro gana (T1 igual, $\\gamma=0{,}02$)",
        "s_k": "6 qubits, peso {k}",
        "z_x": "escala de ruido del desfase",
        "z_y": r"error medio de la lectura filtrada (m-qg)",
        "z_lin": "extrapolación lineal (1, 2)",
        "z_rich": "Richardson (1, 2, 3)",
        "z_title": "ZNE de las lecturas filtradas (T1 igual + desfase 0,03)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "t_x": "sector weight, $k$ (6 qubits)",
        "t_y": r"$\max|$filtered − reference$|$ (m-qg)",
        "t_eq": "equal T1, against noiseless",
        "t_f5": "unequal T1 rescaled, against unequal T1 (F5)",
        "t_f6": "equal T1 + dephasing, against dephasing alone (F6)",
        "t_un": "unequal T1, against noiseless",
        "t_de": "equal T1 + dephasing, against noiseless",
        "t_title": "What the filter leaves exact and what it does not",
        "m_x": "shots per circuit, $S$",
        "m_y": r"mean squared error of the readings (m-qg$^2$)",
        "m_qang": "with qang (simulated)",
        "m_raw": "without qang (simulated)",
        "m_rule_q": r"rule: $(1000^2-m^2)/(KS)$",
        "m_rule_r": r"rule: bias$^2+(1000^2-m^2)/S$",
        "m_title": "6 qubits, weight 2, 12 sublayers, $\\gamma=0.02$",
        "s_x": "sublayers, $d$",
        "s_y": r"break-even shots $S^*$",
        "s_title": "Above $S^*$ the filter wins (equal T1, $\\gamma=0.02$)",
        "s_k": "6 qubits, weight {k}",
        "z_x": "dephasing noise scale",
        "z_y": r"mean error of the filtered reading (m-qg)",
        "z_lin": "linear extrapolation (1, 2)",
        "z_rich": "Richardson (1, 2, 3)",
        "z_title": "ZNE of the filtered readings (equal T1 + dephasing 0.03)",
    },
}


def sector_model(n, k, layers):
    m = QML.WeightQNN(n_qubits=n, weight=k, layers=layers)
    return m


def random_states(m, count, rng):
    psi = np.zeros((count, m.dim))
    idx = m.idx[m.weight]
    a = rng.normal(size=(count, len(idx)))
    psi[:, idx] = a / np.linalg.norm(a, axis=1, keepdims=True)
    return psi


def readings(m, probs, qang):
    return 1000 * m.qg_z(probs, qang=qang)


def theorem_checks(n=6, layers=2, seed=0):
    """Max deviation (m-qg) of the filtered readings for each weight under five conditions."""
    rng = np.random.default_rng(seed)
    u = np.linspace(-1, 1, n)
    g_un = GAMMA * (1 + 0.5 * u)
    out = {}
    for k in range(1, n):
        m = sector_model(n, k, layers)
        psi = random_states(m, 20, rng)
        th = rng.uniform(-math.pi, math.pi, size=m.n_theta)
        clean = readings(m, m.probs(th, psi), False)
        f = lambda **kw: readings(m, m.probs(th, psi, **kw), True)  # noqa: E731
        un = f(gamma=g_un)
        c = 0.85
        un_scaled = f(gamma=1 - c * (1 - g_un))
        deph_only = readings(m, m.probs(th, psi, dephasing=0.03), False)
        out[k] = {"eq": np.abs(f(gamma=GAMMA) - clean).max(), "f5": np.abs(un_scaled - un).max(),
                  "f6": np.abs(f(gamma=GAMMA, dephasing=0.03) - deph_only).max(),
                  "un": np.abs(un - clean).max(), "de": np.abs(f(gamma=GAMMA, dephasing=0.03) - clean).max(),
                  "raw_un_scaled": np.abs(readings(m, m.probs(th, psi, gamma=1 - c * (1 - g_un)), False)
                                          - readings(m, m.probs(th, psi, gamma=g_un), False)).max()}
    return out


def no_jump(n=6, k=2, seed=1):
    """One damping step on a weight-k state: the no-jump part keeps the state up to (1 - gamma)^(k/2)."""
    m = QML.WeightQNN(n_qubits=n, weight=k, layers=1)
    rng = np.random.default_rng(seed)
    psi = random_states(m, 1, rng)[0]
    p_after = m.probs(np.zeros(m.n_theta), psi[None], gamma=GAMMA)[0]
    kept = p_after[m.idx[k]].sum()
    return float(kept), (1 - GAMMA) ** (k * m.depth), m.depth


def mse_monte_carlo(n=6, k=2, layers=4, gamma=0.02, shots=(30, 100, 300, 1000, 3000, 10000), circuits=10, reps=200,
                    seed=2):
    """MSE (m-qg^2) of the per-qubit readings with and without the filter, simulated and from the §94 rule."""
    m = QML.WeightQNN(n_qubits=n, weight=k, layers=layers)
    rng = np.random.default_rng(seed)
    psi = random_states(m, circuits, rng)
    th = rng.uniform(-math.pi, math.pi, size=m.n_theta)
    clean = m.qg_z(m.probs(th, psi), qang=False)
    noisy = m.probs(th, psi, gamma=gamma)
    K = QML.kept_fraction(gamma, k, m.depth)
    z_f = m.qg_z(noisy, qang=True)
    r = m.qg_z(noisy, qang=False)
    sec = m.wt == k
    out = {"K": K, "depth": m.depth, "rows": []}
    for S in shots:
        e_q, e_r = [], []
        for c in range(circuits):
            p = noisy[c] / noisy[c].sum()
            for _ in range(reps):
                cnt = rng.multinomial(S, p)
                raw = (cnt @ m.zsign) / S
                ks = cnt[sec].sum()
                filt = (cnt[sec] @ m.zsign[sec]) / ks if ks else np.zeros(m.n)
                e_q.append(np.mean((filt - clean[c]) ** 2))
                e_r.append(np.mean((raw - clean[c]) ** 2))
        rule_q = np.mean((1 - z_f**2) / (K * S))
        rule_r = np.mean((r - clean) ** 2 + (1 - r**2) / S)
        out["rows"].append((S, 1e6 * np.mean(e_q), 1e6 * rule_q, 1e6 * np.mean(e_r), 1e6 * rule_r))
    return out


def break_even(n=6, k_list=(1, 2, 3), depths=(3, 6, 12, 24, 48), gamma=0.02, circuits=10, seed=3):
    """S* where the rule's MSE with qang equals the raw one: (1 - z^2)/K = bias^2 S + (1 - r^2)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in k_list:
        rows = []
        for d in depths:
            layers = max(1, d // 3)
            m = QML.WeightQNN(n_qubits=n, weight=k, layers=layers)
            psi = random_states(m, circuits, rng)
            th = rng.uniform(-math.pi, math.pi, size=m.n_theta)
            clean = m.qg_z(m.probs(th, psi), qang=False)
            noisy = m.probs(th, psi, gamma=gamma)
            K = QML.kept_fraction(gamma, k, m.depth)
            r = m.qg_z(noisy, qang=False)
            a = np.mean((1 - clean**2) / K - (1 - r**2))
            b2 = np.mean((r - clean) ** 2)
            rows.append((m.depth, K, max(a, 0) / b2))
        out[k] = rows
    return out


def zne(n=5, k=2, layers=3, p=0.03, seed=4):
    """Linear and Richardson ZNE of the filtered readings over dephasing scales 1, 2, 3 (T1 equal, filtered)."""
    m = QML.WeightQNN(n_qubits=n, weight=k, layers=layers)
    rng = np.random.default_rng(seed)
    psi = random_states(m, 40, rng)
    th = rng.uniform(-math.pi, math.pi, size=m.n_theta)
    clean = 1000 * m.qg_z(m.probs(th, psi), qang=False)
    f = [1000 * m.qg_z(m.probs(th, psi, gamma=GAMMA, dephasing=s * p), qang=True) for s in (1, 2, 3)]
    lin = 2 * f[0] - f[1]
    rich = 3 * f[0] - 3 * f[1] + f[2]
    raw = 1000 * m.qg_z(m.probs(th, psi, gamma=GAMMA, dephasing=p), qang=False)
    err = lambda x: float(np.abs(x - clean).mean())  # noqa: E731
    return {"scales": [err(x) for x in f], "lin": err(lin), "rich": err(rich), "raw": err(raw)}


def fig_theorem(t, chk, mc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ks = sorted(chk)
    floor = 1e-14
    for key, c, mk in (("eq", BLUE, "o"), ("f5", GREEN, "s"), ("f6", INK, "^"), ("un", ORANGE, "D"), ("de", RED, "v")):
        ax1.semilogy(ks, [max(chk[k][key], floor) for k in ks], "-" + mk, color=c, label=t["t" + "_" + key])
    ax1.set_xticks(ks)
    ax1.set_xlabel(t["t_x"])
    ax1.set_ylabel(t["t_y"])
    ax1.set_title(t["t_title"], fontsize=10)
    ax1.legend(fontsize=7, frameon=False, loc="center right")
    ax1.grid(alpha=0.3)
    rows = mc["rows"]
    S = [r[0] for r in rows]
    ax2.loglog(S, [r[1] for r in rows], "o", color=BLUE, label=t["m_qang"])
    ax2.loglog(S, [r[2] for r in rows], "-", color=BLUE, lw=1.2, label=t["m_rule_q"])
    ax2.loglog(S, [r[3] for r in rows], "s", color=ORANGE, label=t["m_raw"])
    ax2.loglog(S, [r[4] for r in rows], "-", color=ORANGE, lw=1.2, label=t["m_rule_r"])
    ax2.set_xlabel(t["m_x"])
    ax2.set_ylabel(t["m_y"])
    ax2.set_title(t["m_title"], fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_rule(t, be, z):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for (k, rows), c in zip(be.items(), (BLUE, GREEN, ORANGE)):
        ax1.semilogy([r[0] for r in rows], [max(r[2], 0.5) for r in rows], "-o", color=c, label=t["s_k"].format(k=k))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_yticks([10, 20, 30, 40, 60])
    ax1.set_xlabel(t["s_x"])
    ax1.set_ylabel(t["s_y"])
    ax1.set_title(t["s_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.3, which="both")
    xs = [1, 2, 3]
    ax2.plot(xs, z["scales"], "o-", color=GREY, label="filtro" if "es" in t["dir"] else "filter")
    ax2.plot([0], [z["lin"]], "s", color=BLUE, ms=8, label=t["z_lin"])
    ax2.plot([0], [z["rich"]], "D", color=GREEN, ms=8, label=t["z_rich"])
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xlabel(t["z_x"])
    ax2.set_ylabel(t["z_y"])
    ax2.set_ylim(0, None)
    ax2.set_title(t["z_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    kept, law, d = no_jump()
    print(f"No-jump / kept fraction, 6 qubits weight 2, {d} sublayers of damping: {kept:.6f} vs (1-g)^(k d) = {law:.6f}")
    chk = theorem_checks()
    print("\nMax deviation of the filtered readings (m-qg), 6 qubits, 20 random states, 2 layers (6 sublayers):")
    for k, v in chk.items():
        print(f"  k={k}: equal T1 {v['eq']:.1e} | F5 rescaled {v['f5']:.1e} (raw changes by {v['raw_un_scaled']:.1f}) |"
              f" F6 {v['f6']:.1e} | unequal T1 bias {v['un']:.1f} | dephasing bias {v['de']:.1f}")

    mc = mse_monte_carlo()
    print(f"\nMSE rule (6 qubits, weight 2, {mc['depth']} sublayers, gamma 0.02, K = {mc['K']:.4f}), m-qg^2:")
    for S, q, rq, r, rr in mc["rows"]:
        print(f"  S = {S:6d}: qang {q:9.1f} (rule {rq:9.1f}, ratio {q / rq:.2f}) | raw {r:9.1f} (rule {rr:9.1f}) |"
              f" raw/qang {r / q:.1f}")

    be = break_even()
    print("\nBreak-even shots S* from the rule (6 qubits, gamma 0.02):")
    for k, rows in be.items():
        print(f"  k={k}: " + "  ".join(f"d={d}: K {K:.3f} S* {s:.1f}" for d, K, s in rows))

    z = zne()
    print(f"\nZNE (5 qubits, weight 2, T1 0.08 + dephasing 0.03, 40 random states), mean |error| in m-qg:"
          f" raw {z['raw']:.1f}, filter {z['scales'][0]:.1f}, x2 {z['scales'][1]:.1f}, x3 {z['scales'][2]:.1f},"
          f" linear {z['lin']:.1f} ({z['lin'] / z['scales'][0]:.0%}), Richardson {z['rich']:.1f}"
          f" ({z['rich'] / z['scales'][0]:.0%})")

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap21_teorema", "cap21_regla"), "en": ("cap21_theorem", "cap21_rule")}[lang]
        for name, fig in zip(names, (fig_theorem(tx, chk, mc), fig_rule(tx, be, z))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
