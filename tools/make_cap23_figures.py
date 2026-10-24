"""Figures of chapter 23 of the book (information geometry, the natural gradient and kernels read in qg), Spanish and
English, on the chapter 1 base.

    python tools/make_cap23_figures.py

writes book/es/figuras/cap23_*.png and book/en/figures/cap23_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. Recomputed here: the Fisher information per shot in the angle and in qg (with a Monte
Carlo of the sample mean), the §1 descent from a pole in theta, raw qg and regularised qg, the shift-rule pair that one
evaluation cannot tell apart, the qg_YY block metric against the full Fubini-Study metric, the §159 test seeds of the
natural gradient on the TFIM (with the learning rates stored by the notes' pre-registered run), the §9.2 robustness of
pole-damped descent, the qang angle of the fidelity kernel, and the §23 gradient variance of the global and qg local
costs. The kernel results on device noise models (§161) are quoted from the notes. Runs in a few minutes.
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
sys.path.insert(0, os.path.join(ROOT, "examples"))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402
from qang import fisher as F  # noqa: E402
from qang import gradients as GR  # noqa: E402
from qang import kernels as KR  # noqa: E402

import barren_plateau_qg_local_cost as BP  # noqa: E402
import fisher_natural_gradient_qg as FN  # noqa: E402
import pole_damped_gradient_descent_robustness as PD  # noqa: E402

RED = "#8c2d04"
STORED = os.path.join(ROOT, "examples", "data", "fisher_natural_gradient_qg.json")

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "f_x": r"ángulo qang $\theta$ (rad)",
        "f_y": r"desvío estándar $\times\sqrt{N}$",
        "f_th": r"en el ángulo: $\sigma_\theta\sqrt{N}=1$",
        "f_m": r"en m-qg: $\sigma_m\sqrt{N}/1000=\sin\theta$",
        "f_mc": "simulado (N = 1000)",
        "f_title": "El ángulo es la coordenada plana de Fisher",
        "b_x": "iteración",
        "b_y": r"$\theta$ (rad)",
        "b_title": r"Descenso desde un polo, $E=-\sin\theta$ ($\theta_0=0{,}01$)",
        "b_names": {"theta": r"en $\theta$ (tasa 0,05)", "qg_raw": "en qg, crudo (0,05)",
                    "qg_clipped": "en qg, recortado (0,01)", "qg_tikhonov": "en qg, Tikhonov (0,01)"},
        "q_y": "pasos hasta $E-E_0\\leq0{,}03$ (20 semillas)",
        "q_title": "TFIM de 4 qubits, 12 parámetros",
        "q_names": {"GD-theta": "descenso\nen θ", "GD-qg": "descenso\nen qg", "QNG-block": "QNG, bloques\nqg_YY",
                    "QNG-full": "QNG, métrica\ncompleta"},
        "p_x": "tasa de aprendizaje",
        "p_y": "fracción de éxitos (300 paisajes)",
        "p_plain": "descenso en θ",
        "p_damp": "amortiguado en los polos",
        "p_title": "Empezando cerca de un polo",
        "v_x": "qubits, $n$",
        "v_y": r"Var($\partial C/\partial p_0$)",
        "v_g": "costo global",
        "v_l": "costo local qg",
        "v_title": "Mesetas estériles: costo global contra costo local qg",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "f_x": r"qang angle $\theta$ (rad)",
        "f_y": r"standard deviation $\times\sqrt{N}$",
        "f_th": r"in the angle: $\sigma_\theta\sqrt{N}=1$",
        "f_m": r"in m-qg: $\sigma_m\sqrt{N}/1000=\sin\theta$",
        "f_mc": "simulated (N = 1000)",
        "f_title": "The angle is the Fisher-flat coordinate",
        "b_x": "iteration",
        "b_y": r"$\theta$ (rad)",
        "b_title": r"Descent from a pole, $E=-\sin\theta$ ($\theta_0=0.01$)",
        "b_names": {"theta": r"in $\theta$ (rate 0.05)", "qg_raw": "in qg, raw (0.05)",
                    "qg_clipped": "in qg, clipped (0.01)", "qg_tikhonov": "in qg, Tikhonov (0.01)"},
        "q_y": "steps to $E-E_0\\leq0.03$ (20 seeds)",
        "q_title": "4-qubit TFIM, 12 parameters",
        "q_names": {"GD-theta": "descent\nin θ", "GD-qg": "descent\nin qg", "QNG-block": "QNG, qg_YY\nblocks",
                    "QNG-full": "QNG, full\nmetric"},
        "p_x": "learning rate",
        "p_y": "success fraction (300 landscapes)",
        "p_plain": "descent in θ",
        "p_damp": "pole-damped",
        "p_title": "Starting near a pole",
        "v_x": "qubits, $n$",
        "v_y": r"Var($\partial C/\partial p_0$)",
        "v_g": "global cost",
        "v_l": "qg local cost",
        "v_title": "Barren plateaus: global cost against qg local cost",
    },
}


def fisher_mc(n_shots=1000, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for th in np.linspace(0.15, math.pi - 0.15, 12):
        p0 = math.cos(th / 2) ** 2
        k = rng.binomial(n_shots, p0, size=reps)
        q = 2 * k / n_shots - 1
        out.append((th, float(np.std(q) * math.sqrt(n_shots)), math.sin(th),
                    float(np.std(np.arccos(np.clip(q, -1, 1))) * math.sqrt(n_shots))))
    return out


def descent_runs():
    out = {}
    for space, lr in (("theta", 0.05), ("qg_raw", 0.05), ("qg_clipped", 0.01), ("qg_tikhonov", 0.01)):
        h = GR.run_gradient_descent(space, 0.01, lr=lr, steps=150, eps=0.05)
        out[space] = h
    return out


def metric_check(seeds=5):
    rng = np.random.default_rng(3)
    dmax, diag = 0.0, []
    for _ in range(seeds):
        th = rng.uniform(0, 2 * math.pi, FN.N * FN.L)
        Bm, Fm = F.block_metric(th, FN.N, FN.L), F.full_metric(th, FN.N, FN.L)
        for layer in range(FN.L):
            s = slice(layer * FN.N, (layer + 1) * FN.N)
            dmax = max(dmax, float(np.abs(Bm[s, s] - Fm[s, s]).max()))
        diag.append(np.diag(Fm))
        off = Fm - np.diag(np.diag(Fm))
    return dmax, float(np.abs(np.array(diag) - 0.25).max()), float(np.abs(off).max())


def qng_rerun():
    with open(STORED, encoding="utf-8") as fh:
        best = json.load(fh)["best_lr"]
    return {m: [FN.run(m, best[m], sd)[0] for sd in FN.TEST] for m in FN.METHODS}, best


def pole_damped(lrs=(0.3, 1.0, 2.0, 3.0, 5.0, 10.0)):
    return {lr: PD.success_and_trapped_rates(lr, n_iters=300, seed=0) for lr in lrs}


def plateaus(ns=(2, 4, 6, 8, 10)):
    out = {}
    for name, depth in (("L2", lambda n: 2), ("L4n", lambda n: 4 * n)):
        for cost in ("global", "qg_local"):
            out[(name, cost)] = [BP.gradient_variance(n, depth(n), cost, samples=200) for n in ns]
    return list(ns), out


def fig_fisher(t, mc, runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    th = np.linspace(0.01, math.pi - 0.01, 200)
    ax1.plot(th, np.ones_like(th), color=BLUE, lw=2, label=t["f_th"])
    ax1.plot(th, np.sin(th), color=ORANGE, lw=2, label=t["f_m"])
    ax1.plot([r[0] for r in mc], [r[3] for r in mc], "o", color=BLUE, ms=5, mfc="none", label=t["f_mc"])
    ax1.plot([r[0] for r in mc], [r[1] for r in mc], "o", color=ORANGE, ms=5, mfc="none")
    ax1.set_xlabel(t["f_x"])
    ax1.set_ylabel(t["f_y"])
    ax1.set_ylim(0, 1.25)
    ax1.set_title(t["f_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="lower center")
    ax1.grid(alpha=0.3)
    for (space, h), c in zip(runs.items(), (BLUE, ORANGE, GREEN, INK)):
        raw = space == "qg_raw"   # the raw run jumps pole to pole every step: thin and behind, so the others show
        ax2.plot(h.theta, color=c, lw=0.6 if raw else 2, alpha=0.45 if raw else 1, zorder=1 if raw else 3,
                 label=t["b_names"][space])
    ax2.axhline(math.pi / 2, color=GREY, lw=0.8, ls="--")
    ax2.set_xlabel(t["b_x"])
    ax2.set_ylabel(t["b_y"])
    ax2.set_title(t["b_title"], fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_qng(t, steps, pd):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(steps)
    data = [np.minimum(steps[m], 301) for m in names]
    ax1.boxplot(data, widths=0.5, medianprops={"color": INK})
    for i, d in enumerate(data):
        ax1.plot(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)), d, "o", ms=3,
                 color=(BLUE, ORANGE, GREEN, RED)[i], alpha=0.7)
    ax1.set_xticks(range(1, len(names) + 1))
    ax1.set_xticklabels([t["q_names"][m] for m in names], fontsize=8)
    ax1.axhline(300, color=GREY, lw=0.8, ls=":")
    ax1.set_ylabel(t["q_y"])
    ax1.set_title(t["q_title"], fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    lrs = list(pd)
    ax2.plot(lrs, [pd[lr]["plain"] for lr in lrs], "o-", color=GREY, label=t["p_plain"])
    ax2.plot(lrs, [pd[lr]["damped"] for lr in lrs], "s-", color=BLUE, label=t["p_damp"])
    ax2.set_xscale("log")
    ax2.set_xticks(lrs)
    ax2.set_xticklabels([f"{lr:g}" for lr in lrs])
    ax2.set_xlabel(t["p_x"])
    ax2.set_ylabel(t["p_y"])
    ax2.set_title(t["p_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_plateaus(t, ns, var):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (name, cost), v in var.items():
        c = BLUE if cost == "qg_local" else ORANGE
        ls = "-" if name == "L2" else "--"
        lab = (t["v_l"] if cost == "qg_local" else t["v_g"]) + (", L = 2" if name == "L2" else ", L = 4n")
        ax.semilogy(ns, v, ls, marker="o", color=c, label=lab)
    ax.set_xlabel(t["v_x"])
    ax.set_ylabel(t["v_y"])
    ax.set_title(t["v_title"], fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def main():
    mc = fisher_mc()
    print("Fisher per shot: F_theta(1.0) =", round(F.fisher_theta(1.0), 12), "; F_qg(0.9) =", round(F.fisher_qg(0.9), 4))
    print("Monte Carlo, N = 1000: sd(qg)*sqrt(N) vs sin(theta), sd(arccos qg)*sqrt(N):")
    for th, sq, s, st in mc[::3]:
        print(f"  theta {th:.2f}: {sq:.3f} vs {s:.3f}; angle {st:.3f}")
    print("  angle sd*sqrt(N) over all points:", f"{min(r[3] for r in mc):.3f}..{max(r[3] for r in mc):.3f}")

    runs = descent_runs()
    print("\n§1 descent from theta0 = 0.01 on E = -sin(theta), 150 iterations:")
    for space, h in runs.items():
        steps = np.abs(np.diff(h.theta))
        print(f"  {space:12s} final theta {h.theta[-1]:.4f}  final E {h.energy[-1]:+.4f}  max step {steps.max():.2f}")
    for th0 in (0.1, 0.01, 0.001):
        print(f"  |1/sin| at theta {th0}: raw {abs(GR.inverse_jacobian_raw(th0)):.1f}, clipped "
              f"{abs(GR.inverse_jacobian_clipped(th0)):.1f}, Tikhonov {abs(GR.inverse_jacobian_tikhonov(th0)):.2f}")

    f1, f2, d1, d2 = F.indistinguishable_pair(0.7, math.pi / 2)
    print(f"\nShift rule: f1(theta+s) = {f1(0.7 + math.pi / 2):.6f}, f2(theta+s) = {f2(0.7 + math.pi / 2):.6f};"
          f" derivatives {d1:.4f} vs {d2:.4f}; two-point shift of f1: {F.shift_gradient(f1, 0.7):.6f}")

    dmax, ddiag, off = metric_check()
    print(f"\nMetric, TFIM ansatz n = 4, 3 layers: block (qg_YY/4) vs full in-layer blocks max diff {dmax:.1e};"
          f" every diagonal entry = 1/4 to {ddiag:.1e}; largest off-diagonal entry {off:.3f}")

    steps, best = qng_rerun()
    print(f"\n§159 natural gradient, test seeds 6-25 with the stored learning rates {best}:")
    for m, s in steps.items():
        s = np.array(s)
        print(f"  {m:10s} median {np.median(s):6.1f}  successes {int(np.sum(s <= 300))}/20")

    pd = pole_damped()
    print("\n§9.2 pole-damped descent, 300 landscapes near a pole:")
    for lr, r in pd.items():
        print(f"  lr {lr:5.1f}: plain {r['plain']:.3f}  damped {r['damped']:.3f}  trapped {r['trapped']:.3f}")

    X = np.random.default_rng(5).normal(size=(6, 4))
    E = KR.unary_embedding(X)
    K = KR.exact_kernel(E, E)
    ang = KR.kernel_qang_angle(K)
    phi = np.arccos(np.clip(np.abs(E @ E.T), -1, 1))
    print(f"\nKernel: arccos(2K - 1) vs 2*phi max diff {np.abs(ang - 2 * phi).max():.1e}")

    ns, var = plateaus()
    print("\n§23 gradient variance (200 random initialisations):")
    for key, v in var.items():
        print(f"  {key[0]:3s} {key[1]:8s} " + "  ".join(f"n={n}: {x:.1e}" for n, x in zip(ns, v)))
    r = var[("L2", "qg_local")][-1] / var[("L2", "global")][-1]
    print(f"  ratio local/global at n = {ns[-1]}, L = 2: {r:.0f}")

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap23_fisher", "cap23_qng", "cap23_mesetas"), "en": ("cap23_fisher", "cap23_qng", "cap23_plateaus")}[lang]
        for name, fig in zip(names, (fig_fisher(tx, mc, runs), fig_qng(tx, steps, pd), fig_plateaus(tx, ns, var))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
