"""Figures of chapter 27 of the book (hardware characterization read in qg), Spanish and English, on the chapter 1 base.

    python tools/make_cap27_figures.py

writes book/es/figuras/cap27_*.png and book/en/figures/cap27_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. Recomputed here: the §31 heralded sweep against the standard suite (readout errors,
effective temperature, T1 and T2, by Monte Carlo of the shots), the §18 noise-type detection (mean qg_Z, qg_S, XEB and
HOP as one-threshold classifiers on held-out circuits), and the §17 control grids (single-qubit worst errors and the
45 Grover-Rudolph loading cases). The device-model results (§86, §145, §150) are quoted from the notes. Runs in a few
minutes.
"""

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

import control_quantization_qg_vs_theta as CQ  # noqa: E402
import hardware_characterization_qg as HC  # noqa: E402
import noise_type_detection_qg_vs_xeb as NT  # noqa: E402

RED = "#8c2d04"
REPS = 40

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "c_x": "población térmica real $p$",
        "c_y": r"error de lectura $e_{01}$ estimado",
        "c_true": "verdadero (0,015)",
        "c_std": "calibración estándar",
        "c_qg": "barrido qg con heraldo",
        "c_title": "La temperatura leída como error de lectura (notas §31)",
        "t_x": "temperatura efectiva real (mK)",
        "t_y": "temperatura efectiva estimada (mK)",
        "t_title": r"Temperatura desde $qg_{eq}=\tanh(hf/2k_BT)$",
        "n_y": "acierto en circuitos de prueba",
        "n_feats": {"mean_qg_z": "lectura media\n(sólo conteos)", "qg_s": r"$qg_S$" + "\n(sólo conteos)",
                    "xeb": "XEB\n(simulación ideal)", "hop": "HOP\n(simulación ideal)"},
        "n_chance": "responder siempre «unital»",
        "n_title": "¿T1 o ruido unital? Ruido fuerte (notas §18)",
        "n_shots": "disparos",
        "g_y": "TVD con grilla en θ / TVD con grilla en qg",
        "g_x": "distribución",
        "g_title": "Cargar una distribución: 45 casos (notas §17)",
        "g_bits": "bits",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "c_x": "true thermal population $p$",
        "c_y": r"estimated readout error $e_{01}$",
        "c_true": "true (0.015)",
        "c_std": "standard calibration",
        "c_qg": "heralded qg sweep",
        "c_title": "Temperature read as readout error (notes §31)",
        "t_x": "true effective temperature (mK)",
        "t_y": "estimated effective temperature (mK)",
        "t_title": r"Temperature from $qg_{eq}=\tanh(hf/2k_BT)$",
        "n_y": "accuracy on held-out circuits",
        "n_feats": {"mean_qg_z": "mean readout\n(counts only)", "qg_s": r"$qg_S$" + "\n(counts only)",
                    "xeb": "XEB\n(ideal simulation)", "hop": "HOP\n(ideal simulation)"},
        "n_chance": "always answer 'unital'",
        "n_title": "T1 or unital noise? Strong noise (notes §18)",
        "n_shots": "shots",
        "g_y": "TVD with θ grid / TVD with qg grid",
        "g_x": "distribution",
        "g_title": "Loading a distribution: 45 cases (notes §17)",
        "g_bits": "bits",
    },
}


def characterization():
    return {p: HC.benchmark(p, reps=REPS) for p in HC.P_TH_SWEEP}


def noise_type():
    strong, weak = NT.build_dataset(NT.STRONG), NT.build_dataset(NT.WEAK)
    acc = {n: {f: NT.t1_detection_accuracy(strong, n, f) for f in NT.FEATURES} for n in (100, 1000, 10000)}
    weak_acc = {f: NT.t1_detection_accuracy(weak, 100000, f) for f in NT.FEATURES}
    by_strength = {r: NT.t1_detection_accuracy(strong, 1000, "mean_qg_z", strength_range=r)
                   for r in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.51))}
    return acc, weak_acc, by_strength, NT.ideal_mean_qg_z_spread()


def control():
    rng = np.random.default_rng(0)
    haar = np.arccos(rng.uniform(-1, 1, 200_000))
    single = {b: {g: CQ.single_qubit_errors(haar, b, g) for g in ("theta", "qg")} for b in (4, 8, 10)}
    cases = []
    for n in (4, 6, 8):
        for name, dist in CQ.benchmark_distributions(n).items():
            for b in (4, 6, 8):
                t_th, f_th = CQ.loading_errors(dist, b, "theta")
                t_qg, f_qg = CQ.loading_errors(dist, b, "qg")
                cases.append({"n": n, "dist": name, "bits": b, "tvd": t_th / t_qg, "inf": f_th / f_qg})
    return single, cases


def fig_characterization(t, res):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ps = list(res)
    for proto, c, m in (("standard", ORANGE, "o"), ("qg", BLUE, "s")):
        med = [res[p][1][proto]["e01"][0] for p in ps]
        sd = [res[p][1][proto]["e01"][1] for p in ps]
        ax1.errorbar(ps, med, yerr=sd, fmt=m + "-", color=c, capsize=3, label=t["c_std" if proto == "standard" else "c_qg"])
    ax1.axhline(HC.TRUE["e01"], color=INK, lw=0.8, ls="--", label=t["c_true"])
    ax1.set_xlabel(t["c_x"])
    ax1.set_ylabel(t["c_y"])
    ax1.set_title(t["c_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.3)
    true_T = [res[p][0]["T_eff_mK"] for p in ps]
    for proto, c, m in (("standard", ORANGE, "o"), ("qg", BLUE, "s")):
        med = [res[p][1][proto]["T_eff_mK"][0] for p in ps]
        sd = [res[p][1][proto]["T_eff_mK"][1] for p in ps]
        ax2.errorbar(true_T, med, yerr=sd, fmt=m, color=c, capsize=3, ms=7,
                     label=t["c_std" if proto == "standard" else "c_qg"])
    lim = max(true_T) * 1.1
    ax2.plot([0, lim], [0, lim], color=INK, lw=0.8, ls="--")
    ax2.set_xlabel(t["t_x"])
    ax2.set_ylabel(t["t_y"])
    ax2.set_title(t["t_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_noise_control(t, acc, cases):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    feats = list(NT.FEATURES)
    x = np.arange(len(feats))
    for j, (n, c) in enumerate(zip((100, 10000), (GREEN, BLUE))):
        ax1.bar(x + (j - 0.5) * 0.36, [acc[n][f] for f in feats], 0.34, color=c, label=f"{n:,} {t['n_shots']}".replace(",", "."))
    ax1.axhline(2 / 3, color=INK, lw=0.8, ls="--", label=t["n_chance"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([t["n_feats"][f] for f in feats], fontsize=8)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel(t["n_y"])
    ax1.set_title(t["n_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    names = list(dict.fromkeys(c["dist"] for c in cases))
    for i, name in enumerate(names):
        sub = [c for c in cases if c["dist"] == name]
        for c in sub:
            col = {4: ORANGE, 6: GREEN, 8: BLUE}[c["bits"]]
            ax2.plot(i + {4: -0.15, 6: 0, 8: 0.15}[c["bits"]], c["tvd"], "o", color=col, ms=5)
    for b, col in ((4, ORANGE), (6, GREEN), (8, BLUE)):
        ax2.plot([], [], "o", color=col, label=f"{b} {t['g_bits']}")
    ax2.axhline(1, color=INK, lw=0.8, ls="--")
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax2.set_ylabel(t["g_y"])
    ax2.set_title(t["g_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def main():
    res = characterization()
    print(f"§31 heralded qg sweep against the standard suite ({REPS} repetitions, 2000 shots per circuit):")
    for p, (true, s) in res.items():
        print(f"  p = {p}: true e01 {true['e01']:.4f}, T_eff {true['T_eff_mK']:.1f} mK |"
              f" e01 standard {s['standard']['e01'][0]:.4f} qg {s['qg']['e01'][0]:.4f} |"
              f" T_eff standard {s['standard']['T_eff_mK'][0]:.1f} qg {s['qg']['T_eff_mK'][0]:.1f} ± {s['qg']['T_eff_mK'][1]:.1f} mK |"
              f" T1 sd std/qg {s['standard']['T1'][1]:.1f}/{s['qg']['T1'][1]:.1f}  T2 sd {s['standard']['T2'][1]:.1f}/{s['qg']['T2'][1]:.1f}")

    acc, weak_acc, by_strength, spread = noise_type()
    print("\n§18 T1 vs unital, held-out circuits (0.67 = always 'unital'):")
    for n, a in acc.items():
        print(f"  strong, N = {n}: " + "  ".join(f"{f} {v:.2f}" for f, v in a.items()))
    print("  weak, N = 100000: " + "  ".join(f"{f} {v:.2f}" for f, v in weak_acc.items()))
    print("  mean qg_Z by strength (N = 1000):", {k: round(v, 2) for k, v in by_strength.items()})
    print(f"  ideal mean qg_Z over the 24 circuits: sd {spread[0]:.3f}, range [{spread[1]:+.3f}, {spread[2]:+.3f}]")

    single, cases = control()
    print("\n§17 single qubit, Haar targets (worst |dP| theta / qg, worst infidelity theta / qg):")
    for b, d in single.items():
        print(f"  b = {b}: {d['theta'][0]:.1e} / {d['qg'][0]:.1e}  (ratio {d['theta'][0] / d['qg'][0]:.3f}, pi/2 = {np.pi / 2:.3f});"
              f" {d['theta'][2]:.1e} / {d['qg'][2]:.1e}")
    tvd = np.array([c["tvd"] for c in cases])
    smooth = [c for c in cases if c["dist"] in ("normal", "lognormal", "Dirichlet(1)")]
    sharp = [c for c in cases if c not in smooth]
    print(f"  loading, {len(cases)} cases: qg grid lower TVD in {int(np.sum(tvd > 1))}, median ratio {np.median(tvd):.2f},"
          f" max {tvd.max():.1f}")
    print(f"  smooth: qg lower infidelity in {sum(c['inf'] > 1 for c in smooth)} of {len(smooth)},"
          f" median {np.median([c['inf'] for c in smooth]):.2f};"
          f" sharp: theta lower infidelity in {sum(c['inf'] < 1 for c in sharp)} of {len(sharp)},"
          f" median {np.median([1 / c['inf'] for c in sharp]):.2f}, max {max(1 / c['inf'] for c in sharp):.0f}")

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap27_caracterizacion", "cap27_ruido_control"), "en": ("cap27_characterization", "cap27_noise_control")}[lang]
        for name, fig in zip(names, (fig_characterization(tx, res), fig_noise_control(tx, acc, cases))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
