"""Figures of chapter 26 of the book (error correction and continuous syndromes read in qg), Spanish and English, on the
chapter 1 base.

    python tools/make_cap26_figures.py

writes book/es/figuras/cap26_*.png and book/en/figures/cap26_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. Recomputed here, exactly or by Monte Carlo of the shots: the §27 and §32 logical
infidelities (no code, bit flip, phase flip, Leung) and their boundaries, the T2/T1 rule and the qg-witness policy
(300 instances), the §33 syndromes in closed form, the §59 d = 3 surface code (code capacity, syndrome blindness to
T1, the T1-aware decoder), the §26 cut-selection survey (without qiskit-addon-cutting, which .venv lacks) and the
§11 randomized-benchmarking identity and ZNE table. The circuit-level, qutrit, erasure and device results (§64,
§70-§74, §88-§99) are quoted from the notes. Runs in about a minute.
"""

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
from qang import knitting as KN  # noqa: E402

import circuit_knitting_qg_cut_selection as CK  # noqa: E402
import qec_leung_code_t1_qg as LG  # noqa: E402
import qec_repetition_code_choice_qg as RP  # noqa: E402
import qec_syndrome_drift_tracking_qg as DR  # noqa: E402
import randomized_benchmarking_qg_z as RB  # noqa: E402
import surface_code_d3_qg as SC  # noqa: E402
import zne_qg_vs_theta_space as ZN  # noqa: E402

RED = "#8c2d04"
GAMMA = 0.01

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "c_x": r"desfase sobre amortiguamiento, $p/\gamma$ ($\gamma=0{,}01$)",
        "c_y": "infidelidad lógica",
        "c_names": {"none": "sin código", "bit": "inversión de bit", "phase": "inversión de fase", "leung": "Leung [[4,1]]"},
        "c_rule": (r"$T_2=T_1$", r"$T_2=0{,}4\,T_1$"),
        "c_title": "Qué código conviene bajo T1 y desfase (notas §27, §32)",
        "s_x": r"amortiguamiento por qubit, $\gamma$",
        "s_y": r"error lógico $p_L$",
        "s_mw": "peso mínimo (estándar)",
        "s_t1": "consciente de T1 (errores sólo en bits leídos 0)",
        "s_tw": "modelo de Pauli (twirl)",
        "s_title": "Código de superficie d = 3 bajo T1 (notas §59)",
        "k_x": r"lectura de la compuerta, $qg=\cos\theta$",
        "k_y": r"multiplicador de disparos $\gamma^2$",
        "k_true": r"$\gamma^2=(1+2\sqrt{1-qg^2})^2$",
        "k_wrong": r"atajo incorrecto $1+\sqrt{1-qg^2}$",
        "k_title": "Costo de cortar una RZZ (notas §15.2)",
        "r_y": "multiplicador de disparos (media geométrica)",
        "r_rules": {"count": "menos cortes", "weakest": "enlaces más débiles", "qg": "regla qg (óptimo)"},
        "r_regimes": {"trotter": "Trotter\n(ángulos chicos)", "qaoa": "QAOA\n(ángulos grandes)"},
        "r_title": "Dónde cortar: 8 qubits en bloques de 4 (notas §26)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "c_x": r"dephasing over damping, $p/\gamma$ ($\gamma=0.01$)",
        "c_y": "logical infidelity",
        "c_names": {"none": "no code", "bit": "bit flip", "phase": "phase flip", "leung": "Leung [[4,1]]"},
        "c_rule": (r"$T_2=T_1$", r"$T_2=0.4\,T_1$"),
        "c_title": "Which code to use under T1 and dephasing (notes §27, §32)",
        "s_x": r"damping per qubit, $\gamma$",
        "s_y": r"logical error $p_L$",
        "s_mw": "minimum weight (standard)",
        "s_t1": "T1-aware (errors only on bits read 0)",
        "s_tw": "Pauli model (twirl)",
        "s_title": "d = 3 surface code under T1 (notes §59)",
        "k_x": r"gate readout, $qg=\cos\theta$",
        "k_y": r"shot multiplier $\gamma^2$",
        "k_true": r"$\gamma^2=(1+2\sqrt{1-qg^2})^2$",
        "k_wrong": r"wrong shortcut $1+\sqrt{1-qg^2}$",
        "k_title": "Cost of cutting an RZZ (notes §15.2)",
        "r_y": "shot multiplier (geometric mean)",
        "r_rules": {"count": "fewest cuts", "weakest": "weakest bonds", "qg": "qg rule (optimum)"},
        "r_regimes": {"trotter": "Trotter\n(small angles)", "qaoa": "QAOA\n(large angles)"},
        "r_title": "Where to cut: 8 qubits into blocks of 4 (notes §26)",
    },
}


def code_curves(ratios):
    out = {k: [] for k in ("none", "bit", "phase", "leung")}
    for r in ratios:
        p = r * GAMMA
        for k in ("none", "bit", "phase"):
            out[k].append(RP.logical_infidelity(k, GAMMA, p))
        out["leung"].append(LG.leung_infidelity(GAMMA, p))
    return out


def surface_curves(gammas):
    rows = {g: SC.t1_study(g) for g in gammas}
    return rows


def knit_survey(seeds=range(200)):
    return {kind: CK.survey(kind, seeds=seeds, with_addon=False) for kind in ("trotter", "qaoa")}


def fig_codes(t, ratios, curves, gammas, surf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for (k, v), c, ls in zip(curves.items(), (GREY, ORANGE, GREEN, BLUE), ("--", ":", "-", "-")):
        ax1.loglog(ratios, v, ls, color=c, lw=2, label=t["c_names"][k])
    for x, lab in zip((0.25, 1.0), t["c_rule"]):
        ax1.axvline(x, color=INK, lw=0.7, ls=":")
        ax1.text(x * 1.06, 0.04, lab, fontsize=8, color=INK, transform=ax1.get_xaxis_transform())
    ax1.set_xlabel(t["c_x"])
    ax1.set_ylabel(t["c_y"])
    ax1.set_title(t["c_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.grid(alpha=0.3, which="both")
    ax2.loglog(gammas, [surf[g]["pL_mw"] for g in gammas], "o-", color=ORANGE, label=t["s_mw"])
    ax2.loglog(gammas, [surf[g]["pL_t1"] for g in gammas], "s-", color=BLUE, label=t["s_t1"])
    ax2.loglog(gammas, [surf[g]["pL_twirl"] for g in gammas], "^--", color=GREY, label=t["s_tw"])
    ax2.set_xlabel(t["s_x"])
    ax2.set_ylabel(t["s_y"])
    ax2.set_title(t["s_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_knitting(t, survey):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    q = np.linspace(-1, 1, 401)
    ax1.plot(q, [KN.cut_sampling_overhead(x) for x in q], color=BLUE, lw=2, label=t["k_true"])
    ax1.plot(q, 1 + np.sqrt(1 - q ** 2), color=RED, lw=1.5, ls="--", label=t["k_wrong"])
    ax1.set_xlabel(t["k_x"])
    ax1.set_ylabel(t["k_y"])
    ax1.set_title(t["k_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 10)
    regimes = list(survey)
    x = np.arange(len(regimes))
    for j, (rule, c) in enumerate(zip(("count", "weakest", "qg"), (GREY, ORANGE, BLUE))):
        ax2.bar(x + (j - 1) * 0.26, [survey[r]["geo_mean_overhead"][rule] for r in regimes], 0.25, color=c,
                label=t["r_rules"][rule])
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t["r_regimes"][r] for r in regimes])
    ax2.set_ylabel(t["r_y"])
    ax2.set_title(t["r_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def main():
    print("§27 code capacity, gamma = p = 0.02 and single noises (no code / bit flip / phase flip):")
    for g, p in ((0.02, 0.0), (0.0, 0.02), (0.02, 0.02)):
        print(f"  gamma {g}, p {p}: " + " / ".join(f"{RP.logical_infidelity(k, g, p):.4f}" for k in ("none", "bit", "phase")))
    print("  phase-code boundary p/gamma at gamma = 0.002, 0.01, 0.04:",
          " ".join(f"{RP.boundary_ratio(g):.2f}" for g in (0.002, 0.01, 0.04)))
    print("§32 Leung, pure T1: gamma, no code, Leung, ratio")
    for g in (0.001, 0.01, 0.03):
        a, b = LG.infidelity("none", g, 0.0), LG.infidelity("leung", g, 0.0)
        print(f"  {g}: {a:.2e} {b:.2e} {a / b:.0f}x  (0.92 gamma^2 = {0.92 * g * g:.2e})")
    print("  Leung boundary p/gamma at 0.002, 0.01, 0.04:",
          " ".join(f"{LG.leung_boundary_ratio(g):.3f}" for g in (0.002, 0.01, 0.04)),
          "; crossover with no code at gamma =", f"{LG.crossover_gamma():.2f}")
    print("  best option by T2/T1 (t/T1 = 0.01):", LG.best_by_t2_ratio())
    pt = LG.policy_table()
    print("  policy, 300 instances, 1000 shots:",
          {k: (round(float(v["regret"]), 3), round(v["right_choice"], 2)) for k, v in pt.items() if k != "oracle_share"},
          " oracle share", pt["oracle_share"])

    g, p = 0.01, 0.003
    print(f"\n§33 Leung syndrome flip rates (gamma {g}, p {p}): Kraus {DR.leung_syndrome_rates(g, p)},"
          f" closed form {DR.leung_rates_closed_form(g, p)}")

    print("\n§59 d = 3 surface code: code-capacity p_L at p = 0.001, 0.01, 0.03:",
          " ".join(f"{SC.logical_error_bitflip(x):.2e}" for x in (0.001, 0.01, 0.03)),
          f"; pseudo-threshold {SC.pseudo_threshold():.4f}")
    gm = 0.01
    print(f"  weight-2 syndrome under T1 gamma = {gm}: (1-g)^2 + g^2 = {(1 - gm) ** 2 + gm ** 2:.5f},"
          f" Pauli twirl (1-g)^2 = {(1 - gm) ** 2:.5f}")
    gammas = (0.01, 0.03, 0.1, 0.3)
    surf = surface_curves(gammas)
    for gg in gammas:
        r = surf[gg]
        print(f"  gamma {gg}: twirl {r['pL_twirl']:.2e}  MW {r['pL_mw']:.2e}  T1-aware {r['pL_t1']:.2e}  ML {r['pL_ml']:.2e}"
              f"  ratio {r['pL_mw'] / r['pL_t1']:.2f}  mean qg_Z {r['mean_qg_z']:.3f}")

    survey = knit_survey()
    print("\n§26 cut selection, 200 instances per regime (geometric-mean shot multiplier):")
    for kind, s in survey.items():
        print(f"  {kind}: " + "  ".join(f"{k} {v:,.0f}" for k, v in s["geo_mean_overhead"].items()),
              " qg strictly cheaper than", {k: round(v, 2) for k, v in s["qg_strictly_better"].items()},
              f" {s['seconds_per_instance_qg'] * 1000:.1f} ms per instance")
    print("  gamma^2 at qg = 0: exact", KN.cut_sampling_overhead(0.0), "shortcut", 1 + 1.0)

    vals = [RB.rb_qg_z_survival(m, seed, 0.05) for m in (0, 5, 10) for seed in range(4)]
    print("\n§11 RB qg_Z(m) for 4 seeds at m = 0, 5, 10:", np.round(vals, 10),
          " (1-p)^(m+1):", [round(0.95 ** (m + 1), 10) for m in (0, 5, 10)])
    lams = np.array([1.0, 1.5, 2.0, 3.0])
    th0, q0 = math.pi / 3, 0.5
    sh = ZN.bloch_vector_shrinkage_noise(q0, 0.02, lams)
    dr = ZN.coherent_rotation_drift_noise(th0, 0.02, lams)
    print("  ZNE errors (qg space / theta space): shrinkage",
          f"{abs(ZN.zne_in_qg_space(sh, lams) - q0):.1e} / {abs(ZN.zne_in_theta_space(sh, lams) - q0):.1e};",
          f"drift {abs(ZN.zne_in_qg_space(dr, lams) - q0):.1e} / {abs(ZN.zne_in_theta_space(dr, lams) - q0):.1e}")

    ratios = np.logspace(-2, 1, 40)
    curves = code_curves(ratios)
    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap26_codigos", "cap26_corte"), "en": ("cap26_codes", "cap26_cutting")}[lang]
        for name, fig in zip(names, (fig_codes(tx, ratios, curves, gammas, surf), fig_knitting(tx, survey))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
