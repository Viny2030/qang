"""Figures of chapter 17 of the book (BB84 read in qg: the four readings, the drift/attack monitor, finite keys, attacks
beyond intercept-resend and IonQ's noisy simulator), Spanish and English, on the chapter 1 base.

    python tools/make_cap17_figures.py [--recompute]

writes book/es/figuras/cap17_*.png and book/en/figures/cap17_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. The physics lives in examples/bb84_qg_eve_vs_noise.py (§40), bb84_finite_key_qg.py
(§43), bb84_attacks_beyond_ir_qg.py (§49) and ionq_sim_bb84.py (§44); this script reuses them. The §40 monitor is
recomputed here with a vectorised likelihood (same seeds, same statistic). The IonQ part only analyses the counts
already stored in examples/data/ionq_sim_results.json; nothing is sent anywhere. The Monte Carlo results (a few
minutes) are cached in examples/data/cap17_bb84.json; --recompute rebuilds the cache.
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
from qang import probability as PB  # noqa: E402

import bb84_attacks_beyond_ir_qg as A  # noqa: E402
import bb84_finite_key_qg as F  # noqa: E402
import bb84_qg_eve_vs_noise as B  # noqa: E402
import ionq_sim_bb84 as I  # noqa: E402

CACHE = os.path.join(ROOT, "examples", "data", "cap17_bb84.json")
WINDOWS = (500, 2000, 10000)
RATE_SCEN = ("baseline", "T1 drift 0.04", "T1 drift 0.06", "intercept f = 0.05")
RED = "#8c2d04"

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "e_x": r"centro en Z, $c_Z=(m_0+m_1)/2$ (m-qg)",
        "e_y": r"semieje en Z, $r_Z=(m_0-m_1)/2$ (m-qg)",
        "e_t1": "deriva T1 ($\\gamma$ 0,02 → 0,12)",
        "e_ir": "interceptar y reenviar ($f$ 0 → 0,3)",
        "e_base": "línea de base",
        "e_thr": "umbral de Shor–Preskill",
        "e_bar": "±2σ, ventana de 2000 bits",
        "e_title": "La imagen de la esfera en Z: T1 la corre, Eve la encoge",
        "m_y": "fracción de ventanas con alarma",
        "m_q": "monitor QBER",
        "m_g": "monitor qg",
        "m_title": "Alarmas con N = 2000 bits comparados (1 % en la línea de base)",
        "m_lab": ["base", "T1\n0,04", "T1\n0,06", "IR\n0,02", "IR\n0,05", "IR\n0,10", "IR\n0,20", "imita\nT1"],
        "k_x": "bits de clave por bloque, $n$",
        "k_y": "clave secreta por señal",
        "k_title": "Clave finita (ε = $10^{-10}$); líneas finas: asintótica",
        "k_names": {"baseline": "línea de base", "T1 drift 0.06": "deriva T1 0,06", "intercept f = 0.05": "IR f = 0,05"},
        "d_y": "fracción de bloques",
        "d_title": "Protocolo ajustado en la línea de base, n = $10^5$",
        "d_abort": "el protocolo aborta",
        "d_att": "qg: tipo ataque",
        "d_dr": "qg: tipo deriva",
        "d_lab": ["base", "T1\n0,03", "T1\n0,04", "T1\n0,06", "IR\n0,02", "IR\n0,05", "T1 0,04\n+ IR 0,02", "imita\nT1"],
        "a_title": "Tres familias (n = $10^5$): ataque, T1, T2",
        "a_lab": ["base", "T1\n0,04", "T2\n0,02", "T2\n0,03", "IR\n0,05", "clon\n0,0125", "clon\n0,025", "sólo Z\n0,02",
                  "sólo Z\n0,04", "sólo X\n0,04", "T2 +\nclon"],
        "a_att": "tipo ataque",
        "a_t1": "tipo T1",
        "a_t2": "tipo T2",
        "i_title": "Simulador ruidoso de IonQ, bloques de 4000 bits (16 por escenario)",
        "i_lab": ["base", "T1\n0,03", "T1\n0,04", "T1\n0,06", "IR\n0,02", "IR\n0,05", "IR\n0,10", "T1 0,04\n+ IR 0,05"],
        "i_att": "tipo ataque",
        "i_dr": "tipo deriva",
        "i_pred": "predicho por el modelo ajustado",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "e_x": r"centre in Z, $c_Z=(m_0+m_1)/2$ (m-qg)",
        "e_y": r"semi-axis in Z, $r_Z=(m_0-m_1)/2$ (m-qg)",
        "e_t1": "T1 drift ($\\gamma$ 0.02 → 0.12)",
        "e_ir": "intercept-resend ($f$ 0 → 0.3)",
        "e_base": "baseline",
        "e_thr": "Shor–Preskill threshold",
        "e_bar": "±2σ, 2000-bit window",
        "e_title": "The image of the sphere in Z: T1 shifts it, Eve shrinks it",
        "m_y": "fraction of windows with an alarm",
        "m_q": "QBER monitor",
        "m_g": "qg monitor",
        "m_title": "Alarms with N = 2000 compared bits (1% on the baseline)",
        "m_lab": ["base", "T1\n0.04", "T1\n0.06", "IR\n0.02", "IR\n0.05", "IR\n0.10", "IR\n0.20", "mimics\nT1"],
        "k_x": "key bits per block, $n$",
        "k_y": "secret key per signal",
        "k_title": "Finite key (ε = $10^{-10}$); thin lines: asymptotic",
        "k_names": {"baseline": "baseline", "T1 drift 0.06": "T1 drift 0.06", "intercept f = 0.05": "IR f = 0.05"},
        "d_y": "fraction of blocks",
        "d_title": "Protocol tuned on the baseline, n = $10^5$",
        "d_abort": "protocol aborts",
        "d_att": "qg: attack-like",
        "d_dr": "qg: drift-like",
        "d_lab": ["base", "T1\n0.03", "T1\n0.04", "T1\n0.06", "IR\n0.02", "IR\n0.05", "T1 0.04\n+ IR 0.02", "mimics\nT1"],
        "a_title": "Three families (n = $10^5$): attack, T1, T2",
        "a_lab": ["base", "T1\n0.04", "T2\n0.02", "T2\n0.03", "IR\n0.05", "clone\n0.0125", "clone\n0.025", "Z-only\n0.02",
                  "Z-only\n0.04", "X-only\n0.04", "T2 +\nclone"],
        "a_att": "attack-like",
        "a_t1": "T1-like",
        "a_t2": "T2-like",
        "i_title": "IonQ noisy simulator, 4000-bit blocks (16 per scenario)",
        "i_lab": ["base", "T1\n0.03", "T1\n0.04", "T1\n0.06", "IR\n0.02", "IR\n0.05", "IR\n0.10", "T1 0.04\n+ IR 0.05"],
        "i_att": "attack-like",
        "i_dr": "drift-like",
        "i_pred": "predicted by the fitted model",
    },
}


# --- §40: the four readings and the monitor ----------------------------------------------------------------------


def readings(e):
    """m-qg reading of each sent state in its own basis, and the centre/semi-axis of the image of the sphere."""
    m = 1000 * B.qg_per_state(e)
    return {"m": m.tolist(), "c_z": (m[0] + m[1]) / 2, "r_z": (m[0] - m[1]) / 2,
            "c_x": (m[2] + m[3]) / 2, "r_x": (m[2] - m[3]) / 2, "a_z": m[0] + m[1]}


def sigma_window(e, n_bits):
    """Shot noise (m-qg) of c_Z and r_Z in a window of n_bits compared bits, n_bits/4 per sent state."""
    v = [4e6 * x * (1 - x) / (n_bits // 4) for x in e[:2]]
    return 0.5 * math.sqrt(v[0] + v[1])


def _ll_table(table, k, n):
    pr = np.clip(table, 1e-12, 1 - 1e-12)
    return (k * np.log(pr) + (n - k) * np.log(1 - pr)).sum(axis=-1)


def glrt_fast(k, n):
    """B.glrt_statistic, vectorised over the two grids."""
    ll_a = _ll_table(B._ATTACK_TABLE, k, n).max()
    ll_d = _ll_table(B._DRIFT_TABLE, k, n).max()
    return ll_a - max(ll_d, float(_ll_table(B._BASE_PROBS, k, n)))


def monitor(n_bits):
    """Same seeds and statistic as B.calibrate_thresholds / B.alarm_rates."""
    rng = np.random.default_rng(0)
    tot, stat = [], []
    for _ in range(2000):
        k, n = B.sample_counts(B._BASE_PROBS, n_bits, rng)
        tot.append(k.sum())
        stat.append(glrt_fast(k, n))
    th = float(np.quantile(tot, 0.99)), float(np.quantile(stat, 0.99))
    out = {}
    for name, kw in B.SCENARIOS:
        rng = np.random.default_rng(1)
        aq = ag = 0
        for _ in range(1000):
            k, n = B.sample_counts(B.error_probabilities(**kw), n_bits, rng)
            aq += k.sum() > th[0]
            ag += glrt_fast(k, n) > th[1]
        out[name] = (aq / 1000, ag / 1000)
    return out


# --- cache -------------------------------------------------------------------------------------------------------


def compute():
    res = {"monitor": {}, "rates": {}, "diag": {}}
    for nb in WINDOWS:
        print("  §40 monitor, N =", nb)
        res["monitor"][str(nb)] = monitor(nb)
    print("  §43 finite-key rates")
    sc = dict(F.SCENARIOS)
    for name in RATE_SCEN:
        _, qz, qx = F.channel(**sc[name])
        res["rates"][name] = {"asym": F.asymptotic_rate(qz, qx), "n_min": F.n_min(qz, qx),
                              "grid": {f"{n:.0e}": F.optimize(n, qz, qx)[0] for n in F.N_GRID},
                              "curve": [F.optimize(n, qz, qx)[0] for n in np.logspace(3, 7.5, 28)]}
    for n in (10000, 100000):
        print("  §43 diagnosis, n =", n)
        res["diag"][str(n)] = F.diagnose(n)
    print("  §49 three-family flags")
    res["beyond"] = A.diagnose()
    print("  §44 IonQ noisy simulator (stored counts)")
    res["ionq"] = {}
    for noise in ("aria-1", "forte-1"):
        runs = I.load_runs(noise)
        s = I.analyse(runs)
        s["predicted"] = I.predicted(runs)
        s["predicted_1e5"] = I.predicted(runs, scale=25)
        rates = I.error_rates(runs)
        s["device_error"] = {k: v for k, v in rates.items()}
        res["ionq"][noise] = s
    return res


def load(recompute):
    if os.path.exists(CACHE) and not recompute:
        with open(CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    res = compute()
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


# --- figures -----------------------------------------------------------------------------------------------------


def fig_ellipse(t, res):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    gs = np.linspace(0.02, 0.12, 21)
    fs = np.linspace(0.0, 0.3, 31)
    t1 = [readings(B.error_probabilities(gamma=g)) for g in gs]
    ir = [readings(B.error_probabilities(f=f)) for f in fs]
    ax1.plot([r["c_z"] for r in t1], [r["r_z"] for r in t1], "-o", ms=3, color=ORANGE, label=t["e_t1"])
    ax1.plot([r["c_z"] for r in ir], [r["r_z"] for r in ir], "-s", ms=3, color=RED, label=t["e_ir"])
    b = readings(B.error_probabilities())
    s = 2 * sigma_window(B.error_probabilities(), 2000)
    ax1.errorbar([b["c_z"]], [b["r_z"]], xerr=s, yerr=s, fmt="o", color=BLUE, ms=7, capsize=3,
                 label=t["e_base"] + "; " + t["e_bar"])
    ax1.axhline(780, color=GREY, ls="--", lw=1)
    ax1.text(4, 785, t["e_thr"], fontsize=8, color=GREY, va="bottom")
    ax1.set_xlabel(t["e_x"])
    ax1.set_ylabel(t["e_y"])
    ax1.set_title(t["e_title"], fontsize=10)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax1.grid(alpha=0.3)
    mon = res["monitor"]["2000"]
    names = [n for n, _ in B.SCENARIOS]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, [mon[n][0] for n in names], 0.38, color=GREY, label=t["m_q"])
    ax2.bar(x + 0.2, [mon[n][1] for n in names], 0.38, color=BLUE, label=t["m_g"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(t["m_lab"], fontsize=7.5)
    ax2.set_ylabel(t["m_y"])
    ax2.set_title(t["m_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_finite(t, res):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = np.logspace(3, 7.5, 28)
    for name, col, ls in (("baseline", BLUE, "-"), ("T1 drift 0.06", ORANGE, "--"), ("intercept f = 0.05", RED, ":")):
        r = res["rates"][name]
        ax1.plot(ns, r["curve"], ls, color=col, lw=2, label=t["k_names"][name])
        ax1.axhline(r["asym"], color=col, lw=0.8, alpha=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel(t["k_x"])
    ax1.set_ylabel(t["k_y"])
    ax1.set_title(t["k_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.3)
    d = res["diag"]["100000"]
    names = [n for n, _ in F.SCENARIOS]
    x = np.arange(len(names))
    for off, key, col, lab in ((-0.27, "abort", GREY, t["d_abort"]), (0.0, "attack", RED, t["d_att"]),
                               (0.27, "drift", ORANGE, t["d_dr"])):
        ax2.bar(x + off, [d[n][key] for n in names], 0.26, color=col, label=lab)
    ax2.set_xticks(x)
    ax2.set_xticklabels(t["d_lab"], fontsize=7)
    ax2.set_ylabel(t["d_y"])
    ax2.set_title(t["d_title"], fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_attacks(t, res):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4))
    d = res["beyond"]
    names = [n for n, _ in A.SCENARIOS]
    x = np.arange(len(names))
    for off, key, col, lab in ((-0.27, "attack", RED, t["a_att"]), (0.0, "T1", ORANGE, t["a_t1"]),
                               (0.27, "T2", GREEN, t["a_t2"])):
        ax1.bar(x + off, [d[n][key] for n in names], 0.26, color=col, label=lab)
    ax1.set_xticks(x)
    ax1.set_xticklabels(t["a_lab"], fontsize=7.5)
    ax1.set_ylabel(t["d_y"])
    ax1.set_title(t["a_title"], fontsize=10)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    names = [n for n, _, _ in I.SCENARIOS]
    x = np.arange(len(names))
    w = 0.2
    for j, noise in enumerate(("aria-1", "forte-1")):
        s = res["ionq"][noise]
        for i, (key, col, lab) in enumerate((("attack", RED, t["i_att"]), ("drift", ORANGE, t["i_dr"]))):
            off = (2 * j + i - 1.5) * w
            ax2.bar(x + off, [s[n][key] for n in names], w, color=col, alpha=1.0 if j == 0 else 0.55,
                    label=f"{noise}: {lab}")
            ax2.plot(x + off, [s["predicted"][n][key] for n in names], "_", color=INK, ms=9, mew=1.5,
                     label=t["i_pred"] if (i, j) == (0, 0) else None)
    ax2.set_xticks(x)
    ax2.set_xticklabels(t["i_lab"], fontsize=7.5)
    ax2.set_ylabel(t["d_y"])
    ax2.set_title(t["i_title"], fontsize=10)
    ax2.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax2.set_ylim(0, 1.35)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    res = load("--recompute" in sys.argv)

    print("§40 channel: per-state readings (m-qg), centre/semi-axis, Q, A_Z, Shor-Preskill fraction")
    for name, kw in B.SCENARIOS:
        e = B.error_probabilities(**kw)
        r = readings(e)
        qz, qx = B.qber(e)
        print(f"  {name:28s} m = {np.round(r['m'], 1)}  c_Z {r['c_z']:6.1f} r_Z {r['r_z']:6.1f}  c_X {r['c_x']:6.1f}"
              f" r_X {r['r_x']:6.1f}  A_Z {r['a_z']:6.1f}  Q_Z {qz:.4f} = (1 - r_Z/1000)/2 = {(1 - r['r_z'] / 1000) / 2:.4f}"
              f"  Q_X {qx:.4f}  key {B.shor_preskill(qz, qx):.3f}")
    e0 = B.error_probabilities()
    for nb in WINDOWS:
        print(f"  sigma of c_Z and r_Z in a window of {nb} bits: {sigma_window(e0, nb):.1f} m-qg;"
              f" single-state m(0) sigma {PB.shot_noise_milliqang(math.acos(1 - 2 * e0[0]), nb // 4):.1f}")
    print("  alarm rates (QBER / qg):")
    for name, _ in B.SCENARIOS:
        print(f"    {name:28s} " + "   ".join(f"N={nb}: {res['monitor'][str(nb)][name][0]:.3f}/{res['monitor'][str(nb)][name][1]:.3f}"
                                             for nb in WINDOWS))

    print("\n§43 finite-key rates:")
    for name in RATE_SCEN:
        r = res["rates"][name]
        print(f"  {name:22s} n_min {r['n_min']:.2e}  " + "  ".join(f"{k}: {v:.3f}" for k, v in r["grid"].items())
              + f"  asym {r['asym']:.3f}  kept at 1e4..1e7: "
              + ", ".join(f"{v / r['asym']:.2f}" for v in r["grid"].values()))
    for n, d in res["diag"].items():
        print(f"  n = {n}: k = {d['k']}, Q_tol = {d['q_tol']:.4f} -> r_X threshold {1000 * (1 - 2 * d['q_tol']):.1f} m-qg")
        for name, _ in F.SCENARIOS:
            v = d[name]
            print(f"    {name:28s} abort {v['abort']:.3f} attack {v['attack']:.3f} drift {v['drift']:.3f}")

    print("\n§49 Pauli identities:")
    for D in (0.0125, 0.025):
        print(f"  cloner D = {D}: max |diff vs IR f = 4D| =",
              f"{np.abs(A.error_probabilities(py=D) - B.error_probabilities(f=4 * D)).max():.1e}")
    f = 0.04
    p_eq = (1 - (1 - 2 * B.BASE['p']) * (1 - f)) / 2
    print(f"  Z-only f = {f}: max |diff vs T2 p' = {p_eq:.4f}| =",
          f"{np.abs(A.error_probabilities(pz=f / 2) - A.error_probabilities(p=p_eq)).max():.1e}")
    for name, _ in A.SCENARIOS:
        v = res["beyond"][name]
        r = readings(A.error_probabilities(**dict(A.SCENARIOS)[name]))
        print(f"  {name:32s} Q_Z {v['q_z']:.4f} Q_X {v['q_x']:.4f} key {v['key_fraction']:.3f} | attack {v['attack']:.3f}"
              f" T1 {v['T1']:.3f} T2 {v['T2']:.3f} | §43 attack {v['attack_s43']:.3f}"
              f" | c_Z {r['c_z']:5.1f} r_Z {r['r_z']:5.1f} r_X {r['r_x']:5.1f}")

    print("\n§44 IonQ noisy simulator (observed / predicted; predicted at n x 25):")
    for noise, s in res["ionq"].items():
        print(f"  {noise}: {s['blocks']} blocks; fits (eps_Z, eps_X, gamma0):",
              ", ".join(f"({a:.4f}, {b:.4f}, {g:.4f})" for a, b, g in s["fits"]))
        base = [v for k, v in s["device_error"].items() if k.endswith("|0.02|none")]
        print(f"    mean error per bit, baseline circuits: {np.mean(base):.4f}")
        for name, _, _ in I.SCENARIOS:
            v, w, x = s[name], s["predicted"][name], s["predicted_1e5"][name]
            print(f"    {name:26s} QBER {v['qber']:.2f}/{w['qber']:.2f} attack {v['attack']:.2f}/{w['attack']:.2f}"
                  f" drift {v['drift']:.2f}/{w['drift']:.2f} | 1e5: attack {x['attack']:.2f} drift {x['drift']:.2f}")

    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        names = {"es": ("cap17_elipse", "cap17_clave_finita", "cap17_ataques"),
                 "en": ("cap17_ellipse", "cap17_finite_key", "cap17_attacks")}[lang]
        for name, fig in zip(names, (fig_ellipse(t, res), fig_finite(t, res), fig_attacks(t, res))):
            path = os.path.join(t["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
