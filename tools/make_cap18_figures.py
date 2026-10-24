"""Figures of chapter 18 of the book (certified quantum random numbers read in qg: the private randomness of a qubit,
the readout calibration of §31, and the leak on IonQ's noisy simulator), Spanish and English, on the chapter 1 base.

    python tools/make_cap18_figures.py

writes book/es/figuras/cap18_*.png and book/en/figures/cap18_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. The physics lives in examples/qrng_qg_certified.py (§41) and examples/ionq_sim_qrng.py
(§42); this script reuses them. The §31 readout numbers are exact (the standard calibration's bias is p*b in closed
form). The IonQ part only reads the counts already stored in examples/data/ionq_sim_results.json; nothing is sent
anywhere. Runs in about a minute.
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

import hardware_characterization_qg as HC  # noqa: E402
import ionq_sim_qrng as R  # noqa: E402
import qrng_qg_certified as Q  # noqa: E402

RESULTS = os.path.join(ROOT, "examples", "data", "ionq_sim_results.json")
P_TH = (0.0, 0.01, 0.03, 0.05, 0.08)
TEST_ROUNDS = (10**3, 10**4, 10**5, 10**6, 10**7)

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "h_x": r"lectura (m-qg)",
        "h_y": "bits aleatorios por disparo",
        "h_cert": r"certificados: $H_{\min}(Z|E)$ frente a $m_\perp=\sqrt{m_X^2+m_Y^2}$",
        "h_naive": r"ingenuo: $-\log_2\max(p_0,p_1)$ frente a $|m_Z|$",
        "h_eve": r"lo que sabe Eve: $m_E=\sqrt{1000^2-m_\perp^2}$",
        "h_title": "La aleatoriedad privada vive en el plano transversal",
        "e_x": "visibilidad V (desfase T2)",
        "e_y": "bits certificados por disparo",
        "e_truth": "verdad",
        "e_title": "Estimadores, lectura del §31, $10^4$ rondas de prueba",
        "names": {"naive": "ingenuo (sesgo de salida)", "qg one-sided": "qg, un solo giro",
                  "qg +/- pairs": "qg, pares ±", "qg calibrated": "qg calibrado (§31)"},
        "n_x": "rondas de prueba por ajuste",
        "n_y": "bits certificados (cota a 3σ)",
        "n_title": "Junto al polo la certificación pide disparos",
        "c_x": "temperatura efectiva (mK)",
        "c_y": r"$e_{01}$ estimado",
        "c_std": "calibración estándar: $e_{01}+p\\,b$",
        "c_qg": "protocolo qg del §31",
        "c_true": "verdad, $e_{01}=0{,}015$",
        "c_title": "La calibración estándar lee la temperatura como error",
        "i_x": r"fuga $\theta$ (vueltas de la compuerta MS)",
        "i_title": "Simulador ruidoso de IonQ (3 corridas por modelo)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "h_x": r"reading (m-qg)",
        "h_y": "random bits per shot",
        "h_cert": r"certified: $H_{\min}(Z|E)$ against $m_\perp=\sqrt{m_X^2+m_Y^2}$",
        "h_naive": r"naive: $-\log_2\max(p_0,p_1)$ against $|m_Z|$",
        "h_eve": r"what Eve knows: $m_E=\sqrt{1000^2-m_\perp^2}$",
        "h_title": "Private randomness lives in the transverse plane",
        "e_x": "visibility V (T2 dephasing)",
        "e_y": "certified bits per shot",
        "e_truth": "truth",
        "e_title": "Estimators, §31 readout, $10^4$ test rounds",
        "names": {"naive": "naive (output bias)", "qg one-sided": "qg, one rotation",
                  "qg +/- pairs": "qg, ± pairs", "qg calibrated": "qg calibrated (§31)"},
        "n_x": "test rounds per setting",
        "n_y": "certified bits (3σ bound)",
        "n_title": "Near the pole certification is shot-hungry",
        "c_x": "effective temperature (mK)",
        "c_y": r"estimated $e_{01}$",
        "c_std": "standard calibration: $e_{01}+p\\,b$",
        "c_qg": "qg protocol of §31",
        "c_true": "truth, $e_{01}=0.015$",
        "c_title": "The standard calibration reads temperature as error",
        "i_x": r"leak $\theta$ (turns of the MS gate)",
        "i_title": "IonQ noisy simulator (3 runs per model)",
    },
}
STYLE = {"naive": GREY, "qg one-sided": ORANGE, "qg +/- pairs": GREEN, "qg calibrated": BLUE}


def eve_reading(r_perp):
    """m-qg reading of Eve's guessing probability: P_guess = (1 + sqrt(1 - r^2))/2 = (1000 + m_E)/2000."""
    return 1000 * math.sqrt(max(0.0, 1 - r_perp**2))


def lcb_bits(r_true, n_test, sigmas=3.0):
    """Certified bits from a sigmas lower bound on r_perp with n_test rounds (no readout error)."""
    return Q.h_min_certified(max(0.0, r_true - sigmas / math.sqrt(n_test)))


def standard_vs_qg(p):
    """e01 the standard suite reports (exact: e01 + p b) and the T_eff of the thermal population."""
    a, b = HC.readout_affine(HC.TRUE["e01"], HC.TRUE["e10"])
    qg_eq = 1 - 2 * p
    e01_std = (1 - (a + b * qg_eq)) / 2
    e10_std = (1 + (a - b * qg_eq)) / 2
    return e01_std, e10_std, 1e3 * float(HC.t_eff_from_qg(qg_eq)), 1000 * qg_eq


def unsafe_with_standard_b(reps=300, seed=5):
    """Thermal source: dividing by the standard suite's b (= b qg_eq) inflates r_perp."""
    rng = np.random.default_rng(seed)
    a, b = Q.readout_map()
    out = {}
    for p in (0.01, 0.03, 0.05):
        bloch = Q.device_state(visibility=0.9, p_thermal=p)
        t = Q.truth(bloch)
        b_std = b * (1 - 2 * p)
        v = np.array([Q.estimates(bloch, 10000, rng, b_calibrated=b_std)["qg calibrated"] for _ in range(reps)])
        w = np.array([Q.estimates(bloch, 10000, rng)["qg calibrated"] for _ in range(reps)])
        out[p] = (t, float(v.mean()), float(np.mean(v > t)), float(w.mean()), float(np.mean(w > t)))
    return out


def ionq_summary():
    with open(RESULTS, encoding="utf-8") as fh:
        res = json.load(fh)
    out = {}
    for noise in ("aria-1", "forte-1"):
        entries = [res[k] for k in sorted(res) if k.startswith(f"ionq_sim|{noise}|qrng|")]
        out[noise] = {"summary": R.summarize(entries), "b": [e["readout_b"] for e in entries],
                      "a": [e["readout_a"] for e in entries], "runs": len(entries)}
    return out


def fig_private(t, curve):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))
    m = np.linspace(0, 1000, 401)
    ax1.plot(m, [Q.h_min_certified(x / 1000) for x in m], color=BLUE, lw=2, label=t["h_cert"])
    ax1.plot(m, [Q.h_min_naive((1000 + x) / 2000) for x in m], color=GREY, lw=2, ls="--", label=t["h_naive"])
    ax1b = ax1.twinx()
    ax1b.plot(m, [eve_reading(x / 1000) for x in m], color=ORANGE, lw=1.5, ls=":", label=t["h_eve"])
    ax1b.set_ylabel("$m_E$ (m-qg)", color=ORANGE)
    ax1b.set_ylim(0, 1050)
    ax1.set_xlabel(t["h_x"])
    ax1.set_ylabel(t["h_y"])
    ax1.set_ylim(0, 1.05)
    ax1.set_title(t["h_title"], fontsize=10)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax1.grid(alpha=0.3)
    vs, truth, est = curve
    ax2.plot(vs, truth, color=INK, lw=2, label=t["e_truth"])
    for e in Q.ESTIMATORS:
        ax2.plot(vs, est[e], "o", color=STYLE[e], ms=4, label=t["names"][e])
    ax2.set_xlabel(t["e_x"])
    ax2.set_ylabel(t["e_y"])
    ax2.set_title(t["e_title"], fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_readout(t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = np.logspace(2.5, 7.5, 60)
    for v, ls in ((1.0, "-"), (0.95, "--"), (0.8, ":")):
        r = v
        ax1.plot(ns, [lcb_bits(r, n) for n in ns], ls, color=BLUE, lw=2, label=f"$m_\\perp$ = {1000 * r:.0f} m-qg")
        ax1.axhline(Q.h_min_certified(r), color=BLUE, lw=0.7, alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel(t["n_x"])
    ax1.set_ylabel(t["n_y"])
    ax1.set_title(t["n_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="lower right")
    ax1.grid(alpha=0.3)
    ps = np.linspace(0.002, 0.1, 50)
    rows = [standard_vs_qg(p) for p in ps]
    ax2.plot([r[2] for r in rows], [r[0] for r in rows], color=GREY, lw=2, label=t["c_std"])
    ax2.axhline(HC.TRUE["e01"], color=BLUE, lw=2, label=t["c_qg"])
    ax2.set_xlabel(t["c_x"])
    ax2.set_ylabel(t["c_y"])
    ax2.set_title(t["c_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_ionq(t, ion):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    th = np.array(R.THETAS)
    fine = np.linspace(0, 0.13, 100)
    for ax, noise in zip(axes, ("aria-1", "forte-1")):
        s = ion[noise]["summary"]
        ax.plot(fine, [R.truth(x) for x in fine], color=INK, lw=2, label=t["e_truth"])
        for e in Q.ESTIMATORS:
            if e == "qg calibrated":
                continue
            ax.plot(th, [s[x][e] for x in R.THETAS], "o-", color=STYLE[e], ms=5, lw=1, label=t["names"][e])
        ax.set_xlabel(t["i_x"])
        ax.set_title(f"{t['i_title']}: {noise}", fontsize=9.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(t["e_y"])
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def main():
    print("§41: the private randomness on the chapter 1 scale")
    for mp in (1000, 990, 950, 900, 800, 500, 200):
        r = mp / 1000
        print(f"  m_perp {mp:5d}: H_min {Q.h_min_certified(r):.3f} bits, Eve's reading m_E {eve_reading(r):6.1f} m-qg,"
              f" P_guess {(1000 + eve_reading(r)) / 2000:.4f}, qang angle beta = asin r = {math.asin(r):.3f} rad")
    for mz in (0, 50, 100, 300):
        print(f"  naive at |m_Z| = {mz}: {Q.h_min_naive((1000 + mz) / 2000):.3f} bits")

    t = Q.table()
    print("\n§41 table (mean estimate, fraction unsafe):")
    for name, dev, _ in Q.SCENARIOS:
        r = t[name]
        b = Q.device_state(**dev)
        print(f"  {name:28s} r_perp {math.hypot(b[0], b[1]):.3f} (m_perp {1000 * math.hypot(b[0], b[1]):.0f})"
              f" truth {r['truth']:.3f} | " + " | ".join(f"{e} {r[e][0]:.3f} ({r[e][1]:.2f})" for e in Q.ESTIMATORS))

    print("\nNear-perfect source, 3-sigma bound without readout error (closed form):")
    for n in TEST_ROUNDS:
        print(f"  {n:9d} rounds: {lcb_bits(1.0, n):.3f} bits (m_perp bound {1000 * (1 - 3 / math.sqrt(n)):.0f},"
              f" Eve's reading {eve_reading(1 - 3 / math.sqrt(n)):.0f} m-qg); V = 0.95: {lcb_bits(0.95, n):.3f}"
              f" of {Q.h_min_certified(0.95):.3f}")
    rng = np.random.default_rng(2)
    for n in (10**4, 10**5, 10**6):
        v = np.mean([Q.estimates(Q.device_state(1.0), n, rng)["qg calibrated"] for _ in range(100)])
        print(f"  with §31 readout, calibrated, {n} rounds: {v:.3f}")

    print("\n§31: the standard calibration and the thermal population (e01 = 0.015, e10 = 0.04, 5 GHz):")
    for p in P_TH:
        e01s, e10s, teff, meq = standard_vs_qg(p)
        print(f"  p {p:.2f}: T_eff {teff:5.1f} mK, m_eq {meq:.0f} m-qg; standard e01 {e01s:.4f}, e10 {e10s:.4f}")
    print("  thermal source V = 0.9, dividing by the standard b vs the true b (truth, mean, unsafe):")
    for p, v in unsafe_with_standard_b().items():
        print(f"    p {p:.2f}: truth {v[0]:.3f}; standard b {v[1]:.3f} ({v[2]:.2f} unsafe); true b {v[3]:.3f} ({v[4]:.2f})")

    print("\n§42 IonQ noisy simulator (stored counts):")
    ion = ionq_summary()
    for noise, d in ion.items():
        print(f"  {noise}: {d['runs']} runs, readout b {d['b']}, a {d['a']}")
        for th, s in d["summary"].items():
            frac = s["qg +/- pairs"] / s["truth"]
            print(f"    theta {th:.2f}: truth {s['truth']:.3f} | " + " ".join(f"{e} {s[e]:.3f}" for e in Q.ESTIMATORS)
                  + f" | r {s['r_calibrated']:.3f} vs ideal {abs(math.cos(2 * math.pi * th)):.3f} | +/- captures {frac:.2f}"
                  + f" | unsafe naive {s['unsafe']['naive']:.2f}, qg {max(s['unsafe'][e] for e in Q.ESTIMATORS[1:]):.2f}")

    rng = np.random.default_rng(1)
    vs = np.linspace(0.05, 1.0, 20)
    est = {e: [] for e in Q.ESTIMATORS}
    for v in vs:
        rows = [Q.estimates(Q.device_state(v), 10000, rng) for _ in range(40)]
        for e in Q.ESTIMATORS:
            est[e].append(np.mean([r[e] for r in rows]))
    curve = (vs, [Q.truth(Q.device_state(v)) for v in vs], est)

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap18_privada", "cap18_lectura", "cap18_ionq"),
                 "en": ("cap18_private", "cap18_readout", "cap18_ionq")}[lang]
        for name, fig in zip(names, (fig_private(tx, curve), fig_readout(tx), fig_ionq(tx, ion))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
