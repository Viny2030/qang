"""Figures of chapter 29 of the book (the confirmed empirical advantages), Spanish and English, on the chapter 1 base.

    python tools/make_cap29_figures.py

writes book/es/figuras/cap29_*.png and book/en/figures/cap29_*.png and prints the numbers the chapter quotes. Nothing
is re-simulated: every factor is recomputed, device by device, from the stored results of the notes' pre-registered runs
(§137, §144, §146, §147, §149, §151, §163, §165) and from the chapter 17 BB84 cache (§40). The factor is the raw error
divided by the error with the filter plus readout correction (filter alone where no readout correction exists).
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402

DATA = os.path.join(ROOT, "examples", "data")
RED = "#8c2d04"

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "f_x": "error crudo / error con filtro (+ lectura)",
        "f_title": "Cuánto baja el error, dispositivo por dispositivo",
        "f_studies": {"runtime": "10 circuitos de peso 1\n(§137, 3 disp.)", "fleet": "flota IBM\n(§144, 17 disp.)",
                      "pca": "espectro PCA\n(§146, 21 disp.)", "walk": "Trotter, mejor error\n(§147, 21 disp.)",
                      "inv": "test de inversión\n(§149, 6 disp.)", "nu": "neutrinos 3 sabores\n(§151, 3 disp.)",
                      "sterile": "neutrino estéril\n(§163, 3 disp.)", "iqm": "neutrinos en IQM Garnet\n(§165, hardware real)"},
        "f_med": "mediana",
        "b_y": "fracción de ventanas con alarma",
        "b_qber": "monitor de QBER",
        "b_qg": "monitor qg",
        "b_scen": {"baseline": "línea\nde base", "T1 drift, gamma 0.04": "deriva T1\nγ = 0,04",
                   "T1 drift, gamma 0.06": "deriva T1\nγ = 0,06", "intercept-resend f=0.05": "ataque\nf = 0,05",
                   "intercept-resend f=0.10": "ataque\nf = 0,10", "T1-mimicking attack, +0.04": "ataque que\nimita T1"},
        "b_title": "BB84: falsas alarmas por deriva térmica (§40, 2000 bits)",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "f_x": "raw error / error with filter (+ readout)",
        "f_title": "How much the error falls, device by device",
        "f_studies": {"runtime": "10 weight-1 circuits\n(§137, 3 dev.)", "fleet": "IBM fleet\n(§144, 17 dev.)",
                      "pca": "PCA spectrum\n(§146, 21 dev.)", "walk": "Trotter, best error\n(§147, 21 dev.)",
                      "inv": "inversion test\n(§149, 6 dev.)", "nu": "3-flavour neutrinos\n(§151, 3 dev.)",
                      "sterile": "sterile neutrino\n(§163, 3 dev.)", "iqm": "neutrinos on IQM Garnet\n(§165, real hardware)"},
        "f_med": "median",
        "b_y": "fraction of windows with an alarm",
        "b_qber": "QBER monitor",
        "b_qg": "qg monitor",
        "b_scen": {"baseline": "baseline", "T1 drift, gamma 0.04": "T1 drift\nγ = 0.04",
                   "T1 drift, gamma 0.06": "T1 drift\nγ = 0.06", "intercept-resend f=0.05": "attack\nf = 0.05",
                   "intercept-resend f=0.10": "attack\nf = 0.10", "T1-mimicking attack, +0.04": "T1-mimicking\nattack"},
        "b_title": "BB84: false alarms from thermal drift (§40, 2000 bits)",
    },
}


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def ratios():
    out = {}
    out["runtime"] = [r["error"]["raw"] / r["error"]["filtered_readout"] for r in load("ibm_runtime_qg.json")["rows"]]
    out["fleet"] = [r["error"]["raw"] / r["error"]["filtered_readout"] for r in load("ibm_fleet_qg.json")["rows"]
                    if r.get("error")]   # FakeCusco crashed in the run (§144) and has no result
    pca = load("ibm_pca_spectrum_qg.json")
    ex = np.array(pca["exact"])
    out["pca"] = [np.abs(np.array(r["estimate"]["raw"]) - ex).sum() / np.abs(np.array(r["estimate"]["filtered_readout"]) - ex).sum()
                  for r in pca["rows"]]
    out["walk"] = [r["best_error"]["raw"] / r["best_error"]["filtered_readout"] for r in load("ibm_trotter_walk_qg.json")["rows"]]
    out["inv"] = [r["inversion"]["error"]["raw"] / r["inversion"]["error"]["filtered_readout"]
                  for r in load("ibm_swap_test_qg.json")["rows"]]
    out["nu"] = [r["error"]["raw"] / r["error"]["filtered_readout"] for r in load("neutrino_oscillation_qg.json")["rows"]]
    st = load("neutrino_ordering_majorana_sterile_qg.json")["S_errors"]
    by_dev = {}
    for k, v in st.items():
        dev = k.split(",")[0]
        by_dev.setdefault(dev, [0.0, 0.0])
        by_dev[dev][0] += v["raw"]
        by_dev[dev][1] += v["filtered"]
    out["sterile"] = [a / b for a, b in by_dev.values()]
    iqm = load("iqm_neutrino_qg_run.json")
    out["iqm"] = [iqm["raw_error"] / iqm["filtered_error"]]
    return out


def fig_factors(t, r):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    keys = list(r)
    for i, k in enumerate(keys):
        y = len(keys) - 1 - i
        v = np.array(r[k])
        c = RED if k == "iqm" else BLUE
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(v)) if len(v) > 1 else np.zeros(1)
        ax.plot(v, y + jitter, "*" if k == "iqm" else "o", color=c, ms=12 if k == "iqm" else 5, alpha=0.8)
        ax.plot([np.median(v)] * 2, [y - 0.3, y + 0.3], color=INK, lw=2)
    ax.plot([], [], color=INK, lw=2, label=t["f_med"])
    ax.axvline(1, color=GREY, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([t["f_studies"][k] for k in reversed(keys)], fontsize=8)
    ax.set_xlabel(t["f_x"])
    ax.set_title(t["f_title"], fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def fig_bb84(t, mon):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    scen = list(t["b_scen"])
    x = np.arange(len(scen))
    ax.bar(x - 0.19, [mon[s][0] for s in scen], 0.36, color=ORANGE, label=t["b_qber"])
    ax.bar(x + 0.19, [mon[s][1] for s in scen], 0.36, color=BLUE, label=t["b_qg"])
    ax.set_xticks(x)
    ax.set_xticklabels([t["b_scen"][s] for s in scen], fontsize=8)
    ax.set_ylabel(t["b_y"])
    ax.set_ylim(0, 1.05)
    ax.set_title(t["b_title"], fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    r = ratios()
    print("raw / (filter + readout) error, per device:")
    for k, v in r.items():
        v = np.array(v)
        print(f"  {k:8s} n={len(v):2d}  min {v.min():.2f}  median {np.median(v):.2f}  max {v.max():.2f}")
    fleet = [row for row in load("ibm_fleet_qg.json")["rows"] if row.get("error")]
    best = [row["error"]["raw"] / min(row["error"]["filtered"], row["error"]["filtered_readout"]) for row in fleet]
    print(f"  fleet, raw / best of (filter, filter + readout): {min(best):.1f}..{max(best):.1f}")
    mon = load("cap17_bb84.json")["monitor"]["2000"]
    print("\nBB84 §40 alarm rates, 2000 bits (QBER / qg):")
    for s, (a, b) in mon.items():
        print(f"  {s:30s} {a:.3f} / {b:.3f}")
    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap29_factores", "cap29_bb84"), "en": ("cap29_factors", "cap29_bb84")}[lang]
        for name, fig in zip(names, (fig_factors(tx, r), fig_bb84(tx, mon))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
