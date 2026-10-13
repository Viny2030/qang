"""Figure of the IBM paper, §147: Trotter error against steps, and quantum-walk spreading (raw vs filtered)."""
import json
import os
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "examples", "data", "ibm_trotter_walk_qg.json")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
RAW, FILT = "#eb6834", "#2a78d6"
RS = (1, 2, 3, 4, 6, 8, 12)
TIMES = (0.4, 0.8, 1.2, 1.6)
CLASSICAL = (0.791, 1.513, 2.113, 2.585)

T = {"en": {"a": "(a) Trotter simulation, n = 5, T = 3", "b": "(b) quantum walk, n = 7",
            "steps": "Trotter steps r", "err": "mean |P_k - exact|", "noiseless": "noiseless Trotter",
            "raw": "raw (median of 21)", "filt": "filter + read-out (median of 21)", "t": "time t (1/J)",
            "s2": r"spread $\sigma^2$", "quantum": "exact quantum walk", "classical": "classical random walk"},
     "es": {"a": "(a) simulación de Trotter, n = 5, T = 3", "b": "(b) caminata cuántica, n = 7",
            "steps": "pasos de Trotter r", "err": "|P_k - exacto| medio", "noiseless": "Trotter sin ruido",
            "raw": "crudo (mediana de 21)", "filt": "filtro + lectura (mediana de 21)", "t": "tiempo t (1/J)",
            "s2": r"dispersión $\sigma^2$", "quantum": "caminata cuántica exacta", "classical": "caminata clásica"}}


def main(lang="en"):
    d = json.load(open(DATA))
    rows = [r for r in d["rows"] if "trotter_error" in r]
    L = T[lang]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    med = lambda k, f: np.median([r[f][k] for r in rows], axis=0)  # noqa: E731
    a.plot(RS, d["noiseless_trotter_error"], color=MUTED, ls="--", marker="o", ms=3, label=L["noiseless"])
    a.plot(RS, med("raw", "trotter_error"), color=RAW, marker="o", ms=4, label=L["raw"])
    a.plot(RS, med("filtered_readout", "trotter_error"), color=FILT, marker="o", ms=4, label=L["filt"])
    a.set_xlabel(L["steps"], color=INK2)
    a.set_ylabel(L["err"], color=INK2)
    a.set_title(L["a"], color=INK, fontsize=9.5, loc="left")
    a.set_xticks(RS)
    a.set_ylim(0, 0.42)
    if lang == "es":
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    b.plot(TIMES, d["sigma2_trotterized"], color=MUTED, ls="--", marker="o", ms=3, label=L["quantum"])
    b.plot(TIMES, CLASSICAL, color=INK2, ls=":", marker="s", ms=3, label=L["classical"])
    b.plot(TIMES, med("raw", "sigma2"), color=RAW, marker="o", ms=4, label=L["raw"])
    b.plot(TIMES, med("filtered_readout", "sigma2"), color=FILT, marker="o", ms=4, label=L["filt"])
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks(TIMES)
    b.set_xticklabels([str(t).replace(".", "," if lang == "es" else ".") for t in TIMES])
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ys = (0.3, 0.5, 1, 2, 3, 5)
    b.set_yticks(ys)
    b.set_yticklabels([str(y).replace(".", "," if lang == "es" else ".") for y in ys])
    b.set_xlabel(L["t"], color=INK2)
    b.set_ylabel(L["s2"], color=INK2)
    b.set_title(L["b"], color=INK, fontsize=9.5, loc="left")
    for ax in (a, b):
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    suf = "" if lang == "en" else "_es"
    out = os.path.join(HERE, "..", "ibm" + suf, f"fig_trotter{suf}")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "en")
