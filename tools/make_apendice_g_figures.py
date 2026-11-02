"""Figure of appendix G of the book: the §170 run on IQM Garnet (the Dirac beta qubit and Zitterbewegung).

Reads examples/data/iqm_zitterbewegung_qg_analysis.json (readout-corrected readings of job
01a19c58-a557-776d-8c9c-fb5a54f43715) and draws the exact readings of the model H = p X + m Z (m = 1, p = 0.75) next
to the measured ones. Writes book/es/figuras/apg_zitterbewegung.png and book/en/figures/apg_zitterbewegung.png.

    python tools/make_apendice_g_figures.py
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "examples", "data", "iqm_zitterbewegung_qg_analysis.json")
BLUE, ORANGE, GREEN, GREY = "#2b78d0", "#e8693a", "#1f9d74", "#9a9a9a"
E, THETA = 1.25, math.atan2(0.75, 1.0)

TEXT = {
    "es": {"dir": os.path.join(ROOT, "book", "es", "figuras"), "x": "tiempo $t$ (unidades $\\hbar/mc^2$)",
           "y": "lectura de $\\beta$ (m-qg, corregida)", "zb_exact": "Zitterbewegung exacto",
           "zb_meas": "desde $|0\\rangle$, Garnet", "ep": "autoestado $E_+$, Garnet", "em": "autoestado $E_-$, Garnet",
           "mean": "media $1000/\\gamma^2=+640$", "plus": "$+1000/\\gamma=+800$", "minus": "$-800$",
           "title": "La lectura de $\\beta$ en IQM Garnet (§170): $v=0{,}6c$, 2000 disparos por punto"},
    "en": {"dir": os.path.join(ROOT, "book", "en", "figures"), "x": "time $t$ (units of $\\hbar/mc^2$)",
           "y": "reading of $\\beta$ (m-qg, corrected)", "zb_exact": "exact Zitterbewegung",
           "zb_meas": "from $|0\\rangle$, Garnet", "ep": "eigenstate $E_+$, Garnet", "em": "eigenstate $E_-$, Garnet",
           "mean": "mean $1000/\\gamma^2=+640$", "plus": "$+1000/\\gamma=+800$", "minus": "$-800$",
           "title": "The reading of $\\beta$ on IQM Garnet (§170): $v=0.6c$, 2000 shots per point"},
}


def main():
    r = json.load(open(DATA, encoding="utf-8"))
    t = np.linspace(0, 2 * math.pi / E, 400)
    exact = 1000 * (math.cos(THETA) ** 2 + math.sin(THETA) ** 2 * np.cos(2 * E * t))
    for lang, tx in TEXT.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, exact, color=BLUE, lw=1.6, label=tx["zb_exact"])
        ax.errorbar([z["t"] for z in r["zb"]], [z["m"] for z in r["zb"]], yerr=[z["sigma"] for z in r["zb"]],
                    fmt="o", color=BLUE, ms=5, capsize=2, label=tx["zb_meas"])
        ax.axhline(640, color=BLUE, ls=":", lw=1, label=tx["mean"])
        ax.axhline(800, color=GREEN, ls="--", lw=1, label=tx["plus"])
        ax.errorbar([x["t"] for x in r["eplus"]], [x["m"] for x in r["eplus"]], yerr=[x["sigma"] for x in r["eplus"]],
                    fmt="s", color=GREEN, ms=6, capsize=2, label=tx["ep"])
        ax.axhline(-800, color=ORANGE, ls="--", lw=1, label=tx["minus"])
        ax.errorbar([x["t"] for x in r["eminus"]], [x["m"] for x in r["eminus"]],
                    yerr=[x["sigma"] for x in r["eminus"]], fmt="D", color=ORANGE, ms=6, capsize=2, label=tx["em"])
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_xlabel(tx["x"])
        ax.set_ylabel(tx["y"])
        ax.set_ylim(-1050, 1100)
        ax.set_title(tx["title"], fontsize=11)
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        fig.tight_layout()
        os.makedirs(tx["dir"], exist_ok=True)
        path = os.path.join(tx["dir"], "apg_zitterbewegung.png")
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
