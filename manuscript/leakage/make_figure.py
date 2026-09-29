"""Figure of the leakage note. Numbers transcribed from the outputs of
examples/qutrit_bayes_weight_qg.py (seed 711) and
examples/transmon_leakage_channel_qg.py (seed 72); see RESEARCH_NOTES §71-§72."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

import numpy as np

ES = len(sys.argv) > 1 and sys.argv[1] == "es"
T = (lambda en, es: es) if ES else (lambda en, es: en)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})

models = ["M0", "M(0)", "M(0.25)", "M(0.5)", "M(1)"]
std = [0.0154, 0.0177, 0.0182, 0.0189, 0.0202]
era = [0.0163, 0.0171, 0.0175, 0.0181, 0.0192]
bay = [0.0153, 0.0161, 0.0170, 0.0180, 0.0192]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
x = np.arange(len(models))
for y, c, m, lab in ((std, BLUE, "o", T("standard", "estándar")), (era, ORANGE, "s", T("erasure", "borrado")), (bay, AQUA, "^", T("Bayesian weight", "peso bayesiano"))):
    a.plot(x, np.array(y) * 1e3, color=c, marker=m, ms=5, lw=1.6, label=lab)
a.set_xticks(x, models)
a.set_ylabel(T("logical error", "error lógico") + " ($\\times 10^{-3}$)")
a.set_title(T("(a) flag decoders, d = 3, T1-dominated", "(a) decodificadores con marca, d = 3"), fontsize=9, color=INK, loc="left")
a.grid(axis="y", color=GRID, lw=0.8)
a.legend(frameon=False, fontsize=8, loc="upper left")

cats = ["M0", "M(0)", "M(0.5)", "M(1)", T("transmon\n(§72)", "transmón\n(§72)")]
g3 = [1.65, 1.62, 1.48, 1.37, 1.75]  # ratio_P4 as printed by the scripts
g5 = [2.11, 2.08, 1.77, 1.55, 1.94]
xb = np.arange(len(cats))
w = 0.38
b.bar(xb - w / 2 - 0.01, np.array(g3) - 1, w, bottom=1, color=BLUE, label="d = 3")
b.bar(xb + w / 2 + 0.01, np.array(g5) - 1, w, bottom=1, color=ORANGE, label="d = 5")
b.axhline(1, color=MUTED, lw=0.8)
b.set_xticks(xb, cats)
b.set_ylim(0.95, 2.35)
b.set_ylabel(T("gain of qg + Bayesian weight", "ganancia de qg + peso bayesiano"))
b.set_title(T("(b) over the best flag decoder, T1-dominated", "(b) sobre el mejor con marca, dominado por T1"), fontsize=9, color=INK, loc="left")
b.grid(axis="y", color=GRID, lw=0.8)
b.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
for i in (3, 4):
    b.text(xb[i] - w / 2, g3[i] + 0.03, f"{g3[i]:.2f}", ha="center", fontsize=7, color=INK)
    b.text(xb[i] + w / 2, g5[i] + 0.03, f"{g5[i]:.2f}", ha="center", fontsize=7, color=INK)
for ax in (a, b):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
name = "fig_leakage_es" if ES else "fig_leakage"
fig.savefig(name + ".pdf")
fig.savefig(name + ".png", dpi=200)
print("gains d3", [round(v, 2) for v in g3], "d5", [round(v, 2) for v in g5])
