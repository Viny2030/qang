"""Figure of the QML note. Numbers transcribed from the output of
examples/qnn_seeds_qg.py (seeds 80-84, 60 runs per model); see RESEARCH_NOTES §80."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ES = len(sys.argv) > 1 and sys.argv[1] == "es"
T = (lambda en, es: es) if ES else (lambda en, es: en)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})

# (a) mean accuracy, trained without noise: (without qang, with qang); exact
rows = [
    (T("T1", "T1"), "E", 0.928, 0.952), (T("unequal T1", "T1 desigual"), "E", 0.921, 0.947),
    (T("dephasing", "desfase"), "E", 0.792, 0.918),
    (T("T1", "T1"), "W", 0.757, 0.922), (T("unequal T1", "T1 desigual"), "W", 0.732, 0.916),
    (T("dephasing", "desfase"), "W", 0.656, 0.882),
]
exact = {"E": 0.952, "W": 0.922}
# (b) paired differences, mean and 95% CI across 5 seeds
diffs = [
    (T("qang gain, trained clean, T1", "ganancia qang, sin ruido, T1"), (0.025, 0.008, 0.041), (0.165, 0.118, 0.212)),
    (T("qang gain, trained clean, dephasing", "ganancia qang, sin ruido, desfase"), (0.126, 0.091, 0.162), (0.226, 0.191, 0.262)),
    (T("qang - without, noise-aware training", "qang - sin, entrenada con ruido"), (0.001, -0.006, 0.009), (-0.001, -0.008, 0.005)),
    (T("loss from unequal T1", "pérdida por T1 desigual"), (0.005, -0.002, 0.012), (0.007, -0.000, 0.013)),
    (T("dephasing: aware - clean (qang)", "desfase: con ruido - sin (qang)"), (0.032, 0.011, 0.053), (0.045, 0.034, 0.056)),
]

fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 5.6), gridspec_kw={"height_ratios": [1, 0.95]})
y = np.arange(len(rows))[::-1]
for yi, (cond, m, raw, q) in zip(y, rows):
    c = BLUE if m == "E" else ORANGE
    a.plot([raw, q], [yi, yi], color=c, lw=2, solid_capstyle="round")
    a.plot(raw, yi, "o", mfc="white", mec=c, mew=1.6, ms=7)
    a.plot(q, yi, "o", color=c, ms=7)
    a.plot(exact[m], yi + 0.28, "v", color=MUTED, ms=4)
a.set_yticks(y, [f"{m}, {cond}" for cond, m, _, _ in rows])
a.set_xlim(0.6, 1.0)
a.set_xlabel(T("test accuracy (mean of 60 runs)", "precisión de prueba (media de 60 corridas)"))
a.set_title(T("(a) trained without noise, run under the noise", "(a) entrenada sin ruido, ejecutada con ruido"), fontsize=8.5, color=INK, loc="left")
a.grid(axis="x", color=GRID, lw=0.8)
a.plot([], [], "o", mfc="white", mec=MUTED, mew=1.6, ms=7, label=T("without qang", "sin qang"))
a.plot([], [], "o", color=MUTED, ms=7, label=T("with qang", "con qang"))
a.plot([], [], "v", color=MUTED, ms=5, label=T("noiseless", "sin ruido"))
a.legend(frameon=False, fontsize=8, loc="upper left")

yb = np.arange(len(diffs))[::-1]
for yi, (lab, e, w) in zip(yb, diffs):
    for (mu, lo, hi), c, off, mk in ((e, BLUE, 0.13, "o"), (w, ORANGE, -0.13, "s")):
        b.plot([lo * 100, hi * 100], [yi + off] * 2, color=c, lw=2, solid_capstyle="round")
        b.plot(mu * 100, yi + off, mk, color=c, ms=5.5, mec="white", mew=0.8)
b.axvline(0, color=MUTED, lw=0.8)
b.set_yticks(yb, [d[0] for d in diffs])
b.set_xlabel(T("difference, points (mean and 95% CI over 5 seeds)", "diferencia, puntos (media e IC 95 % sobre 5 semillas)"))
b.set_title(T("(b) paired differences", "(b) diferencias pareadas"), fontsize=8.5, color=INK, loc="left")
b.grid(axis="x", color=GRID, lw=0.8)
b.plot([], [], "o-", color=BLUE, label=T("E, weight 1", "E, peso 1"))
b.plot([], [], "s-", color=ORANGE, label=T("W, weight 2", "W, peso 2"))
b.legend(frameon=False, fontsize=8, loc="lower right")
for ax in (a, b):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
fig.tight_layout()
name = "fig_qml_es" if ES else "fig_qml"
fig.savefig(name + ".pdf")
fig.savefig(name + ".png", dpi=200)
