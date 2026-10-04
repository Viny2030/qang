"""Figure of the radius note: (a) §98, estimated against true deficit;
(b, c) §100, device noise models. python make_figure.py [es]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qang.qml import WeightQNN
from qang.sectors import filter_distribution

ES = len(sys.argv) > 1 and sys.argv[1] == "es"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e6e5e0"
T = {
    "with": "con qang" if ES else "with qang", "without": "sin qang" if ES else "without qang",
    "a_x": "déficit verdadero $1-qg_Z^2$" if ES else "true deficit $1-qg_Z^2$",
    "a_y": "déficit estimado (1000 disparos)" if ES else "estimated deficit (1000 shots)",
    "a_t": "(a) T1 igual, 6 qubits,\npeso 2, profundidad 24" if ES else "(a) equal T1, 6 qubits,\nweight 2, depth 24",
    "b_t": "(b) error del déficit,\ncircuitos entrenados" if ES else "(b) deficit error,\ntrained circuits",
    "c_t": "(c) déficit falso,\nestados producto (eco)" if ES else "(c) false deficit,\nproduct states (echo)",
}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(7.2, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 0.85], wspace=0.25)

# (a)
rng = np.random.default_rng(98)
m = WeightQNN(6, 2, layers=8)
tr, wq, wo = [], [], []
for _ in range(10):
    th = rng.uniform(-np.pi, np.pi, m.n_theta)
    psi = np.zeros((1, m.dim)); psi[0, m.idx[2]] = rng.normal(size=len(m.idx[2])); psi /= np.linalg.norm(psi)
    p0 = m.probs(th, psi)[0]; p1 = m.probs(th, psi, 0.02)[0]; p1 = np.clip(p1, 0, None); p1 /= p1.sum()
    c = rng.multinomial(1000, p1)
    for dist, n, out in ((c, c.sum(), wo), (np.where(m.wt == 2, c, 0), c[m.wt == 2].sum(), wq)):
        q = (dist @ m.zsign) / n
        out.extend(1 - (n * q * q - 1) / (n - 1))
    tr.extend(1 - (p0 @ m.zsign) ** 2)
ax = fig.add_subplot(gs[0])
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
ax.scatter(tr, wo, s=16, marker="^", facecolor="white", edgecolor=ORANGE, lw=1.2, label=T["without"], zorder=2)
ax.scatter(tr, wq, s=14, marker="o", color=BLUE, edgecolor="white", lw=0.6, label=T["with"], zorder=3)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel(T["a_x"]); ax.set_ylabel(T["a_y"]); ax.set_title(T["a_t"], fontsize=8.5, color=INK, loc="left")
ax.grid(color=GRID, lw=0.6)
h, l = ax.get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), fontsize=8)

# (b), (c): §100
names = ["IBM brisbane", "IBM sherbrooke", "IBM torino", "IonQ aria-1", "IonQ forte-1"]
b_q = [0.079, 0.055, 0.051, 0.074, 0.082]; b_r = [0.257, 0.221, 0.197, 0.245, 0.272]
c_q = [0.66, 0.49, 0.37, 0.48, 0.54]; c_r = [0.77, 0.67, 0.56, 0.62, 0.71]
for k, (vq, vr, title, xmax) in enumerate(((b_q, b_r, T["b_t"], 0.3), (c_q, c_r, T["c_t"], 0.9))):
    ax = fig.add_subplot(gs[k + 1])
    y = np.arange(len(names))[::-1]
    for yi, a, b in zip(y, vq, vr):
        ax.plot([a, b], [yi, yi], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    ax.scatter(vr, y, s=26, marker="^", facecolor="white", edgecolor=ORANGE, lw=1.2, label=T["without"], zorder=2)
    ax.scatter(vq, y, s=24, marker="o", color=BLUE, edgecolor="white", lw=0.6, label=T["with"], zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([""] * len(names))
    if k == 0:
        for yi, nm in zip(y, names):
            ax.text(xmax * 0.98, yi + 0.32, nm, ha="right", va="bottom", fontsize=7, color=MUTED)
    else:
        for yi, nm in zip(y, names):
            ax.text(0.02, yi + 0.32, nm, ha="left", va="bottom", fontsize=7, color=MUTED)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.set_xlim(0, xmax); ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.tick_params(axis="y", length=0)
fig.savefig("fig_radius_es.pdf" if ES else "fig_radius.pdf", bbox_inches="tight")
fig.savefig("fig_radius_es.png" if ES else "fig_radius.png", dpi=200, bbox_inches="tight")
print("ok")
