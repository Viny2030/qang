"""Figure of the IBM paper: qg_Z error raw vs filtered (+ readout) on 20 IBM device noise models (§137, §144)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "examples", "data")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
RAW, FILT = "#eb6834", "#2a78d6"


def rows():
    out = []
    for r in json.load(open(os.path.join(DATA, "ibm_runtime_qg.json")))["rows"]:
        out.append((r["backend"].replace("Fake", ""), r["error"]["raw"], r["error"]["filtered_readout"], "§137"))
    for r in json.load(open(os.path.join(DATA, "ibm_fleet_qg.json")))["rows"]:
        if r.get("error"):
            out.append((r["backend"].replace("Fake", ""), r["error"]["raw"], r["error"]["filtered_readout"], "§144"))
    return sorted(out, key=lambda x: x[1])


def main():
    rs = rows()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for i, (name, raw, filt, sec) in enumerate(rs):
        ax.plot([filt, raw], [i, i], color=GRID, lw=2.2, zorder=1)
        ax.scatter(raw, i, s=34, color=RAW, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter(filt, i, s=34, color=FILT, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([f"{n}{'*' if s == '§137' else ''}" for n, _, _, s in rs], color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("mean |qg_Z - ideal| (log scale)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED)
    ax.scatter([], [], s=34, color=RAW, label="raw")
    ax.scatter([], [], s=34, color=FILT, label="qg filter + readout correction")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_fleet.pdf"))
    fig.savefig(os.path.join(HERE, "fig_fleet.png"), dpi=160)


if __name__ == "__main__":
    main()
