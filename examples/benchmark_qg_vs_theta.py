"""
Empirical benchmark for Future Research Direction #3: compares optimizer
convergence in theta-space vs raw qg-space vs two regularized qg-space
variants, all started AT a pole (theta0 = 0.01 rad, i.e. qg_Z ~ +0.99995),
on the toy VQE loss E(theta) = -sin(theta) (minimum at theta = pi/2).

Run:  python examples/benchmark_qg_vs_theta.py
Produces: benchmark_qg_vs_theta.png and a short console summary table.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quang.gradients import benchmark, toy_vqe_grad_theta

# --- dataviz-skill validated categorical palette (light mode, slots 1-4) ---
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, SEC_INK, MUTED, GRID, BASELINE, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)

LABELS = {
    "theta": "θ-space (baseline)",
    "qg_raw": "raw qg-space (paper, Sec. 4.1)",
    "qg_clipped": "qg-space, clipped regularization",
    "qg_tikhonov": "qg-space, Tikhonov regularization",
}
COLORS = {"theta": BLUE, "qg_raw": ORANGE, "qg_clipped": AQUA, "qg_tikhonov": YELLOW}

THETA0 = 0.01   # rad -- deliberately AT a pole (qg_Z ~ 0.99995)
LR = 0.05
STEPS = 150
EPS = 0.05


def main():
    results = {
        "theta": None,  # filled below with a matching lr for a fair baseline
    }
    from quang.gradients import run_gradient_descent

    results["theta"] = run_gradient_descent("theta", THETA0, lr=LR, steps=STEPS)
    results["qg_raw"] = run_gradient_descent("qg_raw", THETA0, lr=LR, steps=STEPS)
    # regularized variants use a smaller lr, sized to their own bound (~lr/eps)
    # so the comparison is "each method at a learning rate appropriate to it",
    # not "every method at one lr that only suits theta-space".
    results["qg_clipped"] = run_gradient_descent("qg_clipped", THETA0, lr=LR / 5, steps=STEPS, eps=EPS)
    results["qg_tikhonov"] = run_gradient_descent("qg_tikhonov", THETA0, lr=LR / 5, steps=STEPS, eps=EPS)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=200)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)

    ax_e, ax_t = axes

    for space in ["theta", "qg_raw", "qg_clipped", "qg_tikhonov"]:
        hist = results[space]
        it = list(range(len(hist.energy)))
        ax_e.plot(it, hist.energy, color=COLORS[space], linewidth=2, label=LABELS[space])
        ax_t.plot(it, hist.theta, color=COLORS[space], linewidth=2, label=LABELS[space])

    ax_e.axhline(-1.0, color=BASELINE, linestyle=":", linewidth=1)
    ax_e.text(STEPS * 0.98, -0.97, "global min E = -1", ha="right", fontsize=8, color=MUTED)
    ax_e.set_xlabel("iteration", fontsize=9.5, color=SEC_INK)
    ax_e.set_ylabel("E(θ)", fontsize=9.5, color=SEC_INK)
    ax_e.set_title("Energy vs. iteration", fontsize=11.5, color=INK, loc="left", fontweight="bold")

    ax_t.axhline(math.pi / 2, color=BASELINE, linestyle=":", linewidth=1)
    ax_t.text(STEPS * 0.98, math.pi / 2 + 0.08, "θ* = π/2", ha="right", fontsize=8, color=MUTED)
    ax_t.set_xlabel("iteration", fontsize=9.5, color=SEC_INK)
    ax_t.set_ylabel("θ (rad)", fontsize=9.5, color=SEC_INK)
    ax_t.set_ylim(-0.1, math.pi + 0.1)
    ax_t.set_title("θ trajectory", fontsize=11.5, color=INK, loc="left", fontweight="bold")

    for ax in axes:
        ax.grid(alpha=0.35, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=8.5)

    ax_e.legend(loc="upper right", frameon=False, fontsize=8.3, labelcolor=SEC_INK)

    fig.suptitle(
        "Optimizer convergence started AT a pole (θ0 = 0.01 rad, qg_Z ≈ +0.9999)",
        fontsize=12.5, color=INK, x=0.02, ha="left", fontweight="bold", y=1.03,
    )
    fig.text(
        0.01, -0.04,
        "Toy loss E(θ) = -sin(θ), minimum at θ=π/2. theta-space and qg_clipped/qg_tikhonov use lr sized to their own step-size "
        "bound; qg_raw uses the same lr as theta-space to show the paper's Section 4.1 divergence directly.",
        fontsize=7.6, color=MUTED,
    )

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_qg_vs_theta.png")
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")

    print(f"\n{'space':16s} {'final theta':>12s} {'final E':>10s} {'|grad| final':>13s} {'max |step|':>11s} {'diverged':>9s}")
    for space in ["theta", "qg_raw", "qg_clipped", "qg_tikhonov"]:
        hist = results[space]
        steps_sizes = [abs(hist.theta[i + 1] - hist.theta[i]) for i in range(len(hist.theta) - 1)]
        max_step = max(steps_sizes) if steps_sizes else 0.0
        final_theta = hist.theta[-1]
        final_e = hist.energy[-1]
        final_grad = abs(toy_vqe_grad_theta(final_theta))
        print(f"{space:16s} {final_theta:12.4f} {final_e:10.4f} {final_grad:13.5f} {max_step:11.4f} {str(hist.diverged):>9s}")


if __name__ == "__main__":
    main()
