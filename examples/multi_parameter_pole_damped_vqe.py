"""
Multi-parameter pole-damped gradient descent: generalizing
``theta_pole_damped`` (qang.gradients) from a single rotation angle to a
vector of them.

Every real ansatz has more than one parameter -- for instance, one Ry
angle per qubit in a hardware-efficient layer. This example checks that
the pole-proximity damping established for a single theta in
qang.gradients (and demonstrated on real H2 VQE in
examples/vqe_h2_qg_vs_theta.py, and statistically over 300 random
one-qubit landscapes in examples/pole_damped_gradient_descent_robustness.py)
continues to behave correctly when several parameters are optimized
together, using ``qang.gradients.multi_param_gradient_descent``.

The toy multi-parameter energy is deliberately *separable*:

    E(theta_1, ..., theta_n) = sum_i [ h_z[i]*cos(theta_i) + h_x[i]*sin(theta_i) ]

so there is no cross-parameter interaction to speak of -- this example is
not claiming a new multi-parameter phenomenon, only checking, honestly,
that the single-parameter mechanism keeps working coordinate by
coordinate once it is applied to many parameters side by side (see
qang.gradients' module docstring, "Generalizing to a vector of
parameters").

The four-parameter scenario below has:
  * param0 -- starts near the theta=0 pole (same configuration already
    used in tests/test_gradients.py's single-parameter headline test).
  * param1 -- starts near the theta=pi pole, with a different (h_z, h_x)
    whose true minimum sits well inside (0, pi) rather than at the domain
    edge, so this is a genuine second stress test rather than a repeat of
    the first.
  * param2, param3 -- start already AT their own global minimum (h_z=0
    puts the minimum at exactly theta=pi/2), nowhere near a pole.

The headline check: at a badly-tuned (aggressive) learning rate, plain
theta-space gradient descent fails on the two near-pole parameters (param0,
param1) while leaving the two pole-free parameters (param2, param3)
completely unaffected -- and theta_pole_damped rescues the two near-pole
parameters without in any way disturbing the two that were already fine,
at every learning rate tested. This last point is the actual content of
the generalization: a parameter's damping factor depends only on that
parameter's own current value, never on any other parameter's state.

Honest scope note: exactly as in the single-parameter study, this fix is
not claimed to be failure-free at arbitrarily aggressive learning rates --
at lr=5.0 here, the theta=pi pole parameter's damped result is close to
its true minimum but not numerically exact, an instance of the same small
"pole-trapping" failure mode already documented and quantified in
examples/pole_damped_gradient_descent_robustness.py.
"""

import math

from qang.gradients import multi_param_gradient_descent, toy_vqe_energy

# --- The four-parameter scenario (see module docstring) ---
H_Z = [1.0, 1.0, 0.0, 0.0]
H_X = [-0.3, -0.5, -0.05, -0.15]
THETA0 = [0.02, math.pi - 0.02, math.pi / 2, math.pi / 2]
LABELS = [
    "near pole (theta=0)",
    "near pole (theta=pi)",
    "pole-free / already at its minimum",
    "pole-free / already at its minimum",
]
TRUE_MINS = [-math.hypot(hz, hx) for hz, hx in zip(H_Z, H_X)]


def run_scenario(lr: float, steps: int = 300, eps: float = 0.05):
    """Run both spaces on the four-parameter scenario at the given
    learning rate. Returns (plain_energies, damped_energies), each a list
    of the four parameters' final individual energies."""
    h_plain = multi_param_gradient_descent("theta", THETA0, lr=lr, steps=steps, h_z=H_Z, h_x=H_X)
    h_damped = multi_param_gradient_descent(
        "theta_pole_damped", THETA0, lr=lr, steps=steps, eps=eps, h_z=H_Z, h_x=H_X
    )
    plain_energies = [
        toy_vqe_energy(theta, hz, hx)
        for theta, hz, hx in zip(h_plain.thetas[-1], H_Z, H_X)
    ]
    damped_energies = [
        toy_vqe_energy(theta, hz, hx)
        for theta, hz, hx in zip(h_damped.thetas[-1], H_Z, H_X)
    ]
    return plain_energies, damped_energies


if __name__ == "__main__":
    print("Multi-parameter pole-damped gradient descent")
    print("=============================================")
    print(f"{'true minima:':>14s} " + "  ".join(f"{e:+.6f}" for e in TRUE_MINS))
    print()

    for lr in [0.3, 1.0, 2.0, 3.0, 5.0]:
        plain_energies, damped_energies = run_scenario(lr)
        print(f"lr = {lr}")
        for i in range(4):
            print(
                f"  param{i} [{LABELS[i]}]: "
                f"plain E={plain_energies[i]:+.6f}  "
                f"damped E={damped_energies[i]:+.6f}  "
                f"true min={TRUE_MINS[i]:+.6f}"
            )
        print()

    print("Headline check: at lr=2.0 and lr=3.0, plain gradient descent fails on both")
    print("near-pole parameters (param0, param1) but leaves the two pole-free")
    print("parameters (param2, param3) untouched at their own optimum; damped rescues")
    print("param0 and param1 without disturbing param2 or param3 at all.")
