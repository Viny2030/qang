"""
Zero-noise extrapolation (ZNE): which space to extrapolate in, qg_Z or
theta, depends on the noise mechanism -- a previously undocumented,
practical connection between the qang unit and standard error mitigation
(Temme, Bravyi, & Gambetta, 2017, "Error Mitigation for Short-Depth
Quantum Circuits", Phys. Rev. Lett. 119, 180509).

Standard ZNE runs a circuit at several artificially amplified noise
levels (scale factors lambda = 1, 2, 3, ... via gate folding or pulse
stretching), measures an observable at each, and extrapolates back to
the noiseless limit lambda -> 0. It is usually done directly on the raw
expectation value -- which, for a single qubit's Z observable, IS qg_Z
by definition (qg_Z = <sigma_z>), so "ZNE on the expectation value" and
"ZNE in qg_Z-space" are the same thing.

The point demonstrated here is different: qg_Z's own inversion relation
(theta = arccos(qg_Z), Section 2.1 of the paper) gives a SECOND natural
space to extrapolate in, and which one is correct depends on where the
noise acts linearly:

  * Noise that shrinks the Bloch vector (depolarizing-like: qg_Z_noisy =
    qg_Z_ideal * (1 - gamma*lambda)) is linear in qg_Z-space. Extrapolating
    there recovers the noiseless value exactly (to machine precision);
    extrapolating in theta-space first (arccos, then a linear fit, then
    cos back) introduces a small but nonzero systematic bias, because
    arccos of a linear function is not itself linear.
  * Noise that is a coherent rotation-angle drift (miscalibration-like:
    theta_noisy = theta_ideal + delta*lambda) is linear in theta-space.
    Here the roles are exactly reversed: extrapolating in theta-space is
    exact, and extrapolating directly in qg_Z-space (cos of a linear
    function) is the one with the systematic bias.

Neither space is "more correct" in general -- the practical guideline is
to extrapolate in whichever space linearizes the actual noise mechanism
at hand, and qg_Z's built-in theta <-> qg_Z round trip (already needed
for Section 2.1's own inversion relation) is what makes checking both
this cheap.
"""

import numpy as np


def linear_extrapolate_to_zero(xs, ys) -> float:
    """Least-squares linear fit y = a + b*x; returns a (the value at x=0),
    i.e. the standard linear zero-noise extrapolation."""
    b, a = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)
    return float(a)


def bloch_vector_shrinkage_noise(qg0: float, gamma: float, lambdas):
    """Model A: depolarizing-like noise, linear shrinkage directly on qg_Z."""
    lambdas = np.asarray(lambdas, dtype=float)
    return qg0 * (1.0 - gamma * lambdas)


def coherent_rotation_drift_noise(theta0: float, delta: float, lambdas):
    """Model B: coherent over/under-rotation, linear drift directly on theta."""
    lambdas = np.asarray(lambdas, dtype=float)
    theta_noisy = theta0 + delta * lambdas
    return np.cos(theta_noisy)


def zne_in_qg_space(qg_noisy, lambdas) -> float:
    """Extrapolate qg_Z(lambda) -> qg_Z(0) directly."""
    return linear_extrapolate_to_zero(lambdas, qg_noisy)


def zne_in_theta_space(qg_noisy, lambdas) -> float:
    """Recover theta(lambda) = arccos(qg_Z(lambda)) at each scale factor,
    extrapolate theta -> theta(0), then convert back to qg_Z."""
    theta_noisy = np.arccos(qg_noisy)
    theta_extrap = linear_extrapolate_to_zero(lambdas, theta_noisy)
    return float(np.cos(theta_extrap))


if __name__ == "__main__":
    theta0 = np.pi / 3.0
    qg0 = np.cos(theta0)
    lambdas = np.array([1.0, 2.0, 3.0])

    print(f"True noiseless value: theta0={theta0:.6f}, qg0={qg0:.6f}")
    print()

    print("Model A -- Bloch-vector shrinkage (depolarizing-like), linear in qg_Z:")
    qg_noisy_A = bloch_vector_shrinkage_noise(qg0, gamma=0.08, lambdas=lambdas)
    qg_via_qgspace_A = zne_in_qg_space(qg_noisy_A, lambdas)
    qg_via_thetaspace_A = zne_in_theta_space(qg_noisy_A, lambdas)
    print(f"  ZNE in qg_Z-space (correct):  {qg_via_qgspace_A:.6f}  error={abs(qg_via_qgspace_A - qg0):.2e}")
    print(f"  ZNE in theta-space (wrong):   {qg_via_thetaspace_A:.6f}  error={abs(qg_via_thetaspace_A - qg0):.2e}")
    print()

    print("Model B -- coherent rotation drift (miscalibration-like), linear in theta:")
    qg_noisy_B = coherent_rotation_drift_noise(theta0, delta=0.05, lambdas=lambdas)
    qg_via_thetaspace_B = zne_in_theta_space(qg_noisy_B, lambdas)
    qg_via_qgspace_B = zne_in_qg_space(qg_noisy_B, lambdas)
    print(f"  ZNE in theta-space (correct): {qg_via_thetaspace_B:.6f}  error={abs(qg_via_thetaspace_B - qg0):.2e}")
    print(f"  ZNE in qg_Z-space (wrong):    {qg_via_qgspace_B:.6f}  error={abs(qg_via_qgspace_B - qg0):.2e}")
    print()

    print("Guideline: extrapolate in whichever space linearizes the actual")
    print("noise mechanism. qg_Z's own theta <-> qg_Z inversion (Section 2.1)")
    print("makes checking both directions essentially free.")
