"""tools/verify_exp40_2.py — Exp. 40.2: ajuste por disparos de la Zitterbewegung del qubit β (sin Qiskit)."""
from __future__ import annotations

import math

import numpy as np

from qang.probability import analyze, shot_noise_milliqang
from qang.qanti import (
    bloch_mqg,
    conjugate,
    eigenstate,
    gamma_of,
    self_conjugate_overlap,
    shots_to_certify_branch,
    singlet_readings,
    zb_analytic_mqg,
    zb_qgz_mqg,
)

M, P = 1.0, 0.75
N = 2000
SEED = 170


def _fit(t: np.ndarray, y: np.ndarray, sigma: np.ndarray, w0: float) -> tuple[float, float, float, float]:
    sw = np.maximum(sigma, 1e-9)
    best = None
    for w in np.linspace(0.95 * w0, 1.05 * w0, 1001):
        x = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
        coef, *_ = np.linalg.lstsq(x / sw[:, None], y / sw, rcond=None)
        chi2 = float(np.sum(((x @ coef - y) / sw) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, w, coef)
    chi2, w, (a, b, c) = best
    return w, a, math.hypot(b, c), chi2 / (len(t) - 4)


def main() -> None:
    print("Exp. 40.2 — Zitterbewegung: qg_Z(t)=1000/γ²+1000β² cos2Et, ajuste Z1–Z5")

    e = math.hypot(M, P)
    gamma = gamma_of(P, M)
    beta = P / e
    w0 = 2.0 * e
    period = math.pi / e
    ts = np.linspace(0.0, 2.0 * period, 33)
    err = max(abs(zb_qgz_mqg(P, M, float(t)) - zb_analytic_mqg(P, M, float(t))) for t in ts)
    assert err < 1e-9
    assert abs(gamma - 1.25) < 1e-12 and abs(beta - 0.6) < 1e-12
    print(f"  E={e:.2f}, γ={gamma:.2f}, β={beta:.2f}, w=2E={w0:.2f}; max error analítico={err:.2e}")

    psi = eigenstate(P, M, +1)
    assert np.allclose(bloch_mqg(conjugate(psi)), -bloch_mqg(psi), atol=1e-9)
    assert abs(abs(np.vdot(conjugate(psi), eigenstate(P, M, -1))) - 1.0) < 1e-12
    rng = np.random.default_rng(SEED)
    mx = 0.0
    for _ in range(2000):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        mx = max(mx, self_conjugate_overlap(v))
    assert mx < 1e-15
    print(f"  conjugación: bloch(Cψ)=-bloch(ψ); |<Cψ|E->|=1; max |<ψ|Cψ>|={mx:.1e}")

    sigmas, readings = [], []
    shot_rng = np.random.default_rng(SEED + 1)
    for t in ts:
        q = zb_analytic_mqg(P, M, float(t)) / 1000.0
        qc = max(-1.0, min(1.0, q))
        theta = math.acos(qc)
        s = shot_noise_milliqang(theta, N)
        k0 = int(shot_rng.binomial(N, (1.0 + qc) / 2.0))
        r = analyze(k0, N, target_qg_z=qc, z=1.0)
        target_m = zb_analytic_mqg(P, M, float(t))
        assert abs(r.milliqang[0] - target_m) < max(5.0 * s, 1e-9)
        sigmas.append(s)
        readings.append(r.milliqang[0])
    w, mean, amp, dof = _fit(ts, np.asarray(readings), np.asarray(sigmas), w0)
    assert abs(w - 2.5) < 0.125 and abs(mean - 640.0) < 60.0
    assert abs(amp - 360.0) < 60.0 and dof < 2.0
    print(f"  ajuste: w={w:.4f} (2.5±5 %), media={mean:.2f} (640±60), amplitud={amp:.2f} (360±60), χ²/dof={dof:.3f}")

    assert abs(shots_to_certify_branch(2.0) - 27.0) < 1e-12
    s99 = shots_to_certify_branch(1.0 / math.sqrt(1.0 - 0.99**2))
    assert 440.0 < s99 < 447.0
    sing, ev = singlet_readings()
    assert all(abs(v) < 1e-12 for v in sing[:2]) and abs(ev + 1000.0) < 1e-12
    print(f"  ramas: N(γ=2)=27, N(β=0.99)={s99:.2f}; singlete=({sing[0]:.0f},{sing[1]:.0f},{sing[2]:.0f}), {ev:.0f} m-qg")
    print("verify_exp40_2: OK")


if __name__ == "__main__":
    main()
