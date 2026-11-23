"""tools/verify_exp38_2.py — Exp. 38.2: eco de Hahn bajo deriva de desintonía, a nivel de disparos."""
from __future__ import annotations

import math

import numpy as np

from qang.probability import analyze, p0_from_milliqang, shot_noise_milliqang
from qang.qtransv import free_decay_qgx, hahn_echo_ensemble, hahn_echo_qgx_drift

N = 20_000
ST = 2.0 * math.pi * 50_000.0 * 20e-6


def main() -> None:
    print("Exp. 38.2 — Hahn: eco estático 1.000000 frente a deriva 10 % ≈0.8209")

    free = free_decay_qgx(ST)
    static = hahn_echo_ensemble(ST, 0.0)
    pulse_err = hahn_echo_ensemble(ST, 0.02)
    drift = hahn_echo_qgx_drift(ST, 0.1)
    assert free < 1e-8
    assert abs(static - 1.0) < 1e-12
    assert abs(pulse_err - 0.99901) < 1e-5
    assert abs(drift - 0.8209) < 1e-4
    print(f"  σ·τ={ST:.6f}: decaimiento libre={free:.3e}; eco estático={static:.6f}")
    print(f"  pulso +2 %: {pulse_err:.5f} ({(1.0 - pulse_err) * 1000:.2f} m-qg); deriva 10 %: {drift:.4f}")

    delta_m = 1000.0 * (static - drift)
    sigma = shot_noise_milliqang(math.pi / 2.0, N)
    assert abs(delta_m - 179.1) < 0.5
    assert delta_m > 25.0 * sigma

    readings = []
    for label, q, seed in (("estático", static, 12), ("deriva 10 %", drift, 13)):
        q_clamped = max(-1.0, min(1.0, q))
        k0 = int(np.random.default_rng(seed).binomial(N, p0_from_milliqang(1000.0 * q_clamped)))
        r = analyze(k0, N, target_qg_z=q_clamped)
        readings.append(r.milliqang[0])
        print(f"  {label:>11}: k0={k0:5d}/{N}  lectura={r.milliqang[0]:+.2f} m-qg")

    assert abs(readings[0] - readings[1] - delta_m) < 8.0 * sigma
    print(f"  separación muestreada={readings[0] - readings[1]:.2f} m-qg (ideal {delta_m:.2f}, σ={sigma:.2f})")
    print("verify_exp38_2: OK")


if __name__ == "__main__":
    main()
