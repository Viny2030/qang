"""tools/verify_exp33_42_1.py — Exp. 33/42.1: conteo de cuerdas, nivel leído y T-dualidad sin qiskit."""
from __future__ import annotations

import numpy as np

from qang.qstring import (
    closed_massless_decomposition,
    colored_partitions,
    critical_dimension,
    dual_exponents,
    level_filter_damped,
    level_from_readings,
    level_from_z,
    level_shots,
    level_variance,
    level_vs_weight_blindness,
    massless_breakdown,
    massless_states,
    mode_readings_mqg,
    open_mass2_alpha,
    t_dual_equal,
    zero_point_energy,
)

R_S = (0.5, 0.7, 1.0, 1.3, 2.0, 3.0)


def main() -> None:
    print("Exp. 33/42.1 — Análogo de cuerda: d_N, D=26, nivel por Z y T-dualidad")

    d = colored_partitions(5, 24)
    dec = closed_massless_decomposition(24)
    assert d == [1, 24, 324, 3200, 25650, 176256]
    assert critical_dimension() == 26
    assert [D for D in range(3, 60) if zero_point_energy(D) == 1] == [26]
    assert open_mass2_alpha(0) == -1 and open_mass2_alpha(1) == 0
    assert dec == {"total": 576, "graviton": 299, "kalb_ramond": 276, "dilaton": 1}
    assert dec["graviton"] + dec["kalb_ramond"] + dec["dilaton"] == d[1] ** 2 == 576
    assert [int(np.ceil(np.log2(x))) for x in d[1:]] == [5, 9, 12, 15, 18]
    print(f"  d_N={d}; D=26; 576={dec['graviton']}+{dec['kalb_ramond']}+{dec['dilaton']}")

    rng = np.random.default_rng(42)
    p = rng.dirichlet(np.ones(64))
    assert abs(level_from_readings(mode_readings_mqg(p, 6)) - level_from_z(p, 6)) < 1e-12
    pu = np.full(64, 1.0 / 64.0)
    var = level_variance(pu, 6)
    assert abs(var - 22.75) < 1e-12
    assert level_shots(pu, 6, 0.1) == 8740
    flt = level_filter_damped(0b101101, 6, 0.3)
    assert abs(flt["kept"] - 0.7**4) < 1e-12 and flt["level_target"] == 14
    blind = level_vs_weight_blindness(4)
    assert blind["level_preserving"] == (3, 3, 1, 2)
    assert blind["weight_preserving"] == (1, 2, 1, 1)
    print(f"  nivel lineal OK; Var={var:.2f}; N(σ=0.1)=8740; retención T1={flt['kept']:.4f}")

    assert all(t_dual_equal(R) for R in R_S)
    counts = {R: massless_states(R) for R in R_S}
    assert counts == {0.5: 580, 0.7: 576, 1.0: 676, 1.3: 576, 2.0: 580, 3.0: 578}
    assert counts[1.0] - 576 == 4 * 24 + 4 == 100
    bd = massless_breakdown(1.0)
    assert sum(o[4] for o in bd if o[2] + o[3] == 1) == 96
    assert sum(o[4] for o in bd if o[2] + o[3] == 0) == 4
    print(f"  T-dualidad en {len(R_S)} radios: {counts}; autodual 676=576+96+4")

    e1, e2 = dual_exponents(1.5)
    assert abs(e1 * e2 - 1.0) < 1e-12
    print(f"  exponentes duales K=3/2: {e1:.6f} × {e2:.6f} = {e1 * e2:.12f}")
    print("verify_exp33_42_1: OK")


if __name__ == "__main__":
    main()
