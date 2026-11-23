"""tools/verify_exp39_1.py — Exp. 39.1: filtro de peso 1 sobre W bajo T1, con lectura por disparos (sin SDK)."""
from __future__ import annotations

import math

import numpy as np

from qang.interop import qg_z_profile, weight_filter
from qang.qintegr import w_state_t1_readings
from qang.qlimits import damp_distribution

T = 0.6
N = 20_000
SEEDS = (11, 12, 13)


def main() -> None:
    print("Exp. 39.1 — W de 3 qubits a t/T1=0.6: crudo +634.1, filtrado +333.333 m-qg")

    r = w_state_t1_readings(T)
    gamma = 1.0 - math.exp(-T)
    p = np.zeros(2**3)
    for q in range(3):
        p[1 << (2 - q)] = 1.0 / 3.0
    pg = damp_distribution(p, 3, gamma)

    assert abs(r["raw_mqg"] - 634.1) < 0.05
    assert abs(r["filtered_mqg"] - 1000.0 / 3.0) < 1e-9
    assert abs(r["kept"] - math.exp(-T)) < 1e-12
    assert abs(r["filtered_mqg"] - r["ideal_mqg"]) < 1e-12
    assert abs(1.0 / r["kept"] - 1.8221188003905087) < 1e-12
    print(f"  exacto: raw={r['raw_mqg']:.4f}, filtrada={r['filtered_mqg']:.9f}, "
          f"kept={r['kept']:.9f}, overhead={1.0 / r['kept']:.9f}")

    for seed in SEEDS:
        z = np.random.default_rng(seed).choice(8, size=N, p=pg)
        counts = {
            tuple(int(b) for b in f"{i:03b}"): int(c)
            for i, c in zip(*np.unique(z, return_counts=True))
        }
        prof = qg_z_profile(counts)
        kept, frac = weight_filter(counts, 1)
        fp = qg_z_profile(kept)
        raw_m, raw_se = 1000.0 * prof[1][0], 1000.0 * prof[1][1]
        fil_m, fil_se = 1000.0 * fp[1][0], 1000.0 * fp[1][1]
        assert abs(raw_m - r["raw_mqg"]) < 5.0 * raw_se
        assert abs(fil_m - r["ideal_mqg"]) < 5.0 * fil_se
        assert abs(frac - r["kept"]) < 5.0 * math.sqrt(frac * (1.0 - frac) / N)
        print(f"  semilla {seed}: raw={raw_m:+.1f}±{raw_se:.2f}, filt={fil_m:+.3f}±{fil_se:.3f}, "
              f"kept={frac:.5f} (k={sum(kept.values())})")
    print("verify_exp39_1: OK")


if __name__ == "__main__":
    main()
