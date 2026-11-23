"""tools/verify_exp12_13_1.py — Exp. 12/13.1: fidelidad de teleportación desde tres correladores y filtro de doble carril."""
from __future__ import annotations

import math

import numpy as np

from qang.qdualrail import (
    apply_loss,
    dual_rail_state,
    readout_sector_filtered,
    readout_unfiltered,
    shots_multiplier_two_photons,
    transmission,
)
from qang.qpon import fidelity_from_correlators
from qang.qtele import shots_for_fidelity, teleport_fidelity_werner_qzz

N = 20_000


def main() -> None:
    print("Exp. 12/13.1 — F=(3+qg_XX−qg_YY+qg_ZZ)/6, sin circuito de teleportación")

    assert fidelity_from_correlators(1.0, -1.0, 1.0) == 1.0
    assert abs(fidelity_from_correlators(1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0) - 2.0 / 3.0) < 1e-15
    assert abs(teleport_fidelity_werner_qzz(1.0 / 3.0) - 2.0 / 3.0) < 1e-15
    assert abs(float(shots_for_fidelity(0.9, 0.01)) - 475.0) < 1e-9
    print("  álgebra exacta: F=1 (Bell), F=2/3 en qg_ZZ=1/3; N(F=0.9, σ=0.01)=475")

    werner = fidelity_from_correlators(0.9, -0.9, 0.9)
    non_werner = fidelity_from_correlators(0.8, -0.8, 1.0)
    one_basis = teleport_fidelity_werner_qzz(1.0)
    assert abs(werner - 0.95) < 1e-12
    assert abs(non_werner - 0.9333333333333332) < 1e-12
    assert one_basis >= non_werner and abs(one_basis - non_werner - 1.0 / 15.0) < 1e-12
    print(f"  Werner V=0.9: F={werner:.6f}; no Werner: F={non_werner:.6f} vs 1 base={one_basis:.6f}")

    rng = np.random.default_rng(0)
    for V in (0.9, 1.0 / 3.0):
        vals, ses = [], []
        for q in (V, -V, V):
            qc = max(-1.0, min(1.0, q))
            k = int(rng.binomial(N, (1.0 + qc) / 2.0))
            vals.append(2.0 * k / N - 1.0)
            ses.append(math.sqrt(max(0.0, 1.0 - qc * qc) / N))
        f_hat = fidelity_from_correlators(*vals)
        f_true = (1.0 + V) / 2.0
        sigma_f = math.sqrt(sum(s * s for s in ses)) / 6.0
        assert abs(f_hat - f_true) < 5.0 * sigma_f
        print(f"  V={V:.4f}: F̂={f_hat:.6f} vs {(1.0 + V) / 2.0:.6f} (±{sigma_f:.6f})")

    eta = float(transmission(50.0))
    assert abs(eta - 0.1) < 1e-14
    rho = apply_loss(dual_rail_state(-0.6, qx=0.3), eta)
    raw = float(readout_unfiltered(rho))
    filt, acc = readout_sector_filtered(rho)
    assert abs(raw - (0.1 * -0.6 + 0.9)) < 1e-14
    assert abs(float(filt) + 0.6) < 1e-13 and abs(float(acc) - 0.1) < 1e-14
    assert abs(shots_multiplier_two_photons(100.0) - 100.0) < 1e-9
    print(f"  doble carril 50 km: η={eta:.2f}, raw={raw:+.2f}, filtrado={float(filt):+.2f} (tasa {float(acc):.2f})")
    print("verify_exp12_13_1: OK")


if __name__ == "__main__":
    main()
