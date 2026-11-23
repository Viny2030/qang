"""tools/verify_exp38_1.py — Exp. 38.1: signo de intercambio Jordan–Wigner bajo ruido de disparos."""
from __future__ import annotations

import math

import numpy as np

from qang.probability import analyze, p0_from_milliqang, shot_noise_milliqang
from qang.qtransv import anticommutation_error, exchange_qgx

N = 20_000
SEED = 11


def main() -> None:
    print("Exp. 38.1 — Jordan–Wigner: qg_X=-1000 m-qg (fermión) frente a +1000 m-qg (SWAP)")

    err = anticommutation_error(4)
    fermionic, swap = exchange_qgx(True), exchange_qgx(False)
    assert err < 1e-15
    assert abs(fermionic + 1.0) < 1e-15 and abs(swap - 1.0) < 1e-15
    print(f"  error de anticomutación (4 modos)={err:.1e}; qg_X fermión={fermionic:+.1f}, SWAP={swap:+.1f}")

    sigma = shot_noise_milliqang(math.pi / 2.0, N)
    assert sigma < 7.08
    for label, x in (("fermión", fermionic), ("SWAP", swap)):
        target = 1000.0 * x
        k0 = int(np.random.default_rng(SEED).binomial(N, p0_from_milliqang(target)))
        reading = analyze(k0, N, target_qg_z=x)
        assert abs(reading.milliqang[0] - target) < 8.0 * sigma
        print(f"  {label:>7}: k0={k0:5d}/{N}  lectura={reading.milliqang[0]:+.2f} m-qg "
              f"(ideal {target:+.0f}, σ={sigma:.2f})")

    assert 1000.0 * (swap - fermionic) == 2000.0
    print(f"  separación fermión↔SWAP = {1000.0 * (swap - fermionic):.0f} m-qg; σ={sigma:.2f} m-qg")
    print("verify_exp38_1: OK")


if __name__ == "__main__":
    main()
