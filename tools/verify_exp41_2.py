"""tools/verify_exp41_2.py — Exp. 41.2: paradoja del abuelo, P-CTC sin éxito y punto fijo D-CTC."""
from __future__ import annotations

import math

import numpy as np

from qang.qtimeloop import I2, SX, bloch_mqg, deutsch_iterate, pctc_success

U = np.kron(I2, SX)
N_ATTEMPTS = 100_000


def _state(theta: float) -> np.ndarray:
    psi = np.array([math.cos(theta / 2.0), math.sin(theta / 2.0)], dtype=complex)
    return np.outer(psi, psi.conj())


def main() -> None:
    print("Exp. 41.2 — Abuelo: P-CTC p_post=0; D-CTC fija (0,0,0) m-qg y deja el CR intacto")

    cases = (("|+>", 0.5 * math.pi), ("RY(0.7)|0>", 0.7), ("|0>", 0.0))
    for name, theta in cases:
        rho = _state(theta)
        p = pctc_success(U, rho)
        c, out = deutsch_iterate(U, rho)
        bc, bo, br = bloch_mqg(c), bloch_mqg(out), bloch_mqg(rho)
        assert p < 1e-15
        assert np.max(np.abs(bc)) < 1e-12
        assert np.max(np.abs(bo - br)) < 1e-9
        accepted = int(np.random.default_rng(7).binomial(N_ATTEMPTS, p))
        assert accepted == 0
        print(f"  {name:>12}: p_post={p:.1e}  CTC={np.round(bc, 12)}  CR_in={np.round(br, 6)}  CR_out={np.round(bo, 6)}")

    rho_plus = _state(0.5 * math.pi)
    c, _ = deutsch_iterate(U, rho_plus, iters=1)
    assert np.max(np.abs(bloch_mqg(c))) < 1e-12
    print(f"  convergencia D-CTC: 1 iteración ya da el punto fijo; {N_ATTEMPTS} intentos P-CTC → 0 aceptados")
    print("verify_exp41_2: OK")


if __name__ == "__main__":
    main()
