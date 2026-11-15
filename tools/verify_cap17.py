"""tools/verify_cap17.py — P-CTC vs D-CTC como mapas leídos en m-qg. python tools/verify_cap17.py"""
import numpy as np
import jax.numpy as jnp
from qang.qctc import (pctc_operator, pctc_map, dctc_map, bloch, state_from_bloch, shots_multiplier_pctc,
                       CNOT, SWAP, I2, X)

# 1. U = I: ambos mapas son la identidad (sanidad)
r = jnp.array([0.3, -0.4, 0.5])
rho = state_from_bloch(r)
U_I = jnp.eye(4, dtype=jnp.complex128)
out_p, p = pctc_map(rho, U_I); out_d, _, it = dctc_map(rho, U_I)
assert np.allclose(np.asarray(bloch(out_p)), np.asarray(r), atol=1e-14) and np.allclose(np.asarray(bloch(out_d)), np.asarray(r), atol=1e-14)
assert abs(float(p) - 1.0) < 1e-14
print(f"[1] U=I: P-CTC y D-CTC = identidad; p_post = 1")

# 2. U = CNOT (CR controla CTC): P-CTC proyecta CR a |0> (no lineal), D-CTC decohere CR (lineal en qg_Z)
C = pctc_operator(CNOT)
assert np.allclose(np.asarray(C), np.asarray(2 * jnp.diag(jnp.array([1.0, 0.0])))), np.asarray(C)
for qz in (0.9, 0.0, -0.6, -0.98):
    rho = state_from_bloch(jnp.array([0.4 * np.sqrt(1 - qz ** 2), 0.0, qz]))
    out_p, p = pctc_map(rho, CNOT)
    out_d, rho_ctc, it = dctc_map(rho, CNOT)
    bp, bd = bloch(out_p), bloch(out_d)
    assert abs(float(bp[2]) - 1.0) < 1e-13 and abs(float(p) - (1 + qz) / 2) < 1e-14
    assert abs(float(bd[2]) - qz) < 1e-13 and abs(float(bd[0])) < 1e-13
    print(f"[2] entrada {1000*qz:+5.0f} m-qg (qg_X={float(bloch(rho)[0]):.3f}): P-CTC → {1000*float(bp[2]):+.0f} m-qg, "
          f"p_post={float(p):.3f} (disparos ×{shots_multiplier_pctc(float(p)):.1f}) | "
          f"D-CTC → {1000*float(bd[2]):+.0f} m-qg, qg_X→{float(bd[0]):.0e}, punto fijo en {it} it.")

# 3. U = SWAP (paradoja del abuelo cronológica): P-CTC C = Tr_CTC(SWAP) = I → identidad, p_post = 1/4 (uniforme);
#    D-CTC punto fijo ρ_CTC = ρ, salida = ρ
rho = state_from_bloch(r)
out_p, p = pctc_map(rho, SWAP); out_d, rho_ctc, it = dctc_map(rho, SWAP)
assert np.allclose(np.asarray(bloch(out_p)), np.asarray(r), atol=1e-14) and abs(float(p) - 0.25) < 1e-14
assert np.allclose(np.asarray(bloch(out_d)), np.asarray(r), atol=1e-13)
print(f"[3] U=SWAP: P-CTC identidad con p_post=0.25 (×4 disparos) | D-CTC identidad, ρ_CTC=ρ en {it} it.")

# 4. U = (X⊗I)·SWAP (abuelo con negación): P-CTC C = Tr_CTC = X·... → ver si C es singular
U = jnp.kron(X, I2) @ SWAP
C = pctc_operator(U)
print(f"[4] U=(X⊗I)SWAP: C = Tr_CTC(U) =\n{np.asarray(C).real}  → mapa P-CTC: ρ → XρX (negación consistente), p_post=0.25")
out_p, p = pctc_map(rho, U)
assert np.allclose(np.asarray(bloch(out_p)), np.asarray([r[0], -r[1], -r[2]]), atol=1e-14)
print("    recursos: P-CTC = 1 par de Bell (2 qubits extra) + postselección ×1/p_post; D-CTC = iteración no lineal (sin análogo de circuito único)")
print("    estatus: identidades de los mapas exact (simulación); CTC físicas unverified")
print("verify_cap17: OK")
