"""tools/verify_cap18.py — Yoshida–Kitaev P·F = 1/4, OTOC y curva de Page (Rényi-2 vs qg_S). python tools/verify_cap18.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qscramble import haar_unitary, yoshida_kitaev, otoc, renyi2_subsystem, qg_S_mean, X, Z

nB, nD = 3, 2          # A = 1 qubit (d_A = 2), B = 3 qubits, radiación D = 2 qubits ⇒ 10 qubits en total
dAB = 2 ** (1 + nB)

# 1. Identidad P_EPR · F_EPR = 1/d_A² = 1/4 para U Haar y para U no aleatoria; F vía 3 correladores qg de (R,R')
for name, U in (("Haar #1", haar_unitary(jax.random.PRNGKey(1), dAB)),
                ("Haar #2", haar_unitary(jax.random.PRNGKey(2), dAB)),
                ("I (sin scrambling)", jnp.eye(dAB, dtype=jnp.complex128))):
    P, F, F_qg = yoshida_kitaev(U, nB, nD)
    assert abs(P * F - 0.25) < 1e-12 and abs(F - F_qg) < 1e-12, (name, P, F)
    print(f"[1] U={name:19s}: P_EPR={P:.4f}  F_EPR={F:.4f}  P·F={P*F:.6f}  F(qg_XX,qg_YY,qg_ZZ)={F_qg:.4f} "
          f"→ disparos ×{1/P:.2f}, {1000*(2*F-1):+.0f} m-qg de fidelidad-eje")
# Haar: P ≈ 1/d_A² + 1/d_D² = 0.3125, F ≈ 0.8  (Yoshida–Kitaev 2017)
Ps = [yoshida_kitaev(haar_unitary(jax.random.PRNGKey(s), dAB), nB, nD)[0] for s in range(10, 30)]
print(f"    <P_EPR>_Haar (20 muestras) = {np.mean(Ps):.4f} ± {np.std(Ps)/np.sqrt(20):.4f}  (teoría 1/d_A² + 1/d_D² = {0.25+1/16:.4f})")
assert abs(np.mean(Ps) - 0.3125) < 0.03

# 2. OTOC como diagnóstico: W = Z en A, V = Z en B_3; Haar → ~1/d², identidad → 1
W = jnp.kron(Z, jnp.eye(2 ** nB, dtype=jnp.complex128))
V = jnp.kron(jnp.eye(2 ** nB, dtype=jnp.complex128), Z)
for name, U in (("Haar", haar_unitary(jax.random.PRNGKey(1), dAB)), ("I", jnp.eye(dAB, dtype=jnp.complex128))):
    print(f"[2] OTOC(U={name}) = {otoc(U, W, V):+.4f}")
assert abs(otoc(jnp.eye(dAB, dtype=jnp.complex128), W, V) - 1.0) < 1e-12

# 3. Curva de Page n=8: S2(k) sube y baja; qg_S(k) ≈ 1 plano (sólo base Z) → qg_S NO mide entrelazamiento (against)
n = 8
psi = haar_unitary(jax.random.PRNGKey(5), 2 ** n)[:, 0]
S2 = [renyi2_subsystem(psi, n, k) for k in range(1, n)]
QS = [qg_S_mean(psi, n, k) for k in range(1, n)]
print("[3] k      : " + " ".join(f"{k:5d}" for k in range(1, n)))
print("    S2(k)  : " + " ".join(f"{s:5.2f}" for s in S2))
print("    qg_S(k): " + " ".join(f"{s:5.2f}" for s in QS))
assert S2[3] > S2[0] and S2[3] > S2[-1] and max(QS) - min(QS) < 0.05
print("    S2 requiere 4^k−1 correladores (pureza); qg_S 1 base: plano → against como medida de entrelazamiento (THEORY_CORE §2.4)")
print("verify_cap18: OK")
