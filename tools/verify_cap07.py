"""tools/verify_cap07.py — QAN / MI / IQAN (JAX). Ejecutar: python tools/verify_cap07.py"""
import itertools
import numpy as np
import jax, jax.numpy as jnp
from qang.qan import (qan_state, qan_readout, mutual_information_qg, mi_matrix_qg,
                     mi_shot_variance_qg, joint_probs_qg, iqan_fit)

# 1. Producto → I = 0 exacto
I0 = float(mutual_information_qg(0.3, -0.6, 0.3 * -0.6))
assert abs(I0) < 1e-12, I0

# 2. Estado Bell-equivalente desde una capa QAN: |++> → RZZ(250 qga) → RY(γ)
n, pairs = 2, ((0, 1),)
best = None
for g0, g1 in itertools.product([-250.0, 0.0, 250.0], repeat=2):
    psi = qan_state(jnp.array([250.0, 250.0]), jnp.array([250.0]), jnp.array([g0, g1]), n, pairs)
    qz, qzz = qan_readout(psi, n)
    I = float(mutual_information_qg(qz[0], qz[1], qzz[0, 1]))
    if best is None or I > best[0]:
        best = (I, g0, g1, float(qzz[0, 1]), np.asarray(qz))
I_bell, g0, g1, qzz_bell, qz_bell = best
assert abs(I_bell - 1.0) < 1e-10, best
assert abs(abs(qzz_bell) - 1.0) < 1e-10 and np.allclose(qz_bell, 0.0, atol=1e-10)
print(f"[2] Bell-equiv: gamma=({g0:+.0f},{g1:+.0f}) qga, qg_ZZ={qzz_bell:+.3f}, I={I_bell:.12f} bit")

# 3. Período 500 qga de RZZ en la lectura: RZZ(β+500) = -i ZZ·RZZ(β); ZZ se absorbe
#    como R_Z(π) virtual (γ → −γ) y es invisible en |ψ|² (THEORY_CORE §1).
psiA = qan_state(jnp.array([120.0, 330.0]), jnp.array([77.0]), jnp.array([40.0, -90.0]), n, pairs)
psiB = qan_state(jnp.array([120.0, 330.0]), jnp.array([577.0]), jnp.array([-40.0, 90.0]), n, pairs)
assert np.allclose(np.abs(np.asarray(psiA)) ** 2, np.abs(np.asarray(psiB)) ** 2, atol=1e-12)

# 4. Varianza método delta vs Monte Carlo (N = 4000 disparos, 4000 réplicas)
qa, qb, qab = 0.2, -0.3, 0.5
p = np.asarray(joint_probs_qg(qa, qb, qab))
N, R = 4000, 4000
rng = np.random.default_rng(1)
counts = rng.multinomial(N, p, size=R) / N
qz_a = counts[:, 0] + counts[:, 1] - counts[:, 2] - counts[:, 3]
qz_b = counts[:, 0] - counts[:, 1] + counts[:, 2] - counts[:, 3]
qzz = counts[:, 0] - counts[:, 1] - counts[:, 2] + counts[:, 3]
I_mc = np.asarray(jax.vmap(mutual_information_qg)(qz_a, qz_b, qzz))
var_delta = float(mi_shot_variance_qg(qa, qb, qab)) / N
ratio = I_mc.var() / var_delta
assert 0.85 < ratio < 1.15, ratio
print(f"[4] I={float(mutual_information_qg(qa,qb,qab)):.4f} bit  sigma_delta={np.sqrt(var_delta):.5f}  "
      f"sigma_MC={I_mc.std():.5f}  ratio={ratio:.3f}")
N_star = float(mi_shot_variance_qg(qa, qb, qab)) / 0.01 ** 2
print(f"    disparos para sigma_I = 0.01 bit: N* = {N_star:.0f}")

# 5. Poda de entrelazadores por MI (4 qubits, capa QAN en cadena, lectura Z única)
n4 = 4
chain = ((0, 1), (1, 2), (2, 3))
alpha = jnp.full((n4,), 250.0)                      # |+>^4
beta = jnp.array([250.0, 60.0, 250.0])              # entrelazador débil en (1,2)
gamma = jnp.array([-250.0, 250.0, -250.0, 250.0])
qz, qzz = qan_readout(qan_state(alpha, beta, gamma, n4, chain), n4)
M = np.asarray(mi_matrix_qg(qz, qzz, n4))
all_pairs = list(itertools.combinations(range(n4), 2))
tau = 0.05
kept = [pq for pq in all_pairs if M[pq] >= tau]
print(f"[5] MI (bit) por par: " + ", ".join(f"{pq}:{M[pq]:.3f}" for pq in all_pairs))
print(f"    RZZ all-to-all = {len(all_pairs)}, retenidos (I>={tau}) = {len(kept)} -> "
      f"-{100*(1-len(kept)/len(all_pairs)):.0f}% compuertas 2Q, 0 ancillas, 1 base de medida")
assert len(kept) < len(all_pairs)

# 6. IQAN: invertir (qg_Z*=0,0 ; I*=1 bit); lr en qga^2 (= 0.1 rad^2 * (1000/2pi)^2 ~ 2.5e3)
conv = 0
for seed in range(8):
    params, loss = iqan_fit(jnp.zeros(2), jnp.array([1.0]), 2, pairs, steps=500, lr=2500.0, seed=seed)
    qz, qzz = qan_readout(qan_state(params[:2], params[2:3], params[3:], 2, pairs), 2)
    I_fit = float(mutual_information_qg(qz[0], qz[1], qzz[0, 1]))
    beta_fit = float(params[2]) % 500.0
    conv += (I_fit > 0.99)
print(f"[6] IQAN (semilla {seed}): loss={loss:.2e}  I={I_fit:.4f} bit  qg_Z={np.asarray(qz).round(3)}  "
      f"beta={beta_fit:.1f} qga (minimo de area; piso mu*(250/500)^2 = 2.5e-4)")
print(f"    convergencias I>0.99: {conv}/8")
assert conv >= 6 and abs(beta_fit - 250.0) < 1.0
print("verify_cap07: OK")
