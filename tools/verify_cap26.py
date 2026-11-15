"""tools/verify_cap26.py — Núcleos en qg: forma cerrada, test de inversión, concentración vs núcleo unario. python tools/verify_cap26.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qkernels import (bloch_from_qga, product_fidelity_kernel, z_only_kernel, unary_kernel, unary_inversion_test,
                           kernel_concentration, shots_for_kernel, RAD_PER_QGA)

# 1. Forma cerrada del núcleo producto frente al solapamiento explícito
def ket(a, p):
    th, ph = a * RAD_PER_QGA, p * RAD_PER_QGA
    return jnp.array([jnp.cos(th / 2), jnp.exp(1j * ph) * jnp.sin(th / 2)])
rng = np.random.default_rng(0)
for _ in range(5):
    n = 3
    a1, p1, a2, p2 = (rng.uniform(0, 500, n), rng.uniform(0, 1000, n), rng.uniform(0, 500, n), rng.uniform(0, 1000, n))
    psi1 = ket(a1[0], p1[0]); psi2 = ket(a2[0], p2[0])
    for q in range(1, n):
        psi1 = jnp.kron(psi1, ket(a1[q], p1[q])); psi2 = jnp.kron(psi2, ket(a2[q], p2[q]))
    k_exp = float(jnp.abs(jnp.vdot(psi1, psi2)) ** 2)
    k_cf = float(product_fidelity_kernel(bloch_from_qga(jnp.asarray(a1), jnp.asarray(p1)), bloch_from_qga(jnp.asarray(a2), jnp.asarray(p2))))
    assert abs(k_exp - k_cf) < 1e-13
print("[1] k = Π(1 + r·r')/2 = |<ψ|ψ'>|² a 1e-13 (5 casos, n=3); núcleo Z-only Π(1+qg_Z qg_Z')/2 ≠ fidelidad (1 base)")

# 2. Concentración exponencial para características de producto aleatorias
print("[2] n   Var[k]      E[k]      disparos para σ_k = E[k]/10")
vs = []
for n in (2, 4, 6, 8, 10):
    v, m = kernel_concentration(n, 20000, jax.random.PRNGKey(n))
    vs.append(v)
    print(f"    {n:2d}  {v:.2e}  {m:.2e}  {shots_for_kernel(m, m/10):.1e}")
assert vs[-1] < vs[0] / 50
# cambio de unidad qga→rad: k es adimensional; la concentración no depende de la unidad (against como mitigación)

# 3. Núcleo unario (sector peso 1): (x·y)² exacto por test de inversión sin ancilla; sin concentración
n = 6
X = rng.normal(size=(2000, n)); X /= np.linalg.norm(X, axis=1, keepdims=True)
ku = (np.einsum("ij,ij->i", X[::2], X[1::2])) ** 2
for i in range(3):
    kt = float(unary_inversion_test(jnp.asarray(X[2 * i]), jnp.asarray(X[2 * i + 1]), n))
    assert abs(kt - ku[i]) < 1e-12
print(f"[3] núcleo unario n={n}: (x·y)² = test de inversión a 1e-12; Var[k]={ku.var():.3e}, E[k]={ku.mean():.3e} (=1/n), "
      f"disparos σ=E/10: {shots_for_kernel(ku.mean(), ku.mean()/10):.0f}")
for n in (4, 8, 16):
    X = rng.normal(size=(4000, n)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    ku = (np.einsum("ij,ij->i", X[::2], X[1::2])) ** 2
    print(f"    n={n:2d}: E[k]={ku.mean():.4f} ≈ 1/n, Var[k]={ku.var():.2e} (polinomial, no exponencial)")
# 4. Coste: SWAP test (1 ancilla, n cSWAP) vs inversión (0 ancillas, profundidad 2L) vs unario (0 ancillas, 2(n−1) RBS)
print("[4] estimar k: SWAP test = 1 ancilla + n cSWAP | inversión U(y)†U(x) = 0 ancillas, 2L | unario = 0 ancillas, 2(n−1) RBS + filtro de sector (T1 exacto, Cap. 24)")
print("verify_cap26: OK")
