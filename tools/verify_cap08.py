"""tools/verify_cap08.py — QAN ↔ distribuciones en espacio qang (JAX). Ejecutar: python tools/verify_cap08.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qan import qan_state, qan_readout
from qang.qdist import (qg_moments, probs_from_moments, moment_order, truncate_k_local, tv_distance,
                        hellinger2_qg, haar_posterior_qz, shots_tv_bound, n_moments_k_local, kl_bits)

n, chain = 4, ((0, 1), (1, 2), (2, 3))
key = jax.random.PRNGKey(11)
a, b, g = [jax.random.uniform(k, (m,), minval=0.0, maxval=500.0)
           for k, m in zip(jax.random.split(key, 3), (n, len(chain), n))]
psi = qan_state(a, b, g, n, chain)
p = jnp.abs(psi) ** 2
q = qg_moments(p, n)
order = np.asarray(moment_order(n))

# 1. Walsh exacta: ida y vuelta, q_∅ = 1, orden 1 = qg_Z, orden 2 = qg_ZZ
assert np.allclose(np.asarray(probs_from_moments(q, n)), np.asarray(p), atol=1e-14)
assert abs(float(q[0]) - 1.0) < 1e-14
qz, qzz = qan_readout(psi, n)
for i in range(n):
    assert abs(float(q[1 << (n - 1 - i)]) - float(qz[i])) < 1e-13   # qubit q ↔ bit n-1-q (como x)
for i in range(n):
    for j in range(i + 1, n):
        assert abs(float(q[(1 << (n - 1 - i)) | (1 << (n - 1 - j))]) - float(qzz[i, j])) < 1e-13
print(f"[1] Walsh exacta: 2^n={2**n} momentos; orden ≤2 = {n_moments_k_local(n, 2)} números")

# 2. Parseval y cota de disparos  E[TV] ≤ ½√(M/N)  (MC, N = 2000 disparos, 2000 réplicas)
rng = np.random.default_rng(0)
N, R = 2000, 2000
pn = np.asarray(p)
counts = rng.multinomial(N, pn, size=R) / N
tv_hist = 0.5 * np.abs(counts - pn).sum(1)
chi = np.asarray(jax.vmap(lambda c: qg_moments(c, n))(jnp.asarray(counts)))
mask2 = (order <= 2)
q_hat2 = np.where(mask2, chi, np.asarray(q)); q_hat2[:, 0] = 1.0        # sólo momentos ≤2 estimados
p_hat2 = np.asarray(jax.vmap(lambda qq: probs_from_moments(qq, n))(jnp.asarray(q_hat2)))
tv_mom = 0.5 * np.abs(p_hat2 - pn).sum(1)
M_hist, M_mom = 2 ** n - 1, n_moments_k_local(n, 2)
b_hist, b_mom = 0.5 * np.sqrt(M_hist / N), 0.5 * np.sqrt(M_mom / N)
assert tv_hist.mean() <= b_hist and tv_mom.mean() <= b_mom
print(f"[2] E[TV] histograma={tv_hist.mean():.4f} (cota {b_hist:.4f}, M={M_hist}); "
      f"momentos ≤2: {tv_mom.mean():.4f} (cota {b_mom:.4f}, M={M_mom})")
tv_trunc = float(tv_distance(truncate_k_local(p, n, 2), p))
print(f"    sesgo de truncación 2-local (sin ruido): TV={tv_trunc:.4f}")

# 3. Hellinger² = 1 − cos(qg_L), qg_L = |θ−θ'|/2 (meridiano común)
th, th2 = 0.7, 2.1
h2 = float(hellinger2_qg(np.cos(th), np.cos(th2)))
assert abs(h2 - (1 - np.cos(abs(th - th2) / 2))) < 1e-14
print(f"[3] Hellinger²={h2:.6f} = 1−cos(qg_L) ✔")

# 4. Posterior Bayes-Haar
m, v = haar_posterior_qz(jnp.array(500.0), jnp.array(1000.0))
assert abs(float(m)) < 1e-14 and abs(float(v) - 1.0 / 1003.0) < 1e-14
print(f"[4] Haar posterior N=1000, k0=500: E[qg_Z]=0, σ={1000*np.sqrt(float(v)):.2f} m-qg")

# 5. IQAN por momentos: ajustar QAN sólo a momentos ≤2 del objetivo y medir TV total
def loss(params):
    aa, bb, gg = params[:n], params[n:n + len(chain)], params[n + len(chain):]
    qq = qg_moments(jnp.abs(qan_state(aa, bb, gg, n, chain)) ** 2, n)
    return jnp.sum(jnp.where(jnp.asarray(mask2), (qq - q) ** 2, 0.0))
vg = jax.jit(jax.value_and_grad(loss))
best = None
for seed in range(6):
    params = jax.random.uniform(jax.random.PRNGKey(100 + seed), (2 * n + len(chain),), minval=0.0, maxval=500.0)
    for _ in range(600):
        L, grad = vg(params); params = params - 2500.0 * grad
    if best is None or L < best[0]:
        best = (float(L), params)
L, params = best
p_fit = jnp.abs(qan_state(params[:n], params[n:n + len(chain)], params[n + len(chain):], n, chain)) ** 2
tv_fit, kl_fit = float(tv_distance(p_fit, p)), float(kl_bits(p, p_fit))
print(f"[5] IQAN momentos ≤2: loss={L:.2e}  TV(p_fit,p)={tv_fit:.4f}  KL={kl_fit:.4f} bit")
assert L < 1e-5 and tv_fit < tv_trunc   # momentos ≤2 no fijan órdenes 3-4: residuo TV (mixed)

# 6. Presupuesto de disparos n=8, TV ≤ 0.05
n8 = 8
N_h, N_m = shots_tv_bound(n8, 2 ** n8 - 1, 0.05), shots_tv_bound(n8, n_moments_k_local(n8, 2), 0.05)
print(f"[6] n=8, TV≤0.05: histograma N={N_h:.0f} (M=255) vs momentos ≤2 N={N_m:.0f} (M=36) → "
      f"-{100*(1-N_m/N_h):.0f}% disparos")
print("verify_cap08: OK")
