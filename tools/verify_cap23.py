"""tools/verify_cap23.py — WeightQNN (RBS) + filtro de simetría bajo T1, y QSP/Chebyshev. python tools/verify_cap23.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qweightqnn import (unary_loader, weight_qnn, weight_mask, sector_readout, readout_unfiltered,
                             amplitude_damping_probs, qsp_readout, RAD_PER_QGA)

n = 4
pairs = ((0, 1), (2, 3), (1, 2)) * 3          # 3 capas, 9 RBS
mask1 = np.asarray(weight_mask(n, 1))

# 1. Cargador unario exacto y conservación del peso por la red
rng = np.random.default_rng(0)
x = rng.normal(size=n); x /= np.linalg.norm(x)
psi = unary_loader(jnp.asarray(x), n)
e = np.array([1 << (n - 1 - k) for k in range(n)])
assert np.allclose(np.asarray(jnp.real(psi[e])), x, atol=1e-13) and abs(float(jnp.sum(jnp.abs(psi) ** 2)) - 1) < 1e-14
theta = jnp.asarray(rng.uniform(0, 1000, len(pairs)))
out = weight_qnn(theta, psi, n, pairs)
p = np.asarray(jnp.abs(out) ** 2)
assert p[~mask1].sum() < 1e-14
print(f"[1] cargador unario: {n-1} RBS, error 1e-13; red de {len(pairs)} RBS: peso fuera del sector = {p[~mask1].sum():.1e}")

# 2. Entrenamiento (JAX) de un clasificador binario en el sector: etiquetas y = sign(x0² + x1² − x2² − x3²) rotadas
def make_data(m, key):
    X = jax.random.normal(key, (m, n)); X = X / jnp.linalg.norm(X, axis=1, keepdims=True)
    R = jnp.asarray(np.linalg.qr(np.random.default_rng(1).normal(size=(n, n)))[0])
    Xr = X @ R
    y = jnp.sign(Xr[:, 0] ** 2 + Xr[:, 1] ** 2 - Xr[:, 2] ** 2 - Xr[:, 3] ** 2)
    return X, y
Xtr, ytr = make_data(200, jax.random.PRNGKey(2)); Xte, yte = make_data(400, jax.random.PRNGKey(3))
def predict(theta, X):
    def one(x):
        p = jnp.abs(weight_qnn(theta, unary_loader(x, n), n, pairs)) ** 2
        return sector_readout(p, n, (0, 1))[0]
    return jax.vmap(one)(X)
loss = jax.jit(lambda th: -jnp.mean(predict(th, Xtr) * ytr))   # correlación margen·etiqueta (forma cuadrática)
vg = jax.jit(jax.value_and_grad(loss))
best = None
for seed in range(6):
    th = jnp.asarray(np.random.default_rng(seed).uniform(0, 1000, len(pairs)))
    for it in range(500):
        L, g = vg(th); th = th - 4000.0 * g
    if best is None or L < best[0]: best = (float(L), th)
L, theta = best
acc_te = float(jnp.mean(jnp.sign(predict(theta, Xte)) == yte))
margin = float(jnp.mean(jnp.abs(predict(theta, Xte))))
print(f"[2] WeightQNN 4 qubits/9 RBS: loss={float(L):.3f}, exactitud test={acc_te:.3f}, |y| medio={margin:.3f}")
assert acc_te > 0.9

# 3. Relajación T1 uniforme γ: sin filtro y → (1−γ)y; con filtro y exacto, tasa (1−γ). Disparos para confirmar signo a 3σ.
def noisy_preds(theta, X, gammas, filtered):
    def one(x):
        p = jnp.abs(weight_qnn(theta, unary_loader(x, n), n, pairs)) ** 2
        pn = amplitude_damping_probs(p, n, gammas)
        return sector_readout(pn, n, (0, 1))[0] if filtered else readout_unfiltered(pn, n, (0, 1))
    return jax.vmap(one)(X)
y0 = predict(theta, Xte)
for gam in (0.1, 0.3):
    yu = noisy_preds(theta, Xte, (gam,) * n, False); yf = noisy_preds(theta, Xte, (gam,) * n, True)
    assert np.allclose(np.asarray(yu), (1 - gam) * np.asarray(y0), atol=1e-12) and np.allclose(np.asarray(yf), np.asarray(y0), atol=1e-12)
    m0 = float(jnp.mean(jnp.abs(y0)))
    N_u = (3 / ((1 - gam) * m0)) ** 2; N_f = (3 / m0) ** 2 / (1 - gam)
    print(f"[3] γ={gam}: sin filtro y→{1-gam:.1f}·y (N_3σ≈{N_u:.0f}) | con filtro y exacto, tasa {1-gam:.1f} (N_3σ≈{N_f:.0f}) → ×{N_u/N_f:.2f} menos disparos")
# 3b. T1 no uniforme: el filtro deja sesgo (mixed)
gams = (0.05, 0.35, 0.10, 0.30)
yf = noisy_preds(theta, Xte, gams, True); yu = noisy_preds(theta, Xte, gams, False)
acc_f = float(jnp.mean(jnp.sign(yf) == yte)); acc_u = float(jnp.mean(jnp.sign(yu) == yte))
print(f"    γ no uniforme {gams}: exactitud con filtro={acc_f:.3f}, sin filtro={acc_u:.3f}, limpia={acc_te:.3f} (sesgo residual: mixed)")

# 4. QSP: con φ ≡ 0, qg_Z = T_d(2a²−1); ajuste de un umbral con d=7 fases en qga
for d in (1, 3, 7):
    a = jnp.linspace(-1, 1, 201)
    q = jax.vmap(lambda aa: qsp_readout(jnp.zeros(d + 1), aa))(a)
    T = np.cos(d * np.arccos(np.clip(2 * np.asarray(a) ** 2 - 1, -1, 1)))
    assert np.allclose(np.asarray(q), T, atol=1e-12)
print("[4] QSP φ≡0: qg_Z = T_d(2a²−1) a 1e-12 para d=1,3,7 (d pulsos físicos W + d+1 RZ virtuales)")
a_tr = jnp.linspace(-1, 1, 101); target = jnp.where(jnp.abs(a_tr) > 0.5, 1.0, -1.0)
d = 7
lossq = jax.jit(lambda ph: jnp.mean((jax.vmap(lambda aa: qsp_readout(ph, aa))(a_tr) - target) ** 2))
vgq = jax.jit(jax.value_and_grad(lossq))
best = None
for seed in range(4):
    ph = jax.random.uniform(jax.random.PRNGKey(seed), (d + 1,), minval=0.0, maxval=1000.0)
    for it in range(400):
        Lq, g = vgq(ph); ph = ph - 2000.0 * g
    if best is None or Lq < best[0]: best = (float(Lq), ph)
Lq, ph = best
pred = jax.vmap(lambda aa: qsp_readout(ph, aa))(a_tr)
acc_q = float(jnp.mean(jnp.sign(pred) == target))
print(f"[4] QSP d=7 ajustado a 1[|a|>0.5]: MSE={Lq:.3f}, exactitud={acc_q:.3f}; fases (qga) = {np.round(np.asarray(ph) % 1000).astype(int)}")
assert acc_q > 0.85
print("verify_cap23: OK")
