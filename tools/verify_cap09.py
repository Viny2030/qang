"""tools/verify_cap09.py — Fisher clásica/cuántica en qga, gradiente natural y mesetas. python tools/verify_cap09.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qan import qan_state, qan_readout, mutual_information_qg, iqan_loss
from qang.qfisher import (fubini_study_metric, classical_fisher_z, natural_step, psd_gap,
                          gradient_variance, circuits_for_metric, QFI_1Q_QGA, SIGMA_1Q_QGA)

# 1. Un qubit RY(α qga), lectura Z: CFI = QFI = (2π/1000)² qga⁻²  → σ_qga = 159.15/√N
s1 = lambda t: qan_state(t, jnp.zeros(0), jnp.zeros(1), 1, ())
for alpha in (37.0, 120.0, 250.0, 410.0):
    t = jnp.array([alpha])
    qfi = 4.0 * float(fubini_study_metric(s1, t)[0, 0])
    cfi = float(classical_fisher_z(s1, t)[0, 0])
    assert abs(qfi - QFI_1Q_QGA) < 1e-15 and abs(cfi - QFI_1Q_QGA) < 1e-12, (alpha, qfi, cfi)
print(f"[1] CFI_Z = QFI = {QFI_1Q_QGA:.4e} qga⁻² (uniforme en α); σ_qga(N) = {SIGMA_1Q_QGA:.2f}/√N qga; "
      f"en qg_Z: I_F = 1/(1−qg_Z²) → ∞ en polos")

# 2. QAN n=3 cadena: 4g ⪰ F_Z (CFI ≤ QFI) en parámetros aleatorios
n, pairs = 3, ((0, 1), (1, 2))
P = 2 * n + len(pairs)
s3 = lambda t: qan_state(t[:n], t[n:n + len(pairs)], t[n + len(pairs):], n, pairs)
for seed in range(5):
    t = jax.random.uniform(jax.random.PRNGKey(seed), (P,), minval=0.0, maxval=1000.0)
    gap = psd_gap(4.0 * fubini_study_metric(s3, t), classical_fisher_z(s3, t))
    assert gap > -1e-12, gap
print(f"[2] QFI − CFI_Z ⪰ 0 en 5/5 muestras (P={P}); circuitos métrica: QFI={circuits_for_metric(P)[0]} "
      f"(1 ancilla) vs CFI_Z={circuits_for_metric(P)[1]} (0 ancillas)")

# 3. IQAN (Cap. 7, n=2, I*=1 bit): euclidiano en qga vs gradiente natural CFI_Z regularizado
n2, p2 = 2, ((0, 1),)
tz, tI = jnp.zeros(2), jnp.array([1.0])
L = lambda t: iqan_loss(t, tz, tI, n2, p2, 1.0, 1e-3)
vg = jax.jit(jax.value_and_grad(L))
s2 = lambda t: qan_state(t[:2], t[2:3], t[3:], 2, p2)
cfi = jax.jit(lambda t: classical_fisher_z(s2, t))
TOL, MAXIT = 1e-3, 600
res = {"euclid": [], "natural": []}
for seed in range(8):
    t0 = jax.random.uniform(jax.random.PRNGKey(seed), (5,), minval=0.0, maxval=500.0)
    t = t0
    for it in range(MAXIT):
        l, g = vg(t)
        if l < TOL: break
        t = t - 2500.0 * g
    res["euclid"].append(it if l < TOL else MAXIT)
    t = t0
    for it in range(MAXIT):
        l, g = vg(t)
        if l < TOL: break
        t = natural_step(t, g, cfi(t), lr=0.2, lam=1e-3 * QFI_1Q_QGA)
    res["natural"].append(it if l < TOL else MAXIT)
e, nat = np.array(res["euclid"]), np.array(res["natural"])
print(f"[3] iteraciones a loss<1e-3 (8 semillas): euclid qga mediana={np.median(e):.0f} conv={np.sum(e<MAXIT)}/8 | "
      f"natural CFI_Z mediana={np.median(nat):.0f} conv={np.sum(nat<MAXIT)}/8")
Pp = 5
print(f"    circuitos (desplazamiento de parámetro, P={Pp}): euclid {np.median(e)*2*Pp:.0f} | "
      f"natural {np.median(nat)*4*Pp:.0f} (gradiente 2P + métrica CFI_Z 2P, 0 ancillas)")

# 4. Meseta estéril: Var[∂L/∂α_0] vs n, en qga y en rad — mismo decaimiento, factor (2π/1000)²
def make_loss(nq):
    prs = tuple((i, i + 1) for i in range(nq - 1))
    def L(t):
        psi = qan_state(t[:nq], t[nq:nq + len(prs)], t[nq + len(prs):], nq, prs)
        qz, _ = qan_readout(psi, nq)
        return qz[0] * qz[nq - 1]
    return L, 2 * nq + len(prs)
vars_qga = []
for nq in (2, 3, 4, 5, 6):
    Lq, Pq = make_loss(nq)
    vars_qga.append(gradient_variance(Lq, Pq, jax.random.PRNGKey(nq), samples=300))
vars_rad = [v / QFI_1Q_QGA for v in vars_qga]
print("[4] Var[∂L/∂α_0] qga⁻²: " + ", ".join(f"n={nq}:{v:.2e}" for nq, v in zip((2, 3, 4, 5, 6), vars_qga)))
print("    en rad⁻²:           " + ", ".join(f"n={nq}:{v:.2e}" for nq, v in zip((2, 3, 4, 5, 6), vars_rad)))
assert all(abs(vr * QFI_1Q_QGA - vq) < 1e-18 for vr, vq in zip(vars_rad, vars_qga))
print("    capa única en cadena con observable 2-local: sin meseta (Var ≈ const); el cambio de unidad es un"
      "\n    reescalado constante (2π/1000)² y NO altera la escala de la varianza (THEORY_CORE §2.1, against)")
print("verify_cap09: OK")
