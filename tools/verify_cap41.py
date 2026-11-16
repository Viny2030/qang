"""tools/verify_cap41.py — Cap. 41: viajes en el tiempo (CTC como mapas) leídos con qang."""
import numpy as np
from qang.qtimeloop import *

rng = np.random.default_rng(41)

# [1] balance de información de la postselección: el Fisher total por copia es exactamente 1
worst = 0.0
for _ in range(2000):
    a, th = rng.uniform(0.05, 0.99), rng.uniform(0.02, np.pi - 0.02)
    fa, p, fb, tot = pctc_fisher(a, th)
    worst = max(worst, abs(tot - 1.0))
    assert p * fa <= 1.0 + 1e-12
assert worst < 1e-12
a = 0.5
fa_s, p_s, fb_s, tot_s = pctc_fisher(a, np.pi - 1e-9)
fa_n, p_n, fb_n, tot_n = pctc_fisher(a, 1e-9)
print(f"[1] 2000 (a, θ) al azar: |F_total − 1| máx = {worst:.1e}. Con a = 0.5: amplificación en el polo sur {pctc_amplification(a, np.pi):.4f} (=1/a), "
      f"p_post = {pctc_p_post(a, np.pi):.4f} (=a²), disparos por éxito ×{1/pctc_p_post(a, np.pi):.1f}; Fisher aceptado × p = {p_s*fa_s:.4f}")
assert abs(pctc_amplification(a, np.pi) - 2.0) < 1e-12 and abs(pctc_p_post(a, np.pi) - 0.25) < 1e-12

# [2] el mapa de ángulos coincide con la operación explícita C ρ C†
th = 1.9
psi = np.array([np.cos(th / 2), np.sin(th / 2)], dtype=complex)
C = np.diag([1.0, a]).astype(complex)
out = C @ psi
out /= np.linalg.norm(out)
th2 = 2 * np.arctan2(abs(out[1]), abs(out[0]))
assert abs(th2 - pctc_theta_out(a, th)) < 1e-12
print(f"[2] θ = {th:.2f} → θ' = {th2:.6f} con C = diag(1, {a}); fórmula {pctc_theta_out(a, th):.6f}")

# [3] paradoja del abuelo: U = I_CR ⊗ X_CTC
U = np.kron(I2, SX)
rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
p_post = pctc_success(U, rho)
c, o = deutsch_iterate(U, rho)
assert abs(p_post) < 1e-15 and np.max(np.abs(bloch_mqg(c))) < 1e-12 and np.allclose(bloch_mqg(o), bloch_mqg(rho), atol=1e-9)
print(f"[3] abuelo (U = I⊗X): P-CTC p_post = {p_post:.1f} (nunca ocurre); D-CTC: ρ_CTC con lecturas {np.round(bloch_mqg(c), 6)} m-qg, CR de salida = entrada {np.round(bloch_mqg(o), 1)} m-qg")

# [4] conjunto de puntos fijos de una rotación en el bucle, por ángulo en qga
dims = {al: fixed_dimension(bloch_rotation([0, 0, 1], al)) for al in (0, 250, 500, 1000)}
assert dims == {0: 3, 250: 1, 500: 1, 1000: 3}, dims
print(f"[4] rotación alrededor de Z en el bucle: dimensión del conjunto de puntos fijos por ángulo (qga): {dims}")
print(f"[5] disparos para 1000 éxitos con p_post = 0.25: {shots_for_successes(1000, 0.25):.0f}")
print("verify_cap41: OK")
