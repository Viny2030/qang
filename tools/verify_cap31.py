"""tools/verify_cap31.py — Cap. 31: falsación pre-registrada."""
import numpy as np
from scipy.stats import binom
from qang.qfalsify import *

# [1] registro
pred = {"id": "F-31-01", "claim": "El filtro N mod 4 devuelve la energía ideal bajo T1 uniforme (N=2, n=4)",
        "metric": "|ΔE|", "threshold": 1e-9, "direction": "<", "alpha": 0.05, "seed": 7, "date": "2026-11-16",
        "script": "tools/verify_cap28.py"}
d = register(pred)
assert d == register(dict(reversed(list(pred.items()))))      # el orden de las claves no cuenta
assert verify_registration(pred, d)
bad = dict(pred, threshold=1e-6)
assert not verify_registration(bad, d) and register(bad) != d
changed = [k for k in pred if register(dict(pred, **{k: str(pred[k]) + " "})) != d]
assert len(changed) == len(pred)
print(f"[1] SHA-256 = {d[:16]}…; independiente del orden; cualquier cambio en {len(changed)}/{len(pred)} campos lo invalida")

# [2] p-valor exacto
for n, k, p0 in ((100, 60, 0.5), (1000, 530, 0.5), (50, 10, 0.1)):
    assert abs(binom_upper(k, n, p0) - binom.sf(k - 1, n, p0)) < 1e-12
    assert abs(binom_lower(k, n, p0) - binom.cdf(k, n, p0)) < 1e-12
print(f"[2] P(X≥60 | n=100, p0=0.5) = {binom_upper(60,100,0.5):.5f}; P(X≥530 | n=1000) = {binom_upper(530,1000,0.5):.5f}")
print("    decisiones:", decide(60, 100, 0.5), "|", decide(530, 1000, 0.5), "|", decide(40, 100, 0.5), "|", decide(55, 100, 0.5))
assert decide(60, 100, 0.5) == "confirmada" and decide(40, 100, 0.5) == "refutada" and decide(55, 100, 0.5) == "no concluyente"

# [3] Holm: FWER
rng = np.random.default_rng(11)
m, trials, n, p0 = 20, 3000, 400, 0.5
unc = holm_fw = 0
for _ in range(trials):
    ks = rng.binomial(n, p0, size=m)
    ps = [binom_upper(int(k), n, p0) for k in ks]
    unc += any(p <= 0.05 for p in ps)
    holm_fw += any(holm(ps, 0.05))
print(f"[3] 20 hipótesis nulas, {trials} repeticiones: P(≥1 falso positivo) sin corregir = {unc/trials:.3f}; con Holm = {holm_fw/trials:.3f}")
assert unc / trials > 0.5 and holm_fw / trials < 0.065
assert holm((0.001, 0.03, 0.04), 0.05) == (True, False, False) and holm((0.001, 0.01, 0.02), 0.05) == (True, True, True)

# [4] potencia
n90 = shots_for_power(0.5, 0.6, 0.05, 0.9)
n99 = shots_for_power(0.5, 0.6, 0.05, 0.99)
assert power(n90, 0.5, 0.6) >= 0.9 and power(n90 - 5, 0.5, 0.6) < 0.9 + 0.05
print(f"[4] detectar 0.6 contra 0.5 (α=0.05): n={n90} para potencia 0.90 (alcanzada {power(n90,0.5,0.6):.3f}); n={n99} para 0.99")
n_small = shots_for_power(0.5, 0.52, 0.05, 0.9)
print(f"    detectar 0.52 contra 0.5: n={n_small} (×{n_small/n90:.1f}); es la regla n ∝ 1/Δ²")
assert 20 < n_small / n90 < 28   # (0.10/0.02)² = 25
print("verify_cap31: OK")
