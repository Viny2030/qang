"""tools/verify_cap30.py — Cap. 30: caracterización en m-qg."""
import numpy as np
from qang.qchar import *

# [1] T1
T1 = 100.0
ts = np.array([20, 50, 100, 150, 250.0])
assert abs(fit_t1(ts, t1_qz(ts, T1)) - T1) < 1e-9
x = optimal_delay_ratio()
assert abs(x - 1.5936) < 1e-3
print(f"[1] ajuste T1 exacto a 1e-9; instante óptimo t*/T1 = {x:.4f}; qg_Z(t*) = {1000*t1_qz(x*T1, T1):+.0f} m-qg")
shots = 2000
for name, grid in (("óptimo único 1.59·T1", [x * T1] * 4), ("rejilla 5 puntos", list(ts)), ("todo corto 0.2·T1", [0.2 * T1] * 4)):
    I = fisher_t1(grid, T1, shots)
    sd = simulate_t1_std(grid, T1, shots, 400, seed=3) if len(set(grid)) > 1 or True else 0
    crb = 1 / np.sqrt(I)
    print(f"    {name:22s}: CRB σ_T1 = {crb:6.2f}, simulado {sd:6.2f}")
    assert sd > 0.9 * crb
I_opt, I_short = fisher_t1([x * T1] * 4, T1, shots), fisher_t1([0.2 * T1] * 4, T1, shots)
assert 3 < I_opt / I_short < 4
print(f"    información (mismos disparos): óptimo {I_opt:.2f} vs 0.2·T1 {I_short:.2f} (×{I_opt/I_short:.1f})")

# [2] lectura
e01, e10, s01, s10 = calibrate_readout(40, 2000, 120, 2000)
print(f"[2] e01={e01:.4f}±{s01:.4f}, e10={e10:.4f}±{s10:.4f}; disparos para σ=0.002 con e=0.02: {shots_for_error(0.02, 0.002):.0f}")
assert abs(shots_for_error(0.02, 0.002) - 4900) < 1e-9
m_true = 600.0
m_read = (1 - e01 - e10) * m_true + 1000 * (e10 - e01)
assert abs(correct_mqg(m_read, e01, e10) - m_true) < 1e-9
print(f"    lectura +600 → {m_read:+.1f} m-qg; corregida {correct_mqg(m_read, e01, e10):+.1f}")

# [3] XEB / HOP
n = 8
for f in (1.0, 0.7, 0.3):
    h = mean_hop(n, f, 300, seed=1)
    pred = f * HOP_PORTER_THOMAS + (1 - f) / 2
    print(f"[3] n=8 f={f:.1f}: HOP simulado {h:.4f}, predicho {pred:.4f}; fidelidad recuperada {fidelity_from_hop(h):.3f}")
    assert abs(h - pred) < 0.01 and abs(fidelity_from_hop(h) - f) < 0.06
assert abs(HOP_PORTER_THOMAS - 0.8466) < 1e-4
assert fidelity_from_hop(2 / 3) > 0 and abs(fidelity_from_hop(2 / 3) - (2 / 3 - 0.5) / (HOP_PORTER_THOMAS - 0.5)) < 1e-15
print(f"    umbral HOP 2/3 ⇒ f ≥ {fidelity_from_hop(2/3):.3f}")

# [4] ZZ
zeta, t = 2 * np.pi * 0.2, 1.3
dphi = np.arccos(zz_ramsey_qgx(zeta, t, 0)) - 0.0
q0, q1 = zz_ramsey_qgx(zeta, t, 0), zz_ramsey_qgx(zeta, t, 1)
assert abs(q0 - np.cos(zeta * t / 2)) < 1e-12 and abs(q1 - np.cos(-zeta * t / 2)) < 1e-12
y0, y1 = zz_ramsey_qgx(zeta, t, 0, "Y"), zz_ramsey_qgx(zeta, t, 1, "Y")
assert abs(y0 + y1) < 1e-12 and abs(abs(y0) - np.sin(zeta * t / 2)) < 1e-12 and abs(q0 - q1) < 1e-12
phi_qga = zz_phase_qga(zeta, t)
rel = 2 * 1000 * (zeta * t / 2) / (2 * np.pi)   # fase relativa 2·(ζt/2) en qga
assert abs(phi_qga - rel) < 1e-9
print(f"[4] ζ/2π=0.2 MHz, t=1.3 µs: fase relativa por vecino = {phi_qga:.1f} qga (= ζ t); qg_X = {1000*q0:+.1f} m-qg en ambos casos; qg_Y = {1000*y0:+.1f} / {1000*y1:+.1f} m-qg (distingue al vecino)")

# [5] cuantización
for b in (6, 8, 10):
    L = 2 ** b
    e_turn = rms_quantization_error(L)
    e_qga = rms_quantization_error(L, unit_period=1000.0)
    e_rad = rms_quantization_error(L, unit_period=2 * np.pi)
    th = 1.0 / L / np.sqrt(12)
    assert abs(e_turn - th) / th < 5e-3 and abs(e_qga / 1000 - e_turn) < 1e-12 and abs(e_rad / (2 * np.pi) - e_turn) < 1e-12
    print(f"[5] {b} bits: σ = {e_qga:.4f} qga = {e_rad:.5f} rad = {e_turn:.3e} vueltas (= 1/(L√12)); cambiar de unidad no la mueve")
print("verify_cap30: OK")
