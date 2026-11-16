"""tools/verify_cap40.py — Cap. 40: materia y antimateria leídas con qang (qubit de Dirac y pares)."""
import numpy as np
from qang.qanti import *

rng = np.random.default_rng(40)

# [1] la conjugación de carga invierte las tres lecturas y lleva +E a −E
worst, err_e = 0.0, 0.0
for _ in range(500):
    p, m = rng.uniform(-3, 3), rng.uniform(0.2, 3)
    psi = eigenstate(p, m, +1)
    chi = conjugate(psi)
    b0, b1 = bloch_mqg(psi), bloch_mqg(chi)
    worst = max(worst, float(np.max(np.abs(b1 + b0))))
    E = np.sqrt(p * p + m * m)
    err_e = max(err_e, abs(float(np.real(np.vdot(chi, dirac_h(p, m) @ chi))) + E))
    assert abs(b0[2] - 1000 * m / E) < 1e-9
assert worst < 1e-9 and err_e < 1e-12
print(f"[1] 500 autoestados +E al azar: C = σy·conj invierte (qg_X, qg_Y, qg_Z) con error {worst:.1e} m-qg y da energía −E (error {err_e:.1e}); lectura de qg_Z = +1000·m/E")

# [2] ningún estado puro de un qubit es su propia antipartícula (Majorana necesita más espacio)
ov = 0.0
for _ in range(2000):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v /= np.linalg.norm(v)
    ov = max(ov, self_conjugate_overlap(v))
assert ov < 1e-15
print(f"[2] 2000 estados puros al azar: máx |⟨ψ|Cψ⟩| = {ov:.1e} (el mapa antípoda no tiene puntos fijos)")

# [3] Zitterbewegung: qg_Z(t) = 1/γ² + β² cos 2Et (en m-qg)
p, m = 0.75, 1.0
g = gamma_of(p, m)
dev = max(abs(zb_qgz_mqg(p, m, t) - zb_analytic_mqg(p, m, t)) for t in np.linspace(0, 20, 81))
assert dev < 1e-9
b2 = (p * p) / (p * p + m * m)
print(f"[3] p/m = 0.75 (γ = {g:.4f}, β = {np.sqrt(b2):.4f}): media {1000/g**2:.1f} m-qg, amplitud {1000*b2:.1f} m-qg, máx |exacto − fórmula| = {dev:.1e} m-qg")

# [4] creación de pares: Landau–Zener numérico frente a exp(−π m²/eE)
rows = []
for m_, eE in ((0.5, 1.0), (0.8, 1.0), (1.0, 1.0)):
    Pa = pair_probability(m_, eE)
    Pn = lz_numeric(2 * m_, 2 * eE, T=250.0, steps=1000000)
    assert abs(Pa - Pn) < 3e-3, (m_, eE, Pa, Pn)
    rows.append((m_, eE, Pa, Pn, pair_reading_mqg(Pa)))
maxerr = max(abs(r_[2] - r_[3]) for r_ in rows)
print(f"[4] (ventana finita ±250; máx |analítico − numérico| = {maxerr:.1e}) pares por modo (c=ħ=1): " + "; ".join(f"m²/eE={m_*m_/eE:.2f}: P={Pa:.4f} (numérico {Pn:.4f}), lectura {r:+.1f} m-qg" for m_, eE, Pa, Pn, r in rows))

# [5] el par en singlete es invisible en las lecturas locales
(z0, z1, zz), c = singlet_readings()
assert abs(z0) < 1e-12 and abs(z1) < 1e-12 and abs(zz + 1000) < 1e-9
print(f"[5] singlete: lecturas locales {z0:+.1f} y {z1:+.1f} m-qg; correlación qg_ZZ = {zz:+.1f} m-qg")

# [6] costo de certificar la rama
Ns = {gm: shots_to_certify_branch(gm) for gm in (2.0, 7.0888120)}
assert abs(Ns[2.0] - 27.0) < 1e-9 and 440 < Ns[7.0888120] < 447
print(f"[6] disparos para ver el signo a 3σ: γ=2 → {Ns[2.0]:.0f}; γ=7.089 (β=0.99) → {Ns[7.0888120]:.0f}")
print("verify_cap40: OK")
