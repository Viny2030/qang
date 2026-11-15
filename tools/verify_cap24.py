"""tools/verify_cap24.py — Filtro de simetría: exacto bajo T1 uniforme (incluso intra-circuito), cocientes T1,
fallo bajo desfase y ZNE. python tools/verify_cap24.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qweightqnn import unary_loader
from qang.qsymfilter import (kraus_amplitude_damping, kraus_dephasing, apply_local_channel, apply_local_channel_per_qubit,
                             run_weight_circuit, filtered_sector_state, sector_observable, sector_indices, richardson_zne)

n = 4
pairs = ((0, 1), (2, 3), (1, 2), (0, 1), (2, 3), (1, 2))
theta = jnp.asarray(np.random.default_rng(0).uniform(0, 1000, len(pairs)))
x = np.array([0.5, -0.3, 0.6, 0.55]); x /= np.linalg.norm(x)
psi0 = unary_loader(jnp.asarray(x), n); rho0 = jnp.outer(psi0, jnp.conj(psi0))
signs = [-1, -1, 1, 1]   # en orden idx=[e_3,e_2,e_1,e_0]: A={e_0,e_1} → +1
rho_ideal = run_weight_circuit(rho0, theta, pairs, n)
S_ideal, _ = filtered_sector_state(rho_ideal, n)
y_ideal = float(sector_observable(S_ideal, signs))
idx = sector_indices(n)

# 1. T1 uniforme tras CADA compuerta: estado filtrado exacto (coherencias incluidas), tasa (1−γ)^L
for g in (0.05, 0.2):
    rho = run_weight_circuit(rho0, theta, pairs, n, lambda r: apply_local_channel(r, kraus_amplitude_damping(g), n))
    S, acc = filtered_sector_state(rho, n)
    err_state = float(jnp.max(jnp.abs(S - S_ideal)))
    y_un = float(jnp.real(jnp.sum(jnp.diag(rho)[idx] * jnp.asarray(signs))))
    assert err_state < 1e-13 and abs(float(acc) - (1 - g) ** len(pairs)) < 1e-13
    print(f"[1] T1 γ={g} intra-circuito: ||ρ_S − ρ_S^ideal||∞={err_state:.1e}, tasa={float(acc):.4f}=(1−γ)^{len(pairs)}; "
          f"y sin filtro={y_un:+.4f} vs ideal {y_ideal:+.4f} ({1000*y_un:+.0f} vs {1000*y_ideal:+.0f} m-qg)")

# 2. T1 no uniforme: cocientes p(e_k)/p(e_j) escalan por (1−γ_k)/(1−γ_j) si el ruido es sólo final; intra-circuito no (mixed)
gams = (0.05, 0.3, 0.1, 0.25)
rho_end = apply_local_channel_per_qubit(rho_ideal, [kraus_amplitude_damping(g) for g in gams], n)
p_end = np.asarray(jnp.real(jnp.diag(rho_end)[idx])); p_id = np.asarray(jnp.real(jnp.diag(rho_ideal)[idx]))
qubit_of = [n - 1 - int(np.log2(i)) for i in idx]          # idx ordenado = [e_3, e_2, e_1, e_0]
ratio_pred = np.array([(1 - gams[k]) for k in qubit_of])
assert np.allclose(p_end / p_id, ratio_pred, atol=1e-13)
p_corr = p_end / ratio_pred; p_corr /= p_corr.sum()
rho_mid = run_weight_circuit(rho0, theta, pairs, n, lambda r: apply_local_channel_per_qubit(r, [kraus_amplitude_damping(g) for g in gams], n))
S_mid, _ = filtered_sector_state(rho_mid, n)
y_mid = float(sector_observable(S_mid, signs))
print(f"[2] T1 no uniforme final: p(e_k) = (1−γ_k)·p_ideal exacto → corrección por cocientes recupera y={float(np.sum(p_corr*signs)):+.4f} (ideal {y_ideal:+.4f})")
print(f"    T1 no uniforme intra-circuito: y filtrado={y_mid:+.4f} (error {abs(y_mid-y_ideal):.4f}) → la corrección final no basta (mixed)")

# 3. Desfase p tras cada compuerta: el filtro NO es exacto (coherencias intra-sector); ZNE de Richardson λ∈{1,2,3}
p = 0.02
def y_at(lam):
    rho = run_weight_circuit(rho0, theta, pairs, n, lambda r: apply_local_channel(r, kraus_dephasing(lam * p), n))
    S, _ = filtered_sector_state(rho, n)
    return float(sector_observable(S, signs))
ys = [y_at(l) for l in (1, 2, 3)]
y_zne = richardson_zne([1, 2, 3], ys)
print(f"[3] desfase p={p}: y(λ=1,2,3)={np.round(ys,4)} ideal={y_ideal:+.4f}; error filtrado={abs(ys[0]-y_ideal):.4f}, "
      f"ZNE={y_zne:+.4f} (error {abs(y_zne-y_ideal):.4f}) → ×{abs(ys[0]-y_ideal)/max(abs(y_zne-y_ideal),1e-12):.1f} mejor")
assert abs(ys[0] - y_ideal) > 1e-3 and abs(y_zne - y_ideal) < abs(ys[0] - y_ideal)
print("verify_cap24: OK")
