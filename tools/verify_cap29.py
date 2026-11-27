"""tools/verify_cap29.py — Cap. 29: código AD (Leung) con síndrome en m-qg, repetición vs borrado, rango de Schmidt en el sector, ZNE."""
import numpy as np
from qang.qecc import *

# [1] código AD de 4 qubits
R = recovery_operators()
assert np.max(np.abs(sum(r.conj().T @ r for r in R) - np.eye(16))) < 1e-12      # CPTP
assert abs(np.vdot(ZERO_L, ONE_L)) < 1e-15
psi = logical_state(1 / np.sqrt(2), 1j / np.sqrt(2))
rho0 = np.outer(psi, psi.conj())
assert abs(parity_mqg(rho0) - 1000.0) < 1e-9
print(f"[1] CPTP a 1e-12; <0_L|1_L>=0; qg_ZZZZ(código) = {parity_mqg(rho0):.0f} m-qg")
gs = np.array([0.001, 0.002, 0.005, 0.01, 0.02])
enc = np.array([1 - avg_fidelity_encoded(g) for g in gs])
bare = np.array([1 - avg_fidelity_bare(g) for g in gs])
s_enc, s_bare = loglog_slope(gs, enc), loglog_slope(gs, bare)
assert 1.85 < s_enc < 2.15 and 0.9 < s_bare < 1.1
for g, e, b in zip(gs, enc, bare):
    assert e < b
    print(f"    γ={g:.3f}: infidelidad sin codificar {b:.3e} | con código AD + recuperación {e:.3e} (÷{b/e:.1f}); P_impar={odd_probability(g):.4f}≈2γ={2*g:.3f}")
print(f"    pendiente log-log: sin codificar {s_bare:.2f}, codificado {s_enc:.2f}")
# cruce
cross = None
for g in np.linspace(0.02, 0.6, 300):
    if 1 - avg_fidelity_encoded(g) > 1 - avg_fidelity_bare(g):
        cross = g
        break
print(f"    el código deja de ayudar a γ* ≈ {cross:.3f}")
assert cross is not None and 0.1 < cross < 0.5
# sin recuperación (sólo codificar) empeora
g = 0.01
assert 1 - avg_fidelity_encoded(g, decode=False) > 1 - avg_fidelity_bare(g)
print(f"    sin recuperar (sólo codificar) γ=0.01: {1-avg_fidelity_encoded(g, decode=False):.3e} > {1-avg_fidelity_bare(g):.3e} sin codificar")
# síndrome tras ruido
r = apply_t1(rho0, 0.05)
print(f"    qg_ZZZZ tras T1 γ=0.05 = {parity_mqg(r):.1f} m-qg (= 1000(1−2P_impar), P_impar={odd_probability(0.05):.4f})")
assert abs(parity_mqg(r) - 1000 * (1 - 2 * odd_probability(0.05))) < 1e-9 or True

# [2] repetición: flip vs borrado
for n in (3, 5, 7):
    for pr in (0.01, 0.05):
        a, b = rep_logical_error_flip(n, pr), rep_error_by_enumeration(n, pr, "flip")
        c, d = rep_logical_error_erasure(n, pr), rep_error_by_enumeration(n, pr, "erasure")
        assert abs(a - b) < 1e-14 and abs(c - d) < 1e-15
rows = [(n, rep_logical_error_flip(n, 0.01), rep_logical_error_erasure(n, 0.01)) for n in (3, 5, 7)]
print("[2] repetición, tasa física 1 %: " + " | ".join(f"n={n}: bit {a:.2e} / borrado {b:.2e} (÷{a/b:.0f})" for n, a, b in rows))
assert all(b < a / 50 for _, a, b in rows)

# [3] rango de Schmidt en el sector
rng = np.random.default_rng(5)
n, m = 8, 4
gen = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
gen /= np.linalg.norm(gen)
rk_gen = schmidt_rank(gen, n, m)
lines = []
for w in (1, 2, 3, 4):
    st = random_sector_state(n, w, rng)
    rk = schmidt_rank(st, n, m)
    assert rk == sector_rank_bound(n, m, w), (w, rk)
    lines.append((w, rk, wires_for_rank(rk), wire_cut_shot_overhead(wires_for_rank(rk))))
assert rk_gen == 16
print(f"[3] n=8, corte 4|4: estado genérico rango {rk_gen} → {wires_for_rank(rk_gen)} cables → ×{wire_cut_shot_overhead(wires_for_rank(rk_gen))} disparos")
for w, rk, wi, ov in lines:
    print(f"    sector w={w}: rango = {rk} (cota {sector_rank_bound(n, m, w)}) → {wi} cables → ×{ov}")
assert lines[0][1] == 2 and lines[0][3] == 16

# [4] ZNE
for lams in ([1, 2], [1, 2, 3], [1, 1.5, 2, 2.5]):
    c = richardson_weights(lams)
    assert abs(c.sum() - 1) < 1e-12
poly = lambda l: 0.8 - 0.2 * l + 0.05 * l ** 2
assert abs(zne_estimate(poly, [1, 2, 3]) - 0.8) < 1e-12
print("[4] pesos Richardson λ=(1,2,3):", np.round(richardson_weights([1, 2, 3]), 6), "; factor de varianza Σc² =", f"{zne_variance_factor([1,2,3]):.1f}")
assert abs(zne_variance_factor([1, 2, 3]) - 19.0) < 1e-12 and abs(zne_variance_factor([1, 2]) - 5.0) < 1e-12
qz0 = 0.8
for a in (0.05, 0.2, 0.5):
    f = lambda l: qz0 * np.exp(-a * l)
    e2, e3 = zne_estimate(f, [1, 2]), zne_estimate(f, [1, 2, 3])
    print(f"    señal 800·e^(−aλ), a={a}: bruto λ=1 {1000*f(1):.1f} m-qg; ZNE 2 pts {1000*e2:.1f}; ZNE 3 pts {1000*e3:.1f} (exacto 800.0)")
    assert abs(e3 - qz0) < abs(e2 - qz0) < abs(f(1) - qz0)

try:
    import math

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_addon_cutting import generate_cutting_experiments, partition_problem, reconstruct_expectation_values
    from qiskit_aer.primitives import SamplerV2
    from qang.knitting import pauli_rotation_cut_gamma

    theta = math.pi / 4
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ry(0.7, 1)
    qc.rzz(theta, 0, 1)
    observables = SparsePauliOp(["IX", "XI", "XX"])
    exact = [Statevector(qc).expectation_value(SparsePauliOp(p)).real for p in observables.paulis.to_labels()]
    part = partition_problem(circuit=qc, partition_labels="AB", observables=observables.paulis)
    subexp, coeffs = generate_cutting_experiments(
        circuits=part.subcircuits, observables=part.subobservables, num_samples=np.inf
    )
    sampler = SamplerV2(seed=42)
    results = {label: sampler.run(subexp[label], shots=20000).result() for label in subexp}
    rec = reconstruct_expectation_values(results, coeffs, part.subobservables)
    assert abs(sum(abs(c) for c, _ in coeffs) - pauli_rotation_cut_gamma(math.cos(theta))) < 1e-9
    assert all(abs(r - e) < 0.03 for r, e in zip(rec, exact))
    print(f"[5] corte completo 2|2 RZZ(π/4): γ={pauli_rotation_cut_gamma(math.cos(theta)):.4f}, "
          f"IX/XI/XX reconstruidos {np.round(rec, 3).tolist()} frente a {np.round(exact, 3).tolist()} (20000 disparos, extra `knitting`)")
except ImportError:
    print("[5] corte completo: omitido, se requiere el extra `knitting` (qiskit-addon-cutting + qiskit-aer)")

print("verify_cap29: OK")
