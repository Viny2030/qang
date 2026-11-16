"""tools/verify_cap28.py — Cap. 28: testigo de electrones, verificación por número, PCA en peso 1, solapamientos, QAOA «exactamente K»."""
import numpy as np
import jax.numpy as jnp
from qang.qchem import *
from qang.qchem import _weights
from qang.qkernels import unary_inversion_test

# [1] Hubbard dímero: estructura y energía exacta
t, U = 1.0, 2.0
H, HZ, Hhop, N, Sz = hubbard_dimer(t, U)
comm = lambda A, B: np.max(np.abs(A @ B - B @ A))
assert comm(H, N) < 1e-12 and comm(H, Sz) < 1e-12
E0, psi = ground_state_sector(H, N, 2)
E_exact = (U - np.sqrt(U**2 + 16 * t**2)) / 2
assert abs(E0 - E_exact) < 1e-12
print(f"[1] [H,N]={comm(H,N):.1e} [H,Sz]={comm(H,Sz):.1e}; E0={E0:.12f} = (U−√(U²+16t²))/2 = {E_exact:.12f}")

# [2] testigo de electrones en m-qg
rho0 = np.outer(psi, psi.conj())
Nd = np.real(np.diag(N))
qz_ideal = np.array([1 - 2 * np.real(np.trace(rho0 @ np.diag(
    np.array([(z >> (3 - q)) & 1 for z in range(16)], dtype=float)))) for q in range(4)])
assert abs(witness_mqg(qz_ideal, 2)) < 1e-12 and abs(electron_count_from_qz(qz_ideal) - 2) < 1e-12
print(f"[2] lectura ideal por qubit (m-qg) = {np.round(1000*qz_ideal).astype(int)}; media ideal = {1000*mean_qz_ideal(4,2):.0f} m-qg")
for g in (0.02, 0.05, 0.10):
    r = apply_t1_all(rho0, g)
    occ = np.array([np.real(np.trace(r @ np.diag(np.array([(z >> (3 - q)) & 1 for z in range(16)], dtype=float)))) for q in range(4)])
    qz = 1 - 2 * occ
    w = witness_mqg(qz, 2)
    assert abs(w - 1000 * g) < 1e-9   # T1 sube la media exactamente γ·1000 m-qg si la ocupación media es 1/2
    print(f"    T1 γ={g:.2f}: testigo = {w:+.1f} m-qg (= 1000γ)")

# [3] filtro de Z (alcance) y verificación por número/paridad/mod4 (alcance total)
print("[3] error de energía (unidades de t) tras T1 sobre el fundamental de N=2 (U=2, t=1)")
P_N = sector_projector(N, [2])
P_par = sector_projector(N, [0], mod=2)
P_m4 = sector_projector(N, [2], mod=4)
for g in (0.02, 0.05, 0.10):
    r = apply_t1_all(rho0, g)
    rows = {}
    for name, P in (("crudo", None), ("paridad", P_par), ("N mod 4", P_m4), ("N exacto", P_N)):
        E, frac = filtered_energy(r, H, P)
        EZ, _ = filtered_energy(r, HZ, P)
        rows[name] = (E - E0, EZ - float(np.real(np.trace(rho0 @ HZ))), frac)
    # N mod 4 y N exacto devuelven el estado ideal: error 0 (exactitud T1 en N=2 de 4 qubits)
    assert abs(rows["N mod 4"][0]) < 1e-12 and abs(rows["N exacto"][0]) < 1e-12
    assert abs(rows["N exacto"][2] - (1 - g) ** 2) < 1e-12
    assert abs(rows["crudo"][0]) > abs(rows["paridad"][0]) > 0
    print(f"    γ={g:.2f}  crudo ΔE={rows['crudo'][0]:+.4f} (Z: {rows['crudo'][1]:+.4f})  paridad ΔE={rows['paridad'][0]:+.5f}"
          f" ({rows['paridad'][2]:.3f})  N mod4 ΔE={rows['N mod 4'][0]:+.1e} ({rows['N mod 4'][2]:.4f})  disparos=(1−γ)²={(1-g)**2:.4f}")
cc = check_costs(4)
print(f"    coste ancilla: paridad {cc['parity_cnot']} CNOT/1 ancilla; N mod 4 {cc['mod4_cnot']} CNOT/2 ancillas")

# [4] PCA en sector de peso 1
rng = np.random.default_rng(3)
X = rng.normal(size=(16, 5)) + np.array([3.0, 1.5, 0.8, 0.3, 0.1])
lam, V, rho = pca_unary(X)
rots, D = givens_decomposition(V)
assert len(rots) == 10
rho2 = apply_givens(rho, rots, D)
err_diag = np.max(np.abs(rho2 - np.diag(lam)))
assert err_diag < 1e-13 and abs(lam.sum() - 1) < 1e-13
qs = [r[2] for r in rots]
print(f"[4] PCA unario n=5, 16 datos: λ={np.round(lam,3)} (=p_exc de cada qubit), error={err_diag:.1e}, RBS={len(rots)}, "
      f"ángulos qga ∈ [{min(qs):.1f}, {max(qs):.1f}]")

# [5] solapamientos: recursos y fórmulas
x = rng.normal(size=5); x /= np.linalg.norm(x)
y = rng.normal(size=5); y /= np.linalg.norm(y)
F = float(np.dot(x, y) ** 2)
assert abs(float(unary_inversion_test(jnp.asarray(x), jnp.asarray(y), 5)) - F) < 1e-12
res = overlap_resources(5)
print(f"[5] F=(x·y)²={F:.6f} (inversión = exacta a 1e-12); recursos n=5: inversión {res['inversion_qubits']} qubits/0 ancillas, swap {res['swap_qubits']} qubits/1 ancilla; "
      f"swap lee P(anc=0)=(1+F)/2={(1+F)/2:.6f}")

# [6] QAOA «exactamente K»
n, K = 8, 3
c = random_cost(n, 7)
assert abs(mean_qz_ideal(n, K) - 0.25) < 1e-15
nfeas = int(np.sum(_weights(n) == K))
out = {}
for p in (1, 2):
    px, psx = qaoa_optimize(c, n, K, p, "xy", restarts=8, seed=1)
    pen_params, pen_state = qaoa_optimize(c, n, K, p, "x", penalty=3.0, restarts=8, seed=1)
    out[p] = (feasible_mass(psx, n, K), approx_ratio(psx, c, n, K), feasible_mass(pen_state, n, K), approx_ratio(pen_state, c, n, K))
    assert abs(out[p][0] - 1.0) < 1e-12
    assert out[p][2] < 1.0
    print(f"    p={p}: XY factible={out[p][0]:.6f} razón={out[p][1]:.3f} | penalización X factible={out[p][2]:.3f} razón(cond.)={out[p][3]:.3f}")
g_r = greedy_ratio(c, n, K)
print(f"    sector factible = C(8,3) = {nfeas}; qg_Z medio exigido = {1000*mean_qz_ideal(n,K):.0f} m-qg; voraz (intercambio) razón = {g_r:.3f}")
assert out[2][1] >= out[1][1] - 1e-9
print("verify_cap28: OK")
