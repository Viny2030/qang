"""tools/verify_exp28_1.py — Exp. 28.1: testigo de electrones bajo T1 con lectura por disparos."""
from __future__ import annotations

import numpy as np

from qang.interop import qg_z_profile, weight_filter
from qang.qchem import (
    apply_t1_all,
    filtered_energy,
    ground_state_sector,
    hubbard_dimer,
    mean_qz_ideal,
    sector_projector,
    witness_mqg,
)

N_QUBITS = 4
N_ELECTRONS = 2
T, U = 1.0, 2.0
SHOTS = 20_000
GAMMAS = (0.02, 0.05, 0.10)
SEEDS = (11, 12, 13)


def _bits(z: int) -> tuple[int, ...]:
    return tuple((z >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS))


def sample_counts(p: np.ndarray, shots: int, seed: int) -> dict[tuple[int, ...], int]:
    outcomes = np.random.default_rng(seed).choice(len(p), size=shots, p=p)
    counts: dict[tuple[int, ...], int] = {}
    for z in outcomes:
        key = _bits(int(z))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _witness_and_se(counts):
    prof = qg_z_profile(counts)
    qz = [q for q, _ in prof]
    se_mean = float(np.sqrt(sum(e * e for _, e in prof))) / len(prof)
    return witness_mqg(qz, N_ELECTRONS), 1000.0 * se_mean


def run(gamma: float, shots: int, seed: int) -> dict:
    H, HZ, Hhop, N, Sz = hubbard_dimer(T, U)
    E0, psi = ground_state_sector(H, N, N_ELECTRONS)
    rho = apply_t1_all(np.outer(psi, psi.conj()), gamma, N_QUBITS)
    p = np.clip(np.real(np.diag(rho)), 0.0, None)
    p /= p.sum()
    counts = sample_counts(p, shots, seed)
    w_raw, se_raw = _witness_and_se(counts)
    kept, frac_obs = weight_filter(counts, N_ELECTRONS)
    w_filt, se_filt = _witness_and_se(kept) if kept else (float("nan"), float("nan"))
    E_filt, frac_exact = filtered_energy(rho, H, sector_projector(N, [N_ELECTRONS]))
    return {
        "gamma": gamma,
        "E0": E0,
        "E_filt": E_filt,
        "w_exact": 1000.0 * gamma,
        "w_raw": w_raw,
        "se_raw": se_raw,
        "w_filt": w_filt,
        "se_filt": se_filt,
        "frac_obs": frac_obs,
        "frac_exact": frac_exact,
        "se_frac": float(np.sqrt(frac_obs * (1 - frac_obs) / shots)),
    }


def main() -> None:
    print("Exp. 28.1 — testigo de electrones bajo T1, Hubbard dímero (4 qubits, t=1, U=2)")
    print(f"qg_Z ideal medio = {mean_qz_ideal(N_QUBITS, N_ELECTRONS):+.1f} m-qg; W = 1000(y - qg_Z_ideal)")
    for gamma in GAMMAS:
        rows = [run(gamma, SHOTS, s) for s in SEEDS]
        avg = {k: float(np.mean([r[k] for r in rows])) for k in ("w_raw", "w_filt", "frac_obs")}
        se_avg = float(np.sqrt(sum(r["se_raw"] ** 2 for r in rows))) / len(rows)
        w_exact = rows[0]["w_exact"]
        frac_exact = rows[0]["frac_exact"]
        assert abs(avg["w_raw"] - w_exact) < 5 * se_avg
        assert abs(avg["w_filt"]) < 5 * se_avg
        assert abs(avg["frac_obs"] - frac_exact) < 5 * rows[0]["se_frac"]
        assert abs(rows[0]["E_filt"] - rows[0]["E0"]) < 1e-12
        print(f"  y={gamma:.2f}: W exacto={w_exact:+6.1f}  W crudo={avg['w_raw']:+6.1f} ± {se_avg:.1f} m-qg  "
              f"W filtrado={avg['w_filt']:+5.1f} m-qg  kept={avg['frac_obs']:.4f} (exacto {frac_exact:.4f})  "
              f"E filtrada={rows[0]['E_filt']:.10f} = E0")
    print("verify_exp28_1: OK")


if __name__ == "__main__":
    main()
