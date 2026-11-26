"""tools/verify_p2_filter_rule.py — pre-registered check of the cap-21 MSE rule."""
from __future__ import annotations

import math

import numpy as np

from qang.qfalsify import decide, register, shots_for_power, verify_registration
from qang.qml import WeightQNN, kept_fraction
from qang.sectors import filter_distribution

N_QUBITS = 5
WEIGHT = 2
LAYERS = 3
SHOTS = 1_000
REPS = 300
CONFIGS = 120
GAMMAS = (0.08, 0.12, 0.16)
P0 = 0.90
ALPHA = 0.05
SEED = 20261125
THRESHOLD = 0.95


def rule(z_filtered: float, z_raw: float, kept: float, shots: int) -> tuple[bool, float, float]:
    mse_f = (1.0 - z_filtered**2) / (kept * shots)
    mse_r = (z_raw - z_filtered) ** 2 + (1.0 - z_raw**2) / shots
    return bool(mse_f < mse_r), mse_f, mse_r


prediction = {
    "id": "P2-FILTER-RULE-01",
    "claim": "cap-21 MSE rule picks the lower-MSE readout for K*S >= 5",
    "metric": "agreement(rule, empirical MSE winner)",
    "threshold": THRESHOLD,
    "direction": ">",
    "null": P0,
    "alpha": ALPHA,
    "seed": SEED,
    "script": "tools/verify_p2_filter_rule.py",
    "configs": CONFIGS,
    "reps": REPS,
    "shots": SHOTS,
    "gammas": list(GAMMAS),
    "n_qubits": N_QUBITS,
    "weight": WEIGHT,
    "layers": LAYERS,
}
digest = register(prediction)
assert verify_registration(prediction, digest)
assert digest != register(dict(prediction, threshold=P0))
print(f"[0] pre-registration SHA-256 = {digest[:16]}…")

rng = np.random.default_rng(SEED)
model = WeightQNN(n_qubits=N_QUBITS, weight=WEIGHT, layers=LAYERS)
successes = 0
worst_kept_shots = float("inf")
ratio_filter_wins = 0

for i in range(CONFIGS):
    gamma = GAMMAS[i % len(GAMMAS)]
    theta = rng.uniform(-math.pi, math.pi, size=model.n_theta)
    x = rng.normal(size=(1, N_QUBITS - 1))
    psi = model.encode(x)
    p_ideal = model.probs(theta, psi)[0]
    p_noisy = model.probs(theta, psi, gamma=gamma)[0]

    z_target = float(np.mean(p_ideal @ model.features))
    p_filtered, kept_measured = filter_distribution(p_noisy, N_QUBITS, WEIGHT)
    z_filtered = float(np.mean(p_filtered @ model.features))
    z_raw = float(np.mean(p_noisy @ model.features))
    kept = kept_fraction(gamma, WEIGHT, model.depth)
    assert abs(kept_measured - kept) < 1e-12
    assert abs(z_filtered - z_target) < 1e-12

    use_filter, _, _ = rule(z_filtered, z_raw, kept, SHOTS)
    worst_kept_shots = min(worst_kept_shots, kept * SHOTS)
    sq_filtered = 0.0
    sq_raw = 0.0
    for _ in range(REPS):
        counts = rng.multinomial(SHOTS, p_noisy / p_noisy.sum())
        p_hat = counts / SHOTS
        p_hat_filtered, _ = filter_distribution(p_hat, N_QUBITS, WEIGHT)
        z_f_hat = float(np.mean(p_hat_filtered @ model.features))
        z_r_hat = float(np.mean(p_hat @ model.features))
        sq_filtered += (z_f_hat - z_target) ** 2
        sq_raw += (z_r_hat - z_target) ** 2

    empirical_filter = sq_filtered < sq_raw
    ratio_filter_wins += int(empirical_filter)
    successes += int(use_filter == empirical_filter)

agreement = successes / CONFIGS
decision = decide(successes, CONFIGS, P0, ALPHA)
print(f"[1] configurations={CONFIGS}, reps={REPS}, shots={SHOTS}, min(K*S)={worst_kept_shots:.1f}")
print(f"[2] filter had lower empirical MSE in {ratio_filter_wins}/{CONFIGS} configurations")
print(f"[3] rule agreement={agreement:.3f} ({successes}/{CONFIGS}); preregistered p0={P0:.2f} -> {decision}")
print(f"[4] n needed for 90% power at p0={P0:.2f}, p1={THRESHOLD:.2f}: {shots_for_power(P0, THRESHOLD, ALPHA, 0.9)}")
assert decision == "confirmada"
print("verify_p2_filter_rule: OK")
