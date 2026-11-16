import pytest; pytest.importorskip("scipy")
import numpy as np
from qang.qecc import recovery_operators, avg_fidelity_encoded, avg_fidelity_bare, rep_logical_error_flip, rep_logical_error_erasure, richardson_weights, zne_variance_factor
def test_cptp(): assert np.max(np.abs(sum(r.conj().T @ r for r in recovery_operators()) - np.eye(16))) < 1e-12
def test_ad_code_helps(): assert 1 - avg_fidelity_encoded(0.01) < (1 - avg_fidelity_bare(0.01)) / 10
def test_erasure_beats_flip(): assert rep_logical_error_erasure(3, 0.01) < rep_logical_error_flip(3, 0.01) / 100
def test_zne_cost(): assert abs(zne_variance_factor([1, 2, 3]) - 19.0) < 1e-12 and np.allclose(richardson_weights([1, 2, 3]), [3, -3, 1])
