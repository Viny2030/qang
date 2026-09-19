import math

import numpy as np
import pytest

from quang.core import Qang
from quang.mixed import (
    density_from_statevector,
    is_valid_density_matrix,
    is_valid_povm,
    mix,
    povm_outcome_probabilities,
    purity,
    qg_s_povm,
    qg_z_density,
    standard_z_povm,
    trine_povm,
    von_neumann_entropy,
)

PI = math.pi


def _pure_state_density(theta, phi=0.0):
    q = Qang.from_angles(theta, phi=phi, mode="polar")
    alpha, beta = q.to_statevector()
    return density_from_statevector([alpha, beta])


# --------------------------------------------------------------------- #
# qg_z_density reduces exactly to qg_Z for pure states
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.0, PI / 2, PI, PI / 4, 1.7])
def test_qg_z_density_matches_pure_state_qg_z(theta):
    rho = _pure_state_density(theta)
    expected = Qang.from_angles(theta, mode="polar").value
    assert qg_z_density(rho) == pytest.approx(expected, abs=1e-9)


def test_qg_z_density_requires_2x2():
    with pytest.raises(ValueError):
        qg_z_density(np.eye(4))


# --------------------------------------------------------------------- #
# von Neumann entropy: 0 for pure states, > 0 for genuinely mixed states
# (this is exactly the quantity the paper's Section 2.2 says qg_S is NOT)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.0, PI / 2, PI, 0.9])
def test_von_neumann_entropy_zero_for_pure_states(theta):
    rho = _pure_state_density(theta)
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-9)
    assert purity(rho) == pytest.approx(1.0, abs=1e-9)


def test_von_neumann_entropy_maximal_for_maximally_mixed_state():
    maximally_mixed = np.eye(2, dtype=complex) / 2.0
    assert von_neumann_entropy(maximally_mixed) == pytest.approx(1.0, abs=1e-9)
    assert purity(maximally_mixed) == pytest.approx(0.5, abs=1e-9)


def test_von_neumann_entropy_of_a_mixture_is_strictly_between():
    rho0 = _pure_state_density(0.0)   # |0><0|
    rho1 = _pure_state_density(PI)    # |1><1|
    rho_mix = mix(rho0, rho1, p=0.5)  # classical 50/50 mixture, NOT a superposition
    assert is_valid_density_matrix(rho_mix)
    # qg_Z of the mixture matches the classical average (0.5*1 + 0.5*(-1) = 0)
    assert qg_z_density(rho_mix) == pytest.approx(0.0, abs=1e-9)
    # but its entropy is maximal, unlike the pure equatorial superposition |+>
    assert von_neumann_entropy(rho_mix) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------- #
# POVM generalization: reduces to standard qg_S at the binary Z-basis POVM
# --------------------------------------------------------------------- #
def test_standard_z_povm_is_valid():
    assert is_valid_povm(standard_z_povm())


@pytest.mark.parametrize("theta", [0.0, PI / 2, PI, PI / 4, 1.1])
def test_qg_s_povm_matches_qg_s_on_standard_z_povm(theta):
    rho = _pure_state_density(theta)
    expected = Qang.from_angles(theta, mode="entropic").value
    got = qg_s_povm(rho, standard_z_povm(), normalize=True)
    assert got == pytest.approx(expected, abs=1e-6)


def test_povm_outcome_probabilities_sum_to_one():
    rho = _pure_state_density(0.6)
    probs = povm_outcome_probabilities(rho, standard_z_povm())
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_trine_povm_is_valid_and_gives_bounded_entropy():
    assert is_valid_povm(trine_povm())
    rho = _pure_state_density(PI / 2)  # maximal-uncertainty state
    h_raw = qg_s_povm(rho, trine_povm(), normalize=False)
    h_norm = qg_s_povm(rho, trine_povm(), normalize=True)
    assert 0.0 <= h_norm <= 1.0 + 1e-9
    assert h_raw == pytest.approx(h_norm * math.log2(3), abs=1e-9)


def test_povm_must_sum_to_identity_on_rho():
    rho = _pure_state_density(0.3)
    bad_povm = [np.array([[0.5, 0], [0, 0.5]], dtype=complex)]  # doesn't sum to identity
    with pytest.raises(ValueError):
        povm_outcome_probabilities(rho, bad_povm)
