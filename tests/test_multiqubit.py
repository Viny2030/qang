import math

import numpy as np
import pytest

from quang.core import Qang
from quang.multiqubit import (
    bell_state,
    joint_qg_s,
    marginal_qg_s,
    marginal_von_neumann_entropy,
    partial_trace,
    per_qubit_qg_z,
    product_state,
)

PI = math.pi


def _single_qubit_statevector(theta, phi=0.0):
    q = Qang.from_angles(theta, phi=phi, mode="polar")
    return np.array(q.to_statevector(), dtype=complex)


# --------------------------------------------------------------------- #
# n_qubits == 1 must reduce exactly to the single-qubit definitions
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.0, PI / 2, PI, PI / 4, 1.2])
def test_single_qubit_reduces_to_core_qg_z(theta):
    psi = _single_qubit_statevector(theta)
    got = per_qubit_qg_z(psi, n_qubits=1)
    expected = Qang.from_angles(theta, mode="polar").value
    assert len(got) == 1
    assert got[0] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("theta", [0.0, PI / 2, PI, PI / 4, 1.2])
def test_single_qubit_reduces_to_core_qg_s(theta):
    psi = _single_qubit_statevector(theta)
    got = marginal_qg_s(psi, n_qubits=1)
    expected = Qang.from_angles(theta, mode="entropic").value
    assert got[0] == pytest.approx(expected, abs=1e-6)
    # joint entropy of a single qubit is the same quantity, normalized
    assert joint_qg_s(psi, n_qubits=1) == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------- #
# product states: independent qubits, zero entanglement
# --------------------------------------------------------------------- #
def test_product_state_per_qubit_qg_z_matches_components():
    psi0 = _single_qubit_statevector(PI / 3)   # qg_Z = cos(pi/3) = 0.5
    psi1 = _single_qubit_statevector(2.0)      # some other qg_Z
    rho = product_state([psi0, psi1])
    profile = per_qubit_qg_z(rho, n_qubits=2)
    assert profile[0] == pytest.approx(0.5, abs=1e-6)
    assert profile[1] == pytest.approx(math.cos(2.0), abs=1e-6)


def test_product_state_qubits_are_individually_pure():
    psi0 = _single_qubit_statevector(0.7)
    psi1 = _single_qubit_statevector(1.9)
    rho = product_state([psi0, psi1])
    marg = marginal_von_neumann_entropy(rho, n_qubits=2)
    assert marg[0] == pytest.approx(0.0, abs=1e-6)
    assert marg[1] == pytest.approx(0.0, abs=1e-6)


def test_product_state_joint_qg_s_equals_average_of_marginals():
    # |0> tensor |+>: qubit 0 is deterministic (qg_S=0), qubit 1 is maximal (qg_S=1)
    psi0 = _single_qubit_statevector(0.0)
    psi1 = _single_qubit_statevector(PI / 2)
    rho = product_state([psi0, psi1])
    marg = marginal_qg_s(rho, n_qubits=2)
    assert marg == pytest.approx([0.0, 1.0], abs=1e-6)
    # joint (normalized) entropy of an independent product is the mean of the marginals
    assert joint_qg_s(rho, n_qubits=2, normalize=True) == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------- #
# Bell state: the entanglement witness this module was built to show
# --------------------------------------------------------------------- #
def test_bell_state_is_globally_pure_but_locally_maximally_mixed():
    psi = bell_state("phi_plus")

    # globally: joint measurement always gives 00 or 11 with equal probability
    assert joint_qg_s(psi, n_qubits=2, normalize=False) == pytest.approx(1.0, abs=1e-6)

    # each qubit's own reduced state is maximally mixed: qg_Z = 0 for both
    profile = per_qubit_qg_z(psi, n_qubits=2)
    assert profile[0] == pytest.approx(0.0, abs=1e-6)
    assert profile[1] == pytest.approx(0.0, abs=1e-6)

    # ... and each qubit's own marginal *von Neumann* entropy is maximal (1 bit),
    # even though the GLOBAL 2-qubit state is exactly pure (entropy 0 as a whole
    # 4-dim density matrix). This gap -- each part maximally mixed, the whole
    # exactly pure -- is impossible for any classical (unentangled) system and
    # is the standard witness that per_qubit_qg_z / marginal_von_neumann_entropy
    # were built to expose.
    marg_vn = marginal_von_neumann_entropy(psi, n_qubits=2)
    assert marg_vn[0] == pytest.approx(1.0, abs=1e-6)
    assert marg_vn[1] == pytest.approx(1.0, abs=1e-6)


def test_partial_trace_of_full_system_is_identity_map():
    psi = bell_state("psi_minus")
    from quang.multiqubit import _as_density

    rho_full = _as_density(psi, 2)
    traced = partial_trace(rho_full, n_qubits=2, keep=[0, 1])
    assert np.allclose(traced, rho_full, atol=1e-9)


def test_partial_trace_rejects_bad_indices():
    psi = bell_state("phi_plus")
    from quang.multiqubit import _as_density

    rho = _as_density(psi, 2)
    with pytest.raises(ValueError):
        partial_trace(rho, n_qubits=2, keep=[5])
