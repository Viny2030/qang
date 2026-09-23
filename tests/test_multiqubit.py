import math

import numpy as np
import pytest

from qang.core import Qang, qg_s_from_qg_z
from qang.multiqubit import (
    bell_state,
    ghz_state,
    joint_qg_s,
    joint_qg_s_from_counts,
    marginal_qg_s,
    marginal_von_neumann_entropy,
    partial_trace,
    per_qubit_qg_z,
    product_state,
    qg_correlation,
    shannon_entropy_miller_madow_bits,
    shannon_entropy_plugin_bits,
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
    from qang.multiqubit import _as_density

    rho_full = _as_density(psi, 2)
    traced = partial_trace(rho_full, n_qubits=2, keep=[0, 1])
    assert np.allclose(traced, rho_full, atol=1e-9)


def test_partial_trace_rejects_bad_indices():
    psi = bell_state("phi_plus")
    from qang.multiqubit import _as_density

    rho = _as_density(psi, 2)
    with pytest.raises(ValueError):
        partial_trace(rho, n_qubits=2, keep=[5])


# --------------------------------------------------------------------- #
# finite-shot qg_S: the plug-in Shannon entropy estimator and its
# Miller-Madow bias correction (see this module's docstring, "Finite-shot
# qg_S", and examples/quantum_volume_qg_s_finite_shots.py for the full
# shot-budget characterization).
# --------------------------------------------------------------------- #
def test_plugin_entropy_matches_hand_computed_value():
    # counts {00: 2, 01: 2}: p = [0.5, 0.5] -> exactly 1 bit
    assert shannon_entropy_plugin_bits({"00": 2, "01": 2}) == pytest.approx(1.0, abs=1e-9)
    # a single observed outcome has exactly zero entropy
    assert shannon_entropy_plugin_bits({"00": 7}) == pytest.approx(0.0, abs=1e-9)


def test_plugin_entropy_accepts_dict_or_plain_sequence_equivalently():
    from_dict = shannon_entropy_plugin_bits({"a": 10, "b": 5, "c": 3, "d": 2})
    from_seq = shannon_entropy_plugin_bits([10, 5, 3, 2])
    assert from_dict == pytest.approx(from_seq, abs=1e-12)


def test_plugin_entropy_rejects_empty_counts():
    with pytest.raises(ValueError):
        shannon_entropy_plugin_bits({})
    with pytest.raises(ValueError):
        shannon_entropy_plugin_bits([0, 0, 0])


def test_miller_madow_correction_is_zero_with_only_one_observed_outcome():
    """K_observed - 1 = 0 when only one outcome ever appears, so the
    Miller-Madow correction vanishes and matches the (also zero) plug-in
    entropy exactly."""
    counts = {"00": 50}
    assert shannon_entropy_miller_madow_bits(counts) == pytest.approx(
        shannon_entropy_plugin_bits(counts), abs=1e-12
    )


@pytest.mark.parametrize("counts", [{"a": 10, "b": 5, "c": 3, "d": 2}, {"a": 1, "b": 1}, {"a": 100, "b": 1, "c": 1}])
def test_miller_madow_is_never_smaller_than_plugin(counts):
    """The correction term (K-1)/(2*N*ln2) is >= 0 whenever more than one
    outcome is observed, so Miller-Madow can only push the estimate up,
    never down -- consistent with correcting a strictly negative bias."""
    assert shannon_entropy_miller_madow_bits(counts) >= shannon_entropy_plugin_bits(counts) - 1e-12


def test_miller_madow_reduces_bias_relative_to_plugin_at_small_sample_size():
    """The headline empirical claim: sampling repeatedly from a KNOWN
    8-outcome distribution, the plug-in estimator's mean is biased well
    below the true entropy at small shot counts, and Miller-Madow cuts
    that bias substantially (not merely nudges it) -- reproduced at two
    different shot budgets."""
    rng = np.random.default_rng(42)
    true_probs = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.02])
    true_entropy = -sum(p * math.log2(p) for p in true_probs if p > 0)

    for n_shots in (20, 50):
        plugins, mms = [], []
        for _ in range(3000):
            sample = rng.multinomial(n_shots, true_probs)
            counts = {i: int(c) for i, c in enumerate(sample) if c > 0}
            plugins.append(shannon_entropy_plugin_bits(counts))
            mms.append(shannon_entropy_miller_madow_bits(counts))

        bias_plugin = np.mean(plugins) - true_entropy
        bias_mm = np.mean(mms) - true_entropy

        assert bias_plugin < -0.05  # the plug-in estimator really is biased low here
        assert abs(bias_mm) < abs(bias_plugin) / 2.0  # Miller-Madow cuts the bias by more than half


def test_joint_qg_s_from_counts_converges_to_exact_joint_qg_s_at_large_shot_count():
    """With enough shots, the finite-sample estimate (with or without the
    Miller-Madow correction, which vanishes as N grows) should agree
    closely with the exact, infinite-shot joint_qg_s of the same state."""
    psi = bell_state("phi_plus")
    exact = joint_qg_s(psi, n_qubits=2, normalize=False)  # exactly 1.0 bit

    probs = np.array([0.5, 0.0, 0.0, 0.5])
    rng = np.random.default_rng(0)
    sample = rng.multinomial(200_000, probs)
    counts = {i: int(c) for i, c in enumerate(sample) if c > 0}

    estimate = joint_qg_s_from_counts(counts, n_qubits=2, normalize=False)
    assert estimate == pytest.approx(exact, abs=0.01)


def test_joint_qg_s_from_counts_rejects_unknown_bias_correction():
    with pytest.raises(ValueError):
        joint_qg_s_from_counts({"00": 5, "11": 5}, n_qubits=2, bias_correction="bogus")


# --------------------------------------------------------------------- #
# the direct qg_Z <-> qg_S identity, applied per-qubit (see qang.core's
# docstring and this module's "qg-native correlation" section), and
# qg_correlation, the always-non-negative gap it leaves open at the joint
# (whole-register) level.
# --------------------------------------------------------------------- #
def test_per_qubit_qg_z_and_marginal_qg_s_satisfy_the_exact_identity_on_random_haar_states():
    """marginal_qg_s[i] must equal qg_s_from_qg_z(per_qubit_qg_z[i]) exactly
    for every qubit of every state -- not just the canonical product/Bell
    examples above -- since both are computed from the very same
    partial-traced single-qubit diagonal. Checked against random Haar-random
    pure states (which reduce to genuinely MIXED single-qubit marginals
    whenever the qubits are entangled), not just hand-picked examples."""
    rng = np.random.default_rng(0)
    n_qubits = 3
    dim = 2 ** n_qubits
    for _ in range(20):
        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        vec = vec / np.linalg.norm(vec)
        qz = per_qubit_qg_z(vec, n_qubits)
        qs = marginal_qg_s(vec, n_qubits)
        for i in range(n_qubits):
            assert qs[i] == pytest.approx(qg_s_from_qg_z(qz[i]), abs=1e-9)


def test_ghz_state_matches_bell_state_at_n_equals_2():
    assert np.allclose(ghz_state(2), bell_state("phi_plus"))


def test_ghz_state_rejects_non_positive_n_qubits():
    with pytest.raises(ValueError):
        ghz_state(0)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_ghz_state_every_qubit_is_maximally_mixed(n_qubits):
    psi = ghz_state(n_qubits)
    profile = per_qubit_qg_z(psi, n_qubits)
    assert profile == pytest.approx([0.0] * n_qubits, abs=1e-9)


def test_qg_correlation_is_zero_for_product_states():
    """No correlation between independent qubits: the per-qubit qg_S
    budget exactly accounts for the joint qg_S, with nothing left over."""
    psi0 = _single_qubit_statevector(0.0)
    psi1 = _single_qubit_statevector(PI / 3)
    rho = product_state([psi0, psi1])
    assert qg_correlation(rho, n_qubits=2) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5, 6])
def test_qg_correlation_of_ghz_state_matches_closed_form(n_qubits):
    """Every qubit of an n-qubit GHZ state is individually maximally mixed
    (1 bit of marginal qg_S each, n bits total), while the joint outcome
    distribution carries only 1 bit (two equally likely outcomes) --
    leaving a gap of exactly n_qubits - 1 bits, for every n."""
    psi = ghz_state(n_qubits)
    assert qg_correlation(psi, n_qubits) == pytest.approx(float(n_qubits - 1), abs=1e-9)


def test_qg_correlation_is_never_negative_on_random_haar_states():
    """Subadditivity of Shannon entropy: this must hold for ANY state, not
    just the hand-picked product/GHZ examples above."""
    rng = np.random.default_rng(1)
    n_qubits = 3
    dim = 2 ** n_qubits
    for _ in range(30):
        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        vec = vec / np.linalg.norm(vec)
        assert qg_correlation(vec, n_qubits) >= -1e-9


# --------------------------------------------------------------------- #
# mean_qg_z / mean_qg_z_from_counts
# --------------------------------------------------------------------- #
def test_mean_qg_z_equals_average_of_per_qubit_profile_on_random_states():
    from qang.multiqubit import mean_qg_z, per_qubit_qg_z
    rng = np.random.default_rng(7)
    for n in (1, 2, 3, 4):
        psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        psi /= np.linalg.norm(psi)
        assert mean_qg_z(psi, n) == pytest.approx(np.mean(per_qubit_qg_z(psi, n)))


def test_mean_qg_z_of_all_zeros_is_one_and_of_ghz_is_zero():
    from qang.multiqubit import mean_qg_z
    zeros = np.zeros(8, dtype=complex)
    zeros[0] = 1.0
    assert mean_qg_z(zeros, 3) == pytest.approx(1.0)
    assert mean_qg_z(ghz_state(3), 3) == pytest.approx(0.0)


def test_mean_qg_z_from_counts_accepts_all_count_formats():
    from qang.multiqubit import mean_qg_z_from_counts
    assert mean_qg_z_from_counts({"000": 5, "111": 5}, 3) == pytest.approx(0.0)
    assert mean_qg_z_from_counts({"0 1": 3, 1: 1}, 2) == pytest.approx(0.0)
    assert mean_qg_z_from_counts([10, 0, 0, 0], 2) == pytest.approx(1.0)
    assert mean_qg_z_from_counts({"11": 4}, 2) == pytest.approx(-1.0)


def test_mean_qg_z_from_counts_converges_to_exact_value_and_is_unbiased():
    from qang.multiqubit import mean_qg_z, mean_qg_z_from_counts
    rng = np.random.default_rng(3)
    n = 3
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    probs = np.abs(psi) ** 2
    exact = mean_qg_z(psi, n)
    # tiny shot budget, many repetitions: the average estimate is unbiased
    estimates = [mean_qg_z_from_counts(rng.multinomial(5, probs), n) for _ in range(20000)]
    se = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - exact) < 5 * se


@pytest.mark.parametrize(
    "counts, n",
    [({"012": 1}, 3), ({"01": 1}, 3), ({4: 1}, 2), ({"00": -1}, 2), ({"00": 0}, 2)],
)
def test_mean_qg_z_from_counts_rejects_bad_input(counts, n):
    from qang.multiqubit import mean_qg_z_from_counts
    with pytest.raises(ValueError):
        mean_qg_z_from_counts(counts, n)
