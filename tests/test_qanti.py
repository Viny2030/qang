import math
import numpy as np
from qang.qanti import (eigenstate, conjugate, bloch_mqg, dirac_h, self_conjugate_overlap, zb_qgz_mqg, zb_analytic_mqg,
                        lz_probability, pair_probability, lz_numeric, singlet_readings, shots_to_certify_branch, gamma_of)


def test_conjugation_flips_readings_and_energy():
    psi = eigenstate(0.7, 1.1, +1)
    chi = conjugate(psi)
    assert np.allclose(bloch_mqg(chi), -bloch_mqg(psi), atol=1e-9)
    assert abs(np.vdot(chi, dirac_h(0.7, 1.1) @ chi).real + math.hypot(0.7, 1.1)) < 1e-12


def test_no_self_conjugate_qubit():
    v = np.array([0.6, 0.8j])
    assert self_conjugate_overlap(v) < 1e-15


def test_zitterbewegung_formula():
    for t in (0.0, 0.3, 2.2):
        assert abs(zb_qgz_mqg(0.75, 1.0, t) - zb_analytic_mqg(0.75, 1.0, t)) < 1e-9


def test_pair_probability_equals_lz():
    assert abs(pair_probability(0.8, 1.0) - lz_probability(1.6, 2.0)) < 1e-15
    assert abs(lz_numeric(1.0, 2.0, 60.0 / math.sqrt(2.0), 200000) - lz_probability(1.0, 2.0)) < 3e-3


def test_singlet_and_shots():
    (z0, z1, zz), _ = singlet_readings()
    assert abs(z0) < 1e-12 and abs(z1) < 1e-12 and abs(zz + 1000) < 1e-9
    assert shots_to_certify_branch(2.0) == 27.0 and abs(gamma_of(0.75, 1.0) - 1.25) < 1e-12
