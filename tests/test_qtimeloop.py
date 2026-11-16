import math
import numpy as np
from qang.qtimeloop import (pctc_fisher, pctc_p_post, pctc_amplification, pctc_theta_out, pctc_success, deutsch_iterate, bloch_mqg,
                            bloch_rotation, fixed_dimension, shots_for_successes, I2, SX)


def test_total_fisher_is_one():
    for a, th in ((0.5, 0.3), (0.2, 2.5), (0.9, 1.0)):
        assert abs(pctc_fisher(a, th)[3] - 1.0) < 1e-12


def test_south_pole_amplification_and_cost():
    assert abs(pctc_amplification(0.5, math.pi) - 2.0) < 1e-12 and abs(pctc_p_post(0.5, math.pi) - 0.25) < 1e-12


def test_theta_map_matches_operator():
    th, a = 1.9, 0.5
    out = np.array([math.cos(th / 2), a * math.sin(th / 2)])
    assert abs(2 * math.atan2(out[1], out[0]) - pctc_theta_out(a, th)) < 1e-12


def test_grandfather():
    U = np.kron(I2, SX)
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    assert abs(pctc_success(U, rho)) < 1e-15
    c, o = deutsch_iterate(U, rho)
    assert np.max(np.abs(bloch_mqg(c))) < 1e-12


def test_fixed_dimension_and_shots():
    assert fixed_dimension(bloch_rotation([0, 0, 1], 0)) == 3 and fixed_dimension(bloch_rotation([0, 0, 1], 250)) == 1
    assert shots_for_successes(1000, 0.25) == 4000
