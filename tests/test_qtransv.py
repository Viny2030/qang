import math
import numpy as np
from qang.qtransv import (anticommutation_error, exchange_qgx, hahn_echo_ensemble, hahn_echo_qgx_drift, tfim_mx_mqg, ab_qgz,
                          wigner_rotation, fresnel, tir_phase_difference, vacuum_weight, gudermannian)


def test_jw_and_exchange_sign():
    assert anticommutation_error(3) < 1e-15 and abs(exchange_qgx(True) + 1) < 1e-15 and abs(exchange_qgx(False) - 1) < 1e-15


def test_hahn_echo():
    st = 2 * math.pi * 50e3 * 20e-6
    assert abs(hahn_echo_ensemble(st, 0.0) - 1) < 1e-12 and abs(hahn_echo_ensemble(st, 0.02) - 0.99901) < 1e-5
    assert abs(hahn_echo_qgx_drift(st, 0.1) - 0.8209) < 1e-4


def test_tfim_and_ab():
    assert abs(tfim_mx_mqg(4) - 810.1) < 0.06
    assert np.max(np.abs(ab_qgz(np.linspace(0, 2, 41)) - ab_qgz(np.linspace(0, 2, 41) + 1))) < 1e-12


def test_wigner():
    num, frm = wigner_rotation([0.6, 0, 0], [0, 0.8, 0])
    assert abs(num - frm) < 1e-12 and abs(num - 0.3303) < 5e-5


def test_fresnel_and_unruh():
    assert abs(fresnel(1.0, 1.5, math.atan(1.5))[1]) < 1e-15
    assert abs(math.sin(abs(tir_phase_difference(1.5, math.radians(45)))) - 0.6) < 1e-9
    assert abs(math.acos(math.sqrt(vacuum_weight(1.0))) - gudermannian(1.0)) < 1e-14
