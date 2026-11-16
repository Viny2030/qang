import numpy as np
from qang.qnano import (barrier_T, theta_from_p0, conductance_from_qg, qpc_T, recover_two_channels, theta_std_sqrtN, occupancy_qgz,
                        thermometry, readout_affine, nv_sigma_mqg, nv_photons_for)


def test_tunnel_amplitude():
    T = barrier_T(1.0)
    assert abs(np.cos(theta_from_p0(T) / 2) - np.sqrt(T)) < 1e-15 and abs(T - 2.85e-3) < 1e-5


def test_landauer_identity_and_two_channels():
    assert all(abs(conductance_from_qg(qpc_T(v)) - qpc_T(v).sum()) < 1e-14 for v in np.linspace(0, 6, 50))
    r = recover_two_channels(0.4031 + 0.9716, 0.4031 * 0.5969 + 0.9716 * 0.0284)
    assert abs(r[0] - 0.9716) < 1e-12 and abs(r[1] - 0.4031) < 1e-12


def test_arcsine_variance_stabilized():
    assert all(1.0 <= theta_std_sqrtN(t, 400) < 1.02 for t in (0.05, 0.3, 0.5, 0.9))


def test_fermi_tanh_and_thermometry_loses():
    x = np.linspace(-6, 6, 50)
    assert np.max(np.abs((2 / (1 + np.exp(x)) - 1) - occupancy_qgz(x))) < 1e-15
    (ml, _), (ar, _) = thermometry(reps=60, seed=5)
    assert abs(ml - 1) < 0.02 and ar < 0.995


def test_readout_and_nv():
    assert np.allclose(readout_affine([1000, -1000], 0.02, 0.08), [960, -840])
    assert abs(nv_sigma_mqg(0.3, nv_photons_for(10, 0.3)) - 10) < 1e-9
