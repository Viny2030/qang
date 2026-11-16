import numpy as np
from qang.qchar import fit_t1, t1_qz, optimal_delay_ratio, correct_mqg, fidelity_from_hop, zz_phase_qga
def test_t1_fit(): ts = np.array([20, 60, 120.0]); assert abs(fit_t1(ts, t1_qz(ts, 90.0)) - 90.0) < 1e-9
def test_optimal_delay(): assert abs(optimal_delay_ratio() - 1.5936) < 1e-3
def test_readout(): m = (1 - 0.02 - 0.06) * 600 + 1000 * (0.06 - 0.02); assert abs(correct_mqg(m, 0.02, 0.06) - 600) < 1e-9
def test_hop_fidelity(): assert abs(fidelity_from_hop(0.5)) < 1e-15
def test_zz_qga(): assert abs(zz_phase_qga(2 * np.pi * 0.2, 1.3) - 260.0) < 1e-9
