"""Tests for examples/qec_syndrome_hardware_qg.py (§88): the Leung-code
syndrome circuits, bit order and floor-corrected estimates (no account)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("scipy")

import qec_syndrome_hardware_qg as S  # noqa: E402


def test_noiseless_code_states_have_no_syndrome_flips():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    for logical in (0, 1):
        counts = sim.run(transpile(S.build(0.0, logical), sim), shots=300, seed_simulator=1).result().get_counts()
        assert S.flips(counts) == (0, 0, 300)


def test_flip_counting_bit_order():
    # key 'c2c1c0': c0 = Z0Z1, c1 = Z2Z3, c2 = XXXX
    assert S.flips({"001": 10, "010": 5, "100": 7, "111": 1}) == (10 + 5 + 2, 7 + 1, 23)
