import pytest

from qang import probability as pb


def test_probability_aliases():
    assert pb.m_qg_from_p0 is pb.milliqang_from_p0
    assert pb.sigma_m is pb.shot_noise_milliqang
    assert pb.qg_tau is pb.dissipative_deficit


def test_gate_aliases():
    pytest.importorskip("qiskit")
    from qang import qiskit_gate

    assert qiskit_gate.R_qg is qiskit_gate.RQangGate

    pytest.importorskip("cirq")
    from qang import cirq_gate

    assert cirq_gate.R_qg is cirq_gate.rqang_gate

    pytest.importorskip("pennylane")
    from qang import pennylane_gate

    assert pennylane_gate.R_qg is pennylane_gate.rqang


@pytest.mark.parametrize(
    "module",
    [
        "qang.qscramble",
        "qang.postselection",
        "qang.qctc",
        "qang.qstring",
    ],
)
def test_analogy_only_demarcation(module):
    import importlib

    pytest.importorskip("jax")
    assert "ANALOGY ONLY" in importlib.import_module(module).__doc__


def test_pqc_is_explicitly_against():
    pytest.importorskip("jax")
    from qang import qdemarc

    assert "AGAINST (direct)" in qdemarc.__doc__
