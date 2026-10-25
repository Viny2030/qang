"""
qang.integrations.qutip — qg readings of QuTiP states and open-system results (book, chapters 19 and 21).

  * ``qg_readings(state)``         per-qubit (qg_X, qg_Y, qg_Z) of a ket or density matrix of n qubits;
  * ``filtered_qg_z(state, w)``    the qg filter on a density matrix: (filtered qg_Z per qubit, kept fraction);
  * ``mesolve_qg(result)``         (times, array of qg_Z per qubit) from the states of a ``qutip.mesolve`` result;
  * ``t1_collapse(n, T1)``         the collapse operators of equal amplitude damping on every qubit.

Qubit 0 is the first factor of the tensor product, and |0> = basis(2, 0) has qg_Z = +1.
"""

from __future__ import annotations

import numpy as np

from . import require
from ._readings import filtered_qg_z_from_probs, qg_z_from_probs

qt = require("qutip", "qutip")

__all__ = ["qg_readings", "filtered_qg_z", "mesolve_qg", "t1_collapse"]


def _rho(state) -> np.ndarray:
    m = state.full()
    if state.isket:
        m = m @ m.conj().T
    return m


def _n_qubits(state) -> int:
    return len(state.dims[0])


def qg_readings(state) -> np.ndarray:
    """(n, 3) array of (qg_X, qg_Y, qg_Z) per qubit."""
    n = _n_qubits(state)
    rho = _rho(state)
    paulis = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    out = np.zeros((n, 3))
    for q in range(n):
        for k, p in enumerate(paulis):
            op = qt.tensor([p if j == q else qt.qeye(2) for j in range(n)]).full()
            out[q, k] = float(np.real(np.trace(rho @ op)))
    return out


def filtered_qg_z(state, weight: int):
    """The qg filter on the computational-basis distribution of ``state``."""
    n = _n_qubits(state)
    return filtered_qg_z_from_probs(np.real(np.diag(_rho(state))), n, weight)


def mesolve_qg(result):
    """(times, (len(times), n) array of qg_Z) from a mesolve/sesolve result that stored its states."""
    states = result.states
    n = _n_qubits(states[0])
    return np.asarray(result.times), np.array([qg_z_from_probs(np.real(np.diag(_rho(s))), n) for s in states])


def t1_collapse(n: int, T1: float):
    """sqrt(1/T1) sigma_minus on every qubit (sigma_minus takes |1> to |0>, so qg_Z relaxes to +1)."""
    lower = qt.basis(2, 0) * qt.basis(2, 1).dag()
    return [np.sqrt(1.0 / T1) * qt.tensor([lower if j == q else qt.qeye(2) for j in range(n)]) for q in range(n)]
