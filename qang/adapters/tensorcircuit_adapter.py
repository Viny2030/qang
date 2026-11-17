"""TensorCircuit-NG: lectura qg_Z (= <Z>) por qubit desde un ``tc.Circuit``.

Import perezoso: este modulo no importa ``tensorcircuit``; solo usa el
metodo ``expectation_ps`` del objeto recibido (duck typing).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def qg_z_from_circuit(circuit: Any, qubits: Sequence[int] | None = None) -> np.ndarray:
    """qg_Z_i = <Z_i> para cada qubit de un circuito de TensorCircuit-NG.

    Parameters
    ----------
    circuit : tensorcircuit.Circuit
        Circuito (backend NumPy, JAX, ...) con ``expectation_ps`` y ``_nqubits``.
    qubits : sequence of int, optional
        Qubits a leer; por defecto todos.

    Returns
    -------
    numpy.ndarray
        Vector real con qg_Z en [-1, 1]. ``1000 * qg_Z`` es la lectura en m-qg
        (no confundir con qga, la escala angular: 1 vuelta = 1000 qga).
    """
    n = getattr(circuit, "_nqubits", None)
    if qubits is None:
        if n is None:
            raise ValueError("circuit sin _nqubits: pase `qubits` explicitamente")
        qubits = range(n)
    out = [float(np.real(np.asarray(circuit.expectation_ps(z=[int(q)])))) for q in qubits]
    return np.asarray(out)
