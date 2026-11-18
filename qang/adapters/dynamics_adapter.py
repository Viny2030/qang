"""Dynamiqs / jaxquantum: lectura qg_Z (= <Z>) por qubit desde un estado simulado.

Solo NumPy. Acepta ket, matriz densidad o un objeto con ``to_numpy()``/``data``/``__array__``
(``dq.QArray``, ``jqt.Qarray``). Convencion: el qubit 0 es el factor tensorial mas
significativo (orden de ``dq.tensor`` y ``jqt`` ``^``), distinto del little-endian de Qiskit.
"""
from __future__ import annotations

import numpy as np


def _as_array(state) -> np.ndarray:
    for attr in ("to_numpy", "data"):
        obj = getattr(state, attr, None)
        if obj is not None:
            return np.asarray(obj() if callable(obj) else obj, dtype=complex)
    return np.asarray(state, dtype=complex)


def qg_z_from_state(state) -> np.ndarray:
    """qg_Z_i = <Z_i> para un estado de n qubits (ket o matriz densidad).

    Returns
    -------
    numpy.ndarray
        Vector real en [-1, 1]; ``1000 * qg_Z`` es la lectura en m-qg
        (qga es la escala angular, 1 vuelta = 1000 qga).
    """
    a = _as_array(state)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim == 1:
        p = np.abs(a) ** 2
    elif a.ndim == 2 and a.shape[0] == a.shape[1]:
        p = np.real(np.diag(a))
    else:
        raise ValueError(f"forma de estado no soportada: {a.shape}")
    dim = p.size
    n = int(round(np.log2(dim)))
    if 2 ** n != dim:
        raise ValueError("la dimension no es potencia de 2")
    p = p / p.sum()
    idx = np.arange(dim)
    return np.array([1.0 - 2.0 * p[((idx >> (n - 1 - i)) & 1) == 1].sum() for i in range(n)])
