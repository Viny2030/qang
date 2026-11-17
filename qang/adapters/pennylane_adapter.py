"""PennyLane: qg_Z por qubit desde un QNode que devuelve probabilidades."""
from __future__ import annotations


def qnode_qg_z(qnode, *args, **kw):
    import numpy as np
    from qang.integrations._readings import qg_z_from_probs
    p = np.asarray(qnode(*args, **kw), dtype=float)
    return qg_z_from_probs(p, int(np.log2(p.size)))
