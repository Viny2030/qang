"""
qang.integrations.qiskit_experiments — the heralded readout and temperature experiment of §31 (book, chapters 18
and 27).

A qubit at rest has residual excited population p, so it reads qg_eq = 1 - 2p = tanh(hf / 2 k_B T_eff), and its
readout is affine in qg: measured = a + b qg, with a = e10 - e01 and b = 1 - e01 - e10. The standard readout
calibration prepares "|0>" and "|1>", which are really the thermal state and X on it, so it counts the thermal
population as readout error (e01 is overestimated by p b) and the qubit always looks cold. Chapter 18 shows that
this makes a QRNG certifier unsafe.

``HeraldedReadout`` measures twice: herald, then I or X, then measure again (two circuits). From the t = 0 pairs,
in closed form,

    m   = E[r1]                       = a + b qg_eq
    V   = E[r1 r2 | I] - m^2          = b^2 (1 - qg_eq^2)
    C_X = E[r1 r2 | X]                = a^2 - b^2
    u = (m^2 - V - C_X) / (2m),  b = sqrt(V + u^2),  a = m - u,  qg_eq = u / b,

and the analysis returns e01, e10, the excited population p, qg_eq in m-qg and T_eff. ``heralded_closed_form``
does the same from counts, without the framework. The method needs mid-circuit measurement and assumes perfect
gates and a QND herald (§31 checks that a 5 % non-QND herald moves the estimates by ~0.001).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from . import require

require("qiskit_experiments", "experiments")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit_experiments.framework import AnalysisResultData, BaseAnalysis, BaseExperiment  # noqa: E402

__all__ = ["HeraldedReadout", "HeraldedReadoutAnalysis", "heralded_closed_form", "t_eff_from_qg"]

H_PLANCK, K_B = 6.62607015e-34, 1.380649e-23
_ORDER = ((+1, +1), (+1, -1), (-1, +1), (-1, -1))


def t_eff_from_qg(qg_eq: float, frequency: float) -> float:
    """T_eff = h f / (2 k_B artanh qg_eq); 0 for qg_eq >= 1, inf for qg_eq <= 0."""
    if qg_eq >= 1:
        return 0.0
    if qg_eq <= 0:
        return math.inf
    return H_PLANCK * frequency / (2 * K_B * math.atanh(qg_eq))


def _pairs(counts: Dict[str, int]) -> np.ndarray:
    """Qiskit counts of a 2-clbit circuit (clbit 0 = herald r1, clbit 1 = r2; key "c1c0") as an array over _ORDER."""
    out = np.zeros(4)
    for key, c in counts.items():
        k = key.replace(" ", "")
        r1, r2 = 1 - 2 * int(k[-1]), 1 - 2 * int(k[-2])
        out[_ORDER.index((r1, r2))] += c
    return out


def heralded_closed_form(counts_i, counts_x, frequency: Optional[float] = None) -> dict:
    """§31 closed form from the heralded I and X circuits (Qiskit counts dicts or arrays over (++, +-, -+, --))."""
    cI = counts_i if isinstance(counts_i, np.ndarray) else _pairs(counts_i)
    cX = counts_x if isinstance(counts_x, np.ndarray) else _pairs(counts_x)
    r1 = np.array([1, 1, -1, -1])
    prod = np.array([1, -1, -1, 1])
    m = (r1 @ cI + r1 @ cX) / (cI.sum() + cX.sum())
    V = prod @ cI / cI.sum() - m**2
    CX = prod @ cX / cX.sum()
    u = (m**2 - V - CX) / (2 * m)
    b = math.sqrt(max(V + u**2, 1e-12))
    a = m - u
    qg_eq = float(np.clip(u / b, -1.0, 1.0))
    out = {"a": float(a), "b": b, "e01": (1 - b - a) / 2, "e10": (1 - b + a) / 2, "qg_eq": qg_eq,
           "m_eq": 1000.0 * qg_eq, "p_excited": (1 - qg_eq) / 2}
    if frequency:
        out["T_eff"] = t_eff_from_qg(qg_eq, frequency)
    return out


def standard_readout(counts_i, counts_x) -> Tuple[float, float]:
    """(e01, e10) as the usual calibration reads them: from the second measurement only, without the herald."""
    cI = counts_i if isinstance(counts_i, np.ndarray) else _pairs(counts_i)
    cX = counts_x if isinstance(counts_x, np.ndarray) else _pairs(counts_x)
    r2_minus = np.array([0, 1, 0, 1])
    return float(r2_minus @ cI / cI.sum()), float(1 - r2_minus @ cX / cX.sum())


class HeraldedReadoutAnalysis(BaseAnalysis):
    """Closed-form analysis of HeraldedReadout (§31)."""

    @classmethod
    def _default_options(cls):
        opts = super()._default_options()
        opts.frequency = None
        return opts

    def _run_analysis(self, experiment_data):
        counts = {d["metadata"]["op"]: d["counts"] for d in experiment_data.data()}
        res = heralded_closed_form(counts["I"], counts["X"], self.options.frequency)
        e01_std, e10_std = standard_readout(counts["I"], counts["X"])
        res["e01_standard"], res["e10_standard"] = e01_std, e10_std
        return [AnalysisResultData(name=k, value=v) for k, v in res.items()], []


class HeraldedReadout(BaseExperiment):
    """Two circuits on one qubit: measure (herald), then I or X, then measure again."""

    def __init__(self, physical_qubits, backend=None, frequency: Optional[float] = None):
        super().__init__(physical_qubits, analysis=HeraldedReadoutAnalysis(), backend=backend)
        self.analysis.set_options(frequency=frequency)

    def circuits(self):
        out = []
        for op in ("I", "X"):
            qc = QuantumCircuit(1, 2)
            qc.measure(0, 0)
            if op == "X":
                qc.x(0)
            else:
                qc.id(0)
            qc.measure(0, 1)
            qc.metadata = {"op": op}
            out.append(qc)
        return out
