"""
qang.integrations.cudaq — qang on NVIDIA CUDA-Q (book, chapter 32).

CUDA-Q runs the same circuits on CPU and NVIDIA GPU simulators (``cudaq.set_target("nvidia")``) and on hardware
providers. This module adds the qang rotation to its kernel builder and reads its sample results:

  * ``rqang(kernel, qubit, qg)``        ``kernel.ry(arccos(qg), qubit)`` on a ``cudaq.make_kernel()`` builder;
  * ``canonical_counts(result)``        {(b_0, ..., b_{n-1}): count} from a ``cudaq.SampleResult``
                                        (CUDA-Q bitstrings put qubit 0 first);
  * ``qg_readout(result, weight)``      qg_Z profile, and the filtered profile with its kept fraction.

CUDA-Q ships wheels for Linux (and WSL on Windows) and macOS, not for native Windows.
"""

from __future__ import annotations

from typing import Optional

from . import require
from ._sdk import readout, theta_from_qg

cudaq = require("cudaq", "cudaq")

__all__ = ["rqang", "canonical_counts", "qg_readout"]


def rqang(kernel, qubit, qg: float):
    kernel.ry(theta_from_qg(qg), qubit)
    return kernel


def canonical_counts(result) -> dict:
    return {tuple(int(b) for b in bits): int(c) for bits, c in result.items()}


def qg_readout(result, weight: Optional[int] = None) -> dict:
    return readout(canonical_counts(result), weight)
