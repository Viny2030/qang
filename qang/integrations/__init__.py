"""
qang.integrations — qang as a module inside other libraries (book, chapter 35).

Each submodule plugs the qang readings (qg_Z, m-qg, the qang angle) and the
qg filter (post-selection on Hamming weight, ``qang.sectors``) into one
library, in that library's own terms:

  =========================  =================================================
  submodule                  what it adds
  =========================  =================================================
  ``mitiq``                  weight post-selection as a mitiq executor, ZNE of
                             filtered readings, the closed-form filter rule
                             (chapter 21)
  ``qiskit_experiments``     the heralded readout/temperature experiment of
                             §31 (chapters 18 and 27)
  ``qiskit_dynamics``        pulse-level qg readings and the pulse-area rule of
                             §160
  ``qutip``                  qg readings of Qobj states and mesolve results,
                             the filter on density matrices (chapters 19, 21)
  ``stim``                   detector and measurement readings in qg, the
                             filter on sampled records (chapters 26-27)
  ``sklearn``                WeightQNN / MultiClassQNN as scikit-learn
                             classifiers (chapter 20)
  ``jax``                    jit/grad-able readings and filtered readout
                             (chapter 20, 23)
  ``torch``                  a weight-conserving PennyLane TorchLayer with the
                             filtered readout (chapter 20)
  ``qibo``, ``pytket``,      the qang rotation and the canonical, filtered
  ``braket``, ``cudaq``      readout on each SDK's own circuits and results
                             (chapter 32)
  =========================  =================================================

Importing ``qang`` never imports any of these libraries; importing one of
these submodules imports only its library, and raises a clear ImportError
naming the extra to install (``pip install "qang[mitiq]"`` and so on).
"""

__all__ = [
    "mitiq",
    "qiskit_experiments",
    "qiskit_dynamics",
    "qutip",
    "stim",
    "sklearn",
    "jax",
    "torch",
    "qibo",
    "pytket",
    "braket",
    "cudaq",
]


def require(module: str, extra: str):
    """Import ``module`` or raise an ImportError that names the qang extra to install."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only when the library is absent
        raise ImportError(
            f"this qang integration needs {module!r}. Install it with "
            f'`pip install "qang[{extra}]"`.'
        ) from exc
