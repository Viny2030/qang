"""
Load scikit-learn and qiskit-aer before anything else. On Linux each bundles its
own libgomp, and each copy needs a slot in glibc's static TLS block; once qiskit
and scipy have loaded their shared libraries the block can be full, and the late
import fails with "cannot allocate memory in static TLS block". The CI workflow
also enlarges the block (GLIBC_TUNABLES); loading both here first covers local
Linux runs without it.
"""

for _mod in ("sklearn", "qiskit_aer"):
    try:
        __import__(_mod)
    except ImportError:  # both are optional; the tests that need them skip themselves
        pass
