"""
Load scikit-learn before anything else. On Linux its bundled libgomp needs a
slot in the static TLS block; once qiskit, scipy and Aer have loaded their own
shared libraries the block can be full, and every later `import sklearn` fails
with "cannot allocate memory in static TLS block".
"""

try:
    import sklearn  # noqa: F401
except ImportError:  # sklearn is optional; the tests that need it skip themselves
    pass
