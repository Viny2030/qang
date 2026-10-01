"""
Prints the qg action of the 15 main gates (qg'_P = qg_{U^dag P U}).

The formulation now lives in the library: ``qang.formulation`` (theory in
manuscript/teoria_es, tests in tests/test_formulation.py). This script is kept
as a quick printout.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang.formulation import *  # noqa: E402,F401,F403
from qang.formulation import GATES_1Q, GATES_MULTI, conjugate, conserves_weight, rx, ry, rz  # noqa: E402

import numpy as np  # noqa: E402


def main():
    print("Gates: qg'_P = qg_{U^dag P U}")
    for name, U in GATES_1Q.items():
        print(f"  {name:7s}", {p: conjugate(U, p) for p in "XYZ"})
    for name, U in (("Rx(t)", rx(0.7)), ("Ry(t)", ry(0.7)), ("Rz(t)", rz(0.7))):
        print(f"  {name:7s} t=0.7", {p: conjugate(U, p) for p in "XYZ"})
    for name, U in GATES_MULTI.items():
        n = int(np.log2(U.shape[0]))
        labels = ["".join(c if j == q else "I" for j in range(n)) for q in range(n) for c in "XZ"]
        print(f"  {name:7s}", {p: conjugate(U, p) for p in labels}, "| conserves Hamming weight:", conserves_weight(U, n))


if __name__ == "__main__":
    main()
