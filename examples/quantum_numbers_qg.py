"""
The four quantum numbers of an electron in qg terms (§155). Exact, no noise, NumPy only.

python examples/quantum_numbers_qg.py

Prints (1) the qang angle arccos(m / l) of |l, m> next to the textbook
vector-model angle arccos(m / sqrt(l (l + 1))) and the transverse spread that
separates them; (2) that a spin-coherent state of 2j qubits has qg_Z = cos
theta for every j; (3) the m_l content of the five p-orbital states (the real
p_x, p_y mix the weight sectors 0 and 2); (4) the l spectrum of the SO(4)
register of hydrogen shells n = 1..4.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang import angular as A  # noqa: E402


def main():
    print("(1) l  m   qang angle  vector-model angle  <Jx^2 + Jy^2>")
    for l in (1, 2, 3):
        for m in range(l, -l - 1, -1):
            print(f"    {l} {m:+d}   {math.degrees(A.qang_angle_of_eigenstate(l, m)):6.1f}      "
                  f"{math.degrees(A.vector_model_angle(l, m)):6.1f}            {A.transverse_spread(l, m):g}")
    print("(2) spin-coherent states, theta = 1.0: qg_Z by j")
    for j in (0.5, 1, 2, 5):
        print(f"    j = {j:<4g} qg_Z = {A.qg_z_spin(A.coherent_m_distribution(j, 1.0)):.12f}   cos(1) = {math.cos(1):.12f}")
    print("(3) p orbitals on 2 qubits: P(m_l = +1, 0, -1) = P(weight 0, 1, 2)")
    for name in ("p+1", "p0", "p-1", "px", "py", "pz"):
        pm = A.m_distribution(np.abs(A.p_orbital(name)) ** 2, 2)
        print(f"    {name:4s} {np.round(pm, 3)}  qg_Z = {A.qg_z_spin(pm):+.3f}")
    print("(4) hydrogen shell n: 2(n - 1) qubits, l values found")
    for n in (1, 2, 3, 4):
        ls = A.shell_l_spectrum(n)
        vals = sorted({int(round(x)) for x in ls})
        print(f"    n = {n}: {A.shell_register_qubits(n)} qubits, {len(ls)} states, l = {vals}")


if __name__ == "__main__":
    main()
