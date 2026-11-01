"""Independent check of the addition of 2026-11-01 to chapter 7 of the book (the geometric phase in qga).

Claim: the geometric phase of a loop at constant height is 1000·p0 qga modulo 1000, i.e. (1000 + m)/2 qga. The phase
is computed from the discrete overlap product (Pancharatnam) on states built in numpy and in QuTiP; qang is not used.
Prints PASA/FALLA per claim and exits with 1 if any fails.

    .venv-int/Scripts/python tools/verify_cap07.py
"""

import cmath
import math
import sys

import numpy as np
import qutip

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


def loop_phase_qga(states):
    """Geometric phase −arg Π⟨ψ_k|ψ_{k+1}⟩ of a closed discrete loop, in qga modulo 1000."""
    prod = 1.0 + 0j
    for a, b in zip(states, states[1:] + states[:1]):
        prod *= complex(a.conj() @ b)
    return (-cmath.phase(prod) * 1000 / (2 * math.pi)) % 1000


def cone(theta, n=4000, lib="numpy"):
    out = []
    for k in range(n):
        ph = 2 * math.pi * k / n
        if lib == "numpy":
            out.append(np.array([math.cos(theta / 2), np.exp(1j * ph) * math.sin(theta / 2)]))
        else:
            st = (math.cos(theta / 2) * qutip.basis(2, 0) + np.exp(1j * ph) * math.sin(theta / 2) * qutip.basis(2, 1))
            out.append(st.full().ravel())
    return out


for lib in ("numpy", "QuTiP"):
    worst = 0.0
    for m in (800, 500, 0, -300, -900):
        th = math.acos(m / 1000)
        got = loop_phase_qga(cone(th, lib=lib))
        want = ((1000 + m) / 2) % 1000
        worst = max(worst, min(abs(got - want), 1000 - abs(got - want)))
    check(f"{lib}: la fase de un bucle a altura constante es (1000 + m)/2 qga (5 alturas)", worst < 0.01,
          f"desvío máx {worst:.4f} qga")
check("+500 m-qg → 750 qga; ecuador → 500 qga", (1000 + 500) / 2 == 750 and (1000 + 0) / 2 == 500)


def bloch(v):
    x, y, z = v
    th, ph = math.acos(z), math.atan2(y, x)
    return np.array([math.cos(th / 2), np.exp(1j * ph) * math.sin(th / 2)])


octant = [bloch(v) for v in ((0, 0, 1), (1, 0, 0), (0, 1, 0))]
got = loop_phase_qga(octant)
check("octante (Pancharatnam, tres estados): −π/4 = −125 qga ≡ 875 qga", abs(got - 875) < 1e-9, f"{got:.4f} qga")

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
