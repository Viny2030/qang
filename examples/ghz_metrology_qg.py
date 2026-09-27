"""
GHZ metrology in qg: when entanglement beats N independent qubits, and
what readout error and T1 do to it.

Sensing a frequency omega with N qubits for an interrogation time t:

  product  every qubit in |+>, Ramsey, read in X: each is a qubit with
           qg_X = V_1 cos(omega t),  V_1 = e^{-t/T2} (1 - 2e)
  GHZ      (|0...0> + |1...1>)/sqrt 2 picks up N omega t; the parity
           P = X_1 ... X_N (one qg of the whole register) reads
           qg_P = V_N cos(N omega t),  V_N = e^{-N t/T2} (1 - 2e)^N

where e is the symmetric readout error per qubit. The Fisher information
of a +-1 readout with qg = V cos(phi) is, at the best operating point
(qg = 0, mid-fringe; the §36 rule F = (V^2 - qg^2)/(1 - qg^2)), V^2 per
shot in phi. Per unit time, with t free:

  product  N t e^{-2t/T2} (1 - 2e)^2         (N independent Ramsey qubits)
  GHZ      N^2 t e^{-2N t/T2} (1 - 2e)^{2N}  (one parity shot)

A. Markovian dephasing (e^{-t/T2}): optimising t, both give N T2/(2e_E)
   (e_E = Euler's number) without readout error: the textbook result that
   GHZ gains nothing under independent Markovian dephasing.
B. Non-Markovian (Gaussian) dephasing, e^{-(t/T2)^2}, typical of slow
   noise: GHZ gains sqrt(N) in Fisher rate (sensitivity N^{-3/4} instead of
   N^{-1/2}).
C. Readout error enters the parity as (1 - 2e)^N (the readout gain b of
   §31 raised to the N): it caps the useful GHZ size at
   N* = 3 / (4 |ln(1 - 2e)|) ~ 3/(8e) in case B, and makes GHZ worse than
   product in case A for every N > 1. Calibrating b and dividing it out
   (§31) removes the bias but not the lost information.
D. A fixed short interrogation window (t_max << T2/N, e.g. a transient
   signal): GHZ gains N (Heisenberg scaling), until readout caps it.
E. T1 during interrogation. Amplitude damping shrinks the GHZ coherence
   by (1 - gamma)^{N/2} and moves every qubit's qg_Z from 0 to +gamma: the
   register-mean qg (§20) is a free T1 witness on a GHZ sensor, invisible
   in the parity signal.

Exact density-matrix checks for N = 3, 4 verify the parity formulas with
dephasing, amplitude damping and readout error.

Findings (python examples/ghz_metrology_qg.py):

  Exact N = 4 density-matrix checks: parity = cos(N phi)(1-2p)^N
  (1-gamma)^{N/2}(1-2e)^N to 6 digits for dephasing, amplitude damping,
  readout error and all three together; mean qg_Z = gamma exactly (0 with
  dephasing alone).

  A. Markovian dephasing, t optimised: gain 1.000 for every N (Huelga et
     al. 1997). With readout error it is < 1: 0.93 at N = 10 and 0.45 at
     N = 100 for e = 0.002; 0.70 and 0.02 for e = 0.01. N independent
     qubits are strictly better.
  B. Gaussian dephasing: gain sqrt(N) exactly (1.41, 3.16, 10.0, 17.3 at
     N = 2, 10, 100, 300), the Zeno-limit N^{-3/4} of Matsuzaki et al. 2011
     and Chin, Huelga, Plenio 2012.
  C. With readout error the Gaussian gain peaks and collapses. Grid N* =
     187, 75, 37, 18 for e = 0.002, 0.005, 0.01, 0.02, identical to
     3/(4|ln(1-2e)|); the gain there is only 3.1, 2.0, 1.4, 1.06. At
     today's trapped-ion readout (e ~ 0.005) a GHZ sensor is worth at most
     ~2x in Fisher rate (1.4x in sensitivity), at N ~ 75.
  D. Window t <= 0.01 T2 (Markovian): gain 1.96, 8.4, 16.8 at N = 2, 10,
     30 (close to N) and then flat at 1/(2 e_E 0.0098) = 18.8 once
     T2/(2N) < t_max, i.e. the window stops binding. Readout cuts it to
     13.3 (e = 0.002) and 5.2 (e = 0.01) at N = 30, below 1 at N = 100
     for e = 0.01.
  E. T1 and dephasing both only shrink the parity fringe, so the parity
     alone cannot tell them apart; the register-mean qg_Z (read in the
     computational basis on a few spare shots) is gamma under T1 and 0
     under dephasing. It is the same witness as §20, applied to a sensor.

  What is new here and what is not. A, B and D are textbook. The qg
  contributions are modest and stated as such: the closed-form readout
  cap N* = 3/(4|ln(1-2e)|) with the small gain it leaves (C), and the
  T1/dephasing separation from mean qg_Z (E). Limitations: independent
  noise only (correlated dephasing, the case where GHZ is really
  hurt or helped, is not modelled); GHZ preparation errors (depth ~log N
  or N) are not included and would lower every GHZ number here; the
  parity is taken as a single-shot product of N readouts, no
  error-mitigated or ancilla readout.
"""

import math

import numpy as np

EULER = math.e


# --------------------------------------------------------------------- #
# Fisher rates with the interrogation time optimised
# --------------------------------------------------------------------- #
def _decay(t, t2, kind):
    return math.exp(-t / t2) if kind == "markov" else math.exp(-((t / t2) ** 2))


def fisher_rate(n, t, e=0.0, t2=1.0, kind="markov", state="ghz", overhead=0.0):
    """Fisher information about omega per unit time at the best operating point."""
    if state == "ghz":
        v = _decay(t, t2, kind) ** n * (1 - 2 * e) ** n
        return n * n * t * t * v * v / (t + overhead)
    v = _decay(t, t2, kind) * (1 - 2 * e)
    return n * t * t * v * v / (t + overhead)


_T = np.geomspace(1e-4, 5.0, 4000)


def best_rate(n, e=0.0, kind="markov", state="ghz", t_max=None, overhead=0.0):
    ts = _T if t_max is None else _T[_T <= t_max]
    vals = np.array([fisher_rate(n, t, e, 1.0, kind, state, overhead) for t in ts])
    i = int(np.argmax(vals))
    return float(vals[i]), float(ts[i])


def gain(n, e=0.0, kind="markov", t_max=None, overhead=0.0):
    """Fisher-rate ratio GHZ / product (sensitivity gain is its square root)."""
    return best_rate(n, e, kind, "ghz", t_max, overhead)[0] / best_rate(n, e, kind, "product", t_max, overhead)[0]


def n_star(e, kind="gauss", n_max=400):
    """GHZ size that maximises the absolute GHZ Fisher rate."""
    ns = np.arange(1, n_max + 1)
    rates = [best_rate(int(n), e, kind, "ghz")[0] for n in ns]
    return int(ns[int(np.argmax(rates))])


# --------------------------------------------------------------------- #
# exact checks (small N)
# --------------------------------------------------------------------- #
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def _on(op, q, n):
    m = np.array([[1.0]])
    for k in range(n - 1, -1, -1):
        m = np.kron(m, op if k == q else I2)
    return m


def _channel(rho, kraus, q, n):
    out = np.zeros_like(rho)
    for K in kraus:
        F = _on(K, q, n)
        out = out + F @ rho @ F.conj().T
    return out


def ghz_parity_exact(n, phi, dephase_p=0.0, gamma=0.0, e=0.0):
    """Parity <X...X> (with readout error e on every qubit) and mean qg_Z of a
    noisy GHZ state that accumulated phase phi on every qubit."""
    v = np.zeros(2**n, dtype=complex)
    v[0] = v[-1] = 1 / math.sqrt(2)
    rho = np.outer(v, v.conj())
    for q in range(n):
        Rz = np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])
        rho = _on(Rz, q, n) @ rho @ _on(Rz, q, n).conj().T
        rho = _channel(rho, [math.sqrt(1 - dephase_p) * I2, math.sqrt(dephase_p) * Z], q, n)
        rho = _channel(rho, [np.array([[1, 0], [0, math.sqrt(1 - gamma)]]), np.array([[0, math.sqrt(gamma)], [0, 0]])], q, n)
    P = np.array([[1.0]])
    for _ in range(n):
        P = np.kron(P, X)
    parity = float(np.real(np.trace(rho @ P))) * (1 - 2 * e) ** n
    mean_qg = float(np.mean([np.real(np.trace(rho @ _on(Z, q, n))) for q in range(n)]))
    return parity, mean_qg


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ns = np.unique(np.round(np.geomspace(1, 300, 40)).astype(int))
    for kind, ls in (("markov", "-"), ("gauss", "--")):
        for e, c in ((0.0, "#1f6fb2"), (0.002, "#e0a030"), (0.01, "#8c2d04")):
            ax1.plot(ns, [gain(int(n), e, kind) for n in ns], ls, color=c, lw=1.8,
                     label=f"{'Markovian' if kind == 'markov' else 'Gaussian'}, e = {e}")
    ax1.axhline(1, color="k", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-2, 40)
    ax1.set_xlabel("number of qubits N")
    ax1.set_ylabel("Fisher-rate gain, GHZ / N independent qubits")
    ax1.set_title("Time optimised: GHZ helps only with slow (Gaussian) noise", fontsize=9)
    ax1.legend(fontsize=6)
    ax1.grid(alpha=0.3)
    for e, c in ((0.0, "#1f6fb2"), (0.002, "#e0a030"), (0.01, "#8c2d04")):
        ax2.plot(ns, [gain(int(n), e, "markov", t_max=0.01) for n in ns], color=c, lw=1.8, label=f"e = {e}")
    ax2.plot(ns, ns, "k:", lw=1, label="N (Heisenberg)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-2, 500)
    ax2.set_xlabel("number of qubits N")
    ax2.set_title("Window t <= 0.01 T2: gain ~N up to N ~ T2/(2 t_max) or the readout cap", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    print("Exact checks (N = 4, phi = 0.3):")
    for p, g, e in ((0.05, 0.0, 0.0), (0.0, 0.1, 0.0), (0.05, 0.1, 0.01)):
        par, mq = ghz_parity_exact(4, 0.3, p, g, e)
        pred = math.cos(4 * 0.3) * (1 - 2 * p) ** 4 * (1 - g) ** 2 * (1 - 2 * e) ** 4
        print(f"  dephase {p}, gamma {g}, e {e}: parity {par:+.6f} (formula {pred:+.6f}), mean qg_Z {mq:+.4f}")
    print("\nA/B/C. Fisher-rate gain GHZ / product with the interrogation time optimised")
    print(f"{'N':>5} | {'Markov e=0':>10} {'e=0.002':>8} {'e=0.01':>8} | {'Gauss e=0':>9} {'e=0.002':>8} {'e=0.01':>8}")
    for n in (2, 4, 10, 30, 100, 300):
        row = [gain(n, e, k) for k in ("markov", "gauss") for e in (0.0, 0.002, 0.01)]
        print(f"{n:5d} | {row[0]:10.3f} {row[1]:8.3f} {row[2]:8.3f} | {row[3]:9.3f} {row[4]:8.3f} {row[5]:8.3f}")
    for e in (0.002, 0.005, 0.01, 0.02):
        ns_ = n_star(e)
        print(f"  Gaussian dephasing, e = {e}: best GHZ size N* = {ns_} (formula 3/(4|ln(1-2e)|) = "
              f"{3 / (4 * abs(math.log(1 - 2 * e))):.0f}), gain there {gain(ns_, e, 'gauss'):.2f}")
    print("\nD. fixed window t <= 0.01 T2 (Markovian): gain")
    for n in (2, 10, 30, 100, 300):
        print(f"  N = {n:3d}: e=0 {gain(n, 0.0, 'markov', 0.01):7.2f}   e=0.002 {gain(n, 0.002, 'markov', 0.01):7.2f}"
              f"   e=0.01 {gain(n, 0.01, 'markov', 0.01):7.2f}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
