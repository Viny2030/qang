"""
Where does a real transmon sit on the §71 map? A coherent three-level CZ
and the leakage channel it implies (§72).

§70-§71 used classical leakage models with free parameters: the leak
asymmetry a (leak from |0> relative to |1>), where a leaked qubit returns,
how |2> is read, and how a leaked control kicks its partner. §71 showed
that these parameters decide whether the |2> flag helps. Here they are
derived instead of chosen.

Part A (coherent, no free leakage parameters): two transmons truncated at
three levels, H = D1(t) n1 + (alpha/2) n1(n1 - 1) + (alpha/2) n2(n2 - 1)
+ g (a1^dag a2 + a1 a2^dag), alpha = -2pi 0.3 GHz, g = 2pi 10 MHz. A
diabatic CZ: the data transmon (1) is flux-tuned from 0.8 GHz above its
partner to the |11> <-> |20> resonance (D1 = -alpha) with 4 ns tanh ramps,
held, and tuned back; the hold time is set by maximising the CZ fidelity
after virtual-Z corrections. From the 9 x 9 unitary:
  * leakage from each computational input, and a_eff = mean leak with the
    data qubit in |0> over mean leak with it in |1>;
  * the kick a leaked data qubit gives its ancilla in a CNOT (H CZ H):
    kappa = sin^2(theta_2 / 2), theta_2 the ancilla's conditional phase
    when the data transmon is in |2> (theta_1 = pi, theta_0 = 0 for the
    computational states).
Physics inputs not simulated: |2> decays to |1> at twice the |1> -> |0>
rate (harmonic-oscillator matrix element), so a leaked qubit returns to
|1> with probability 2 gamma per layer; a two-level dispersive readout
assigns |2> to 1; when |11> leaks to |20> the partner goes to |0>, which
in the H CZ H frame is a random kick on the ancilla.

Part B: the §71 surface-code memory with this derived channel (a = a_eff,
kick kappa, return to |1> at rate 2 gamma, |2> read as 1, partner kicked
at the leak event), in the two §71 regimes, ell = 0.002 per CNOT, d = 3
(3 rounds) and d = 5 (5 rounds, T1-dominated only).

Pre-registered predictions (written and committed before any run of this
file; seed 72, 60 000 shots per point):
  Q1  a_eff < 0.05: a transmon CZ leaks only from |1> (from |11> to |20>).
  Q2  in the derived channel the |2> flag is nearly useless: bayes is never
      worse than standard (z > -3) and gains less than 1.1x over it, in
      every regime and distance.
  Q3  erasure of the flag is harmful: worse than standard with z <= -3 in
      the T1-dominated regime at d = 3.
  Q4  the qang gain is a qubit gain: qg (no flags) beats standard by at
      least 1.3x with z >= 3 in the T1-dominated regime at d = 3 and d = 5.
If Q1-Q3 hold, the §70b gain of the |2> readout does not apply to
transmons of this kind, and qang needs no qutrit readout there.

Needs scipy and pymatching.

Findings (python examples/transmon_leakage_channel_qg.py):

  Part A. Best CZ: hold 31.75 ns + two 4 ns ramps, conditional phase
  0.990 pi, average gate fidelity 0.99967, mean leakage 1.3e-4 per gate.
    leakage by input: |00>, |01>, |10>: 0 (exactly); |11>: 5.4e-4
    a_eff = 0; leaked data qubit: ancilla phase theta_2 = 0.89 pi, kick
    kappa = 0.97; |21> and |20> stay put (0.999, 1.000).

  Part B (seed 72, 60 000 shots per point, ell = 0.002 per CNOT):
    d  regime (witness)      standard  erasure  bayes   qg      qg+eras  qg+bayes
    3  T1-dominated (1.68)   0.0153    0.0163   0.0153  0.0084  0.0092   0.0087
    3  depolarizing (0.34)   0.0031    0.0035   0.0030  0.0031  0.0035   0.0030
    5  T1-dominated (2.30)   0.0079    0.0084   0.0079  0.0039  0.0043   0.0041

  Predictions: Q1 PASS, Q2 PASS, Q3 PASS, Q4 PASS.
  * Q1: a_eff = 0. This is structural, not a fitted number: the exchange
    coupling conserves the excitation number, so only |11> (two
    excitations) can reach |20>. Leakage from |0> needs another mechanism
    (drive-induced, heating), not in this model.
  * A leaked transmon is almost invisible to its stabilizers: in |2> it
    still gives the ancilla a phase of 0.89 pi, so the ancilla flips with
    probability 0.97, nearly as if the qubit were still |1>. And it returns
    to |1> (by T1). Its value is preserved: this is the §70 / M0 corner of
    the §71 map.
  * Q2: bayes vs standard 1.001x, 1.011x, 0.998x (z = +1.0, +1.4, -1.0).
    The |2> flag is useless here, as the §71 rule predicts for a = 0.
  * Q3: erasure is harmful: z = -7.2 (T1-dominated), -4.9 (depolarizing),
    -5.7 (d = 5).
  * Q4: the qg gain is a qubit gain and survives realistic leakage: 1.81x
    (z = +15.3) at d = 3 and 2.02x (z = +12.4) at d = 5. Adding the flag to
    qg costs a little (0.0087 vs 0.0084, 0.0041 vs 0.0039), because a
    flagged qubit's T1 weight is set to neutral although its bit is right.

  Verdict. For flux-tuned transmons with a diabatic CZ, the leakage channel
  sits where the |2> readout carries no useful information: qang needs no
  qutrit readout there, and erasure-style leakage handling hurts. The
  §70b / §71 gain applies to platforms where leakage scrambles the qubit
  value. Together, §70-§72: whether a three-level readout helps is decided
  by one number, the leak asymmetry a, together with whether a leaked
  qubit keeps its value; for these transmons a = 0 and it does.
  Limitations: one pulse shape and parameter set (kappa depends on the
  |21> phase, i.e. on the design), three-level truncation, the leak rate in
  Part B (0.002) is set 4x above the coherent value for statistics, the
  partner kick at a leak event and the |2> -> 1 readout assignment are
  assumed, not simulated.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qutrit_bayes_weight_qg as B  # noqa: E402
import surface_code_circuit_t1_qg as SC  # noqa: E402

TWO_PI = 2 * math.pi
ALPHA = -TWO_PI * 0.3
G = TWO_PI * 0.010
D_IDLE = TWO_PI * 0.8
D_RES = -ALPHA
RAMP = 4.0
DT = 0.02


# --------------------------------------------------------------------- #
# Part A: coherent three-level CZ
# --------------------------------------------------------------------- #
def _ops():
    a = np.diag([1.0, math.sqrt(2.0)], 1)
    n = a.T @ a
    one = np.eye(3)
    A1, A2 = np.kron(a, one), np.kron(one, a)
    N1, N2 = np.kron(n, one), np.kron(one, n)
    anh = lambda N: 0.5 * ALPHA * N @ (N - np.eye(9))  # noqa: E731
    H0 = anh(N1) + anh(N2) + G * (A1.T @ A2 + A1 @ A2.T)
    return H0, N1


def detuning(t, hold):
    """Flux pulse: D_IDLE -> D_RES in RAMP ns (tanh), hold, back."""
    total = 2 * RAMP + hold
    up = 0.5 * (1 + np.tanh((t - RAMP / 2) / (RAMP / 6)))
    down = 0.5 * (1 + np.tanh((total - RAMP / 2 - t) / (RAMP / 6)))
    s = np.minimum(up, down)
    return D_IDLE + (D_RES - D_IDLE) * s, total


def cz_unitary(hold):
    from scipy.linalg import expm

    H0, N1 = _ops()
    _, total = detuning(0.0, hold)
    steps = int(round(total / DT))
    U = np.eye(9, dtype=complex)
    for k in range(steps):
        t = (k + 0.5) * total / steps
        d1, _ = detuning(t, hold)
        U = expm(-1j * (H0 + d1 * N1) * (total / steps)) @ U
    # remove the idle-frame dynamics of qubit 1 (it sits at D_IDLE outside the pulse)
    return U


def idx(i, j):
    return 3 * i + j


def analyse(U):
    comp = [idx(0, 0), idx(0, 1), idx(1, 0), idx(1, 1)]
    ph = {k: np.angle(U[k, k]) for k in range(9)}
    z1 = ph[idx(1, 0)] - ph[idx(0, 0)]
    z2 = ph[idx(0, 1)] - ph[idx(0, 0)]
    cphase = (ph[idx(1, 1)] - ph[idx(1, 0)] - ph[idx(0, 1)] + ph[idx(0, 0)]) % TWO_PI
    # fidelity to CZ after virtual Z (2 x 2 subspace average gate fidelity)
    Z = np.diag(np.exp(-1j * np.array([0, z2, z1, z1 + z2]))) * np.exp(-1j * ph[idx(0, 0)])
    Uc = Z @ U[np.ix_(comp, comp)]
    cz = np.diag([1, 1, 1, -1])
    m = np.trace(cz.conj().T @ Uc)
    leak_all = 1 - np.real(np.trace(Uc.conj().T @ Uc)) / 4
    fid = (abs(m) ** 2 + np.real(np.trace(Uc.conj().T @ Uc))) / 20
    lk = {}
    lvl2 = [k for k in range(9) if k // 3 == 2 or k % 3 == 2]
    for (i, j) in ((0, 0), (0, 1), (1, 0), (1, 1)):
        lk[(i, j)] = float(np.sum(np.abs(U[lvl2, idx(i, j)]) ** 2))
    a_eff = (lk[(0, 0)] + lk[(0, 1)]) / max(lk[(1, 0)] + lk[(1, 1)], 1e-300)
    theta2 = (ph[idx(2, 1)] - ph[idx(2, 0)] - z2) % TWO_PI
    kappa = math.sin(theta2 / 2) ** 2
    stay21 = float(abs(U[idx(2, 1), idx(2, 1)]) ** 2)
    stay20 = float(abs(U[idx(2, 0), idx(2, 0)]) ** 2)
    return {"cphase": cphase, "fidelity": fid, "leak_mean": leak_all, "leak": lk, "a_eff": a_eff,
            "theta2": theta2, "kappa": kappa, "stay_21": stay21, "stay_20": stay20}


def best_cz(holds=None):
    if holds is None:
        t0 = math.pi / (math.sqrt(2) * G)
        holds = np.linspace(0.8 * t0, 1.2 * t0, 41)
    best = None
    for h in holds:
        r = analyse(cz_unitary(h))
        if best is None or r["fidelity"] > best[1]["fidelity"]:
            best = (h, r)
    # refine
    h0 = best[0]
    step = (holds[1] - holds[0])
    for h in np.linspace(h0 - step, h0 + step, 21):
        r = analyse(cz_unitary(h))
        if r["fidelity"] > best[1]["fidelity"]:
            best = (h, r)
    return best


# --------------------------------------------------------------------- #
# Part B: the surface-code memory with the derived channel
# --------------------------------------------------------------------- #
def make_sim(p2, gamma, q, ell, a_eff, kappa):
    r_back = 2 * gamma  # |2> -> |1> decay per layer

    def sim(code, rounds, shots, rng, logical=0):
        nd, na = code.nd, code.na
        st = np.zeros((shots, code.nq), dtype=bool)
        st[:, :nd] = code.random_codewords(shots, rng, logical)
        leak = np.zeros((shots, nd), dtype=bool)
        meas = np.zeros((shots, rounds, na), dtype=bool)
        g_idle, g_m = gamma, 2 * gamma

        def decay(cols, g):
            if g > 0:
                cols = np.asarray(cols)
                hit = rng.random((shots, len(cols))) < g
                dm = cols < nd
                if dm.any():
                    hit[:, dm] &= ~leak[:, cols[dm]]
                st[:, cols] = st[:, cols] & ~hit

        def back():
            b = leak & (rng.random((shots, nd)) < r_back)
            st[:, :nd] |= b  # returns to |1>
            leak[b] = False

        for op in SC.ops_list(code, rounds):
            if op[0] == "reset":
                st[:, nd:] = False
                if q > 0:
                    st[:, nd:] = rng.random((shots, na)) < q
                decay(list(range(nd)), g_idle)
                back()
            elif op[0] == "cx":
                busy = set()
                for c, tq in op[3]:
                    kick = rng.random(shots) < kappa
                    st[:, tq] ^= np.where(leak[:, c], kick, st[:, c])
                    busy |= {c, tq}
                    if p2 > 0:
                        pick = rng.integers(1, 16, size=shots)
                        err = rng.random(shots) < p2
                        pa, pb = pick // 4, pick % 4
                        st[:, c] ^= err & ((pa == 1) | (pa == 2)) & ~leak[:, c]
                        st[:, tq] ^= err & ((pb == 1) | (pb == 2))
                    if gamma > 0:
                        hit = rng.random(shots) < gamma
                        st[:, c] &= ~(hit & ~leak[:, c])
                        st[:, tq] &= ~(rng.random(shots) < gamma)
                    if ell > 0:
                        pl = np.where(st[:, c], ell, a_eff * ell)
                        new = ~leak[:, c] & (rng.random(shots) < pl)
                        leak[:, c] |= new
                        st[:, tq] ^= new & (rng.random(shots) < 0.5)  # partner kicked at the leak event
                idle = [x for x in range(code.nq) if x not in busy]
                if idle:
                    decay(idle, g_idle)
                back()
            elif op[0] == "meas":
                decay(list(range(nd, code.nq)), g_m)
                decay(list(range(nd)), g_m)
                out = st[:, nd:].copy()
                if q > 0:
                    out ^= rng.random((shots, na)) < q
                meas[:, op[1]] = out
            elif op[0] == "final":
                decay(list(range(nd)), g_m)
                final = st[:, :nd] | leak  # |2> read as 1
                if q > 0:
                    final = final ^ (rng.random((shots, nd)) < q)
        return meas, final, leak.copy()

    return sim


def surface(a_eff, kappa, regime, d, shots=60000, seed=72, ell=0.002):
    kw = B.REGIMES[regime]
    sim = make_sim(ell=ell, a_eff=a_eff, kappa=kappa, **kw)
    return B.run((a_eff, "one", "one"), d=d, shots=shots, seed=seed, sim=sim, **kw)


POINTS = [("T1-dominated", 3), ("depolarizing", 3), ("T1-dominated", 5)]


def verdict(chan, res):
    q1 = chan["a_eff"] < 0.05
    q2 = all(r["z_bayes_vs_std"] > -3 and r["standard"] / max(r["bayes"], 1e-12) < 1.1 for r in res.values())
    q3 = res[("T1-dominated", 3)]["z_erasure_vs_std"] <= -3
    q4 = all(res[("T1-dominated", d)]["standard"] / res[("T1-dominated", d)]["qg"] >= 1.3
             and res[("T1-dominated", d)].get("z_qg_vs_std", 0) >= 3 for d in (3, 5))
    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}


def main(shots=60000, seed=72):
    hold, chan = best_cz()
    print(f"Part A: CZ hold {hold:.2f} ns (+ 2 x {RAMP:.0f} ns ramps), conditional phase {chan['cphase'] / math.pi:.4f} pi,"
          f" fidelity {chan['fidelity']:.6f}, mean leakage {chan['leak_mean']:.2e}")
    print("  leakage by input: " + ", ".join(f"|{i}{j}> {v:.2e}" for (i, j), v in chan["leak"].items()))
    print(f"  a_eff = {chan['a_eff']:.2e} | leaked data qubit: ancilla phase theta_2 = {chan['theta2'] / math.pi:.3f} pi,"
          f" kick kappa = {chan['kappa']:.3f} | |21> stays {chan['stay_21']:.3f}, |20> stays {chan['stay_20']:.3f}")
    res = {}
    for regime, d in POINTS:
        r = surface(chan["a_eff"], chan["kappa"], regime, d, shots=shots, seed=seed)
        res[(regime, d)] = r
        print(f"Part B d={d} {regime:13s} (witness {r['witness']:.2f}, leaked at end {r['leaked_at_end']:.3f}): "
              + " | ".join(f"{k} {r[k]:.4f}" for k in B.DECODERS))
        print(f"   z: bayes vs std {r['z_bayes_vs_std']:+.1f}, erasure vs std {r['z_erasure_vs_std']:+.1f},"
              f" qg vs std {r['z_qg_vs_std']:+.1f}; std/bayes {r['standard'] / max(r['bayes'], 1e-12):.3f},"
              f" std/qg {r['standard'] / max(r['qg'], 1e-12):.2f}", flush=True)
    v = verdict(chan, res)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return chan, res, v


if __name__ == "__main__":
    main()
