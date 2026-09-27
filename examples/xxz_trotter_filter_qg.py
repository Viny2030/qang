"""
Trotterized XXZ dynamics read in qg: a case where the symmetry filter
reaches the whole observable.

H = sum_i J (X_i X_{i+1} + Y_i Y_{i+1}) + Delta Z_i Z_{i+1}, open chain of n
qubits. H conserves the number of 1s, N. Start from the Neel state
|0101...>, N = n/2, and follow the imbalance

    I(t) = (1/n) sum_i (-1)^i qg_Z,i(t)

(qg_Z,i = <Z_i>). Every quantity is read in the Z basis, so every shot can
be checked: the qg filter keeps shots whose register mean qg equals
1 - 2N/n = 0 (§20). In LiH (§50) the error sat in the X/Y groups the
filter could not reach; here nothing is out of reach.

Trotter step: even bonds then odd bonds, each bond
exp(-i dt (J XX + J YY + Delta ZZ)) compiled to 3 CNOTs. Noise on every
CNOT: two-qubit depolarizing p2 and amplitude damping gamma on both
qubits; symmetric readout error e. Exact density-matrix simulation
(qiskit-aer).

Compared, as the error in I(t) against the noiseless Trotter circuit:
  raw        all shots
  filter     keep shots with N = n/2
  ZNE        CNOT folding (1x, 3x), linear Richardson
  filter+ZNE both
The kept fraction is itself a qg noise meter.

Findings (python examples/xxz_trotter_filter_qg.py):

  Setup: n = 6 (n = 8 in brackets where quoted), dt = 0.25, Delta = J = 1,
  1-8 Trotter steps; errors are |I - I_noiseless Trotter|. The 3-CNOT
  bond and the number-conserving bond reproduce exp(-i dt H_bond) to
  1e-15 (fidelity 1.0000000).

  A. One noise at a time (4 steps). Share of the error the filter removes:
       2q depolarizing p2 = 0.01    41 % (36 %)   either compilation
       readout e = 0.02             94 %
       Z dephasing                   0-2 %        (N-preserving, invisible)
       amplitude damping, 3 CNOT    49 % (38 %)
       amplitude damping, N-cons.   99.5 % (99.4 %)
     Depolarizing: the Paulis that keep N (Z-type, and XX/YY on |01>,|10>)
     pass. Amplitude damping is the new point: T1 always lowers N, and the
     no-jump part is uniform inside a fixed-N sector (checked: with decay
     applied between Trotter steps the filter removes it to 1e-15). With
     the 3-CNOT compilation the intermediate states inside a bond are not
     in the sector, so a decay mid-gate can be rotated back into N = n/2
     and pass the filter: half of the T1 error leaks through. With gates
     that keep N at every point (an XY interaction and a ZZ interaction,
     as on hardware with native iSWAP/fSim-type gates), it cannot.
  B. Mixed noise, 3 CNOT, p2 = 0.01, gamma = 0.005, e = 0.01: error at
     1/2/4/6/8 steps raw 0.036/0.031/0.048/0.224/0.112, filter
     0.015/0.016/0.030/0.146/0.090, ZNE 0.011/0.008/0.029/0.168/0.091,
     filter+ZNE 0.001/0.003/0.006/0.059/0.058. Filter + ZNE is 2.6-8x below
     the best single method at 1-4 steps; past ~90 CNOTs all methods keep
     errors of 0.06-0.2.
  C. Same T1-dominated budget per bond (p2 = 0.002, gamma = 0.02,
     e = 0.01), filter alone: 3 CNOT 0.031/0.025/0.050/0.207/0.116, number-
     conserving 0.005/0.005/0.012/0.081/0.052. Against raw, the filter
     cuts the error 2.3-9x with number-conserving gates and 1.1-1.9x with
     3 CNOTs. With filter + ZNE the number-conserving circuit stays below
     0.011 up to 4 steps and reaches 0.026 and 0.006 at 6 and 8 steps
     (3 CNOT: 0.135 and 0.094). n = 8 gives the same picture.
  D. The price is the kept fraction, also a qg noise meter: 0.70, 0.52,
     0.30, 0.17, 0.11 at 1-8 steps for the number-conserving circuit
     (0.61 to 0.05 at n = 8), lower than for 3 CNOTs because more of the
     decays are now caught. With 2000 shots per circuit at 4 steps the
     RMSE (bias and shot noise together) is raw 0.050, filter 0.028, ZNE
     0.035, filter+ZNE 0.048 for the number-conserving circuit: once the
     filter has removed the bias, ZNE adds only variance. For 3 CNOTs,
     filter+ZNE is the best (0.028 and 0.041 for the two cases).

  What is new and what is not. Symmetry verification by post-selection is
  known (Bonet-Monroig et al. 2018, McArdle et al. 2019) and was used with
  number-conserving fSim gates in Google's Fermi-Hubbard experiment (Arute
  et al. 2020). Contributed: the per-noise reach of the qg filter on a
  dynamics problem where every observable is in Z (the contrast with LiH,
  §50), the measured leak of mid-gate decay through the filter under a
  CNOT compilation (about half of the T1 error) and its closure with
  number-conserving gates, and the combination with ZNE at a finite shot
  budget. Limitations: small chains (n = 6, 8), density-matrix simulation
  with gate-attached noise models, the number-conserving gates are ideal
  unitaries with the same per-bond error budget (no model of how a
  specific device implements them), open boundary conditions, one
  initial state and one observable.
"""

import math

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error, depolarizing_error, pauli_error
except ImportError:  # pragma: no cover
    QuantumCircuit = None


def bond(qc, a, b, theta_xy, theta_zz, fold=1):
    """exp(-i (theta_xy (XX + YY) + theta_zz ZZ)) with 3 CNOTs (Vatan-Williams
    form); each CNOT repeated `fold` times (odd) for ZNE."""

    def cx(c, t):
        for _ in range(fold):
            qc.cx(c, t)

    qc.rz(-math.pi / 2, b)
    cx(b, a)
    qc.rz(2 * theta_zz - math.pi / 2, a)
    qc.ry(math.pi / 2 - 2 * theta_xy, b)
    cx(a, b)
    qc.ry(2 * theta_xy - math.pi / 2, b)
    cx(b, a)
    qc.rz(math.pi / 2, a)


def bond_native(qc, a, b, theta_xy, theta_zz, fold=1):
    """The same bond from two number-conserving native gates, an XY
    (XX+YY) interaction and a ZZ interaction (they commute). Every
    intermediate state keeps N. Folding: G G^dag G."""
    from qiskit.circuit.library import RZZGate, UnitaryGate, XXPlusYYGate
    from qiskit.quantum_info import Operator

    # custom unitaries labelled so the noise model can attach to them
    for g, lab in ((XXPlusYYGate(4 * theta_xy), "xyg"), (RZZGate(2 * theta_zz), "zzg")):
        u = Operator(g).data
        qc.append(UnitaryGate(u, label=lab), [a, b])
        for _ in range((fold - 1) // 2):
            qc.append(UnitaryGate(u.conj().T, label=lab), [a, b])
            qc.append(UnitaryGate(u, label=lab), [a, b])


def trotter_circuit(n, steps, dt, J=1.0, delta=1.0, fold=1, measure=True, native=False):
    qc = QuantumCircuit(n)
    for i in range(1, n, 2):
        qc.x(i)
    for _ in range(steps):
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                (bond_native if native else bond)(qc, i, i + 1, J * dt, delta * dt, fold)
    if measure:
        qc.measure_all()
    return qc


def noise_model(p2, gamma, e, pz=0.0, gates=("cx",)):
    nm = NoiseModel()
    err = depolarizing_error(p2, 2)
    if gamma > 0:
        ad = amplitude_damping_error(gamma)
        err = err.compose(ad.tensor(ad))
    if pz > 0:
        zf = pauli_error([("Z", pz), ("I", 1 - pz)])
        err = err.compose(zf.tensor(zf))
    if p2 > 0 or gamma > 0 or pz > 0:
        nm.add_all_qubit_quantum_error(err, list(gates))
    if e > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]))
    return nm


def probabilities(n, steps, dt, p2=0.0, gamma=0.0, e=0.0, fold=1, delta=1.0, pz=0.0, native=False):
    """Exact outcome distribution (no shot noise) as an array over bitstrings
    (bit i of the index = qubit i). native=True compiles each bond to two
    number-conserving gates and scales the per-gate noise by 3/2, so the
    noise budget per bond equals that of the 3-CNOT compilation."""
    qc = trotter_circuit(n, steps, dt, delta=delta, fold=fold, measure=False, native=native)
    if p2 == 0 and gamma == 0 and e == 0 and pz == 0:
        sim = AerSimulator(method="statevector")
        qc2 = qc.copy()
        qc2.save_probabilities()
        return np.asarray(sim.run(transpile(qc2, sim, optimization_level=0)).result().data()["probabilities"])
    if native:
        nm = noise_model(1.5 * p2, 1.5 * gamma, 0.0, 1.5 * pz, gates=("xyg", "zzg"))
        basis = None
    else:
        nm = noise_model(p2, gamma, 0.0, pz)
        basis = ["cx", "rz", "ry", "x"]
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    t = qc.copy() if basis is None else transpile(qc, basis_gates=basis, optimization_level=0)
    t.save_probabilities()
    p = np.asarray(sim.run(t).result().data()["probabilities"])
    if e > 0:
        p = apply_readout(p, n, e)
    return p


def apply_readout(p, n, e):
    p = p.reshape([2] * n)
    m = np.array([[1 - e, e], [e, 1 - e]])
    for ax in range(n):
        p = np.moveaxis(np.tensordot(m, p, axes=([1], [ax])), 0, ax)
    return p.reshape(-1)


def _tables(n):
    idx = np.arange(1 << n)
    bits = (idx[:, None] >> np.arange(n)) & 1
    z = 1 - 2 * bits  # qg_Z per qubit per outcome
    sign = np.array([(-1) ** i for i in range(n)])
    # Neel |0101..>: qubit i odd is 1 -> Z = -1, so I(0) = (1/n) sum (-1)^i Z_i = +1
    imb = (z * sign).mean(axis=1)
    N = bits.sum(axis=1)
    return imb, N


def imbalance(p, n, filtered=False):
    imb, N = _tables(n)
    if filtered:
        keep = N == n // 2
        return float((p[keep] * imb[keep]).sum() / p[keep].sum()), float(p[keep].sum())
    return float((p * imb).sum()), 1.0


def study(n=6, steps_list=(1, 2, 4, 6, 8), dt=0.25, p2=0.01, gamma=0.005, e=0.01, delta=1.0, native=False):
    rows = []
    for s in steps_list:
        ideal, _ = imbalance(probabilities(n, s, dt, delta=delta), n)
        p1 = probabilities(n, s, dt, p2, gamma, e, 1, delta, native=native)
        p3 = probabilities(n, s, dt, p2, gamma, e, 3, delta, native=native)
        raw1, _ = imbalance(p1, n)
        raw3, _ = imbalance(p3, n)
        f1, kept1 = imbalance(p1, n, True)
        f3, _ = imbalance(p3, n, True)
        zne = 1.5 * raw1 - 0.5 * raw3
        fzne = 1.5 * f1 - 0.5 * f3
        rows.append(dict(steps=s, t=s * dt, cnots=(2 if native else 3) * (n - 1) * s, ideal=ideal, raw=raw1, filt=f1,
                         zne=zne, fzne=fzne, kept=kept1))
    return rows


def noise_decomposition(n=6, steps=4, dt=0.25):
    """Share of the imbalance error the filter removes, one noise at a time,
    for the 3-CNOT and the number-conserving compilation."""
    ideal, _ = imbalance(probabilities(n, steps, dt), n)
    out = []
    for name, kw in (("2q depolarizing p2 = 0.01", dict(p2=0.01)),
                     ("amplitude damping gamma = 0.01", dict(gamma=0.01)),
                     ("readout e = 0.02", dict(e=0.02)),
                     ("Z dephasing pz = 0.01", dict(pz=0.01))):
        for native in (False, True):
            if native and "readout" in name:
                continue
            p = probabilities(n, steps, dt, native=native, **kw)
            raw, _ = imbalance(p, n)
            f, kept = imbalance(p, n, True)
            out.append((name, "N-conserving" if native else "3 CNOT", abs(raw - ideal), abs(f - ideal), kept))
    return ideal, out


def idle_decay_check(n=6, steps=4, dt=0.25, gamma=0.03):
    """Amplitude damping applied BETWEEN Trotter steps (never inside a
    bond): the filtered imbalance equals the noiseless one, because the
    no-jump part is uniform on a fixed-N sector and every jump lowers N.
    Returns (|err| raw, |err| filtered)."""
    qc = QuantumCircuit(n)
    for i in range(1, n, 2):
        qc.x(i)
    err = amplitude_damping_error(gamma)
    for _ in range(steps):
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                bond(qc, i, i + 1, dt, dt)
        for q in range(n):
            qc.append(err.to_instruction(), [q])
    qc.save_probabilities()
    sim = AerSimulator(method="density_matrix")
    p = np.asarray(sim.run(transpile(qc, sim, optimization_level=0)).result().data()["probabilities"])
    ideal, _ = imbalance(probabilities(n, steps, dt), n)
    return abs(imbalance(p, n)[0] - ideal), abs(imbalance(p, n, True)[0] - ideal)


def shot_noise(n, steps, dt, p2, gamma, e, shots, reps, rng, delta=1.0, native=False):
    """Std of the estimators with a finite number of shots per circuit."""
    p1 = probabilities(n, steps, dt, p2, gamma, e, 1, delta, native=native)
    p3 = probabilities(n, steps, dt, p2, gamma, e, 3, delta, native=native)
    imb, N = _tables(n)
    keep = N == n // 2
    out = {"raw": [], "filt": [], "zne": [], "fzne": []}
    for _ in range(reps):
        c1 = rng.multinomial(shots, p1 / p1.sum())
        c3 = rng.multinomial(shots, p3 / p3.sum())
        r1, r3 = (c1 * imb).sum() / shots, (c3 * imb).sum() / shots
        f1 = (c1[keep] * imb[keep]).sum() / max(c1[keep].sum(), 1)
        f3 = (c3[keep] * imb[keep]).sum() / max(c3[keep].sum(), 1)
        out["raw"].append(r1)
        out["filt"].append(f1)
        out["zne"].append(1.5 * r1 - 0.5 * r3)
        out["fzne"].append(1.5 * f1 - 0.5 * f3)
    return {k: float(np.std(v)) for k, v in out.items()}


def make_figure(path, rows_by_case):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rows_by_case), figsize=(5.5 * len(rows_by_case), 4))
    if len(rows_by_case) == 1:
        axes = [axes]
    for ax, (title, rows) in zip(axes, rows_by_case):
        t = [r["t"] for r in rows]
        ax.plot(t, [r["ideal"] for r in rows], "k-", lw=2, label="noiseless Trotter")
        ax.plot(t, [r["raw"] for r in rows], "o-", color="#8c2d04", label="raw")
        ax.plot(t, [r["zne"] for r in rows], "s--", color="#e0a030", label="ZNE (1x, 3x)")
        ax.plot(t, [r["filt"] for r in rows], "o-", color="#1f6fb2", label="qg filter (N = n/2)")
        ax.plot(t, [r["fzne"] for r in rows], "d--", color="#2c7a3a", label="filter + ZNE")
        ax2 = ax.twinx()
        ax2.plot(t, [r["kept"] for r in rows], ":", color="grey")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("kept fraction (dotted)", color="grey", fontsize=8)
        ax.set_xlabel("time t (J = 1)")
        ax.set_ylabel("imbalance I(t)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


CASES = [
    ("3 CNOT: p2 = 0.01, gamma = 0.005, e = 0.01", dict(p2=0.01, gamma=0.005, e=0.01)),
    ("3 CNOT, T1-dominated: p2 = 0.002, gamma = 0.02, e = 0.01", dict(p2=0.002, gamma=0.02, e=0.01)),
    ("N-conserving gates, same T1-dominated budget", dict(p2=0.002, gamma=0.02, e=0.01, native=True)),
]


if __name__ == "__main__":
    import sys

    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6
    results = []
    for title, kw in CASES:
        rows = study(n=n, **kw)
        results.append((title, rows))
        print(f"\n{title}   (n = {n}, dt = 0.25)")
        print(f"{'steps':>5} {'2q g':>5} {'ideal':>7} | {'|err| raw':>9} {'filter':>7} {'ZNE':>7} {'f+ZNE':>7} | kept")
        for r in rows:
            print(f"{r['steps']:5d} {r['cnots']:5d} {r['ideal']:+7.3f} | {abs(r['raw'] - r['ideal']):9.4f}"
                  f" {abs(r['filt'] - r['ideal']):7.4f} {abs(r['zne'] - r['ideal']):7.4f}"
                  f" {abs(r['fzne'] - r['ideal']):7.4f} | {r['kept']:.3f}")
    er, ef = idle_decay_check(n)
    print(f"\nDecay between Trotter steps only (gamma = 0.03): |err| raw {er:.4f}, filtered {ef:.1e}")
    ideal4, dec = noise_decomposition(n)
    print(f"\nOne noise at a time (4 steps, ideal I = {ideal4:+.3f}):")
    for name, comp, er, ef, kept in dec:
        print(f"  {name:32s} {comp:13s} |err| raw {er:.4f}  filter {ef:.4f}  removed {1 - ef / er:6.1%}  kept {kept:.3f}")
    rng = np.random.default_rng(3)
    print("\nFinite shots, 4 steps, 2000 shots per circuit (std, bias, RMSE = sqrt(bias^2 + std^2)):")
    for ci, (title, kw) in enumerate(CASES):
        rows4 = {r["steps"]: r for r in results[ci][1]}[4]
        sn = shot_noise(n, 4, 0.25, kw["p2"], kw["gamma"], kw["e"], 2000, 300, rng, native=kw.get("native", False))
        print(f"  {title}")
        for k in ("raw", "filt", "zne", "fzne"):
            bias = abs(rows4[k] - rows4["ideal"])
            print(f"    {k:5s} std {sn[k]:.4f}  bias {bias:.4f}  RMSE {math.hypot(bias, sn[k]):.4f}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), results)
