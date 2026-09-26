"""
IonQ noisy cloud simulator: BB84 with a stored qubit, T1 drift and an
intercept-resend eavesdropper, diagnosed block by block with the qg flags
of §43 (aria-1, forte-1 noise models).

Why. §40 and §43 used an analytic single-qubit channel. Here every piece
is a circuit on trapped-ion noise models, and the model the flags use is
fitted, not given:

  memory (T1)   amplitude damping with an ancilla m:
                CRY(2 asin sqrt(gamma)) q -> m, then CX m -> q
                (gamma0 = 0.02 is the "healthy" memory; drift 0.03-0.06)
  Eve           measure-and-resend in Z = CX q -> e; in X = H CX H. Both
                are exact: Eve's ancilla is never read, so its record is
                equivalent to a measurement. A fraction f of rounds is
                attacked (half in each basis): the block takes that share
                of its shots from the Eve circuits (without replacement).
  Bob           measures in the basis Alice used (sifted rounds only).

Circuits: 4 states x (4 memories without Eve + 2 memories x 2 Eve bases)
= 32, each with 4 independent copies (12 qubits), 2000 shots, in IonQ's
native gates after a barrier-respecting transpile. Each copy of each
repetition is one block of n = 4000 key (Z) bits and k = 4000 test (X)
bits.

The flags. The device adds its own noise (gates, readout), so the
§43 model gets two symmetric flip rates eps_Z, eps_X on top of
(gamma, f), and the healthy gamma0 is fitted too. Fitting uses the
baseline blocks of the OTHER repetitions (leave-one-out), and the 1 %
thresholds come from a parametric bootstrap of that fitted model. Then

  attack-like   f > 0 with gamma free       (symmetric error beyond any T1)
  drift-like    gamma > gamma0 with f free  (T1 beyond the healthy memory)
  QBER          total errors above the 99 % bootstrap quantile

Modes: "local" (Aer, generic all-to-all noise of §29: dry run and tests)
and "ionq_sim" (free; key in IONQ_API_KEY or the git-ignored .ionq_key).
Nothing here submits to a QPU. Counts are stored per run in
examples/data/ionq_sim_results.json under "ionq_sim|<noise>|bb84|<tag>".

Findings (4 repetitions x 4 copies = 16 blocks of n = k = 4000 per
scenario and noise model; flag rates observed / predicted by the fitted
analytic model for the same block size):

  The device adds a nearly symmetric error of 2.3-2.4 % (aria-1) and
  2.8-3.1 % (forte-1) per bit, mostly from the two MS gates of the
  memory circuit; the fit recovers the designed healthy memory
  (gamma0 = 0.020-0.0225) in every leave-one-out fold.

  aria-1                      QBER       attack-like  drift-like
    baseline                  0.00/0.01  0.00/0.00    0.00/0.02
    T1 drift 0.03             0.56/0.36  0.00/0.00    0.31/0.12
    T1 drift 0.04             0.94/0.93  0.00/0.01    0.81/0.73
    T1 drift 0.06             1.00/1.00  0.00/0.00    1.00/1.00
    intercept f = 0.02        0.62/0.58  0.38/0.21    0.00/0.02
    intercept f = 0.05        1.00/1.00  0.88/0.96    0.00/0.01
    intercept f = 0.10        1.00/1.00  1.00/1.00    0.00/0.01
    drift 0.04 + f = 0.05     1.00/1.00  0.81/0.94    0.81/0.57
  forte-1
    baseline                  0.00/0.01  0.06/0.00    0.06/0.00
    T1 drift 0.04             0.38/0.75  0.00/0.01    0.38/0.46
    T1 drift 0.06             1.00/1.00  0.00/0.00    1.00/0.99
    intercept f = 0.05        1.00/0.99  0.81/0.88    0.06/0.01
    intercept f = 0.10        1.00/1.00  1.00/1.00    0.06/0.01
    drift 0.04 + f = 0.05     1.00/1.00  0.75/0.84    0.31/0.42

  * The attribution transfers to trapped-ion noise: a drifting memory is
    never taken for an attack (0 of 48 aria-1 drift blocks, 1 of 48 on
    forte-1), and an attack is taken for drift in at most 1 of 16
    blocks. Large changes (drift 0.06, f = 0.10) are attributed in every
    block.
  * With 4000-bit blocks and 2-3 % device noise, small changes are only
    partly detected (drift 0.03-0.04: 31-81 %; f = 0.05: 81-88 %), and in
    the mixed block the drift flag drops to 31-81 %. The QBER monitor
    sees changes as often or more, but cannot say which.
  * The fitted analytic model predicts most rates within the sampling
    spread of 16 blocks (+-0.12); the largest misses are forte-1 at
    drift 0.04 (QBER 0.38 observed vs 0.75 predicted) and aria-1 at drift
    0.03 (drift flag 0.31 vs 0.12), i.e. the model is right about which
    flag rises, less so about how often near threshold. §43's numbers
    therefore carry over approximately:
    extrapolated to n = 1e5 key bits with the fitted aria-1 and forte-1
    noise, every scenario is attributed correctly in 99-100 % of blocks,
    with 0.3-2 % cross-flags (predicted(..., scale=25)).

These are vendor noise models run on IonQ's simulator, not hardware; the
memory and the attack are designed circuits.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bb84_qg_eve_vs_noise as B  # noqa: E402

RESULTS = os.path.join(HERE, "data", "ionq_sim_results.json")

STATES = ("0", "1", "+", "-")
GAMMAS = (0.02, 0.03, 0.04, 0.06)
EVE_GAMMAS = (0.02, 0.04)
GAMMA0 = 0.02
COPIES = 4
SHOTS = 2000  # the IonQ noisy simulator caps shots at 2000

SCENARIOS = [("baseline", 0.02, 0.0), ("T1 drift 0.03", 0.03, 0.0), ("T1 drift 0.04", 0.04, 0.0),
             ("T1 drift 0.06", 0.06, 0.0), ("intercept f = 0.02", 0.02, 0.02),
             ("intercept f = 0.05", 0.02, 0.05), ("intercept f = 0.10", 0.02, 0.10),
             ("drift 0.04 + intercept 0.05", 0.04, 0.05)]


# --------------------------------------------------------------------- #
# circuits
# --------------------------------------------------------------------- #
def circuit_keys():
    keys = [(s, g, "none") for s in STATES for g in GAMMAS]
    keys += [(s, g, b) for s in STATES for g in EVE_GAMMAS for b in ("Z", "X")]
    return keys


def bb84_circuit(state, gamma, eve, copies=COPIES):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3 * copies, copies)
    theta = 2 * math.asin(math.sqrt(gamma))
    for c in range(copies):
        q, m, e = 3 * c, 3 * c + 1, 3 * c + 2
        if state in ("1", "-"):
            qc.x(q)
        if state in ("+", "-"):
            qc.h(q)
        qc.barrier()
        if eve == "Z":
            qc.cx(q, e)
        elif eve == "X":
            qc.h(q)
            qc.cx(q, e)
            qc.h(q)
        qc.barrier()
        qc.cry(theta, q, m)
        qc.cx(m, q)
        qc.barrier()
        if state in ("+", "-"):
            qc.h(q)
        qc.measure(q, c)
    return qc


def error_counts(counts, state):
    """Errors per copy (Bob's bit differs from Alice's) and shots."""
    wrong = 0 if state in ("1", "-") else 1  # bit value that is an error
    err = np.zeros(COPIES, dtype=int)
    n = 0
    for key, cnt in counts.items():
        k = key.replace(" ", "")
        v = int(k, 16) if k.startswith("0x") else int(k, 2)
        n += cnt
        for i in range(COPIES):
            if (v >> i) & 1 == wrong:
                err[i] += cnt
    return err.tolist(), n


def load_runs(noise, path=RESULTS):
    """Recorded runs of one noise model (needs only numpy)."""
    import json

    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    return [res[k] for k in sorted(res) if k.startswith(f"ionq_sim|{noise}|bb84|")]


def run(noise, mode, key=None):
    import ionq_sim_hubbard_qaoa as I

    keys = circuit_keys()
    circs = [bb84_circuit(*k) for k in keys]
    if mode == "local":
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        import hubbard_trotter_qg_filters as HB

        # one copy per circuit (Aer's statevector sampler fails on the 12-qubit noisy
        # version); its shots are regrouped into COPIES-bit keys
        sim = AerSimulator(noise_model=HB.all_to_all_noise_model(), method="density_matrix")
        single = [bb84_circuit(*k, copies=1) for k in keys]
        tc = transpile(single, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
        res = sim.run(tc, shots=SHOTS * COPIES, seed_simulator=11, memory=True).result()
        counts = []
        for i in range(len(tc)):
            bits = np.array([int(b) for b in res.get_memory(i)]).reshape(SHOTS, COPIES)
            keys_i = ["".join(str(b) for b in row[::-1]) for row in bits]
            c = {}
            for kk in keys_i:
                c[kk] = c.get(kk, 0) + 1
            counts.append(c)
    else:
        import ionq_sim_zne_grover as G
        from qiskit import transpile

        nb = G.native_backend()
        native = transpile(circs, backend=nb, optimization_level=1)
        counts = I._run(nb, native, noise, "ionq_sim", key, prebuilt=True)
    out = {}
    for k, c in zip(keys, counts):
        err, n = error_counts(c, k[0])
        out["|".join((k[0], f"{k[1]:.2f}", k[2]))] = {"errors": err, "shots": n}
    return out


# --------------------------------------------------------------------- #
# blocks and flags
# --------------------------------------------------------------------- #
def _get(run_data, state, gamma, eve):
    return run_data["|".join((state, f"{gamma:.2f}", eve))]


def make_block(run_data, copy, gamma, f, rng):
    """Error counts (0, 1, +, -) and shots for one block: a fraction f of the
    rounds is taken from the Eve circuits (half per basis), without replacement."""
    k, n = np.zeros(4, dtype=int), np.zeros(4, dtype=int)
    for i, s in enumerate(STATES):
        parts = [("none", 1 - f)] if f == 0 else [("none", 1 - f), ("Z", f / 2), ("X", f / 2)]
        for eve, w in parts:
            d = _get(run_data, s, gamma, eve)
            tot, bad = d["shots"], d["errors"][copy]
            m = int(round(w * tot))
            k[i] += rng.hypergeometric(bad, tot - bad, m) if m < tot else bad
            n[i] += m
    return k, n


G_GRID = np.round(np.arange(0.0, 0.1501, 0.0025), 4)
F_GRID = np.round(np.arange(0.0, 0.3001, 0.0025), 4)
_IDEAL = np.array([[B.error_probabilities(gamma=g, p=0.0, e01=0.0, e10=0.0, f=f) for f in F_GRID]
                   for g in G_GRID])  # (gamma, f, 4)


def model_table(eps_z, eps_x):
    eps = np.array([eps_z, eps_z, eps_x, eps_x])
    return _IDEAL * (1 - eps) + (1 - _IDEAL) * eps


def _ll(table, k, n):
    pr = np.clip(table, 1e-12, 1 - 1e-12)
    return (k * np.log(pr) + (n - k) * np.log(1 - pr)).sum(axis=-1)


def fit_baseline(blocks):
    """(eps_Z, eps_X, gamma0 index) by maximum likelihood with f = 0 on pooled baseline blocks."""
    k = np.sum([b[0] for b in blocks], axis=0)
    n = np.sum([b[1] for b in blocks], axis=0)
    best = (-np.inf, None)
    ideal0 = _IDEAL[:, 0, :]
    for ez in np.arange(0.0, 0.05, 0.0005):
        for ex in np.arange(0.0, 0.05, 0.0005):
            eps = np.array([ez, ez, ex, ex])
            ll = _ll(ideal0 * (1 - eps) + (1 - ideal0) * eps, k, n)
            i = int(np.argmax(ll))
            if ll[i] > best[0]:
                best = (ll[i], (float(ez), float(ex), i))
    return best[1]


def statistics(table, i_g0, k, n):
    ll = _ll(table, k, n)
    best = ll.max()
    total = k.sum()
    return float(best - ll[:, 0].max()), float(best - ll[i_g0, :].max()), int(total)


def calibrate(table, i_g0, n, reps=500, seed=0):
    rng = np.random.default_rng(seed)
    p = table[i_g0, 0]
    s = np.array([statistics(table, i_g0, rng.binomial(n, p), n) for _ in range(reps)])
    return tuple(float(np.quantile(s[:, j], 0.99)) for j in range(3))


def analyse(runs, seed=2):
    """runs: list of per-run data (same noise model). Leave-one-run-out fit."""
    rng = np.random.default_rng(seed)
    out = {name: {"attack": [], "drift": [], "qber": []} for name, _, _ in SCENARIOS}
    fits = []
    for r, data in enumerate(runs):
        others = [runs[j] for j in range(len(runs)) if j != r] or [data]
        base = [make_block(d, c, GAMMA0, 0.0, rng) for d in others for c in range(COPIES)]
        ez, ex, i_g0 = fit_baseline(base)
        fits.append((ez, ex, float(G_GRID[i_g0])))
        table = model_table(ez, ex)
        n = base[0][1]
        th = calibrate(table, i_g0, n)
        for name, g, f in SCENARIOS:
            for c in range(COPIES):
                k, nn = make_block(data, c, g, f, rng)
                sa, sd, tot = statistics(table, i_g0, k, nn)
                out[name]["attack"].append(sa > th[0])
                out[name]["drift"].append(sd > th[1])
                out[name]["qber"].append(tot > th[2])
    summary = {name: {kk: float(np.mean(v)) for kk, v in d.items()} for name, d in out.items()}
    summary["blocks"] = len(runs) * COPIES
    summary["fits"] = fits
    return summary


def predicted(runs, reps=400, seed=3, scale=1):
    """Flag rates the fitted analytic model predicts for the same block size
    (fit on all baseline blocks), to compare with what the simulator gives;
    scale > 1 extrapolates to blocks scale times larger."""
    rng = np.random.default_rng(seed)
    base = [make_block(d, c, GAMMA0, 0.0, rng) for d in runs for c in range(COPIES)]
    ez, ex, i_g0 = fit_baseline(base)
    table = model_table(ez, ex)
    n = base[0][1]
    n = n * scale
    th = calibrate(table, i_g0, n)
    g0 = float(G_GRID[i_g0])
    out = {}
    for name, g, f in SCENARIOS:
        ig = int(np.argmin(np.abs(G_GRID - (g0 + g - GAMMA0))))
        jf = int(np.argmin(np.abs(F_GRID - f)))
        hits = np.array([statistics(table, i_g0, rng.binomial(n, table[ig, jf]), n) for _ in range(reps)])
        out[name] = {"attack": float(np.mean(hits[:, 0] > th[0])), "drift": float(np.mean(hits[:, 1] > th[1])),
                     "qber": float(np.mean(hits[:, 2] > th[2]))}
    return out


def error_rates(runs):
    """Mean error probability per state for each circuit, pooled."""
    out = {}
    for key in runs[0]:
        e = sum(sum(r[key]["errors"]) for r in runs)
        n = sum(r[key]["shots"] * COPIES for r in runs)
        out[key] = e / n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("local", "ionq_sim"), default="local")
    ap.add_argument("--noise", choices=("aria-1", "forte-1"), default="aria-1")
    ap.add_argument("--tag", default="run0")
    ap.add_argument("--analyse", action="store_true", help="analyse all recorded runs of this noise model")
    args = ap.parse_args(argv)
    if args.analyse:
        runs = load_runs(args.noise)
        s = analyse(runs)
        print(f"{args.noise}: {s['blocks']} blocks per scenario; fits (eps_Z, eps_X, gamma0): "
              + ", ".join(f"({a:.4f}, {b:.4f}, {g:.4f})" for a, b, g in s["fits"]))
        pr = predicted(runs)
        print("  (observed / predicted by the fitted model)")
        for name, _, _ in SCENARIOS:
            v, w = s[name], pr[name]
            print(f"  {name:30s} QBER {v['qber']:.2f}/{w['qber']:.2f}  attack-like {v['attack']:.2f}/{w['attack']:.2f}"
                  f"  drift-like {v['drift']:.2f}/{w['drift']:.2f}")
        s["predicted"] = pr
        return s
    import ionq_sim_hubbard_qaoa as I

    key = f"{args.mode}|{args.noise if args.mode == 'ionq_sim' else 'generic'}|bb84|{args.tag}"
    res = I.load_results()
    if key in res:
        print("already done:", key)
        return res[key]
    try:
        out = run(args.noise, args.mode, key)
    except I.Pending as exc:
        print("pending:", key, "-", exc)
        return None
    if args.mode == "ionq_sim":
        res[key] = out
        I.save_results(res)
        print("saved", key)
    rates = error_rates([out])
    for s in STATES:
        print(s, " ".join(f"{g:.2f}:{rates['|'.join((s, f'{g:.2f}', 'none'))]:.4f}" for g in GAMMAS))
    return out


if __name__ == "__main__":
    main()
