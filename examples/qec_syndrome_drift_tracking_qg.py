"""
The syndrome as a continuous qg witness: tracking noise drift while a
code runs, and switching code when T2/T1 changes.

§32 gave a rule on T2/T1 alone: Leung code if T2 > T1, no code if
0.4 T1 < T2 < T1, phase-flip code if T2 < 0.4 T1. On real qubits T2
fluctuates (two-level-system defects switch T_phi on and off), so a
choice made at calibration time goes stale. Here the codes' own
syndrome measurements are used as a free, in-situ witness.

In qg units every syndrome ancilla returns the expectation of a
stabilizer S, qg_Z(ancilla) = <S>, and its "flip rate" is (1 - <S>)/2:

  * Leung code: Z1Z2 and Z3Z4 flip under damping jumps (not under
    dephasing); XXXX (also a stabilizer) flips under phase errors.
    Two rates -> both gamma and p.
  * phase-flip code: X1X2, X2X3 flip under dephasing (and the Y part of
    damping). One rate -> p, given gamma.
  * no code: no syndrome -> blind.

Model: code capacity (as §27, §32), one noise round of amplitude damping
gamma = t/T1 and dephasing p = (1 - exp(-t/T_phi))/2 per QEC round,
exact rates and infidelities from Kraus operators. Time is divided into
windows of R QEC rounds; the syndrome counts of a window are binomial
draws from the exact rates. T1 is constant; T2/T1 follows a random
telegraph process between three levels (1.6, 0.6, 0.3 = the three
regimes of §32) with mean dwell of 20 windows.

Strategies (logical infidelity per round, averaged over time):
  oracle              knows T2/T1 in every window
  fixed none/phase/leung
  calibrate once      the §27/§32 qg witness at t = 0, then fixed
  probe every K       dedicated qg_Z(|1>), qg_X(|+>) probes every K
                      windows (2000 extra shots each time)
  probe every window  the same probes in every window (2000 shots/window)
  syndrome tracking   estimates (gamma, p) from the syndromes of the
                      code that is running; when it has chosen "no
                      code" it is blind, so it spends one window in the
                      Leung code every E windows to look again
  hybrid              syndromes while a code runs, the dedicated probes
                      only in windows spent without a code

Findings:

  * Closed forms (exact, checked against the Kraus computation): every
    stabilizer expectation is a power of the single-qubit witness
    qg_X = sqrt(1 - gamma)(1 - 2p) of §27:
        Leung:  <Z0Z1> = 1 - 2 gamma (1 - gamma),  <XXXX> = qg_X^4
        phase:  <X1X2> = qg_X^2
    so the ancilla qg_Z inverts in one line to gamma and p. From one
    window of 1000 rounds the Leung syndromes give gamma to ±23% and p
    to ±28% (T2/T1 = 0.6), enough to separate the three regimes.
  * Drift (T2/T1 telegraph between 1.6, 0.6, 0.3; 400 windows x 5
    seeds; T1 stable, so all gamma information is pooled), regret vs
    the oracle and extra probe shots per window:
        probe every window                                 +6.6%   2000
        hybrid (syndromes + probes only without a code)    +7.2%    481
        syndrome tracking only                            +13.1%      5
        probe every 10 windows                            +20.4%    200
        fixed no code                                     +25.9%      0
        fixed phase code                                  +46.0%      0
        calibrate once (§27/§32 witness at t = 0)          +57.7%      5
        fixed Leung code                                 +129.5%      0
  * The syndromes are free information: syndrome-only tracking beats
    recalibrating every 10 windows with no probe shots at all, and the
    hybrid comes within 0.6 points of probing every window with 4x fewer
    probe shots. A one-time calibration is worse than never coding at
    all once the noise drifts.
  * What limits it: the lag of one window after each switch, and the
    blindness without a code (hence the exploration every 5 windows, or
    the hybrid). Smoothing the p counts over past windows (forgetting
    0.5, 0.8) made it worse (+18.5%, +27.6%): here the drift is abrupt.
    Pooling gamma matters: without it (gamma re-estimated from each
    1000-shot probe, ±30%) the hybrid got stuck in the phase code near
    the T2 = 0.4 T1 boundary (+11.3%).

Scope: code capacity, perfect syndrome extraction, a synthetic drift
model. On hardware the syndrome ancillas have their own readout error
(§31 calibrates it) and extraction adds noise.
"""

import math
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qec_leung_code_t1_qg import (  # noqa: E402
    OPTIONS,
    _noise_kraus,
    infidelity,
    leung_encoder,
    rates_from_times,
)
from qec_repetition_code_choice_qg import (  # noqa: E402
    estimate_noise,
    phase_syndrome_rate,
    sample_witness,
)

T_ROUND = 0.01          # QEC round length in units of T1  -> gamma ~ 0.01
LEVELS = (1.6, 0.6, 0.3)  # T2/T1: Leung / none / phase regime
ROUNDS_PER_WINDOW = 1000
MEAN_DWELL = 20


# --------------------------------------------------------------------- #
# exact syndrome rates (flip probability = (1 - qg_Z(ancilla)) / 2)
# --------------------------------------------------------------------- #
def _pauli_string(ops):
    out = np.array([[1.0]])
    for o in ops:
        out = np.kron(out, o)
    return out


_Zm, _Xm, _I = np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [1.0, 0.0]]), np.eye(2)
# qubit order in leung_encoder: index bits b3 b2 b1 b0; |0011> has qubits 0,1 set.
LEUNG_STABILIZERS = {
    "Z0Z1": _pauli_string([_I, _I, _Zm, _Zm]),
    "Z2Z3": _pauli_string([_Zm, _Zm, _I, _I]),
    "XXXX": _pauli_string([_Xm, _Xm, _Xm, _Xm]),
}


def leung_syndrome_rates(gamma, p):
    """Flip rates of Z0Z1, Z2Z3 (averaged) and XXXX after one noise round
    on the maximally mixed logical state."""
    v = leung_encoder()
    rho = v @ v.T / 2
    out = sum(e @ rho @ e.conj().T for e in _noise_kraus(gamma, p))
    exp = {k: float(np.real(np.trace(s @ out))) for k, s in LEUNG_STABILIZERS.items()}
    return 0.5 * ((1 - exp["Z0Z1"]) / 2 + (1 - exp["Z2Z3"]) / 2), (1 - exp["XXXX"]) / 2


def phase_rate(gamma, p):
    return phase_syndrome_rate(gamma, p)


# closed forms: every stabilizer expectation is a product of single-qubit qg
def leung_rates_closed_form(gamma, p):
    """<Z0Z1> = 1 - 2 gamma (1 - gamma);  <XXXX> = (1 - gamma)^2 (1 - 2p)^4
    = qg_X^4, with qg_X = sqrt(1 - gamma)(1 - 2p) the §27 witness."""
    return gamma * (1 - gamma), (1 - (1 - gamma) ** 2 * (1 - 2 * p) ** 4) / 2


def phase_rate_closed_form(gamma, p):
    """<X1X2> = (1 - gamma)(1 - 2p)^2 = qg_X^2."""
    return (1 - (1 - gamma) * (1 - 2 * p) ** 2) / 2


# --------------------------------------------------------------------- #
# estimators from syndrome counts (inverting the closed forms)
# --------------------------------------------------------------------- #
def estimate_from_leung(n_zz_flips, n_xxxx_flips, rounds):
    """(gamma, p) from the Leung syndromes (ZZ checks: 2 * rounds trials)."""
    rz = min(n_zz_flips / (2 * rounds), 0.2499)
    gamma = (1 - math.sqrt(1 - 4 * rz)) / 2
    s_x = max(1 - 2 * n_xxxx_flips / rounds, 1e-12)
    p = (1 - min(s_x / (1 - gamma) ** 2, 1.0) ** 0.25) / 2
    return float(gamma), float(p)


def estimate_p_from_phase(n_flips, rounds, gamma):
    """p from the X-check flip rate of the phase code (2 * rounds trials), gamma known."""
    s_x = max(1 - 2 * n_flips / (2 * rounds), 1e-12)
    return float((1 - min(s_x / (1 - gamma), 1.0) ** 0.5) / 2)


# --------------------------------------------------------------------- #
# drift and strategies
# --------------------------------------------------------------------- #
def telegraph_series(n_windows, seed=0, levels=LEVELS, mean_dwell=MEAN_DWELL):
    rng = np.random.default_rng(seed)
    out, cur = [], int(rng.integers(len(levels)))
    for _ in range(n_windows):
        out.append(levels[cur])
        if rng.random() < 1 / mean_dwell:
            cur = int(rng.choice([k for k in range(len(levels)) if k != cur]))
    return out


def _best(g, p):
    return min(OPTIONS, key=lambda k: infidelity(k, g, p))


class _Cost:
    """Cache of exact infidelities per (option, T2/T1 level)."""

    def __init__(self):
        self.c = {}

    def __call__(self, option, ratio):
        key = (option, ratio)
        if key not in self.c:
            self.c[key] = infidelity(option, *rates_from_times(T_ROUND, ratio))
        return self.c[key]


class GammaPool:
    """T1 is stable, so every piece of gamma information is pooled: decays
    seen by the |1> probe (Binomial(shots, gamma)) and flips of the Leung
    ZZ checks (Binomial(trials, gamma (1 - gamma)))."""

    def __init__(self):
        self.probe = [0, 0]
        self.zz = [0, 0]

    def add_probe(self, qg_z, shots):
        self.probe[0] += round((1 + qg_z) / 2 * shots)
        self.probe[1] += shots

    def add_zz(self, flips, trials):
        self.zz[0] += flips
        self.zz[1] += trials

    def estimate(self):
        (k1, n1), (k2, n2) = self.probe, self.zz

        def nll(g):
            r = g * (1 - g)
            return -(k1 * math.log(g) + (n1 - k1) * math.log(1 - g) + k2 * math.log(r) + (n2 - k2) * math.log(1 - r))

        return float(minimize_scalar(nll, bounds=(1e-6, 0.5), method="bounded").x)


def _probe(g, p, shots, rng, pool):
    qz, qx = sample_witness(g, p, shots, rng)
    pool.add_probe(qz, shots)
    g_hat = pool.estimate()
    return g_hat, min(max(0.5 * (1 - qx / math.sqrt(1 - g_hat)), 0.0), 0.5)


def run_strategy(name, series, rng, cost, rounds=ROUNDS_PER_WINDOW, probe_every=10, explore_every=5,
                 probe_shots=1000, forget=0.0):
    """Returns (per-window infidelity list, per-window option list, extra probe shots)."""
    inf, opts, extra = [], [], 0
    g_est, p_est = None, None
    choice = None
    since_explore = 0
    acc_x, acc_p = [0.0, 0.0], [0.0, 0.0]
    pool = GammaPool()
    for w, ratio in enumerate(series):
        g, p = rates_from_times(T_ROUND, ratio)
        if name == "oracle":
            choice = min(OPTIONS, key=lambda k: cost(k, ratio))
        elif name.startswith("fixed"):
            choice = name.split()[1]
        elif name == "calibrate once":
            if w == 0:
                g_est, p_est = estimate_noise(*sample_witness(g, p, probe_shots, rng))
                extra += 2 * probe_shots
                choice = _best(g_est, p_est)
        elif name == "probe every K":
            if w % probe_every == 0:
                g_est, p_est = _probe(g, p, probe_shots, rng, pool)
                extra += 2 * probe_shots
                choice = _best(g_est, p_est)
        elif name == "hybrid":
            if w == 0 or choice == "none":
                g_est, p_est = _probe(g, p, probe_shots, rng, pool)
                extra += 2 * probe_shots
                choice = _best(g_est, p_est)
        elif name == "syndrome tracking":
            if w == 0:
                g_est, p_est = _probe(g, p, probe_shots, rng, pool)
                extra += 2 * probe_shots
                choice = _best(g_est, p_est)
            elif choice == "none":
                since_explore += 1
                if since_explore >= explore_every:
                    choice, since_explore = "leung", 0  # look again
        else:
            raise ValueError(name)
        inf.append(cost(choice, ratio))
        opts.append(choice)
        # syndrome information collected during this window, used for the next one
        if name in ("syndrome tracking", "hybrid"):
            if choice == "leung":
                rz, rx = leung_rates_closed_form(g, p)
                pool.add_zz(rng.binomial(2 * rounds, rz), 2 * rounds)
                g_est = pool.estimate()
                acc_x = [forget * acc_x[0] + rng.binomial(rounds, rx), forget * acc_x[1] + rounds]
                s_x = max(1 - 2 * acc_x[0] / acc_x[1], 1e-12)
                p_est = (1 - min(s_x / (1 - g_est) ** 2, 1.0) ** 0.25) / 2
                choice = _best(g_est, p_est)
            elif choice == "phase":
                acc_p = [forget * acc_p[0] + rng.binomial(2 * rounds, phase_rate_closed_form(g, p)),
                         forget * acc_p[1] + rounds]
                p_est = estimate_p_from_phase(acc_p[0], acc_p[1], g_est)
                choice = _best(g_est, p_est)
    return inf, opts, extra


STRATEGIES = ("oracle", "hybrid", "syndrome tracking", "probe every K", "probe every window", "calibrate once", "fixed none",
              "fixed phase", "fixed leung")


def compare(n_windows=400, seeds=range(5), **kw):
    cost = _Cost()
    out = {s: [] for s in STRATEGIES}
    extra = {s: [] for s in STRATEGIES}
    for seed in seeds:
        series = telegraph_series(n_windows, seed)
        for s in STRATEGIES:
            rng = np.random.default_rng(1000 + seed)
            if s == "probe every window":
                inf, _, ex = run_strategy("probe every K", series, rng, cost, **dict(kw, probe_every=1))
            else:
                inf, _, ex = run_strategy(s, series, rng, cost, **kw)
            out[s].append(np.mean(inf))
            extra[s].append(ex / n_windows)
    oracle = np.mean(out["oracle"])
    return {s: {"mean_infidelity": float(np.mean(out[s])), "regret": float(np.mean(out[s]) / oracle - 1),
                "extra_shots_per_window": float(np.mean(extra[s]))} for s in STRATEGIES}


def make_figure(path, n_windows=160, seed=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost = _Cost()
    series = telegraph_series(n_windows, seed)
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 1.2]})
    axes[0].step(range(n_windows), series, where="post", color="k")
    axes[0].axhline(1.0, color="#1f6fb2", ls="--", lw=0.8)
    axes[0].axhline(0.4, color="#2e8b57", ls="--", lw=0.8)
    axes[0].set_ylabel("T2 / T1")
    axes[0].set_title("Drifting dephasing; dashed: §32 boundaries (T2 = T1, T2 = 0.4 T1)", fontsize=9)
    ymap = {"none": 0, "phase": 1, "leung": 2}
    for ax, s in ((axes[1], "syndrome tracking"), (axes[2], "hybrid")):
        rng = np.random.default_rng(1000 + seed)
        inf, opts, _ = run_strategy(s, series, rng, cost)
        best = [min(OPTIONS, key=lambda k: cost(k, r)) for r in series]
        ax.step(range(n_windows), [ymap[o] for o in best], where="post", color="#c7c7c7", lw=4, label="oracle")
        ax.step(range(n_windows), [ymap[o] for o in opts], where="post", color="#1f6fb2" if s.startswith("syn")
                else "#2e8b57", lw=1.5, label=s)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["no code", "phase", "Leung"])
        ax.legend(fontsize=8, loc="upper right")
    axes[2].set_xlabel(f"window ({ROUNDS_PER_WINDOW} QEC rounds each)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    print("Exact syndrome flip rates at gamma = 0.01 (per round):")
    for ratio in LEVELS:
        g, p = rates_from_times(T_ROUND, ratio)
        rz, rx = leung_syndrome_rates(g, p)
        print(f"  T2/T1 = {ratio}: gamma {g:.4f}, p {p:.4f}; Leung ZZ {rz:.4f}, XXXX {rx:.4f}; "
              f"phase XX {phase_rate(g, p):.4f}; best {_best(g, p)}")
    print("\nEstimator check (one window of 1000 rounds, 200 repetitions, T2/T1 = 0.6):")
    rng = np.random.default_rng(0)
    g, p = rates_from_times(T_ROUND, 0.6)
    rz, rx = leung_rates_closed_form(g, p)
    est = np.array([estimate_from_leung(rng.binomial(2000, rz), rng.binomial(1000, rx), 1000) for _ in range(200)])
    print(f"  true gamma {g:.4f} p {p:.4f}; estimated gamma {est[:, 0].mean():.4f} ± {est[:, 0].std():.4f}, "
          f"p {est[:, 1].mean():.4f} ± {est[:, 1].std():.4f}")
    print("\nStrategies (400 windows x 5 drift seeds):")
    t = compare()
    for s in STRATEGIES:
        r = t[s]
        print(f"  {s:18s} infidelity/round {r['mean_infidelity']:.2e}  regret {100 * r['regret']:+7.1f}%  "
              f"extra probe shots/window {r['extra_shots_per_window']:.0f}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
