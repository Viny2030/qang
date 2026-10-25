"""Figures of chapter 33 of the book (nanoquantum: transmission, partition noise and nanoscale qubits), Spanish and
English, on the chapter 1 base.

    python tools/make_cap33_figures.py

writes book/es/figuras/cap33_*.png and book/en/figures/cap33_*.png and prints the numbers the chapter quotes. There is
no notes section and no qang module for this chapter: every check is a closed form or a Monte Carlo written here.

* Tunnelling: rectangular barrier, V0 = 1 eV, E = 0.5 eV, electron mass; T(a), its reading in m-qg, and the identity
  cos(theta_T / 2) = sqrt(T) = |t| (the qang half-angle is the transmission amplitude).
* Landauer: saddle-point quantum point contact, T_n = 1 / (1 + exp(-2 pi (V - n) / w)); G / G0 = N / 2 + sum qg_n / 2;
  two partly open channels recovered exactly from G and the shot-noise numerator sum T_n (1 - T_n).
* Partition noise: N = 1000 electrons per window, 20 000 windows; Fano factor Var(n_t) / <n_t> = 1 - T, and the spread of
  the estimated T against that of the estimated qang angle.
* Coulomb blockade: level occupation f = 1 / (1 + e^x), x = eps / kT, qg_Z = 2 f - 1 = -tanh(x / 2); temperature from a
  maximum-likelihood Fermi fit and from a straight line through artanh(qg_Z), 400 repetitions.
* Spin or charge qubit read by a charge sensor: Rabi oscillation with readout errors e0, e1; affine readout in m-qg and
  its inversion from the two poles.
* NV centre: m-qg budget of photon-counting readout, sigma_m = 2000 / (C sqrt(N_bright)), checked by Monte Carlo; ODMR line
  centre fitted on simulated photon counts against the Cramer-Rao bound.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit, minimize_scalar  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402

HBAR, ME, EV = 1.054571817e-34, 9.1093837015e-31, 1.602176634e-19
GAMMA_NV = 28.0e9            # Hz per tesla

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "q_x": "voltaje de compuerta (unidades del espaciado de subbandas)",
        "q_g": r"$G/G_0$",
        "q_m": "lectura de cada canal (m-qg)",
        "q_title_g": "La escalera de conductancia",
        "q_title_m": "Cada canal va de un polo al otro",
        "q_ch": "canal",
        "r_x": r"transmisión $T$",
        "r_y": r"desvío $\times\sqrt{N}$",
        "r_t": r"de $\hat T$: $\sqrt{T(1-T)}$",
        "r_th": r"del ángulo qang $\hat\theta$: 1",
        "r_title": "Ruido de partición: el ángulo lo estabiliza",
        "c_x": r"$\varepsilon/k_BT$",
        "c_y": r"$qg_Z$ del nivel",
        "c_exact": r"$-\tanh(x/2)$",
        "c_data": "200 lecturas por punto",
        "c_title": "Ocupación de un nivel en el bloqueo de Coulomb",
        "c_lin": r"$\operatorname{artanh}(qg_Z)$: una recta de pendiente $-1/2k_BT$",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "q_x": "gate voltage (units of the subband spacing)",
        "q_g": r"$G/G_0$",
        "q_m": "reading of each channel (m-qg)",
        "q_title_g": "The conductance staircase",
        "q_title_m": "Each channel goes from one pole to the other",
        "q_ch": "channel",
        "r_x": r"transmission $T$",
        "r_y": r"spread $\times\sqrt{N}$",
        "r_t": r"of $\hat T$: $\sqrt{T(1-T)}$",
        "r_th": r"of the qang angle $\hat\theta$: 1",
        "r_title": "Partition noise: the angle stabilizes it",
        "c_x": r"$\varepsilon/k_BT$",
        "c_y": r"level $qg_Z$",
        "c_exact": r"$-\tanh(x/2)$",
        "c_data": "200 readouts per point",
        "c_title": "Occupation of a level in Coulomb blockade",
        "c_lin": r"$\operatorname{artanh}(qg_Z)$: a line of slope $-1/2k_BT$",
    },
}


# ── tunnelling ─────────────────────────────────────────────────────────────────────────────────────────────────────
def barrier_T(a_nm, E_eV=0.5, V_eV=1.0):
    k = math.sqrt(2 * ME * (V_eV - E_eV) * EV) / HBAR
    s = np.sinh(k * np.asarray(a_nm) * 1e-9)
    return 1.0 / (1.0 + V_eV ** 2 * s ** 2 / (4 * E_eV * (V_eV - E_eV))), k * 1e-9


def tunnelling():
    a = np.array([0.2, 0.5, 1.0, 1.5])
    T, kappa = barrier_T(a)
    theta = 2 * np.arccos(np.sqrt(T))
    m = 2000 * T - 1000
    aa = np.linspace(0.6, 1.6, 50)
    Ta, _ = barrier_T(aa)
    slope = np.polyfit(aa, np.log(np.pi - 2 * np.arccos(np.sqrt(Ta))), 1)[0]
    return {"a": a, "T": T, "m": m, "theta_deg": np.degrees(theta), "kappa": kappa, "slope": slope,
            "amp_err": float(np.abs(np.cos(theta / 2) - np.sqrt(T)).max())}


# ── Landauer ───────────────────────────────────────────────────────────────────────────────────────────────────────
def qpc_T(V, n_channels=4, w=0.45):
    V = np.asarray(V, float)
    return np.array([1 / (1 + np.exp(-2 * np.pi * (V - n - 0.5) / w)) for n in range(n_channels)])


def landauer():
    V = np.linspace(0, 4, 801)
    T = qpc_T(V)
    G = T.sum(0)
    qg = 2 * T - 1
    ident = float(np.abs(G - (T.shape[0] / 2 + qg.sum(0) / 2)).max())
    plateaus = {v: float(qpc_T([v]).sum()) for v in (1.0, 2.0, 3.0)}
    # two partly open channels: overlap them by widening the steps
    Tw = qpc_T([1.4], w=1.6)[:, 0]
    open_ = Tw[(Tw > 0.02) & (Tw < 0.98)]
    if open_.size != 2:
        raise RuntimeError("expected two partly open channels")
    Gp = open_.sum()
    noise = (open_ * (1 - open_)).sum()          # S / (2 e V G0): sum T (1 - T)
    sq = Gp - noise                              # sum T^2
    disc = math.sqrt(2 * sq - Gp ** 2)
    rec = sorted(((Gp + disc) / 2, (Gp - disc) / 2))
    return {"V": V, "T": T, "G": G, "ident": ident, "plateaus": plateaus, "pair": sorted(open_.tolist()),
            "recovered": rec, "fano": noise / Gp}


# ── partition noise ────────────────────────────────────────────────────────────────────────────────────────────────
def partition(N=1000, reps=20000, seed=33):
    rng = np.random.default_rng(seed)
    Ts = np.linspace(0.02, 0.98, 25)
    sT, sth, fano = [], [], []
    for T in Ts:
        n = rng.binomial(N, T, size=reps)
        that = n / N
        sT.append(that.std() * math.sqrt(N))
        sth.append((2 * np.arccos(np.sqrt(that))).std() * math.sqrt(N))
        fano.append(n.var() / n.mean())
    return {"T": Ts, "sT": np.array(sT), "sth": np.array(sth), "fano": np.array(fano), "N": N}


# ── Coulomb blockade thermometry ──────────────────────────────────────────────────────────────────────────────────
def coulomb(N=200, reps=400, kT=1.0, seed=34):
    rng = np.random.default_rng(seed)
    eps = np.linspace(-6, 6, 25)
    f = 1 / (1 + np.exp(eps / kT))
    mle, lin = [], []
    example = None
    for r in range(reps):
        k = rng.binomial(N, f)
        if example is None:
            example = k / N

        def nll(t):
            p = np.clip(1 / (1 + np.exp(eps / t)), 1e-12, 1 - 1e-12)
            return -(k * np.log(p) + (N - k) * np.log(1 - p)).sum()

        mle.append(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        q = 2 * k / N - 1
        ok = np.abs(q) < 1 - 1e-9                 # all-equal points carry no slope information
        y = np.arctanh(q[ok])
        wts = N * (1 - q[ok] ** 2)                # 1 / Var(artanh(qg)) to first order
        slope = np.polyfit(eps[ok], y, 1, w=np.sqrt(wts))[0]
        lin.append(-1 / (2 * slope))
    mle, lin = np.array(mle), np.array(lin)
    ident = float(np.abs((2 * f - 1) + np.tanh(eps / (2 * kT))).max())
    return {"eps": eps, "example": example, "ident": ident, "mle": (mle.mean(), mle.std()),
            "lin": (lin.mean(), lin.std()), "N": N, "reps": reps}


# ── charge-sensor readout of a dot qubit ──────────────────────────────────────────────────────────────────────────
def dot_readout(e0=0.02, e1=0.08, N=2000, seed=35):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, 41)
    p0 = np.cos(t / 2) ** 2                       # Rabi, no decay
    m = 2000 * p0 - 1000
    pr0 = p0 * (1 - e0) + (1 - p0) * e1          # probability of reading 0
    raw = 2000 * rng.binomial(N, pr0) / N - 1000
    pred = (1 - e0 - e1) * m + 1000 * (e1 - e0)
    # calibration from the two poles, N shots each
    e0h = rng.binomial(N, e0) / N
    e1h = rng.binomial(N, e1) / N
    corr = (raw - 1000 * (e1h - e0h)) / (1 - e0h - e1h)
    return {"amp_raw": (raw.max() - raw.min()) / 2, "offset": 1000 * (e1 - e0), "vis": 1 - e0 - e1,
            "pred_err": float(np.abs((2000 * pr0 - 1000) - pred).max()),
            "raw_rms": float(np.sqrt(np.mean((raw - m) ** 2))), "corr_rms": float(np.sqrt(np.mean((corr - m) ** 2))),
            "noise_gain": 1 / (1 - e0 - e1), "fidelity": 1 - (e0 + e1) / 2}


# ── NV centre ──────────────────────────────────────────────────────────────────────────────────────────────────────
def nv_budget(C=0.3, Nb=444_444, reps=4000, seed=36):
    """Photon counts n ~ Poisson(Nb (1 - C (1 - p0))): sigma of the m-qg estimate of p0 at p0 = 1/2."""
    rng = np.random.default_rng(seed)
    p0 = 0.5
    n = rng.poisson(Nb * (1 - C * (1 - p0)), size=reps)
    p0h = 1 - (1 - n / Nb) / C
    return {"mc": float((2000 * p0h).std()), "formula": 2000 / (C * math.sqrt(Nb)) * math.sqrt(1 - C * (1 - p0)),
            "simple": 2000 / (C * math.sqrt(Nb)), "Nb_10": (2000 / (C * 10)) ** 2}


def nv_odmr(C=0.2, fwhm=1.0e6, counts=20000, reps=300, seed=37):
    rng = np.random.default_rng(seed)
    nu = np.linspace(-3e6, 3e6, 61)
    g = fwhm / 2

    def model(x, nu0, a, c):
        return a * (1 - c * g ** 2 / ((x - nu0) ** 2 + g ** 2))

    lam = model(nu, 0.0, counts, C)
    d = counts * C * g ** 2 * 2 * nu / (nu ** 2 + g ** 2) ** 2      # d lambda / d nu0 at nu0 = 0
    crb = 1 / math.sqrt((d ** 2 / lam).sum())
    fits = []
    for _ in range(reps):
        y = rng.poisson(lam)
        fits.append(curve_fit(model, nu, y, p0=(1e5, counts, C), sigma=np.sqrt(np.maximum(y, 1)))[0][0])
    fits = np.array(fits)
    return {"std": float(fits.std()), "crb": crb, "bias": float(fits.mean()), "sigmaB_nT": fits.std() / GAMMA_NV * 1e9,
            "photons": float(lam.sum())}


# ── figures ────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_qpc(t, L):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    a.plot(L["V"], L["G"], color=INK, lw=2)
    a.set_xlabel(t["q_x"], color=INK, fontsize=9)
    a.set_ylabel(t["q_g"], color=INK)
    a.set_title(t["q_title_g"], color=INK, fontsize=11)
    a.grid(alpha=0.25)
    for n, c in enumerate((BLUE, ORANGE, GREEN, GREY)):
        b.plot(L["V"], 2000 * L["T"][n] - 1000, color=c, lw=1.8, label=f"{t['q_ch']} {n + 1}")
    b.axhline(0, color=INK, lw=0.7, ls=":")
    b.set_xlabel(t["q_x"], color=INK, fontsize=9)
    b.set_ylabel(t["q_m"], color=INK)
    b.set_title(t["q_title_m"], color=INK, fontsize=11)
    b.legend(frameon=False, fontsize=8)
    b.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_noise(t, P):
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    Tc = np.linspace(0.01, 0.99, 200)
    ax.plot(Tc, np.sqrt(Tc * (1 - Tc)), color=GREY, lw=1.2)
    ax.plot(P["T"], P["sT"], "o", color=GREY, ms=5, label=t["r_t"])
    ax.axhline(1, color=BLUE, lw=1.2)
    ax.plot(P["T"], P["sth"], "s", color=BLUE, ms=5, label=t["r_th"])
    ax.set_xlabel(t["r_x"], color=INK)
    ax.set_ylabel(t["r_y"], color=INK)
    ax.set_ylim(0, 1.3)
    ax.set_title(t["r_title"], color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_coulomb(t, Cb):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    x = np.linspace(-6, 6, 300)
    a.plot(x, -np.tanh(x / 2), color=INK, lw=1.6, label=t["c_exact"])
    q = 2 * Cb["example"] - 1
    a.plot(Cb["eps"], q, "o", color=BLUE, ms=4.5, label=t["c_data"])
    a.set_xlabel(t["c_x"], color=INK)
    a.set_ylabel(t["c_y"], color=INK)
    a.set_title(t["c_title"], color=INK, fontsize=11)
    a.legend(frameon=False, fontsize=8.5)
    a.grid(alpha=0.25)
    ok = np.abs(q) < 1
    b.plot(x, -x / 2, color=INK, lw=1.4)
    b.plot(Cb["eps"][ok], np.arctanh(q[ok]), "o", color=ORANGE, ms=4.5)
    b.set_xlabel(t["c_x"], color=INK)
    b.set_title(t["c_lin"], color=INK, fontsize=10)
    b.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def main():
    tu = tunnelling()
    print(f"tunnelling (V0 1 eV, E 0.5 eV): kappa = {tu['kappa']:.3f} /nm; max |cos(theta/2) - sqrt(T)| = {tu['amp_err']:.1e}")
    for a, T, m, th in zip(tu["a"], tu["T"], tu["m"], tu["theta_deg"]):
        print(f"  a = {a:.1f} nm: T = {T:.3e}, m = {m:+.2f} m-qg, theta_T = {th:.2f} deg")
    print(f"  slope of ln(pi - theta_T) against a: {tu['slope']:.3f} /nm (-kappa = {-tu['kappa']:.3f})")
    L = landauer()
    print(f"\nLandauer: max |G/G0 - (N/2 + sum qg/2)| = {L['ident']:.1e}; G at V = 1, 2, 3: "
          + ", ".join(f"{v:.4f}" for v in L["plateaus"].values()))
    print(f"  two partly open channels {np.round(L['pair'], 4)} recovered from G and noise as {np.round(L['recovered'], 4)}; "
          f"Fano {L['fano']:.4f}")
    P = partition()
    k = (P["T"] >= 0.05) & (P["T"] <= 0.95)
    print(f"\npartition noise, N = {P['N']}: max |Fano - (1 - T)| = {np.abs(P['fano'] - (1 - P['T'])).max():.3f}")
    print(f"  sqrt(N) std of T-hat: {P['sT'].min():.3f}..{P['sT'].max():.3f}; of the angle for T in [0.05, 0.95]: "
          f"{P['sth'][k].min():.3f}..{P['sth'][k].max():.3f}; at T = {P['T'][0]:.2f} and {P['T'][-1]:.2f}: "
          f"{P['sth'][0]:.3f}, {P['sth'][-1]:.3f}")
    Cb = coulomb()
    print(f"\nCoulomb blockade: max |qg_Z + tanh(x/2)| = {Cb['ident']:.1e}; kT (true 1), {Cb['reps']} repetitions of "
          f"25 points x {Cb['N']} readouts:")
    print(f"  Fermi MLE {Cb['mle'][0]:.4f} +- {Cb['mle'][1]:.4f};  straight line in artanh(qg) {Cb['lin'][0]:.4f} +- "
          f"{Cb['lin'][1]:.4f}")
    D = dot_readout()
    print(f"\ndot qubit, e0 0.02, e1 0.08: affine identity error {D['pred_err']:.1e}; raw amplitude {D['amp_raw']:.0f} m-qg "
          f"(visibility {D['vis']:.2f}), offset {D['offset']:+.0f} m-qg; rms error raw {D['raw_rms']:.1f}, corrected "
          f"{D['corr_rms']:.1f} m-qg; noise x{D['noise_gain']:.3f}; readout fidelity {D['fidelity']:.3f}")
    B = nv_budget()
    print(f"\nNV readout budget, C 0.3, 444 444 bright photons: sigma_m MC {B['mc']:.2f}, formula {B['formula']:.2f}, "
          f"2000/(C sqrt(N)) {B['simple']:.2f} m-qg; photons for 10 m-qg: {B['Nb_10']:.0f}")
    O = nv_odmr()
    print(f"NV ODMR, C 0.2, FWHM 1 MHz, 61 points, {O['photons']:.3g} photons: fitted centre std {O['std']:.0f} Hz, "
          f"Cramer-Rao {O['crb']:.0f} Hz, bias {O['bias']:.0f} Hz; field {O['sigmaB_nT']:.2f} nT")
    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap33_qpc", "cap33_ruido", "cap33_coulomb"), "en": ("cap33_qpc", "cap33_noise", "cap33_coulomb")}[lang]
        for name, fig in zip(names, (fig_qpc(tx, L), fig_noise(tx, P), fig_coulomb(tx, Cb))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
