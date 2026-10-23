"""Figures of chapter 16 of the book (the demarcation between quantum cryptography and post-quantum cryptography),
Spanish and English, on the chapter 1 base.

    python tools/make_cap16_figures.py

writes book/es/figuras/cap16_*.png and book/en/figures/cap16_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. Everything here is computed in this script (exact statevectors, exact distributions and
a classical Monte Carlo); there are no device runs. The ML-KEM failure probabilities are quoted from FIPS 203, not
computed: the toy scheme below only shows how such a probability is (or is not) measured by counting.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import erfc  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402
from qang import algorithms as AL  # noqa: E402
from qang import probability as PB  # noqa: E402

# Decryption failure probabilities of ML-KEM (FIPS 203, table 1), log2.
MLKEM = (("ML-KEM-512", -138.8), ("ML-KEM-768", -164.8), ("ML-KEM-1024", -174.8))

# Toy Regev/Kyber-like noise without compression: e = sum_{i<2n} a_i b_i + c, a, b, c ~ CBD(eta), modulus q.
Q, N_DIM = 3329, 256
ETAS = (8, 12, 16, 20, 24, 28, 32, 40)
MC_TRIALS = 200_000

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "f_x": "lectura de la base en BB84, $m=1000\\,|qg|$ (m-qg)",
        "f_y": "tasa de clave asintótica, $1-2h(Q)$",
        "f_thr": "umbral de Shor–Preskill,\nQ = 11 %: 780 m-qg",
        "f_ir": "interceptar y reenviar,\nQ = 25 %: 500 m-qg",
        "f_bar": "±2σ con 2000 disparos",
        "l_x": r"ancho del ruido de descifrado, $\sigma/q$",
        "l_y": "probabilidad de falla de descifrado",
        "l_ex": "exacta (juguete, n = 256)",
        "l_ga": "aproximación gaussiana",
        "l_mc": f"conteo, {MC_TRIALS:,} cifrados".replace(",", "."),
        "l_floor": "lo que puede contarse:\n$10^{12}$ ensayos",
        "g_x": "iteraciones de Grover, $k$",
        "g_y": r"ángulo qang de la probabilidad de fallar (rad)",
        "g_law": r"ley: $2(2k+1)\arcsin 2^{-n/2}$",
        "g_m": "probabilidad de fallar en m-qg",
        "g_kx": r"$k/k_{\mathrm{opt}}$",
        "s_y": "lectura de cada qubit de conteo (m-qg)",
        "s_before": r"antes de $\mathrm{QFT}^{-1}$: radio local",
        "s_after": r"después: $m_Z$",
        "s_qpe": r"QPE de $T$ ($\varphi=1/8$): $m_X$ antes",
        "s_q": "qubit",
        "s_t7": "Shor, N = 15, a = 7 (r = 4)",
        "s_t11": "Shor, N = 15, a = 11 (r = 2)",
        "s_tq": r"QPE de la compuerta $T$",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "f_x": "BB84 basis reading, $m=1000\\,|qg|$ (m-qg)",
        "f_y": "asymptotic key rate, $1-2h(Q)$",
        "f_thr": "Shor–Preskill threshold,\nQ = 11%: 780 m-qg",
        "f_ir": "intercept-resend,\nQ = 25%: 500 m-qg",
        "f_bar": "±2σ with 2000 shots",
        "l_x": r"width of the decryption noise, $\sigma/q$",
        "l_y": "decryption failure probability",
        "l_ex": "exact (toy, n = 256)",
        "l_ga": "Gaussian approximation",
        "l_mc": f"counting, {MC_TRIALS:,} encryptions",
        "l_floor": "what can be counted:\n$10^{12}$ trials",
        "g_x": "Grover iterations, $k$",
        "g_y": "qang angle of the failure probability (rad)",
        "g_law": r"law: $2(2k+1)\arcsin 2^{-n/2}$",
        "g_m": "failure probability in m-qg",
        "g_kx": r"$k/k_{\mathrm{opt}}$",
        "s_y": "reading of each counting qubit (m-qg)",
        "s_before": r"before $\mathrm{QFT}^{-1}$: local radius",
        "s_after": r"after: $m_Z$",
        "s_qpe": r"QPE of $T$ ($\varphi=1/8$): $m_X$ before",
        "s_q": "qubit",
        "s_t7": "Shor, N = 15, a = 7 (r = 4)",
        "s_t11": "Shor, N = 15, a = 11 (r = 2)",
        "s_tq": r"QPE of the $T$ gate",
    },
}


# --- BB84: the security frontier in m-qg -------------------------------------------------------------------------


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def bb84_rate(m):
    """Shor-Preskill asymptotic rate as a function of the basis reading m = 1000|qg| (QBER = (1 - |qg|)/2)."""
    qber = (1 - np.asarray(m) / 1000) / 2
    return np.maximum(1 - 2 * h2(qber), 0)


# --- Toy LWE decryption failure ----------------------------------------------------------------------------------


def cbd_pmf(eta):
    k = np.arange(-eta, eta + 1)
    from math import comb

    pmf = np.array([sum(comb(eta, i) * comb(eta, i + d) for i in range(eta + 1) if 0 <= i + d <= eta) for d in k])
    return k, pmf / 4**eta


def toy_failure_exact(eta):
    """P(noise mod q in (q/4, 3q/4)) for e = sum_{2n} a b + c, computed exactly mod q by FFT (valid above ~1e-14)."""
    k, p = cbd_pmf(eta)
    prod = np.zeros(Q)
    for i, pi in zip(k, p):
        for j, pj in zip(k, p):
            prod[(i * j) % Q] += pi * pj
    c = np.zeros(Q)
    for i, pi in zip(k, p):
        c[i % Q] += pi
    spec = np.fft.rfft(prod) ** (2 * N_DIM) * np.fft.rfft(c)
    dist = np.clip(np.fft.irfft(spec, Q), 0, None)
    dist /= dist.sum()
    lo, hi = math.ceil(Q / 4), math.floor(3 * Q / 4)
    return float(dist[lo : hi + 1].sum())


def toy_sigma(eta):
    return math.sqrt(2 * N_DIM * (eta / 2) ** 2 + eta / 2)


def gauss_failure(sigma):
    return erfc((Q / 4) / (sigma * math.sqrt(2)))


def toy_failure_mc(eta, trials, rng):
    """Count failures of the toy decryption noise, sampling every a_i, b_i, c."""
    fails, done, chunk = 0, 0, 5000
    while done < trials:
        n = min(chunk, trials - done)
        a = rng.binomial(2 * eta, 0.5, (n, 2 * N_DIM)) - eta  # CBD(eta) = B(2 eta, 1/2) - eta
        b = rng.binomial(2 * eta, 0.5, (n, 2 * N_DIM)) - eta
        c = rng.binomial(2 * eta, 0.5, n) - eta
        e = ((a * b).sum(axis=1) + c) % Q
        fails += int(((e > Q / 4) & (e < 3 * Q / 4)).sum())
        done += n
    return fails


# --- Grover: the attack as a probability equation ---------------------------------------------------------------


def grover_angles(n):
    kopt = AL.grover_optimal_iterations(n)
    ks = np.arange(0, kopt + 1)
    p_fail = np.array([1 - AL.grover_success_probability(n, 1, int(k)) for k in ks])
    theta = np.arccos(np.clip(2 * p_fail - 1, -1, 1))
    law = np.arccos(np.cos(2 * (2 * ks + 1) * math.asin(2 ** (-n / 2))))  # folded into [0, pi], like arccos
    return ks, kopt, theta, law, p_fail


# --- Shor N = 15 and QPE: what single counting qubits read ------------------------------------------------------


def shor_readings(a, n_mod=15, t=4):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFT, UnitaryGate
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

    def mod_mult(power):
        mult = pow(a, power, n_mod)
        u = np.zeros((2**w, 2**w))
        for y in range(2**w):
            u[(mult * y) % n_mod if y < n_mod else y, y] = 1
        return UnitaryGate(u).control(1)

    w = math.ceil(math.log2(n_mod))
    qc = QuantumCircuit(t + w)
    qc.h(range(t))
    qc.x(t)  # work register = |1>
    for k in range(t):
        qc.append(mod_mult(2**k), [k] + list(range(t, t + w)))
    before = Statevector.from_instruction(qc)
    qc.append(QFT(t, inverse=True, do_swaps=True), range(t))
    after = Statevector.from_instruction(qc)
    out = {"before": [], "after": [], "work": []}
    for name, sv, qubits in (("before", before, range(t)), ("after", after, range(t)), ("work", after, range(t, t + w))):
        for q in qubits:
            rho = DensityMatrix(partial_trace(sv, [i for i in range(t + w) if i != q])).data
            x, y, z = 2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real
            out[name].append((1000 * x, 1000 * y, 1000 * z, 1000 * math.sqrt(x * x + y * y + z * z)))
    probs = after.probabilities(range(t))
    out["outcomes"] = {j: round(float(p), 6) for j, p in enumerate(probs) if p > 1e-9}
    return out


def qpe_t_readings(t=3):
    """QPE of T on |1> (phi = 1/8): counting qubit k holds cos(2 pi 2^k phi) in X before the inverse QFT."""
    phi = 1 / 8
    return [1000 * math.cos(2 * math.pi * 2**k * phi) for k in range(t)], -1000.0


# --- Figures -----------------------------------------------------------------------------------------------------


def fig_frontier(t, toy):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    m = np.linspace(0, 1000, 600)
    a1.plot(m, bb84_rate(m), color=BLUE, lw=2)
    a1.axvline(780, color=ORANGE, ls="--", lw=1.2)
    a1.axvline(500, color=GREY, ls=":", lw=1.2)
    a1.text(790, 0.55, t["f_thr"], color=ORANGE, fontsize=8.5)
    a1.text(510, 0.55, t["f_ir"], color=GREY, fontsize=8.5)
    for m0 in (850, 900, 950):
        s = PB.shot_noise_milliqang(math.acos(m0 / 1000), 2000)
        a1.errorbar([m0], [bb84_rate(m0)], xerr=[[2 * s], [2 * s]], fmt="o", color=GREEN, ms=4, capsize=3)
    a1.plot([], [], "o", color=GREEN, label=t["f_bar"])
    a1.legend(loc="upper left", fontsize=8.5, frameon=False)
    a1.set_xlabel(t["f_x"])
    a1.set_ylabel(t["f_y"])
    a1.set_xlim(400, 1000)
    a1.set_ylim(0, 1.02)

    sig = np.linspace(0.02, 0.4, 300)
    a2.semilogy(sig, [gauss_failure(s * Q) for s in sig], color=GREY, ls="--", lw=1.2, label=t["l_ga"])
    ex = [(toy_sigma(e) / Q, p) for e, p, _, _ in toy if p > 1e-12]
    a2.semilogy(*zip(*ex), "-", color=BLUE, lw=2, label=t["l_ex"])
    for e, p, k, n in toy:
        if k > 0:
            _, lo, hi = PB.analyze(n - k, n).p0
            a2.errorbar([toy_sigma(e) / Q], [k / n], yerr=[[k / n - (1 - hi)], [(1 - lo) - k / n]], fmt="s",
                        color=GREEN, ms=4, capsize=3)
    a2.plot([], [], "s", color=GREEN, label=t["l_mc"])
    a2.axhline(1e-12, color=INK, lw=0.8, ls=":")
    a2.text(0.25, 3e-15, t["l_floor"], fontsize=8.5, color=INK)
    for name, l2 in MLKEM:
        a2.axhline(2.0**l2, color=ORANGE, lw=1.0)
        a2.text(0.25, 2.0**l2 * 3, name, fontsize=8, color=ORANGE)
    a2.set_ylim(1e-55, 1)
    a2.set_xlim(0.02, 0.4)
    a2.set_xlabel(t["l_x"])
    a2.set_ylabel(t["l_y"])
    a2.legend(loc="center right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    return fig


def fig_grover(t, gro):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    cols = (BLUE, ORANGE, GREEN, INK)
    for (n, (ks, kopt, theta, law, p_fail)), c in zip(gro.items(), cols):
        a1.plot(ks, law, color=c, lw=1.2)
        a1.plot(ks, theta, "o", color=c, ms=4, label=f"n = {n}")
        a2.plot(ks / kopt, 1000 * (2 * p_fail - 1), "o-", color=c, ms=3.5, lw=1, label=f"n = {n}")
    a1.plot([], [], color=GREY, label=t["g_law"])
    a1.set_xlabel(t["g_x"])
    a1.set_ylabel(t["g_y"])
    a1.legend(fontsize=8.5, frameon=False)
    a2.axhline(0, color=GREY, lw=0.6)
    a2.set_xlabel(t["g_kx"])
    a2.set_ylabel(t["g_m"])
    a2.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    return fig


def fig_shor(t, s7, s11, qpe):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    w = 0.38
    for ax, data, title in ((axes[0], s7, t["s_t7"]), (axes[1], s11, t["s_t11"])):
        q = np.arange(len(data["before"]))
        ax.bar(q - w / 2, [r[3] for r in data["before"]], w, color=GREY, label=t["s_before"])
        ax.bar(q + w / 2, [r[2] for r in data["after"]], w, color=BLUE, label=t["s_after"])
        ax.set_xticks(q)
        ax.set_xlabel(t["s_q"])
        ax.set_title(title, fontsize=10)
    xs, _ = qpe
    q = np.arange(len(xs))
    axes[2].bar(q, xs, 0.5, color=GREEN, label=t["s_qpe"])
    axes[2].set_xticks(q)
    axes[2].set_xlabel(t["s_q"])
    axes[2].set_title(t["s_tq"], fontsize=10)
    for ax in axes:
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_ylim(-1100, 1150)
    axes[0].set_ylabel(t["s_y"])
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[2].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def main():
    rng = np.random.default_rng(16)

    print("BB84 frontier: threshold Q = 11.00% -> m =", round(1000 * (1 - 2 * 0.11)), "m-qg; rate at 900 m-qg:",
          round(float(bb84_rate(900)), 4), "; at 950:", round(float(bb84_rate(950)), 4))
    q_thr = 0.110028
    print("  exact Shor-Preskill zero: Q =", q_thr, "-> m =", round(1000 * (1 - 2 * q_thr), 1))
    for m0 in (850, 900, 950):
        print(f"  sigma_m at m = {m0}, N = 2000: {PB.shot_noise_milliqang(math.acos(m0 / 1000), 2000):.1f} m-qg")

    print("\nToy LWE (q = 3329, n = 256, no compression):")
    toy = []
    for eta in ETAS:
        p = toy_failure_exact(eta)
        k = toy_failure_mc(eta, MC_TRIALS, rng) if p > 1e-7 else 0
        n = MC_TRIALS if p > 1e-7 else 0
        toy.append((eta, p, k, n))
        r = PB.analyze(n - k, n) if n else None
        print(f"  eta {eta:2d}  sigma/q {toy_sigma(eta) / Q:.3f}  exact {p:.3e}  gauss {gauss_failure(toy_sigma(eta)):.3e}"
              + (f"  counted {k}/{n} = {k / n:.3e}  m = {r.milliqang[0]:.1f} [{r.milliqang[1]:.1f}, {r.milliqang[2]:.1f}] m-qg" if n else ""))
    for name, l2 in MLKEM:
        d = 2.0**l2
        print(f"  {name}: delta = 2^{l2} = {d:.2e}; m = 1000 - {2000 * d:.1e} m-qg; trials for one failure {1 / d:.1e}")

    print("\nGrover:")
    gro = {n: grover_angles(n) for n in (4, 6, 8, 10)}
    for n, (ks, kopt, theta, law, p_fail) in gro.items():
        print(f"  n = {n}: k_opt = {kopt}, max |angle - law| = {np.abs(theta - law).max():.2e} rad,"
              f" failure at k_opt = {p_fail[-1]:.4f} ({1000 * (2 * p_fail[-1] - 1):.0f} m-qg)")
    for bits in (128, 256):
        th = math.asin(2 ** (-bits / 2))
        print(f"  AES-{bits}: angle step 4 arcsin 2^-{bits // 2} = {4 * th:.3e} rad/iteration;"
              f" k_opt = {math.pi / (4 * th):.3e}; k to the equator = {math.pi / (8 * th):.3e}")

    print("\nShor N = 15 (4 counting qubits) and QPE of T:")
    s7, s11 = shor_readings(7), shor_readings(11)
    for a, s in ((7, s7), (11, s11)):
        print(f"  a = {a}: outcomes {s['outcomes']}")
        for key in ("before", "after", "work"):
            print("    " + key + ": " + "; ".join(f"x {r[0]:.0f} y {r[1]:.0f} z {r[2]:.0f} r {r[3]:.0f}" for r in s[key]))
    s21 = shor_readings(2, 21, 6)
    print("  N = 21, a = 2 (r = 6), 6 counting qubits: outcomes", s21["outcomes"])
    print("    before, local radius:", [round(r[3]) for r in s21["before"]], "; after, m_Z:", [round(r[2]) for r in s21["after"]])
    qpe = qpe_t_readings()
    print("  QPE T: m_X before inverse QFT per counting qubit", [round(x) for x in qpe[0]], "; target m_Z", qpe[1])

    for lang, t in TEXT.items():
        os.makedirs(t["dir"], exist_ok=True)
        names = {"es": ("cap16_frontera", "cap16_grover", "cap16_shor"), "en": ("cap16_frontier", "cap16_grover", "cap16_shor")}[lang]
        for name, fig in zip(names, (fig_frontier(t, toy), fig_grover(t, gro), fig_shor(t, s7, s11, qpe))):
            path = os.path.join(t["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
