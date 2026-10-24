"""Figures of chapter 19 of the book (qubit quantum batteries read in qg: where the energy of a qubit goes, storage
under T1 and T2, certification from shots and the charge locked in registers), Spanish and English, on the chapter 1
base.

    python tools/make_cap19_figures.py

writes book/es/figuras/cap19_*.png and book/en/figures/cap19_*.png and prints the numbers the chapter quotes. Same
style as make_cap01_figures.py. The physics lives in examples/battery_ergotropy_qg.py (§56); this script reuses it.
Everything is exact (closed forms, density matrices) except the certification, a Monte Carlo of shots. Runs in about
ten seconds.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples"))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402

import battery_ergotropy_qg as BT  # noqa: E402
import bell_pairs_network_qg as BP  # noqa: E402

RED = "#8c2d04"
SHOTS = (100, 300, 1000, 3000, 10000)

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "s_y": r"energía (unidades de $\omega$)",
        "s_ext": r"extraíble: $(m_r-m_Z)/2000$",
        "s_lock": r"atrapada por la mezcla: $(1000-m_r)/2000$",
        "s_title": "Dónde está la energía de un qubit",
        "s_lab": ["puro\ninvertido", "puro\necuador", "puro\n$\\theta=2\\pi/3$", "mezclado\nabajo",
                  "mezclado\necuador", "invertido\n$t=0{,}8\\,T_1$"],
        "t_x": r"tiempo de almacenamiento, $t/T_1$",
        "t_y": r"ergotropía (unidades de $\omega$)",
        "t_best": r"mejor ángulo, $T_2={r}\,T_1$",
        "dec": "{,}",
        "t_inv": r"invertida ($\theta=\pi$)",
        "t_eq": r"$m_Z$ de la invertida cruza el ecuador: $t=T_1\ln2$",
        "t_title": "Almacenar: la inversión muere en el ecuador; la coherencia guarda carga",
        "c_x": "disparos por eje",
        "c_y": r"ergotropía estimada (unidades de $\omega$)",
        "c_truth": "verdad",
        "c_plug": "estimación directa",
        "c_lcb": "cota inferior a 3σ",
        "c_inv": r"invertida, $t=0{,}5\,T_1$",
        "c_pas": r"pasiva ($t=0{,}8\,T_1$)",
        "c_title": "Certificar con disparos",
        "r_x": r"tiempo de almacenamiento, $t/T_1$ ($T_2=T_1$)",
        "r_glob": "global",
        "r_loc": "suma de las locales",
        "r_dicke": "Dicke D(4,2)",
        "r_prod": "producto, misma energía",
        "r_title": "Registro de 4 qubits con 2 excitaciones",
        "n_raw": "sin destilar",
        "n_dist": "una ronda DEJMPS",
        "n_y": "fracción secreta BBM92 por par de entrada",
        "n_title": "Un par de Bell guardado en dos memorias",
        "n_title2": "Las lecturas del par guardado ($T_2=T_1$)",
        "n_cut": "corte sin destilar",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "s_y": r"energy (units of $\omega$)",
        "s_ext": r"extractable: $(m_r-m_Z)/2000$",
        "s_lock": r"locked by mixing: $(1000-m_r)/2000$",
        "s_title": "Where the energy of a qubit is",
        "s_lab": ["pure\ninverted", "pure\nequator", "pure\n$\\theta=2\\pi/3$", "mixed\ndown",
                  "mixed\nequator", "inverted\n$t=0.8\\,T_1$"],
        "t_x": r"storage time, $t/T_1$",
        "t_y": r"ergotropy (units of $\omega$)",
        "t_best": r"best angle, $T_2={r}\,T_1$",
        "dec": ".",
        "t_inv": r"inverted ($\theta=\pi$)",
        "t_eq": r"$m_Z$ of the inverted one crosses the equator: $t=T_1\ln2$",
        "t_title": "Storage: inversion dies at the equator; coherence keeps charge",
        "c_x": "shots per axis",
        "c_y": r"estimated ergotropy (units of $\omega$)",
        "c_truth": "truth",
        "c_plug": "plug-in estimate",
        "c_lcb": "3σ lower bound",
        "c_inv": r"inverted, $t=0.5\,T_1$",
        "c_pas": r"passive ($t=0.8\,T_1$)",
        "c_title": "Certifying from shots",
        "r_x": r"storage time, $t/T_1$ ($T_2=T_1$)",
        "r_glob": "global",
        "r_loc": "sum of the local ones",
        "r_dicke": "Dicke D(4,2)",
        "r_prod": "product, same energy",
        "r_title": "4-qubit register with 2 excitations",
        "n_raw": "no distillation",
        "n_dist": "one DEJMPS round",
        "n_y": "BBM92 secret fraction per input pair",
        "n_title": "A Bell pair stored in two memories",
        "n_title2": "The readings of the stored pair ($T_2=T_1$)",
        "n_cut": "cutoff without distillation",
    },
}


def split(qx, qy, qz):
    """Energy, ergotropy and the part locked by mixing, in units of omega, from the readings."""
    r = math.sqrt(qx * qx + qy * qy + qz * qz)
    return (1 - qz) / 2, (r - qz) / 2, (1 - r) / 2


def example_states():
    th = 2 * math.pi / 3
    rp, qz = BT.aged(math.pi, 0.8, 1.0)
    return [(0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (math.sin(th), 0.0, math.cos(th)), (0.0, 0.0, -0.5), (0.5, 0.0, 0.0),
            (rp, 0.0, qz)]


def fig_charge(t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    states = example_states()
    x = np.arange(len(states))
    ext = [split(*s)[1] for s in states]
    lock = [split(*s)[2] for s in states]
    ax1.bar(x, ext, 0.6, color=BLUE, label=t["s_ext"])
    ax1.bar(x, lock, 0.6, bottom=ext, color=GREY, label=t["s_lock"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(t["s_lab"], fontsize=7.5)
    ax1.set_ylabel(t["s_y"])
    ax1.set_ylim(0, 1.15)
    ax1.set_title(t["s_title"], fontsize=10)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ts = np.linspace(0, 2.5, 126)
    for ratio, c in ((2.0, BLUE), (1.0, GREEN), (0.5, ORANGE)):
        ax2.plot(ts, [BT.best_angle(s, ratio)[1] for s in ts], color=c, lw=2, label=t["t_best"].replace("{r}", "" if ratio == 1 else f"{ratio:g}".replace(".", t["dec"])))
    ax2.plot(ts, [BT.stored_ergotropy(math.pi, s, 1.0) for s in ts], color=INK, ls=":", lw=1.5, label=t["t_inv"])
    ax2.axvline(math.log(2), color=GREY, lw=0.8)
    ax2.text(math.log(2) + 0.04, 0.85, t["t_eq"], fontsize=7.5, color=GREY)
    ax2.set_xlabel(t["t_x"])
    ax2.set_ylabel(t["t_y"])
    ax2.set_title(t["t_title"], fontsize=9.5)
    ax2.legend(fontsize=7.5, frameon=False, loc="center right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_register(t, cert, reg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))
    for key, c, lab in (("inv", BLUE, t["c_inv"]), ("pas", ORANGE, t["c_pas"])):
        d = cert[key]
        ax1.axhline(d[SHOTS[0]]["truth"], color=c, lw=0.8)
        ax1.plot(SHOTS, [d[n]["plug_mean"] for n in SHOTS], "o-", color=c, lw=1.5, label=f"{lab}: {t['c_plug']}")
        ax1.plot(SHOTS, [d[n]["lcb_mean"] for n in SHOTS], "s--", color=c, lw=1.5, label=f"{lab}: {t['c_lcb']}")
    ax1.set_xscale("log")
    ax1.set_xlabel(t["c_x"])
    ax1.set_ylabel(t["c_y"])
    ax1.set_title(t["c_title"], fontsize=10)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax1.grid(alpha=0.3)
    tt, rows = reg
    for name, c, lab in (("Dicke", RED, t["r_dicke"]), ("product", BLUE, t["r_prod"])):
        ax2.plot(tt, [v[0] for v in rows[name]], color=c, lw=2, label=f"{lab}: {t['r_glob']}")
        ax2.plot(tt, [v[1] for v in rows[name]], color=c, lw=1.5, ls="--", label=f"{lab}: {t['r_loc']}")
    ax2.set_xlabel(t["r_x"])
    ax2.set_ylabel(t["t_y"])
    ax2.set_title(t["r_title"], fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def distil_crossover(ratio):
    """First storage time at which one DEJMPS round gives more key per input pair than none."""
    lo, hi = 1e-4, BP.cutoff(ratio)
    for _ in range(50):
        mid = (lo + hi) / 2
        if BP.key_fraction_after_dejmps(BP.aged_pair(mid, ratio)) > BP.key_fraction(BP.aged_pair(mid, ratio)):
            hi = mid
        else:
            lo = mid
    return hi


def fig_network(t, ratios=(2.0, 1.0, 0.5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ts = np.linspace(0, 0.45, 91)
    for ratio, c in zip(ratios, (BLUE, GREEN, ORANGE)):
        num = "" if ratio == 1 else f"{ratio:g}".replace(".", t["dec"])
        lab = f"$T_2={num}\,T_1$"
        ax1.plot(ts, [BP.key_fraction(BP.aged_pair(s, ratio)) for s in ts], color=c, lw=2, label=f"{lab}: {t['n_raw']}")
        ax1.plot(ts, [BP.key_fraction_after_dejmps(BP.aged_pair(s, ratio)) for s in ts], color=c, lw=1.5, ls="--",
                 label=f"{lab}: {t['n_dist']}")
    ax1.set_xlabel(t["t_x"])
    ax1.set_ylabel(t["n_y"])
    ax1.set_title(t["n_title"], fontsize=10)
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(alpha=0.3)
    ts = np.linspace(0, 1.0, 101)
    cl = [BP.aged_correlations_closed(s, 1.0) for s in ts]
    ax2.plot(ts, [1000 * c[0] for c in cl], color=BLUE, lw=2, label=r"$1000\,c_x=-1000\,c_y$")
    ax2.plot(ts, [1000 * c[2] for c in cl], color=GREEN, lw=2, label=r"$1000\,c_z$")
    ax2.plot(ts, [1000 * c[3] for c in cl], color=RED, lw=2, label=r"$m_A=m_B$")
    ax2.axvline(BP.cutoff(1.0), color=GREY, lw=0.8)
    ax2.text(BP.cutoff(1.0) + 0.02, 520, t["n_cut"], fontsize=7.5, color=GREY)
    ax2.set_xlabel(t["r_x"])
    ax2.set_ylabel("m-qg")
    ax2.set_title(t["n_title2"], fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 0.0))
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    print("Energy split from the readings (units of omega): energy, ergotropy, locked by mixing; m_r, m_Z")
    for s in example_states():
        e, w, lk = split(*s)
        r = math.sqrt(sum(v * v for v in s))
        print(f"  ({s[0]:+.3f}, {s[1]:+.3f}, {s[2]:+.3f}): E {e:.3f}  W {w:.3f}  locked {lk:.3f}  m_r {1000 * r:.0f} m_Z {1000 * s[2]:.0f}")
    th = np.linspace(0, math.pi, 7)
    dev = max(abs(BT.ergotropy(math.sin(x), 0, math.cos(x)) - math.sin(x / 2) ** 2) for x in th)
    print(f"  pure state: W = omega sin^2(theta/2) = omega p1, max deviation {dev:.1e}")

    print("\nStorage (best angle / W_best / W_inverted / W_equator):")
    for ratio in BT.T2_RATIOS:
        tc, tca = BT.crossover_time(ratio), BT.crossover_time_analytic(ratio)
        cells = []
        for s in (0.5, 1.0, 2.0):
            a, w = BT.best_angle(s, ratio)
            cells.append(f"t={s}: {a / math.pi:.2f}pi {w:.3f}/{BT.stored_ergotropy(math.pi, s, ratio):.3f}/"
                         f"{BT.stored_ergotropy(math.pi / 2, s, ratio):.3f}")
        print(f"  T2 = {ratio} T1: t_c {tc:.3f} (closed {tca:.3f})  " + "  ".join(cells))
    print(f"  ln(4/3) {math.log(4 / 3):.3f}, ln(3/2) {math.log(3 / 2):.3f}, ln 2 {math.log(2):.3f}")
    for s in (0.5, 0.8, math.log(2)):
        print(f"  inverted battery at t = {s:.3f}: m_Z = {1000 * (1 - 2 * math.exp(-s)):+.0f} m-qg,"
              f" W = {BT.stored_ergotropy(math.pi, s, 1.0):.3f}")

    print("\nCertification (truth | plug-in mean, P(over) | 3-sigma LCB mean, P(unsafe)):")
    cert = {"inv": {}, "pas": {}}
    for key, (th0, s) in (("inv", (math.pi, 0.5)), ("pas", (math.pi, 0.8))):
        rp, qz = BT.aged(th0, s, 1.0)
        for n in SHOTS:
            c = cert[key][n] = BT.certify((rp, 0.0, qz), n)
            print(f"  {key} N = {n:5d}: {c['truth']:.4f} | {c['plug_mean']:.4f} ({c['plug_over']:.2f}) | "
                  f"{c['lcb_mean']:.4f} ({c['lcb_unsafe']:.2f})")

    print("\nRegister n = 4, k = 2, T2 = T1 (global, sum of local, mean qg_Z):")
    tab = BT.register_table()
    for name, rows in tab.items():
        print(f"  {name:8s} " + "  ".join(f"t={s}: ({g:.3f}, {lo:.3f}, {m:+.3f})" for s, (g, lo, m) in rows.items()))
    for name, rho in (("Dicke", BT.dicke(4, 2)), ("product", BT.product_equal_energy(4, 2))):
        b = BT._local_bloch(rho, 4, 0)
        print(f"  {name}: local reading of qubit 0 (m_X, m_Y, m_Z) = ({1000 * b[0]:.0f}, {1000 * b[1]:.0f}, {1000 * b[2]:.0f}),"
              f" local radius {1000 * math.sqrt(sum(v * v for v in b)):.0f} m-qg")
    tt = np.linspace(0, 2, 41)
    reg = (tt, {name: [BT.register_ergotropy(BT._local_channel(rho, 4, s, 1.0), 4) for s in tt]
                for name, rho in (("Dicke", BT.dicke(4, 2)), ("product", BT.product_equal_energy(4, 2)))})

    print("\nA stored Bell pair (§57): readings at the BBM92 cutoff and when distillation starts to pay (units of T1)")
    for ratio in BT.T2_RATIOS:
        tc, td, tx_ = BP.cutoff(ratio), BP.cutoff(ratio, True), distil_crossover(ratio)
        cx, _, cz, m = BP.aged_correlations_closed(tc, ratio)
        rho = BP.aged_pair(0.3, ratio)
        cc = BP.correlations(rho)
        dev = max(abs(a - b) for a, b in zip(BP.aged_correlations_closed(0.3, ratio), (cc[0], cc[1], cc[2], cc[3])))
        print(f"  T2 = {ratio} T1: cutoff {tc:.3f}, with one DEJMPS round {td:.3f}, distil only after {tx_:.3f};"
              f" at the cutoff 1000 c_x {1000 * cx:.0f}, 1000 c_z {1000 * cz:.0f}, m_A {1000 * m:.0f} m-qg;"
              f" closed form vs density matrix {dev:.1e}")

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        names = {"es": ("cap19_carga", "cap19_registro", "cap19_red"),
                 "en": ("cap19_charge", "cap19_register", "cap19_network")}[lang]
        for name, fig in zip(names, (fig_charge(tx), fig_register(tx, cert, reg), fig_network(tx))):
            path = os.path.join(tx["dir"], name + ".png")
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
