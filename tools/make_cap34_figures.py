"""Figures of chapter 34 of the book (cross-domain tests: chemistry, condensed matter, relativistic optics,
thermodynamics and geometry), Spanish and English, on the chapter 1 base.

    python tools/make_cap34_figures.py

writes book/es/figuras/cap34_*.png and book/en/figures/cap34_*.png and prints the numbers the chapter quotes. There is
no notes section and no qang module for these topics: every check is a closed form, an exact diagonalization or a Monte
Carlo written here. Each block checks one claim of the proposal the chapter answers and prints what holds.

* Jordan-Wigner: anticommutators of four modes; exchange sign read as the ancilla qg_X of a Hadamard test.
* UCC double excitation for H2 (STO-3G, 0.735 A, the Hamiltonian of examples/chemistry_qg_symmetry_witness.py).
* Larmor precession and the Hahn echo: static detuning, pulse error, slowly diffusing detuning.
* Transverse-field Ising chain: half-chain entropy and transverse magnetization, N = 2..12, open chain.
* Kitaev chain: parity splitting against length; braid gate exp(pi/4 gamma_2 gamma_3) with an angle error.
* Aharonov-Bohm ring: qg_Z = cos(2 pi Phi / Phi0); flux from shots through the qang angle.
* Wigner rotation: composition of two boosts as 4x4 matrices and as SL(2,C) spinor matrices.
* Fresnel: Brewster zero, total internal reflection phases, Fresnel rhomb and the Artmann Goos-Hanchen shift.
* Bogoliubov / Unruh: two-mode squeezed vacuum; the qang half-angle of the vacuum weight is the Gudermannian of r.
* Jarzynski: two-point measurement on a driven qubit, exact and with finite samples.
* Quantum Fisher information of GHZ states and the classical Fisher information of the parity reading.
* Fubini-Study: geodesic length pole to pole and Gaussian curvature.
* OTOC: integrable (transverse Ising) against chaotic (tilted field) chain, N = 8, infinite temperature.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import expm, polar  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0 + 0j, -1.0])
RNG = np.random.default_rng(34)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def site(op, j, n):
    return kron(*[op if k == j else I2 for k in range(n)])


def m_from_qg(qg):
    return 1000.0 * qg


# ---------------------------------------------------------------- 1. Jordan-Wigner
def jordan_wigner(n=4):
    sp = (X - 1j * Y) / 2          # sigma^+ maps |0> -> ... ; a^dagger creates |1>
    adag = []
    for j in range(n):
        ops = [-Z if k < j else (sp if k == j else I2) for k in range(n)]
        adag.append(kron(*ops))
    a = [m.conj().T for m in adag]
    err_car = max(np.abs(a[i] @ adag[j] + adag[j] @ a[i] - (i == j) * np.eye(2 ** n)).max()
                  for i in range(n) for j in range(n))
    err_aa = max(np.abs(a[i] @ a[j] + a[j] @ a[i]).max() for i in range(n) for j in range(n))
    vac = np.zeros(2 ** n, complex)
    vac[0] = 1
    s12 = adag[0] @ adag[1] @ vac
    s21 = adag[1] @ adag[0] @ vac
    sym = np.linalg.norm(s12 + s21)
    # fermionic swap of modes 0,1 acting on the two-fermion state, read by a Hadamard test: qg_X(ancilla) = Re<psi|U|psi>
    f_swap = np.eye(2 ** n) - adag[0] @ a[0] - adag[1] @ a[1] + adag[0] @ a[1] + adag[1] @ a[0]
    psi = s12 / np.linalg.norm(s12)
    qgx_fermion = float(np.real(psi.conj() @ f_swap @ psi))
    qubit_swap = np.zeros((4, 4))
    for b in range(4):
        qubit_swap[((b & 1) << 1) | (b >> 1), b] = 1
    qubit_swap = np.kron(qubit_swap, np.eye(2 ** (n - 2)))
    qgx_qubit = float(np.real(psi.conj() @ qubit_swap @ psi))
    return dict(err_car=err_car, err_aa=err_aa, sym=sym, qgx_fermion=qgx_fermion, qgx_qubit=qgx_qubit)


# ---------------------------------------------------------------- 2. UCC double excitation for H2
H2_TERMS = [("IIII", -0.090578986088), ("IIIZ", 0.172183932619), ("IIZI", 0.172183932619),
            ("IIZZ", 0.168927538701), ("IZII", -0.225753492224), ("IZIZ", 0.120912632618),
            ("IZZI", 0.166145432564), ("XXYY", -0.045232799946), ("XYYX", 0.045232799946),
            ("YXXY", 0.045232799946), ("YYXX", -0.045232799946), ("ZIII", -0.225753492224),
            ("ZIIZ", 0.166145432564), ("ZIZI", 0.120912632618), ("ZZII", 0.174643430683)]
FCI_H2 = -1.1373060357534004
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_string(label):
    return kron(*[PAULI[c] for c in label])      # leftmost character = qubit 3 (Qiskit order)


def uccd_h2():
    h = sum(c * pauli_string(l) for l, c in H2_TERMS)
    hf = np.zeros(16, complex)
    hf[0b0011] = 1                               # qubits 0, 1 occupied
    dbl = np.zeros(16, complex)
    dbl[0b1100] = 1
    gen = np.outer(dbl, hf) - np.outer(hf, dbl)  # T - T^dagger for the double excitation, antihermitian
    ts = np.linspace(-np.pi / 2, np.pi / 2, 200001)
    e = np.array([np.real(v.conj() @ h @ v) for v in (np.cos(t) * hf + np.sin(t) * dbl for t in ts[::1000])])
    # E(t) = A + B cos 2t + C sin 2t, exact: fit the three coefficients from three angles
    def energy(t):
        v = expm(t * gen) @ hf
        return float(np.real(v.conj() @ h @ v))
    e0, e1, e2 = energy(0.0), energy(np.pi / 4), energy(-np.pi / 4)
    a_ = (e1 + e2) / 2
    b_ = e0 - a_
    c_ = (e1 - e2) / 2
    t_star = 0.5 * math.atan2(-c_, -b_)
    e_min = energy(t_star)
    p_hf = math.cos(t_star) ** 2
    evals = np.linalg.eigvalsh(h)
    # gradient in t and in the HF reading m = 1000 (2 p_HF - 1) = 1000 cos 2t
    eps = 1e-6
    grad_t = (energy(t_star + eps) - energy(t_star - eps)) / (2 * eps)
    return dict(e_hf=e0, e_min=e_min, err_fci=e_min - FCI_H2, ground=evals[0], t_star=t_star,
                theta_qang=2 * abs(t_star), m_hf=m_from_qg(2 * p_hf - 1), grad_t=grad_t, coarse_min=e.min())


# ---------------------------------------------------------------- 3. Larmor and Hahn echo
def hahn_echo(n_spins=20000, sigma=2 * np.pi * 50e3, tau=20e-6, pulse_err=0.02, d_corr=None):
    """Return the echo readout qg_X after pi/2 - tau - pi - tau for an ensemble of detunings (rad/s)."""
    dw = RNG.normal(0, sigma, n_spins)

    def run(pi_angle, drift):
        # spin starts along +x after a perfect pi/2 about y; free precession by phi = dw t
        phi1 = dw * tau
        dw2 = dw + drift
        phi2 = dw2 * tau
        # Bloch vector after first free period
        v = np.stack([np.cos(phi1), np.sin(phi1), np.zeros_like(phi1)])
        # rotation by pi_angle about x
        c, s = np.cos(pi_angle), np.sin(pi_angle)
        v = np.stack([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])
        # second free period: rotation about z by phi2
        c2, s2 = np.cos(phi2), np.sin(phi2)
        v = np.stack([c2 * v[0] - s2 * v[1], s2 * v[0] + c2 * v[1], v[2]])
        return float(v[0].mean())          # the echo forms along +x for a pi pulse about x: x -> x, y -> -y

    fid = float(np.mean(np.cos(dw * tau)))     # free induction decay at time tau, ensemble qg_X
    static = run(np.pi, 0.0)
    wrong = run(np.pi * (1 + pulse_err), 0.0)
    drift = run(np.pi, RNG.normal(0, sigma * 0.1, n_spins))
    return dict(fid=fid, static=static, wrong=wrong, drift=drift,
                fid_theory=math.exp(-(sigma * tau) ** 2 / 2),
                drift_theory=math.exp(-(0.1 * sigma * tau) ** 2 / 2),
                wrong_theory=(1 + math.cos(np.pi * pulse_err)) / 2 + (1 - math.cos(np.pi * pulse_err)) / 2 * math.exp(-2 * (sigma * tau) ** 2))


def larmor():
    gamma_p = 2 * np.pi * 42.577e6            # proton, rad s^-1 T^-1
    b0, t = 1.0, 5e-9
    phi = gamma_p * b0 * t
    psi0 = np.array([1, 1]) / np.sqrt(2)
    u = np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])
    psi = u @ psi0
    qz = float(np.real(psi.conj() @ Z @ psi))
    qx = float(np.real(psi.conj() @ X @ psi))
    return dict(phi=phi, qz=qz, qx=qx, qx_theory=math.cos(phi))


# ---------------------------------------------------------------- 4. Transverse-field Ising chain
def tfim_h(n, g, j=1.0, hz=0.0):
    h = np.zeros((2 ** n, 2 ** n), complex)
    for k in range(n - 1):
        h -= j * site(Z, k, n) @ site(Z, k + 1, n)
    for k in range(n):
        h -= g * site(X, k, n)
        if hz:
            h -= hz * site(Z, k, n)
    return h


def half_entropy(psi, n):
    m = psi.reshape(2 ** (n // 2), 2 ** (n - n // 2))
    s = np.linalg.svd(m, compute_uv=False) ** 2
    s = s[s > 1e-15]
    return float(-(s * np.log2(s)).sum())


def tfim_sparse(n):
    """Sparse -sum ZZ and -sum X of the open chain, so the scan only adds them."""
    from scipy import sparse
    dim = 2 ** n
    idx = np.arange(dim)
    bits = (idx[:, None] >> np.arange(n)[None, :]) & 1
    spins = 1 - 2 * bits
    zz = -np.sum(spins[:, :-1] * spins[:, 1:], axis=1).astype(float)
    hzz = sparse.diags(zz)
    rows = np.concatenate([idx] * n)
    cols = np.concatenate([idx ^ (1 << k) for k in range(n)])
    hx = sparse.csr_matrix((-np.ones(n * dim), (rows, cols)), shape=(dim, dim))
    sx = -hx / n
    # spin-flip parity prod X maps |b> to |~b>; a penalty keeps the solver in the even sector, so the ground state
    # below the transition is the symmetric cat and not an arbitrary mix of the two near-degenerate levels
    flip = sparse.csr_matrix((np.ones(dim), (idx ^ (dim - 1), idx)), shape=(dim, dim))
    penalty = 5.0 * (sparse.identity(dim) - flip) / 2
    return hzz + penalty, hx, sx


def tfim_scan(ns=(2, 4, 6, 8, 10, 12), gs=None):
    from scipy.sparse.linalg import eigsh
    gs = np.linspace(0.05, 2.5, 99) if gs is None else gs
    out = {}
    for n in ns:
        hzz, hx, sx = tfim_sparse(n)
        ent, mx = [], []
        v0 = None
        for g in gs:
            h = hzz + g * hx
            if n <= 6:
                w, v = np.linalg.eigh(h.toarray())
                psi = v[:, 0]
            else:
                w, v = eigsh(h, k=1, which="SA", v0=v0, tol=1e-12)
                psi = v[:, 0]
                v0 = psi
            # half_entropy reshapes with the first index as the most significant bit; site k = bit k, so reverse
            # the bit order is irrelevant for a cut in the middle of a symmetric open chain
            ent.append(half_entropy(psi, n))
            mx.append(float(np.real(psi.conj() @ (sx @ psi))))
        ent, mx = np.array(ent), np.array(mx)
        d_ent = np.gradient(ent, gs)
        d_mx = np.gradient(mx, gs)
        out[n] = dict(g=gs, ent=ent, mx=mx, g_ent=float(gs[np.argmin(d_ent)]), g_mx=float(gs[np.argmax(d_mx)]),
                      m_x_at_1=m_from_qg(float(np.interp(1.0, gs, mx))))
    return out


# ---------------------------------------------------------------- 5. Kitaev chain and braiding
def majoranas(n):
    g = []
    for j in range(n):
        string = [Z] * j
        g.append(kron(*string, X, *[I2] * (n - j - 1)))
        g.append(kron(*string, Y, *[I2] * (n - j - 1)))
    return g


def kitaev_splitting(lengths=(2, 3, 4, 5, 6, 7, 8), mu=0.5, t=1.0, delta=1.0, disorder=0.0):
    out = []
    for n in lengths:
        g = majoranas(n)
        h = np.zeros((2 ** n, 2 ** n), complex)
        mus = mu + disorder * RNG.uniform(-1, 1, n)
        for j in range(n):
            h += -mus[j] / 2 * (np.eye(2 ** n) + 1j * g[2 * j] @ g[2 * j + 1])
        for j in range(n - 1):
            # t = delta: H = i t sum gamma_{2j+1} gamma_{2j+2} (the Kitaev form), up to the mu terms
            h += 1j * t / 2 * g[2 * j + 1] @ g[2 * j + 2] if delta == t else 0
        par = np.eye(2 ** n, dtype=complex)
        for j in range(n):
            par = par @ (-1j * g[2 * j] @ g[2 * j + 1])
        w, v = np.linalg.eigh(h)
        evs = []
        for p in (+1, -1):
            proj = (np.eye(2 ** n) + p * par) / 2
            hp = proj @ h @ proj
            wp = np.linalg.eigvalsh(hp + 10.0 * (np.eye(2 ** n) - proj))
            evs.append(wp[0])
        out.append((n, abs(evs[0] - evs[1])))
    return out


def braid_gate(eps_list=(0.0, 0.01, 0.05, 0.1)):
    g = majoranas(2)
    gen = g[1] @ g[2]                       # gamma_2 gamma_3 (0-based gamma[1], gamma[2])
    b = expm(np.pi / 4 * gen)
    alt = (np.eye(4) + gen) / np.sqrt(2)
    phases = np.angle(np.linalg.eigvals(b))
    res = []
    for eps in eps_list:
        be = expm((np.pi / 4) * (1 + eps) * gen)
        f = abs(np.trace(b.conj().T @ be)) ** 2 / 16
        res.append((eps, 1 - f))
    return dict(err_closed=np.abs(b - alt).max(), phases=np.sort(phases), infid=res,
                square_is_parity=np.abs(b @ b - gen).max())


# ---------------------------------------------------------------- 6. Aharonov-Bohm
def aharonov_bohm(n_shots=2000, reps=4000):
    fluxes = np.array([0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45])
    rows = []
    for f in fluxes:
        p0 = math.cos(math.pi * f) ** 2              # qg_Z = cos(2 pi Phi/Phi0)
        k = RNG.binomial(n_shots, p0, reps)
        th = 2 * np.arccos(np.sqrt(k / n_shots))
        f_hat = th / (2 * np.pi)
        rows.append((f, m_from_qg(math.cos(2 * math.pi * f)), float(f_hat.std() * math.sqrt(n_shots)),
                     float(f_hat.mean() - f)))
    per = max(abs(math.cos(2 * math.pi * (f + 1)) - math.cos(2 * math.pi * f)) for f in np.linspace(0, 1, 101))
    return dict(rows=rows, bound=1 / (2 * math.pi), period_err=per)


# ---------------------------------------------------------------- 7. Wigner rotation
ETA = np.diag([1.0, -1, -1, -1])


def boost(v):
    v = np.asarray(v, float)
    b2 = v @ v
    g = 1 / math.sqrt(1 - b2)
    lam = np.eye(4)
    lam[0, 0] = g
    lam[0, 1:] = lam[1:, 0] = -g * v
    lam[1:, 1:] += (g - 1) * np.outer(v, v) / b2
    return lam


def spinor_boost(v):
    v = np.asarray(v, float)
    s = math.sqrt(v @ v)
    eta = math.atanh(s)
    n = v / s
    ns = n[0] * X + n[1] * Y + n[2] * Z
    return math.cosh(eta / 2) * I2 + math.sinh(eta / 2) * ns


def wigner(v1=(0.6, 0, 0), v2=(0, 0.8, 0)):
    l1, l2 = boost(v1), boost(v2)
    lam = l2 @ l1
    g = lam[0, 0]
    v = -lam[1:, 0] / g                       # velocity of the composed frame
    rot = np.linalg.inv(boost(v)) @ lam
    r3 = rot[1:, 1:]
    omega = math.acos(max(-1, min(1, (np.trace(r3) - 1) / 2)))
    g1 = 1 / math.sqrt(1 - np.dot(v1, v1))
    g2 = 1 / math.sqrt(1 - np.dot(v2, v2))
    gam = g1 * g2 * (1 + np.dot(v1, v2))
    cos_half = (1 + gam + g1 + g2) / math.sqrt(2 * (1 + gam) * (1 + g1) * (1 + g2))
    # spinor representation: product of Hermitian boosts = (Hermitian boost) x (unitary rotation)
    s = spinor_boost(v2) @ spinor_boost(v1)
    u, p = polar(s, side="left")             # s = p u, p Hermitian positive, u unitary
    omega_spin = 2 * math.acos(min(1, abs(np.trace(u)) / 2))
    # spin read-out: spin along the composed velocity's normal... use z axis ⟂ to the xy rotation plane? rotation is about z
    up_x = np.array([1, 1]) / np.sqrt(2)
    out = u @ up_x
    qg_x = float(np.real(out.conj() @ X @ out))
    return dict(omega=omega, omega_formula=2 * math.acos(cos_half), omega_spin=omega_spin, gamma=gam,
                lorentz_err=np.abs(lam.T @ ETA @ lam - ETA).max(), rot_err=np.abs(r3.T @ r3 - np.eye(3)).max(),
                unitary_err=np.abs(u.conj().T @ u - I2).max(), det=abs(np.linalg.det(s) - 1), qg_x=qg_x)


# ---------------------------------------------------------------- 8. Fresnel
def fresnel(n1, n2, ti):
    st = n1 / n2 * np.sin(ti) + 0j
    ct = np.sqrt(1 - st ** 2)
    ct = np.where(np.imag(ct) < 0, -ct, ct)
    ci = np.cos(ti)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return rs, rp


def fresnel_tests(n=1.5, lam=633e-9):
    tb = math.atan(n)
    rs, rp = fresnel(1.0, n, np.array([tb]))
    # TIR from glass to air
    tc = math.asin(1 / n)
    th = np.linspace(tc + 1e-6, np.pi / 2 - 1e-6, 200001)
    rs_t, rp_t = fresnel(n, 1.0, th)
    mod_err = max(np.abs(np.abs(rs_t) - 1).max(), np.abs(np.abs(rp_t) - 1).max())
    dphi = np.angle(rp_t) - np.angle(rs_t)
    dphi = np.unwrap(dphi)
    rel = np.abs(np.mod(dphi + np.pi, 2 * np.pi) - np.pi)
    k = int(np.argmax(rel))
    # Fresnel rhomb: relative phase 45 deg per reflection; first crossing above the critical angle and second
    target = np.pi / 4
    cross = th[np.where(np.diff(np.sign(rel - target)))[0]]
    # Artmann: D = -(lam / 2 pi n) d delta / d theta, with delta the phase of r
    ds = np.gradient(np.unwrap(np.angle(rs_t)), th)
    d_gh = -lam / (2 * np.pi * n) * ds
    t45 = math.radians(45)
    i45 = int(np.argmin(abs(th - t45)))
    # Stokes reading after one reflection of 45-degree linear light: qg_Y = S3/S0 = sin(delta_p - delta_s)
    s3 = math.sin(rel[i45])
    r_normal = ((n - 1) / (n + 1)) ** 2
    return dict(theta_b=math.degrees(tb), rp_b=abs(rp[0]), rs_b2=abs(rs[0]) ** 2, theta_c=math.degrees(tc),
                mod_err=mod_err, rel_max=math.degrees(rel[k]), rel_max_at=math.degrees(th[k]),
                rhomb=[math.degrees(c) for c in cross], gh_45=d_gh[i45], gh_45_lam=d_gh[i45] / lam,
                qgy_45=s3, m_refl_normal=m_from_qg(2 * r_normal - 1))


# ---------------------------------------------------------------- 9. Bogoliubov / Unruh
def bogoliubov(rs=(0.25, 0.5, 1.0, 1.5), nmax=80):
    rows = []
    for r in rs:
        lam = math.tanh(r)
        pn = (1 - lam ** 2) * lam ** (2 * np.arange(nmax))       # reduced thermal distribution
        s_num = float(-(pn[pn > 0] * np.log(pn[pn > 0])).sum())
        c2, s2 = math.cosh(r) ** 2, math.sinh(r) ** 2
        s_closed = c2 * math.log(c2) - s2 * math.log(s2)
        p_vac = 1 / c2
        theta = 2 * math.acos(math.sqrt(p_vac))
        gd = math.atan(math.sinh(r))
        rows.append(dict(r=r, nbar=s2, s_num=s_num, s_closed=s_closed, m_vac=m_from_qg(2 * p_vac - 1),
                         half=theta / 2, gd=gd, boltz=lam ** 2))
    m = np.array([[math.cosh(0.7), math.sinh(0.7)], [math.sinh(0.7), math.cosh(0.7)]])
    k = np.diag([1.0, -1.0])
    return dict(rows=rows, su11_err=np.abs(m.T @ k @ m - k).max())


# ---------------------------------------------------------------- 10. Jarzynski
def jarzynski(beta=1.0, h0=1.0, h1=2.0, phi=1.0, tau=1.5, steps=400, samples=(10, 100, 1000, 10000), reps=2000):
    def ham(h, ang):
        return -0.5 * h * (math.cos(ang) * Z + math.sin(ang) * X)
    hi, hf = ham(h0, 0.0), ham(h1, phi)
    dt = tau / steps
    u = np.eye(2, dtype=complex)
    for s in range(steps):
        f = (s + 0.5) / steps
        u = expm(-1j * dt * ham(h0 + f * (h1 - h0), f * phi)) @ u
    ei, vi = np.linalg.eigh(hi)
    ef, vf = np.linalg.eigh(hf)
    pi = np.exp(-beta * ei)
    zi = pi.sum()
    pi /= zi
    zf = np.exp(-beta * ef).sum()
    trans = np.abs(vf.conj().T @ u @ vi) ** 2            # trans[n, m] = |<n_f|U|m_i>|^2
    works, probs = [], []
    for m_ in range(2):
        for n_ in range(2):
            works.append(ef[n_] - ei[m_])
            probs.append(pi[m_] * trans[n_, m_])
    works, probs = np.array(works), np.array(probs)
    lhs = float((probs * np.exp(-beta * works)).sum())
    rhs = zf / zi
    d_f = -math.log(rhs) / beta
    mean_w = float((probs * works).sum())
    qg_trans = [2 * trans[0, 0] - 1, 2 * trans[1, 1] - 1]
    est = []
    for ns in samples:
        idx = RNG.choice(4, size=(reps, ns), p=probs)
        val = -np.log(np.exp(-beta * works[idx]).mean(axis=1)) / beta
        est.append((ns, float(val.mean() - d_f), float(val.std())))
    return dict(lhs=lhs, rhs=rhs, err=abs(lhs - rhs), d_f=d_f, mean_w=mean_w, qg_trans=qg_trans, est=est,
                p_neg=float(probs[works < d_f].sum()))


# ---------------------------------------------------------------- 11. Fisher information of GHZ
def ghz_fisher(ns=(1, 2, 3, 4, 6, 8), phi=0.3):
    rows = []
    for n in ns:
        d = 2 ** n
        gen = sum(site(Z, k, n) for k in range(n)) / 2
        ghz = np.zeros(d, complex)
        ghz[0] = ghz[-1] = 1 / math.sqrt(2)
        plus = np.ones(d, complex) / math.sqrt(d)

        def qfi(psi):
            u = expm(-1j * phi * gen)
            p = u @ psi
            dp = -1j * gen @ p
            return float(4 * (np.real(dp.conj() @ dp) - abs(p.conj() @ dp) ** 2))
        # parity reading of GHZ after phase: qg_parity = cos(n phi); classical Fisher of a +-1 reading
        q = math.cos(n * phi)
        dq = -n * math.sin(n * phi)
        f_cl = dq ** 2 / (1 - q ** 2)
        rows.append((n, qfi(ghz), qfi(plus), f_cl))
    return rows


# ---------------------------------------------------------------- 12. Fubini-Study
def fubini_study(k=20000):
    th = np.linspace(0, np.pi, k + 1)
    states = np.stack([np.cos(th / 2), np.sin(th / 2)], axis=1)
    ov = np.abs(np.sum(states[:-1] * states[1:], axis=1))
    length_fs = float(np.sum(np.arccos(np.clip(ov, -1, 1))))       # sum of FS distances arccos|<a|b>|
    length_ds = float(np.sum(np.sqrt(1 - ov ** 2)))                 # sum of sqrt(1 - |<a|b>|^2)
    # Gaussian curvature of ds^2 = E dth^2 + G dph^2, E = 1/4, G = sin^2 th / 4 (Brioschi for orthogonal metrics)
    t0 = 0.7
    h = 1e-4
    sqrt_g = lambda t: abs(math.sin(t)) / 2  # noqa: E731
    sqrt_e = 0.5
    d2 = (sqrt_g(t0 + h) - 2 * sqrt_g(t0) + sqrt_g(t0 - h)) / h ** 2
    kgauss = -d2 / (sqrt_g(t0) * sqrt_e ** 2)
    # meridian length in the qg coordinate: ds = dqg / (2 sqrt(1 - qg^2))
    q = np.cos(th)
    dq = np.abs(np.diff(q))
    mid = np.cos((th[:-1] + th[1:]) / 2)
    length_qg = float(np.sum(dq / (2 * np.sqrt(1 - mid ** 2))))
    return dict(length_fs=length_fs, length_ds=length_ds, length_qg=length_qg, k=kgauss)


# ---------------------------------------------------------------- 13. OTOC
def otoc(n=8, ts=None, hz=0.5, g=1.05):
    ts = np.linspace(0, 30, 301) if ts is None else ts
    res = {}
    for name, hzz in (("integrable", 0.0), ("chaotic", hz)):
        h = tfim_h(n, g, hz=hzz)
        w, v = np.linalg.eigh(h)
        wop = site(Z, 0, n)
        vop = site(Z, n - 1, n)
        wv = v.conj().T @ wop @ v
        vv = v.conj().T @ vop @ v
        f = []
        for t in ts:
            ph = np.exp(-1j * w * t)
            wt = (ph.conj()[:, None] * wv) * ph[None, :]       # W(t) in the eigenbasis
            f.append(float(np.real(np.trace(wt @ vv @ wt @ vv)) / 2 ** n))
        # level spacing ratio in the even-parity, reflection-symmetric sector is not resolved here: use all levels of
        # the spin-flip-even sector (hz = 0) or all levels (hz != 0) as a rough diagnostic
        e = np.sort(w)
        sp = np.diff(e)
        sp = sp[sp > 1e-9]
        r = np.minimum(sp[1:], sp[:-1]) / np.maximum(sp[1:], sp[:-1])
        res[name] = dict(t=ts, f=np.array(f), late=float(np.mean(f[len(ts) // 3:])), r=float(r.mean()),
                         t_half=float(ts[np.argmax(np.array(f) < 0.5)]))
    return res


def level_ratio_sector(n=10, g=1.05, hz=0.5):
    """Mean level-spacing ratio in the reflection-even, momentum-free open chain: resolve reflection only."""
    h = tfim_h(n, g, hz=hz)
    refl = np.zeros((2 ** n, 2 ** n))
    for b in range(2 ** n):
        rb = int(format(b, "0%db" % n)[::-1], 2)
        refl[rb, b] = 1
    w, v = np.linalg.eigh(refl)
    even = v[:, w > 0]
    he = even.T @ h @ even
    e = np.linalg.eigvalsh(he)
    sp = np.diff(e)
    r = np.minimum(sp[1:], sp[:-1]) / np.maximum(sp[1:], sp[:-1])
    return float(r.mean())


# ---------------------------------------------------------------- figures
TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "t_g": r"$g/J$", "t_dent": r"$-dS/dg$ (entropía de media cadena)", "t_dmx": r"$d\,qg_X/dg$",
        "t_title1": "Entropía: el pico se corre con N", "t_title2": "Lectura qg_X: el pico se corre con N",
        "o_t": r"tiempo $Jt$", "o_f": r"$F(t)=\mathrm{Re}\,\langle W(t)VW(t)V\rangle$",
        "o_title": "OTOC en una cadena de 8 espines", "o_int": "integrable (Ising transverso)",
        "o_cha": "caótica (campo inclinado)",
        "a_f": r"flujo $\Phi/\Phi_0$", "a_m": "lectura (m-qg)", "a_s": r"desvío de $\hat\Phi/\Phi_0$ $\times\sqrt{N}$",
        "a_title1": "Aharonov–Bohm: una franja por cuanto", "a_title2": "El ángulo estabiliza el error",
        "a_bound": r"$1/2\pi$",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "t_g": r"$g/J$", "t_dent": r"$-dS/dg$ (half-chain entropy)", "t_dmx": r"$d\,qg_X/dg$",
        "t_title1": "Entropy: the peak moves with N", "t_title2": "qg_X reading: the peak moves with N",
        "o_t": r"time $Jt$", "o_f": r"$F(t)=\mathrm{Re}\,\langle W(t)VW(t)V\rangle$",
        "o_title": "OTOC on an 8-spin chain", "o_int": "integrable (transverse Ising)",
        "o_cha": "chaotic (tilted field)",
        "a_f": r"flux $\Phi/\Phi_0$", "a_m": "reading (m-qg)", "a_s": r"spread of $\hat\Phi/\Phi_0$ $\times\sqrt{N}$",
        "a_title1": "Aharonov–Bohm: one fringe per quantum", "a_title2": "The angle stabilizes the error",
        "a_bound": r"$1/2\pi$",
    },
}


def figures(tf, ot, ab):
    colors = [GREY, ORANGE, GREEN, BLUE, INK, "#7a4fc4"]
    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for c, (n, d) in zip(colors, tf.items()):
            ax[0].plot(d["g"], -np.gradient(d["ent"], d["g"]), color=c, label="N = %d" % n)
            ax[1].plot(d["g"], np.gradient(d["mx"], d["g"]), color=c, label="N = %d" % n)
        for a, t, y in ((ax[0], tx["t_title1"], tx["t_dent"]), (ax[1], tx["t_title2"], tx["t_dmx"])):
            a.axvline(1.0, color=GREY, ls=":", lw=1)
            a.set_xlabel(tx["t_g"])
            a.set_ylabel(y)
            a.set_title(t)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(tx["dir"], "cap34_ising.png"), dpi=180, facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(ot["integrable"]["t"], ot["integrable"]["f"], color=BLUE, label=tx["o_int"])
        ax.plot(ot["chaotic"]["t"], ot["chaotic"]["f"], color=ORANGE, label=tx["o_cha"])
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel(tx["o_t"])
        ax.set_ylabel(tx["o_f"])
        ax.set_title(tx["o_title"])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(tx["dir"], "cap34_otoc.png"), dpi=180, facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        f = np.linspace(0, 2, 401)
        ax[0].plot(f, 1000 * np.cos(2 * np.pi * f), color=BLUE)
        ax[0].set_xlabel(tx["a_f"])
        ax[0].set_ylabel(tx["a_m"])
        ax[0].set_title(tx["a_title1"])
        fl = [r[0] for r in ab["rows"]]
        ax[1].plot(fl, [r[2] for r in ab["rows"]], "o-", color=ORANGE)
        ax[1].axhline(ab["bound"], color=GREY, ls=":", label=tx["a_bound"])
        ax[1].set_xlabel(tx["a_f"])
        ax[1].set_ylabel(tx["a_s"])
        ax[1].set_ylim(0, 0.25)
        ax[1].set_title(tx["a_title2"])
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(tx["dir"], "cap34_ab.png"), dpi=180, facecolor="white")
        plt.close(fig)


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("1 JW", jordan_wigner())
    print("2 UCC", uccd_h2())
    print("3 Larmor", larmor())
    print("3 Hahn", hahn_echo())
    tf = tfim_scan()
    print("4 TFIM steepest entropy drop, qg_X susceptibility peak, m_X at g=1",
          {n: (round(d["g_ent"], 3), round(d["g_mx"], 3), round(d["m_x_at_1"], 1)) for n, d in tf.items()})
    print("4 TFIM thermodynamic m_X at g = 1: 1000 * 2/pi =", 2000 / math.pi)
    print("5 Kitaev clean", kitaev_splitting())
    print("5 Kitaev mu=0 disorder", kitaev_splitting(mu=0.0, disorder=0.3))
    print("5 braid", braid_gate())
    ab = aharonov_bohm()
    print("6 AB", ab)
    print("7 Wigner", wigner())
    print("7 Wigner parallel", wigner((0.6, 0, 0), (0.5, 0, 0)))
    print("8 Fresnel", fresnel_tests())
    print("9 Bogoliubov", bogoliubov())
    print("10 Jarzynski", jarzynski())
    print("11 GHZ", ghz_fisher())
    print("12 FS", fubini_study())
    ot = otoc()
    print("13 OTOC late mean, first t with F < 0.5", {k: (round(v["late"], 3), v["t_half"]) for k, v in ot.items()})
    print("13 r sector chaotic", level_ratio_sector(), "integrable", level_ratio_sector(hz=0.0))
    figures(tf, ot, ab)


if __name__ == "__main__":
    main()
