# Capítulo 12 — Oscilación de sabor en el vacío: un qubit por par de sabores, lectura en m-qg y punto de operación para Δ m²
# Chapter 12 — Vacuum Flavor Oscillation: One Qubit per Flavor Pair, m-qg Readout and the Operating Point for Δ m²

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 9 sobre la base de Cap. 1–11. Módulo: `qang.qneutrino` (JAX, `jax<0.9`; alineado con `qang.neutrino`). Verificación: `tools/verify_cap12.py`. Hardware: §165 IQM Garnet (`tested on hardware`).

---

## 12.1 Dos sabores = un qubit: la fase de oscilación en qga

**ES.** El estado de sabor $|\nu_e\rangle=\cos\theta|\nu_1\rangle+\sin\theta|\nu_2\rangle$ es un qubit cuyo eje de masa forma un ángulo de Bloch $2\theta$ con el eje de sabor. La propagación es una rotación $R_Z(\phi)$ en la base de masa,
$$\phi=2\cdot1.267\,\frac{\Delta m^2[\mathrm{eV^2}]\,L[\mathrm{km}]}{E[\mathrm{GeV}]}\ \text{rad},\qquad \phi_{\text{qga}}=\frac{1000}{2\pi}\phi,\qquad L_{\text{osc}}=\frac{4\pi E}{\Delta m^2}\equiv1000\ \text{qga}.$$
Circuito exacto: $R_Y(-2\theta)\,R_Z(\phi)\,R_Y(2\theta)|0\rangle$ — dos pulsos físicos y una $R_Z$ **virtual** (Cap. 2). Lectura:
$$P(\nu_\alpha\to\nu_\beta)=\sin^22\theta\,\sin^2\tfrac\phi2,\qquad qg_Z=1-2P,\qquad m\text{-qg}=1000-2000P .$$
Toda la fenomenología de dos sabores es, por tanto, una lectura `m-qg` de un qubit con un solo parámetro angular $\phi_{\text{qga}}$ (`exact`).

**EN.** Two-flavor oscillation is one qubit: mass axis tilted by $2\theta$, propagation a virtual $R_Z(\phi)$, and the transition probability an m-qg readout. One oscillation length is one qga turn (`exact`).

## 12.2 Punto de operación: θ frente a Δ m²

Fisher por evento (Cap. 9, vía `jax.grad`):
$$I_F(\theta)=\frac{(\partial_\theta P)^2}{P(1-P)},\quad\max\ \text{en }\phi=\pi\ (\text{máximo de oscilación}),\qquad I_F(\phi)=\frac{\sin^22\theta\cos^2\frac\phi2}{1-\sin^22\theta\sin^2\frac\phi2},\quad\text{decreciente en }[0,\pi],\ I_F(\pi)=0 .$$
**Hallazgo**: el máximo de oscilación, óptimo para $\theta$ ($16.0$/evento para $\theta_{12}$), es el **peor** punto para $\Delta m^2$ ($1.4\cdot10^{-6}$/evento). En $\phi=\pi/2$ (fase $250$ qga) $I_F(\phi)=0.731$ frente a $0.444$ en $\phi=0.75\pi$: $\times1.65$ menos eventos para la misma $\sigma_{\Delta m^2}$ (`exact`, por evento; el flujo y la sección eficaz entran aparte).

## 12.3 Tres sabores en dos qubits y el filtro de sector

Codificación $e,\mu,\tau\mapsto|00\rangle,|01\rangle,|10\rangle$; $|11\rangle$ (peso de Hamming 2) es **no físico**: $P(|11\rangle)=0$ idealmente, de modo que todo peso allí es ruido diagnosticable y el filtro de sector (`qang.sectors`) lo descarta y renormaliza. Bajo despolarización $\varepsilon$ el error medio por sabor cae $\times1.18$–$1.22$ en simulación; en hardware (IQM Garnet, §165, $8\times300$ disparos) cayó $\times1.59$ ($0.062\to0.039$) y recuperó el mínimo de $P_{\mu e}$ en $2.6$ GeV (`tested on hardware`; THEORY_CORE §3). PMNS de referencia: $\theta_{12}=33.4^\circ,\theta_{13}=8.57^\circ,\theta_{23}=49^\circ,\delta=195^\circ$, $\Delta m^2_{21}=7.42\cdot10^{-5}$, $\Delta m^2_{31}=2.51\cdot10^{-3}$ eV$^2$.

## 12.4 Contraste operativo — Sin `qang` vs. con `qang`

| Métrica / Metric | Estándar (sin `qang`) | `qang` (`qang.qneutrino` + JAX) | Ganancia / Estatus |
| :--- | :--- | :--- | :--- |
| Circuito de 2 sabores | 3 rotaciones físicas | $R_Y,R_Y$ + $R_Z$ virtual: 2 pulsos | $-1$ pulso (`exact`) |
| Lectura, DUNE-like ($1300$ km, $2.5$ GeV, sector atm.) | $P_{\mu\tau}=0.9739$ | $\phi=526.4$ qga, $-948$ m-qg | (`exact`) |
| Eventos para $\sigma_{\Delta m^2}$ dada, $\theta_{12}$ | en $\phi=0.75\pi$: $I_F=0.444$ | en $\phi=\pi/2$: $0.731$ | $\times1.65$ menos eventos (`exact`, por evento) |
| Error por sabor, 3 sabores / 2 qubits, $\varepsilon=0.10$ | $0.0371$ | filtro de sector: $0.0307$ | $\times1.21$ sim.; hw §165 $\times1.59$ (`tested on hardware`) |
| Qubits para 3 sabores | 3 (one-hot) | 2 + sector físico | $-1$ qubit (`exact`) |

## 12.5 Bloque JAX (`qang/qneutrino.py`)

```python
@jax.jit
def osc_phase_qga(dm2, L, E):                # 1000/(2π) · 2·1.267·Δm² L/E
    return 2.0 * 1.267 * dm2 * L / E / RAD_PER_QGA

@jax.jit
def survival_qz(theta, phi):                 # qg_Z = 1 − 2 sin²2θ sin²(φ/2)
    return 1.0 - 2.0 * jnp.sin(2 * theta) ** 2 * jnp.sin(phi / 2) ** 2

@jax.jit
def fisher_phi(theta, phi):                  # (∂φ P)² / (P(1−P)) vía jax.grad
    p = p_transition(theta, phi); dp = jax.grad(p_transition, argnums=1)(theta, phi)
    return dp ** 2 / jnp.clip(p * (1 - p), 1e-300)

def p3_vacuum(U, m2, L, E, alpha):           # P(ν_α→ν_β) = |Σ_i U_βi e^{−iφ_i} U*_αi|²
    ph = jnp.exp(-1j * 2.0 * 1.267 * m2 * L / E)
    return jnp.abs(U @ (ph * jnp.conj(U)[alpha, :])) ** 2

@jax.jit
def sector_filter(p4):                       # descarta |11>, renormaliza {|00>,|01>,|10>}
    q = p4.at[3].set(0.0); return q / jnp.sum(q)
```

## 12.6 Verificación — `tools/verify_cap12.py` (salida reproducible)

```
[1] circuito 1 qubit = fórmula (1e-14); L_osc(Δm²21, 2.5 GeV) = 83543 km ≡ 1000 qga; 2 pulsos físicos + RZ virtual
[2] θ12: I_F(θ) máx en φ=3.1416 (=π), 16.000/evento
    I_F(φ) por evento: φ→0: 0.8448 (=sin²2θ) | φ=π/2 (250 qga): 0.7313 | φ=0.75π: 0.4436 | φ=π: 1.36e-06
[3] L=1300 km, E=2.5 GeV: φ=3.307 rad = 526.4 qga; P(νμ→ντ)=0.9739; qg_Z=-0.9478 → -948 m-qg
[4] ε=0.05: 0.0185 → 0.0151 (×1.22) | ε=0.10: 0.0371 → 0.0307 (×1.21) | ε=0.20: 0.0742 → 0.0629 (×1.18)
    hardware IQM Garnet §165 (8×300 shots): 0.062 → 0.039 (×1.59) [tested on hardware]
verify_cap12: OK
```

### `tests/test_qneutrino_vacuum.py`
```python
import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qneutrino import osc_phase_qga, survival_qz, pmns, p3_vacuum, sector_filter, embed_two_qubits

def test_one_turn_is_one_oscillation_length():
    dm2, E = 7.42e-5, 2.5; L = 4 * np.pi * E / (2 * 1.267 * 2 * dm2)
    assert abs(float(osc_phase_qga(dm2, L, E)) - 1000.0) < 1e-9

def test_unitarity_three_flavors():
    U = pmns(0.583, 0.1496, 0.855, 3.4); p = p3_vacuum(U, jnp.array([0.0, 7.42e-5, 2.51e-3]), 1300.0, 2.5, 1)
    assert abs(float(p.sum()) - 1) < 1e-12

def test_sector_filter_removes_11():
    q = sector_filter(jnp.array([0.4, 0.3, 0.2, 0.1])); assert float(q[3]) == 0.0 and abs(float(q.sum()) - 1) < 1e-15
```

## 12.7 Apéndice H — entradas nuevas en `qang.measures`

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `osc_phase_qga` | neutrinos | $\phi_{\text{qga}}=\frac{1000}{2\pi}\cdot2.534\,\Delta m^2L/E$ | `osc_phase_qga` | `exact` |
| `survival_mqg` | neutrinos | $m=1000-2000P$ | `survival_qz` | `exact` |
| `dm2_operating_point` | metrología | $I_F(\phi)\downarrow$ en $[0,\pi]$; $I_F(\pi)=0$ | `fisher_phi`, `fisher_theta` | `exact` (por evento) |
| `flavor_sector_filter` | compilación | descarte de $|11\rangle$ | `sector_filter`, `flavor_error` | sim. `exact`; §165 `tested on hardware` |

### `CHANGELOG.md` — `[Unreleased]`
```
### Added
- `qang.qneutrino` (parte vacío): `osc_phase_rad/qga`, `p_transition`, `survival_qz`, `fisher_phi`, `fisher_theta`,
  `pmns`, `p3_vacuum`, `embed_two_qubits`, `depolarize`, `sector_filter`, `flavor_error` (JAX, `jax<0.9`).
- `tools/verify_cap12.py`, `tests/test_qneutrino_vacuum.py`; Apéndice H: 4 entradas.
```
Commit: `docs(cap12): oscilación en el vacío reescrita sobre Cap. 1–11 + qang.qneutrino + verify_cap12`.
