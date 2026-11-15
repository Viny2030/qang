# Capítulo 13 — Efectos de materia (MSW): la resonancia como ecuador, la lectura adiabática como producto de dos qg y el coste de Trotter
# Chapter 13 — Matter Effects (MSW): Resonance as the Equator, Adiabatic Readout as a Product of Two qg Values, and the Trotter Cost

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 10 sobre la base de Cap. 1–12. Módulo: `qang.qneutrino` (parte materia; JAX, `jax<0.9`). Verificación: `tools/verify_cap13.py`. Oscilaciones colectivas (§156, §163) quedan fuera de este capítulo: estatus `unverified`, sin afirmaciones.

---

## 13.1 Hamiltoniano efectivo y resonancia en qga

**ES.** En materia con densidad electrónica $N_e$ el potencial $A=2\sqrt2G_FN_eE$ ($=1.52\cdot10^{-4}\,\mathrm{eV^2}\cdot(N_e/N_A\,\mathrm{cm^{-3}})\cdot E[\mathrm{GeV}]$) inclina el eje de masa efectivo:
$$\cos2\theta_m=\frac{\cos2\theta-A/\Delta m^2}{\sqrt{(\cos2\theta-A/\Delta m^2)^2+\sin^22\theta}},\qquad \sin^22\theta_m=\frac{\sin^22\theta}{(\cos2\theta-A/\Delta m^2)^2+\sin^22\theta}.$$
$\cos2\theta_m$ **es** el $qg_Z$ del eje efectivo. La resonancia $A_{\text{res}}=\Delta m^2\cos2\theta$ equivale a $\theta_m=\pi/4$: el eje efectivo cruza el **ecuador** ($qg_Z=0$, $250$ qga). Para $\theta_{12}$: $2\theta_m$ va de $185.6$ qga (vacío) a $250.0$ qga (resonancia); $E_{\text{res}}=1.92$ MeV a $N_e=100N_A$ cm$^{-3}$ (`exact`).

**EN.** The MSW resonance is the equator crossing of the effective mass axis; $\cos2\theta_m$ is literally the $qg_Z$ of that axis (`exact`).

## 13.2 Lectura adiabática: producto de dos qg

Para propagación adiabática desde producción (densidad $N_e^{\text{prod}}$) hasta el vacío,
$$P_{ee}=\tfrac12\big(1+\cos2\theta_m^{\text{prod}}\cos2\theta\big)\quad\Longleftrightarrow\quad qg_Z^{(ee)}=qg_Z^{\text{eje}}(A_{\text{prod}})\cdot qg_Z^{\text{eje}}(0),$$
un **producto de dos lecturas qg** del eje efectivo. Neutrinos solares ($N_e^{\text{prod}}=100N_A$):

| fuente | $A/A_{\text{res}}$ | $\cos2\theta_m$ | $P_{ee}$ | lectura |
| :--- | :--- | :--- | :--- | :--- |
| pp $0.4$ MeV | $0.21$ | $+0.321$ | $0.563$ | $+127$ m-qg |
| $^7$Be $0.86$ MeV | $0.45$ | $+0.231$ | $0.545$ | $+91$ m-qg |
| $^8$B $10$ MeV | $5.20$ | $-0.874$ | $0.328$ | $-344$ m-qg |

Límites reproducidos: $P_{ee}\to1-\tfrac12\sin^22\theta$ (promedio de vacío) a baja $E$ y $P_{ee}\to\sin^2\theta$ a alta $E$ (`exact` bajo adiabaticidad, $\gamma_{\text{res}}\approx2.7\cdot10^3\gg1$).

## 13.3 Coste de simular el perfil: Trotter frente a la fórmula de dos qg

Evolución por tramos con $H(L)\propto\big[-\cos2\theta+A(L)/\Delta m^2,\ \sin2\theta\big]$ sobre $N_e(r)=100N_A\,e^{-r/r_0}$, $r_0=R_\odot/10.54$: cada tramo es una rotación de eje arbitrario (2 pulsos $R_Y$ + $R_Z$ virtual). Convergencia a $\langle P_{ee}\rangle$ (banda $\pm10\%$ en $E$, que lava la fase de vacío):

| $N$ tramos | pulsos | $\langle P_{ee}\rangle$ | $|\Delta|$ vs adiabático |
| :--- | :--- | :--- | :--- |
| 50 | 100 | 0.3186 | 0.0092 |
| 200 | 400 | 0.3403 | 0.0124 |
| 1000 | 2000 | 0.3266 | 0.0012 |
| 4000 | 8000 | 0.3294 | 0.0015 |
| adiabático | **0** (2 valores qg) | 0.3278 | — |

Cuando $\gamma\gg1$, la lectura de dos qg sustituye $\ge2000$ pulsos por una identidad algebraica (`exact`); cuando $\gamma\lesssim1$ (perfiles abruptos, supernovas) la fórmula falla y el Trotter es obligatorio — la decisión se toma con $\gamma$, no a ciegas.

## 13.4 Contraste operativo — Sin `qang` vs. con `qang`

| Métrica / Metric | Estándar (sin `qang`) | `qang` (`qang.qneutrino` + JAX) | Ganancia / Estatus |
| :--- | :--- | :--- | :--- |
| Localizar la resonancia | barrido de $\sin^22\theta_m$ | $qg_Z^{\text{eje}}=0$ (250 qga) | (`exact`) |
| $P_{ee}$ solar adiabático | integrar perfil ($N\ge1000$ tramos, $\ge2000$ pulsos) | producto de 2 qg: $0$ tramos | $-2000$ pulsos a $\Delta<1.5\cdot10^{-3}$ (`exact`, $\gamma\gg1$) |
| Lectura $^8$B / pp | $P_{ee}=0.328/0.563$ | $-344$ / $+127$ m-qg | (`exact`) |
| Criterio de validez | — | $\gamma_{\text{res}}=2.7\cdot10^3$, $P_{LZ}\approx e^{-\pi\gamma/2}$ | (`exact`) |
| Oscilaciones colectivas | — | fuera de alcance | `unverified` |

## 13.5 Bloque JAX (`qang/qneutrino.py`, parte materia)

```python
@jax.jit
def cos2theta_m(theta, dm2, A):              # qg_Z del eje efectivo
    s, c = jnp.sin(2 * theta), jnp.cos(2 * theta); x = c - A / dm2
    return x / jnp.sqrt(x ** 2 + s ** 2)

@jax.jit
def adiabatic_survival_qz(theta, dm2, A_prod):   # qg_Z^{ee} = cos2θ_m(A_prod)·cos2θ
    return cos2theta_m(theta, dm2, A_prod) * jnp.cos(2 * theta)

def trotter_survival_qz(theta, dm2, E, A_profile, dL_km):   # lax.scan de rotaciones de eje arbitrario
    s, c = jnp.sin(2 * theta), jnp.cos(2 * theta)
    def step(psi, A):
        hz, hx = (-c + A / dm2), s; h = jnp.sqrt(hz ** 2 + hx ** 2)
        ang = 1.267 * dm2 * dL_km / E * h
        U = jnp.cos(ang) * jnp.eye(2) - 1j * jnp.sin(ang) * jnp.array([[hz, hx], [hx, -hz]]) / h
        return U @ psi, None
    psi, _ = jax.lax.scan(step, jnp.array([1.0 + 0j, 0j]), A_profile)
    return 2.0 * jnp.abs(psi[0]) ** 2 - 1.0
```

## 13.6 Verificación — `tools/verify_cap13.py` (salida reproducible)

```
[1] A_res = 2.923e-05 eV² → cos2θ_m = 0 (250 qga); E_res(N_e=100 N_A) = 1.92 MeV; 2θ_m en qga: vacío 185.6 → resonancia 250.0
[2] pp 0.4 MeV   : A/A_res= 0.21  cos2θ_m=+0.321  P_ee=0.563  qg_Z=+0.127 → +127 m-qg
[2] 8B 10 MeV    : A/A_res= 5.20  cos2θ_m=-0.874  P_ee=0.328  qg_Z=-0.344 → -344 m-qg
[3] Trotter N= 1000 tramos (2000 pulsos): <P_ee>=0.3266 | adiabático (0 tramos, 2 qg): 0.3278 | Δ=0.0012
    γ_adiab(res) = 2661 ≫ 1 → P_LZ ≈ e^(−πγ/2) ≈ 0
verify_cap13: OK
```
Asserts: $\cos2\theta_m(A_{\text{res}})=0$ y $\sin^22\theta_m=1$ a $10^{-14}$; límites pp/$^8$B a $0.02/0.03$; $|\langle P_{ee}\rangle_{N=4000}-P_{\text{ad}}|<0.01$.

### `tests/test_qneutrino_matter.py`
```python
import pytest; jax = pytest.importorskip("jax")
import numpy as np
from qang.qneutrino import cos2theta_m, resonance_A, adiabatic_survival_qz

def test_resonance_is_equator():
    th, dm2 = 0.583, 7.42e-5
    assert abs(float(cos2theta_m(th, dm2, resonance_A(th, dm2)))) < 1e-14

def test_vacuum_limit():
    th, dm2 = 0.583, 7.42e-5
    assert abs(float(adiabatic_survival_qz(th, dm2, 0.0)) - np.cos(2 * th) ** 2) < 1e-14
```

## 13.7 Apéndice H — entradas nuevas en `qang.measures`

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `msw_axis_qz` | neutrinos | $\cos2\theta_m$ | `cos2theta_m` | `exact` |
| `msw_resonance_equator` | neutrinos | $A_{\text{res}}=\Delta m^2\cos2\theta\Rightarrow qg_Z=0$ | `resonance_A` | `exact` |
| `adiabatic_two_qg` | compilación | $qg_Z^{(ee)}=qg_Z(A_{\text{prod}})\,qg_Z(0)$; 0 tramos | `adiabatic_survival_qz` | `exact` si $\gamma\gg1$ |
| `trotter_profile_cost` | compilación | $2N$ pulsos, $\Delta(N)$ | `trotter_survival_qz` | `exact` sim. |
| `collective_oscillations` | neutrinos | — | — | `unverified` |

### `CHANGELOG.md` — `[Unreleased]`
```
### Added
- `qang.qneutrino` (parte materia): `matter_potential_A`, `sin2_2theta_m`, `cos2theta_m`, `resonance_A`,
  `adiabatic_survival_qz`, `trotter_survival_qz` (JAX `lax.scan`).
- `tools/verify_cap13.py`, `tests/test_qneutrino_matter.py`; Apéndice H: 5 entradas.
```
Commit: `docs(cap13): efectos de materia MSW reescritos sobre Cap. 1–12 + verify_cap13`.
