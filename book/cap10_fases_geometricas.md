# Capítulo 10 — Fases geométricas y el teorema de Stokes en la esfera: holonomía en qga, invariante de Bargmann y lectura con una sola base
# Chapter 10 — Geometric Phases and Stokes' Theorem on the Sphere: Holonomy in qga, Bargmann Invariant and Single-Basis Readout

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 7 sobre la base de Cap. 1–9. Módulo: `qang.qgeom` (JAX, `jax<0.9`; alineado con `qang.geometric`, `qang.phase`). Verificación: `tools/verify_cap10.py`.

---

## 10.1 Curvatura de Berry como monopolo de carga ½ en coordenadas qga

**ES.** Con $|\psi(\alpha,\varphi)\rangle=\cos\tfrac\theta2|0\rangle+e^{i\phi}\sin\tfrac\theta2|1\rangle$, $\theta=2\pi\alpha/1000$, $\phi=2\pi\varphi/1000$ (ambos en qga), la conexión y la curvatura de Berry son
$$A=i\langle\psi|d\psi\rangle=-\sin^2\tfrac\theta2\,d\phi=-\tfrac{1-qg_Z}{2}\,d\phi,\qquad F=dA=\tfrac12\sin\theta\,d\theta\wedge d\phi .$$
Teorema de Stokes: para todo lazo cerrado $\mathcal C=\partial\Sigma$,
$$\gamma=\oint_{\mathcal C}A=-\iint_\Sigma F=-\tfrac12\,\Omega(\Sigma),$$
con $\Omega$ el ángulo sólido orientado. En la coordenada de lectura la 1-forma es **lineal en $qg_Z$**: un lazo de latitud (una vuelta en $\varphi$: $1000$ qga) da
$$\gamma_{\text{lat}}=-\pi\,(1-qg_Z)\quad\text{[rad]}\qquad(\texttt{berry\_phase\_latitude}),$$
de modo que la fase geométrica se lee con **una sola base** ($Z$) y sin ancilla (`exact`). Curvatura total $\iint_{S^2}F=2\pi$: carga $\tfrac12$ (Cap. 1: curvatura métrica de Fubini–Study $=4$ es otra cantidad; no confundir).

**EN.** Berry's connection is linear in the readout coordinate $qg_Z$; the latitude-loop phase $\gamma=-\pi(1-qg_Z)$ follows from Stokes' theorem and is read from a single $Z$ histogram (`exact`). The Berry curvature is a charge-½ monopole, distinct from the Fubini–Study metric curvature $4$ of Chapter 1.

## 10.2 Invariante de Bargmann y fase de Pancharatnam

$$\gamma_{123}=\arg\big(\langle\psi_2|\psi_1\rangle\langle\psi_3|\psi_2\rangle\langle\psi_1|\psi_3\rangle\big)=-\tfrac12\,\Omega(\triangle_{123}),\qquad \Omega=2\arctan\frac{\mathbf r_1\cdot(\mathbf r_3\times\mathbf r_2)}{1+\mathbf r_1\!\cdot\mathbf r_2+\mathbf r_2\!\cdot\mathbf r_3+\mathbf r_3\!\cdot\mathbf r_1}$$
(Van Oosterom–Strackee; $\mathbf r_k$ = vectores de Bloch $(qg_X,qg_Y,qg_Z)$). Invariante de gauge a $10^{-12}$ y convergente al límite continuo como $O(K^{-2})$ para un polígono de $K$ vértices (`exact`). Convención fijada en código: $\gamma=i\oint\langle\psi|d\psi\rangle$; con $\varphi$ creciente, $\gamma<0$ y $\Omega>0$.

## 10.3 Compuerta Z geométrica frente a Z virtual: hallazgo negativo de área

Una fase relativa $\Omega$ entre $|0\rangle$ y $|1\rangle$ ($R_Z(\Omega)$) se obtiene (a) **virtual**: $0$ pulsos, $0$ qga de área (Cap. 2); (b) **física**: $1$ pulso, área $\Omega\cdot1000/2\pi$ qga; (c) **geométrica** por el lazo polo $\to(\theta,0)\to(\theta,\Delta\phi)\to$ polo: $3$ pulsos, área $(2\theta+d)\cdot1000/2\pi$ qga con $d=\arccos(\cos^2\theta+\sin^2\theta\cos\Delta\phi)$. Instancia ($\theta=150$ qga, $\Delta\phi=0.6$ rad): $0$ / $39.4$ / $376.8$ qga. **Estatus**: área `no gain` (×9.6 peor que la física); la robustez holonómica frente a ruido de amplitud queda `unverified` en este tratado (no se afirma).

## 10.4 Presupuesto de disparos para medir $\gamma$

Lazo de latitud: $\sigma_\gamma=\pi\,\sigma_{qg_Z}=\pi\sqrt{(1-qg_Z^2)/N}$ ⇒ $N=\pi^2(1-qg_Z^2)/\sigma_\gamma^2$ (1 base, 0 ancillas). Lazo general de 3 estados: tomografía de $\mathbf r_{1,2,3}$, $\ge9$ configuraciones de medida. Nótese que la precisión **mejora hacia los polos** ($qg_Z\to\pm1$), al contrario que la estimación angular del Cap. 9 (uniforme en qga): $\gamma$ es lineal en $qg_Z$, no en $\alpha$.

## 10.5 Contraste operativo — Sin `qang` vs. con `qang`

| Métrica / Metric | Estándar (sin `qang`) | `qang` (`qang.qgeom` + JAX) | Ganancia / Estatus |
| :--- | :--- | :--- | :--- |
| Lectura de $\gamma$ en lazo de latitud | interferometría con ancilla o tomografía (≥3 bases) | $\gamma=-\pi(1-qg_Z)$: 1 base, 0 ancillas | $-2$ bases, $-1$ ancilla (`exact`) |
| Disparos para $\sigma_\gamma=0.01$ rad | — | $N=9\,425$ ($\alpha=50$ qga) / $64\,597$ ($150$) / $98\,696$ ($250$) | (`exact`) |
| Holonomía de lazo discreto de $K$ vértices | producto de $K$ unitarias | $\sum$ triángulos geodésicos (`solid_angle_polygon`), error $2.7\cdot10^{-5}$ a $K=384$ | (`exact`, $O(K^{-2})$) |
| $R_Z(\Omega)$ geométrica vs virtual | 3 pulsos, $376.8$ qga | virtual: $0$ pulsos | geométrica: `no gain`; robustez `unverified` |

## 10.6 Bloque JAX (`qang/qgeom.py`)

```python
@jax.jit
def berry_phase_latitude(qz):            # γ = −π(1 − qg_Z)  [rad], Stokes exacto
    return -jnp.pi * (1.0 - qz)

@jax.jit
def solid_angle_triangle(r1, r2, r3):    # Van Oosterom–Strackee, orientación φ creciente ⇒ Ω>0
    num = jnp.dot(r1, jnp.cross(r3, r2))
    den = 1.0 + jnp.dot(r1, r2) + jnp.dot(r2, r3) + jnp.dot(r3, r1)
    return 2.0 * jnp.arctan2(num, den)

@jax.jit
def bargmann_phase(p1, p2, p3):          # arg(<ψ2|ψ1><ψ3|ψ2><ψ1|ψ3>) = −Ω/2
    return jnp.angle(jnp.vdot(p2, p1) * jnp.vdot(p3, p2) * jnp.vdot(p1, p3))

def shots_for_berry_latitude(qz, sigma_gamma):   # N = π²(1−qg_Z²)/σ_γ²
    return float(jnp.pi ** 2 * (1.0 - qz ** 2) / sigma_gamma ** 2)
```

## 10.7 Verificación — `tools/verify_cap10.py` (salida reproducible)

```
[1] K= 384: γ_disc=-1.294984  γ_Stokes=-1.295011  |Δ|=2.70e-05
    Ω polígono=2.589968 = 2π(1−qg_Z)=2.590022  (qg_Z=0.5878, α=150 qga)
[2] arg(<ψ2|ψ1><ψ3|ψ2><ψ1|ψ3>) = −Ω/2 en 5/5 triángulos, invariante de gauge a 1e-12
[3] ∬F = 1.295010823 = −γ ✔ (monopolo de carga ½)
[4] R_Z(Ω=0.2473 rad = 39.4 qga): virtual 0 qga | física 39.4 qga (1 pulso) | geométrica 376.8 qga (3 pulsos)
[5] α= 50 qga (qg_Z=+0.951): N = 9425 ; α=150: 64597 ; α=250: 98696
verify_cap10: OK
```

### `tests/test_qgeom.py`
```python
import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qgeom import state_from_qga, bloch_from_state, bargmann_phase, solid_angle_triangle, berry_phase_latitude

def test_bargmann_equals_minus_half_solid_angle():
    ps = [state_from_qga(a, p) for a, p in ((40., 10.), (200., 300.), (330., 700.))]
    Om = float(solid_angle_triangle(*[bloch_from_state(q) for q in ps]))
    assert abs(float(bargmann_phase(*ps)) + Om / 2) < 1e-12

def test_latitude_equator():
    assert abs(float(berry_phase_latitude(0.0)) + np.pi) < 1e-15
```

## 10.8 Apéndice H — entradas nuevas en `qang.measures`

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `berry_latitude_qz` | geometría | $\gamma=-\pi(1-qg_Z)$ | `berry_phase_latitude` | `exact` (1 base, 0 ancillas) |
| `bargmann_triangle` | geometría | $\arg\prod\langle\psi_{k+1}|\psi_k\rangle=-\Omega/2$ | `bargmann_phase`, `solid_angle_triangle` | `exact` |
| `berry_shots_latitude` | estadística | $N=\pi^2(1-qg_Z^2)/\sigma_\gamma^2$ | `shots_for_berry_latitude` | `exact` |
| `holonomic_z_area` | compilación | área $(2\theta+d)$ vs $0$ virtual | `verify_cap10.py` §4 | `no gain`; robustez `unverified` |

### `CHANGELOG.md` — `[Unreleased]`
```
### Added
- `qang.qgeom`: `state_from_qga`, `bloch_from_state`, `berry_phase_latitude`, `solid_angle_triangle`,
  `solid_angle_polygon`, `bargmann_phase`, `shots_for_berry_latitude` (JAX, `jax<0.9`).
- `tools/verify_cap10.py`, `tests/test_qgeom.py`; Apéndice H: 4 entradas.
```
Commit: `docs(cap10): fases geométricas y Stokes reescritos sobre Cap. 1–9 + qang.qgeom + verify_cap10`.
