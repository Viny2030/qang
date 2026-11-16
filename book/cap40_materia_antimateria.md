# Capítulo 40 — Materia y antimateria leídas con qang: conjugación de carga, Zitterbewegung, creación de pares y el par invisible
# Chapter 40 — Matter and Antimatter Read with qang: Charge Conjugation, Zitterbewegung, Pair Creation and the Invisible Pair

**Actualizado:** 16 de noviembre de 2026 · **Updated:** November 16, 2026

> Capítulo nuevo, sobre la base de Cap. 1–39 y el Apéndice G. Módulo: `qang.qanti` (NumPy exacto, $c=\hbar=1$). Verificación: `tools/verify_cap40.py`. **Estado: `exact`** para todo lo calculado; **no hay ningún resultado en hardware** y nada de esto es una propuesta nueva sobre la física: son resultados estándar leídos en m-qg. Lo que el Apéndice G ya hace (⟨β⟩ = cos θ_r, CPT en cuatro componentes, el «qubit de Pitágoras») no se repite aquí.
>
> **EN abstract.** On a 1+1 Dirac qubit $H=p\sigma_x+m\sigma_z$, charge conjugation $\psi\mapsto\sigma_y\psi^*$ maps energy $+E$ to $-E$ and flips all three readings $(qg_X,qg_Y,qg_Z)$: the antipodal map of the Bloch sphere, which no unitary implements. Hence no pure one-qubit state is its own antiparticle (max overlap $7\cdot10^{-17}$ over 2000 random states): a Majorana particle needs a larger space. Zitterbewegung is a Rabi problem: $qg_Z(t)=1/\gamma^2+\beta^2\cos2Et$ (mean 640.0, amplitude 360.0 m-qg at $\gamma=1.25$; error $4.5\cdot10^{-13}$). Pair creation per mode is Landau–Zener, $P=e^{-\pi m^2/eE}$ (numerical propagation agrees within $2\cdot10^{-3}$), read as $qg=1-2P$. A spin singlet pair has local readings $0$ and correlation $-1000$ m-qg: it is invisible to single-qubit $qg_Z$ (Chapter 34). Seeing the sign of $\pm1000/\gamma$ at $3\sigma$ costs $N=9(\gamma^2-1)$ shots: 27 at $\gamma=2$, 443 at $\beta=0.99$. No hardware claim.

---

## 40.1 Reglas de lectura

1. **qga no es m-qg.** El ángulo se da en qga (1 vuelta = 1000 qga); las lecturas, en m-qg ($1000\,qg_Z$).
2. **La antipartícula es una lectura invertida, no un ángulo sumado.** $C$ lleva $\theta\mapsto\pi-\theta$ en el polo-ecuador del qubit de Dirac y niega $qg_X,qg_Y,qg_Z$ a la vez.
3. **Lo que se afirma aquí es estándar.** Landau–Zener, Zitterbewegung y la conjugación de carga son físicas conocidas; qang aporta la escala de lectura y el presupuesto de disparos.

## 40.2 Conjugación de carga: el mapa antípoda

Para $H=p\sigma_x+m\sigma_z$, $C\psi=\sigma_y\psi^*$ cumple $\sigma_yH^*\sigma_y=-H$, así que lleva un autoestado de energía $+E$ ($E=\sqrt{p^2+m^2}$) a uno de $-E$ (`conjugate`). En el qubit actúa sobre las tres lecturas como $(qg_X,qg_Y,qg_Z)\mapsto-(qg_X,qg_Y,qg_Z)$: en 500 autoestados al azar, el error es $1{,}1\cdot10^{-13}$ m-qg, y $qg_Z=+1000\,m/E$ en la rama positiva. Ese mapa es **antiunitario**: ninguna compuerta lo implementa (la antípoda de un estado desconocido no se obtiene con una unitaria). Consecuencia exacta: $|\langle\psi|C\psi\rangle|=0$ para todo estado puro de un qubit (máximo observado $6{,}8\cdot10^{-17}$ en 2000 estados), así que **ningún estado puro de un qubit es su propia antipartícula**. Una partícula de Majorana necesita más espacio (Cap. 14, ex-11, fases de Majorana).

## 40.3 Zitterbewegung como un problema de Rabi

Un estado que empieza en $|0\rangle$ (el polo, $qg_Z=+1000$ m-qg) bajo $H$ precede alrededor de $\hat n=(p,0,m)/E$ con frecuencia $2E$:

$$
qg_Z(t)=\frac{m^2}{E^2}+\frac{p^2}{E^2}\cos 2Et=\frac1{\gamma^2}+\beta^2\cos2Et .
$$

Con $p/m=0{,}75$ ($\gamma=1{,}25$, $\beta=0{,}6$) la lectura oscila alrededor de $640{,}0$ m-qg con amplitud $360{,}0$ m-qg; la diagonalización exacta (`zb_qgz_mqg`) coincide con la fórmula (`zb_analytic_mqg`) a $4{,}5\cdot10^{-13}$ m-qg. Es la misma geometría que la regla del área del Cap. 2: el detuning es $m$ y el acoplamiento $p$.

## 40.4 Creación de pares: Landau–Zener

En un campo uniforme, cada modo es un cruce evitado: con $H=\tfrac{vt}{2}\sigma_z+\tfrac\Delta2\sigma_x$, la probabilidad de seguir en el estado diabático es $e^{-\pi\Delta^2/2v}$ (`lz_probability`). Con $\Delta=2m$ y $v=2eE$ da la tasa de Schwinger por modo, $P=e^{-\pi m^2/eE}$ (`pair_probability`). Propagando el sistema $2\times2$ entre $\pm250$ (`lz_numeric`), el acuerdo es de $2{,}0\cdot10^{-3}$ en el peor caso (ventana finita). La ocupación vacío/par se lee como $qg=1-2P$ (`pair_reading_mqg`).

| $m^2/eE$ | $P$ (analítico) | $P$ (numérico) | lectura $1-2P$ (m-qg) |
|---|---|---|---|
| 0,25 | 0,4559 | 0,4541 | $+88{,}1$ |
| 0,64 | 0,1339 | 0,1359 | $+732{,}2$ |
| 1,00 | 0,0432 | 0,0429 | $+913{,}6$ |

: Creación de pares por modo ($c=\hbar=1$). {#tbl-c40-pares}

## 40.5 El par en singlete es invisible en las lecturas locales

Un par partícula–antipartícula en singlete de espín tiene $qg_Z=0$ en cada qubit (0.0 y 0.0 m-qg), pero $qg_{ZZ}=-1000$ m-qg (`singlet_readings`). Es la ceguera de la base $Z$ del Cap. 34: la lectura local no ve el par; la correlación sí.

## 40.6 Cuánto cuesta ver el signo

Distinguir la rama $+1000/\gamma$ de la $-1000/\gamma$ a $z$ desviaciones del cero exige $N=z^2(\gamma^2-1)$ disparos (`shots_to_certify_branch`). A $z=3$: 27 disparos con $\gamma=2$ y 443 con $\gamma=7{,}089$ ($\beta=0{,}99$). Cuanto más relativista la partícula, más cerca del ecuador queda la lectura y más cuesta la rama.

## 40.7 Sin qang vs con qang

| aspecto | sin qang | con qang | diferencia |
|---|---|---|---|
| conjugar una partícula | «signo de la carga» | $(qg_X,qg_Y,qg_Z)\mapsto-(\cdot)$, $\theta\mapsto\pi-\theta$ | error $1{,}1\cdot10^{-13}$ m-qg; mapa antiunitario |
| ¿Una partícula puede ser su antipartícula con un qubit? | se discute | $\lvert\langle\psi\vert C\psi\rangle\rvert=0$ siempre | $6{,}8\cdot10^{-17}$ en 2000 estados |
| Zitterbewegung | frecuencia $2E$ | media $1000/\gamma^2$, amplitud $1000\beta^2$ m-qg | 640,0 y 360,0 m-qg a $\gamma=1{,}25$ |
| certificar la rama a 3σ | sin presupuesto | $N=9(\gamma^2-1)$ | 27 ($\gamma=2$) · 443 ($\beta=0{,}99$) disparos |
| par en singlete | «entrelazado» | locales 0, $qg_{ZZ}=-1000$ m-qg | no se ve con $qg_Z$ local |

## 40.8 Código

```python
from qang.qanti import (dirac_h, eigenstate, conjugate, bloch_mqg, self_conjugate_overlap, gamma_of,
                        zb_qgz_mqg, zb_analytic_mqg, lz_probability, pair_probability, lz_numeric,
                        pair_reading_mqg, singlet_readings, shots_to_certify_branch)

psi = eigenstate(0.75, 1.0, +1);  bloch_mqg(conjugate(psi))      # = -bloch_mqg(psi)
zb_analytic_mqg(0.75, 1.0, 0.7)                                  # 640.0 + 360.0*cos(2Et)
pair_reading_mqg(pair_probability(0.8, 1.0))                     # +732.2 m-qg
shots_to_certify_branch(2.0)                                     # 27.0
```

## 40.9 Verificación

`tools/verify_cap40.py` (salida real):

```text
[1] 500 autoestados +E al azar: C = σy·conj invierte (qg_X, qg_Y, qg_Z) con error 1.1e-13 m-qg y da energía −E (error 1.3e-15); lectura de qg_Z = +1000·m/E
[2] 2000 estados puros al azar: máx |⟨ψ|Cψ⟩| = 6.8e-17 (el mapa antípoda no tiene puntos fijos)
[3] p/m = 0.75 (γ = 1.2500, β = 0.6000): media 640.0 m-qg, amplitud 360.0 m-qg, máx |exacto − fórmula| = 4.5e-13 m-qg
[4] (ventana finita ±250; máx |analítico − numérico| = 2.0e-03) pares por modo (c=ħ=1): m²/eE=0.25: P=0.4559 (numérico 0.4541), lectura +88.1 m-qg; m²/eE=0.64: P=0.1339 (numérico 0.1359), lectura +732.2 m-qg; m²/eE=1.00: P=0.0432 (numérico 0.0429), lectura +913.6 m-qg
[5] singlete: lecturas locales 0.0 y 0.0 m-qg; correlación qg_ZZ = -1000.0 m-qg
[6] disparos para ver el signo a 3σ: γ=2 → 27; γ=7.089 (β=0.99) → 443
verify_cap40: OK
```

`tests/test_qanti.py` (5 pruebas): conjugación (lecturas y energía), ausencia de estado autoconjugado, fórmula de Zitterbewegung, Landau–Zener frente a Schwinger y numérico, singlete y disparos.

## 40.10 Apéndice H

| id | familia | fórmula | función | estatus |
|---|---|---|---|---|
| `charge_conj_antipode` | física | $\psi\mapsto\sigma_y\psi^*$ invierte las tres lecturas | `qang.qanti.conjugate`, `qang.qanti.bloch_mqg` | `exact` |
| `no_self_conjugate_qubit` | física | $\lvert\langle\psi\vert C\psi\rangle\rvert=0$ | `qang.qanti.self_conjugate_overlap` | `exact` |
| `zitterbewegung_qgz` | física | $qg_Z=1/\gamma^2+\beta^2\cos2Et$ | `qang.qanti.zb_analytic_mqg`, `qang.qanti.zb_qgz_mqg` | `exact` |
| `pair_creation_lz` | física | $P=e^{-\pi m^2/eE}$; $qg=1-2P$ | `qang.qanti.pair_probability`, `qang.qanti.pair_reading_mqg`, `qang.qanti.lz_numeric` | `exact` (numérico a $2\cdot10^{-3}$) |
| `singlet_local_blind` | física | $qg_Z=0$ local, $qg_{ZZ}=-1000$ | `qang.qanti.singlet_readings` | `exact` |
| `branch_shots` | ruido | $N=z^2(\gamma^2-1)$ | `qang.qanti.shots_to_certify_branch` | `exact` |

## 40.11 CHANGELOG y commit

```markdown
### Added
- `qang.qanti` (Cap. 40): matter and antimatter read with qang (charge conjugation as the antipodal map, Zitterbewegung, Landau–Zener pair creation, singlet blindness, branch shot budget); `tools/verify_cap40.py`; 5 tests.
```

```text
feat(book): add Cap. 40 (matter/antimatter) with qang.qanti
```
