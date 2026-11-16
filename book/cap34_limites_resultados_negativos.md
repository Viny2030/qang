# Capítulo 34 — Límites estructurales, equivalencias y resultados negativos: qué no puede ver la lectura qg (y dónde el filtro no llega)
# Chapter 34 — Structural Limits, Equivalences and Negative Results: What the qg Reading Cannot See (and Where the Filter Does Not Reach)

**Actualizado:** 16 de noviembre de 2026 · **Updated:** November 16, 2026

> Reescritura del capítulo de límites sobre la base de Cap. 1–33. Módulo: `qang.qlimits` (NumPy exacto). Verificación: `tools/verify_cap34.py`. **Resumen: casi todos los límites tienen una raíz común: qg lee una probabilidad de la base Z ($p_0$), y lo que no está en esa probabilidad no lo ve. Lo que sigue lo demuestra con código donde se puede, y marca como `unverified` lo que sólo se cita de las notas.**
> **EN abstract.** Almost every limit has the same root: $qg$ reads a Z-basis probability $p_0$, so anything outside it is invisible. This chapter proves the checkable ones with exact code (graph states, $qg_S$ under $T_1$, the $\arccos$ range trap, the weight-preservation analyzer, the weight-1 quadratic form, in-sector errors) and flags the rest as cited, not re-run.

---

## 34.1 Reglas de lectura

1. Cada límite entra con su función en `qang.qlimits` y su línea en `tools/verify_cap34.py`. Lo que sólo viene de las notas de investigación (§) se marca `unverified` en este capítulo: **aquí no se re-ejecutó**.
2. `qga` es la escala angular (1 vuelta = 1000 qga); `m-qg` es la lectura $1000\,qg_Z$. Ninguno de los límites de abajo cambia al pasar de una a otra: son límites de **qué** se mide, no de cómo se escribe.
3. Un resultado negativo no se borra: se registra en el Apéndice H con su estatus.

## 34.2 Ceguera de la base Z: el estado de grafo

Un estado de grafo es $H^{\otimes n}$ seguido de una CZ por arista. La CZ es diagonal: sólo cambia fases. Con el anillo de 6 qubits (`ring_graph_state`):

- la distribución en Z coincide con la de $|+\rangle^{\otimes 6}$ con diferencia **exactamente 0**;
- $qg_Z=0$ m-qg en cada qubit y $qg_S=1.00000$, igual que el estado producto;
- $\langle X_iZ_{i-1}Z_{i+1}\rangle=+1.0$ en el grafo y $+0.0$ en el producto (`stabilizer_expectation`): lo separan, pero **en base X**.

Sin `qang` ocurre lo mismo: toda estadística de la base Z, cruda o tomográfica, es ciega a este entrelazamiento. `qang` hereda el límite, no lo crea.

## 34.3 $qg_S$ no es monótona bajo $T_1$ (pero $qg_Z$ medio sí)

El amortiguamiento de amplitud actúa sobre las poblaciones de Z como una cadena de Markov $1\to0$ con probabilidad $\gamma$ por qubit (`damp_distribution`); las coherencias no cambian la distribución en Z. Con 24 estados aleatorios de 4 qubits y $\gamma\in[0,0.5]$ (`t1_noise_scan`):

- $qg_S$ **sube y baja en 24/24**. Ejemplo (estado 0): $0.782\to0.847$ ($\gamma=0.18$) $\to0.736$ ($\gamma=0.50$);
- el $qg_Z$ medio crece **monótono en 24/24**, y es exacto: cada qubit pasa de $1-2P_1$ a $1-2(1-\gamma)P_1$. En el ejemplo va de $+251$ a $+543$ m-qg;
- por eso el par ($qg_S$, $qg_Z$ medio) separa «poco ruido» de «mucho $T_1$», donde $qg_S$ sola no puede.

**Alcance honesto.** Este resultado es para amortiguamiento aplicado al estado de salida. En circuitos con $T_1$ entre compuertas (volumen cuántico, notas §10.4 D) el $qg_Z$ medio fue monótono en 21–23 de 24 circuitos: ese número se cita, no se re-ejecuta aquí (`unverified`).

## 34.4 La trampa del rango de arccos

$\theta=\arccos(qg_Z)\in[0,\pi]$. Si la cota de un mínimo está en $\theta^*<0$, la actualización en $qg$ no puede llegar. Con $E(\theta)=-\cos(\theta-\theta^*)$ y $\theta^*=-0.5$ (`best_cost_clipped`):

| | valor mínimo | dónde |
| :--- | :--- | :--- |
| sin recorte (descenso en $\theta$) | $-1.0000$ | $\theta=-0.5$ |
| con $\theta\in[0,\pi]$ (vía $qg_Z$) | $-0.8776$ | $\theta=0$ |

Pérdida: $0.1224$ de $E$. Las afirmaciones de que «converge en qg donde $\theta$ no converge» venían de este recorte. No se propone optimización directa en $qg_Z$ sin regularización.

## 34.5 Conservación de peso: dónde el filtro se aplica y dónde no

El filtro de sector exige que cada compuerta conserve el peso de Hamming. `preserves_weight` lo decide sobre la unitaria, y `weight_breakers` cuenta las compuertas culpables:

| circuito (4 qubits) | ¿conserva el peso? | compuertas que lo rompen |
| :--- | :--- | :--- |
| QFT | no | 4 (las Hadamard) |
| GHZ | no | 4 (H + 3 CNOT) |
| escalera RBS (unitaria completa) | sí | 0 |

Las CPhase y SWAP de la QFT sí conservan el peso; las Hadamard mezclan todos los pesos. La estimación de fase (QPE, y lo que se apoya en ella) queda fuera del filtro.

## 34.6 Error dentro del sector

El filtro descarta disparos fuera del sector; un error que **conserva el peso** pasa entero. Con RBS ideal de $\pi/2$ frente a RBS$(\pi/2+0.3)$ (`in_sector_error`):

- retención del filtro: **1.0** en los dos casos;
- $qg_Z$ del qubit 0: $+1000.0\to+825.3$ m-qg: desvío de $-174.7$ m-qg, **invisible al filtro**.

Es el mismo mecanismo que en LiH (notas §50: el error está en los grupos X/Y, fuera del alcance del filtro en Z). Esos números se citan: **LiH, 60 compuertas: 76.6 mHa crudo frente a 85.0 mHa con filtro** (`unverified` en este capítulo).

## 34.7 La red de peso 1 es una forma cuadrática clásica

En el sector de una excitación, la red RBS actúa como una matriz ortogonal $O$ sobre los datos cargados $v$, y

$$qg_Z^{(i)}=1-2\,(Ov)_i^2 .$$

`weight1_qgz` la evalúa en $O(n^2)$; `simulate_qgz_weight1` simula el circuito completo ($2^n$ amplitudes). Con $n=5$ y 20 redes al azar el desvío máximo es $4.4\cdot10^{-16}$. Lo que el filtro protege exactamente del $T_1$ es, por ese mismo motivo, simulable. No se afirma ventaja cuántica.

## 34.8 Equivalencias y fallos citados (sin re-ejecutar)

Estos puntos vienen de las notas (§) y de los capítulos donde se midieron; **no** tienen verificador en este capítulo:

- **Gradiente natural en qg = descenso en $\theta$** (§159, 20 semillas, Ising de 4 qubits): 103 pasos (mediana) en $\theta$; el descenso en qg no llegó nunca; gradiente natural completo 22.5 pasos; versión leída de bloques $qg_{YY}$ 79.5 pasos ($1.3\times$, no los $1.5\times$ predichos: G2 falló). La ganancia es del gradiente natural, no de `qang`.
- **Predicciones pre-registradas §157–§164:** se cumplieron 26 de 45 (19 falladas, publicadas).
- **Hardware real (IQM Garnet), §165–§170:** 12 de 13 predicciones; la fallada es M2 de §168.

## 34.9 Inventario de negativos del libro de cuentas

`negative_inventory` lee las tablas del Apéndice H de los capítulos 7–33 (`qang.qledger`): 97 submedidas; `against` 4, `no gain` 3, `mixed` 1, `unverified` 1, `not adopted` 1; **0 `measured gain`**. Las filas: `plateau_unit_rescale` (Cap. 9), `page_curve_qgS` (18), `pqc_out_of_scope` (19), `local_reading_blind_to_c` (33) como `against`; `holonomic_z_area` (10), `ghz_dephasing` (11), `control_quantization_unit` (30) como `no gain`; `one_basis_bias` (15) `mixed`; `collective_oscillations` (13) `unverified`; `string_theory_claims` (33) `not adopted`.

## 34.10 Contraste operativo: sin `qang` frente a con `qang`

| Límite | Sin `qang` | Con `qang` (`qang.qlimits`) | Estatus |
| :--- | :--- | :--- | :--- |
| Estado de grafo (anillo 6) | $P_Z$ idéntica a $\lvert+\rangle^{\otimes6}$ | $qg_Z=0$ m-qg, $qg_S=1.00000$; estabilizador en X: $+1$ frente a $0$ | `exact` |
| $T_1$ sobre estados de 4 qubits | $H(p)$ no monótona | $qg_S$ no monótona (24/24); $qg_Z$ medio monótono (24/24) | `exact` |
| Rango de arccos | mín. $-1.0000$ | $-0.8776$ con $\theta\in[0,\pi]$ (pérdida $0.1224$) | `exact` |
| QFT / GHZ (4 qubits) | — | no conservan el peso (4 compuertas cada una): filtro no aplicable | `exact` |
| Error dentro del sector | — | retención 1.0; desvío $-174.7$ m-qg sin detectar | `exact` |
| Red de peso 1 | simulación $2^n$ | forma cuadrática $O(n^2)$, error $4.4\cdot10^{-16}$ | `exact` |
| Ventaja medida del libro | — | 0 `measured gain` en 97 submedidas | `exact` (auditoría) |

## 34.11 Bloque (`qang/qlimits.py`)

```python
def ring_graph_state(n)                      # H^n + CZ en anillo
def stabilizer_expectation(psi, i, n)        # <X_i Z_{i-1} Z_{i+1}>
def damp_distribution(p, n, gamma)           # T1 sobre las poblaciones de Z
def t1_noise_scan(p, n, gammas)              # (qg_S, qg_Z medio) frente a gamma
def best_cost_clipped(theta_star)            # trampa del rango de arccos
def preserves_weight(U, n); weight_breakers(n, gates)
def weight1_qgz(O, v); simulate_qgz_weight1(n, thetas, v)
def in_sector_error(eps)                     # retención 1.0, desvío en m-qg
def negative_inventory(rows)                 # lee qang.qledger
```

## 34.12 Verificación — `tools/verify_cap34.py`

```
[1] anillo n=6: |P_Z(grafo) − P_Z(producto)| = 0.0; qg_Z medio 0 m-qg, qg_S 1.00000; ⟨X_i Z_(i−1) Z_(i+1)⟩ = +1.0 (grafo) frente a +0.0 (producto)
[2] 24 estados de 4 qubits, γ∈[0,0.5]: qg_S sube y baja en 24/24; qg_Z medio monótono creciente en 24/24 (exacto: 1−2(1−γ)P1)
    ejemplo (estado 0): qg_S 0.782 → 0.847 (γ=0.18) → 0.736 (γ=0.50); qg_Z medio +251 → +543 m-qg
[3] E(θ)=−cos(θ−θ*), θ*=−0.5: mínimo exacto -1.0000 (θ<0); con θ=arccos(qg_Z)∈[0,π] el mejor valor es -0.8776 en θ=0 → pérdida 0.1224
[4] QFT 4q: conserva=False, compuertas que lo rompen=4; GHZ 4q: conserva=False, compuertas que lo rompen=4; RBS escalera 4q: conserva=True, compuertas que lo rompen=0
[5] cargador unario + red RBS, n=5, 20 redes al azar: |forma cuadrática − simulación completa| ≤ 4.4e-16 (coste O(n²) frente a 2^n)
[6] RBS(π/2+0.3): retención del filtro 1.0; qg_Z del qubit 0: +1000.0 → +825.3 m-qg (desvío -174.7 m-qg, invisible al filtro)
[7] libro de cuentas Cap. 7–33: 97 submedidas; against: 4, no gain: 3, mixed: 1, unverified: 1, not adopted: 1; measured gain: 0
verify_cap34: OK
```

### `tests/test_qlimits.py`
```python
import numpy as np
from qang.qlimits import (ring_graph_state, plus_state, z_distribution, stabilizer_expectation, t1_noise_scan, is_monotone,
                          best_cost_clipped, qft_gates, circuit_unitary, preserves_weight, weight_breakers, in_sector_error)
def test_graph_state_blind_in_z():
    g, p = ring_graph_state(6), plus_state(6)
    assert np.max(np.abs(z_distribution(g) - z_distribution(p))) == 0.0
    assert np.allclose([stabilizer_expectation(g, i, 6) for i in range(6)], 1.0)
def test_t1_mean_qgz_monotone_and_qgs_not():
    rng = np.random.default_rng(1); psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    s, z = t1_noise_scan(z_distribution(psi / np.linalg.norm(psi)), 4, np.linspace(0, .5, 51))
    assert is_monotone(z, True) and not is_monotone(s, True) and not is_monotone(s, False)
def test_arccos_range_trap(): assert abs(best_cost_clipped(-0.5)[0] + np.cos(0.5)) < 1e-9
def test_weight_analyzer():
    assert not preserves_weight(circuit_unitary(4, qft_gates(4)), 4) and len(weight_breakers(4, qft_gates(4))) == 4
def test_in_sector_error_passes_filter():
    r = in_sector_error(0.3); assert abs(r["error_retained"] - 1) < 1e-12 and abs(r["shift_mqg"]) > 100
```

## 34.13 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `graph_state_z_blind` | límites | $P_Z(\text{grafo})=P_Z(\lvert+\rangle^{\otimes n})$ | `ring_graph_state`, `stabilizer_expectation` | `exact` |
| `qgs_t1_nonmonotone` | límites | $qg_S$ sube y baja bajo $T_1$ (24/24) | `t1_noise_scan`, `damp_distribution` | `exact` (numérico) |
| `mean_qgz_t1_monotone` | límites | $1-2(1-\gamma)P_1$ crece con $\gamma$ | `mean_qgz`, `t1_noise_scan` | `exact` |
| `arccos_range_trap` | límites | mín. alcanzable $-\cos(\theta^*)$ si $\theta^*<0$ | `best_cost_clipped` | `exact` |
| `weight_preservation_analyzer` | filtro | $U$ diagonal por bloques de peso | `preserves_weight`, `weight_breakers` | `exact` |
| `in_sector_error_unseen` | filtro | error que conserva el peso: retención 1 | `in_sector_error` | `exact` |
| `weight1_quadratic_form` | redes | $qg_Z^{(i)}=1-2(Ov)_i^2$ | `weight1_qgz`, `simulate_qgz_weight1` | `exact` |
| `negative_inventory_ledger` | auditoría | 0 `measured gain` en Cap. 7–33 | `negative_inventory` | `exact` |
| `lih_filter_scope` | filtro | LiH 60 compuertas: 76.6 → 85.0 mHa | — | `unverified` (notas §50) |
| `natural_gradient_equivalence` | optimización | qg = descenso en $\theta$; 103 pasos frente a 22.5 | — | `unverified` (notas §159) |
| `prereg_failures_157_164` | metodología | 26 de 45 predicciones cumplidas | — | `unverified` (notas §157–§164) |

`CHANGELOG.md [Unreleased]`: `qang.qlimits`; `tools/verify_cap34.py`; `tests/test_qlimits.py`.
Commit: `docs(cap34): límites estructurales y resultados negativos + qang.qlimits + verify_cap34`.
