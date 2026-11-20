# Capítulo 19 — Demarcación: criptografía cuántica (QKD/QRNG) frente a post-cuántica (PQC). Dónde hay un qubit que leer en m-qg y dónde no
# Chapter 19 — Demarcation: Quantum Cryptography (QKD/QRNG) versus Post-Quantum Cryptography (PQC). Where There Is a Qubit to Read in m-qg and Where There Is None

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 16 sobre la base de Cap. 1–18. Módulo: `qang.qdemarc` (JAX, `jax<0.9`). Verificación: `tools/verify_cap19.py`. Detalle de BB84 y QRNG: Cap. 20–21.

---

## 19.1 Criterio de demarcación

**ES.** `qang` mide ángulos y probabilidades de **qubits físicos** (Cap. 1). Aplica donde la seguridad descansa en una lectura proyectiva: QKD (el QBER es un $qg_Z$) y QRNG (la entropía mínima es una función de $|qg_Z|$). **No aplica** a PQC (ML-KEM, ML-DSA, SLH-DSA, HQC): aritmética clásica en retículos/hashes/códigos, sin qubit, con parámetros de seguridad $\delta\le2^{-138.8}$ cuyo "presupuesto de disparos" sería $N=(3/2\delta)^2=2^{278.8}$: incontable (`against`, THEORY_CORE §2.6). Un número que no puede medirse con disparos no es una lectura `m-qg`.

**EN.** `qang` applies where security rests on a projective readout of a physical qubit (QKD, QRNG) and not to PQC, whose security parameters are uncountable in shots (`against`).

## 19.2 QKD: el QBER como lectura m-qg y el umbral de 780 m-qg

$e\leftrightarrow qg_Z=1-2e$ en el eje de la base reconciliada. Tasa asintótica BB84 (un solo paso, ataques colectivos): $r=1-2h(e)>0\iff e<0.1100\iff qg_Z>0.780$ ($780$ m-qg). Tabla: QBER $0.02\to+960$ m-qg, $r=0.717$; $0.08\to+840$, $r=0.196$; $0.15\to+700$, $r=-0.220$ (abortar). Decisión de protocolo = umbral en m-qg, con intervalos Wilson/Bayes-Haar (Cap. 4, 8) para la cobertura cerca del polo (`exact`).

## 19.3 QRNG: entropía mínima desde |qgZ|

$H_{\min}=-\log_2\frac{1+|qg_Z|}{2}$: $0$ m-qg → $1.000$ bit; $20$ m-qg → $0.971$; $100$ m-qg → $0.863$; $500$ m-qg → $0.415$. Certificar un sesgo de $20$ m-qg ($\delta=0.01$) a $3\sigma$ cuesta $N=22\,500$ disparos (`exact`).

## 19.4 Contraste operativo

| Dominio | Objeto medido | Lectura `qang` | Estatus |
| :--- | :--- | :--- | :--- |
| QKD (BB84) | QBER $e$ | $qg_Z=1-2e$; clave $\iff qg_Z>780$ m-qg | `exact` |
| QRNG | sesgo de un qubit | $H_{\min}(|qg_Z|)$; $N=(z/2\delta)^2$ | `exact` |
| PQC (ML-KEM/ML-DSA/SLH-DSA/HQC) | $\delta\le2^{-138.8}$ | ninguna: $2^{278.8}$ disparos, sin qubit | `against` |
| Baterías/redes cuánticas (Cap. 22) | $qg_Z$ de celdas | aplica | ver Cap. 22 |

## 19.5 Bloque JAX (`qang/qdemarc.py`)

```python
@jax.jit
def bb84_rate_from_qz(qz):            # r = 1 − 2 h((1 − qg_Z)/2)
    return 1.0 - 2.0 * h2((1.0 - qz) / 2.0)

def bb84_threshold_qz(tol=1e-12):     # bisección: qg_Z* = 0.7799
    ...

@jax.jit
def qrng_min_entropy_from_qz(qz):     # −log2((1 + |qg_Z|)/2)
    return -jnp.log2((1.0 + jnp.abs(qz)) / 2.0)

def shots_to_resolve_bias(delta, z=3.0):   # (z/(2δ))²
    return (z / (2.0 * delta)) ** 2
```

## 19.6 Verificación — `tools/verify_cap19.py`

```
[1] BB84: r = 1 − 2h(e) = 0 en e* = 0.1100  ⇔  qg_Z* = 0.7799 (780 m-qg)
    QBER=0.08 → +840 m-qg → r = +0.1956 (clave) | QBER=0.15 → +700 m-qg → r = -0.2197 (abortar)
[2] |qg_Z| = 20 m-qg → H_min = 0.9714 bit; certificar δ=0.01 a 3σ: N = 22500 disparos
[3] PQC: δ = 2^-138.8 → N = 2^278.8 'disparos' — incontable; sin qubit: qang no aplica (against)
verify_cap19: OK
```
Asserts: $e^\star=0.1100\pm2\cdot10^{-4}$ con $1-2h(e^\star)=0$ a $10^{-9}$; $H_{\min}(0)=1$; $\log_2N_{\text{PQC}}>270$.

### `tests/test_qdemarc.py`
```python
import pytest; jax = pytest.importorskip("jax")
from qang.qdemarc import bb84_threshold_qz, qrng_min_entropy_from_qz, shots_to_resolve_bias

def test_bb84_threshold(): assert abs(bb84_threshold_qz() - 0.7799) < 1e-3
def test_qrng_unbiased(): assert abs(float(qrng_min_entropy_from_qz(0.0)) - 1.0) < 1e-15
def test_pqc_uncountable(): import math; assert math.log2(shots_to_resolve_bias(2.0 ** -138.8)) > 270
```

## 19.7 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `qber_mqg_threshold` | criptografía | $qg_Z>0.780$ ⇔ $r>0$ | `bb84_rate_from_qz`, `bb84_threshold_qz` | `exact` |
| `qrng_hmin_qz` | criptografía | $-\log_2\frac{1+|qg_Z|}{2}$ | `qrng_min_entropy_from_qz` | `exact` |
| `pqc_out_of_scope` | demarcación | $N=2^{278.8}$, sin qubit | `shots_to_resolve_bias` | `against` |

`CHANGELOG.md [Unreleased]`: `qang.qdemarc` (`h2`, `qber_to_qz`, `qz_to_qber`, `bb84_rate_from_qz`, `bb84_threshold_qz`, `qrng_min_entropy_from_qz`, `shots_to_resolve_bias`); `tools/verify_cap19.py`; `tests/test_qdemarc.py`.
Commit: `docs(cap19): demarcación QKD/QRNG vs PQC reescrita sobre Cap. 1–18 + qang.qdemarc + verify_cap19`.
