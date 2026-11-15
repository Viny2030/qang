# Capítulo 16 — Doble carril, pérdida como borrado anunciado y fidelidad a larga distancia: tasa y fidelidad se separan tras el filtro de sector
# Chapter 16 — Dual Rail, Loss as Heralded Erasure and Long-Distance Fidelity: Rate and Fidelity Decouple after the Sector Filter

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 13 sobre la base de Cap. 1–15. Módulo: `qang.qdualrail` (JAX, `jax<0.9`; alineado con `qang.qpon`, `qang.sectors`, §158). Verificación: `tools/verify_cap16.py`.

---

## 16.1 El qubit de doble carril vive en el sector de peso de Hamming 1

**ES.** $|0_L\rangle=|10\rangle$, $|1_L\rangle=|01\rangle$ (modos $a,b$). La pérdida con transmisión $\eta=10^{-\alpha L/10}$ ($\alpha=0.2$ dB/km) es amortiguamiento de amplitud independiente por modo: el peso sale del sector $\{|10\rangle,|01\rangle\}$ hacia el vacío $|00\rangle$ y **nunca** se mezcla dentro del sector. Por tanto (`exact`):
$$\text{sin anuncio:}\quad qg_Z'=\eta\,qg_Z+(1-\eta)\ \ (\text{vacío leído como }0_L),\qquad \text{con filtro de sector:}\quad qg_Z'=qg_Z,\ \ \text{tasa}=\eta .$$
El sesgo sin filtro es $(1-\eta)(1-qg_Z)$: a $50$ km una entrada de $-600$ m-qg se lee $+840$ m-qg; a $200$ km, $+1000$ m-qg (toda la información destruida). Con filtro la lectura es exacta a cualquier distancia, y también $qg_X,qg_Y$ (la matriz densidad postseleccionada es $\rho_L$).

**EN.** Photon loss moves weight out of the Hamming-weight-1 sector without mixing inside it; sector postselection restores the full Bloch vector exactly at rate $\eta$, while unheralded readout is biased by $(1-\eta)(1-qg_Z)$ (`exact`).

## 16.2 Teleportación a larga distancia: la pérdida cuesta tasa, el ruido intra-sector cuesta fidelidad

Par de Werner $V$ (Cap. 15) con un fotón por brazo: tasa de anuncio $\eta(L/2)^2=\eta(L)$. Tras el filtro, $\bar F=(1+V)/2$ **no depende de $L$**; el presupuesto de disparos se multiplica por $1/\eta(L)$: $\times10$ (50 km), $\times100$ (100 km), $\times10^4$ (200 km). El criterio de utilidad sigue siendo $qg_{ZZ}>333$ m-qg (Cap. 15) medido **dentro del sector**. Regla de diseño: optimizar $\eta$ y $V$ por separado; no existe compensación entre ambos tras el filtro (modelo de borrado puro; cuentas oscuras y pérdida dependiente del modo rompen la separación: `mixed` en hardware, §158).

## 16.3 Contraste operativo — Sin `qang` vs. con `qang`

| Métrica / Metric | Estándar (sin filtro) | `qang` (`qang.qdualrail`, filtro de sector) | Ganancia / Estatus |
| :--- | :--- | :--- | :--- |
| Lectura a 50 km, entrada $-600$ m-qg | $+840$ m-qg (sesgo $+1440$) | $-600$ m-qg | sesgo eliminado (`exact`) |
| Coherencias $qg_X,qg_Y$ tras pérdida | atenuadas $\times\eta$ | intactas | (`exact`) |
| $\bar F$ de teleportación vs $L$ | decae con $\eta$ | $(1+V)/2$ constante; coste $\times1/\eta$ en disparos | (`exact`, borrado) |
| Diagnóstico | mezclado | pérdida ↔ tasa; ruido ↔ $qg_{ZZ}$ | separables (`exact`); hw `mixed` |

## 16.4 Bloque JAX (`qang/qdualrail.py`)

```python
def apply_loss(rho4, eta):                       # Kraus de amortiguamiento por modo, producto a⊗b
    K0 = jnp.array([[1, 0], [0, jnp.sqrt(eta)]]); K1 = jnp.array([[0, jnp.sqrt(1 - eta)], [0, 0]])
    return sum(jnp.kron(Ka, Kb) @ rho4 @ jnp.conj(jnp.kron(Ka, Kb)).T for Ka in (K0, K1) for Kb in (K0, K1))

@jax.jit
def readout_unfiltered(rho4):                    # qg_Z' = 1 − 2 P(click en b)
    p = jnp.real(jnp.diag(rho4)); return 1.0 - 2.0 * (p[1] + p[3])

@jax.jit
def readout_sector_filtered(rho4):               # (qg_Z, tasa) postseleccionando peso 1
    p = jnp.real(jnp.diag(rho4)); acc = p[1] + p[2]; return (p[2] - p[1]) / acc, acc
```

## 16.5 Verificación — `tools/verify_cap16.py`

```
[1] L=   50 km η=0.1000: entrada −600 m-qg → sin filtro +840 m-qg (sesgo +1440) | filtrado −600 m-qg con tasa 0.1000
[1] L=  200 km η=0.0001: entrada −600 m-qg → sin filtro +1000 m-qg (sesgo +1600) | filtrado −600 m-qg con tasa 0.0001
[2] pérdida pura: ρ postseleccionada = ρ_L exacta (qg_X, qg_Y, qg_Z intactos) → borrado anunciado
[3] L= 100 km: F̄ filtrada = 0.950 (V=0.9, sin dependencia de L); salida +540 m-qg; disparos × 100 por anuncio
verify_cap16: OK
```
Asserts: sesgo $=\eta qg_Z+(1-\eta)$ a $10^{-14}$ (12 casos); $qg_Z$ filtrado a $10^{-13}$ y tasa $=\eta$ a $10^{-14}$; $qg_X$ postseleccionado intacto; multiplicador $1/\eta(100\text{ km})=100$.

### `tests/test_qdualrail.py`
```python
import pytest; jax = pytest.importorskip("jax")
from qang.qdualrail import dual_rail_state, apply_loss, readout_unfiltered, readout_sector_filtered

def test_bias_and_filter():
    rho = apply_loss(dual_rail_state(-0.6), 0.1)
    assert abs(float(readout_unfiltered(rho)) - (0.1 * -0.6 + 0.9)) < 1e-14
    q, acc = readout_sector_filtered(rho); assert abs(float(q) + 0.6) < 1e-13 and abs(float(acc) - 0.1) < 1e-14
```

## 16.6 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `dual_rail_loss_bias` | óptica | $qg_Z'=\eta qg_Z+(1-\eta)$ | `readout_unfiltered` | `exact` |
| `dual_rail_sector_filter` | compilación | $qg_Z$ exacto, tasa $\eta$ | `readout_sector_filtered` | `exact`; hw §158 `mixed` |
| `teleport_distance_budget` | estadística | disparos $\times1/\eta(L)$, $\bar F$ constante | `shots_multiplier_two_photons` | `exact` (borrado) |

`CHANGELOG.md [Unreleased]`: `qang.qdualrail` (`transmission`, `dual_rail_state`, `apply_loss`, `readout_unfiltered`, `readout_sector_filtered`, `shots_multiplier_two_photons`); `tools/verify_cap16.py`; `tests/test_qdualrail.py`.
Commit: `docs(cap16): doble carril y filtro de sector reescritos sobre Cap. 1–15 + qang.qdualrail + verify_cap16`.
