# Capítulo 25 — Calibración por eco y el sesgo de ruido: el arrastre de lectura como mapa afín en m-qg, cuándo **no** invertir (§108) y la lectura sin arrastre
# Chapter 25 — Echo Calibration and Noise Bias: Readout Drag as an Affine Map in m-qg, When **Not** to Invert (§108) and Drag-Free Readout

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 22 (F7) sobre la base de Cap. 1–24. Módulo: `qang.qecho` (NumPy; compatible con el core). Verificación: `tools/verify_cap25.py`. Prohibición THEORY_CORE §2.2 incorporada como criterio cuantitativo.

---

## 25.1 El error de asignación es un mapa afín en m-qg

**ES.** Con $e_{01}=P(\text{leer }1|0)$, $e_{10}=P(\text{leer }0|1)$:
$$m'=(1-e_{01}-e_{10})\,m+1000\,(e_{10}-e_{01}),\qquad m^\star=\frac{1000\,(e_{10}-e_{01})}{e_{01}+e_{10}}\ \ (\text{punto fijo, sesgo cero}).$$
Instancia $e_{01}=0.02,e_{10}=0.06$: $-1000\to-880$, $0\to+40$, $+1000\to+960$ m-qg; $m^\star=+500$ m-qg. El "arrastre" (*pull effect*) es la contracción $0.92$ más el desplazamiento $+40$ m-qg; el coste de una compuerta $X$ medido en §168 ($1.64$ m-qg lineales) es este mismo mapa con $e\sim10^{-3}$ (`exact`, `tested on hardware` §168).

**EN.** Assignment error is an affine contraction of the m-qg axis with a drag-free fixed point $m^\star$ (`exact`).

## 25.2 Inversión por eco: criterio cuantitativo de §108

La calibración por eco (circuitos $I$ y $X$, o eco $XX$) estima $(e_{01},e_{10})$ con $N_{\text{cal}}$ disparos y permite invertir el mapa; la inversión amplifica la varianza por $(1-e_{01}-e_{10})^{-2}$ y añade la varianza de calibración. Decisión:
$$\text{invertir}\iff \underbrace{b^2+\sigma_m^2}_{\text{MSE}_{\text{raw}}}>\underbrace{\frac{\sigma_m^2}{(1-e_{01}-e_{10})^2}+\mathrm{Var}_{\text{cal}}}_{\text{MSE}_{\text{inv}}},\qquad b=m'-m,\ \ \sigma_m=1000\sqrt{(1-qg_Z'^2)/N}.$$

| $m$ | $N$ | $N_{\text{cal}}$ | sesgo | $\sigma_m$ | MSE$_{\text{raw}}$ | MSE$_{\text{inv}}$ | decisión |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $+600$ | 200 | 2000 | $-8.0$ | 57.0 | 3312 | 3872 | **NO invertir** (§108) |
| $+600$ | 20000 | 2000 | $-8.0$ | 5.7 | 96 | 73 | invertir |
| $+600$ | 20000 | 200 | $-8.0$ | 5.7 | 96 | 388 | NO invertir (calibración pobre) |
| $+500=m^\star$ | 20000 | 2000 | $0$ | 6.1 | 38 | 79 | NO invertir (sin arrastre) |
| $-900$ | 1000 | 1000 | $+112$ | 19.5 | 12923 | 689 | invertir |

Monte Carlo ($4000$ réplicas, fila 1): MSE$_{\text{raw}}=3400$ (teoría 3312), MSE$_{\text{inv}}=4009$ (teoría 3872) (`exact` asintótico). La prohibición de §108 es exactamente la fila 1: sesgo $\lesssim\sigma_m$ ⇒ la inversión empeora.

## 25.3 Lectura sin arrastre (F7): el filtro de sector baja el error de $O(e)$ a $O(e_{01}e_{10})$

En el sector de peso 1, **un** flip de lectura saca el string del sector (lo elimina el filtro); llegar a **otro** string de peso 1 exige dos flips, probabilidad $e_{10}e_{01}$. Sesgo máximo filtrado: $0.0013$ ($\approx e_{01}e_{10}=0.0012$) frente a $0.060$ de primer orden sin filtro ($\times46$); con $e=0.01$: $\times196$. Combinado con Cap. 24: el filtro elimina $T_1$ exactamente y la lectura a segundo orden **sin calibrar ni invertir** — la "lectura sin arrastre" (`exact` modelo de flips independientes; hardware `unverified`).

## 25.4 Contraste operativo

| Métrica / Metric | Estándar (invertir siempre) | `qang` (`qang.qecho`) | Estatus |
| :--- | :--- | :--- | :--- |
| $m=600$, $N=200$ | MSE 3872 | sin invertir: 3312 | $-14\%$ (`exact`, §108) |
| $m=-900$, $N=1000$ | — | invertir: 12923 → 689 | $-95\%$ (`exact`) |
| Punto de operación | arbitrario | $m^\star=+500$ m-qg: sesgo 0 | (`exact`) |
| Error de lectura en sector | $O(e)$: 0.060 | $O(e_{01}e_{10})$: 0.0013 | $\times46$ (`exact` modelo) |
| Hardware | — | §168: $X=1.64$ m-qg | `tested on hardware` |

## 25.5 Bloque (`qang/qecho.py`)

```python
def readout_affine(m, e01, e10):  return (1 - e01 - e10) * m + 1000 * (e10 - e01)
def drag_free_point(e01, e10):    return 1000 * (e10 - e01) / (e01 + e10)
def invert_readout(m_obs, e01, e10): return (m_obs - 1000 * (e10 - e01)) / (1 - e01 - e10)
def mse_raw_vs_inverted(m, e01, e10, N, N_cal)   # (MSE_raw, MSE_inv, sesgo, σ_m) — criterio §108
def sector_readout_error_order(p_sector, e01, e10) # (p filtrada, sesgo máx., cota e01·e10)
```

## 25.6 Verificación — `tools/verify_cap25.py`

```
[1] m=-1000 → m'=-880.0 (sesgo +120) | m=0 → +40.0 | punto sin arrastre m* = +500.0 m-qg
[2] +600 N=200 N_cal=2000: MSE_raw=3312 MSE_inv=3872 → NO invertir | +600 N=20000: 96 vs 73 → INVERTIR
    MC: MSE_raw=3400 (teoría 3312), MSE_inv=4009 (teoría 3872)
[3] e01=0.02, e10=0.06: sesgo filtrado=0.00130 (≈ e01·e10) vs 0.0600 sin filtro → ×46
verify_cap25: OK
```
Asserts: punto fijo a $10^{-9}$; decisiones de las filas 1, 2 y 4; MC dentro del $15\%$ de la teoría; sesgo filtrado $<3\,e_{01}e_{10}$.

### `tests/test_qecho.py`
```python
from qang.qecho import readout_affine, drag_free_point, mse_raw_vs_inverted, sector_readout_error_order
import numpy as np
def test_fixed_point(): m = drag_free_point(0.02, 0.06); assert abs(readout_affine(m, 0.02, 0.06) - m) < 1e-9
def test_108_rule(): r, i, _, _ = mse_raw_vs_inverted(600, 0.02, 0.06, 200, 2000); assert r < i
def test_second_order(): _, b, c = sector_readout_error_order(np.array([.5, .3, .15, .05]), 0.02, 0.06); assert b < 3 * c
```

## 25.7 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `readout_drag_affine` | lectura | $m'=(1-e_{01}-e_{10})m+1000(e_{10}-e_{01})$ | `readout_affine` | `exact`; §168 `tested on hardware` |
| `drag_free_point` | lectura | $m^\star=1000(e_{10}-e_{01})/(e_{01}+e_{10})$ | `drag_free_point` | `exact` |
| `echo_inversion_rule_108` | mitigación | invertir ⇔ MSE$_{\text{raw}}>$ MSE$_{\text{inv}}$ | `mse_raw_vs_inverted` | `exact` (sustituye la prohibición cualitativa) |
| `sector_readout_second_order` | mitigación | sesgo $O(e_{01}e_{10})$ | `sector_readout_error_order` | `exact` modelo |

`CHANGELOG.md [Unreleased]`: `qang.qecho`; `tools/verify_cap25.py`; `tests/test_qecho.py`.
Commit: `docs(cap25): arrastre de lectura, criterio §108 y lectura sin arrastre sobre Cap. 1–24 + qang.qecho + verify_cap25`.
