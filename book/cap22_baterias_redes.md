# Capítulo 22 — Baterías cuánticas y redes: ergotropía como $(r-qg_Z)/2$ y cadenas de pares con visibilidad $V^k$
# Chapter 22 — Quantum Batteries and Networks: Ergotropy as $(r-qg_Z)/2$ and Repeater Chains with Visibility $V^k$

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 19 sobre la base de Cap. 1–21 (§56, §57). Módulo: `qang.qbattery` (JAX, `jax<0.9`; usa `qang.qtele`). Verificación: `tools/verify_cap22.py`.

---

## 22.1 Energía y ergotropía de una celda en qg

**ES.** Con $H=\omega|1\rangle\langle1|$: $E/\omega=(1-qg_Z)/2$ y, con $r$ el radio local de Bloch (Cap. 5),
$$\frac{W}{\omega}=\frac{r-qg_Z}{2},\qquad \text{fracción pasiva}=\frac{1-r}{1-qg_Z}.$$
Carga completa $qg_Z=-1$ ($-1000$ m-qg): $W=\omega$. **Hallazgo**: la lectura $Z$ sola no determina la carga extraíble: $|+\rangle$ ($0$ m-qg, $r=1$) tiene $W=\omega/2$; $I/2$ ($0$ m-qg, $r=0$) tiene $W=0$. La ergotropía exige $r$ (3 bases o cuadrados insesgados, THEORY_CORE §1), igual que $qg_\tau$ (`exact`). Potencia de carga colectiva vs paralela: `unverified` aquí (no se afirma ventaja).

**EN.** Ergotropy is $(r-qg_Z)/2$ in units of $\omega$; the $Z$ readout alone cannot distinguish a coherent half-charged cell from a passive one (`exact`).

## 22.2 Redes: el swapping multiplica visibilidades

Cadena de $k$ enlaces de Werner $V$ con medidas de Bell ideales: el canal compuesto es depolarizante con $V_{\text{end}}=V^k$ (verificado numéricamente $0.9\times0.8=0.720$: $qg_{ZZ}=720$ m-qg). Utilidad (Cap. 15): $V^k>1/3$ ⇒ $k_{\max}=\lfloor\ln(1/3)/\ln V\rfloor$: $V=0.99\to109$ enlaces ($334$ m-qg), $0.95\to21$, $0.9\to10$ ($349$ m-qg; el 11.º cae a $314$), $0.8\to4$. Planificación de red: medir $qg_{ZZ}$ por enlace (1 base), multiplicar, comparar con $333$ m-qg (`exact` para Werner; estados no Werner: cota superior, Cap. 15 §4). Pérdida por enlace: tasa $\times\eta^k$ (Cap. 16), no fidelidad.

## 22.3 Contraste operativo

| Métrica / Metric | Estándar | `qang` (`qang.qbattery` + JAX) | Estatus |
| :--- | :--- | :--- | :--- |
| Carga extraíble | diagonalizar $\rho$ | $(r-qg_Z)/2$ | (`exact`) |
| Lectura $0$ m-qg | ambigua | $r$ discrimina $W=\omega/2$ vs $0$ | (`exact`) |
| Enlaces máximos a $V=0.9$ | simular cadena | $k_{\max}=10$; $V^{10}=349$ m-qg | (`exact`, Werner) |
| Diseño de repetidores | — | $qg_{ZZ}$ por enlace (1 base) + producto | (`exact`); no Werner `mixed` |
| Ventaja colectiva de carga | — | — | `unverified` |

## 22.4 Bloque JAX (`qang/qbattery.py`)

```python
@jax.jit
def ergotropy_qg(r, qz):            # (r − qg_Z)/2
    return (r - qz) / 2.0
def swap_visibility(V, k):          # V**k
def max_links(V):                   # floor(ln(1/3)/ln V)
def chain_two_links_numeric(V1, V2) # (id ⊗ Λ_{V2})(ρ_{V1}) → correladores = V1·V2
```

## 22.5 Verificación — `tools/verify_cap22.py`

```
[1] qg_Z= -600 m-qg: E/ω=0.80 | W/ω puro=0.80 | W/ω diagonal=0.60 | fracción pasiva=0.25
[2] misma lectura 0 m-qg: |+> W=0.5ω vs I/2 W=0
[3] V=0.9: k_max=10 enlaces; V^k_max=0.349 ⇒ 349 m-qg; k_max+1 → 314 m-qg
[4] dos enlaces 0.9 y 0.8: qg_ZZ extremo = 0.720 = V1·V2 (720 m-qg)
verify_cap22: OK
```
Asserts: $W_{\text{puro}}=E$ y $W_{\text{diag}}=\max(0,-qg_Z)$ a $10^{-15}$; $k_{\max}(0.9)=10$, $k_{\max}(0.95)=21$; correladores $0.72$ a $10^{-13}$.

### `tests/test_qbattery.py`
```python
import pytest; jax = pytest.importorskip("jax")
from qang.qbattery import ergotropy_qg, max_links, chain_two_links_numeric
def test_plus_state_half(): assert abs(float(ergotropy_qg(1.0, 0.0)) - 0.5) < 1e-15
def test_links(): assert max_links(0.9) == 10
def test_chain(): assert abs(float(chain_two_links_numeric(0.9, 0.8)[2]) - 0.72) < 1e-13
```

## 22.6 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `ergotropy_qg` | energía | $(r-qg_Z)/2$ | `ergotropy_qg` | `exact` |
| `passive_fraction_qg` | energía | $(1-r)/(1-qg_Z)$ | `passive_fraction` | `exact` |
| `repeater_visibility` | redes | $V^k>1/3$, $k_{\max}$ | `swap_visibility`, `max_links` | `exact` (Werner) |

`CHANGELOG.md [Unreleased]`: `qang.qbattery`; `tools/verify_cap22.py`; `tests/test_qbattery.py`.
Commit: `docs(cap22): baterías y redes reescritos sobre Cap. 1–21 + qang.qbattery + verify_cap22`.
