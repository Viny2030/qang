# Capítulo 15 — Teleportación óptica y el límite del par de Bell: el canal como contracción lineal en m-qg y el umbral de 333 m-qg
# Chapter 15 — Optical Teleportation and the Bell-Pair Limit: the Channel as a Linear m-qg Contraction and the 333 m-qg Threshold

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 12 sobre la base de Cap. 1–14. Módulo: `qang.qtele` (JAX, `jax<0.9`; alineado con `qang.qpon`, §57, §157). Verificación: `tools/verify_cap15.py`. Doble carril y filtro de sector a larga distancia: Cap. 16.

---

## 15.1 El recurso de Werner leído en tres correladores

**ES.** $\rho_V=V|\Phi^+\rangle\langle\Phi^+|+(1-V)\tfrac{I}{4}$ tiene $qg_{XX}=V$, $qg_{YY}=-V$, $qg_{ZZ}=V$ (Cap. 7: correladores de una base cada uno, 0 ancillas). Fracción de singlete y fidelidad media de teleportación (Horodecki):
$$F_s=\tfrac14\big(1+qg_{XX}-qg_{YY}+qg_{ZZ}\big),\qquad \bar F=\frac{2F_s+1}{3},\qquad \bar F_{\text{Werner}}=\frac{1+V}{2}.$$
Umbral clásico $\bar F=2/3\iff F_s=1/2\iff V=1/3$: **el par de Bell es útil si y sólo si $qg_{ZZ}>333$ m-qg** bajo el modelo de Werner (`exact`). Para un recurso general se necesitan las 3 bases; la lectura de una sola base es cota superior (§15.4).

**EN.** The Werner resource is read as three single-basis correlators; the classical threshold is $qg_{ZZ}=333$ m-qg under the Werner model (`exact`), and an upper bound otherwise.

## 15.2 El canal de teleportación en m-qg: m'=V m

Teleportación completa (medida de Bell sobre entrada⊗A, corrección de Pauli en B) con recurso $\rho_V$ es el canal depolarizante $\mathbf r\mapsto V\mathbf r$; en lectura: $m\text{-qg}'=V\cdot m\text{-qg}$ para cualquier eje, verificado a $10^{-13}$ sobre entradas aleatorias. Ejemplo: $+600$ m-qg $\to+420$ ($V=0.7$) $\to+200$ ($V=1/3$). El déficit disipativo del estado teleportado (Cap. 5) es $qg_\tau=\sqrt{1-V^2r^2}$. La fidelidad promedio sobre el 2-diseño de 6 autoestados de Pauli reproduce $(1+V)/2$ a $10^{-13}$ (`exact`).

## 15.3 Presupuesto de disparos y configuraciones

Desde $qg_{ZZ}$: $\sigma_{\bar F}=\tfrac12\sqrt{(1-V^2)/N}$ ⇒ $N=(1-V^2)/(4\sigma_{\bar F}^2)$: $475$ disparos ($V=0.9$) o $1\,600$ ($V=0.6$) para $\sigma_{\bar F}=0.01$ en **una** configuración. Tomografía de proceso del canal: 4 entradas $\times$ 3 bases $=12$ configuraciones, $\ge12N$ disparos para la misma precisión por celda (`exact`, conteo).

## 15.4 Hallazgo `mixed`: la lectura de una base sobreestima recursos no Werner

Recurso con desfase puro ($qg_{ZZ}=1$, $qg_{XX}=0.8$): $\bar F$ real (3 bases) $=0.9333$; $\bar F$ supuesta Werner desde $qg_{ZZ}$ $=1.0000$ (sesgo $+0.067$). Regla: **certificar con 3 bases; planificar con 1**. Un recurso con $qg_{ZZ}>1/3$ pero $qg_{XX}+qg_{ZZ}\le1$ no está certificado como entrelazado (testigo de 2 bases, Cap. 5).

## 15.5 Contraste operativo — Sin `qang` vs. con `qang`

| Métrica / Metric | Estándar (sin `qang`) | `qang` (`qang.qtele` + JAX) | Ganancia / Estatus |
| :--- | :--- | :--- | :--- |
| Decidir si el par sirve | tomografía de estado (9 bases) | $qg_{ZZ}>333$ m-qg (1 base, Werner) / $F_s>\tfrac12$ (3 bases) | $-8$ / $-6$ bases (`exact`) |
| Disparos para $\sigma_{\bar F}=0.01$, $V=0.9$ | proceso: 12 config., $\ge5\,700$ | 1 config., $475$ | $-92\%$ (`exact`) |
| Predicción de la salida | simular el canal | $m'=V\,m$ en cualquier eje | (`exact`) |
| Recurso no Werner | — | 1 base sesga $+0.067$; usar 3 bases | `mixed` |
| Hardware óptico (§57, §157) | — | referenciado, no reproducido aquí | `unverified` en este cap. |

## 15.6 Bloque JAX (`qang/qtele.py`)

```python
@jax.jit
def resource_correlators(rho2):                 # (qg_XX, qg_YY, qg_ZZ)
    f = lambda P: jnp.real(jnp.trace(rho2 @ jnp.kron(P, P)))
    return f(X), f(Y), f(Z)

@jax.jit
def singlet_fraction_from_qg(qxx, qyy, qzz):    # (1 + qg_XX − qg_YY + qg_ZZ)/4
    return (1.0 + qxx - qyy + qzz) / 4.0

@jax.jit
def teleport_fidelity_from_singlet(Fs):         # (2F_s + 1)/3 ; umbral 2/3
    return (2.0 * Fs + 1.0) / 3.0

def teleport(rho_in, rho_res):                  # medida de Bell + corrección de Pauli, ρ_out 2×2
    rho = kron(rho_in, rho_res); corr = [I2, Z, X, X @ Z]; out = 0
    for k, b in enumerate(bell_basis()):
        M = kron(jnp.outer(b, jnp.conj(b)), I2)
        rb = jnp.einsum("iaib->ab", (M @ rho @ M).reshape(4, 2, 4, 2))
        out = out + corr[k] @ rb @ jnp.conj(corr[k]).T
    return out

def shots_for_fidelity(V, sigma_F):             # (1 − V²)/(4σ_F²)
    return (1.0 - V ** 2) / (4.0 * sigma_F ** 2)
```

## 15.7 Verificación — `tools/verify_cap15.py` (salida reproducible)

```
[1] Werner: qg_ZZ = V, F_s = (1+3V)/4, F̄ = (2F_s+1)/3 = (1+V)/2  (1e-14)
[2] V=0.700: entrada +600 m-qg → salida +420 m-qg | V=0.333: → +200 m-qg  (m' = V·m)
[3] V=0.333 (qg_ZZ=333 m-qg): F̄=0.6667 → límite clásico (2/3) | V=0.900: F̄=0.9500 → cuántico
[4] V=0.9: N(σ_F=0.01) = 475 disparos en 1 configuración | tomografía de proceso: 12 configuraciones ≥ 5700
[5] no Werner: qg_XX=0.80, qg_ZZ=1.00; F̄(3 bases)=0.9333, F̄(1 base)=1.0000 → sesgo +0.0667 (mixed)
verify_cap15: OK
```
Asserts: correladores y $F_s$ a $10^{-14}$; Bloch de salida $=V\mathbf r$ a $10^{-13}$ (9 entradas aleatorias); $\bar F$ sobre 2-diseño a $10^{-13}$; $\bar F(V=1/3)=2/3$ a $10^{-15}$; $\bar F_{1\text{ base}}\ge\bar F_{3\text{ bases}}$.

### `tests/test_qtele.py`
```python
import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qtele import werner, teleport, bloch, state_from_bloch, teleport_fidelity_werner_qzz, resource_correlators

def test_channel_contracts_bloch():
    r = jnp.array([0.3, -0.4, 0.5]); out = teleport(state_from_bloch(r), werner(0.7))
    assert np.allclose(np.asarray(bloch(out)), 0.7 * np.asarray(r), atol=1e-13)

def test_classical_threshold_333_mqg():
    assert abs(float(teleport_fidelity_werner_qzz(1 / 3)) - 2 / 3) < 1e-15

def test_werner_correlators():
    qxx, qyy, qzz = resource_correlators(werner(0.5))
    assert abs(float(qxx) - 0.5) < 1e-14 and abs(float(qyy) + 0.5) < 1e-14 and abs(float(qzz) - 0.5) < 1e-14
```

## 15.8 Apéndice H — entradas nuevas en `qang.measures`

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `bell_threshold_mqg` | teleportación | $qg_{ZZ}>1/3$ (333 m-qg) ⇔ $\bar F>2/3$ (Werner) | `teleport_fidelity_werner_qzz` | `exact` (modelo) |
| `singlet_fraction_3bases` | teleportación | $(1+qg_{XX}-qg_{YY}+qg_{ZZ})/4$ | `singlet_fraction_from_qg` | `exact` |
| `teleport_channel_mqg` | teleportación | $m'=V\,m$ | `teleport`, `bloch` | `exact` |
| `teleport_shots` | estadística | $N=(1-V^2)/(4\sigma_F^2)$; 1 vs 12 config. | `shots_for_fidelity` | `exact` |
| `one_basis_bias` | teleportación | $\bar F_{1b}\ge\bar F_{3b}$ | `verify_cap15.py` §5 | `mixed` |

### `CHANGELOG.md` — `[Unreleased]`
```
### Added
- `qang.qtele`: `werner`, `resource_correlators`, `singlet_fraction_from_qg`, `teleport_fidelity_from_singlet`,
  `teleport_fidelity_werner_qzz`, `teleport`, `bloch`, `state_from_bloch`, `shots_for_fidelity` (JAX, `jax<0.9`).
- `tools/verify_cap15.py`, `tests/test_qtele.py`; Apéndice H: 5 entradas.
### Fixed
- `.github/workflows/book.yml`: `GH_TOKEN` en `quarto-actions/setup` (403 al resolver TinyTeX).
```
Commit: `docs(cap15): teleportación y límite del par de Bell reescritos sobre Cap. 1–14 + qang.qtele + verify_cap15`.
