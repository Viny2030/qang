# Capítulo 17 — Curvas temporales cerradas por postselección (P-CTC) frente a D-CTC: dos mapas, dos lecturas en m-qg y el precio en disparos
# Chapter 17 — Postselected Closed Timelike Curves (P-CTC) versus D-CTC: Two Maps, Two m-qg Readouts and the Shot Price

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 14 sobre la base de Cap. 1–16. Módulo: `qang.qctc` (JAX, `jax<0.9`; alineado con `qang.postselection`, §164). Verificación: `tools/verify_cap17.py`. Se simulan los **mapas**; la existencia física de CTC es `unverified` y no se afirma.

---

## 17.1 Los dos mapas

**ES.** Sea $U$ la interacción entre el qubit cronológico CR y el qubit CTC.
$$\text{P-CTC (Lloyd):}\quad \rho\mapsto\frac{C\rho C^\dagger}{\mathrm{Tr}(C\rho C^\dagger)},\ \ C=\mathrm{Tr}_{\text{CTC}}U,\ \ p_{\text{post}}=\tfrac14\mathrm{Tr}(C\rho C^\dagger);\qquad \text{D-CTC (Deutsch):}\quad \rho_{\text{CTC}}=\mathrm{Tr}_{\text{CR}}\big[U(\rho\otimes\rho_{\text{CTC}})U^\dagger\big].$$
P-CTC es la teleportación del Cap. 15 con la medida de Bell **postseleccionada**: cuesta un par de Bell (2 qubits) y $\times1/p_{\text{post}}$ disparos. D-CTC es un punto fijo no lineal (iteración desde $I/2$, selección de máxima entropía) sin análogo de circuito único.

**EN.** P-CTC is postselected teleportation (Bell pair + $1/p_{\text{post}}$ shots); D-CTC is a nonlinear fixed point. Both are simulated as maps and read in m-qg.

## 17.2 Tres interacciones, leídas en m-qg

| $U$ | $C=\mathrm{Tr}_{\text{CTC}}U$ | P-CTC | $p_{\text{post}}$ | D-CTC |
| :--- | :--- | :--- | :--- | :--- |
| $I$ | $2I$ | identidad | $1$ | identidad |
| CNOT (CR controla CTC) | $2|0\rangle\langle0|$ | **proyecta** a $+1000$ m-qg, cualquier entrada | $(1+qg_Z)/2$ | conserva $qg_Z$, **borra** $qg_X,qg_Y$ (1 iteración) |
| SWAP | $I$ | identidad | $1/4$ | identidad ($\rho_{\text{CTC}}=\rho$) |
| $(X\otimes I)\,$SWAP | $X$ | $\rho\mapsto X\rho X$ ($qg_Y,qg_Z\to-$) | $1/4$ | — |

Instancia CNOT: entrada $-980$ m-qg → P-CTC $+1000$ m-qg con $p_{\text{post}}=0.010$ ($\times100$ disparos); D-CTC $-980$ m-qg con $qg_X\to0$. La no linealidad de P-CTC se paga en el presupuesto: $N\propto1/p_{\text{post}}=2/(1+qg_Z)$, divergente en el polo $|1\rangle$ (`exact` para el mapa).

## 17.3 Contraste operativo

| Métrica / Metric | Estándar | `qang` (`qang.qctc` + JAX) | Estatus |
| :--- | :--- | :--- | :--- |
| Recursos P-CTC | — | 1 par de Bell + postselección $\times1/p_{\text{post}}$ | `exact` (conteo) |
| Recursos D-CTC | — | iteración de punto fijo (convergió en $\le2$ it. en los 3 casos) | `exact` sim. |
| Lectura discriminante | tomografía | $qg_Z$ y $qg_X$ de CR: P-CTC mueve $qg_Z$, D-CTC mata $qg_X$ | `exact` |
| CTC físicas | — | — | `unverified` |

## 17.4 Bloque JAX (`qang/qctc.py`)

```python
def pctc_operator(U):                            # C = Tr_CTC(U)
    return U.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)

def pctc_map(rho, U):                            # (ρ_out, p_post)
    C = pctc_operator(U); M = C @ rho @ jnp.conj(C).T
    return M / jnp.trace(M), jnp.real(jnp.trace(M)) / 4.0

def dctc_map(rho, U, iters=200):                 # punto fijo desde I/2
    rho_ctc = I2 / 2
    for k in range(iters):
        new = ptrace_first(U @ jnp.kron(rho, rho_ctc) @ jnp.conj(U).T)
        if float(jnp.max(jnp.abs(new - rho_ctc))) < 1e-14: rho_ctc = new; break
        rho_ctc = new
    return ptrace_second(U @ jnp.kron(rho, rho_ctc) @ jnp.conj(U).T), rho_ctc, k + 1
```

## 17.5 Verificación — `tools/verify_cap17.py`

```
[1] U=I: P-CTC y D-CTC = identidad; p_post = 1
[2] entrada  -980 m-qg: P-CTC → +1000 m-qg, p_post=0.010 (disparos ×100.0) | D-CTC → -980 m-qg, qg_X→0, punto fijo en 1 it.
[3] U=SWAP: P-CTC identidad con p_post=0.25 (×4 disparos) | D-CTC identidad, ρ_CTC=ρ en 2 it.
[4] U=(X⊗I)SWAP: C = X → ρ → XρX, p_post=0.25
verify_cap17: OK
```
Asserts: $C_{\text{CNOT}}=2|0\rangle\langle0|$; P-CTC $qg_Z\to1$ a $10^{-13}$ y $p_{\text{post}}=(1+qg_Z)/2$ a $10^{-14}$; D-CTC conserva $qg_Z$ y anula $qg_X$ a $10^{-13}$; SWAP: ambos identidad, $p_{\text{post}}=1/4$.

### `tests/test_qctc.py`
```python
import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qctc import pctc_map, dctc_map, bloch, state_from_bloch, CNOT, SWAP

def test_cnot_pctc_projects_dctc_decoheres():
    rho = state_from_bloch(jnp.array([0.4, 0.0, -0.6]))
    op, p = pctc_map(rho, CNOT); od, _, _ = dctc_map(rho, CNOT)
    assert abs(float(bloch(op)[2]) - 1) < 1e-13 and abs(float(p) - 0.2) < 1e-14
    assert abs(float(bloch(od)[2]) + 0.6) < 1e-13 and abs(float(bloch(od)[0])) < 1e-13

def test_swap_identity():
    rho = state_from_bloch(jnp.array([0.3, -0.4, 0.5])); op, p = pctc_map(rho, SWAP)
    assert np.allclose(np.asarray(bloch(op)), [0.3, -0.4, 0.5], atol=1e-14) and abs(float(p) - 0.25) < 1e-14
```

## 17.6 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `pctc_map_mqg` | postselección | $C\rho C^\dagger/\mathrm{Tr}$, $p_{\text{post}}=\mathrm{Tr}(C\rho C^\dagger)/4$ | `pctc_map` | `exact` (mapa); físico `unverified` |
| `dctc_fixed_point` | postselección | punto fijo de Deutsch desde $I/2$ | `dctc_map` | `exact` sim. |
| `pctc_shot_price` | estadística | $N\times1/p_{\text{post}}$ | `shots_multiplier_pctc` | `exact` |

`CHANGELOG.md [Unreleased]`: `qang.qctc` (`pctc_operator`, `pctc_map`, `dctc_map`, `bloch`, `state_from_bloch`, `shots_multiplier_pctc`); `tools/verify_cap17.py`; `tests/test_qctc.py`.
Commit: `docs(cap17): P-CTC vs D-CTC reescritos sobre Cap. 1–16 + qang.qctc + verify_cap17`.
