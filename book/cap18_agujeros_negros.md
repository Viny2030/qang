# Capítulo 18 — Agujeros negros como scramblers: Hayden–Preskill, la ley $P\cdot F=1/4$ del decodificador de Yoshida–Kitaev, OTOC y por qué $qg_S$ no ve la curva de Page
# Chapter 18 — Black Holes as Scramblers: Hayden–Preskill, the $P\cdot F=1/4$ Law of the Yoshida–Kitaev Decoder, OTOC and Why $qg_S$ Cannot See the Page Curve

> Reescritura del antiguo Cap. 15 sobre la base de Cap. 1–17. Módulo: `qang.qscramble` (JAX, `jax<0.9`; §164). Verificación: `tools/verify_cap18.py` (10 qubits en statevector).

---

## 18.1 Decodificador probabilístico y la identidad $P_{\text{EPR}}F_{\text{EPR}}=1/d_A^2$

**ES.** Mensaje $A$ ($d_A=2$) entrelazado con la referencia $R$; agujero negro $B$ ($n_B=3$) maximalmente entrelazado con la radiación temprana $B'$; dinámica $U$ en $AB$; radiación tardía $D$ ($n_D=2$). El decodificador aplica $U^*$ sobre $A'B'$ (con $A'R'$ en EPR) y **proyecta** $DD'$ sobre EPR: éxito con probabilidad $P_{\text{EPR}}$ y fidelidad $F_{\text{EPR}}=\langle\text{EPR}|\rho_{RR'}|\text{EPR}\rangle$. Identidad (Yoshida–Kitaev):
$$P_{\text{EPR}}\cdot F_{\text{EPR}}=\frac1{d_A^2}=\frac14\quad\text{para toda }U,$$
verificada a $10^{-12}$ para $U$ Haar y $U=I$. Para $U$ Haar: $P\approx d_A^{-2}+d_D^{-2}=0.3125$ (medido $0.298\pm0.003$ con correcciones de tamaño finito), $F\approx0.84$. La fidelidad se lee con **tres correladores** de $(R,R')$ (Cap. 15): $F=(1+qg_{XX}-qg_{YY}+qg_{ZZ})/4$, coincidente con la proyección directa a $10^{-12}$ (`exact`). Presupuesto: $\times1/P_{\text{EPR}}\approx3.4$ disparos por decodificación anunciada.

**EN.** The probabilistic decoder obeys $P\cdot F=1/d_A^2$ exactly; the fidelity is read from three $(R,R')$ correlators, no tomography (`exact`).

## 18.2 OTOC como diagnóstico de scrambling

$\mathrm{OTOC}=\tfrac1d\mathrm{Re\,Tr}(W^\dagger V_t^\dagger WV_t)$ con $W=Z_A$, $V=Z_{B_3}$: $+1.000$ para $U=I$, $-0.009\approx O(1/d^2)$ para $U$ Haar. Es una lectura $qg$ de un circuito compuesto ($U^\dagger VU$ seguido de $W$), coste 1 configuración (`exact`).

## 18.3 Hallazgo negativo: $qg_S$ es plano donde la curva de Page sube y baja

Estado Haar de $n=8$ qubits. Rényi-2 del subsistema de $k$ qubits (desde la pureza, $4^k-1$ correladores): $0.98,1.90,2.61,\mathbf{3.00},2.61,1.89,0.95$ — curva de Page. Media de $qg_S$ (una base, Cap. 1): $1.00$ para todo $k$. **$qg_S$ no mide entrelazamiento** (THEORY_CORE §2.4, `against`); su uso aquí es sólo como coordenada de lectura, nunca como entropía de entrelazamiento.

## 18.4 Contraste operativo

| Métrica / Metric | Estándar | `qang` (`qang.qscramble` + JAX) | Estatus |
| :--- | :--- | :--- | :--- |
| Fidelidad de decodificación | tomografía de $\rho_{RR'}$ (9 bases) | 3 correladores $qg$ | $-6$ bases (`exact`) |
| Verificación de $P\cdot F=1/4$ | — | $10^{-12}$ en 3 $U$; 20 Haar | `exact` |
| Coste del decodificador | — | $\times1/P\approx3.4$ disparos; 2 copias del sistema | `exact` (conteo) |
| Scrambling | — | OTOC: 1 configuración | `exact` |
| Curva de Page con $qg_S$ | — | plano $1.00$ | `against` |

## 18.5 Bloque JAX (`qang/qscramble.py`, núcleo)

```python
def yoshida_kitaev(U, nB, nD):          # (P_EPR, F_EPR, F vía qg_XX, qg_YY, qg_ZZ de (R,R'))
    psi = initial_state(nB)             # EPR(R,A) ⊗ EPR(B_i,B'_i) ⊗ EPR(A',R')
    psi = _apply(psi, U, AB, n); psi = _apply(psi, jnp.conj(U), ABp, n)
    for j in range(nD): psi = _apply(psi, jnp.outer(EPR, jnp.conj(EPR)), (AB[j], ABp[j]), n)
    P = ‖psi‖²; rho = ρ_{RR'}(psi/√P)
    return P, <EPR|rho|EPR>, (1 + qg_XX − qg_YY + qg_ZZ)/4

def otoc(U, W, V):                      # (1/d) Re Tr(W† V_t† W V_t), V_t = U† V U
def renyi2_subsystem(psi, n, k)         # −log2 Tr ρ_k²
def qg_S_mean(psi, n, k)                # media de H((1+qg_Z)/2) sobre k qubits (sólo base Z)
```

## 18.6 Verificación — `tools/verify_cap18.py`

```
[1] U=Haar #1: P_EPR=0.2964  F_EPR=0.8435  P·F=0.250000  F(qg)=0.8435 → disparos ×3.37
[1] U=I      : P_EPR=0.2500  F_EPR=1.0000  P·F=0.250000  (A ⊂ D: trivialmente decodificable)
    <P_EPR>_Haar (20 muestras) = 0.2980 ± 0.0027  (teoría asintótica 0.3125)
[2] OTOC(Haar) = -0.0090 | OTOC(I) = +1.0000
[3] S2(k):  0.98 1.90 2.61 3.00 2.61 1.89 0.95 | qg_S(k): 1.00 × 7
verify_cap18: OK
```
Asserts: $|PF-\tfrac14|<10^{-12}$, $|F-F_{qg}|<10^{-12}$ (3 $U$); $|\langle P\rangle-0.3125|<0.03$; OTOC$(I)=1$; $S_2$ con máximo central y $\max qg_S-\min qg_S<0.05$.

### `tests/test_qscramble.py`
```python
import pytest; jax = pytest.importorskip("jax")
import jax.numpy as jnp
from qang.qscramble import yoshida_kitaev, haar_unitary

def test_pf_law():
    P, F, Fq = yoshida_kitaev(haar_unitary(jax.random.PRNGKey(0), 16), 3, 2)
    assert abs(P * F - 0.25) < 1e-12 and abs(F - Fq) < 1e-12
```

## 18.7 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `yk_pf_law` | scrambling | $P_{\text{EPR}}F_{\text{EPR}}=1/d_A^2$ | `yoshida_kitaev` | `exact` |
| `yk_fidelity_3qg` | scrambling | $F=(1+qg_{XX}-qg_{YY}+qg_{ZZ})/4$ | `yoshida_kitaev` | `exact` |
| `otoc_qg` | scrambling | $\tfrac1d\mathrm{Tr}(W^\dagger V_t^\dagger WV_t)$ | `otoc` | `exact` |
| `page_curve_qgS` | información | $qg_S$ plano vs $S_2$ | `renyi2_subsystem`, `qg_S_mean` | `against` |

`CHANGELOG.md [Unreleased]`: `qang.qscramble` (`haar_unitary`, `initial_state`, `yoshida_kitaev`, `otoc`, `renyi2_subsystem`, `qg_S_mean`); `tools/verify_cap18.py`; `tests/test_qscramble.py`.
Commit: `docs(cap18): Hayden–Preskill, ley P·F=1/4 y curva de Page reescritos sobre Cap. 1–17 + qang.qscramble + verify_cap18`.
