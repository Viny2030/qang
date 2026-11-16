# Capítulo 37 — Nanocuántica: túnel, Landauer, ruido de partición, ocupación de un nivel, espines en puntos cuánticos y centros NV leídos con la ecuación de qang
# Chapter 37 — Nanoquantum Systems: Tunnelling, Landauer, Partition Noise, Level Occupation, Quantum-Dot Spins and NV Centres Read with the qang Equation

**Actualizado:** 16 de noviembre de 2026 · **Updated:** November 16, 2026

> Reescritura del capítulo de nanocuántica sobre la base de Cap. 1–36. Módulo: `qang.qnano` (NumPy; el ruido binomial se calcula con la distribución exacta, sin Monte Carlo, salvo la termometría). Verificación: `tools/verify_cap37.py`. **Todo es cálculo y simulación: ninguna cifra viene de un experimento. En el único caso en que `qang` compite como estimador (la termometría), pierde.**
> **EN abstract.** Transmissions, occupations and spin populations are probabilities, so the qang equation reads them all alike: the half-angle of a tunnelling probability is the transmission amplitude modulus; Landauer's conductance is the mean $qg$ of the channels; partition noise is the binomial variance, stabilised by the angle ($\sqrt N\sigma_\theta\in[1.0005,1.0100]$ for $T\in[0.02,0.98]$). As a thermometer, the artanh line is biased ($-1.9\%$) and noisier than the maximum-likelihood fit. Calculation and simulation only.

---

## 37.1 Reglas de lectura

1. $p_0$ es, según el caso, la transmisión $T$, la ocupación $f$ de un nivel o la población de un estado de espín. Con $p_0$: $\theta=2\arccos\sqrt{p_0}$, $qg_Z=2p_0-1$, m-qg $=1000\,qg_Z$; 1 m-qg es siempre $\Delta p_0=0.0005$.
2. `qga` (1 vuelta = 1000 qga) sólo aparece como escala del ángulo: $\sigma_\theta=1/\sqrt N$ equivale a $159.15/\sqrt N$ qga.
3. Cada afirmación lleva su función en `qang.qnano`; el contraste con la forma usual va al lado, y se dice cuando `qang` pierde.

## 37.2 Túnel: el medio ángulo es la amplitud

Barrera rectangular de 1 eV, electrón de 0.5 eV ($\kappa=3.623$ nm⁻¹), fórmula exacta (`barrier_T`):

| ancho | $T$ | lectura | $\theta_T$ |
| :--- | :--- | :--- | :--- |
| 0.2 nm | 0.616 | +231.9 m-qg | 76.6° |
| 0.5 nm | 0.101 | −797.3 m-qg | 142.9° |
| 1.0 nm | $2.85\cdot10^{-3}$ | −994.3 m-qg | 173.9° |
| 1.5 nm | $7.62\cdot10^{-5}$ | −999.8 m-qg | 179.0° |

$\cos(\theta_T/2)=\sqrt T=|t|$ a $10^{-15}$: **el medio ángulo es el módulo de la amplitud de transmisión**. Sin `qang` se grafica $\ln T$, con pendiente $-2\kappa$ ($-7.242$ medida, $-7.245$ esperada). Con `qang`, la distancia al polo $\pi-\theta_T\approx2|t|$ cae con pendiente $-\kappa$ ($-3.622$ frente a $-3.623$ nm⁻¹): la amplitud en vez de la probabilidad, la misma información.

## 37.3 Landauer: un canal, una lectura

$G/G_0=\sum_nT_n=N/2+\tfrac12\sum_n qg_n$, con el modelo de punto de silla $T_n=1/(1+e^{-2\pi(V-n-1/2)/w})$ ($w=0.45$): identidad a $4.4\cdot10^{-16}$ (`conductance_from_qg`). Las mesetas están en $V=2,3,4$ con $G/G_0=1,2,3$; cada escalón es un canal que cruza el ecuador. La conductancia sola da $\sum T_n$; con el ruido de disparo $S=\sum T_n(1-T_n)$ se recuperan dos canales abiertos a la vez (`recover_two_channels`): de $G=1.3747$, $S=0.2682$ salen $T=0.9716$ y $0.4031$ exactos. Es la técnica usual; en qg, la conductancia lee el $qg$ medio y el ruido el $qg^2$ medio.

## 37.4 Ruido de partición: la varianza binomial, estabilizada

$T(1-T)=(1-qg^2)/4=\sin^2\theta/4$. Con $N=1000$ electrones por ventana y la distribución binomial **exacta** (`theta_std_sqrtN`):

| $T$ | 0.02 | 0.05 | 0.2 | 0.5 | 0.8 | 0.95 | 0.98 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\sqrt N\,\sigma(\hat T)$ | 0.140 | 0.218 | 0.400 | 0.500 | 0.400 | 0.218 | 0.140 |
| $\sqrt N\,\sigma(\hat\theta)$ | 1.0100 | 1.0038 | 1.0009 | 1.0005 | 1.0009 | 1.0038 | 1.0100 |

La barra de $\hat T$ depende de lo que se mide y cerca de los polos engaña; la de $\hat\theta$ es la misma en todo el rango. Es la transformación arcoseno, conocida en estadística: `qang` no la inventa; aporta que el ángulo estabilizado es la coordenada natural del estado.

## 37.5 Bloqueo de Coulomb: la ocupación de un nivel

$qg_Z=2f-1=-\tanh(x/2)$, con $f=1/(1+e^x)$, $x=\varepsilon/k_BT$ (a $10^{-15}$): el mismo estado térmico que $qg_Z=\tanh(\beta h)$ del Cap. 11 (ex-8), con el signo de la convención. Como termómetro (25 niveles, 200 lecturas por nivel, 400 repeticiones, $k_BT=1$):

| método | $k_BT$ estimado (semilla 34) | 4 semillas |
| :--- | :--- | :--- |
| ajuste de Fermi por máxima verosimilitud (sin `qang`) | $1.0000\pm0.0275$ | 0.9997–1.0008 |
| recta de $\operatorname{artanh}(qg_Z)$ por el origen (con `qang`) | $0.9808\pm0.0366$ | 0.9805–0.9813 |

**`qang` pierde.** La recta es cerrada pero tiene un sesgo de $-1.9\%$ y un desvío $1.3\times$ mayor: $\operatorname{artanh}$ amplifica el ruido cerca de $\pm1$ y hay que descartar los puntos saturados. La versión anterior del capítulo daba $+2.9\%$ con otra receta (recta con ordenada); **el signo y el tamaño del sesgo dependen de la receta, no el resultado**: ninguna recta en artanh supera al ajuste de Fermi. La lectura qg sirve para ver el termómetro, no para estimar con él.

## 37.6 Qubits de espín en puntos cuánticos

La lectura por sensor de carga tiene errores $e_0=P(1\mid0)$, $e_1=P(0\mid1)$, y la regla del Cap. 30 (ex-27) en m-qg es $m'=(1-e_0-e_1)m+1000(e_1-e_0)$ (`readout_affine`). Con $e_0=0.02$, $e_1=0.08$ (valores ilustrativos): $+1000\to960$ y $-1000\to-840$ m-qg, es decir **amplitud 900 m-qg y desplazamiento +60 m-qg** (cada punto porcentual de error de lectura cuesta 10 m-qg de visibilidad); corregir amplifica el ruido $1/(1-e_0-e_1)=1.11\times$. Es la corrección estándar escrita lineal.

## 37.7 Centros NV: el contraste como presupuesto

Con $n\sim\text{Poisson}(N_b[1-C(1-p_0)])$, la lectura de $p_0$ en m-qg tiene (`nv_sigma_mqg`)

$$\sigma_m=\frac{2000\sqrt{1-C(1-p_0)}}{C\sqrt{N_b}}\ \approx\ \frac{2000}{C\sqrt{N_b}}.$$

Con $C=0.3$ y $N_b=444\,444$: la aproximación da **10.00 m-qg** y la fórmula exacta **9.22 m-qg**; para exactamente 10 m-qg hacen falta **377 778 fotones** (`nv_photons_for`). El contraste $C=0.3$ cuesta ≈$1/C^2=11.1\times$ los fotones de $C=1$. En ODMR (línea lorentziana de 1 MHz, $C=0.2$, 61 frecuencias en $\pm3$ anchos, 1.16 M fotones), la cota de Cramér–Rao del centro es **8.55 kHz = 305 nT** con $\gamma=28$ GHz/T (`odmr_fisher_crb_hz`). El ajuste de la línea no cambia con `qang`; la escala agrega la cuenta previa en una unidad lineal en la población.

## 37.8 Contraste operativo: sin `qang` frente a con `qang`

| Tema | Sin `qang` | Con `qang` (`qang.qnano`) | Estatus |
| :--- | :--- | :--- | :--- |
| Túnel | $\ln T$, pendiente $-2\kappa$ | $\cos(\theta_T/2)=\lvert t\rvert$; $\pi-\theta_T$ con pendiente $-\kappa$ | `exact` |
| Landauer | $G/G_0=\sum T_n$ | $N/2+\sum qg_n/2$; mesetas en los polos | `exact` |
| Dos canales | $G$ y $S$ | los mismos; recupera 0.9716 y 0.4031 | `exact` |
| Ruido de partición | barra de $\hat T$: 0.14 a 0.50 | barra de $\hat\theta$: 1.0005 a 1.0100 | `exact` |
| Termometría | Fermi ML: $1.0000\pm0.0275$ | artanh: $0.9808\pm0.0366$ | `against` |
| Lectura de espín | 900 m-qg y +60 m-qg sin corregir | corregida con los polos, ruido ×1.11 | `exact` |
| NV | contraste y fotones | 377 778 fotones para 10 m-qg ($C=0.3$) | `exact` |
| ODMR | CRB 8.55 kHz | lo mismo (305 nT) | `exact` |

## 37.9 Bloque (`qang/qnano.py`)

```python
def barrier_T(a_nm, E_eV=0.5, V_eV=1.0)       # transmisión exacta
def conductance_from_qg(T); recover_two_channels(G, S)
def theta_std_sqrtN(T, N)                      # σ(θ̂)·√N con la binomial exacta
def occupancy_qgz(x)                           # −tanh(x/2)
def fit_kT_mle(eps, ones, N); fit_kT_artanh(eps, ones, N); thermometry(...)
def readout_affine(m, e0, e1)
def nv_sigma_mqg(C, Nb, p0=0.5); nv_photons_for(sigma_mqg, C); odmr_fisher_crb_hz(...)
```

## 37.10 Verificación — `tools/verify_cap37.py`

```
[1] κ = 3.623 nm⁻¹; 0.2 nm: T=0.616, +231.9 m-qg, θ=76.6°; 0.5 nm: T=0.101, -797.3 m-qg, θ=142.9°; 1.0 nm: T=0.00285, -994.3 m-qg, θ=173.9°; 1.5 nm: T=7.62e-05, -999.8 m-qg, θ=179.0°
    cos(θ_T/2) = √T = |t| a 1e-15; pendiente de ln T: -7.242 (−2κ = -7.245); de ln(π−θ_T): -3.622 (−κ = -3.623) nm⁻¹
[2] G/G0 = N/2 + Σqg/2 a 4.4e-16; mesetas en V=2,3,4: [1.0, 2.0, 3.0]; dos canales (G=1.3747, S=0.2682) → T = 0.9716, 0.4031
[3] √N·σ(θ̂) = [1.01, 1.0038, 1.0009, 1.0005, 1.0009, 1.0038, 1.01] para T=[0.02, 0.05, 0.2, 0.5, 0.8, 0.95, 0.98]; √N·σ(T̂) = [0.14, 0.218, 0.4, 0.5, 0.4, 0.218, 0.14]
[4] qg_Z = 2f − 1 = −tanh(x/2) a 1e-15
[5] k_BT: máxima verosimilitud 1.0000 ± 0.0275; recta en artanh 0.9808 ± 0.0366 (4 semillas: ML 0.9997–1.0008, artanh 0.9805–0.9813)
[6] +1000 → 960, −1000 → -840 m-qg: amplitud 900 m-qg, desplazamiento +60 m-qg; ruido ×1.11 al corregir
[7] NV C=0.3, Nb=444 444: σ aproximada 10.00 m-qg, exacta 9.22 m-qg; para 10 m-qg exactos hacen falta 377,778 fotones (C=0.3 pide 11.1× los de C=1); ODMR 1 MHz, C=0.2, 61 puntos, 1.16 M fotones: CRB 8.55 kHz = 305 nT (γ=28 GHz/T)
verify_cap37: OK
```

### `tests/test_qnano.py`
```python
import numpy as np
from qang.qnano import barrier_T, theta_from_p0, recover_two_channels, theta_std_sqrtN, thermometry, nv_sigma_mqg, nv_photons_for
def test_tunnel_amplitude():
    T = barrier_T(1.0); assert abs(np.cos(theta_from_p0(T) / 2) - np.sqrt(T)) < 1e-15 and abs(T - 2.85e-3) < 1e-5
def test_two_channels():
    r = recover_two_channels(0.4031 + 0.9716, 0.4031 * 0.5969 + 0.9716 * 0.0284); assert abs(r[0] - 0.9716) < 1e-12
def test_arcsine_variance_stabilized(): assert all(1.0 <= theta_std_sqrtN(t, 400) < 1.02 for t in (0.05, 0.3, 0.5, 0.9))
def test_thermometry_loses():
    (ml, _), (ar, _) = thermometry(reps=60, seed=5); assert abs(ml - 1) < 0.02 and ar < 0.995
def test_nv(): assert abs(nv_sigma_mqg(0.3, nv_photons_for(10, 0.3)) - 10) < 1e-9
```

(El resto, `test_landauer_identity_and_two_channels`, `test_fermi_tanh_and_thermometry_loses` y `test_readout_and_nv`, está en `tests/test_qnano.py`.)

## 37.11 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `tunnel_amplitude` | nano | $\cos(\theta_T/2)=\lvert t\rvert$ | `barrier_T`, `theta_from_p0` | `exact` |
| `landauer_qg_identity` | nano | $G/G_0=N/2+\sum qg_n/2$ | `conductance_from_qg`, `qpc_T` | `exact` |
| `two_channel_recovery` | nano | $G=\sum T$, $S=\sum T(1-T)$ → $T_1,T_2$ | `recover_two_channels` | `exact` |
| `arcsine_partition_noise` | nano | $\sqrt N\sigma_\theta\in[1.0005,1.0100]$ | `theta_std_sqrtN` | `exact` |
| `fermi_qgz_tanh` | nano | $qg_Z=-\tanh(x/2)$ | `occupancy_qgz` | `exact` |
| `thermometry_artanh_loses` | nano | artanh $0.9808\pm0.0366$ frente a ML $1.0000\pm0.0275$ | `fit_kT_artanh`, `fit_kT_mle`, `thermometry` | `against` |
| `spin_readout_affine` | nano | $m'=(1-e_0-e_1)m+1000(e_1-e_0)$ | `readout_affine` | `exact` |
| `nv_photon_budget` | nano | $\sigma_m=2000\sqrt{1-C(1-p_0)}/(C\sqrt{N_b})$ | `nv_sigma_mqg`, `nv_photons_for` | `exact` |
| `odmr_crb` | nano | CRB 8.55 kHz (305 nT) | `odmr_fisher_crb_hz` | `exact` |

`CHANGELOG.md [Unreleased]`: `qang.qnano`; `tools/verify_cap37.py`; `tests/test_qnano.py`.
Commit: `docs(cap37): nanocuántica leída con qang + qang.qnano + verify_cap37`.
