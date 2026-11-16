# Capítulo 21 — Números aleatorios certificados y auditoría de canal: entropía mínima con intervalo creíble y deriva por bloques en m-qg
# Chapter 21 — Certified Random Numbers and Channel Audit: Min-Entropy with a Credible Interval and Block Drift in m-qg

**Actualizado:** 15 de noviembre de 2026 · **Updated:** November 15, 2026

> Reescritura del antiguo Cap. 18 sobre la base de Cap. 1–20 (§31, §41, §42). Módulo: `qang.qrng` (NumPy/SciPy opcional; compatible con el core). Verificación: `tools/verify_cap21.py`.

---

## 21.1 Punto frente a certificado

**ES.** $H_{\min}=-\log_2\frac{1+|qg_Z|}{2}$ (Cap. 19) evaluada en el **punto** $\hat{qg}_Z=2k_0/N-1$ sobreestima: con $k_0=N/2$ da $1.000$ bit para cualquier $N$. La versión certificada toma el máximo de $|qg_Z|$ sobre el intervalo creíble al $99.9\%$ del posterior Bayes-Haar $\mathrm{Beta}(1+k_0,1+N-k_0)$ (Cap. 8):
$$H_{\min}^{\text{cert}}=-\log_2\frac{1+\max(|qg_Z^{\text{lo}}|,|qg_Z^{\text{hi}}|)}{2}:\quad N=10^3\to0.866,\ 10^4\to0.956,\ 10^5\to0.986,\ 10^6\to0.9955\ \text{bit}.$$
Presupuesto para certificar $H_{\min}\ge0.99$ ($|qg_Z|\le7.0$ m-qg al $99.9\%$): $N\approx1.97\cdot10^5$ disparos (`exact` para el modelo i.i.d.). Con sesgo real de $30$ m-qg ($N=2\cdot10^5$): punto $0.9528$, certificado $0.9431$, verdadero $0.9574$ — el certificado es cota inferior.

**EN.** Point min-entropy overestimates; the certified value uses the $99.9\%$ Bayes-Haar credible interval of $qg_Z$ and is a lower bound (`exact`, i.i.d. model).

## 21.2 Auditoría de canal: estacionariedad de $qg_Z$ por bloques

$\chi^2$ de homogeneidad de $p_0$ entre bloques de $10^4$ ($19$ g.d.l.): canal estacionario $27.1$; deriva lineal de $0\to80$ m-qg: $130.7$, con lecturas por bloque $+7,-32,-22,-49,-71$ m-qg. La deriva invalida la hipótesis i.i.d. de §21.1 antes de que el punto lo muestre (`exact` estadístico). Extracción (Toeplitz/Trevisan) y certificación por desigualdades de Bell quedan referenciadas (§41, §42), no reproducidas.

## 21.3 Contraste operativo

| Métrica / Metric | Estándar | `qang` (`qang.qrng`) | Estatus |
| :--- | :--- | :--- | :--- |
| $H_{\min}$, $N=10^4$, sin sesgo | punto: $1.000$ | certificado $99.9\%$: $0.956$ | cota honesta (`exact`) |
| Disparos para $H_{\min}\ge0.99$ | — | $1.97\cdot10^5$ | (`exact`) |
| Sesgo oculto de $30$ m-qg | punto $0.953>$ verdadero | cert $0.943<$ verdadero | (`exact`) |
| Deriva de canal | — | $\chi^2$ $27\to131$; bloques en m-qg | (`exact`) |
| Certificación por Bell / extractores | §41–42 | referenciado | `unverified` aquí |

## 21.4 Bloque (`qang/qrng.py`)

```python
def posterior_qz_quantiles(k0, N, q=0.999):   # cuantiles de Beta(1+k0, 1+N−k0) → qg_Z = 2p0 − 1
def certified_min_entropy(k0, N, q=0.999):    # −log2((1 + max|qg_Z| en el intervalo)/2)
def shots_for_certified_hmin(h, z=3.09):      # q_max = 2·2^−h − 1 ; N = (z/q_max)²
def block_drift_chi2(bits, block):            # χ² de homogeneidad; devuelve qg_Z por bloque en m-qg
```

## 21.5 Verificación — `tools/verify_cap21.py`

```
[1] N=    10000, k0=N/2: H_min punto = 1.0000 | intervalo 99.9% qg_Z = [-30.9, +30.9] m-qg | H_min cert = 0.9561 bit
[2] H_min ≥ 0.99 certificado: |qg_Z| ≤ 7.0 m-qg ⇒ N ≈ 197358 disparos
[3] sesgo real 30 m-qg, N=2e5: H_min punto=0.9528, cert=0.9431, verdadero=0.9574
[4] χ²(19 g.d.l.): estacionario 27.1 | deriva 130.7
verify_cap21: OK
```
Asserts: cert$(10^6)>0.99$, cert$(10^3)<0.9$; cert $\le$ verdadero con sesgo; $\chi^2_{\text{deriva}}>3\chi^2_{\text{estac}}$.

### `tests/test_qrng.py`
```python
import pytest; jax = pytest.importorskip("jax")
from qang.qrng import certified_min_entropy, point_min_entropy
def test_cert_below_point(): assert certified_min_entropy(5000, 10000) < point_min_entropy(5000, 10000)
def test_cert_grows_with_N(): assert certified_min_entropy(500000, 1000000) > certified_min_entropy(5000, 10000)
```

## 21.6 Apéndice H

| id | familia | fórmula | función | estatus |
| :--- | :--- | :--- | :--- | :--- |
| `hmin_certified_haar` | criptografía | intervalo creíble Beta → $H_{\min}$ | `certified_min_entropy` | `exact` (i.i.d.) |
| `hmin_shots` | estadística | $N=(z/(2\cdot2^{-h}-1))^2$ | `shots_for_certified_hmin` | `exact` |
| `channel_drift_chi2` | auditoría | $\chi^2$ por bloques en m-qg | `block_drift_chi2` | `exact` |

`CHANGELOG.md [Unreleased]`: `qang.qrng`; `tools/verify_cap21.py`; `tests/test_qrng.py`.
Commit: `docs(cap21): QRNG certificado y auditoría de canal reescritos sobre Cap. 1–20 + qang.qrng + verify_cap21`.
