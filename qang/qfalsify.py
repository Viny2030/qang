"""qang.qfalsify — Cap. 31: protocolo de falsación pre-registrada: registro con SHA-256, p-valor binomial exacto,
decisión confirmada/refutada/no concluyente, corrección de Holm y potencia (disparos mínimos). Sólo biblioteca estándar.
"""
from __future__ import annotations

import hashlib
import json
from math import comb, exp, lgamma, log
from typing import Dict, Sequence, Tuple


# ------------------------------------------------------------------ registro
def canonical(prediction: Dict) -> str:
    """JSON canónico (claves ordenadas, sin espacios, UTF-8): lo que se hashea."""
    return json.dumps(prediction, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def register(prediction: Dict) -> str:
    """SHA-256 hexadecimal de la predicción canónica. Debe guardarse (commit) ANTES de medir."""
    return hashlib.sha256(canonical(prediction).encode("utf-8")).hexdigest()


def verify_registration(prediction: Dict, digest: str) -> bool:
    return register(prediction) == digest


# ------------------------------------------------------------------ pruebas exactas
def _logpmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else float("-inf")
    if p >= 1.0:
        return 0.0 if k == n else float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p)


def binom_upper(k: int, n: int, p0: float) -> float:
    """P(X ≥ k) con X ~ Bin(n, p0), exacto."""
    return float(min(1.0, sum(exp(_logpmf(j, n, p0)) for j in range(k, n + 1))))


def binom_lower(k: int, n: int, p0: float) -> float:
    """P(X ≤ k)."""
    return float(min(1.0, sum(exp(_logpmf(j, n, p0)) for j in range(0, k + 1))))


def decide(k: int, n: int, p0: float, alpha: float = 0.05) -> str:
    """Predicción «p > p0»: confirmada si P(X≥k) ≤ α; refutada si P(X≤k) ≤ α (los datos favorecen p < p0); si no, no concluyente."""
    if binom_upper(k, n, p0) <= alpha:
        return "confirmada"
    if binom_lower(k, n, p0) <= alpha:
        return "refutada"
    return "no concluyente"


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> Tuple[bool, ...]:
    """Holm–Bonferroni: controla la probabilidad de al menos un falso positivo ≤ α."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    rej = [False] * m
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return tuple(rej)


def _upper_tails(n: int, p: float):
    """[P(X ≥ k) for k = 0..n+1] en O(n)."""
    pm = [exp(_logpmf(k, n, p)) for k in range(n + 1)]
    out = [0.0] * (n + 2)
    acc = 0.0
    for k in range(n, -1, -1):
        acc += pm[k]
        out[k] = min(1.0, acc)
    return out


def power(n: int, p0: float, p1: float, alpha: float = 0.05) -> float:
    """Potencia exacta de la prueba unilateral «p > p0» cuando el valor verdadero es p1."""
    t0 = _upper_tails(n, p0)
    kc = next((k for k in range(n + 1) if t0[k] <= alpha), n + 1)
    return _upper_tails(n, p1)[kc] if kc <= n else 0.0


def shots_for_power(p0: float, p1: float, alpha: float = 0.05, target: float = 0.9, n_max: int = 20000) -> int:
    """Menor n con potencia ≥ target (el primero que la alcanza y la mantiene en n..n+20)."""
    for n in range(5, n_max):
        if power(n, p0, p1, alpha) >= target and all(power(n + d, p0, p1, alpha) >= target for d in range(1, 21)):
            return n
    raise ValueError("n_max insuficiente")
