"""qang.qgarnet — Cap. 35: recálculo, desde los archivos guardados, de las corridas en hardware real (IQM Garnet, §165, §168,
§170). Sólo biblioteca estándar y NumPy: no hace falta iqm-client ni créditos. m-qg = 1000·qg_Z.

Convención de lectura (qubit de calibración): e0 = P(1|0) (se mide 1 con el qubit en |0>), e1 = P(0|1). Una lectura en
m-qg se mapea a m' = (1−e0−e1)·m + 1000·(e1−e0).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ lecturas de un qubit
def shots(c: Dict[str, int]) -> int:
    return int(sum(c.values()))


def p1(c: Dict[str, int]) -> float:
    return c.get("1", 0) / shots(c)


def raw_mqg(c: Dict[str, int]) -> float:
    """Lectura cruda en m-qg: 1000·(1 − 2 P(1))."""
    return 1000.0 * (1.0 - 2.0 * p1(c))


def readout_from_poles(c0: Dict[str, int], c1: Dict[str, int]) -> Tuple[float, float]:
    """(e0, e1) a partir de los circuitos de calibración «todo en |0>» y «todo en |1>»."""
    return c0.get("1", 0) / shots(c0), c1.get("0", 0) / shots(c1)


def contrast(e0: float, e1: float) -> float:
    return 1.0 - e0 - e1


def predicted_raw_mqg(target: float, e0: float, e1: float) -> float:
    """Lectura cruda que se espera si el qubit tiene exactamente `target` m-qg y sólo falla la lectura."""
    return contrast(e0, e1) * target + 1000.0 * (e1 - e0)


def correct_mqg(raw: float, e0: float, e1: float) -> float:
    """Inversión de la corrección de lectura."""
    return (raw - 1000.0 * (e1 - e0)) / contrast(e0, e1)


def binomial_sigma_mqg(p: float, n: int) -> float:
    """σ de una lectura en m-qg con n disparos: 2000·sqrt(p(1−p)/n)."""
    return 2000.0 * math.sqrt(p * (1.0 - p) / n)


# ------------------------------------------------------------------ §168: copias del ecuador y cadenas de X
def equator_dispersion_ratio(copies: Sequence[Dict[str, int]]) -> float:
    """Desvío observado de las lecturas crudas de las copias del ecuador dividido por el σ binomial al p agrupado (M2)."""
    m = np.array([raw_mqg(c) for c in copies])
    n = shots(copies[0])
    pooled = sum(c.get("1", 0) for c in copies) / (n * len(copies))
    return float(np.std(m, ddof=1) / binomial_sigma_mqg(pooled, n))


def chain_shortfall(ck: Dict[str, int], c_pole1: Dict[str, int], c: float) -> Tuple[float, float]:
    """(déficit en m-qg corregido por contraste, σ) de una cadena de X frente al polo medido."""
    s = (raw_mqg(ck) - raw_mqg(c_pole1)) / c
    return s, binomial_sigma_mqg(p1(ck), shots(ck)) / c


def fit_slope_origin(x: Sequence[float], y: Sequence[float], sigma: Sequence[float]) -> Tuple[float, float]:
    """Pendiente por el origen, ponderada, y χ²/gl."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), 1.0 / np.asarray(sigma, float) ** 2
    a = float((w * x * y).sum() / (w * x * x).sum())
    chi2 = float((w * (y - a * x) ** 2).sum())
    return a, chi2 / (len(x) - 1)


# ------------------------------------------------------------------ §165: neutrinos
def neutrino_summary(run: dict) -> Dict[str, object]:
    raw, fil, ex = (np.asarray(run[k], float) for k in ("raw", "filtered", "exact"))
    e = np.asarray(run["energies"], float)
    err_raw, err_fil = float(np.abs(raw - ex).mean()), float(np.abs(fil - ex).mean())
    return {
        "raw_error": err_raw, "filtered_error": err_fil, "factor": err_raw / err_fil,
        "per_flavour_raw": np.abs(raw - ex).mean(0), "per_flavour_filtered": np.abs(fil - ex).mean(0),
        "emin_raw": float(e[np.argmin(raw[:, 1])]), "emin_filtered": float(e[np.argmin(fil[:, 1])]),
        "emin_exact": float(e[np.argmin(ex[:, 1])]), "kept": float(run["kept"]),
        "kept_shots": run["kept"] * run["shots"] * len(e),
    }


# ------------------------------------------------------------------ presupuesto de créditos
def budget_summary(steps: Sequence[Tuple[str, float]], initial: float, tests: Iterable[str]) -> Dict[str, float]:
    """steps = [(nombre, créditos)]; devuelve gasto total, saldo y fracciones de pruebas y de corridas."""
    tests = set(tests)
    spent = sum(c for _, c in steps)
    t = sum(c for n, c in steps if n in tests)
    return {"spent": spent, "left": initial - spent, "tests_frac": t / initial, "runs_frac": (spent - t) / initial}


# ------------------------------------------------------------------ pre-registro frente a git
def sha256_text_blob(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prereg_hash(prereg_path: str) -> str:
    return open(prereg_path, encoding="utf-8").read().split()[0]


def blob_hash_at_commit(repo: str, commit: str, path: str) -> str:
    """sha256 del archivo `path` tal como estaba en `commit` (necesita `git` y la historia completa); '' si no se puede."""
    try:
        out = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, timeout=60)
        return sha256_text_blob(out.stdout) if out.returncode == 0 and out.stdout else ""
    except Exception:
        return ""
