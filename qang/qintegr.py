"""qang.qintegr — Cap. 39: auditoría de `qang.integrations` (qang como módulo dentro de otras librerías) y una comprobación cerrada
del filtro bajo T1 que no necesita ningún SDK. Sólo biblioteca estándar y NumPy (reusa `qang.qarch` y `qang.qlimits`).
m-qg = 1000·qg_Z.
"""
from __future__ import annotations

import ast
import glob
import math
import os
from typing import Dict, List, Set, Tuple

import numpy as np

from .qarch import _imports_in, parse_pyproject, requirement_root, third_party_roots

# submódulo de qang.integrations -> (extra de pyproject, paquete que se importa)
INTEGRATIONS: Dict[str, Tuple[str, str]] = {
    "mitiq": ("mitiq", "mitiq"),
    "qiskit_experiments": ("experiments", "qiskit_experiments"),
    "qiskit_dynamics": ("dynamics", "qiskit_dynamics"),
    "qutip": ("qutip", "qutip"),
    "stim": ("stim", "stim"),
    "sklearn": ("sklearn", "sklearn"),
    "jax": ("jax", "jax"),
    "torch": ("torch", "torch"),
    "qibo": ("qibo", "qibo"),
    "pytket": ("pytket", "pytket"),
    "braket": ("braket", "braket"),
    "cudaq": ("cudaq", "cudaq"),
}


def integration_modules(qang_dir: str) -> List[str]:
    d = os.path.join(qang_dir, "integrations")
    return sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(d, "*.py")) if not os.path.basename(f).startswith("_"))


def import_levels(path: str) -> Tuple[Set[str], Set[str]]:
    """(paquetes de terceros, salvo NumPy, importados a nivel de módulo, paquetes importados sólo de forma diferida)."""
    top = third_party_roots(path, top_only=True) - {"numpy"}
    allr = third_party_roots(path, top_only=False) - {"numpy"}
    return top, allr - top


def lazy_report(qang_dir: str) -> Dict[str, Tuple[Set[str], Set[str]]]:
    d = os.path.join(qang_dir, "integrations")
    return {m: import_levels(os.path.join(d, m + ".py")) for m in integration_modules(qang_dir)}


def extra_report(extras: Dict[str, List[str]]) -> Dict[str, bool]:
    """¿Existe el extra de cada integración y exige el paquete que el módulo importa?"""
    out = {}
    for mod, (extra, root) in INTEGRATIONS.items():
        reqs = {requirement_root(r) for r in extras.get(extra, [])}
        out[mod] = extra in extras and root in reqs
    return out


def bundle_libraries(extras: Dict[str, List[str]]) -> Set[str]:
    return {requirement_root(r) for r in extras.get("integrations", [])}


def guard_report(test_path: str) -> Dict[str, List[str]]:
    """Por función de prueba: integraciones de qang que importa y paquetes que protege con importorskip. Devuelve las
    funciones cuya integración usa un paquete sin protección (lista vacía = todo guardado)."""
    with open(test_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    bad: Dict[str, List[str]] = {}

    def _skips(fn: ast.AST) -> Set[str]:
        return {str(n.args[0].value).split(".")[0] for n in ast.walk(fn)
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "importorskip" and n.args
                and isinstance(n.args[0], ast.Constant)}

    helpers = {n.name: _skips(n) for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("test_")}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]:
        used, guarded = set(), set()
        for mod, names in _imports_in(fn.body, top_only=False):
            if mod == "qang.integrations":
                used |= set(names) & set(INTEGRATIONS)
            elif mod.startswith("qang.integrations."):
                used.add(mod.split(".")[2])
        used &= set(INTEGRATIONS)
        guarded |= _skips(fn)
        for node in ast.walk(fn):  # llamada a un ayudante del módulo que protege con importorskip
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in helpers:
                guarded |= helpers[node.func.id]
        miss = sorted(INTEGRATIONS[u][1] for u in used if INTEGRATIONS[u][1] not in guarded)
        if miss:
            bad[fn.name] = miss
    return bad


def modules_with_tests(test_path: str) -> Set[str]:
    with open(test_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    used: Set[str] = set()
    for mod, names in _imports_in(tree.body, top_only=False):
        if mod == "qang.integrations":
            used |= set(names) & set(INTEGRATIONS)
        elif mod.startswith("qang.integrations."):
            used.add(mod.split(".")[2])
    return used & set(INTEGRATIONS)


def missing_library_message(extra: str) -> str:
    """Mensaje de error que da `qang.integrations.require` cuando falta la librería."""
    from .integrations import require

    try:
        require("a_package_that_does_not_exist_xyz", extra)
    except ImportError as exc:
        return str(exc)
    return ""


# ------------------------------------------------------------------ filtro bajo T1 sin SDK (estado W)
def w_state_t1_readings(t_over_T1: float, n: int = 3) -> Dict[str, float]:
    """Estado W de n qubits con amortiguamiento parejo γ = 1 − e^{−t/T1}: lectura cruda y filtrada (peso 1) del qubit 1, en
    m-qg, y fracción conservada. Cálculo exacto sobre las poblaciones de Z (las coherencias no cambian esas poblaciones)."""
    from .qlimits import bit, damp_distribution, popcount

    gamma = 1.0 - math.exp(-t_over_T1)
    p = np.zeros(2 ** n)
    for q in range(n):
        p[1 << (n - 1 - q)] = 1.0 / n
    pg = damp_distribution(p, n, gamma)
    idx = np.arange(2 ** n)
    raw = 1.0 - 2.0 * pg[bit(idx, 1, n) == 1].sum()
    w1 = popcount(idx) == 1
    kept = float(pg[w1].sum())
    filt = 1.0 - 2.0 * pg[w1 & (bit(idx, 1, n) == 1)].sum() / kept
    return {"raw_mqg": 1000.0 * raw, "filtered_mqg": 1000.0 * filt, "kept": kept, "ideal_mqg": 1000.0 * (1.0 - 2.0 / n)}


def require_guard_report(qang_dir: str) -> Dict[str, bool]:
    """Por submódulo: si importa un SDK a nivel de módulo, ¿llama antes a `require(<paquete>, <extra>)`? Así `import qang` no
    arrastra el SDK y, al importar el submódulo sin él, el error dice qué extra instalar."""
    d = os.path.join(qang_dir, "integrations")
    out: Dict[str, bool] = {}
    for m in integration_modules(qang_dir):
        path = os.path.join(d, m + ".py")
        top, _ = import_levels(path)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        def _is_req(n):
            v = n.value if isinstance(n, (ast.Expr, ast.Assign)) else None
            return isinstance(v, ast.Call) and getattr(v.func, "id", "") == "require"

        req_line = min((n.lineno for n in tree.body if _is_req(n)), default=None)
        first = min((n.lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                     and any(r in (getattr(n, "module", None) or "").split(".")[0:1] + [a.name.split(".")[0] for a in n.names]
                             for r in top)), default=None)
        out[m] = (not top) or (req_line is not None and first is not None and req_line < first)
    return out
