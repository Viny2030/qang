"""qang.qarch — Cap. 36: auditoría de la arquitectura de software. ¿De qué paquetes depende cada módulo de `qang` al
importarlo? ¿Qué extras instala el CI (`pip install -e ".[all]"`)? ¿Qué pruebas importan un módulo cuyo paquete el CI no
instala y no se protegen con `pytest.importorskip` (error de recolección, código de salida 2)? Sólo biblioteca estándar
(`ast`, `re`). Y una medida en m-qg: el error de leer un resultado con la convención de qubits equivocada (usa
`qang.interop` sólo dentro de `convention_error_mqg`). m-qg = 1000·qg_Z.
"""
from __future__ import annotations

import ast
import glob
import os
import re
import sys
from typing import Dict, Iterable, List, Set, Tuple

_FALLBACK_STDLIB = {
    "abc", "ast", "cmath", "collections", "dataclasses", "functools", "glob", "hashlib", "importlib", "itertools", "json",
    "math", "os", "random", "re", "string", "sys", "time", "typing", "warnings", "__future__", "fractions", "statistics",
    "heapq", "bisect", "copy", "enum", "io", "pathlib", "decimal", "contextlib", "operator", "textwrap", "datetime",
    "subprocess", "tempfile", "shutil", "argparse", "inspect", "types", "numbers", "struct", "itertools", "unittest",
}
STDLIB: Set[str] = set(getattr(sys, "stdlib_module_names", ())) | _FALLBACK_STDLIB

# nombre en PyPI (minúsculas, guion bajo) -> paquete que se importa
_PIP_TO_IMPORT = {
    "scikit_learn": "sklearn", "amazon_braket_sdk": "braket", "iqm_client": "iqm", "qiskit_aer": "qiskit_aer",
    "qiskit_ibm_runtime": "qiskit_ibm_runtime", "qiskit_ionq": "qiskit_ionq", "qiskit_addon_cutting": "qiskit_addon_cutting",
    "qiskit_machine_learning": "qiskit_machine_learning", "qiskit_experiments": "qiskit_experiments",
    "qiskit_dynamics": "qiskit_dynamics", "openfermionpyscf": "openfermionpyscf", "cudaq": "cudaq",
}


# ------------------------------------------------------------------ imports de un módulo
def _imports_in(nodes: Iterable[ast.stmt], top_only: bool) -> List[Tuple[str, List[str]]]:
    """[(módulo, [nombres])] de las sentencias import. Con top_only no entra en def/class (imports diferidos)."""
    out: List[Tuple[str, List[str]]] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            out += [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append(("." * node.level + (node.module or ""), [a.name for a in node.names]))
        elif top_only and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        else:
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(node, field, None)
                if isinstance(sub, list):
                    out += _imports_in(sub, top_only)
            for h in getattr(node, "handlers", []) or []:
                out += _imports_in(h.body, top_only)
    return out


def module_imports(path: str, top_only: bool = True) -> List[Tuple[str, List[str]]]:
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return _imports_in(tree.body, top_only)


def third_party_roots(path: str, top_only: bool = True) -> Set[str]:
    """Paquetes de terceros importados (raíz del nombre), sin biblioteca estándar ni `qang`."""
    roots = set()
    for mod, _ in module_imports(path, top_only):
        if mod.startswith("."):
            continue
        r = mod.split(".")[0]
        if r and r not in STDLIB and r != "qang":
            roots.add(r)
    return roots


def qang_neighbors(path: str, top_only: bool = True) -> Set[str]:
    """Módulos de `qang` que este importa (`from .x import`, `from qang.x import`, `from . import x`, `import qang.x`)."""
    out = set()
    for mod, names in module_imports(path, top_only):
        if mod.startswith("."):
            base = mod.lstrip(".")
            if base:
                out.add(base.split(".")[0])
            else:
                out |= set(names)
        elif mod == "qang" or mod.startswith("qang."):
            parts = mod.split(".")
            if len(parts) > 1:
                out.add(parts[1])
            else:
                out |= set(names)
    return out


# ------------------------------------------------------------------ dependencias transitivas
def package_files(qang_dir: str) -> Dict[str, str]:
    return {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(qang_dir, "*.py"))}


def requires(qang_dir: str) -> Dict[str, Set[str]]:
    """Paquetes de terceros que exige importar cada módulo `qang.x` (imports de nivel de módulo, transitivos).
    Importar `qang.x` importa antes `qang/__init__.py`, así que sus requisitos entran en todos."""
    files = package_files(qang_dir)
    direct = {m: third_party_roots(f) for m, f in files.items()}
    nbrs = {m: qang_neighbors(f) & set(files) for m, f in files.items()}
    cache: Dict[str, Set[str]] = {}

    def rec(m: str, seen: Tuple[str, ...]) -> Set[str]:
        if m in cache:
            return cache[m]
        r = set(direct[m])
        for n in nbrs[m]:
            if n not in seen and n != m:
                r |= rec(n, seen + (m,))
        cache[m] = r
        return r

    out = {m: rec(m, ()) for m in files}
    init = out.get("__init__", set())
    return {m: (r | init if m != "__init__" else r) for m, r in out.items()}


def group_modules(req: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Módulos por grupo de dependencia: 'core' (sólo NumPy y biblioteca estándar) o la lista de paquetes extra."""
    groups: Dict[str, List[str]] = {}
    for m, r in sorted(req.items()):
        extra = sorted(r - {"numpy"})
        groups.setdefault("core" if not extra else "+".join(extra), []).append(m)
    return groups


# ------------------------------------------------------------------ pyproject
def _list_literal(text: str, start: int) -> List[str]:
    """Cadenas de la lista TOML que abre en text[start] == '[' (respeta comillas y corchetes dentro de ellas)."""
    items, depth, quote, buf = [], 0, None, ""
    for ch in text[start:]:
        if quote:
            if ch == quote:
                items.append(buf)
                buf, quote = "", None
            else:
                buf += ch
        elif ch in "\"'":
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
        elif ch == "#":
            continue
    return items


def parse_pyproject(text: str) -> Dict[str, List[str]]:
    """{'dependencies': [...], extra: [...]} sin dependencias externas (explorador de listas sobre el TOML de qang)."""
    text = text.replace("\r\n", "\n")
    out: Dict[str, List[str]] = {}
    m = re.search(r"^dependencies\s*=\s*\[", text, re.M)
    out["dependencies"] = _list_literal(text, m.end() - 1) if m else []
    sec = re.search(r"\[project\.optional-dependencies\]\n(.*?)(?=^\[)", text, re.M | re.S)
    if sec:
        base = sec.start(1)
        for em in re.finditer(r"^(\w[\w-]*)\s*=\s*\[", sec.group(1), re.M):
            out[em.group(1)] = _list_literal(text, base + em.end() - 1)
    return out


def requirement_root(req: str) -> str:
    """'jax[cpu]>=0.4.30,<0.9; python_version >= "3.10"' -> 'jax'; 'scikit-learn>=1.2' -> 'sklearn'."""
    name = re.split(r"[\[<>=!~;\s]", req.strip(), maxsplit=1)[0].lower().replace("-", "_")
    return _PIP_TO_IMPORT.get(name, name)


def installed_by(extras: Dict[str, List[str]], extra: str = "all") -> Set[str]:
    """Paquetes que instala `pip install -e ".[extra]"` (incluye las dependencias base)."""
    return {requirement_root(r) for r in extras.get("dependencies", []) + extras.get(extra, [])}


def check_extras(extras: Dict[str, List[str]]) -> Dict[str, bool]:
    return {
        "core_is_numpy_only": [requirement_root(r) for r in extras["dependencies"]] == ["numpy"],
        "iqm_isolated_from_all": "iqm" not in installed_by(extras, "all") and "iqm_client" not in installed_by(extras, "all"),
        "jax_below_0_9": any("jax" in r and "<0.9" in r.replace(" ", "") for r in extras.get("jax", [])),
    }


# ------------------------------------------------------------------ pruebas que fallarían al recolectar
def guarded_roots(path: str) -> Set[str]:
    """Paquetes protegidos con pytest.importorskip("x") en una prueba."""
    with open(path, encoding="utf-8") as fh:
        return set(re.findall(r"""importorskip\(\s*["']([\w\.]+)["']""", fh.read()))


def local_modules(dirs: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.py")):
            out.setdefault(os.path.basename(f)[:-3], f)
    return out


def test_requirements(path: str, req: Dict[str, Set[str]], local: Dict[str, str]) -> Set[str]:
    """Paquetes de terceros que exige importar una prueba: los suyos, los de los `qang.x` que importa y los de los
    scripts locales (examples/, tests/) que importa, de forma transitiva."""
    need: Set[str] = set(req.get("__init__", set()))
    seen: Set[str] = set()
    stack = [path]
    while stack:
        f = stack.pop()
        for m in qang_neighbors(f):
            need |= req.get(m, set())
        for r in third_party_roots(f):
            if r in local:
                if local[r] not in seen:
                    seen.add(local[r])
                    stack.append(local[r])
            else:
                need.add(r)
    return need


def unguarded_tests(tests_dir: str, qang_dir: str, available: Set[str], local_dirs: Iterable[str] = ()) -> List[Tuple[str, List[str]]]:
    """Pruebas que importan (a nivel de módulo) algo que exige un paquete que el CI no instala (`available`) y no lo
    protegen con importorskip: error de recolección → pytest sale con código 2."""
    req = requires(qang_dir)
    local = local_modules(list(local_dirs) + [tests_dir])
    bad = []
    for f in sorted(glob.glob(os.path.join(tests_dir, "test_*.py"))):
        need = {r for r in test_requirements(f, req, local) if r not in available and r != "pytest"}
        guarded = {g.split(".")[0] for g in guarded_roots(f)}
        missing = sorted(need - guarded)
        if missing:
            bad.append((os.path.basename(f), missing))
    return bad


# ------------------------------------------------------------------ convención de qubits
def convention_error_mqg(counts, sdk_true: str, sdk_wrong: str, n_qubits: int = None) -> float:
    """Desvío máximo en m-qg del perfil qg_Z por qubit al leer `counts` como `sdk_wrong` en vez de `sdk_true`."""
    from .interop import canonical_counts, qg_z_profile

    a = qg_z_profile(canonical_counts(counts, sdk=sdk_true, n_qubits=n_qubits))
    b = qg_z_profile(canonical_counts(counts, sdk=sdk_wrong, n_qubits=n_qubits))
    return 1000.0 * max(abs(x[0] - y[0]) for x, y in zip(a, b))
