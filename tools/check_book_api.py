"""tools/check_book_api.py — sincroniza el manuscrito con la API pública: extrae los bloques ```python de book/ y comprueba, con ast e
inspect, que cada llamada a `qang` (alias de módulo, `from qang.x import f`) apunte a un nombre que existe y acepte esos argumentos
(posicionales y por nombre). No ejecuta los bloques. Los módulos de `qang` que no se pueden importar en este entorno (SDK ausente) se
listan aparte: correr el script también en el entorno con los extras.

    python tools/check_book_api.py [raíz_del_repositorio]        (código de salida 1 si hay desajustes)
"""
from __future__ import annotations

import ast
import glob
import importlib
import importlib.util
import inspect
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

FENCE = re.compile(r"^```\s*(?:\{python[^}]*\}|python[^\n]*)\s*$")


def code_blocks(text: str) -> List[Tuple[int, str]]:
    """[(línea inicial, código)] de los bloques ```python / ```{python}."""
    out, cur, start = [], None, 0
    for i, line in enumerate(text.split("\n"), 1):
        if cur is None and FENCE.match(line.strip()):
            cur, start = [], i + 1
        elif cur is not None and line.strip().startswith("```"):
            out.append((start, "\n".join(cur)))
            cur = None
        elif cur is not None:
            cur.append(line)
    return out


def _clean(code: str) -> str:
    return "\n".join("pass" if l.lstrip().startswith(("!", "%")) else l for l in code.split("\n"))


def _aliases(tree: ast.AST) -> Dict[str, Tuple[str, Optional[str]]]:
    """nombre local -> (módulo qang, atributo o None)."""
    al: Dict[str, Tuple[str, Optional[str]]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "qang" or a.name.startswith("qang."):
                    al[a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], None)
        elif isinstance(n, ast.ImportFrom) and n.module and (n.module == "qang" or n.module.startswith("qang.")):
            for a in n.names:
                al[a.asname or a.name] = (n.module, a.name)
    return al


def _chain(node: ast.AST) -> Optional[List[str]]:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


def _resolve(chain: List[str], al: Dict[str, Tuple[str, Optional[str]]], unimportable: set):
    """Devuelve (objeto, descripción) o ('MISSING', texto) o None si no es de qang."""
    root = chain[0]
    if root not in al:
        return None
    mod, attr = al[root]
    try:
        obj = importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        unimportable.add("%s (%s)" % (mod, type(exc).__name__))
        return None
    path = ([attr] if attr else []) + chain[1:]
    for p in path:
        if hasattr(obj, p):
            obj = getattr(obj, p)
            continue
        full = getattr(obj, "__name__", mod) + "." + p
        try:  # submódulo no importado todavía
            obj = importlib.import_module(full)
        except Exception as exc:  # noqa: BLE001
            try:
                exists = importlib.util.find_spec(full) is not None
            except Exception:  # noqa: BLE001
                exists = False
            if exists:  # el archivo está, pero falta su SDK: no verificable aquí
                unimportable.add("%s (%s)" % (full, type(exc).__name__))
                return None
            return "MISSING", full
    return obj, ".".join([mod] + path)


def check_call(node: ast.Call, al, unimportable: set) -> Optional[str]:
    chain = _chain(node.func)
    if chain is None:
        return None
    r = _resolve(chain, al, unimportable)
    if r is None:
        return None
    obj, desc = r
    if obj == "MISSING":
        return "no existe: %s" % desc
    if inspect.ismodule(obj) or not callable(obj):
        return None
    if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
        return None
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    try:
        sig.bind(*[object()] * len(node.args), **{k.arg: object() for k in node.keywords})
    except TypeError as exc:
        # los argumentos que faltan no son un desajuste de nombre si el bloque es un fragmento; sólo nombres/exceso
        msg = str(exc)
        if msg.startswith("missing a required argument"):
            return None
        return "%s%s -> %s" % (desc, sig, msg)
    return None


def scan(root: str) -> Tuple[List[str], List[str], int, int]:
    sys.path.insert(0, root)
    files = sorted(glob.glob(os.path.join(root, "book", "es", "*.qmd")) + glob.glob(os.path.join(root, "book", "en", "*.qmd"))
                   + glob.glob(os.path.join(root, "book", "*.md")))
    problems: List[str] = []
    unimportable: set = set()
    nblocks = ncalls = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        for start, code in code_blocks(text):
            nblocks += 1
            try:
                tree = ast.parse(_clean(code))
            except SyntaxError:
                continue
            al = _aliases(tree)
            if not al:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    ncalls += 1
                    msg = check_call(node, al, unimportable)
                    if msg:
                        problems.append("%s:%d: %s" % (os.path.relpath(f, root), start + node.lineno - 1, msg))
    return problems, sorted(unimportable), nblocks, ncalls


def main() -> int:
    root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
    problems, unimp, nb, nc = scan(root)
    print("bloques python: %d; llamadas a qang revisadas: %d; desajustes: %d" % (nb, nc, len(problems)))
    for p in problems:
        print("  ✗", p)
    if unimp:
        print("no verificables en este entorno (módulo sin importar): " + "; ".join(unimp))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
