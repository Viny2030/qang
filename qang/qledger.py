"""qang.qledger — Cap. 32: libro de cuentas de las submedidas de los capítulos qang-JAX. Lee las tablas «Apéndice H» de
book/capNN_*.md, cuenta los estatus y comprueba que cada función citada existe en `qang/` y que cada capítulo tiene su
verificador. Sólo biblioteca estándar. Estatus válidos: los de THEORY_CORE.md / qang.measures.
"""
from __future__ import annotations

import glob
import os
import re
from collections import Counter
from typing import Dict, List

STATUSES = ("exact", "measured gain", "tested on hardware", "mixed", "no gain", "against", "unverified", "not adopted")


def parse_appendix_h(text: str, cap: int) -> List[Dict]:
    """Filas de la tabla que sigue al encabezado «Apéndice H» (id, familia, fórmula, función, estatus)."""
    rows, inside = [], False
    for line in text.split("\n"):
        if re.match(r"^#{2,3} .*Apéndice H", line):
            inside = True
            continue
        if inside and line.startswith("#"):
            break
        if inside and line.startswith("| `"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) == 5:
                rows.append({"cap": cap, "id": c[0].strip("`"), "family": c[1], "formula": c[2], "functions": c[3], "status": c[4]})
    return rows


def statuses_of(cell: str) -> List[str]:
    return [t for t in re.findall(r"`([^`]+)`", cell)]


def function_names(cell: str) -> List[str]:
    out = []
    for t in re.findall(r"`([^`]+)`", cell):
        t = t.split(" §")[0].strip()
        if t.endswith(".py"):
            out.append(os.path.basename(t))
            continue
        t = t.split("(")[0].split(".")[-1].strip()
        if t:
            out.append(t)
    return out


def defined_names(qang_dir: str) -> set:
    names = set()
    for f in glob.glob(os.path.join(qang_dir, "*.py")):
        for m in re.finditer(r"^\s*(?:def|class)\s+(\w+)|^(\w+)\s*=", open(f, encoding="utf-8").read(), re.M):
            names.add(m.group(1) or m.group(2))
    return names


def load_ledger(book_dir: str, first: int = 7, last: int = 31) -> List[Dict]:
    rows: List[Dict] = []
    for f in sorted(glob.glob(os.path.join(book_dir, "cap[0-9][0-9]_*.md"))):
        n = int(os.path.basename(f)[3:5])
        if first <= n <= last:
            rows += parse_appendix_h(open(f, encoding="utf-8").read(), n)
    return rows


def primary_status(cell: str) -> str:
    """Primer estatus de la celda (el del resultado principal); el resto son matices («; hardware `unverified`»)."""
    for t in statuses_of(cell):
        if t in STATUSES:
            return t
    return "?"


def summarize(rows: List[Dict]) -> Counter:
    return Counter(primary_status(r["status"]) for r in rows)


def unresolved(rows: List[Dict], qang_dir: str, tools_dir: str) -> List[tuple]:
    """Filas cuya función (o verificador `x.py`) no existe."""
    defs = defined_names(qang_dir)
    bad = []
    for r in rows:
        for n in function_names(r["functions"]):
            ok = os.path.exists(os.path.join(tools_dir, n)) if n.endswith(".py") else n in defs
            if not ok:
                bad.append((r["cap"], r["id"], n))
    return bad


def caveats(rows: List[Dict]) -> List[Dict]:
    """Filas con más de un estatus en la celda (matices sobre el resultado principal)."""
    return [r for r in rows if len([t for t in statuses_of(r["status"]) if t in STATUSES]) > 1]


def clean_exact(rows: List[Dict]) -> List[Dict]:
    """Filas `exact` sin ningún matiz."""
    cv = {(r["cap"], r["id"]) for r in caveats(rows)}
    return [r for r in rows if primary_status(r["status"]) == "exact" and (r["cap"], r["id"]) not in cv]
