"""tools/check_baselines.py — audita la tabla de dominios propuestos del Apéndice H.

Comprueba que los dominios del cuadro de investigación (razones, finanzas,
cognición, navegación, espacio, biofinanzas) estén declarados con dataset,
baseline sin qang, estatus y evidencia, y que las ediciones ES/EN concuerden.
Ninguna fila puede afirmar una ganancia medida sin dataset, baseline y
prerregistro citado. Sólo biblioteca estándar.

    python tools/check_baselines.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    "es": ROOT / "book" / "es" / "apendice_h.qmd",
    "en": ROOT / "book" / "en" / "appendix_h.qmd",
}
HEADERS = {
    "es": "| dominio | dataset | baseline sin qang | estatus | evidencia |",
    "en": "| domain | dataset | without-qang baseline | status | evidence |",
}
STATUS = {
    "es": {
        "no ganancia": "no gain",
        "no adoptada": "not adopted",
        "sin verificar": "unverified",
        "medida": "measured gain",
    },
    "en": {
        "no gain": "no gain",
        "not adopted": "not adopted",
        "unverified": "unverified",
        "measured gain": "measured gain",
    },
}
EXPECTED = [
    ("razones / proporciones", "proportions", "no gain"),
    ("finanzas", "finance", "not adopted"),
    ("cognición / qagent", "cognition / qagent", "not adopted"),
    ("navegación / SLAM", "navigation / SLAM", "unverified"),
    ("espacio", "space", "not adopted"),
    ("biofinanzas", "biofinance", "not adopted"),
]
_PATH = re.compile(r"`((?:tools|tests|examples|book)/[^`]+)`")


def table(lang: str) -> list[list[str]]:
    """Lee la tabla de dominios de una edición y devuelve sus filas."""
    lines = SOURCES[lang].read_text(encoding="utf-8").split("\n")
    start = lines.index(HEADERS[lang]) + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def check() -> list[str]:
    """Devuelve los problemas encontrados en las dos ediciones."""
    errors = []
    parsed = {}
    for lang in ("es", "en"):
        rows = table(lang)
        parsed[lang] = rows
        if len(rows) != len(EXPECTED):
            errors.append(f"{lang}: {len(rows)} filas, se esperaban {len(EXPECTED)}")
        for row in rows:
            if len(row) != 5:
                errors.append(f"{lang}: fila con {len(row)} celdas: {row}")
                continue
            domain, dataset, baseline, status, evidence = row
            if status not in STATUS[lang]:
                errors.append(f"{lang}/{domain}: estatus desconocido {status!r}")
                continue
            if not evidence:
                errors.append(f"{lang}/{domain}: sin evidencia")
            measured = STATUS[lang]["medida" if lang == "es" else "measured gain"]
            if status == measured:
                for name, value in (("dataset", dataset), ("baseline", baseline)):
                    if value == "—":
                        errors.append(f"{lang}/{domain}: ganancia medida sin {name}")
                if ".prereg" not in evidence:
                    errors.append(f"{lang}/{domain}: ganancia medida sin prerregistro")
            for cell in (dataset, baseline, evidence):
                for path in _PATH.findall(cell):
                    if ":" in path:
                        path = path.split(":", 1)[0]
                    if not (ROOT / path).exists():
                        errors.append(f"{lang}/{domain}: ruta inexistente {path}")

    if len(parsed["es"]) == len(parsed["en"]):
        for i, (row_es, row_en) in enumerate(zip(parsed["es"], parsed["en"])):
            if STATUS["es"].get(row_es[3]) != STATUS["en"].get(row_en[3]):
                errors.append(f"fila {i + 1}: estatus ES/EN distinto")
            if len(row_es) == 5 and len(row_en) == 5:
                if (row_es[1] == "—") != (row_en[1] == "—"):
                    errors.append(f"fila {i + 1}: dataset ES/EN distinto")
                if (row_es[2] == "—") != (row_en[2] == "—"):
                    errors.append(f"fila {i + 1}: baseline ES/EN distinto")

    domains = {
        lang: [row[0] for row in rows if len(row) == 5]
        for lang, rows in parsed.items()
    }
    for i, (name_es, name_en, _) in enumerate(EXPECTED):
        for lang, name in (("es", name_es), ("en", name_en)):
            if i >= len(domains[lang]):
                continue
            if domains[lang][i] != name:
                errors.append(f"{lang}: fila {i + 1} es {domains[lang][i]!r}, se esperaba {name!r}")
    return errors


def main() -> int:
    """Punto de entrada del verificador."""
    errors = check()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("check_baselines: OK (6 dominios, 0 ganancias medidas, ES/EN de acuerdo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
