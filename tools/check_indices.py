"""tools/check_indices.py — comprueba que los índices maestros siguen al libro.

Los índices `book/INDICE_MAESTRO_{ES,EN}.md` deben cubrir todos los capítulos
que `_quarto.yml` publica y declarar las partes XIV y XV. Este checker evita
que un bloque nuevo quede fuera del índice maestro, como pasó con los caps.
36–50. Sólo biblioteca estándar.

    python tools/check_indices.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHAPTER = re.compile(r"^\s+- (?:cap|qjax)(\d+)\.qmd$", re.MULTILINE)
HEADING = re.compile(r"^###\s+(?:Capítulo|Chapter)\s+(\d+)\.", re.MULTILINE)
PATH = re.compile(r"`((?:tools|tests|examples|book)/[^`]+\.(?:py|qmd|md))`")


def quarto_chapters(lang: str) -> set[int]:
    """Números de capítulo publicados en una edición."""
    text = (ROOT / "book" / lang / "_quarto.yml").read_text(encoding="utf-8")
    return {int(number) for number in CHAPTER.findall(text)}


def index_chapters(path: Path) -> set[int]:
    """Números de capítulo documentados en un índice maestro."""
    return {int(number) for number in HEADING.findall(path.read_text(encoding="utf-8"))}


def check() -> list[str]:
    """Devuelve las discrepancias entre índices y libro."""
    errors = []
    published = {}
    for lang in ("es", "en"):
        published[lang] = quarto_chapters(lang)
        path = ROOT / "book" / ("INDICE_MAESTRO_ES.md" if lang == "es" else "INDICE_MAESTRO_EN.md")
        documented = index_chapters(path)
        text = path.read_text(encoding="utf-8")
        missing = sorted(published[lang] - documented)
        extra = sorted(documented - published[lang])
        if missing:
            errors.append(f"{lang}: capítulos publicados sin índice: {missing}")
        if extra:
            errors.append(f"{lang}: capítulos en el índice que no publica _quarto.yml: {extra}")
        marker = "PARTE XIV" if lang == "es" else "PART XIV"
        marker_xv = "PARTE XV" if lang == "es" else "PART XV"
        if marker not in text:
            errors.append(f"{lang}: falta la parte XIV")
        if marker_xv not in text:
            errors.append(f"{lang}: falta la parte XV")
        for reference in PATH.findall(text):
            if "*" in reference:
                continue
            if not (ROOT / reference).exists():
                errors.append(f"{lang}: ruta inexistente {reference}")
    if published["es"] != published["en"]:
        errors.append("las dos ediciones publican capítulos distintos")
    return errors


def main() -> int:
    """Punto de entrada del verificador."""
    errors = check()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    count = len(quarto_chapters("es"))
    print(f"check_indices: OK ({count}/{count} capítulos por edición, XIV y XV presentes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
