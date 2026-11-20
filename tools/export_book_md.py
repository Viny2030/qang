"""tools/export_book_md.py — genera `book/capNN_*.md` a partir de `book/es/qjaxNN.qmd`.

El libro se escribe en Quarto (`book/es/qjaxNN.qmd` y `book/en/qjaxNN.qmd`, que es
lo que renderiza `quarto render book/es` y `quarto render book/en`). La raíz de
`book/` guarda además una copia markdown de cada capítulo, para leerlos en
GitHub y para las rutas que cita `BOOK_OUTLINE.md`.

Esa copia era una segunda fuente editada a mano, y ya se había separado del
`.qmd`: en 18 de los 38 capítulos le faltaba contenido, y en los caps. 31 y 34
faltaban las tablas de comparación y los estados `sin_verificar` — justo la
parte que evita afirmar de más. Este script la convierte en un export
generado: la fuente es el `.qmd` y el `.md` se regenera desde él.

La transformación reproduce el formato que ya usaban los `.md` correctos:

  1. se quita el anclaje Quarto de la línea 1 (`{#sec-qjaxNN .unnumbered}`);
  2. el subtítulo inglés `*Chapter N — …*` pasa a ser encabezado de nivel 1
     `# Chapter N — …`;
  3. se suprime la línea en blanco que seguía al subtítulo en el `.qmd`;
  4. el resto del capítulo se copia literal.

El nombre del fichero (`capNN_<slug>.md`) no se inventa: se toma del `.md` que ya
existe en `book/`, emparejado por número de capítulo. Así el export nunca
renombra nada.

    python tools/export_book_md.py            # regenera los .md
    python tools/export_book_md.py --check    # sale con 1 si algún .md está
                                              # desfasado (para el job `checks`)

Sólo biblioteca estándar.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
QMD_DIR = BOOK / "es"
MD_DIR = BOOK

_ANCHOR = re.compile(r"\s*\{[^}]*\}")
_SUBTITLE = re.compile(r"^\*(.+)\*$")


def chapter_number(path: Path) -> int | None:
    """El número de capítulo de un `qjaxNN.qmd` o de un `capNN_slug.md`."""
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else None


def transform(qmd_text: str) -> str:
    """Convierte un capítulo Quarto en su copia markdown."""
    lines = qmd_text.split("\n")
    out: list[str] = []

    for i, line in enumerate(lines):
        if i == 0:
            # 1. fuera el anclaje Quarto del título.
            out.append(_ANCHOR.sub("", line))
            continue
        if i == 1:
            # 2. el subtítulo inglés pasa a encabezado de nivel 1.
            match = _SUBTITLE.match(line.strip())
            out.append(f"# {match.group(1)}" if match else line)
            continue
        if i == 2 and not line.strip():
            # 3. fuera la línea en blanco que seguía al subtítulo.
            continue
        out.append(line)

    return "\n".join(out)


def collect() -> tuple[list[tuple[Path, Path, str]], list[str]]:
    """([(qmd, md, texto generado)], avisos)."""
    pairs: list[tuple[Path, Path, str]] = []
    avisos: list[str] = []

    md_by_number: dict[int, Path] = {}
    for md in sorted(MD_DIR.glob("cap*.md")):
        number = chapter_number(md)
        if number is not None:
            md_by_number[number] = md

    for qmd in sorted(QMD_DIR.glob("qjax*.qmd")):
        number = chapter_number(qmd)
        if number is None:
            continue
        md = md_by_number.get(number)
        if md is None:
            avisos.append(f"capítulo {number}: no hay cap*.md en book/ — omitido")
            continue
        pairs.append((qmd, md, transform(qmd.read_text(encoding="utf-8"))))

    used = {md for _, md, _ in pairs}
    for number, md in sorted(md_by_number.items()):
        if md not in used:
            avisos.append(f"{md.name}: sin qjax{number}.qmd — se deja como está")

    return pairs, avisos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="no escribir nada; salir con 1 si algún .md está desfasado",
    )
    args = parser.parse_args(argv)

    pairs, avisos = collect()
    for aviso in avisos:
        print(f"  aviso: {aviso}")

    changed: list[str] = []
    for qmd, md, generated in pairs:
        current = md.read_text(encoding="utf-8") if md.exists() else ""
        if current == generated:
            continue
        changed.append(md.name)
        if not args.check:
            md.write_text(generated, encoding="utf-8")

    verb = "desfasado" if args.check else "regenerado"
    print(f"capítulos exportados: {len(pairs)}")
    print(f"{verb}s: {len(changed)}")
    for name in changed:
        print(f"  {name}")

    if args.check and changed:
        print(
            "\nFAIL: ejecuta `python tools/export_book_md.py` para regenerar "
            "book/capNN_*.md desde book/es/qjaxNN.qmd"
        )
        return 1

    print("OK" if not args.check else "OK: los .md están al día")
    return 0


if __name__ == "__main__":
    sys.exit(main())
