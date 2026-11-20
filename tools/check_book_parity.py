"""tools/check_book_parity.py — comprueba que las dos ediciones del libro van juntas.

Dos comprobaciones sobre los 38 capítulos (book/es/qjaxNN.qmd y su gemelo
book/en/qjaxNN.qmd):

  1. **Paridad de secciones**: el capítulo español y el inglés tienen las mismas
     secciones (los mismos números §N.M). Sin esto, la edición inglesa se queda
     atrás en silencio: los caps. 43 y 44 llegaron a tener §43.10/§44.10 sólo
     en español.

  2. **Referencias vivas**: cada ruta que el capítulo cita entre acentos graves
     existe de verdad en el repositorio — módulos `qang.x`, `tools/...`,
     `tests/...`, `examples/...` y `book/...`. El cap. 38 citaba
     `book/es/cap34.qmd`, una ruta del nombre antiguo que ya no existe (ahora es
     `book/es/qjax34.qmd`), en las tres copias del capítulo.

Sale con 1 si algo falla, para que el job `checks` lo pare. Sólo biblioteca
estándar.

    python tools/check_book_parity.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"

# `qang.x`, `qang.y`... son marcadores de posición en prosa ("los qang.x que
# importa"), no rutas: no se comprueban.
_PLACEHOLDER = re.compile(r"^qang\.[xyz]$")

_MODULE = re.compile(r"^qang((?:\.[A-Za-z_][A-Za-z_0-9]*)+)$")
_PATH = re.compile(
    r"^(?:tools|tests|examples|book|docs|notebooks|manuscript)/[A-Za-z0-9_./-]*"
    r"\.(?:py|qmd|md|bib|json|prereg|ipynb|txt|csv)$"
)
_SECTION = re.compile(r"^#{2,3}\s+(\d+\.\d+)")


def sections(path: Path) -> list[str]:
    """Los números de sección (N.M) de un capítulo, en orden."""
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        match = _SECTION.match(line)
        if match:
            out.append(match.group(1))
    return out


def _module_names(path: Path) -> set[str]:
    """Nombres públicos definidos en un módulo (funciones, clases, constantes)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def module_exists(dotted: str) -> bool:
    """¿Existe `qang.a.b`? Puede ser un módulo, un paquete o un atributo.

    `qang.qan` es el módulo qang/qan.py; `qang.qan.iqan_fit` es la función
    iqan_fit de ese módulo; `qang.integrations.mitiq.execute_with_filter_zne` es
    un atributo del módulo qang/integrations/mitiq.py.
    """
    parts = dotted.split(".")[1:]  # sin el "qang" inicial
    if not parts:
        return False

    for k in range(len(parts), 0, -1):
        base = ROOT.joinpath("qang", *parts[:k])
        source = base.with_suffix(".py")
        if source.exists():
            attrs = parts[k:]
            if not attrs:
                return True
            return all(a in _module_names(source) for a in attrs)
        if (base / "__init__.py").exists():
            attrs = parts[k:]
            if not attrs:
                return True
            # en un paquete, el atributo puede estar en un submódulo
            return all(
                (base / f"{a}.py").exists() or (base / a / "__init__.py").exists()
                for a in attrs
            )
    return False


def references(path: Path) -> list[str]:
    """Las rutas entre acentos graves que el capítulo cita."""
    text = path.read_text(encoding="utf-8")
    found = []
    for raw in re.findall(r"`([^`\n]+)`", text):
        ref = raw.strip()
        if not ref or " " in ref:
            continue
        if _PLACEHOLDER.match(ref):
            continue
        if _MODULE.match(ref):
            found.append(ref)
        elif _PATH.match(ref):
            found.append(ref)
    return found


def main() -> int:
    chapters = sorted(BOOK.glob("es/qjax*.qmd"))
    if not chapters:
        print("FAIL: no hay capítulos en book/es/")
        return 1

    parity: list[str] = []
    broken: dict[str, set[str]] = {}
    missing_en: list[str] = []
    total_refs = 0

    for es_path in chapters:
        number = re.search(r"(\d+)", es_path.name).group(1)
        en_path = BOOK / "en" / es_path.name
        if not en_path.exists():
            missing_en.append(en_path.name)
            continue

        es_sections = sections(es_path)
        en_sections = sections(en_path)
        if es_sections != en_sections:
            only_es = [s for s in es_sections if s not in en_sections]
            only_en = [s for s in en_sections if s not in es_sections]
            parity.append(
                f"{es_path.name}: solo ES {only_es or '-'} | solo EN {only_en or '-'}"
            )

        for ref in references(es_path) + references(en_path):
            total_refs += 1
            if _MODULE.match(ref):
                if not module_exists(ref):
                    broken.setdefault(es_path.name, set()).add(ref)
            elif not (ROOT / ref).exists():
                broken.setdefault(es_path.name, set()).add(ref)

    print(f"capítulos comprobados: {len(chapters)}")
    print(f"referencias comprobadas: {total_refs}")

    print()
    if missing_en:
        print(f"FAIL ({len(missing_en)}): capítulos sin traducción")
        for name in missing_en:
            print(f"        {name}")
    else:
        print("OK   paridad de secciones ES/EN")

    if parity:
        print(f"FAIL ({len(parity)}): secciones desalineadas entre ES y EN")
        for line in parity:
            print(f"        {line}")
    elif not missing_en:
        print("OK   todas las secciones existen en las dos ediciones")

    if broken:
        total = sum(len(v) for v in broken.values())
        print(f"FAIL ({total}): referencias rotas")
        for name, refs in sorted(broken.items()):
            print(f"        {name}")
            for ref in sorted(refs):
                print(f"            {ref}")
    else:
        print("OK   todas las rutas citadas existen")

    failed = bool(missing_en or parity or broken)
    print()
    print("FAIL" if failed else "OK: el libro está sincronizado")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
