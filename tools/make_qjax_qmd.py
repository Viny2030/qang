"""tools/make_qjax_qmd.py — convierte book/capNN_*.md (capítulos qang-JAX) en book/es/qjaxNN.qmd para Quarto.

Uso: python tools/make_qjax_qmd.py [--force] [book_dir] [out_dir]   (por defecto book y book/es)

Reglas: (1) el primer H1 (ES) lleva ancla #sec-qjaxNN y .unnumbered; el segundo H1 (EN) pasa a cursiva.
(2) Los TÍTULOS no pueden llevar matemática: en PDF (scrreprt + xelatex) un símbolo como \\gamma en un encabezado
corriente rompe la compilación («Improper alphabetic constant»). Se sustituye por texto Unicode; el cuerpo no cambia.

AVISO — la dirección de la fuente cambió (2026-11-20). Esta herramienta va de `book/capNN_*.md` a
`book/es/qjaxNN.qmd`, pero el trabajo real ya se hace en el `.qmd`: las últimas revisiones (los caps. 31 y 34,
«Contexto 2026») se escribieron directamente en `book/es/qjaxNN.qmd` y la copia markdown de la raíz se quedó
atrás. Por eso `book/capNN_*.md` es ahora un EXPORT generado por `tools/export_book_md.py`, y esta herramienta
es el paso inverso.

Consecuencia práctica: ejecutar esto sobrescribe el `.qmd` con la copia de la raíz, y con ella se pierden las
ediciones directas y los arreglos de títulos (por ejemplo `qg_Z=tanh(x/2)`, que aquí se degradaría a `qgZ`).
Por eso, sin `--force` la herramienta sólo informa de lo que cambiaría y no escribe nada:

    python tools/make_qjax_qmd.py            # informa, no escribe
    python tools/make_qjax_qmd.py --force    # escribe (revisa antes el diff)
"""
import glob
import os
import re
import sys

MACROS = [(r"\\gamma", "γ"), (r"\\Delta", "Δ"), (r"\\theta", "θ"), (r"\\nu", "ν"), (r"\\beta", "β"),
          (r"\\cdot", "·"), (r"\\rangle", "⟩"), (r"\\langle", "⟨"), (r"\\mathbf\s*\{?(\w)\}?", r"\1"),
          (r"\\mathrm\{([^}]*)\}", r"\1"), (r"\\text\{([^}]*)\}", r"\1"), (r"\\tanh", "tanh"),
          (r"\\,", " "), (r"\\preceq", "⪯"), (r"\^2", "²"), (r"\\", "")]


def plain_heading(h: str) -> str:
    def repl(m):
        s = m.group(1)
        for a, b in MACROS:
            s = re.sub(a, b, s)
        s = re.sub(r"_\{([^}]*)\}", r"\1", s)
        s = s.replace("_", "")        # sin subíndices: evita énfasis de Markdown
        return s.replace("{", "").replace("}", "")
    return re.sub(r"\$([^$]+)\$", repl, h)


def convert(text: str, n: int) -> str:
    L = text.replace("\r\n", "\n").split("\n")
    h = [i for i, x in enumerate(L[:6]) if x.startswith("# ")]
    assert len(h) == 2, "se esperan dos H1 (ES, EN)"
    L[h[0]] = "# " + plain_heading(L[h[0]][2:]) + f" {{#sec-qjax{n:02d} .unnumbered}}"
    L[h[1]] = "*" + plain_heading(L[h[1]][2:]) + "*\n"
    out = []
    fence = False
    for x in L:
        if x.startswith("```"):
            fence = not fence
        if not fence and re.match(r"^#{2,6} ", x) and "$" in x:
            x = plain_heading(x)
        out.append(x)
    return "\n".join(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv[1:]
    book = args[0] if len(args) > 0 else "book"
    dst = args[1] if len(args) > 1 else os.path.join(book, "es")
    cambios = []
    for f in sorted(glob.glob(os.path.join(book, "cap[0-9][0-9]_*.md"))):
        n = int(os.path.basename(f)[3:5])
        with open(f, encoding="utf-8") as fh:
            q = convert(fh.read(), n)
        destino = os.path.join(dst, f"qjax{n:02d}.qmd")
        actual = ""
        if os.path.exists(destino):
            with open(destino, encoding="utf-8") as fh:
                actual = fh.read()
        if actual == q:
            continue
        cambios.append((n, destino, q))
    if not cambios:
        print("qjax*.qmd ya coinciden con book/cap*.md: no hay nada que escribir.")
        sys.exit(0)
    for n, destino, _ in cambios:
        print(f"cambiaría {destino}")
    if not force:
        print("\nNo se escribe nada sin --force: estas conversiones degradan la matemática de los títulos "
              "y pueden perder ediciones hechas directamente en el .qmd.")
        print("Revisa el diff y, si de verdad quieres regenerar, ejecuta con --force.")
        sys.exit(1)
    for n, destino, q in cambios:
        with open(destino, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(q)
        print("qjax%02d.qmd" % n)
