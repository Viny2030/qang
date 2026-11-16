"""tools/make_qjax_qmd.py — convierte book/capNN_*.md (capítulos qang-JAX) en book/es/qjaxNN.qmd para Quarto.

Uso: python tools/make_qjax_qmd.py [book_dir] [out_dir]   (por defecto book y book/es)

Reglas: (1) el primer H1 (ES) lleva ancla #sec-qjaxNN y .unnumbered; el segundo H1 (EN) pasa a cursiva.
(2) Los TÍTULOS no pueden llevar matemática: en PDF (scrreprt + xelatex) un símbolo como \\gamma en un encabezado
corriente rompe la compilación («Improper alphabetic constant»). Se sustituye por texto Unicode; el cuerpo no cambia.
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
    book = sys.argv[1] if len(sys.argv) > 1 else "book"
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(book, "es")
    for f in sorted(glob.glob(os.path.join(book, "cap[0-9][0-9]_*.md"))):
        n = int(os.path.basename(f)[3:5])
        with open(f, encoding="utf-8") as fh:
            q = convert(fh.read(), n)
        with open(os.path.join(dst, f"qjax{n:02d}.qmd"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(q)
        print("qjax%02d.qmd" % n)
