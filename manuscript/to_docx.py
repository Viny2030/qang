"""LaTeX -> Word for the qang notes: pandoc with citeproc, then fonts and captions."""
import re, subprocess, sys
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

src, out, fig_png = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(src, encoding="utf-8").read()
t = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]*\}", r"\\includegraphics[width=0.9\\textwidth]{" + fig_png + "}", t)
t = t.replace(">{\\raggedright\\arraybackslash}p{", "p{").replace("\\rm ", "\\mathrm ")
t = t.replace("\\newcommand{\\qZ}{qg_Z}", "").replace("\\qZ", "qg_Z")
tmp = src.replace(".tex", "_pandoc.tex")
open(tmp, "w", encoding="utf-8").write(t)
subprocess.run(["pandoc", tmp, "-o", out, "--citeproc", "--bibliography", sys.argv[4],
                "--resource-path", ".", "-M", "link-citations=true"], check=True)
d = Document(out)
for st in d.styles:
    try:
        st.font.name = "Times New Roman"
    except Exception:
        pass
for p in d.paragraphs:
    for r in p.runs:
        if r.font.name is None or "Courier" not in (r.font.name or ""):
            if p.style.name not in ("Source Code",):
                r.font.name = r.font.name if (r.font.name and "Mono" in r.font.name) else "Times New Roman"
    if p.style.name in ("Title",):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p.style.name in ("Image Caption", "Table Caption", "Captioned Figure"):
        for r in p.runs:
            r.font.size = Pt(9)
d.save(out)
print("wrote", out)
