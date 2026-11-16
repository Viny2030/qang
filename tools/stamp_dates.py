"""tools/stamp_dates.py — escribe bajo el título de cada capítulo (book/es/cap*.qmd y book/en/cap*.qmd) la fecha de su
última modificación sustantiva, en español e inglés. La fecha sale de git (último commit que cambió alguna línea que no
sea la propia marca), así que ejecutarlo de nuevo no mueve las fechas. Uso: python tools/stamp_dates.py
Los capítulos qjax (book/es/qjaxNN.qmd) ya traen su fecha desde tools/make_qjax_qmd.py y se omiten."""
import glob
import re
import subprocess
from datetime import date

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MARK = re.compile(r"^\*\*Actualizado:\*\*.*$")


def last_substantive(path: str) -> date:
    out = subprocess.run(["git", "log", "-1", "--format=%cs", "-G", r"^([^*]|\*[^*])", "--", path],
                         capture_output=True, text=True).stdout.strip()
    if not out:  # sin historial (archivo nuevo): hoy
        return date.today()
    y, m, d = map(int, out.split("-"))
    return date(y, m, d)


def stamp(path: str) -> bool:
    raw = open(path, encoding="utf-8", newline="").read()
    crlf = "\r\n" in raw
    L = raw.replace("\r\n", "\n").split("\n")
    L = [x for x in L if not MARK.match(x)]
    i = next(k for k, x in enumerate(L) if x.startswith("# "))
    d = last_substantive(path)
    line = f"**Actualizado:** {d.day} de {MESES[d.month - 1]} de {d.year} · **Updated:** {MONTHS[d.month - 1]} {d.day}, {d.year}"
    # quitar blancos tras el título y reinsertar: título, blanco, marca, blanco
    j = i + 1
    while j < len(L) and L[j].strip() == "":
        j += 1
    L[i + 1:j] = ["", line, ""]
    new = "\n".join(L)
    new = new.replace("\n", "\r\n") if crlf else new
    if new != raw:
        open(path, "w", encoding="utf-8", newline="").write(new)
        return True
    return False


if __name__ == "__main__":
    n = 0
    for f in sorted(glob.glob("book/es/cap[0-9][0-9].qmd") + glob.glob("book/en/cap[0-9][0-9].qmd")):
        n += stamp(f)
    print("capítulos marcados:", n)
