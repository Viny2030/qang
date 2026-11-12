"""Write appendix H of the book (the catalogue of qang sub-measures) in Spanish and English from qang.measures.

The tables come from ``qang.measures.CATALOG``, the single source for both languages, so the two appendices cannot
drift apart. Run it after changing the registry:

    .venv-int/Scripts/python tools/make_apendice_h.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qang import measures as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "book")

TEXT = {
    "es": {
        "path": os.path.join(ROOT, "es", "apendice_h.qmd"),
        "title": "# Catálogo de submedidas qang {#sec-apendice-h}",
        "intro": r"""La unidad qang no reformula ninguna ley física. Cada submedida de este catálogo es una lectura de una
probabilidad, o de unas pocas, y es igual a una magnitud que la física o la estadística ya tienen. Lo que una
submedida puede aportar es una **decisión**: menos disparos, un pulso más corto, una compilación elegida antes de correr,
una lectura que el ruido no sesga. El catálogo registra, para cada una, su definición, a qué equivale, qué decisión
mejora y con qué evidencia, y la clasifica con uno de los estados de la @tbl-h-estados.

Las tablas se generan desde el registro `qang.measures` con `tools/make_apendice_h.py`, la misma fuente para las dos
lenguas del libro. Una submedida nueva entra al catálogo como **sin verificar** y sólo cambia de estado con una prueba,
de preferencia pre-registrada (capítulo 28).""",
        "status_head": "| estado | significado | submedidas |",
        "status_caption": ": Estados del catálogo. {#tbl-h-estados}",
        "meaning": {
            "exacta": "identidad exacta: nombra o re-escala algo; no se atribuye ganancia",
            "ganancia": "probada en simulación o análisis, con ganancia medida frente a la práctica usual",
            "hardware": "probada en un dispositivo real",
            "mixta": "ganancia en algunos regímenes, ninguna o pérdida en otros",
            "sin_ganancia": "probada: equivalente a la práctica usual",
            "en_contra": "probada: peor que la práctica usual",
            "sin_verificar": "propuesta sin prueba: candidata a un experimento pre-registrado",
            "no_adoptada": "propuesta descartada por errónea o sin respaldo",
        },
        "head": "| submedida | definición | equivale a | decisión que mejora | estado | evidencia | dónde |",
        "caption": ": {name}. {{#tbl-h-{fam}}}",
        "balance_title": "## Balance {#sec-h-balance}",
        "balance": r"""De {n} submedidas, {exact} son identidades exactas, {gain} tienen una ganancia medida en simulación o
análisis, {hw} se probaron en un dispositivo real (con un resultado mixto en el caso del mili-qang), {mixed} son mixtas, {nogain} resultaron equivalentes a la práctica usual, {against} perdió frente a
ella, {open} están sin verificar y {rejected} no se adoptó. Las ganancias no vienen de la unidad sino de las decisiones que
la lectura permite tomar antes de gastar recursos: elegir el representante de menor área, la compilación de menor
exposición, el código según dos lecturas, o leer sólo el sector de peso que el T1 no contamina. Las submedidas sin
verificar son la lista de experimentos pendientes; el catálogo las mantiene a la vista para que no se citen como
resultados.""",
        "code_title": "## En código {#sec-h-codigo}",
        "code": """```python
from qang import measures

measures.summary()                       # submedidas por estado
measures.by_status("sin_verificar")      # los experimentos pendientes
m = measures.find("min_area")
m.decision[0], m.evidence[0]             # la decisión que mejora y la evidencia, en español
measures.resolve(m)                      # qang.pulse_area.minimal_area_circuit
```""",
    },
    "en": {
        "path": os.path.join(ROOT, "en", "appendix_h.qmd"),
        "title": "# Catalogue of qang sub-measures {#sec-appendix-h}",
        "intro": r"""The qang unit does not reformulate any physical law. Each sub-measure in this catalogue is a reading of
one probability, or of a few, and equals a quantity that physics or statistics already has. What a sub-measure can
bring is a **decision**: fewer shots, a shorter pulse, a compilation chosen before running, a reading that noise does
not bias. For each one the catalogue records its definition, what it equals, which decision it improves and with what
evidence, and classifies it with one of the statuses of @tbl-h-statuses.

The tables are generated from the registry `qang.measures` with `tools/make_apendice_h.py`, the same source for both
languages of the book. A new sub-measure enters the catalogue as **unverified** and changes status only with a test,
preferably pre-registered (Chapter 28).""",
        "status_head": "| status | meaning | sub-measures |",
        "status_caption": ": Statuses of the catalogue. {#tbl-h-statuses}",
        "meaning": {
            "exacta": "exact identity: it names or re-scales something; no gain is claimed",
            "ganancia": "tested in simulation or analysis, with a measured gain over the usual practice",
            "hardware": "tested on a real device",
            "mixta": "a gain in some regimes, none or a loss in others",
            "sin_ganancia": "tested: equivalent to the usual practice",
            "en_contra": "tested: worse than the usual practice",
            "sin_verificar": "untested proposal: a candidate for a pre-registered experiment",
            "no_adoptada": "proposal rejected as wrong or unsupported",
        },
        "head": "| sub-measure | definition | equals | decision it improves | status | evidence | where |",
        "caption": ": {name}. {{#tbl-h-{fam}}}",
        "balance_title": "## Balance {#sec-h-balance}",
        "balance": r"""Of {n} sub-measures, {exact} are exact identities, {gain} have a measured gain in simulation or
analysis, {hw} were tested on a real device (with a mixed result for the milli-qang), {mixed} are mixed, {nogain} turned out equivalent to the usual practice, {against} lost to it, {open}
are unverified and {rejected} was not adopted. The gains do not come from the unit but from the decisions the reading
allows before spending resources: choosing the minimum-area representative, the compilation of least exposure, the
code from two readings, or reading only the weight sector that T1 does not contaminate. The unverified sub-measures
are the list of pending experiments; the catalogue keeps them in sight so that they are not cited as results.""",
        "code_title": "## In code {#sec-h-code}",
        "code": """```python
from qang import measures

measures.summary()                       # sub-measures per status
measures.by_status("sin_verificar")      # the pending experiments
m = measures.find("min_area")
m.decision[1], m.evidence[1]             # the decision it improves and the evidence, in English
measures.resolve(m)                      # qang.pulse_area.minimal_area_circuit
```""",
    },
}


def cell(s):
    return s.replace("|", r"\vert{}")


def where_en(s):
    return s.replace("caps.", "chs.").replace("cap.", "ch.").replace("ap.", "app.").replace(" y ", " and ")


def build(lang):
    t = TEXT[lang]
    i = 0 if lang == "es" else 1
    counts = M.summary()
    out = [t["title"], "", t["intro"], "", t["status_head"], "|---|---|---|"]
    for s, names in M.STATUSES.items():
        out.append(f"| {names[i]} | {t['meaning'][s]} | {counts[s]} |")
    out += ["", t["status_caption"], ""]
    for fam, names in M.FAMILIES.items():
        out += [f"## {names[i]} {{#sec-h-{fam}}}", "", t["head"], "|---|---|---|---|---|---|---|"]
        for m in M.by_family(fam):
            where = m.where if lang == "es" else where_en(m.where)
            out.append("| " + " | ".join(cell(x) for x in (
                m.text("symbol", lang), m.text("definition", lang), m.equals[i], m.decision[i], f"**{M.STATUSES[m.status][i]}**",
                m.evidence[i], where)) + " |")
        out += ["", t["caption"].format(name=names[i], fam=fam), ""]
    out += [t["balance_title"], "", t["balance"].format(
        n=len(M.CATALOG), exact=counts["exacta"], gain=counts["ganancia"], hw=counts["hardware"],
        mixed=counts["mixta"], nogain=counts["sin_ganancia"], against=counts["en_contra"],
        open=counts["sin_verificar"], rejected=counts["no_adoptada"]), ""]
    out += [t["code_title"], "", t["code"], ""]
    with open(t["path"], "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    return t["path"]


if __name__ == "__main__":
    for lang in ("es", "en"):
        print(build(lang))
