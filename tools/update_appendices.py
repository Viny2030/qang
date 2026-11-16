"""tools/update_appendices.py — añade (de forma idempotente) a los apéndices A–H la sección «Actualización a la reescritura
(Cap. 7–41)». Cada cifra se calcula aquí con funciones de `qang` y se comprueba con `assert`; si algo cambia, el script se detiene.
Ejecutar DESPUÉS de tools/make_appendices.py y tools/make_apendice_h.py (que reescriben los archivos):

    python tools/update_appendices.py [raíz_del_repositorio]

Sólo NumPy y biblioteca estándar (qang.qlimits, qgarnet, qchar, qtransv, qfalsify, qledger, formulation).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from qang import formulation as F  # noqa: E402
from qang import qchar, qfalsify, qgarnet, qledger, qlimits, qtransv  # noqa: E402
from qang.measures import STATUSES  # noqa: E402

BEGIN, END = "<!-- qang-update-39 -->", "<!-- /qang-update-39 -->"
FILES = {k: (os.path.join("book", "es", "apendice_%s.qmd" % k), os.path.join("book", "en", "appendix_%s.qmd" % k)) for k in "abcdefgh"}


def put_section(path: str, body: str) -> bool:
    """Sustituye (o agrega al final) el bloque marcado. Devuelve True si el archivo cambió."""
    with open(path, encoding="utf-8", newline="") as fh:
        s = fh.read().replace("\r\n", "\n")
    s0 = s
    s = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", s, flags=re.S).rstrip("\n") + "\n\n"
    s += BEGIN + "\n" + body.strip("\n") + "\n" + END + "\n"
    if s == s0:
        return False
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(s)
    return True


# ------------------------------------------------------------------ cifras (cada una comprobada)
def facts() -> dict:
    f = {}
    keep = sorted(k for k, U in F.GATES_MULTI.items() if F.conserves_weight(U, int(np.log2(np.shape(U)[0]))))
    f["multi_keep"] = keep
    assert keep == sorted(["CZ", "SWAP", "iSWAP", "Fredkin"]), keep
    psi = qlimits.ring_graph_state(4)
    p = qlimits.z_distribution(psi)
    f["ring_qgz"] = float(qlimits.mean_qgz(p, 4))
    f["ring_stab"] = [float(qlimits.stabilizer_expectation(psi, i, 4)) for i in range(4)]
    assert abs(f["ring_qgz"]) < 1e-12 and all(abs(v - 1) < 1e-12 for v in f["ring_stab"])
    e0, e1, target = 0.05, 0.08, 300.0
    raw = qgarnet.predicted_raw_mqg(target, e0, e1)
    f["raw"], f["back"] = raw, qgarnet.correct_mqg(raw, e0, e1)
    assert abs(raw - 291.0) < 1e-9 and abs(f["back"] - target) < 1e-9
    f["cal"] = qchar.calibrate_readout(50, 1000, 80, 1000)
    assert abs(f["cal"][0] - 0.05) < 1e-12 and abs(f["cal"][1] - 0.08) < 1e-12
    n = 3
    pw = np.zeros(2 ** n)
    for q in range(n):
        pw[1 << (n - 1 - q)] = 1.0 / n
    gam = np.linspace(0.0, 0.6, 13)
    qgs, mean_z = qlimits.t1_noise_scan(pw, n, gam)
    f["qgs_mono"], f["z_mono"] = qlimits.is_monotone(qgs, True), qlimits.is_monotone(mean_z, True)
    assert (not f["qgs_mono"]) and f["z_mono"]
    f["qgs_peak"] = float(np.max(qgs))
    r = 1.0
    f["gd"], f["vac"] = float(qtransv.gudermannian(r)), float(qtransv.vacuum_weight(r))
    assert abs(f["gd"] - 2 * np.arctan(np.tanh(r / 2))) < 1e-12 and abs(f["vac"] - 1 / np.cosh(r) ** 2) < 1e-12
    pred = {"id": "demo", "p0": 0.5, "n": 100}
    dig = qfalsify.register(pred)
    f["reg_ok"], f["reg_bad"] = qfalsify.verify_registration(pred, dig), qfalsify.verify_registration({**pred, "n": 101}, dig)
    assert f["reg_ok"] and not f["reg_bad"]
    rows = qledger.load_ledger(os.path.join(ROOT, "book"), 7, 41)
    f["rows"] = rows
    f["summary"] = qledger.summarize(rows)
    assert f["summary"].get("measured gain", 0) == 0 and sum(f["summary"].values()) == len(rows)
    return f


def fmt(x: float, d: int = 3) -> str:
    return ("%.*f" % (d, x))


def note(lang: str) -> str:
    return ("Las referencias «capítulo N» de este apéndice usan la numeración de `cap01.qmd`–`cap36.qmd`. La reescritura "
            "(Parte XV, `qjax07`–`qjax41`) usa la numeración nueva: 1–6 igual; 7–29 pasan a +3 (7–9 son nuevos); 30–35 pasan a +4; "
            "el 33 nuevo es la teoría de cuerdas.") if lang == "es" else (
        "The “chapter N” references in this appendix use the numbering of `cap01.qmd`–`cap36.qmd`. The rewrite (Part XV, "
        "`qjax07`–`qjax41`) uses the new numbering: 1–6 unchanged; 7–29 shift by +3 (7–9 are new); 30–35 shift by +4; the new "
        "33 is string theory.")


def section(key: str, lang: str, f: dict) -> str:
    es = lang == "es"
    h = {"a": "sec-a-act", "b": "sec-b-act", "c": "sec-c-act", "d": "sec-d-act", "e": "sec-e-act", "f": "sec-f-act",
         "g": "sec-g-act", "h": "sec-h-act"}[key]
    title = ("## Actualización a la reescritura (Cap. 7–41) {#%s}" if es else "## Update for the rewrite (Chapters 7–41) {#%s}") % h
    out = [title, "", note(lang), ""]
    if key == "a":
        k = ", ".join(f["multi_keep"])
        out += [("**Compuertas que conservan el peso:** %s (`formulation.conserves_weight`). El resto de las de varios qubits "
                 "(CX, Toffoli) cambia el peso." % k) if es else
                ("**Weight-conserving gates:** %s (`formulation.conserves_weight`). The other multi-qubit gates (CX, Toffoli) change "
                 "the weight." % k), "",
                ("**Límite de lectura (Cap. 34, ex-30; `qang.qlimits`).** Que una compuerta entrelace no se ve en $qg_Z$: el estado de "
                 "grafo de anillo de 4 qubits, armado con CZ, tiene $qg_Z=%s$ en todos los qubits (%s m-qg) y cuatro estabilizadores "
                 "$X Z Z$ iguales a $1$. Con la lectura de $Z$ solamente, ese estado es indistinguible de $|+\\rangle^{\\otimes4}$." %
                 (fmt(f["ring_qgz"], 1), fmt(1000 * f["ring_qgz"], 1))) if es else
                ("**Reading limit (Ch. 34, formerly 30; `qang.qlimits`).** Entangling cannot be seen in $qg_Z$: the 4-qubit ring graph "
                 "state, built with CZ, has $qg_Z=%s$ on every qubit (%s m-qg) and four $XZZ$ stabilizers equal to $1$. With the "
                 "$Z$ reading alone it cannot be told from $|+\\rangle^{\\otimes4}$." % (fmt(f["ring_qgz"], 1), fmt(1000 * f["ring_qgz"], 1))), "",
                ("Orden de qubits entre SDKs (Cap. 36, ex-32; `qang.qarch.convention_error_mqg`): la tabla de arriba usa el qubit 0 a la "
                 "izquierda, como Cirq y PennyLane; Qiskit lo pone a la derecha.") if es else
                ("Qubit order across SDKs (Ch. 36, formerly 32; `qang.qarch.convention_error_mqg`): the table above puts qubit 0 on the "
                 "left, like Cirq and PennyLane; Qiskit puts it on the right.")]
    elif key == "b":
        out += [("**Lectura con error de medida (Cap. 35 y 30; `qang.qgarnet.correct_mqg`, `qang.qchar.correct_mqg`).** Con "
                 "$e_0=P(1|0)$ y $e_1=P(0|1)$, la lectura cruda es afín: $m'=(1-e_0-e_1)\\,m+1000\\,(e_1-e_0)$. Para $m=300$ m-qg, "
                 "$e_0=0{,}05$, $e_1=0{,}08$: $m'=%s$ m-qg, y `correct_mqg` devuelve $%s$ m-qg (error $<10^{-9}$)." %
                 (fmt(f["raw"], 1), fmt(f["back"], 1))) if es else
                ("**Reading with measurement error (Ch. 35 and 30; `qang.qgarnet.correct_mqg`, `qang.qchar.correct_mqg`).** With "
                 "$e_0=P(1|0)$ and $e_1=P(0|1)$ the raw reading is affine: $m'=(1-e_0-e_1)\\,m+1000\\,(e_1-e_0)$. For $m=300$ m-qg, "
                 "$e_0=0.05$, $e_1=0.08$: $m'=%s$ m-qg, and `correct_mqg` returns $%s$ m-qg (error $<10^{-9}$)." % (fmt(f["raw"], 1), fmt(f["back"], 1))), "",
                ("El gradiente natural regularizado y la distancia TV de los Cap. 8–9 (`qang.qdist`, `qang.qfisher`) no agregan fórmulas "
                 "a este apéndice: usan las de §B.1 y la métrica de Fisher del capítulo 8.") if es else
                ("The regularized natural gradient and the TV distance of Ch. 8–9 (`qang.qdist`, `qang.qfisher`) add no formulas to this "
                 "appendix: they use those of §B.1 and the Fisher metric of Chapter 8.")]
    elif key == "c":
        e01, e10 = f["cal"][0], f["cal"][1]
        out += [("**Protocolo anunciado (Cap. 30, ex-27; `qang.qchar.calibrate_readout`).** Con 50 unos sobre 1000 preparaciones de "
                 "$|0\\rangle$ y 80 ceros sobre 1000 de $|1\\rangle$: $e_{01}=%s$, $e_{10}=%s$. La corrección posterior es la afín del "
                 "apéndice B (`correct_mqg`). Con esos errores, 300 m-qg se leen 291,0 m-qg sin corregir." % (fmt(e01, 2), fmt(e10, 2))) if es else
                ("**Announced protocol (Ch. 30, formerly 27; `qang.qchar.calibrate_readout`).** With 50 ones out of 1000 preparations of "
                 "$|0\\rangle$ and 80 zeros out of 1000 of $|1\\rangle$: $e_{01}=%s$, $e_{10}=%s$. The later correction is the affine one "
                 "of Appendix B (`correct_mqg`). With those errors, 300 m-qg read as 291.0 m-qg uncorrected." % (fmt(e01, 2), fmt(e10, 2))), "",
                ("En el Cap. 35 (ex-31) la misma corrección se aplica a las lecturas de IQM Garnet guardadas "
                 "(`qang.qgarnet.readout_from_poles`); el estado de esas cifras es `tested on hardware`, de una sola corrida por día.") if es else
                ("In Ch. 35 (formerly 31) the same correction is applied to the stored IQM Garnet readings "
                 "(`qang.qgarnet.readout_from_poles`); the status of those figures is `tested on hardware`, a single run per day.")]
    elif key == "d":
        out += [("La tabla de módulos se regenera con `tools/make_appendices.py`, que lee el código; los módulos de la reescritura "
                 "(`qan`, `qdist`, `qfisher`, `qcft`, `qanti`, `qtimeloop`, `qlimits`, `qgarnet`, `qarch`, `qnano`, `qtransv`, `qintegr`, `qledger`, `qfalsify`, "
                 "`qchar`, `qecc`, `qchem`) aparecen allí con sus nombres públicos. El módulo `qang.qarch` comprueba por código que "
                 "`import qang` sólo necesita NumPy (Cap. 36) y `qang.qintegr`, que las 12 integraciones se importan con guarda (Cap. 39).") if es else
                ("The module table is regenerated by `tools/make_appendices.py`, which reads the code; the rewrite's modules (`qan`, "
                 "`qdist`, `qfisher`, `qcft`, `qanti`, `qtimeloop`, `qlimits`, `qgarnet`, `qarch`, `qnano`, `qtransv`, `qintegr`, `qledger`, `qfalsify`, `qchar`, "
                 "`qecc`, `qchem`) appear there with their public names. `qang.qarch` checks by code that `import qang` needs only NumPy "
                 "(Ch. 36) and `qang.qintegr` that the 12 integrations are imported behind a guard (Ch. 39).")]
    elif key == "e":
        s = f["summary"]
        order = [x for x in STATUSES]
        names = {"exacta": "exact", "ganancia": "measured gain", "hardware": "tested on hardware", "mixta": "mixed",
                 "sin_ganancia": "no gain", "en_contra": "against", "sin_verificar": "unverified", "no_adoptada": "not adopted"}
        cnt = ", ".join("`%s` %d" % (STATUSES[k][0 if es else 1], s.get(names[k], 0)) for k in order)
        out += [("**Maquinaria (Cap. 31, ex-28; `qang.qfalsify`).** `register` fija una predicción por su hash SHA-256 y "
                 "`verify_registration` detecta cualquier cambio posterior (con $n=101$ en vez de $100$ la comprobación da `False`). "
                 "`decide` y `holm` fijan la regla de decisión con el nivel $\\alpha=0{,}05$ y la corrección de comparaciones múltiples.") if es else
                ("**Machinery (Ch. 31, formerly 28; `qang.qfalsify`).** `register` fixes a prediction by its SHA-256 hash and "
                 "`verify_registration` detects any later change (with $n=101$ instead of $100$ the check gives `False`). `decide` and "
                 "`holm` fix the decision rule at level $\\alpha=0.05$ with a multiple-comparison correction."), "",
                ("**Libro de cuentas de las submedidas (Cap. 32, ex-29; `qang.qledger`).** Sobre las %d filas del Apéndice H de los Cap. "
                 "7–41: %s. Ninguna es `ganancia medida`; la falsación del programa entero sigue como se declaró arriba." % (len(f["rows"]), cnt)) if es else
                ("**Sub-measure ledger (Ch. 32, formerly 29; `qang.qledger`).** Over the %d Appendix H rows of Ch. 7–41: %s. None is "
                 "`measured gain`; the falsification of the whole programme stands as declared above." % (len(f["rows"]), cnt)), "",
                ("**Hardware (Cap. 35, `qang.qgarnet`).** Los pre-registros de IQM Garnet se comprueban contra el hash del *blob* en el "
                 "commit de registro (`blob_hash_at_commit`), no contra el archivo actual: el *docstring* del script se editó después.") if es else
                ("**Hardware (Ch. 35, `qang.qgarnet`).** The IQM Garnet pre-registrations are checked against the blob hash at the "
                 "registration commit (`blob_hash_at_commit`), not against the current file: the script's docstring was edited afterwards.")]
    elif key == "f":
        out += [("**Dónde falla la representación (Cap. 34, ex-30; `qang.qlimits.t1_noise_scan`).** Para el estado W de 3 qubits, con "
                 "amortiguamiento $\\gamma$ de 0 a 0,6, la media de $qg_Z$ crece de forma monótona (de 333,3 a 733,3 m-qg) y $qg_S$ por qubit "
                 "no: sube hasta $%s$ y baja. $qg_Z$ es un funcional lineal con representante (§F.3); $qg_S$ no (§F.4), y esa es la "
                 "diferencia que la lectura bajo ruido hace visible." % fmt(f["qgs_peak"], 4)) if es else
                ("**Where the representation fails (Ch. 34, formerly 30; `qang.qlimits.t1_noise_scan`).** For the 3-qubit W state, with "
                 "damping $\\gamma$ from 0 to 0.6, the mean $qg_Z$ grows monotonically (from 333.3 to 733.3 m-qg) while $qg_S$ per qubit "
                 "does not: it rises to $%s$ and falls. $qg_Z$ is a linear functional with a representative (§F.3); $qg_S$ is not "
                 "(§F.4), and that is the difference that readout under noise makes visible." % fmt(f["qgs_peak"], 4))]
    elif key == "g":
        out += [("**Estado.** La propuesta sigue siendo `unverified`; lo único en hardware es la lectura de $\\beta$ del §170, "
                 "`tested on hardware`. La reescritura agrega una relación exacta del mismo tipo (Cap. 38, ex-34; `qang.qtransv`): en "
                 "el vacío de Unruh, $\\theta/2=\\mathrm{gd}(r)=2\\arctan\\tanh(r/2)$; para $r=1$ vale $%s$ rad y el peso del vacío "
                 "$\\cos^2(\\mathrm{gd}\\,r)=\\mathrm{sech}^2 r=%s$. Es una identidad de lectura, no una verificación de la energía "
                 "relativista." % (fmt(f["gd"], 6), fmt(f["vac"], 6))) if es else
                ("**Status.** The proposal remains `unverified`; the only hardware item is the $\\beta$ reading of §170, `tested on "
                 "hardware`. The rewrite adds an exact relation of the same kind (Ch. 38, formerly 34; `qang.qtransv`): in the Unruh "
                 "vacuum, $\\theta/2=\\mathrm{gd}(r)=2\\arctan\\tanh(r/2)$; for $r=1$ it is $%s$ rad and the vacuum weight "
                 "$\\cos^2(\\mathrm{gd}\\,r)=\\mathrm{sech}^2 r=%s$. It is a reading identity, not a verification of relativistic "
                 "energy." % (fmt(f["gd"], 6), fmt(f["vac"], 6)))]
    elif key == "h":
        rows = f["rows"]
        s = f["summary"]
        names = {k: k for k in STATUSES}
        out += [("Submedidas de las tablas «Apéndice H» de los capítulos reescritos (`qang.qledger.load_ledger`, %d filas). Los "
                 "números de capítulo son los nuevos. El catálogo de arriba (`qang.measures`) no se modificó." % len(rows)) if es else
                ("Sub-measures from the “Appendix H” tables of the rewritten chapters (`qang.qledger.load_ledger`, %d rows). Chapter "
                 "numbers are the new ones. The catalogue above (`qang.measures`) was not modified." % len(rows)), "",
                "| " + ("estatus principal | filas" if es else "main status | rows") + " |", "|---|---|"]
        trans = {"exact": STATUSES["exacta"], "measured gain": STATUSES["ganancia"], "tested on hardware": STATUSES["hardware"],
                 "mixed": STATUSES["mixta"], "no gain": STATUSES["sin_ganancia"], "against": STATUSES["en_contra"],
                 "unverified": STATUSES["sin_verificar"], "not adopted": STATUSES["no_adoptada"]}
        for k in qledger.STATUSES:
            out.append("| %s | %d |" % (trans[k][0 if es else 1], s.get(k, 0)))
        out += ["", ": " + ("Estatus de las submedidas de los Cap. 7–41." if es else "Status of the sub-measures of Ch. 7–41."), "",
                "| cap. | id | " + ("función" if es else "function") + " | " + ("estatus" if es else "status") + " |", "|---|---|---|---|"]
        for r in rows:
            cell = lambda t: t.replace("|", "\\|")  # noqa: E731
            out.append("| %d | `%s` | %s | %s |" % (r["cap"], r["id"], cell(r["functions"]), cell(r["status"])))
        out += ["", ": " + ("Submedidas de la reescritura, por capítulo." if es else "Rewrite sub-measures, by chapter.")]
    return "\n".join(out)


def main() -> None:
    f = facts()
    changed = 0
    for key, (es_p, en_p) in FILES.items():
        for lang, rel in (("es", es_p), ("en", en_p)):
            path = os.path.join(ROOT, rel)
            if not os.path.exists(path):
                print("  (falta %s)" % rel)
                continue
            changed += put_section(path, section(key, lang, f))
    print("update_appendices: %d archivos cambiados; %d filas del libro de cuentas; resumen %s" % (changed, len(f["rows"]), dict(f["summary"])))


if __name__ == "__main__":
    main()
