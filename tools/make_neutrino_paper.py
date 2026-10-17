"""
Builds manuscript/qang_neutrinos.docx and manuscript/qang_neutrinos_es.docx (and the PDFs through Microsoft Word,
with docx2pdf, when it is available): the neutrino paper, RESEARCH_NOTES §151-§156 and §163.

python tools/make_neutrino_paper.py [--pdf]
"""

import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "manuscript")

T = {
    "title": ("Neutrino oscillations as weight-preserving quantum circuits: what the qg symmetry filter reads, "
              "and what no oscillation circuit can see",
              "Oscilaciones de neutrinos como circuitos cuánticos que conservan el peso: qué lee el filtro de "
              "simetría qg y qué no puede ver ningún circuito de oscilación"),
    "author": ("Vicente Humberto Monteverde\nUniversidad del Museo Social Argentino (UMSA), Argentina\n"
               "ORCID 0000-0001-8884-4811",) * 2,
    "date": ("Draft of 17 October 2026", "Borrador del 17 de octubre de 2026"),
}

ABSTRACT = (
    "A neutrino is produced and detected in a flavour state and propagates in mass states, so flavour oscillation "
    "is a rotation. We write it in the measurement-level units of the open-source library qang, where a qubit is "
    "described by qg_Z = <Z>, and place every flavour on its own qubit with a single excitation (the weight-1 "
    "sector). Every gate of the oscillation circuit then conserves the Hamming weight, and the qg symmetry filter "
    "(discard the shots that left the sector) applies. Six results follow, all on the noise models of IBM devices, "
    "each pre-registered and reported as passed or failed. (i) In vacuum qg_Z = cos²2θ + sin²2θ cos Δ is the "
    "spherical law of cosines, and at the oscillation maximum the mixing angle is a quarter of the qang angle "
    "arccos(qg_Z), with the θ-independent error bar 1/(4√(N+½)). (ii) Three flavours on three qubits: the filter cuts "
    "the error of the transition probabilities about 3× and the CP asymmetry at a DUNE-like baseline survives. "
    "(iii) Earth matter enters as a weight-preserving Trotter circuit, and matter fakes CP violation. (iv) In dense "
    "neutrino gases the conserved number is the number of ν₂, not the flavour number, so the filter applies in the "
    "mass basis (2.2-2.8×). (v) For solar neutrinos the phases are lost before detection, so the whole Sun is one "
    "rotation and the filter, though it works, is not needed. (vi) The mass ordering is read correctly through matter "
    "on three device models; Majorana phases change no oscillation probability, and the transpiler deletes them; a "
    "sterile neutrino is a fourth qubit in the weight-1 sector, where the filter cuts the error 5.5×. The link between "
    "neutrinos and qang is modelling, not new physics: the oscillation is a rotation, and qang reads rotations.",
    "Un neutrino se produce y se detecta en un estado de sabor y se propaga en estados de masa, así que la oscilación "
    "de sabor es una rotación. La escribimos en las unidades de nivel de medición de la biblioteca de código abierto "
    "qang, donde un qubit se describe por qg_Z = <Z>, y ponemos cada sabor en su propio qubit con una sola excitación "
    "(el sector de peso 1). Todas las compuertas del circuito de oscilación conservan entonces el peso de Hamming, y "
    "el filtro de simetría qg (descartar los disparos que salieron del sector) se aplica. Seis resultados, todos en "
    "modelos de ruido de dispositivos IBM, cada uno preregistrado e informado como aprobado o fallido. (i) En vacío "
    "qg_Z = cos²2θ + sin²2θ cos Δ es la ley de cosenos esférica, y en el máximo de oscilación el ángulo de mezcla es "
    "un cuarto del ángulo qang arccos(qg_Z), con la barra de error 1/(4√(N+½)) independiente de θ. (ii) Tres sabores en "
    "tres qubits: el filtro reduce unas 3× el error de las probabilidades de transición y la asimetría CP a una "
    "distancia tipo DUNE se mantiene. (iii) La materia de la Tierra entra como un circuito de Trotter que conserva el "
    "peso, y la materia simula violación CP. (iv) En gases densos de neutrinos lo que se conserva es el número de ν₂, "
    "no el número de sabor, así que el filtro se aplica en la base de masa (2.2-2.8×). (v) En los neutrinos solares "
    "las fases se pierden antes de la detección, así que todo el Sol es una rotación y el filtro, aunque funciona, no "
    "hace falta. (vi) La jerarquía de masas se lee bien a través de la materia en tres modelos de dispositivo; las "
    "fases de Majorana no cambian ninguna probabilidad de oscilación, y el transpilador las borra; un neutrino "
    "estéril es un cuarto qubit en el sector de peso 1, donde el filtro reduce el error 5.5×. El vínculo entre los "
    "neutrinos y qang es de modelado, no física nueva: la oscilación es una rotación, y qang lee rotaciones.",
)

SECTIONS = [
    (("1. Oscillation is a rotation", "1. La oscilación es una rotación"), [
        ("With two flavours on one qubit (|0⟩ = ν_e), going to the mass basis (RY(−2θ)), letting the mass states pick "
         "up their relative phase Δ = Δm²L/2E (RZ) and coming back (RY(2θ)) gives P(e→μ) = sin²2θ sin²(Δ/2) and",
         "Con dos sabores en un qubit (|0⟩ = ν_e), pasar a la base de masa (RY(−2θ)), dejar que los estados de masa "
         "acumulen su fase relativa Δ = Δm²L/2E (RZ) y volver (RY(2θ)) da P(e→μ) = sin²2θ sin²(Δ/2) y"),
        ("EQ qg_Z = 1 − 2P = cos²(2θ) + sin²(2θ) cos Δ,", "EQ qg_Z = 1 − 2P = cos²(2θ) + sin²(2θ) cos Δ,"),
        ("the spherical law of cosines: the Bloch vector precesses on a cone of half-angle 2θ about the mass axis, and "
         "the qang angle arccos(qg_Z) is the great-circle distance travelled. At the maximum (Δ = π) it equals 4θ "
         "exactly, so θ = arccos(qg_Z)/4 inherits the θ-independent shot-noise standard deviation 1/(4√(N+½)) of the "
         "arcsine transformation. With N = 1000 shots, 2000 repetitions gave 0.0077-0.0081 against 0.0079 predicted. "
         "Elsewhere the relation is not linear: qg removes the sin² only at the maximum.",
         "la ley de cosenos esférica: el vector de Bloch precesa en un cono de semiángulo 2θ alrededor del eje de "
         "masa, y el ángulo qang arccos(qg_Z) es la distancia de círculo máximo recorrida. En el máximo (Δ = π) vale "
         "exactamente 4θ, así que θ = arccos(qg_Z)/4 hereda la desviación estándar de ruido de disparo 1/(4√(N+½)), "
         "independiente de θ, de la transformación arcoseno. Con N = 1000 disparos, 2000 repeticiones dieron "
         "0.0077-0.0081 contra 0.0079 predicho. Fuera del máximo la relación no es lineal: qg quita el sin² solo en "
         "el máximo."),
    ]),
    (("2. Three flavours in the weight-1 sector", "2. Tres sabores en el sector de peso 1"), [
        ("Three flavours go on three qubits, qubit k excited = flavour k. The PMNS matrix U = R23 U_δ R13 U_δ† R12 is "
         "three RBS (XX+YY) rotations and two phase gates, and propagation is one phase gate per qubit, so every gate "
         "conserves the weight. At L = 1300 km, 0.8-5 GeV, from ν_μ, the filter with readout correction lowered the "
         "mean error of the nine transition probabilities from 0.025-0.048 to 0.009-0.013 on FakeBrisbane, FakeTorino "
         "and FakeFez (median 3.4×; 18 two-qubit gates, 86-94% of the shots kept). The CP asymmetry at 2.6 GeV "
         "(exact 0.025) read 0.022-0.026 filtered; it also survives without the filter, because it is a difference of "
         "two nearly identical circuits and the common shrinkage cancels.",
         "Tres sabores van en tres qubits, qubit k excitado = sabor k. La matriz PMNS U = R23 U_δ R13 U_δ† R12 son tres "
         "rotaciones RBS (XX+YY) y dos compuertas de fase, y la propagación es una compuerta de fase por qubit, así "
         "que todas conservan el peso. A L = 1300 km, 0.8-5 GeV, desde ν_μ, el filtro con corrección de lectura bajó "
         "el error medio de las nueve probabilidades de transición de 0.025-0.048 a 0.009-0.013 en FakeBrisbane, "
         "FakeTorino y FakeFez (mediana 3.4×; 18 compuertas de dos qubits, 86-94% de los disparos conservados). La "
         "asimetría CP en 2.6 GeV (exacta 0.025) dio 0.022-0.026 filtrada; también sobrevive sin el filtro, porque es "
         "una diferencia de dos circuitos casi idénticos y la contracción común se cancela."),
    ]),
    (("3. Earth matter and dense neutrino gases", "3. Materia de la Tierra y gases densos de neutrinos"), [
        ("Electrons add A = 2√2 G_F N_e E to the ν_e entry of the Hamiltonian. Alternating the exact vacuum step with "
         "a phase gate on the ν_e qubit (second-order Trotter) keeps every gate weight-preserving. At the DUNE-like "
         "baseline matter alone produces an asymmetry of 0.042, larger than the vacuum CP signal (0.025), and the "
         "filtered circuit sees it (0.029-0.035 at two Trotter steps, whose own value is 0.033). The filter gained 2.8× "
         "(median). One Trotter step already had less error than the noise, so there was no depth trade-off.",
         "Los electrones suman A = 2√2 G_F N_e E a la entrada ν_e del Hamiltoniano. Alternar el paso exacto de vacío "
         "con una compuerta de fase en el qubit ν_e (Trotter de segundo orden) mantiene todas las compuertas "
         "conservando el peso. A la distancia tipo DUNE la materia sola produce una asimetría de 0.042, mayor que la "
         "señal CP de vacío (0.025), y el circuito filtrado la ve (0.029-0.035 con dos pasos de Trotter, cuyo valor "
         "propio es 0.033). El filtro ganó 2.8× (mediana). Un paso de Trotter ya tenía menos error que el ruido, así "
         "que no hubo compromiso de profundidad."),
        ("In a supernova the neutrino density is so high that neutrinos oscillate together. With the vacuum term, what "
         "is conserved is the number of ν₂, not the flavour number, so the filter applies to mass-basis read-outs. All "
         "seven pre-registered predictions failed for one reason: the noiseless Trotter error (0.095) exceeded the "
         "noise. Scored post hoc against the noiseless circuit, the mass-basis filter gained 2.2-2.8× and reweighting by "
         "known sector probabilities 1.6-1.7×; the flavour basis, where nothing is conserved, gained nothing beyond "
         "readout correction.",
         "En una supernova la densidad de neutrinos es tan alta que oscilan juntos. Con el término de vacío, lo que se "
         "conserva es el número de ν₂, no el número de sabor, así que el filtro se aplica a lecturas en la base de "
         "masa. Las siete predicciones preregistradas fallaron por una sola razón: el error de Trotter sin ruido (0.095) "
         "superó al ruido. Evaluado a posteriori contra el circuito sin ruido, el filtro en base de masa ganó 2.2-2.8× y "
         "el reponderado por probabilidades de sector conocidas 1.6-1.7×; la base de sabor, donde nada se conserva, no "
         "ganó nada más allá de la corrección de lectura."),
    ]),
    (("4. Solar neutrinos: the whole Sun is one rotation", "4. Neutrinos solares: todo el Sol es una rotación"), [
        ("A ν_e born at the solar core follows the heavier matter eigenstate out of the Sun above a few MeV, so P_ee "
         "falls from 0.54 to 0.30. The mass states lose their coherence on the way, so the read-out is in the vacuum "
         "mass basis and P_ee = c13⁴(cos²θ12 p₁ + sin²θ12 p₂) + s13⁴ is linear in qg_Z of the ν₂ qubit. A slab circuit "
         "(exact 2×2 propagators, weight-preserving) gained 1.9-2.3× with the filter; but since only one population is "
         "observed, the whole Sun compiles to the single rotation that loads it, which beats every slab circuit even "
         "without the filter. A quantum computer could add something only where coherence survives.",
         "Un ν_e nacido en el núcleo solar sigue al autoestado de materia más pesado hasta salir del Sol por encima de "
         "unos MeV, así que P_ee cae de 0.54 a 0.30. Los estados de masa pierden su coherencia en el camino, así que la "
         "lectura es en la base de masa de vacío y P_ee = c13⁴(cos²θ12 p₁ + sin²θ12 p₂) + s13⁴ es lineal en el qg_Z del "
         "qubit ν₂. Un circuito por capas (propagadores 2×2 exactos, que conservan el peso) ganó 1.9-2.3× con el filtro; "
         "pero como se observa una sola población, todo el Sol se compila en la única rotación que la carga, que le "
         "gana a todo circuito por capas incluso sin filtro. Una computadora cuántica podría aportar algo solo donde "
         "la coherencia sobrevive."),
    ]),
    (("5. What oscillation circuits can and cannot see", "5. Qué pueden ver y qué no los circuitos de oscilación"), [
        ("Mass ordering. With Δm²31 < 0 (inverted ordering) matter enhances the anti-neutrino μ→e instead of the "
         "neutrino. On three device models the filtered circuits read the ordering correctly, with the right sign of "
         "the ν/ν̄ asymmetry; the values sit 0.010-0.033 high, but the common bias cancels in NO − IO (0.034-0.036, "
         "exact 0.037).",
         "Jerarquía de masas. Con Δm²31 < 0 (orden invertido) la materia realza el antineutrino μ→e en lugar del "
         "neutrino. En tres modelos de dispositivo los circuitos filtrados leen bien el orden, con el signo correcto de "
         "la asimetría ν/ν̄; los valores quedan 0.010-0.033 arriba, pero el sesgo común se cancela en NO − IO "
         "(0.034-0.036, exacto 0.037)."),
        ("Majorana phases. U_M = U diag(1, e^{iα21/2}, e^{iα31/2}) is diagonal in the mass basis, commutes with the "
         "propagation phases and cancels in U_M D U_M†: no oscillation probability depends on it, in vacuum or matter "
         "(4·10⁻¹⁶). On all three device models the transpiler merges the Majorana phase gates with their inverses and "
         "outputs the same circuit. Neutrinoless double beta decay sees them: m_ββ = |Σ U_ei² m_i| = 1.4-3.7 meV (normal "
         "ordering) and 19-48 meV (inverted) for a massless lightest neutrino.",
         "Fases de Majorana. U_M = U diag(1, e^{iα21/2}, e^{iα31/2}) es diagonal en la base de masa, conmuta con las "
         "fases de propagación y se cancela en U_M D U_M†: ninguna probabilidad de oscilación depende de ella, en vacío "
         "o en materia (4·10⁻¹⁶). En los tres modelos de dispositivo el transpilador fusiona las compuertas de fase de "
         "Majorana con sus inversas y produce el mismo circuito. La desintegración doble beta sin neutrinos sí las ve: "
         "m_ββ = |Σ U_ei² m_i| = 1.4-3.7 meV (orden normal) y 19-48 meV (invertido) para un neutrino más liviano sin "
         "masa."),
        ("A sterile neutrino (3+1) is a fourth flavour that does not interact: one more qubit in the weight-1 sector. "
         "With hypothetical θ14 = θ24 = 0.15 and Δm²41 = 1 eV² at 0.5 km, the filter cut the error of P(μ→μ) 5.5× "
         "(median) and found the oscillation dip at the right energy (0.4 GeV) on all three device models.",
         "Un neutrino estéril (3+1) es un cuarto sabor que no interactúa: un qubit más en el sector de peso 1. Con "
         "θ14 = θ24 = 0.15 y Δm²41 = 1 eV² hipotéticos a 0.5 km, el filtro redujo el error de P(μ→μ) 5.5× (mediana) y "
         "encontró la caída de oscilación en la energía correcta (0.4 GeV) en los tres modelos de dispositivo."),
    ]),
    (("6. Discussion and limits", "6. Discusión y límites"), [
        ("The link between neutrinos and qang is modelling: oscillation is a rotation, and qang reads rotations and "
         "filters on conserved numbers. That gives a θ-independent error bar for the mixing angle, a 2-5× noise filter "
         "wherever the circuit conserves a number (flavour in vacuum and matter, ν₂ in dense gases, the sterile "
         "sector), and a clear statement of what a circuit cannot see (Majorana phases). It does not remove the "
         "trigonometry in general, does not speed up anything, and every result here is on simulated device noise "
         "with fixed parameters, not a fit to data. Of the pre-registered predictions behind this paper, 4 of 4 (§151), "
         "4 of 5 (§152), 0 of 7 (§154), 6 of 6 (§156) and 5 of 6 (§163) passed; all are reported.",
         "El vínculo entre los neutrinos y qang es de modelado: la oscilación es una rotación, y qang lee rotaciones y "
         "filtra con números conservados. Eso da una barra de error independiente de θ para el ángulo de mezcla, un "
         "filtro de ruido de 2-5× donde el circuito conserva un número (el sabor en vacío y materia, ν₂ en gases densos, "
         "el sector estéril), y una afirmación clara de lo que un circuito no puede ver (las fases de Majorana). No "
         "elimina la trigonometría en general, no acelera nada, y todos los resultados son con ruido de dispositivo "
         "simulado y parámetros fijos, no un ajuste a datos. De las predicciones preregistradas detrás de este "
         "artículo pasaron 4 de 4 (§151), 4 de 5 (§152), 0 de 7 (§154), 6 de 6 (§156) y 5 de 6 (§163); todas se "
         "informan."),
        ("Code and data: github.com/Viny2030/qang (qang ≥ 0.12, module qang.neutrino); the scripts "
         "examples/neutrino_*_qg.py with their pre-registration hashes; RESEARCH_NOTES §151-§156 and §163; notebooks "
         "notebooks/qang_neutrinos.ipynb (English) and qang_neutrinos_es.ipynb (Spanish).",
         "Código y datos: github.com/Viny2030/qang (qang ≥ 0.12, módulo qang.neutrino); los scripts "
         "examples/neutrino_*_qg.py con sus hashes de preregistro; RESEARCH_NOTES §151-§156 y §163; notebooks "
         "notebooks/qang_neutrinos.ipynb (inglés) y qang_neutrinos_es.ipynb (español)."),
    ]),
]

REFS = [
    "B. Pontecorvo, Sov. Phys. JETP 26, 984 (1968).",
    "Z. Maki, M. Nakagawa, S. Sakata, Prog. Theor. Phys. 28, 870 (1962).",
    "L. Wolfenstein, Phys. Rev. D 17, 2369 (1978).",
    "S. P. Mikheyev, A. Yu. Smirnov, Sov. J. Nucl. Phys. 42, 913 (1985).",
    "I. Esteban, M. C. Gonzalez-Garcia, M. Maltoni, T. Schwetz, A. Zhou, JHEP 09, 178 (2020) (NuFIT).",
    "H. Duan, G. M. Fuller, Y.-Z. Qian, Annu. Rev. Nucl. Part. Sci. 60, 569 (2010).",
    "C. A. Argüelles, B. J. P. Jones, Phys. Rev. Research 1, 033176 (2019).",
    "X. Bonet-Monroig, R. Sagastizabal, M. Singh, T. E. O'Brien, Phys. Rev. A 98, 062339 (2018).",
    "M. J. Dolinski, A. W. P. Poon, W. Rodejohann, Annu. Rev. Nucl. Part. Sci. 69, 219 (2019).",
    "V. H. Monteverde, qang: the Qang (qg) unit and metric, software, doi:10.5281/zenodo.22832150 (2026).",
]


def build(lang):
    i = 0 if lang == "en" else 1
    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    h = d.add_heading(T["title"][i], level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in (T["author"][i] + "\n" + T["date"][i]).split("\n"):
        p = d.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.add_heading("Abstract" if lang == "en" else "Resumen", level=1)
    d.add_paragraph(ABSTRACT[i])
    for (head, paras) in SECTIONS:
        d.add_heading(head[i], level=1)
        for par in paras:
            text = par[i]
            if text.startswith("EQ "):
                p = d.add_paragraph(text[3:])
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                d.add_paragraph(text)
    d.add_heading("Summary of results" if lang == "en" else "Resumen de resultados", level=1)
    rows = [("§", "question" if lang == "en" else "pregunta", "result" if lang == "en" else "resultado"),
            ("151", "vacuum, 3 flavours" if lang == "en" else "vacío, 3 sabores", "filter 3.4×; θ error bar 1/(4√(N+½))"),
            ("152", "Earth matter" if lang == "en" else "materia de la Tierra", "filter 2.8×; matter A_CP 0.042 seen"),
            ("154", "dense gases" if lang == "en" else "gases densos", "mass-basis filter 2.2–2.8× (post hoc)"),
            ("156", "solar" if lang == "en" else "solar", "one rotation; filter 1.9–2.3× not needed"),
            ("163", "ordering, Majorana, sterile" if lang == "en" else "jerarquía, Majorana, estéril",
             "ordering read; Majorana invisible; sterile 5.5×")]
    t = d.add_table(rows=len(rows), cols=3)
    t.style = "Light Grid Accent 1"
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    d.add_heading("References" if lang == "en" else "Referencias", level=1)
    for k, ref in enumerate(REFS, 1):
        d.add_paragraph(f"[{k}] {ref}")
    path = os.path.join(OUT, "qang_neutrinos" + ("" if lang == "en" else "_es") + ".docx")
    d.save(path)
    print("wrote", path)
    return path


if __name__ == "__main__":
    paths = [build("en"), build("es")]
    if "--pdf" in sys.argv:  # through a separate, hidden Microsoft Word instance (pywin32)
        import win32com.client

        word = win32com.client.DispatchEx("Word.Application")
        word.Visible, word.DisplayAlerts = False, 0
        try:
            for p in paths:
                doc = word.Documents.Open(os.path.abspath(p), ReadOnly=True, AddToRecentFiles=False)
                doc.SaveAs2(os.path.abspath(p[:-5] + ".pdf"), FileFormat=17)
                doc.Close(0)
                print("wrote", p[:-5] + ".pdf")
        finally:
            word.Quit()
