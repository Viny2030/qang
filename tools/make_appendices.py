"""Appendices A to F of the book (ES/EN), generated from the code so they cannot drift from it.

    .venv-int/Scripts/python tools/make_appendices.py

* Appendix A: the qg action of the 15 gates of qang.formulation, qg'_P = qg_{U^dag P U}, from formulation.gate_table,
  with Clifford, weight conservation and the radius class (formulation.gate_radius_class).
* Appendix D: every module of the qang package with its public names (``__all__``, or the public functions and
  classes it defines). Run it in an environment where the optional SDKs import (``.venv-int``), so that every module
  is listed; a module that fails to import is reported, not skipped silently.

* Appendix B: the closed forms of qang.algebra (universal gate, reflections, transition probability), the diagonal
  Gell-Mann matrices and the Grover trajectory as Chebyshev polynomials (§37, §166), each checked against direct linear
  algebra on random states.
* Appendix C: the echo-calibration protocol of qang.sectors (§105-§108), its cost, an exact worked example on a synthetic
  in-sector channel, and the four pre-registered runs with their commits checked.
* Appendix E: the falsification register, from the same audit functions as tools/make_cap28_figures.py.
* Appendix F: the Riesz-Frechet foundation (Appendix A of the notes) with its numerical checks.

Writes book/es/apendice_{a..f}.qmd and book/en/appendix_{a..f}.qmd. Appendix B needs Qiskit.
"""

import importlib
import inspect
import math
import os
import pkgutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import qang  # noqa: E402
import qang.formulation as F  # noqa: E402


# --------------------------------------------------------------------------- Appendix A
def coef(c):
    for v, s in ((1.0, ""), (-1.0, "-"), (0.5, "\\tfrac12"), (-0.5, "-\\tfrac12"), (1 / math.sqrt(2), "\\tfrac{1}{\\sqrt2}"),
                 (-1 / math.sqrt(2), "-\\tfrac{1}{\\sqrt2}")):
        if abs(c - v) < 1e-9:
            return s
    return "%.4f" % c


def expr(d):
    terms = []
    for k, c in sorted(d.items(), key=lambda kv: kv[0]):
        s = coef(c)
        sign = "-" if s.startswith("-") else "+"
        body = s.lstrip("-")
        terms.append((sign, body + "qg_{\\mathrm{%s}}" % k))
    out = ""
    for i, (sign, t) in enumerate(terms):
        out += ("-" if sign == "-" else "") + t if i == 0 else (" - " if sign == "-" else " + ") + t
    return "$" + out + "$"


def from_zero(U):
    n = int(np.log2(U.shape[0]))
    psi = U[:, 0]
    q = F.local_qg(psi, n, 0)
    return 1000.0 * q[2]


def gate_rows_1q():
    rows = []
    for name, U in F.GATES_1Q.items():
        t = F.gate_table(U)
        rows.append((name, expr(t["X"]), expr(t["Y"]), expr(t["Z"]), "%+.1f" % from_zero(U), F.is_clifford(U),
                     F.conserves_weight(U, 1)))
    return rows


def gate_rows_multi():
    rows = []
    for name, U in F.GATES_MULTI.items():
        n = int(np.log2(U.shape[0]))
        t = F.gate_table(U)
        r = F.gate_radius_class(U)
        z = "; ".join("%s → %s" % ("qg_{\\mathrm{%s}}" % k if False else "$qg_{\\mathrm{%s}}$" % k, expr(v))
                      for k, v in t.items() if "Z" in k)
        rows.append((name, n, z, F.is_clifford(U), F.conserves_weight(U, n), r["class"], r["max deficit"]))
    return rows


TXT_A = {
    "es": dict(
        file=os.path.join(ROOT, "book", "es", "apendice_a.qmd"),
        title="# Catálogo de las 15 compuertas fundamentales en representación qg {#sec-apendice-a}",
        intro="""Este apéndice reúne la acción de las compuertas de `qang.formulation` sobre las lecturas qg. Una compuerta $U$
transforma cada lectura de Pauli por conjugación,

$$
qg'_P = qg_{U^\\dagger P U},
$$

así que alcanza con decir en qué se convierte cada lectura local. Las tablas no están escritas a mano: las genera
`tools/make_appendices.py` a partir de `formulation.gate_table`, `is_clifford`, `conserves_weight` y `gate_radius_class`.
Si el código cambia, se regeneran. En las cadenas de Pauli, la letra de la izquierda es el qubit 0, que es el control de
CX y de CZ, los dos controles de Toffoli y el control de Fredkin.

**Por qué son 15.** El módulo define 16 matrices: seis fijas de un qubit, cuatro paramétricas y seis de varios qubits.
$R_Z(\\varphi)$ y la fase $P(\\varphi)$ difieren sólo en una fase global, $P(\\varphi)=e^{i\\varphi/2}R_Z(\\varphi)$, así que
actúan igual sobre todas las lecturas y en qg cuentan como una.""",
        h1="## Compuertas fijas de un qubit {#sec-a-fijas}",
        t1_head="| compuerta | $qg'_X$ | $qg'_Y$ | $qg'_Z$ | desde $\\lvert0\\rangle$ (m-qg) | Clifford | conserva el peso |",
        t1_cap=": Compuertas fijas de un qubit. {#tbl-a-fijas}",
        after1="""Las compuertas de Clifford permutan las lecturas con signo, y $T$ las mezcla. Las diagonales ($Z$, $S$, $T$) dejan
quieta $qg_Z$: no mueven poblaciones, sólo la fase, que se lee en $qg_X$ y $qg_Y$. Conservar el peso es lo que permite usar el
filtro de los capítulos 21 y 22.""",
        h2="## Compuertas paramétricas {#sec-a-parametricas}",
        t2="""| compuerta | $qg'_X$ | $qg'_Y$ | $qg'_Z$ | desde $\\lvert0\\rangle$ |
|---|---|---|---|---|
| $R_X(\\theta)$ | $qg_X$ | $\\cos\\theta\\,qg_Y-\\sin\\theta\\,qg_Z$ | $\\sin\\theta\\,qg_Y+\\cos\\theta\\,qg_Z$ | $m=1000\\cos\\theta$ |
| $R_Y(\\theta)$ | $\\cos\\theta\\,qg_X+\\sin\\theta\\,qg_Z$ | $qg_Y$ | $-\\sin\\theta\\,qg_X+\\cos\\theta\\,qg_Z$ | $m=1000\\cos\\theta$ |
| $R_Z(\\varphi)$, $P(\\varphi)$ | $\\cos\\varphi\\,qg_X-\\sin\\varphi\\,qg_Y$ | $\\sin\\varphi\\,qg_X+\\cos\\varphi\\,qg_Y$ | $qg_Z$ | $m=+1000$ |

: Compuertas paramétricas; el script comprueba cada coeficiente con $\\theta=0{,}7$. {#tbl-a-parametricas}""",
        after2="""Desde $|0\\rangle$, $R_X(\\theta)$ y $R_Y(\\theta)$ son la ecuación de probabilidad del capítulo 1: el ángulo de la
compuerta es el ángulo qang y la lectura es $m=1000\\cos\\theta$. $R_Z$ y $P$ no cambian la lectura en $Z$. Ninguna rotación
alrededor de $X$ o de $Y$ conserva el peso: mueve poblaciones entre $|0\\rangle$ y $|1\\rangle$. Por eso las capas que se usan
con el filtro son de rotaciones de Givens (RBS) entre pares, no de rotaciones de un qubit.""",
        h3="## Compuertas de varios qubits {#sec-a-multi}",
        t3_head="| compuerta | qubits | qué pasa con cada $qg_Z$ local | Clifford | conserva el peso | radio | déficit máximo |",
        t3_cap=": Compuertas de varios qubits: la transformación de las lecturas $Z$ locales (las de $X$ e $Y$ están en `gate_table`). {#tbl-a-multi}",
        classes={"entangling": "entrelaza", "permutes": "permuta", "preserves": "conserva"},
        after3="""- **Radio.** "Conserva": todo radio local queda en 1 con entradas producto puras. "Permuta": los radios se intercambian
  (SWAP). "Entrelaza": alguna entrada producto termina con déficit $1-r_q^2>0$. El déficit máximo, 1, lleva un qubit al centro
  de la esfera (capítulo 5).
- **Sobre estados de la base**, ninguna de estas compuertas crea déficit: actúan como permutaciones clásicas.
- **Conservan el peso** CZ, SWAP, iSWAP y Fredkin. CX y Toffoli no, porque cambian el bit objetivo. Un circuito armado sólo con
  las primeras tiene un sector de peso fijo y admite el filtro qg.
- Toffoli y Fredkin no son de Clifford: una lectura local pasa a ser una mezcla de cuatro cadenas con coeficientes $\\pm\\tfrac12$.""",
        yes="sí", no="no",
    ),
    "en": dict(
        file=os.path.join(ROOT, "book", "en", "appendix_a.qmd"),
        title="# Catalogue of the 15 fundamental gates in qg representation {#sec-appendix-a}",
        intro="""This appendix collects the action of the gates of `qang.formulation` on the qg readings. A gate $U$ transforms each
Pauli reading by conjugation,

$$
qg'_P = qg_{U^\\dagger P U},
$$

so it is enough to say what each local reading becomes. The tables are not written by hand: `tools/make_appendices.py`
generates them from `formulation.gate_table`, `is_clifford`, `conserves_weight` and `gate_radius_class`. If the code changes,
they are regenerated. In Pauli strings the leftmost letter is qubit 0, which is the control of CX and CZ, both controls of
Toffoli and the control of Fredkin.

**Why fifteen.** The module defines 16 matrices: six fixed single-qubit gates, four parametric ones and six multi-qubit
gates. $R_Z(\\varphi)$ and the phase gate $P(\\varphi)$ differ only by a global phase, $P(\\varphi)=e^{i\\varphi/2}R_Z(\\varphi)$,
so they act identically on every reading and count as one in qg.""",
        h1="## Fixed single-qubit gates {#sec-a-fixed}",
        t1_head="| gate | $qg'_X$ | $qg'_Y$ | $qg'_Z$ | from $\\lvert0\\rangle$ (m-qg) | Clifford | conserves weight |",
        t1_cap=": Fixed single-qubit gates. {#tbl-a-fixed}",
        after1="""Clifford gates permute the readings with signs, and $T$ mixes them. The diagonal gates ($Z$, $S$, $T$) leave $qg_Z$
alone: they do not move populations, only the phase, which is read in $qg_X$ and $qg_Y$. Conserving the weight is what allows
the filter of Chapters 21 and 22.""",
        h2="## Parametric gates {#sec-a-parametric}",
        t2="""| gate | $qg'_X$ | $qg'_Y$ | $qg'_Z$ | from $\\lvert0\\rangle$ |
|---|---|---|---|---|
| $R_X(\\theta)$ | $qg_X$ | $\\cos\\theta\\,qg_Y-\\sin\\theta\\,qg_Z$ | $\\sin\\theta\\,qg_Y+\\cos\\theta\\,qg_Z$ | $m=1000\\cos\\theta$ |
| $R_Y(\\theta)$ | $\\cos\\theta\\,qg_X+\\sin\\theta\\,qg_Z$ | $qg_Y$ | $-\\sin\\theta\\,qg_X+\\cos\\theta\\,qg_Z$ | $m=1000\\cos\\theta$ |
| $R_Z(\\varphi)$, $P(\\varphi)$ | $\\cos\\varphi\\,qg_X-\\sin\\varphi\\,qg_Y$ | $\\sin\\varphi\\,qg_X+\\cos\\varphi\\,qg_Y$ | $qg_Z$ | $m=+1000$ |

: Parametric gates; the script checks each coefficient at $\\theta=0.7$. {#tbl-a-parametric}""",
        after2="""From $|0\\rangle$, $R_X(\\theta)$ and $R_Y(\\theta)$ are the Chapter 1 probability equation: the gate angle is the qang
angle and the reading is $m=1000\\cos\\theta$. $R_Z$ and $P$ do not change the $Z$ reading. No rotation about $X$ or $Y$
conserves the weight: it moves population between $|0\\rangle$ and $|1\\rangle$. That is why the layers used with the filter
are made of Givens (RBS) rotations between pairs, not of single-qubit rotations.""",
        h3="## Multi-qubit gates {#sec-a-multi}",
        t3_head="| gate | qubits | what happens to each local $qg_Z$ | Clifford | conserves weight | radius | maximal deficit |",
        t3_cap=": Multi-qubit gates: how the local $Z$ readings transform (those of $X$ and $Y$ are in `gate_table`). {#tbl-a-multi}",
        classes={"entangling": "entangles", "permutes": "permutes", "preserves": "preserves"},
        after3="""- **Radius.** "Preserves": every local radius stays 1 for pure product inputs. "Permutes": the radii are exchanged (SWAP).
  "Entangles": some product input ends with deficit $1-r_q^2>0$. The maximal deficit, 1, takes a qubit to the centre of the
  sphere (Chapter 5).
- **On basis states**, none of these gates creates a deficit: they act as classical permutations.
- **CZ, SWAP, iSWAP and Fredkin conserve the weight.** CX and Toffoli do not, because they flip the target bit. A circuit made
  only of the former has a fixed weight sector and admits the qg filter.
- Toffoli and Fredkin are not Clifford: a local reading becomes a mixture of four strings with coefficients $\\pm\\tfrac12$.""",
        yes="yes", no="no",
    ),
}


def check_parametric():
    t = 0.7
    c, s = math.cos(t), math.sin(t)
    expect = {
        "Rx": {"X": {"X": 1}, "Y": {"Y": c, "Z": -s}, "Z": {"Y": s, "Z": c}},
        "Ry": {"X": {"X": c, "Z": s}, "Y": {"Y": 1}, "Z": {"X": -s, "Z": c}},
        "Rz": {"X": {"X": c, "Y": -s}, "Y": {"X": s, "Y": c}, "Z": {"Z": 1}},
        "P": {"X": {"X": c, "Y": -s}, "Y": {"X": s, "Y": c}, "Z": {"Z": 1}},
    }
    for name, e in expect.items():
        got = F.gate_table(F.GATES_PARAMETRIC[name](t))
        for k in "XYZ":
            for kk, v in e[k].items():
                assert abs(got[k][kk] - v) < 1e-9, (name, k, kk)
        for which in ("Rx", "Ry"):
            U = F.GATES_PARAMETRIC[which](t)
            assert abs(from_zero(U) - 1000 * c) < 1e-9
    U = F.GATES_PARAMETRIC["P"](t) @ np.linalg.inv(F.GATES_PARAMETRIC["Rz"](t))
    assert np.allclose(U, np.exp(1j * t / 2) * np.eye(2))


def write_a():
    check_parametric()
    r1, r3 = gate_rows_1q(), gate_rows_multi()
    for lang, tx in TXT_A.items():
        yn = lambda b: tx["yes"] if b else tx["no"]  # noqa: E731
        lines = [tx["title"], "", tx["intro"], "", tx["h1"], "", tx["t1_head"], "|---|---|---|---|---|---|---|"]
        for name, x, y, z, m, cl, w in r1:
            lines.append("| $%s$ | %s | %s | %s | %s | %s | %s |" % (name, x, y, z, m.replace(".", ",") if lang == "es"
                                                                        else m, yn(cl), yn(w)))
        lines += ["", tx["t1_cap"], "", tx["after1"], "", tx["h2"], "", tx["t2"], "", tx["after2"], "", tx["h3"], "",
                  tx["t3_head"], "|---|---|---|---|---|---|---|"]
        for name, n, z, cl, w, cls, dmax in r3:
            d = ("%.0f" % dmax)
            lines.append("| %s | %d | %s | %s | %s | %s | %s |" % (name, n, z, yn(cl), yn(w), tx["classes"][cls], d))
        lines += ["", tx["t3_cap"], "", tx["after3"], ""]
        open(tx["file"], "w", encoding="utf-8").write("\n".join(lines))
    print("appendix A: %d fixed, 4 parametric, %d multi-qubit gates" % (len(r1), len(r3)))


# --------------------------------------------------------------------------- Appendix D
DESC = {
    "algebra": ("Álgebra cerrada en qg: la compuerta universal, reflexiones y probabilidades de transición; subálgebra de Cartan (§166).",
                "Closed-form qg algebra: the universal gate, reflections and transition probabilities; Cartan subalgebra (§166)."),
    "algorithms": ("Algoritmos de libro de texto en Qiskit: teleportación, codificación superdensa, Grover.",
                   "Textbook algorithms in Qiskit: teleportation, superdense coding, Grover."),
    "angular": ("Los cuatro números cuánticos de un electrón en qg (§155).", "The four quantum numbers of an electron in qg (§155)."),
    "ansatze": ("Ansätze variacionales reutilizables, incluidos los de capa qg.", "Reusable variational ansätze, including qg layers."),
    "channels": ("Canales de un qubit como mapas afines del vector qg (§162).", "Single-qubit channels as affine maps of the qg vector (§162)."),
    "circuits": ("Constructores de circuitos (Bell, GHZ, W, grafo, Dicke) y sus lecturas.",
                 "Circuit constructors (Bell, GHZ, W, graph, Dicke) and their readings."),
    "cirq_gate": ("qg como compuertas nativas de Cirq.", "qg as native Cirq gates."),
    "clock_comparison": ("Fase diferencial entre dos relojes y el corrimiento gravitatorio.",
                         "Differential phase between two clocks and the gravitational shift."),
    "complementarity": ("Dualidad onda–partícula e incertidumbre entrópica en qg.", "Wave–particle duality and entropic uncertainty in qg."),
    "core": ("La unidad qang (qg) y la clase `Qang`.", "The qang (qg) unit and the `Qang` class."),
    "fisher": ("Información de Fisher, reglas de desplazamiento y gradiente natural en qg (§159).",
               "Fisher information, shift rules and the natural gradient in qg (§159)."),
    "formulation": ("Las 15 compuertas y los 14 algoritmos en lenguaje qg; el radio local (apéndice A, capítulo 5).",
                    "The 15 gates and 14 algorithms in qg language; the local radius (Appendix A, Chapter 5)."),
    "geometric": ("Fase geométrica (Berry, Pancharatnam) en unidades qg.", "Geometric phase (Berry, Pancharatnam) in qg units."),
    "gradients": ("La singularidad de coordenadas en los polos y sus regularizaciones.", "The coordinate singularity at the poles and its regularizations."),
    "ibm": ("Lecturas qg en IBM Runtime: pase de preservación del peso, corrección de lectura, cadenas, ZNE (§137).",
            "qg readings on IBM Runtime: weight-preservation pass, readout correction, chains, ZNE (§137)."),
    "integrations": ("qang dentro de otras librerías: un submódulo por librería (capítulo 35).",
                     "qang inside other libraries: one submodule per library (Chapter 35)."),
    "interop": ("Lecturas qg de los resultados de cualquier SDK con un solo orden de qubits (§130).",
                "qg readings from any SDK's results with one qubit order (§130)."),
    "kernels": ("Núcleos cuánticos con el filtro qg (§161).", "Quantum kernels with the qg filter (§161)."),
    "knitting": ("Costo de muestreo del corte de circuitos.", "Sampling cost of circuit cutting."),
    "mixed": ("La unidad qang para estados mixtos y POVMs.", "The qang unit for mixed states and POVMs."),
    "multiqubit": ("La unidad qang para registros de varios qubits: marginales, entropías, correlaciones.",
                   "The qang unit for multi-qubit registers: marginals, entropies, correlations."),
    "neutrino": ("Oscilaciones de sabor de neutrinos en qg (§151; capítulos 9–11).",
                 "Neutrino flavour oscillations in qg (§151; Chapters 9–11)."),
    "pennylane_gate": ("qg como operaciones nativas de PennyLane, diferenciables; lectura filtrada.",
                       "qg as native, differentiable PennyLane operations; filtered readout."),
    "phase": ("La unidad de fase diferida qg_Φ.", "The deferred qg_Φ phase unit."),
    "polarization": ("Luz polarizada en unidades qg: Stokes, Jones, Mueller (§83; capítulo 6).",
                     "Polarized light in qg units: Stokes, Jones, Mueller (§83; Chapter 6)."),
    "postselection": ("Teleportación con posselección: bucles temporales y decodificación de agujeros negros (capítulos 14–15).",
                      "Teleportation with postselection: closed timelike curves and black-hole decoding (Chapters 14–15)."),
    "probability": ("La ecuación de probabilidad qang y el mili-qang (§167; capítulo 1).",
                    "The qang probability equation and the milli-qang (§167; Chapter 1)."),
    "proportions": ("El ángulo qang como transformación arcoseno de una proporción.", "The qang angle as the arcsine transformation of a proportion."),
    "pulse_area": ("Área de pulso de compuertas fraccionarias: la regla del área mínima (§160).",
                   "Pulse area of fractional gates: the minimal-area rule (§160)."),
    "qec": ("El código de bit-flip de 3 qubits con síndromes leídos en qg.", "The 3-qubit bit-flip code with syndromes read in qg."),
    "qiskit_gate": ("qg como compuerta nativa de Qiskit.", "qg as a native Qiskit gate."),
    "qiskit_ml": ("La QNN de peso dentro de qiskit-machine-learning (§140).", "The weight QNN inside qiskit-machine-learning (§140)."),
    "qml": ("Redes neuronales cuánticas que conservan el peso, con y sin filtro (§75–§80; capítulo 20).",
            "Weight-conserving quantum neural networks, with and without the filter (§75–§80; Chapter 20)."),
    "qpon": ("Teleportación por una red óptica pasiva, variables discretas (§157; capítulos 12–13).",
             "Teleportation through a passive optical network, discrete variables (§157; Chapters 12–13)."),
    "qsp": ("Procesamiento cuántico de señales en qg (§134).", "Quantum signal processing in qg (§134)."),
    "sectors": ("Herramientas de sectores conservados para el filtro qg y el eco (§20, §60, §66, §68; apéndice C).",
                "Conserved-sector tools for the qg filter and the echo (§20, §60, §66, §68; Appendix C)."),
    "statistics": ("Propagación de errores entre probabilidad y ángulo, estimadores e intervalos.",
                   "Error propagation between probability and angle, estimators and intervals."),
    "transformations": ("Transformaciones cerradas entre estados parametrizados en qang.", "Closed-form transformations between qang-parameterized states."),
    "varqte": ("Evolución variacional en tiempo real (McLachlan) con ansatz que conserva el peso (§162).",
               "Variational real-time evolution (McLachlan) with a weight-preserving ansatz (§162)."),
}

TXT_D = {
    "es": dict(
        file=os.path.join(ROOT, "book", "es", "apendice_d.qmd"),
        title="# Referencia rápida de la API de qang 0.13.0 {#sec-apendice-d}",
        intro="""Esta es la lista de todos los módulos del paquete `qang` en la versión %s, con sus nombres públicos: los de
`__all__` o, si el módulo no lo define, las funciones y clases públicas que declara. La genera `tools/make_appendices.py`
importando cada módulo, así que la lista es la del código y no una copia. La firma y la documentación de cada nombre están en
`help(qang.<módulo>.<nombre>)`.

- `import qang` sólo necesita NumPy. Los módulos de SDK (`qiskit_gate`, `cirq_gate`, `pennylane_gate`, `ibm`, `qiskit_ml`,
  `qec`, `circuits`, `algorithms`) y los de `qang.integrations` importan su librería cuando se los usa.
- No hay módulo `crypto`: lo criptográfico del libro está en los ejemplos (capítulos 16–18). Tampoco existe todavía `qang/iqm.py`
  (capítulo 31): el hardware de IQM se usa a través de `qang.interop` y los ejemplos.
- La columna "nombres" da la cantidad de nombres públicos, y la lista completa sigue a la tabla.""",
        head="| módulo | qué hace | nombres |", cap=": Los módulos de qang %s. {#tbl-d-modulos}",
        h_full="## Nombres públicos por módulo {#sec-d-nombres}",
        h_int="## `qang.integrations` {#sec-d-integraciones}",
        int_intro="Cada submódulo importa sólo su librería (capítulo 35); el extra de `pip` va entre paréntesis.",
        failed="**No se pudieron importar en el entorno del script:** %s.",
    ),
    "en": dict(
        file=os.path.join(ROOT, "book", "en", "appendix_d.qmd"),
        title="# Quick reference of the qang 0.13.0 API {#sec-appendix-d}",
        intro="""This is the list of every module of the `qang` package at version %s, with its public names: those in `__all__`
or, if the module does not define it, the public functions and classes it declares. `tools/make_appendices.py` generates it by
importing each module, so the list is the code's own and not a copy. The signature and documentation of each name are in
`help(qang.<module>.<name>)`.

- `import qang` needs only NumPy. The SDK modules (`qiskit_gate`, `cirq_gate`, `pennylane_gate`, `ibm`, `qiskit_ml`, `qec`,
  `circuits`, `algorithms`) and those of `qang.integrations` import their library when used.
- There is no `crypto` module: the book's cryptography is in the examples (Chapters 16–18). Nor does `qang/iqm.py` exist yet
  (Chapter 31): IQM hardware is used through `qang.interop` and the examples.
- The "names" column gives the number of public names, and the full list follows the table.""",
        head="| module | what it does | names |", cap=": The modules of qang %s. {#tbl-d-modules}",
        h_full="## Public names by module {#sec-d-names}",
        h_int="## `qang.integrations` {#sec-d-integrations}",
        int_intro="Each submodule imports only its library (Chapter 35); the `pip` extra is in parentheses.",
        failed="**Could not be imported in the script's environment:** %s.",
    ),
}

EXTRAS = {"mitiq": "mitiq", "qiskit_experiments": "experiments", "qiskit_dynamics": "dynamics", "qutip": "qutip",
          "stim": "stim", "sklearn": "sklearn", "jax": "jax", "torch": "torch", "qibo": "qibo", "pytket": "pytket",
          "braket": "braket", "cudaq": "cudaq"}


def public_names(mod):
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n, o in vars(mod).items() if not n.startswith("_") and (inspect.isfunction(o) or inspect.isclass(o))
                 and getattr(o, "__module__", "") == mod.__name__]
    return list(names)


def module_names(modname):
    """Public names without importing the module (for integrations whose library is absent): read __all__ from source."""
    import ast
    path = os.path.join(ROOT, *modname.split(".")) + ".py"
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            return [e.value for e in node.value.elts]
    return []


def write_d():
    mods = sorted(m.name for m in pkgutil.iter_modules(qang.__path__))
    info, failed = {}, []
    for m in mods:
        try:
            info[m] = public_names(importlib.import_module("qang." + m))
        except Exception:  # noqa: BLE001
            failed.append(m)
    missing = set(mods) - set(DESC)
    assert not missing, "describe these modules in DESC: %s" % sorted(missing)
    subs = [s for s in qang.integrations.__all__] if "integrations" in info else []
    sub_names = {s: module_names("qang.integrations." + s) for s in subs}
    for lang, tx in TXT_D.items():
        k = 0 if lang == "es" else 1
        lines = [tx["title"], "", tx["intro"] % qang.__version__, "", tx["head"], "|---|---|---|"]
        for m in mods:
            n = len(info.get(m, []))
            lines.append("| `%s` | %s | %s |" % (m, DESC[m][k], n if m in info else "—"))
        lines += ["", tx["cap"] % qang.__version__, ""]
        if failed:
            lines += [tx["failed"] % ", ".join("`%s`" % f for f in failed), ""]
        lines += [tx["h_full"], ""]
        for m in mods:
            if m in info and m != "integrations":
                lines.append("- **`%s`**: %s" % (m, ", ".join("`%s`" % x for x in info[m])))
        lines += ["", tx["h_int"], "", tx["int_intro"], ""]
        for s in subs:
            lines.append("- **`integrations.%s`** (`qang[%s]`): %s" % (s, EXTRAS[s], ", ".join("`%s`" % x for x in sub_names[s])))
        lines.append("")
        open(tx["file"], "w", encoding="utf-8").write("\n".join(lines))
    print("appendix D: %d modules, %d failed to import %s, %d integrations" % (len(mods), len(failed), failed, len(subs)))


# --------------------------------------------------------------------------- shared helpers for B, C, E, F
FILES = {"b": ("apendice_b", "appendix_b"), "c": ("apendice_c", "appendix_c"), "e": ("apendice_e", "appendix_e"),
         "f": ("apendice_f", "appendix_f")}


def nf(x, spec, lang, math_mode=False):
    s = format(x, spec)
    return s.replace(".", "{,}" if math_mode else ",") if lang == "es" else s


def sci(x, lang):
    """A small error as $a\\cdot10^{b}$ (or 0)."""
    if x == 0:
        return "0"
    m, e = ("%.1e" % x).split("e")
    return "$%s\\cdot10^{%d}$" % (m.replace(".", "{,}") if lang == "es" else m, int(e))


def table_safe(line):
    """A '|' inside $...$ would split a pipe-table cell: write it as \\vert on table rows."""
    if not line.startswith("|"):
        return line
    parts = line.split("$")
    return "$".join(p.replace("|", "\\vert ") if i % 2 else p for i, p in enumerate(parts))


def write(key, lang, lines):
    lines = [table_safe(ln) for ln in lines]
    sub, name = ("es", FILES[key][0]) if lang == "es" else ("en", FILES[key][1])
    open(os.path.join(ROOT, "book", sub, name + ".qmd"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def psi(q, phi=0.0):
    """Pure state with polar bias q = qg_Z and azimuth phi."""
    return np.array([math.sqrt((1 + q) / 2), np.exp(1j * phi) * math.sqrt((1 - q) / 2)])


# --------------------------------------------------------------------------- Appendix B
def check_b(n=2000, seed=166):
    """Every closed form of qang.algebra against direct linear algebra on n random states."""
    import qang.algebra as AL
    from qiskit.circuit.library import UGate

    rng = np.random.default_rng(seed)
    err = dict.fromkeys(("u3", "trans", "house", "diff", "tp", "merid"), 0.0)
    for _ in range(n):
        q, q2 = rng.uniform(-1, 1, 2)
        phi, lam, phi2 = rng.uniform(0, 2 * np.pi, 3)
        U = AL.u3_qang(q, phi, lam)
        err["u3"] = max(err["u3"], np.abs(U - UGate(math.acos(q), phi, lam).to_matrix()).max())
        err["trans"] = max(err["trans"], np.abs(np.abs(U) ** 2 - AL.transition_matrix(q)).max())
        v = psi(q)
        P = np.outer(v, v.conj())
        err["house"] = max(err["house"], np.abs(np.eye(2) - 2 * P - AL.householder_qg(q)).max())
        err["diff"] = max(err["diff"], np.abs(2 * P - np.eye(2) - AL.grover_diffuser_qg(q)).max())
        exact = abs(np.vdot(psi(q, phi), psi(q2, phi2))) ** 2
        err["tp"] = max(err["tp"], abs(exact - AL.transition_probability(q, q2, phi, phi2)))
        mer = (1 + q * q2 + math.sqrt((1 - q * q) * (1 - q2 * q2))) / 2
        err["merid"] = max(err["merid"], abs(abs(np.vdot(psi(q, phi), psi(q2, phi))) ** 2 - mer))
    mats = {(1, 0, 0): np.eye(2), (-1, 0, math.pi): np.array([[0, 1], [1, 0]]), (1, 0, math.pi): np.diag([1, -1]),
            (0, 0, math.pi): np.array([[1, 1], [1, -1]]) / math.sqrt(2)}
    for a, m in mats.items():
        assert np.allclose(AL.u3_qang(*a), m, atol=1e-12), a
    assert max(err.values()) < 1e-12, err
    return err


def gellmann_rows(dmax=4):
    import qang.algebra as AL

    rows = []
    for d in range(2, dmax + 1):
        G = AL.cartan_generators(d)
        assert np.allclose(G @ G.T, 2 * np.eye(d - 1))
        for k in range(1, d):
            N = k * (k + 1) // 2
            ints = G[k - 1] * math.sqrt(N)
            assert np.allclose(ints, np.round(ints))
            pref = "" if N == 1 else "\\tfrac{1}{\\sqrt{%d}}\\," % N
            rows.append((d, k, "$%s\\mathrm{diag}(%s)$" % (pref, ", ".join("%d" % v for v in np.round(ints)))))
    return rows


def check_cartan(seed=3):
    import qang.algebra as AL

    rng = np.random.default_rng(seed)
    e = 0.0
    for d in range(2, 7):
        for _ in range(200):
            p = rng.dirichlet(np.ones(d))
            e = max(e, np.abs(AL.populations_from_cartan(AL.cartan_values(p)) - p).max())
    assert abs(AL.cartan_values([0.7, 0.3])[0] - 0.4) < 1e-15  # d = 2: the single value is qg_Z
    assert e < 1e-14
    return e, AL.cartan_values([0.5, 0.3, 0.2])


def cheb(m):
    return [0] * m + [1]


def cheb_tex(m):
    from numpy.polynomial import chebyshev as C

    c = C.cheb2poly(cheb(m))
    s = ""
    for p in range(m, -1, -1):
        a = int(round(c[p]))
        if a == 0:
            continue
        body = ("" if abs(a) == 1 and p > 0 else str(abs(a))) + ("q" if p == 1 else "q^{%d}" % p if p > 1 else "")
        s += ("-" if a < 0 else "") + body if not s else (" - " if a < 0 else " + ") + body
    return "$" + s + "$"


def check_grover(seed=37):
    """qg after k Grover iterations is T_{2k+1}(qg_0); the one-shot Fisher information is m^2/(1 - qg_0^2)."""
    import qang.algebra as AL
    from numpy.polynomial import chebyshev as C

    rng = np.random.default_rng(seed)
    Z = np.diag([1.0, -1.0])
    e = 0.0
    for _ in range(200):
        q0 = rng.uniform(-1, 1)
        G = AL.grover_diffuser_qg(q0) @ Z
        v = psi(q0)
        for k in range(6):
            e = max(e, abs(np.real(np.vdot(v, Z @ v)) - C.chebval(q0, cheb(2 * k + 1))))
            v = G @ v
    ef, h = 0.0, 1e-6
    for m in (1, 3, 5, 7, 9):
        for q0 in (-0.6, 0.1, 0.4, 0.8):
            T = C.chebval(q0, cheb(m))
            dp = (C.chebval(q0 + h, cheb(m)) - C.chebval(q0 - h, cheb(m))) / (4 * h)
            ef = max(ef, abs(dp ** 2 / ((1 + T) / 2 * (1 - T) / 2) / (m * m / (1 - q0 * q0)) - 1))
    assert e < 1e-12 and ef < 1e-6, (e, ef)
    return e, ef


def write_b():
    from numpy.polynomial import chebyshev as C

    err = check_b()
    gm = gellmann_rows()
    ec, qutrit = check_cartan()
    eg, ef = check_grover()
    for lang in ("es", "en"):
        t = (lambda es_, en_: es_) if lang == "es" else (lambda es_, en_: en_)  # noqa: E731
        L = [t("# Fórmulas de transición, matrices de Gell-Mann y polinomios de Chebyshev {#sec-apendice-b}",
               "# Transition formulas, Gell-Mann matrices and Chebyshev polynomials {#sec-appendix-b}"), ""]
        L += [t("""Este apéndice junta en un lugar las fórmulas cerradas que usan los capítulos 2 y 3: la compuerta universal y las
reflexiones escritas en $q=qg_Z$, la probabilidad de transición, la generalización de $qg_Z$ a qudits con las matrices de
Gell-Mann diagonales y la trayectoria de Grover como polinomio de Chebyshev (notas, §37 y §166). No hay resultados nuevos.
Lo que agrega es la comprobación: `tools/make_appendices.py` evalúa cada fórmula de `qang.algebra` contra el álgebra lineal
directa en 2000 estados al azar y escribe acá el error máximo. Si una fórmula estuviera mal, el script se detiene.""",
                """This appendix gathers in one place the closed forms used in Chapters 2 and 3: the universal gate and the
reflections written in $q=qg_Z$, the transition probability, the generalization of $qg_Z$ to qudits with the diagonal
Gell-Mann matrices, and the Grover trajectory as a Chebyshev polynomial (notes, §37 and §166). There are no new results.
What it adds is the check: `tools/make_appendices.py` evaluates each formula of `qang.algebra` against direct linear
algebra on 2000 random states and writes the maximal error here. If a formula were wrong, the script would stop."""), ""]
        L += [t("## Fórmulas de un qubit {#sec-b-un-qubit}", "## Single-qubit formulas {#sec-b-single}"), ""]
        L += [t("Con $q=qg_Z=\\cos\\theta$ vale $\\cos\\tfrac\\theta2=\\sqrt{(1+q)/2}$ y $\\sin\\tfrac\\theta2=\\sqrt{(1-q)/2}$. "
                "El estado puro de sesgo polar $q$ y azimut $\\varphi$ es",
                "With $q=qg_Z=\\cos\\theta$, $\\cos\\tfrac\\theta2=\\sqrt{(1+q)/2}$ and $\\sin\\tfrac\\theta2=\\sqrt{(1-q)/2}$. "
                "The pure state with polar bias $q$ and azimuth $\\varphi$ is"), "",
              "$$", "|\\psi(q,\\varphi)\\rangle=\\sqrt{\\tfrac{1+q}{2}}\\,|0\\rangle+e^{i\\varphi}\\sqrt{\\tfrac{1-q}{2}}\\,|1\\rangle,"
              "\\qquad \\vec r=\\big(\\sqrt{1-q^2}\\cos\\varphi,\\ \\sqrt{1-q^2}\\sin\\varphi,\\ q\\big).", "$$", "",
              t("Las fórmulas son:", "The formulas are:"), "",
              "$$", "U_3^{qg}(q,\\varphi,\\lambda)=\\begin{pmatrix}\\sqrt{\\tfrac{1+q}{2}} & -e^{i\\lambda}\\sqrt{\\tfrac{1-q}{2}}\\\\"
              " e^{i\\varphi}\\sqrt{\\tfrac{1-q}{2}} & e^{i(\\varphi+\\lambda)}\\sqrt{\\tfrac{1+q}{2}}\\end{pmatrix},\\qquad"
              " \\big|U_{ij}\\big|^2=\\begin{pmatrix}\\tfrac{1+q}{2} & \\tfrac{1-q}{2}\\\\ \\tfrac{1-q}{2} & \\tfrac{1+q}{2}\\end{pmatrix}",
              "$$ {#eq-b-u3}", "",
              "$$", "I-2|\\psi\\rangle\\langle\\psi|=-q\\,Z-\\sqrt{1-q^2}\\,X,\\qquad 2|\\psi\\rangle\\langle\\psi|-I=q\\,Z+\\sqrt{1-q^2}\\,X",
              "$$ {#eq-b-reflexiones}", "",
              "$$", "|\\langle\\psi_1|\\psi_2\\rangle|^2=\\frac{1+\\vec r_1\\cdot\\vec r_2}{2}"
              "\\ \\xrightarrow{\\ \\varphi_1=\\varphi_2\\ }\\ \\frac{1+q_1q_2+\\sqrt{(1-q_1^2)(1-q_2^2)}}{2}", "$$ {#eq-b-transicion}", "",
              t("| fórmula | función de `qang.algebra` | comparada con | error máximo |",
                "| formula | `qang.algebra` function | compared with | maximal error |"), "|---|---|---|---|",
              t("| compuerta universal | `u3_qang` | `UGate(arccos q, φ, λ)` de Qiskit | %s |",
                "| universal gate | `u3_qang` | Qiskit's `UGate(arccos q, φ, λ)` | %s |") % sci(err["u3"], lang),
              t("| matriz de transición | `transition_matrix` | $|U_{ij}|^2$ | %s |", "| transition matrix | `transition_matrix` | $|U_{ij}|^2$ | %s |")
              % sci(err["trans"], lang),
              "| Householder | `householder_qg` | $I-2|\\psi\\rangle\\langle\\psi|$ | %s |" % sci(err["house"], lang),
              t("| difusor de Grover | `grover_diffuser_qg` | $2|\\psi\\rangle\\langle\\psi|-I$ | %s |",
                "| Grover diffuser | `grover_diffuser_qg` | $2|\\psi\\rangle\\langle\\psi|-I$ | %s |") % sci(err["diff"], lang),
              t("| probabilidad de transición | `transition_probability` | $|\\langle\\psi_1|\\psi_2\\rangle|^2$ | %s |",
                "| transition probability | `transition_probability` | $|\\langle\\psi_1|\\psi_2\\rangle|^2$ | %s |") % sci(err["tp"], lang),
              t("| misma, sobre un meridiano | (forma cerrada) | $|\\langle\\psi_1|\\psi_2\\rangle|^2$ | %s |",
                "| same, on one meridian | (closed form) | $|\\langle\\psi_1|\\psi_2\\rangle|^2$ | %s |") % sci(err["merid"], lang), "",
              t(": Comprobación de las fórmulas en 2000 estados al azar. {#tbl-b-formulas}",
                ": Check of the formulas on 2000 random states. {#tbl-b-formulas}"), "",
              t("""La matriz de transición es doblemente estocástica, racional en $q$ y no depende de las fases: la probabilidad que
deja una compuerta se lee sin trigonometría. Con $(q,\\varphi,\\lambda)=(1,0,0)$, $(-1,0,\\pi)$, $(1,0,\\pi)$ y $(0,0,\\pi)$,
@eq-b-u3 da exactamente $I$, $X$, $Z$ y $H$; el script lo comprueba. El reflector y el difusor son el mismo operador con
signo opuesto, y los dos son combinaciones lineales de $Z$ y $X$ con coeficientes $q$ y $\\sqrt{1-q^2}$ (capítulo 3).""",
                """The transition matrix is doubly stochastic, rational in $q$ and independent of the phases: the probability a gate
leaves is read without trigonometry. With $(q,\\varphi,\\lambda)=(1,0,0)$, $(-1,0,\\pi)$, $(1,0,\\pi)$ and $(0,0,\\pi)$,
@eq-b-u3 gives exactly $I$, $X$, $Z$ and $H$; the script checks it. The reflector and the diffuser are the same operator
with opposite sign, and both are linear combinations of $Z$ and $X$ with coefficients $q$ and $\\sqrt{1-q^2}$ (Chapter 3)."""), ""]
        L += [t("## Matrices de Gell-Mann diagonales {#sec-b-gellmann}", "## Diagonal Gell-Mann matrices {#sec-b-gellmann}"), "",
              t("""Un qudit de $d$ niveles tiene $d^2-1$ generadores de $\\mathfrak{su}(d)$, las matrices de Gell-Mann generalizadas.
Sólo $d-1$ son diagonales: forman la subálgebra de Cartan y son las únicas que se leen midiendo en la base computacional.
Son""",
                """A $d$-level qudit has $d^2-1$ generators of $\\mathfrak{su}(d)$, the generalized Gell-Mann matrices. Only $d-1$ of
them are diagonal: they form the Cartan subalgebra and are the only ones read by measuring in the computational basis.
They are"""), "",
              "$$", "\\lambda_k=\\sqrt{\\tfrac{2}{k(k+1)}}\\;\\mathrm{diag}(\\underbrace{1,\\dots,1}_{k},-k,0,\\dots,0),"
              "\\qquad k=1,\\dots,d-1,\\qquad \\mathrm{Tr}(\\lambda_j\\lambda_k)=2\\delta_{jk},", "$$ {#eq-b-cartan}", "",
              t("y sus valores esperados son lineales en las poblaciones $p$ y las determinan:",
                "and their expectation values are linear in the populations $p$ and determine them:"), "",
              "$$", "\\langle\\lambda_k\\rangle=\\sum_i p_i\\,(\\lambda_k)_{ii},\\qquad p=\\frac1d+\\frac12\\sum_{k=1}^{d-1}"
              "\\langle\\lambda_k\\rangle\\,\\mathrm{diag}(\\lambda_k).", "$$ {#eq-b-poblaciones}", "",
              t("| $d$ | $k$ | $\\lambda_k$ |", "| $d$ | $k$ | $\\lambda_k$ |"), "|---|---|---|"]
        L += ["| %d | %d | %s |" % r for r in gm]
        L += ["", t(": Las matrices diagonales hasta $d=4$, generadas por `cartan_generators`. {#tbl-b-gellmann}",
                    ": The diagonal matrices up to $d=4$, generated by `cartan_generators`. {#tbl-b-gellmann}"), "",
              t("""Para $d=2$ la única es $\\lambda_1=Z$ y su valor es $qg_Z$. Para $d=3$ son las $\\lambda_3$ y $\\lambda_8$ de Gell-Mann.
El script recupera las poblaciones desde los $d-1$ valores para $d=2,\\dots,6$, 200 distribuciones al azar cada una, con error
máximo %s. Un ejemplo: el qutrit con $p=(0{,}5;\\ 0{,}3;\\ 0{,}2)$ lee $\\langle\\lambda_3\\rangle=%s$ y
$\\langle\\lambda_8\\rangle=%s$.

Esto es una definición y una identidad. Si leer más niveles le da a qang alguna ventaja se preguntó en §70 para la fuga, y
no se encontró ninguna (capítulo 2).""",
                """For $d=2$ the only one is $\\lambda_1=Z$ and its value is $qg_Z$. For $d=3$ they are Gell-Mann's $\\lambda_3$ and
$\\lambda_8$. The script recovers the populations from the $d-1$ values for $d=2,\\dots,6$, 200 random distributions each,
with maximal error %s. An example: the qutrit with $p=(0.5, 0.3, 0.2)$ reads $\\langle\\lambda_3\\rangle=%s$ and
$\\langle\\lambda_8\\rangle=%s$.

This is a definition and an identity. Whether reading more levels gives qang any advantage was asked in §70 for leakage,
and none was found (Chapter 2).""") % (sci(ec, lang), nf(qutrit[0], ".4f", lang, True), nf(qutrit[1], ".4f", lang, True)), ""]
        L += [t("## Polinomios de Chebyshev y la trayectoria de Grover {#sec-b-chebyshev}",
                "## Chebyshev polynomials and the Grover trajectory {#sec-b-chebyshev}"), "",
              t("""Con amplitud $a=\\sin^2t$ y la bandera bueno/malo leída como un qubit, $qg_0=1-2a=\\cos2t$. Cada iteración de Grover
suma $2t$ al ángulo, así que tras $k$ iteraciones""",
                """With amplitude $a=\\sin^2t$ and the good/bad flag read as a qubit, $qg_0=1-2a=\\cos2t$. Each Grover iteration adds
$2t$ to the angle, so after $k$ iterations"""), "",
              "$$", "qg_k=T_{2k+1}(qg_0),\\qquad T_m(\\cos x)=\\cos mx,", "$$ {#eq-b-grover}", "",
              t("""con $T_m$ el polinomio de Chebyshev de primera especie. Como $T_m'=m\\,U_{m-1}$ y $1-T_m^2=(1-q^2)\\,U_{m-1}^2$, la
información de Fisher de un disparo a profundidad $m$ es""",
                """with $T_m$ the Chebyshev polynomial of the first kind. Since $T_m'=m\\,U_{m-1}$ and $1-T_m^2=(1-q^2)\\,U_{m-1}^2$, the
one-shot Fisher information at depth $m$ is"""), "",
              "$$", "F_m(qg_0)=\\frac{m^2}{1-qg_0^2},", "$$ {#eq-b-fisher}", "",
              t("""el factor $m^2$ sobre la información de profundidad 1, igual en todo el recorrido. Es la forma qg del conocido $4m^2$
para el ángulo [@suzuki2020].""",
                """the factor $m^2$ over the depth-1 information, the same along the whole fringe. It is the qg form of the known $4m^2$
for the angle [@suzuki2020]."""), "",
              t("| $k$ | $m=2k+1$ | $T_m(q)$ | $F_m/F_1$ | $qg_k$ con $a=0{,}1$ (m-qg) |",
                "| $k$ | $m=2k+1$ | $T_m(q)$ | $F_m/F_1$ | $qg_k$ at $a=0.1$ (m-qg) |"), "|---|---|---|---|---|"]
        for k in range(5):
            m = 2 * k + 1
            L.append("| %d | %d | %s | %d | %s |" % (k, m, cheb_tex(m), m * m, nf(1000 * C.chebval(0.8, cheb(m)), "+.1f", lang)))
        L += ["", t(": Los primeros polinomios de la trayectoria. Con $a=0{,}1$, $qg_0=0{,}8$; el éxito es $-1000$ m-qg. {#tbl-b-chebyshev}",
                    ": The first polynomials of the trajectory. At $a=0.1$, $qg_0=0.8$; success is $-1000$ m-qg. {#tbl-b-chebyshev}"), "",
              t("""El script aplica el difusor de @eq-b-reflexiones y el oráculo $Z$ a 200 estados iniciales al azar, hasta $k=5$, y
encuentra @eq-b-grover con error máximo %s. Comprueba @eq-b-fisher por diferencias finitas para $m=1,\\dots,9$, con error
relativo máximo %s. Lo práctico de §37 está en el capítulo 3: el ruido tiene que entrar en la verosimilitud, y la ventaja de
la estimación de amplitud termina cerca de la profundidad $1/p$.""",
                """The script applies the diffuser of @eq-b-reflexiones and the oracle $Z$ to 200 random initial states, up to $k=5$,
and finds @eq-b-grover with maximal error %s. It checks @eq-b-fisher by finite differences for $m=1,\\dots,9$, with maximal
relative error %s. The practical side of §37 is in Chapter 3: the noise has to enter the likelihood, and the advantage of
amplitude estimation ends near depth $1/p$.""") % (sci(eg, lang), sci(ef, lang)), ""]
        L += [t("## Notas bibliográficas {.unnumbered}", "## Bibliographic notes {.unnumbered}"), "",
              t("""Las matrices son las de @gellmann1962. El reflector es el de @householder1958; la búsqueda de Grover, @grover1996; la
estimación de amplitud, @brassard2002, y su versión por máxima verosimilitud, @suzuki2020. La probabilidad de transición es
la distancia de @wootters1981. Las notas numeradas (§) y el script de este apéndice están en el repositorio [@qangrepo].""",
                """The matrices are those of @gellmann1962. The reflector is that of @householder1958; Grover search, @grover1996;
amplitude estimation, @brassard2002, and its maximum-likelihood version, @suzuki2020. The transition probability is the
distance of @wootters1981. The numbered notes (§) and the script of this appendix are in the repository [@qangrepo].""")]
        write("b", lang, L)
    print("appendix B: max formula error %.1e, Cartan round trip %.1e, Grover %.1e, Fisher %.1e"
          % (max(err.values()), ec, eg, ef))


# --------------------------------------------------------------------------- Appendix C
ECHO_RUNS = [  # §, predictions, commit, passed, outcome (es, en); numbers from RESEARCH_NOTES §105-§108
    ("105", "C1–C4", "052752d", 4, ("peso 1, 5 qubits, cinco dispositivos simulados: el eco quita 38–47 % del error que deja el "
                                    "filtro; filtro y eco, 6–7 veces más exactos que la lectura cruda",
                                    "weight 1, 5 qubits, five simulated devices: the echo removes 38–47% of the error the filter "
                                    "leaves; filter plus echo, 6–7× more accurate than the raw readout")),
    ("106", "L1–L4", "fa8eda6", 4, ("el clasificador de peso 1, 189 entradas: el error del valor de decisión baja 43–56 %; "
                                    "decisiones distintas del modelo sin ruido, 15 → 7 → 3",
                                    "the weight-1 classifier, 189 inputs: the decision-value error falls 43–56%; decisions "
                                    "differing from the noiseless model, 15 → 7 → 3")),
    ("107", "W1–W4", "4d83029", 4, ("peso 2, matriz de 10 × 10: el eco quita 37–43 % del error de $qg_Z$ y 39–48 % del de "
                                    "$qg_{ZZ}$", "weight 2, a 10 × 10 matrix: the echo removes 37–43% of the $qg_Z$ error and "
                                    "39–48% of the $qg_{ZZ}$ error")),
    ("108", "M1–M4", "4ae2674", 0, ("entradas de margen estrecho: el filtro baja las decisiones invertidas de 47 a 17 y el eco "
                                    "no agrega nada (19)", "narrow-margin inputs: the filter cuts flipped decisions from 47 to 17 "
                                    "and the echo adds nothing (19)")),
]


def qg_z_all(p, n):
    """Local qg_Z of every qubit from a full distribution (qubit q is bit q of the index)."""
    idx = np.arange(2 ** n)
    return np.array([np.sum(p * (1 - 2 * ((idx >> q) & 1))) for q in range(n)])


def echo_example(seed=105, leak=0.15, diag=0.90, shots=1000, trials=300):
    """Exact worked example of the protocol on a synthetic in-sector channel K (n = 5, weight 1).

    The forward circuit applies K inside the sector and loses `leak` of the shots to |00000> (T1); the echo applies K
    twice. Returns the qg_Z error of each readout, and at finite shots how often a plain inverse goes negative."""
    from qang.sectors import echo_transfer_matrix, filter_distribution, matrix_power_stochastic, sector_states, unmix_sector

    rng = np.random.default_rng(seed)
    n, w = 5, 1
    st = sector_states(n, w)
    D = len(st)
    K = np.zeros((D, D))
    for j in range(D):
        off = rng.dirichlet(np.ones(D - 1)) * (1 - diag)
        K[:, j] = np.insert(off, j, diag)

    def full(sector_part, lost):
        p = np.zeros(2 ** n)
        p[st] = sector_part * (1 - lost)
        p[0] += lost
        return p

    echo = [full((K @ K)[:, j], leak) for j in range(D)]
    M = echo_transfer_matrix(echo, n, w)
    assert np.allclose(M, K @ K, atol=1e-12)
    A = matrix_power_stochastic(M, 0.5)
    assert np.allclose(A, K, atol=1e-9)  # the principal root of the echo is the forward channel
    x = rng.dirichlet(np.ones(D))
    truth = qg_z_all(full(x, 0.0), n)
    obs = full(K @ x, leak)
    f, kept = filter_distribution(obs, n, w)
    reads = {"raw": qg_z_all(obs, n), "filter": qg_z_all(f, n),
             "half": qg_z_all(unmix_sector(obs, M, n, w, 0.5), n), "full": qg_z_all(unmix_sector(obs, M, n, w, 1.0), n)}
    exact = {k: 1000 * float(np.mean(np.abs(v - truth))) for k, v in reads.items()}
    neg, e_inv, e_nnls = 0, [], []
    for _ in range(trials):
        c = rng.multinomial(shots, obs) / shots
        fs, _ = filter_distribution(c, n, w)
        y = np.linalg.solve(A, fs[st])
        neg += bool(np.any(y < 0))
        e_inv.append(np.mean(np.abs(qg_z_all(full(y, 0.0), n) - truth)))
        e_nnls.append(np.mean(np.abs(qg_z_all(unmix_sector(c, M, n, w, 0.5), n) - truth)))
    return dict(exact=exact, kept=float(kept), leak=leak, k_diag=diag, diag=float(np.diag(M).mean()), neg=neg, trials=trials, shots=shots,
                e_inv=1000 * float(np.mean(e_inv)), e_nnls=1000 * float(np.mean(e_nnls)))


def write_c():
    sys.path.insert(0, os.path.dirname(__file__))
    from make_cap28_figures import commit_pairs
    from qang.sectors import sector_states

    pairs = {(s, c): ok for s, c, ok in commit_pairs()}
    for sec, _, c, _, _ in ECHO_RUNS:
        assert pairs.get((sec, c)), "commit %s of §%s is not a consistent pre-registration" % (c, sec)
    ex = echo_example()
    cost = [(5, 1, "§105, §106, §108"), (5, 2, "§107"), (8, 1, "—"), (8, 2, "§115")]
    for lang in ("es", "en"):
        t = (lambda es_, en_: es_) if lang == "es" else (lambda es_, en_: en_)  # noqa: E731
        L = [t("# Protocolo de calibración por eco e inversión en sectores de Hamming (§105–§108) {#sec-apendice-c}",
               "# The echo-calibration protocol and inversion in Hamming sectors (§105–§108) {#sec-appendix-c}"), "",
             t("""El capítulo 22 explica por qué hace falta el eco: el filtro qg descarta los disparos que salieron del sector, pero
no ve los errores que mueven una excitación de un qubit a otro sin cambiar el peso. Este apéndice da el protocolo paso a
paso, lo que cuesta, un ejemplo exacto generado por `tools/make_appendices.py` con las funciones de `qang.sectors`, y el
registro de las cuatro corridas pre-registradas de las notas.""",
               """Chapter 22 explains why the echo is needed: the qg filter discards the shots that left the sector, but it does
not see the errors that move an excitation from one qubit to another without changing the weight. This appendix gives the
protocol step by step, what it costs, an exact example generated by `tools/make_appendices.py` with the functions of
`qang.sectors`, and the record of the four pre-registered runs of the notes."""), "",
             t("## El protocolo {#sec-c-protocolo}", "## The protocol {#sec-c-protocol}"), "",
             t("""1. **Sector.** $n$ qubits y peso $w$: el sector tiene $D=\\binom{n}{w}$ estados de la base, $|e_1\\rangle,\\dots,|e_D\\rangle$
   (`sector_states`).
2. **Ecos.** Para cada $j$: preparar $|e_j\\rangle$ con compuertas $X$, aplicar la red $W$ y después $W^\\dagger$, y medir en
   $Z$. Sin errores el circuito vuelve a $|e_j\\rangle$. Las notas usan 4000 disparos por eco.
3. **Matriz.** Filtrar cada eco al sector (`filter_distribution`). Su distribución sobre los $D$ estados es la columna $j$ de
   $M$, con $M_{ij}=P(\\text{se mide }e_i\\mid\\text{se preparó }e_j)$ (`echo_transfer_matrix`).
4. **Raíz.** El circuito que se quiere leer tiene la mitad de las compuertas del eco, así que se corrige con $A=M^{1/2}$: la
   raíz principal, real, no negativa y normalizada por columnas (`matrix_power_stochastic`).
5. **Desmezcla.** Filtrar la distribución del circuito, $f$, y resolver $Ax=f$ con $x\\ge0$ por mínimos cuadrados no
   negativos; normalizar $x$ (`unmix_sector`).
6. **Lectura.** Cada qubit lee $m_Z(q)=1000\\,(1-2p_q)$ con las poblaciones de $x$.
7. **Antes de usarlo.** Simular sin ruido con los mismos disparos para conocer el error de muestreo. El eco ayuda cuando el
   error filtrado está bien por encima de ese nivel, y no cuando ya es casi todo ruido de disparos (§108).""",
               """1. **Sector.** $n$ qubits and weight $w$: the sector has $D=\\binom{n}{w}$ basis states, $|e_1\\rangle,\\dots,|e_D\\rangle$
   (`sector_states`).
2. **Echoes.** For each $j$: prepare $|e_j\\rangle$ with $X$ gates, apply the network $W$ and then $W^\\dagger$, and measure in
   $Z$. Without errors the circuit returns to $|e_j\\rangle$. The notes use 4000 shots per echo.
3. **Matrix.** Filter each echo to the sector (`filter_distribution`). Its distribution over the $D$ states is column $j$ of
   $M$, with $M_{ij}=P(e_i\\text{ measured}\\mid e_j\\text{ prepared})$ (`echo_transfer_matrix`).
4. **Root.** The circuit to be read has half the gates of the echo, so it is corrected with $A=M^{1/2}$: the principal,
   real, non-negative, column-normalized root (`matrix_power_stochastic`).
5. **Unmixing.** Filter the circuit's distribution, $f$, and solve $Ax=f$ with $x\\ge0$ by non-negative least squares;
   normalize $x$ (`unmix_sector`).
6. **Reading.** Each qubit reads $m_Z(q)=1000\\,(1-2p_q)$ with the populations of $x$.
7. **Before using it.** Simulate without noise at the same shots to know the sampling error. The echo helps when the
   filtered error is well above that level, and not when it is already almost all shot noise (§108)."""), "",
             t("## Costo {#sec-c-costo}", "## Cost {#sec-c-cost}"), "",
             t("| qubits | peso | estados del sector $D$ | circuitos de eco | disparos de eco (4000 c/u) | usado en |",
               "| qubits | weight | sector states $D$ | echo circuits | echo shots (4000 each) | used in |"), "|---|---|---|---|---|---|"]
        for n, w, where in cost:
            D = len(sector_states(n, w))
            assert D == math.comb(n, w)
            L.append("| %d | %d | %d | %d | %s | %s |" % (n, w, D, D, ("{:,}".format(4000 * D)).replace(",", "." if lang == "es"
                                                                                                        else ","), where))
        L += ["", t(": Un eco por estado del sector; la matriz se mide una vez por red, no por entrada. {#tbl-c-costo}",
                    ": One echo per sector state; the matrix is measured once per network, not per input. {#tbl-c-cost}"), "",
              t("""El costo crece como $\\binom{n}{w}$, no como $2^n$: es lo que hace practicable la inversión. En peso 2 con 8 qubits
son 28 circuitos; una matriz de confusión de lectura completa en 8 qubits necesitaría 256.""",
                """The cost grows as $\\binom{n}{w}$, not as $2^n$: that is what makes the inversion practical. At weight 2 with 8 qubits
it is 28 circuits; a full readout confusion matrix on 8 qubits would need 256."""), "",
              t("## Un ejemplo exacto {#sec-c-ejemplo}", "## An exact example {#sec-c-example}"), "",
              t("""El script arma un canal dentro del sector $K$ en 5 qubits y peso 1, con %s de probabilidad de quedarse en el qubit
correcto, y una pérdida T1 de %s de los disparos hacia $|00000\\rangle$, que lee $+1000$ m-qg en todos los qubits. El eco
aplica $K$ dos veces, así que $M=K^2$ (diagonal media %s), y la raíz principal recupera $K$ exactamente. Es la suposición detrás de $M^{1/2}$:
que la vuelta tenga el mismo error que la ida. En §105 el $M$ completo fue mejor en Forte-1, así que no siempre se cumple.""",
                """The script builds an in-sector channel $K$ on 5 qubits and weight 1, with probability %s of staying on the right
qubit, and a T1 loss of %s of the shots to $|00000\\rangle$, which reads $+1000$ m-qg on every qubit. The echo applies $K$
twice, so $M=K^2$ (mean diagonal %s), and the principal root recovers $K$ exactly. That is the assumption behind $M^{1/2}$: that the way back
has the same error as the way out. In §105 the full $M$ did better on Forte-1, so it does not always hold.""")
              % (nf(ex["k_diag"], ".2f", lang), nf(100 * ex["leak"], ".0f", lang) + (" %" if lang == "es" else "%"),
                 nf(ex["diag"], ".2f", lang)), "",
              t("| lectura | error medio de $qg_Z$ (m-qg) |", "| readout | mean $qg_Z$ error (m-qg) |"), "|---|---|",
              t("| sin qang | %s |", "| without qang | %s |") % nf(ex["exact"]["raw"], ".1f", lang),
              t("| filtro qg | %s |", "| qg filter | %s |") % nf(ex["exact"]["filter"], ".1f", lang),
              t("| filtro y eco, $M^{1/2}$ | %s |", "| filter and echo, $M^{1/2}$ | %s |") % sci(ex["exact"]["half"], lang),
              t("| filtro y eco, $M$ | %s |", "| filter and echo, $M$ | %s |") % nf(ex["exact"]["full"], ".1f", lang), "",
              t(": Sin ruido de disparos. El filtro quita la pérdida; $M^{1/2}$ deshace $K$; $M$ entero corrige de más y queda cerca del filtro solo. {#tbl-c-ejemplo}",
                ": No shot noise. The filter removes the loss; $M^{1/2}$ undoes $K$; the full $M$ over-corrects and ends near the "
                "filter alone. {#tbl-c-example}"), ""]
        L += [t("""**Por qué mínimos cuadrados no negativos y no la inversa.** Con %d disparos por corrida, en %d de %d corridas la
inversa directa $A^{-1}f$ da alguna población negativa, que no es una probabilidad. La solución no negativa nunca lo hace.
El error medio de $qg_Z$ es casi el mismo: %s m-qg con la inversa y %s m-qg con la solución no negativa. Lo que se gana es
que el resultado siempre es una distribución, no exactitud; por eso la mitigación de lectura usual también restringe la
solución al símplex.""",
                """**Why non-negative least squares and not the inverse.** With %d shots per run, in %d of %d runs the direct inverse
$A^{-1}f$ gives some negative population, which is not a probability. The non-negative solution never does. The mean
$qg_Z$ error is almost the same: %s m-qg with the inverse and %s m-qg with the non-negative solution. What is gained is that
the result is always a distribution, not accuracy; that is why the usual readout mitigation also restricts the solution to
the simplex.""")
              % (ex["shots"], ex["neg"], ex["trials"], nf(ex["e_inv"], ".1f", lang), nf(ex["e_nnls"], ".1f", lang)), ""]
        L += [t("## Las cuatro corridas pre-registradas {#sec-c-registro}", "## The four pre-registered runs {#sec-c-register}"), "",
              t("| § | predicciones (commit) | cumplidas | resultado |", "| § | predictions (commit) | passed | result |"),
              "|---|---|---|---|"]
        for sec, preds, c, ok, txt in ECHO_RUNS:
            L.append("| §%s | %s (`%s`) | %d de 4 | %s |" % (sec, preds, c, ok, txt[0]) if lang == "es"
                     else "| §%s | %s (`%s`) | %d of 4 | %s |" % (sec, preds, c, ok, txt[1]))
        tot = sum(r[3] for r in ECHO_RUNS)
        L += ["", t(": Las corridas de las notas, en modelos de ruido de tres dispositivos de IBM y dos de IonQ. {#tbl-c-registro}",
                    ": The runs of the notes, on the noise models of three IBM and two IonQ devices. {#tbl-c-register}"), "",
              t("""En total, %d de 16 predicciones cumplidas. El script comprueba que cada commit citado existe y que en él las notas
todavía no tenían la sección: las cuatro anterioridades son consistentes (apéndice E). Los números de la tabla vienen de las
notas; los scripts de `examples/` que nombra cada sección los reproducen, pero tardan horas y este apéndice no los corre.

La falla de §108 es la regla del paso 7. En ese modelo el valor de decisión se dispersa unos 0,22 por los disparos solos, y
con el filtro el error ya era 0,26–0,35: quedaba poco sesgo dentro del sector, e invertir la matriz sólo agregó varianza. En
§106 el error filtrado era 0,47–0,87, casi todo sesgo, y el eco lo redujo a la mitad.""",
                """In total, %d of 16 predictions passed. The script checks that each cited commit exists and that in it the notes did
not yet have the section: the four 'before's are consistent (Appendix E). The numbers in the table come from the notes;
the `examples/` scripts each section names reproduce them, but they take hours and this appendix does not run them.

The failure of §108 is the rule of step 7. In that model the decision value scatters by about 0.22 from shots alone, and
with the filter the error was already 0.26–0.35: little in-sector bias was left, and inverting the matrix only added
variance. In §106 the filtered error was 0.47–0.87, mostly bias, and the echo halved it.""") % tot, "",
              t("## Notas bibliográficas {.unnumbered}", "## Bibliographic notes {.unnumbered}"), "",
              t("""El eco es el de @hahn1950; la mitigación de errores de lectura por inversión de una matriz de confusión, @bravyi2021.
El filtro de sector es la verificación de simetría de @bonetmonroig2018. Las notas numeradas (§) y los scripts están en el
repositorio [@qangrepo].""",
                """The echo is that of @hahn1950; readout-error mitigation by inverting a confusion matrix, @bravyi2021. The sector
filter is the symmetry verification of @bonetmonroig2018. The numbered notes (§) and the scripts are in the repository
[@qangrepo].""")]
        write("c", lang, L)
    print("appendix C: exact errors %s, negative inverses %d/%d, echo runs %d/16"
          % ({k: round(v, 3) for k, v in ex["exact"].items()}, ex["neg"], ex["trials"], sum(r[3] for r in ECHO_RUNS)))


# --------------------------------------------------------------------------- Appendix E
def section_of(name):
    """The notes section whose heading names the script that wrote examples/data/<name>.json (or .prereg)."""
    import re
    base = re.sub(r"_(analysis|run)$", "", name)
    heads = [ln for ln in open(os.path.join(ROOT, "RESEARCH_NOTES.md"), encoding="utf-8").read().splitlines()
             if ln.startswith("## ")]
    for cand in (base, base + "_qg"):
        for h in heads:
            if "examples/%s.py" % cand in h:
                return h.split()[1].rstrip(".")
    return "—"


def write_e():
    import json
    import re
    sys.path.insert(0, os.path.dirname(__file__))
    from make_cap28_figures import DATA, commit_pairs, sha_audit, stored_verdicts

    runs = stored_verdicts()
    keys = {}
    for r in runs:
        d = json.load(open(os.path.join(DATA, r[0] + ".json"), encoding="utf-8"))["verdict"]
        keys[r[0]] = [k for k, v in d.items() if v is False]
    p, f = sum(r[1] for r in runs), sum(r[2] for r in runs)
    sha = sha_audit()
    ver = sum(1 for r in sha if r["match_at_prereg"] and r["before_results"])
    pairs = commit_pairs()
    by_sec = {}
    for s, c, ok in pairs:
        by_sec.setdefault(s, []).append((c, ok))
    secs = sorted(by_sec, key=lambda x: float(re.sub(r"[^0-9.]", "", x)))
    ok_c = sum(ok for _, _, ok in pairs)
    for lang in ("es", "en"):
        t = (lambda es_, en_: es_) if lang == "es" else (lambda es_, en_: en_)  # noqa: E731
        yn = (lambda b: "sí" if b else "no") if lang == "es" else (lambda b: "yes" if b else "no")  # noqa: E731
        L = [t("# Registro de falsación: las predicciones que el repositorio permite comprobar {#sec-apendice-e}",
               "# Falsification register: the predictions the repository allows one to check {#sec-appendix-e}"), "",
             t("""El plan del libro anunciaba aquí un recuento completo de §1–§164. El capítulo 28 explica por qué no se hizo: las
secciones anteriores a §71 casi no usan pre-registro, y muchas posteriores dejan sus veredictos sólo en prosa, con redacciones
distintas. Este apéndice lista lo que se puede comprobar con código. Lo genera `tools/make_appendices.py` con las mismas
funciones que la figura del capítulo 28 (`tools/make_cap28_figures.py`), así que sus totales son los del capítulo.""",
               """The book's plan announced a full count of §1–§164 here. Chapter 28 explains why it was not done: the sections before
§71 hardly use pre-registration, and many later ones leave their verdicts only in prose, worded differently. This appendix
lists what can be checked with code. `tools/make_appendices.py` generates it with the same functions as the Chapter 28
figure (`tools/make_cap28_figures.py`), so its totals are the chapter's."""), "",
             t("## Declaración de falsación {#sec-e-declaracion}", "## Falsification statement {#sec-e-statement}"), "",
             t("""- **Las definiciones y las identidades no se falsan con experimentos.** La unidad qg, el mili-qang, la ecuación de
  probabilidad y las fórmulas de los apéndices A y B se demuestran o se comprueban a precisión de máquina en `tests/`. Si una
  falla, es un error de código, no un resultado.
- **Las afirmaciones empíricas sí.** Cada una dice que una lectura con qang es más exacta, más barata o más estable que sin
  qang, y se fija con un umbral antes de correr. Una predicción fallada se publica con su número y no se reescribe.
- **Qué falsaría el programa entero.** Que las ventajas del capítulo 29 no se repitieran en hardware real con umbrales fijados
  antes. Hoy esa prueba tiene dos corridas en IQM Garnet (§165 y §168) con 7 de 8 predicciones cumplidas. Es poco para
  decidir.""",
               """- **Definitions and identities are not falsified by experiments.** The qg unit, the milli-qang, the probability
  equation and the formulas of Appendices A and B are proven or checked to machine precision in `tests/`. If one fails, it
  is a code bug, not a result.
- **Empirical claims are.** Each says that a reading with qang is more accurate, cheaper or more stable than without qang,
  and it is fixed with a threshold before running. A failed prediction is published with its number and not rewritten.
- **What would falsify the programme as a whole.** That the advantages of Chapter 29 did not repeat on real hardware with
  thresholds fixed in advance. Today that test has two runs on IQM Garnet (§165 and §168), with 7 of 8 predictions passed.
  That is too little to decide."""), "",
             t("## Veredictos guardados por el código {#sec-e-veredictos}", "## Verdicts stored by the code {#sec-e-verdicts}"), "",
             t("""Cada archivo de `examples/data/` con un diccionario `verdict` es una corrida cuyas predicciones juzgó el propio
script. La columna § es la sección de las notas que nombra al script.""",
               """Each file of `examples/data/` with a `verdict` dictionary is a run whose predictions the script itself scored. The §
column is the notes section that names the script."""), "",
             t("| corrida | § | cumplidas | falladas | cuáles fallaron |", "| run | § | passed | failed | which failed |"),
             "|---|---|---|---|---|"]
        for name, ok, bad in runs:
            sec = section_of(name)
            L.append("| `%s` | %s | %d | %d | %s |" % (name, "§" + sec if sec != "—" else sec, ok, bad,
                                                      ", ".join(keys[name]) or "—"))
        L += ["| **%s** | | **%d** | **%d** | |" % (t("total", "total"), p, f), "",
              t(": %d corridas, %d predicciones: %d cumplidas (%s) y %d falladas. {#tbl-e-veredictos}",
                ": %d runs, %d predictions: %d passed (%s) and %d failed. {#tbl-e-verdicts}")
              % (len(runs), p + f, p, nf(100 * p / (p + f), ".0f", lang) + (" %" if lang == "es" else "%"), f), "",
              t("## Pre-registros por sha256 {#sec-e-sha256}", "## Pre-registrations by sha256 {#sec-e-sha256}"), "",
              t("""Cada `examples/data/*.prereg` guarda el sha256 de un script congelado. La anterioridad está comprobada si el hash
coincide con el script en el commit que agregó el `.prereg` y ese commit es anterior al de los resultados.""",
                """Each `examples/data/*.prereg` holds the sha256 of a frozen script. The 'before' is checked if the hash matches the
script at the commit that added the `.prereg` and that commit is earlier than the one of the results."""), "",
              t("| pre-registro | § | coincide hoy | coincide en su commit | antes de los resultados | anterioridad |",
                "| pre-registration | § | matches today | matches at its commit | before the results | 'before' |"),
              "|---|---|---|---|---|---|"]
        for r in sha:
            sec = section_of(r["name"])
            good = r["match_at_prereg"] and r["before_results"]
            L.append("| `%s` | %s | %s | %s | %s | %s |" % (r["name"], "§" + sec if sec != "—" else sec, yn(r["match_now"]),
                                                          yn(r["match_at_prereg"]), yn(r["before_results"]),
                                                          t("comprobada", "checked") if good else
                                                          t("sólo fecha propia", "own timestamp only")))
        L += ["", t(": %d pre-registros: %d comprobados en git, %d sólo con la fecha del propio archivo. {#tbl-e-sha256}",
                    ": %d pre-registrations: %d checked in git, %d with the file's own timestamp only. {#tbl-e-sha256}")
              % (len(sha), ver, len(sha) - ver), "",
              t("""Que ninguno coincida hoy es lo esperable: después de la corrida se agregan los hallazgos al comentario inicial del
script. En los que tienen sólo fecha propia, el `.prereg` y los resultados entraron en el mismo commit.""",
                """That none matches today is expected: after the run the findings are added to the script's opening comment. In those
with their own timestamp only, the `.prereg` and the results entered in the same commit."""), "",
              t("## Pre-registros por commit {#sec-e-commit}", "## Pre-registrations by commit {#sec-e-commit}"), "",
              t("""Cada "committed … (abcdef1)" de las notas: el commit existe y en él las notas todavía no tienen la sección.""",
                """Each "committed … (abcdef1)" in the notes: the commit exists and in it the notes do not yet have the section."""), "",
              t("| § | commits | consistentes |", "| § | commits | consistent |"), "|---|---|---|"]
        for s in secs:
            cs = by_sec[s]
            L.append("| §%s | %s | %d de %d |" % (s, ", ".join("`%s`" % c for c, _ in cs), sum(o for _, o in cs), len(cs))
                     if lang == "es" else
                     "| §%s | %s | %d of %d |" % (s, ", ".join("`%s`" % c for c, _ in cs), sum(o for _, o in cs), len(cs)))
        L += ["", t(": %d citas en %d secciones, §%s–§%s: %d consistentes. {#tbl-e-commit}",
                    ": %d citations in %d sections, §%s–§%s: %d consistent. {#tbl-e-commit}")
              % (len(pairs), len(secs), secs[0], secs[-1], ok_c), "",
              t("## Lo que este registro no incluye {#sec-e-fuera}", "## What this register leaves out {#sec-e-outside}"), "",
              t("""- **La línea de QML (§75–§123)** lleva su propia cuenta en el README: 176 predicciones juzgadas, 123 cumplidas y 53
  falladas, con 4 más esperando la corrida en hardware de IonQ (§116). Se superpone en parte con las tablas de arriba y no
  debe sumarse.
- **Los veredictos en prosa** ("Four passed; V1 failed") de las secciones que no guardan un diccionario. Están en las notas,
  sección por sección, pero contarlos exige leerlos uno a uno.
- **Las secciones anteriores a §71**, que casi no usan pre-registro: sus números son reproducibles, pero no tienen un umbral
  fijado antes.""",
                """- **The QML line (§75–§123)** keeps its own count in the README: 176 predictions judged, 123 passed and 53 failed,
  with 4 more waiting for the IonQ hardware run (§116). It partly overlaps the tables above and must not be added to them.
- **The verdicts in prose** ("Four passed; V1 failed") of the sections that store no dictionary. They are in the notes,
  section by section, but counting them requires reading them one by one.
- **The sections before §71**, which hardly use pre-registration: their numbers are reproducible, but they have no threshold
  fixed in advance."""), "",
              t("## Notas bibliográficas {.unnumbered}", "## Bibliographic notes {.unnumbered}"), "",
              t("""El criterio de falsación es el de @popper1959, y el pre-registro como práctica, @nosek2018. Las notas numeradas (§),
los archivos de `examples/data/` y los scripts están en el repositorio [@qangrepo].""",
                """The falsification criterion is that of @popper1959, and pre-registration as a practice, @nosek2018. The numbered
notes (§), the files of `examples/data/` and the scripts are in the repository [@qangrepo].""")]
        write("e", lang, L)
    print("appendix E: %d runs, %d predictions (%d passed, %d failed); sha256 %d/%d checked; commit %d/%d consistent"
          % (len(runs), p + f, p, f, ver, len(sha), ok_c, len(pairs)))


# --------------------------------------------------------------------------- Appendix F
def h2(p):
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def check_f(seed=7):
    """Riesz representatives of linear readouts, and the bias of the plug-in entropy (no representative)."""
    from scipy.stats import binom

    paulis = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    basis = [np.kron(paulis[a], paulis[b]) / 2 for a in "IXYZ" for b in "IXYZ"]  # Hilbert-Schmidt orthonormal, 2 qubits

    def representative(f):
        return sum(np.conj(f(E)) * E for E in basis)

    ZI = np.kron(paulis["Z"], np.eye(2))
    e1 = np.abs(representative(lambda r: np.trace(r @ ZI)) - ZI).max()
    w = np.random.default_rng(seed).normal(size=4)
    e2 = np.abs(representative(lambda r: np.sum(w * np.real(np.diag(r)))) - np.diag(w)).max()
    assert e1 < 1e-14 and e2 < 1e-14
    bias = []
    for p0 in (0.5, 0.9):
        for N in (10, 100, 1000):
            k = np.arange(N + 1)
            pk = binom.pmf(k, N, p0)
            bz = float(np.sum(pk * (2 * k / N - 1)) - (2 * p0 - 1))
            bs = float(np.sum(pk * np.array([h2(x / N) for x in k])) - h2(p0))
            bias.append((p0, N, bz, bs))
    return e1, e2, bias


def write_f():
    e1, e2, bias = check_f()
    for lang in ("es", "en"):
        t = (lambda es_, en_: es_) if lang == "es" else (lambda es_, en_: en_)  # noqa: E731
        L = [t("# Fundamento de análisis funcional: Riesz–Fréchet {#sec-apendice-f}",
               "# Functional-analysis foundation: Riesz–Fréchet {#sec-appendix-f}"), "",
             t("""Este apéndice es conceptual y no agrega resultados. Es el apéndice A de las notas de investigación, que el capítulo 4
cita: explica por qué $qg_Z$ es la lectura lineal canónica de un estado y por qué $qg_S$ no puede serlo. Las tres
comprobaciones numéricas las hace `tools/make_appendices.py`.""",
               """This appendix is conceptual and adds no results. It is Appendix A of the research notes, which Chapter 4 cites: it
explains why $qg_Z$ is the canonical linear readout of a state and why $qg_S$ cannot be one. The three numerical checks are
done by `tools/make_appendices.py`."""), "",
             t("## El marco {#sec-f-marco}", "## The setting {#sec-f-setting}"), "",
             t("""Para $n$ qubits, los operadores sobre $\\mathbb{C}^{2^n}$ forman un espacio de Hilbert de dimensión finita con el
producto de Hilbert–Schmidt""",
               """For $n$ qubits, the operators on $\\mathbb{C}^{2^n}$ form a finite-dimensional Hilbert space with the Hilbert–Schmidt
inner product"""), "",
             "$$", "\\langle A,B\\rangle=\\mathrm{Tr}(A^\\dagger B),", "$$", "",
             t("y las matrices densidad viven en él.", "and density matrices live in it."), "",
             t("## El teorema {#sec-f-teorema}", "## The theorem {#sec-f-theorem}"), "",
             t("""**Representación de Riesz–Fréchet** [@riesz1907; @frechet1907]. Todo funcional lineal continuo $f$ sobre un
espacio de Hilbert $H$ es de la forma $f(x)=\\langle y_f,x\\rangle$ para un único $y_f\\in H$. En dimensión finita la
continuidad es automática y el teorema se reduce a álgebra lineal: con una base ortonormal $\\{E_i\\}$,
$y_f=\\sum_i\\overline{f(E_i)}\\,E_i$. Por eso es un fundamento y no un resultado.""",
               """**Riesz–Fréchet representation** [@riesz1907; @frechet1907]. Every continuous linear functional $f$ on a Hilbert
space $H$ has the form $f(x)=\\langle y_f,x\\rangle$ for a unique $y_f\\in H$. In finite dimension continuity is automatic and
the theorem reduces to linear algebra: with an orthonormal basis $\\{E_i\\}$, $y_f=\\sum_i\\overline{f(E_i)}\\,E_i$. That is why
it is a foundation and not a result."""), "",
             t("## Aplicación: $qg_Z$ tiene representante {#sec-f-qgz}", "## Application: $qg_Z$ has a representative {#sec-f-qgz}"), "",
             t("""La lectura $\\rho\\mapsto qg_Z(\\rho)=\\mathrm{Tr}(\\rho\\,\\sigma_z)$ es lineal en $\\rho$, así que tiene un único
representante de Hilbert–Schmidt: $\\sigma_z$ en un qubit, y $\\sigma_z^{(i)}\\otimes I$ en el qubit $i$ de un registro. Toda
lectura lineal de un estado, sea un valor esperado, una población o un promedio por disparo como el XEB lineal, tiene esta
forma. $qg_Z$ es aquella cuyo representante es el observable $Z$.

El script lo comprueba en dos qubits con la base de Pauli normalizada, $E_P=P/2$. Reconstruye el representante de
$\\rho\\mapsto\\mathrm{Tr}(\\rho\\,Z\\otimes I)$ y obtiene $Z\\otimes I$ con error %s. Para una lectura de poblaciones con pesos
al azar, $\\rho\\mapsto\\sum_x w_x\\langle x|\\rho|x\\rangle$, obtiene $\\mathrm{diag}(w)$ con error %s.""",
               """The readout $\\rho\\mapsto qg_Z(\\rho)=\\mathrm{Tr}(\\rho\\,\\sigma_z)$ is linear in $\\rho$, so it has a unique
Hilbert–Schmidt representative: $\\sigma_z$ on one qubit, and $\\sigma_z^{(i)}\\otimes I$ on qubit $i$ of a register. Every
linear readout of a state, be it an expectation value, a population or a per-shot average such as linear XEB, has this form.
$qg_Z$ is the one whose representative is the $Z$ observable.

The script checks it on two qubits with the normalized Pauli basis, $E_P=P/2$. It rebuilds the representative of
$\\rho\\mapsto\\mathrm{Tr}(\\rho\\,Z\\otimes I)$ and gets $Z\\otimes I$ with error %s. For a population readout with random
weights, $\\rho\\mapsto\\sum_x w_x\\langle x|\\rho|x\\rangle$, it gets $\\mathrm{diag}(w)$ with error %s.""") % (sci(e1, lang), sci(e2, lang)), "",
             t("## Lo que no tiene representante: $qg_S$ {#sec-f-qgs}", "## What has no representative: $qg_S$ {#sec-f-qgs}"), "",
             t("""$qg_S(\\rho)=H(\\langle0|\\rho|0\\rangle)$ es estrictamente cóncava en $\\rho$, no lineal, así que ningún operador la
representa. El ejemplo más corto: $|0\\rangle\\langle0|$ y $|1\\rangle\\langle1|$ tienen $qg_S=0$ los dos, y su mezcla por mitades
tiene $qg_S=1$ bit; una lectura lineal daría el promedio, 0. Con $qg_Z$ el promedio sí se cumple: $(+1-1)/2=0$.

El mismo hecho estructural está detrás de dos resultados de las notas:

- **§10.3 B:** las lecturas lineales tienen estimadores insesgados por promedio de disparos, y la entropía no (desigualdad de
  Jensen). La tabla da el sesgo exacto, por suma binomial, de los dos estimadores directos.
- **§7.1:** la identidad $qg_S=H\\big((1+qg_Z)/2\\big)$ escribe la métrica no lineal como una función escalar fija de la lineal.
  $qg_S$ no lleva información del estado más allá de la que ya extrae el representante de $qg_Z$.""",
               """$qg_S(\\rho)=H(\\langle0|\\rho|0\\rangle)$ is strictly concave in $\\rho$, not linear, so no operator represents it. The
shortest example: $|0\\rangle\\langle0|$ and $|1\\rangle\\langle1|$ both have $qg_S=0$, and their equal mixture has $qg_S=1$ bit;
a linear readout would give the average, 0. With $qg_Z$ the average does hold: $(+1-1)/2=0$.

The same structural fact is behind two results of the notes:

- **§10.3 B:** linear readouts have unbiased shot-average estimators, and entropy does not (Jensen's inequality). The table
  gives the exact bias, by binomial sum, of the two direct estimators.
- **§7.1:** the identity $qg_S=H\\big((1+qg_Z)/2\\big)$ writes the non-linear metric as a fixed scalar function of the linear
  one. $qg_S$ carries no information about the state beyond what the representative of $qg_Z$ already extracts."""), "",
             t("| $p_0$ | disparos $N$ | sesgo del promedio de $qg_Z$ | sesgo de $H(k/N)$ (bits) |",
               "| $p_0$ | shots $N$ | bias of the $qg_Z$ average | bias of $H(k/N)$ (bits) |"), "|---|---|---|---|"]
        for p0, N, bz, bs in bias:
            L.append("| %s | %d | %s | %s |" % (nf(p0, ".1f", lang), N, sci(abs(bz), lang) if abs(bz) > 1e-15 else "0",
                                               nf(bs, "+.5f", lang)))
        L += ["", t(": El promedio de $qg_Z$ es insesgado a precisión de máquina; la entropía de la frecuencia subestima siempre. "
                    "{#tbl-f-sesgo}",
                    ": The $qg_Z$ average is unbiased to machine precision; the entropy of the frequency always underestimates. "
                    "{#tbl-f-bias}"), "",
              t("## Sobre el lema de Riesz (1918) {#sec-f-lema}", "## On Riesz's lemma (1918) {#sec-f-lemma}"), "",
              t("""El lema del vector casi ortogonal sirve para mostrar que la bola unidad cerrada de un espacio normado de dimensión
infinita no es compacta. En las dimensiones finitas de este libro es trivial, porque siempre existe un vector exactamente
ortogonal. No cumple ningún papel en qang y no debería citarse como si lo cumpliera.""",
                """The almost-orthogonal-vector lemma is used to show that the closed unit ball of an infinite-dimensional normed
space is not compact. In the finite dimensions of this book it is trivial, because an exactly orthogonal vector always
exists. It plays no role in qang and should not be cited as if it did."""), "",
              t("## Notas bibliográficas {.unnumbered}", "## Bibliographic notes {.unnumbered}"), "",
              t("""El teorema de representación es @riesz1907 y @frechet1907. La entropía binaria es la de @shannon1948. Las notas
numeradas (§) y el script están en el repositorio [@qangrepo].""",
                """The representation theorem is @riesz1907 and @frechet1907. The binary entropy is that of @shannon1948. The
numbered notes (§) and the script are in the repository [@qangrepo].""")]
        write("f", lang, L)
    print("appendix F: representative errors %.1e, %.1e; entropy bias at p0=0.5, N=100: %+.5f bits" % (e1, e2, bias[1][3]))


if __name__ == "__main__":
    write_a()
    write_b()
    write_c()
    write_d()
    write_e()
    write_f()
