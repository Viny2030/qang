"""Appendices A and D of the book (ES/EN), generated from the code so they cannot drift from it.

    .venv-int/Scripts/python tools/make_appendices.py

* Appendix A: the qg action of the 15 gates of qang.formulation, qg'_P = qg_{U^dag P U}, from formulation.gate_table,
  with Clifford, weight conservation and the radius class (formulation.gate_radius_class).
* Appendix D: every module of the qang package with its public names (``__all__``, or the public functions and
  classes it defines). Run it in an environment where the optional SDKs import (``.venv-int``), so that every module
  is listed; a module that fails to import is reported, not skipped silently.

Writes book/es/apendice_a.qmd, book/es/apendice_d.qmd, book/en/appendix_a.qmd and book/en/appendix_d.qmd.
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


if __name__ == "__main__":
    write_a()
    write_d()
