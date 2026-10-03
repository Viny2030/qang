"""Adds sections 31-38 (RESEARCH_NOTES §75-§93) to notebooks/qang_avances_colab.ipynb and renumbers the
honest summary as section 39. Run from the repository root: python tools/extend_avances_notebook.py"""
import sys

import nbformat as nbf

PATH = sys.argv[1] if len(sys.argv) > 1 else "notebooks/qang_avances_colab.ipynb"

NEW = [
    ("md", """## 31. Redes neuronales cuánticas con y sin qang (§75–§80)

Una QNN que conserva el peso de Hamming mantiene los datos en un sector; el filtro qg se queda solo con los disparos que siguieron en él. Con T1 igual en todos los qubits, la lectura filtrada es **exactamente** la sin ruido (F1/F4). Se entrena en simulador (sin ruido) y se lee bajo T1 (γ = 0.08 por qubit y subcapa) **con qang** y **sin qang**. Librería: `qang.qml`."""),
    ("code", """subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"], check=True)
import qnn_classifier_qg as Q
from qang.qml import WeightQNN, kept_fraction
Xa, ya, use_pca = Q.load("cancer", np.random.default_rng(80))
Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, 8000)
modelos = {"E (peso 1)": WeightQNN(5, 1), "W (peso 2)": WeightQNN(5, 2)}
for nombre, m in modelos.items():
    m.fit(Xtr, ytr, epochs=120, seed=1)
    con, sin = m.score(Xte, yte, gamma=0.08, qang=True), m.score(Xte, yte, gamma=0.08, qang=False)
    print(f"{nombre}: sin ruido {m.score(Xte, yte):.3f} | bajo T1: con qang {con:.3f}, sin qang {sin:.3f} "
          f"(diferencia {con - sin:+.3f}) | disparos conservados {kept_fraction(0.08, m.weight, m.depth):.2f}")"""),
    ("md", """**Lectura.** En 60 corridas (§80): entrenando sin ruido, qang suma +2.5 puntos en peso 1 y +16.5 en peso 2, y recupera exactamente la precisión sin ruido. Entrenando con el ruido calibrado, el modelo sin qang lo alcanza (diferencia 0.1 puntos); con una lectura rica (qg_ZZ) el modelo sin qang llega a ser 1.2 puntos mejor (§81), porque los disparos decaídos todavía informan. Estos modelos son simulables clásicamente: es robustez, no ventaja cuántica."""),
    ("md", """## 32. Cuánta diferencia de T1 entre qubits aguanta el filtro (§78, §82)

γ_q = 0.08 (1 + s·u_q), con u_q repartido en [−1, 1]: s es la dispersión relativa de las tasas 1/T1."""),
    ("code", """m = modelos["E (peso 1)"]
u = np.random.default_rng(3).permutation(np.linspace(-1, 1, 5))
for s in (0.0, 0.5, 1.0):
    g = 0.08 * (1 + s * u)
    print(f"s = {s:.1f}: con qang {m.score(Xte, yte, gamma=g, qang=True):.3f}, sin qang {m.score(Xte, yte, gamma=g, qang=False):.3f}"
          f" (sin ruido {m.score(Xte, yte):.3f})")"""),
    ("md", """**Lectura.** En 60 corridas (§82) la pérdida del filtro es 0.3 puntos con ±50 % y 1.4 con ±100 %: el filtro solo alcanza hasta una dispersión de ±80 % de 1/T1. Con más, hay que entrenar con el ruido calibrado. El desfase no lo corrige (§78)."""),
    ("md", """## 33. Peso 2: la carga de datos decide (§81, §87, §90)

Productos de pares (§79) contra la codificación "dual" que llena los 10 estados de peso 2 sin deformar los datos (x en el anillo, x² en las cuerdas), con lectura qg_Z o qg_ZZ."""),
    ("code", """variantes = {"W pares, qg_Z": dict(weight=2), "W dual, qg_Z": dict(weight=2, encoding="dual"),
             "W dual, qg_ZZ": dict(weight=2, encoding="dual", readout="zz")}
print(f"E (peso 1): {modelos['E (peso 1)'].score(Xte, yte):.3f}")
for nombre, kw in variantes.items():
    w = WeightQNN(5, **kw).fit(Xtr, ytr, epochs=120, seed=1)
    print(f"{nombre}: sin ruido {w.score(Xte, yte):.3f} | bajo T1 con qang {w.score(Xte, yte, gamma=0.08, qang=True):.3f}, "
          f"sin qang {w.score(Xte, yte, gamma=0.08, qang=False):.3f}")"""),
    ("md", """**Lectura.** La brecha del peso 2 venía de la carga de datos (el techo de información de los productos de pares es 2.4 puntos menor, §87). Con la codificación dual y la lectura qg_ZZ completa, el peso 2 iguala al peso 1 (0.947 = 0.947 en 60 corridas, §90). Iguala, no supera."""),
    ("md", """## 34. El QNN filtrado en modelos de ruido de equipos (§85, §91)

El modelo E se compila a Qiskit (cascada de 4 RBS que carga los datos + 15 RBS entrenadas; el circuito coincide con el modelo a 10⁻¹⁶) y se corre en un backend simulado con la calibración de un equipo IBM. Cada entrada se lee con y sin qang de los mismos disparos."""),
    ("code", """import qnn_hardware_qg as HW
r, v = HW.main(["--mode", "fake", "--shots", "1000"])"""),
    ("md", """**Lectura.** En tres backends de IBM y en el simulador de IonQ con ruido Aria-1 y Forte-1, con iris la precisión no cambia (los márgenes son amplios) y el error del qg_Z medido baja 2.9–3.6 veces con qang, conservando ~70 % de los disparos. En 189 entradas de cuatro conjuntos con ruido Forte-1 (§91), qang es 1.1 puntos más preciso (0.968 contra 0.958; sin ruido 0.979) y el error baja 2.7–2.9 veces. Son modelos de ruido, no equipos: las corridas en hardware están preparadas (`--mode ibm`, `--mode ionq_qpu`) y pendientes."""),
    ("md", """## 35. Síndromes del código de Leung en circuitos, sin ancillas (§88, §92, §93)

§33 en circuitos: codificar en el código de Leung, esperar t y leer los estabilizadores Z0Z1, Z2Z3 (base Z) y XXXX (base X) **destructivamente**, sin ancillas ni SWAPs. Un ajuste conjunto de todas las esperas da T1 y T_φ."""),
    ("code", """import qec_syndrome_destructive_qg as D93
b = D93.S.get_backend("fake", "fake_torino")
path = D93.best_path(b)
per_t, sin_ruteo = D93.run(b, "fake", 2000, path)
fit, rows = D93.analyse(b, per_t, path)
print("qubits", path, "| sin SWAPs:", sin_ruteo, "| T1 ajustado", round(fit["T1"], 1), "us")
for x in rows:
    print(f"t = {x['t_us']:.0f} us: gamma calibración {x['gamma calibration']:.3f}, síndromes {x['gamma joint fit']:.3f}")"""),
    ("md", """**Lectura.** Con ancillas (§88) el circuito necesitaba SWAPs y la comparación fallaba (§92, registrado). Sin ancillas ni SWAPs (§93), los síndromes leen el amortiguamiento a 1–4 % de la calibración en tres backends; el desfase sale 1.2–1.5× alto. Pendiente: un equipo IBM real."""),
    ("md", """## 36. Luz polarizada en unidades qg (§83)

La esfera de Poincaré es la esfera de Bloch: los parámetros de Stokes normalizados **son** valores qg, y una matriz de Mueller sin pérdidas es la regla de compuertas qg. Ley de Malus en qg: I = I₀ (1 + qg_Z cos 2θ + qg_X sin 2θ)/2."""),
    ("code", """from qang import polarization as P
from qang import formulation as F
E = np.array([0.6, 0.8 * np.exp(0.7j)])
q = P.jones_to_qg(E)
ref = F.qg_values(E.astype(complex))
print("Stokes -> qg:", {k: round(q[k], 4) for k in "XYZ"}, "| <P> del estado:", {k: round(ref.get(k, 0), 4) for k in "XYZ"})
for t in (0.0, 0.5, 1.0):
    mueller = (P.mueller_from_jones(P.linear_polarizer(t)) @ P.qg_to_stokes(q))[0]
    print(f"Malus theta = {t}: fórmula qg {P.malus_intensity(q, t):.4f} | matriz de Mueller {mueller:.4f}")"""),
    ("md", """## 37. Fase geométrica y teorema de Stokes (§84)

La fase de un lazo en la esfera es −½ del ángulo sólido (teorema de Stokes para la curvatura de Berry). Para un lazo a qg_Z constante: **(qg_Z − 1)/2 de vuelta**."""),
    ("code", """from qang import geometric as G
for z in (0.8, 0.0, -0.6):
    s = np.sqrt(1 - z**2); t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    lazo = np.stack([s * np.cos(t), s * np.sin(t), np.full_like(t, z)], axis=1)
    print(f"qg_Z = {z:+.1f}: solapamientos {G.geometric_phase_qg(lazo).phi:.5f} vueltas | (qg_Z - 1)/2 = {G.cone_phase_turns(z):.5f}")"""),
    ("md", """**Lectura (36–37).** Física estándar en las unidades de qang, verificada con 26 pruebas; conecta qang con los qubits fotónicos. Nota: `manuscript/qang_optics_es.pdf`; cuaderno propio: `qang_optics_es.ipynb`."""),
    ("md", """## 38. Lo que falta: hardware real

Todo lo anterior es simulación, incluidos los modelos de ruido de IBM e IonQ. Los scripts de hardware están listos, con confirmación explícita de costo:

* `examples/qnn_hardware_qg.py --mode ionq_qpu` (Forte-1, 30 circuitos × 1000 disparos) y `--mode ibm`
* `examples/hardware_characterization_ibm.py --mode ibm` (§86)
* `examples/qec_syndrome_destructive_qg.py --mode ibm` (§93)

Las predicciones de cada uno están registradas antes de correr."""),
]

CONTENTS = """31. Redes neuronales cuánticas con y sin qang — §75–§80
32. Cuánta diferencia de T1 entre qubits aguanta el filtro — §78, §82
33. Peso 2: la carga de datos decide — §81, §87, §90
34. El QNN filtrado en modelos de ruido de equipos — §85, §91
35. Síndromes del código de Leung en circuitos, sin ancillas — §88, §92, §93
36. Luz polarizada en unidades qg — §83
37. Fase geométrica y teorema de Stokes — §84
38. Lo que falta: hardware real
39. Resumen honesto"""

ROWS = """| QNN (§75–§80) | entrenando en simulador, qang recupera exactamente la precisión bajo T1 (+2.5 / +16.5 puntos) | entrenando con ruido no suma; con lectura rica cuesta 1.2 puntos (§81) |
| Dispersión de T1 (§82) | el filtro solo alcanza hasta ±80 % de 1/T1 | más allá, entrenar con ruido |
| Peso 2 (§87, §90) | con carga dual + qg_ZZ iguala al peso 1 | no lo supera |
| Modelos de ruido de equipos (§85, §91) | error del qg_Z ÷2.7–3.6 en IBM e IonQ simulados; +1.1 puntos en 189 entradas (Forte-1) | falta hardware real |
| Síndromes en circuitos (§93) | T1 a 1–4 % sin ancillas | desfase 1.2–1.5× alto |
| Óptica (§83–§84) | Stokes = qg; Mueller = regla de compuertas; fase = (qg_Z − 1)/2 | física conocida, sin novedad física |
"""


def main():
    nb = nbf.read(PATH, as_version=4)
    if any("## 31. Redes neuronales cuánticas" in c.source for c in nb.cells):
        print("already extended")
        return
    nb.cells[0].source = nb.cells[0].source.replace("avances §20–§74", "avances §20–§93").replace(
        "# El Qang (qg): avances §20–§74", "# El Qang (qg): avances §20–§93").replace("31. Resumen honesto", CONTENTS)
    summary = nb.cells[-1]
    summary.source = summary.source.replace("## 31. Resumen honesto", "## 39. Resumen honesto")
    lines = summary.source.split("\n")
    last_row = max(i for i, l in enumerate(lines) if l.startswith("|"))
    lines.insert(last_row + 1, ROWS.rstrip("\n"))
    summary.source = "\n".join(lines)
    new = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in NEW]
    nb.cells = nb.cells[:-1] + new + [summary]
    nbf.write(nb, PATH)
    print("added", len(new), "cells")


if __name__ == "__main__":
    main()
