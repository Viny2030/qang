"""Builds notebooks/qang_qml.ipynb (English) and notebooks/qang_qml_es.ipynb (Spanish)."""
import sys

import nbformat as nbf

REPO = "https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/"

T = {
    "en": {
        "file": "qang_qml.ipynb",
        "title": "# Quantum neural networks with and without qang\n\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](" + REPO + "qang_qml.ipynb)\n\n"
        "This notebook uses the published library (`pip install qang`, module `qang.qml`). A weight-conserving QNN "
        "keeps its data in one Hamming-weight sector, so the **qg filter** can keep only the shots that stayed in that "
        "sector. Every result below is shown **with qang** (filtered readout) and **without qang** (raw readout), "
        "with their difference.\n\n"
        "What to expect (RESEARCH_NOTES §75–§80): no quantum advantage (these models are classically simulable); "
        "trained on a simulator and run under T1 relaxation, qang recovers the noiseless accuracy exactly; trained "
        "under the calibrated noise, a model without qang catches up; dephasing and unequal T1 are not corrected by "
        "the filter.",
        "install": "## 1. Install",
        "data": "## 2. Data: breast cancer, 4 PCA features scaled to [-1, 1]",
        "w1": "## 3. Weight-1 QNN, trained without noise, run under T1 (gamma = 0.08)\n\n"
        "With qang the readout is exactly the noiseless one (F1); the price is the discarded shots, "
        "known in advance: (1 - gamma)^(weight x depth).",
        "w2": "## 4. Weight-2 QNN: the effect of qang grows with the weight\n\n"
        "A weight-2 state decays twice as fast, so without qang the readout is more biased; with qang it is still exact.",
        "aware": "## 5. Noise-aware training: training under the noise, with and without qang\n\n"
        "If the device noise is known and the model is trained under it, the model without qang learns the decay "
        "and catches up (§79–§80). This takes a couple of minutes.",
        "real": "## 6. What the filter does not correct: unequal T1 across qubits, and dephasing",
        "shots": "## 7. Finite shots (200 per input)",
        "multi": "## 8. Multiclass: each qubit is a class (§110)\n\n"
        "`MultiClassQNN` reads C classes from one excitation spread over the qubits. Readout `\"qubit\"`: the score "
        "of class c is the excitation probability of qubit c. Readout `\"head\"`: a linear softmax head on the five "
        "qg_Z. The filter keeps the shots with exactly one excitation, so the class probabilities are exactly the "
        "noiseless ones under equal T1. Digits 0-4 (5 classes, one per qubit), 4 PCA features. Over 15 runs (§110) the filter added 5.3 points (qubit readout) and 9.1 (head) on average, and 15 and 18 on digits; one split, like the one below, varies from run to run. On 3-class iris the difference is small.",
        "big": "## 9. Eight qubits: the effect of qang grows with size at weight 2 (§111)\n\n"
        "Same QNN with 8 qubits (7 PCA features), weight 1 and weight 2 (dual encoding, qg_ZZ readout), trained without "
        "noise and run under T1. Over 54 runs (§111) the filter kept every model exactly at its noiseless accuracy; "
        "without it the loss stayed near 3 points at weight 1 but reached 26 points at weight 2 with 8 qubits "
        "(0.965 against 0.705). Wine (classes 0 and 1). Training the weight-2 model takes about a minute.",
        "grad": "## 10. Gradients under T1: what the filter gives back (§112)\n\n"
        "Random weight-conserving circuits, gradient of qg_Z of qubit 0 with respect to the first angle. With qang the "
        "gradient is exactly the noiseless one; without qang it is K times smaller at weight 1 (K = kept fraction). The "
        "last columns are the median shots needed to resolve the gradient: the filter is cheaper while K is not too "
        "small; at half filling with 8 qubits and long depth (K < 0.15) the raw readout needed fewer shots in §112. With "
        "only 40 draws, as here, the shot medians are noisy (in §112, with 200 draws, n = 6 at weight 3 needed 4215 "
        "shots with qang against 5570 without).",
        "summary": "## Summary\n\n"
        "* **Trained on a simulator, run under T1:** qang is decisive, exact under equal T1, and more so at higher "
        "weight.\n"
        "* **Trained under the calibrated noise:** with and without qang are within about one test sample.\n"
        "* **Cost:** a known fraction of the shots, 1 - (1 - gamma)^(weight x depth).\n"
        "* **Limits:** unequal T1 and dephasing are not corrected; no quantum advantage.\n\n"
        "Details, pre-registered predictions and failures: `RESEARCH_NOTES.md` §75–§80, scripts in `examples/qnn_*_qg.py`.",
        "cols": ("reading", "with qang", "without qang", "difference"),
    },
    "es": {
        "file": "qang_qml_es.ipynb",
        "title": "# Redes neuronales cuánticas con y sin qang\n\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](" + REPO + "qang_qml_es.ipynb)\n\n"
        "Este cuaderno usa la librería publicada (`pip install qang`, módulo `qang.qml`). Una QNN que conserva el "
        "peso de Hamming mantiene los datos en un sector, y el **filtro qg** puede quedarse solo con los disparos que "
        "siguieron en ese sector. Cada resultado se muestra **con qang** (lectura filtrada) y **sin qang** (lectura "
        "cruda), con su diferencia.\n\n"
        "Qué esperar (RESEARCH_NOTES §75–§80): no hay ventaja cuántica (estos modelos son simulables clásicamente); "
        "entrenada en simulador y ejecutada con relajación T1, qang recupera exactamente la precisión sin ruido; "
        "entrenada con el ruido calibrado, la versión sin qang la alcanza; el desfase y el T1 desigual no los corrige "
        "el filtro.",
        "install": "## 1. Instalación",
        "data": "## 2. Datos: cáncer de mama, 4 componentes PCA escaladas a [-1, 1]",
        "w1": "## 3. QNN de peso 1, entrenada sin ruido, ejecutada con T1 (gamma = 0.08)\n\n"
        "Con qang la lectura es exactamente la sin ruido (F1); el precio son los disparos descartados, "
        "conocidos de antemano: (1 - gamma)^(peso x profundidad).",
        "w2": "## 4. QNN de peso 2: el efecto de qang crece con el peso\n\n"
        "Un estado de peso 2 decae el doble de rápido: sin qang la lectura está más sesgada; con qang sigue siendo exacta.",
        "aware": "## 5. Entrenamiento con ruido, con y sin qang\n\n"
        "Si se conoce el ruido del equipo y se entrena con él, la versión sin qang aprende el decaimiento y la "
        "alcanza (§79–§80). Tarda un par de minutos.",
        "real": "## 6. Lo que el filtro no corrige: T1 distinto en cada qubit, y desfase",
        "shots": "## 7. Disparos finitos (200 por entrada)",
        "multi": "## 8. Varias clases: cada qubit es una clase (§110)\n\n"
        "`MultiClassQNN` lee C clases a partir de una excitación repartida entre los qubits. Lectura `\"qubit\"`: el "
        "puntaje de la clase c es la probabilidad de excitación del qubit c. Lectura `\"head\"`: una capa lineal "
        "softmax sobre los cinco qg_Z. El filtro conserva los disparos con exactamente una excitación, así que con T1 "
        "igual las probabilidades de las clases son exactamente las sin ruido. Dígitos 0-4 (5 clases, una por qubit), 4 componentes PCA. En 15 corridas (§110) el filtro sumó 5.3 puntos (lectura por qubit) y 9.1 (capa lineal) en promedio, y 15 y 18 en dígitos; una sola partición, como la de abajo, varía de corrida en corrida. Con iris (3 clases) la diferencia es chica.",
        "big": "## 9. Ocho qubits: el efecto de qang crece con el tamaño en peso 2 (§111)\n\n"
        "La misma QNN con 8 qubits (7 componentes PCA), peso 1 y peso 2 (codificación dual, lectura qg_ZZ), entrenada sin "
        "ruido y ejecutada con T1. En 54 corridas (§111) el filtro mantuvo cada modelo exactamente en su precisión sin "
        "ruido; sin el filtro la pérdida quedó cerca de 3 puntos en peso 1 pero llegó a 26 puntos en peso 2 con 8 "
        "qubits (0.965 contra 0.705). Vino (clases 0 y 1). Entrenar el modelo de peso 2 tarda alrededor de un minuto.",
        "grad": "## 10. Gradientes con T1: lo que el filtro devuelve (§112)\n\n"
        "Circuitos aleatorios que conservan el peso, gradiente del qg_Z del qubit 0 respecto del primer ángulo. Con qang "
        "el gradiente es exactamente el sin ruido; sin qang es K veces más chico en peso 1 (K = fracción conservada). Las "
        "últimas columnas son la mediana de disparos necesarios para resolver el gradiente: el filtro es más barato "
        "mientras K no sea muy chico; a mitad de llenado con 8 qubits y mucha profundidad (K < 0.15) la lectura cruda "
        "necesitó menos disparos en §112. Con solo 40 sorteos, como aquí, las medianas de disparos tienen ruido (en §112, con "
        "200 sorteos, n = 6 en peso 3 necesitó 4215 disparos con qang contra 5570 sin qang).",
        "summary": "## Resumen\n\n"
        "* **Entrenada en simulador, ejecutada con T1:** qang es decisivo, exacto con T1 igual, y más cuanto mayor "
        "es el peso.\n"
        "* **Entrenada con el ruido calibrado:** con y sin qang quedan a menos de una muestra de prueba.\n"
        "* **Costo:** una fracción conocida de disparos, 1 - (1 - gamma)^(peso x profundidad).\n"
        "* **Límites:** el T1 desigual y el desfase no se corrigen; no hay ventaja cuántica.\n\n"
        "Detalles, predicciones pre-registradas y fallas: `RESEARCH_NOTES.md` §75–§80, scripts en `examples/qnn_*_qg.py`.",
        "cols": ("lectura", "con qang", "sin qang", "diferencia"),
    },
}

CODE = {
    "install": """import importlib, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "qang>=0.6.16", "scikit-learn"], check=False)
importlib.invalidate_caches()
import qang.qml
if not (hasattr(qang.qml, "MultiClassQNN") and hasattr(qang.qml.WeightQNN, "block_unitaries")):  # multiclass is on GitHub before the next PyPI release
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                    "git+https://github.com/Viny2030/qang.git"], check=True)
    print("installed qang from GitHub; if an import fails below, restart the runtime once")
import qang
print('qang', qang.__version__)""",
    "data": """import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qang.qml import WeightQNN, compare_qang, kept_fraction

d = load_breast_cancer()
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(d.target == c)[0], 100, replace=False) for c in (0, 1)])
Xtr, Xte, ytr, yte = train_test_split(d.data[idx], d.target[idx], test_size=0.3, stratify=d.target[idx], random_state=0)
sc = StandardScaler().fit(Xtr); pca = PCA(4, random_state=0).fit(sc.transform(Xtr))
Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
lo, hi = Ztr.min(0), Ztr.max(0)
scale = lambda Z: np.clip(2 * (Z - lo) / (hi - lo) - 1, -1, 1)
Xtr, Xte = scale(Ztr), scale(Zte)
print(Xtr.shape, Xte.shape)

def show(rows, cols=COLS):
    print(f"{cols[0]:<34}{cols[1]:>11}{cols[2]:>13}{cols[3]:>12}")
    for name, q, r in rows:
        print(f"{name:<34}{q:>11.3f}{r:>13.3f}{q - r:>+12.3f}")""",
    "w1": """GAMMA = 0.08
m1 = WeightQNN(n_qubits=5, weight=1).fit(Xtr, ytr, epochs=120, seed=0)
exact1 = m1.score(Xte, yte)
show([("weight 1, T1, trained clean", m1.score(Xte, yte, gamma=GAMMA, qang=True), m1.score(Xte, yte, gamma=GAMMA, qang=False))])
print(f"noiseless accuracy {exact1:.3f}; kept fraction with qang {kept_fraction(GAMMA, 1, m1.depth):.3f}")""",
    "w2": """m2 = WeightQNN(n_qubits=5, weight=2).fit(Xtr, ytr, epochs=120, seed=0)
exact2 = m2.score(Xte, yte)
show([("weight 2, T1, trained clean", m2.score(Xte, yte, gamma=GAMMA, qang=True), m2.score(Xte, yte, gamma=GAMMA, qang=False))])
print(f"noiseless accuracy {exact2:.3f}; kept fraction with qang {kept_fraction(GAMMA, 2, m2.depth):.3f}")""",
    "aware": """r = compare_qang(WeightQNN(5, 2), Xtr, ytr, Xte, yte, gamma=GAMMA, epochs=120, seed=0, noise_aware=True)
show([("weight 2, T1, trained under noise", r["with qang"], r["without qang"])])""",
    "real": """unequal = [0.04, 0.06, 0.08, 0.10, 0.12]
show([
    ("weight 1, unequal T1", m1.score(Xte, yte, gamma=unequal, qang=True), m1.score(Xte, yte, gamma=unequal, qang=False)),
    ("weight 1, T1 + dephasing 0.03", m1.score(Xte, yte, gamma=GAMMA, dephasing=0.03, qang=True), m1.score(Xte, yte, gamma=GAMMA, dephasing=0.03, qang=False)),
    ("weight 2, unequal T1", m2.score(Xte, yte, gamma=unequal, qang=True), m2.score(Xte, yte, gamma=unequal, qang=False)),
    ("weight 2, T1 + dephasing 0.03", m2.score(Xte, yte, gamma=GAMMA, dephasing=0.03, qang=True), m2.score(Xte, yte, gamma=GAMMA, dephasing=0.03, qang=False)),
])
print(f"noiseless: weight 1 {exact1:.3f}, weight 2 {exact2:.3f}")""",
    "multi": """from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from qang.qml import MultiClassQNN
dd = load_digits(); keep = dd.target < 5
Ar, Ae, br, be = train_test_split(dd.data[keep], dd.target[keep], test_size=0.3, stratify=dd.target[keep], random_state=0)
sd = StandardScaler().fit(Ar); pc = PCA(4, random_state=0).fit(sd.transform(Ar))
Ar, Ae = pc.transform(sd.transform(Ar)), pc.transform(sd.transform(Ae))
mm = MinMaxScaler((-1, 1)).fit(Ar); Ar, Ae = mm.transform(Ar), np.clip(mm.transform(Ae), -1, 1)
rows = []
for ro in ("qubit", "head"):
    mc = MultiClassQNN(n_qubits=5, n_classes=5, readout=ro).fit(Ar, br, epochs=120, seed=0)
    print(f"readout {ro}: noiseless accuracy {mc.score(Ae, be):.3f}")
    rows += [(f"{ro}: T1, trained clean", mc.score(Ae, be, gamma=GAMMA, qang=True), mc.score(Ae, be, gamma=GAMMA, qang=False)),
             (f"{ro}: unequal T1", mc.score(Ae, be, gamma=unequal, qang=True), mc.score(Ae, be, gamma=unequal, qang=False)),
             (f"{ro}: T1, 200 shots", mc.score(Ae, be, gamma=GAMMA, qang=True, shots=200, seed=1),
              mc.score(Ae, be, gamma=GAMMA, qang=False, shots=200, seed=1))]
show(rows)""",
    "big": """from sklearn.datasets import load_wine
dw = load_wine(); kw = dw.target < 2
Br, Be, cr, ce = train_test_split(dw.data[kw], dw.target[kw], test_size=0.3, stratify=dw.target[kw], random_state=0)
sw = StandardScaler().fit(Br); pw = PCA(7, random_state=0).fit(sw.transform(Br))
Br, Be = pw.transform(sw.transform(Br)), pw.transform(sw.transform(Be))
lo8, hi8 = Br.min(0), Br.max(0)
Br, Be = np.clip(2 * (Br - lo8) / (hi8 - lo8) - 1, -1, 1), np.clip(2 * (Be - lo8) / (hi8 - lo8) - 1, -1, 1)
unequal8 = GAMMA * (1 + 0.5 * np.linspace(-1, 1, 8))
rows = []
for label, kw8 in (("8 qubits, weight 1", dict(weight=1)), ("8 qubits, weight 2", dict(weight=2, encoding="dual", readout="zz"))):
    m8 = WeightQNN(n_qubits=8, **kw8).fit(Br, cr, epochs=120, seed=0)
    print(f"{label}: noiseless accuracy {m8.score(Be, ce):.3f}, kept fraction with qang {kept_fraction(GAMMA, m8.weight, m8.depth):.3f}")
    rows += [(f"{label}, T1", m8.score(Be, ce, gamma=GAMMA, qang=True), m8.score(Be, ce, gamma=GAMMA, qang=False)),
             (f"{label}, unequal T1", m8.score(Be, ce, gamma=unequal8, qang=True), m8.score(Be, ce, gamma=unequal8, qang=False))]
show(rows)""",
    "grad": """def grad_cell(n, k, L, draws=40, gamma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    g = WeightQNN(n, k, layers=L); K = kept_fraction(gamma, k, g.depth); h = 1e-5
    out = {"noiseless": [], "with qang": [], "without qang": []}; shots = {"with qang": [], "without qang": []}
    for _ in range(draws):
        th = rng.uniform(-np.pi, np.pi, g.n_theta); psi = np.zeros((1, g.dim))
        psi[0, g.idx[k]] = rng.normal(size=len(g.idx[k])); psi /= np.linalg.norm(psi)
        def f(t, gam, q):
            t2 = th.copy(); t2[0] = t
            return float(g.qg_z(g.probs(t2, psi, gam), q)[0, 0])
        out["noiseless"].append((f(th[0] + h, None, False) - f(th[0] - h, None, False)) / (2 * h))
        for lab, q, r in (("with qang", True, K), ("without qang", False, 1.0)):
            out[lab].append((f(th[0] + h, gamma, q) - f(th[0] - h, gamma, q)) / (2 * h))
            fp, fm = f(th[0] + 0.3, gamma, q), f(th[0] - 0.3, gamma, q)
            shots[lab].append(4 * ((1 - fp**2) + (1 - fm**2)) / (max((fp - fm)**2, 1e-300) * r))
    v0 = np.var(out["noiseless"])
    return K, v0, np.var(out["with qang"]) / v0, np.var(out["without qang"]) / v0, np.median(shots["with qang"]), np.median(shots["without qang"])
print(f"{'circuit':<22}{'K':>7}{'Var(grad)':>11}{'ratio qang':>12}{'ratio raw':>11}{'shots qang':>12}{'shots raw':>11}")
for n, k, L in ((4, 1, 8), (6, 1, 12), (6, 3, 6), (8, 1, 8)):
    K, v0, rq, rr, sq, sr = grad_cell(n, k, L)
    print(f"n={n}, weight {k}, L={L:<5}{K:>7.3f}{v0:>11.2e}{rq:>12.3f}{rr:>11.3f}{sq:>12.0f}{sr:>11.0f}")""",
    "shots": """show([
    ("weight 1, T1, 200 shots", m1.score(Xte, yte, gamma=GAMMA, qang=True, shots=200, seed=1), m1.score(Xte, yte, gamma=GAMMA, qang=False, shots=200, seed=1)),
    ("weight 2, T1, 200 shots", m2.score(Xte, yte, gamma=GAMMA, qang=True, shots=200, seed=1), m2.score(Xte, yte, gamma=GAMMA, qang=False, shots=200, seed=1)),
])""",
}


def build(lang):
    t = T[lang]
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(t["title"])]
    for key in ("install", "data", "w1", "w2", "aware", "real", "shots", "multi", "big", "grad"):
        cells.append(nbf.v4.new_markdown_cell(t[key]))
        code = CODE[key].replace("COLS", repr(t["cols"]))
        cells.append(nbf.v4.new_code_cell(code))
    cells.append(nbf.v4.new_markdown_cell(t["summary"]))
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}
    return nb


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "notebooks"  # run from the repository root: python tools/<this file>
    for lang in T:
        nbf.write(build(lang), f"{out}/{T[lang]['file']}")
        print("wrote", T[lang]["file"])
