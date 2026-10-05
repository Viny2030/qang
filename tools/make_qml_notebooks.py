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
if not hasattr(qang.qml, "MultiClassQNN"):  # multiclass is on GitHub before the next PyPI release
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
    "shots": """show([
    ("weight 1, T1, 200 shots", m1.score(Xte, yte, gamma=GAMMA, qang=True, shots=200, seed=1), m1.score(Xte, yte, gamma=GAMMA, qang=False, shots=200, seed=1)),
    ("weight 2, T1, 200 shots", m2.score(Xte, yte, gamma=GAMMA, qang=True, shots=200, seed=1), m2.score(Xte, yte, gamma=GAMMA, qang=False, shots=200, seed=1)),
])""",
}


def build(lang):
    t = T[lang]
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(t["title"])]
    for key in ("install", "data", "w1", "w2", "aware", "real", "shots", "multi"):
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
