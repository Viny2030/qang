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
    "install": '!pip install -q -U "qang>=0.6.0" scikit-learn\n'
    "import qang\nprint('qang', qang.__version__)",
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
    "shots": """show([
    ("weight 1, T1, 200 shots", m1.score(Xte, yte, gamma=GAMMA, qang=True, shots=200, seed=1), m1.score(Xte, yte, gamma=GAMMA, qang=False, shots=200, seed=1)),
    ("weight 2, T1, 200 shots", m2.score(Xte, yte, gamma=GAMMA, qang=True, shots=200, seed=1), m2.score(Xte, yte, gamma=GAMMA, qang=False, shots=200, seed=1)),
])""",
}


def build(lang):
    t = T[lang]
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(t["title"])]
    for key in ("install", "data", "w1", "w2", "aware", "real", "shots"):
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
