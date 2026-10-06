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
        "echo": "## 11. Class confusions inside the sector: filter plus echo calibration (§113)\n\n"
        "The filter removes the shots that left the sector, but not the errors that move an excitation to another qubit "
        "*inside* the sector: in the multiclass QNN those swap one class for another. An **echo** (the circuit followed "
        "by its inverse, one run per sector state) measures that mixing as a transfer matrix M; `unmix_sector` undoes "
        "half of it (M^1/2) after the filter. Below, a synthetic device: T1 plus nearest-neighbour hops of 6-16% "
        "(echo diagonal 0.74-0.82, as on the device models of §113), echo measured with 2000 shots per state. "
        "Accuracy and, in brackets, decisions that differ from the noiseless model.\n\n"
        "Measured in §113 on five device noise models (100 digits each, 1000 shots, noiseless 0.82 / 0.91):\n\n"
        "| readout | without qang | qang | qang + echo |\n|---|---|---|---|\n"
        "| qubit | 0.792 (46 flips) | 0.818 (19) | 0.836 (15) |\n"
        "| head | 0.858 (51) | 0.890 (24) | 0.900 (15) |\n\n"
        "The filter does most of the work; the echo removes a further third of the flips. Per backend the counts are "
        "small and move both ways (E2 and E4 failed on one or two backends).",
        "multi8": "## 12. Eight classes on eight qubits (§114)\n\n"
        "`MultiClassQNN` with 8 qubits, one class per qubit, digits 0-7 (100 per class, 7 PCA features), head readout, trained "
        "without noise and run under T1, plus the model trained under T1 without qang. Over 30 runs (§114) the filter "
        "kept every model exactly at its noiseless accuracy; without it the head readout lost 11.1 points with 8 classes "
        "(15.9 with 5, 0.7 with 3; large spread between seeds), and the model trained under the noise without qang stayed "
        "about 3 points behind. The single split below loses more than that average (seeds ranged from 2 to 34 points with 5 classes). The two trainings take about four minutes.",
        "t1fix": "## 13. Correcting unequal T1: train with qang under the calibrated rates (§117)\n\n"
        "Conditioned on no decay, the kept state depends only on the *ratios* of the decay rates 1 - gamma_q, so a model "
        "trained with the filter under the device's calibrated T1s does not need their absolute level. Weight 2 (dual "
        "encoding, qg_ZZ readout), decay 0 to 0.16 across the qubits, calibration with a 10% error, and a drift in which "
        "every T1 gets 1.5 times shorter. Over 40 runs (§117) this model was within 0.5 points of noiseless (filter alone: "
        "2.7 points below) and, at weight 2, lost 0.7 points under the drift against 10 for the raw noise-aware model. "
        "Below: one split, mean of three initializations (one test sample is worth 1.7 points). In this split calibrated training lifts the filtered model from 0.872 to 0.939 (noiseless 0.944); the drift effect of §117 does not show here (one dataset, 60 test inputs). The nine trainings take about eight minutes.",
        "zne": "## 14. Dephasing: filter plus zero-noise extrapolation (§118)\n\n"
        "With equal T1 the filtered readout equals the readout with dephasing alone (exactly), so the filter leaves a "
        "single noise to extrapolate. ZNE runs the circuit at noise scales 1 and 2 (as gate folding would) and "
        "extrapolates every feature to zero noise. Weight 2 (dual, qg_ZZ), T1 0.08 plus dephasing 0.03. Over 40 runs "
        "(§118) filter + ZNE beat raw + ZNE by 3-10 points and, with exact probabilities, came within 1.3 points of "
        "noiseless; with 1000 shots per scale its variance cancelled the gain at weight 2. Training with the filter "
        "under the calibrated noise recovered the noiseless accuracy. At 8 qubits (§122) the same holds. Below: one split, mean of three initializations; "
        "the first column of the last row is the noise-aware model, compared with the clean model read with the filter. "
        "The six trainings take about four minutes.",
        "device": "## 15. The whole pipeline on an IBM device noise model (§119)\n\n"
        "The model of section 2 compiled to Qiskit (a loader cascade of 4 RBS gates, then the 15 trained RBS gates) and run on "
        "the FakeBrisbane noise model (Aer), 1000 shots per input, with 5 echo circuits per model. A second model is trained "
        "with qang under the device's published T1 and T2 (converted to decay and dephasing per sublayer). Over 189 inputs "
        "on three IBM noise models (§119) the filter lifted the accuracy from 0.958 to 0.972 (noiseless 0.979), the echo cut "
        "the decision error without changing decisions, and the T1/T2-calibrated training added nothing (0.968): on these "
        "devices gate errors dominate, and the published decay is only 0.002-0.017 per sublayer. Calibrating with the device itself (echo-aware training, or refitting the readout on 40 device inputs) did not do better either (§120). In this split (60 inputs, one input = 1.7 points) the accuracy is the same with "
        "and without qang and the echo moves one decision the wrong way, while the decision error falls from 1.72 (raw) to "
        "0.95 (filter) and 0.50 (filter + echo). Installs Qiskit; about four minutes.",
        "pull": "## 16. Why the loss without qang varies between seeds (§121)\n\n"
        "At weight 1 under equal T1 every decayed shot lands in |0...0>, where every qg_Z is +1. So the raw logits are exactly "
        "K L + (1 - K) v: the noiseless logits L, scaled by the kept fraction K, plus a constant vector v fixed by training "
        "(v = W^T 1 + b for the head readout). Without the filter every input is pulled toward the class with the largest v_c. "
        "Below, the head model of section 8: the raw decisions predicted by this formula match the simulated ones, and v "
        "shows the pull. Over 20 seeds (§121) a pull index built from v and the training margins ranked the seeds by their "
        "loss without qang (Spearman 0.65 for the head, 0.98 for the qubit readout; losses of 3-42 points). No training here.",
        "summary": "## Summary\n\n"
        "* **Trained on a simulator, run under T1:** qang is decisive, exact under equal T1, and more so at higher "
        "weight.\n"
        "* **Trained under the calibrated noise:** with and without qang are within about one test sample.\n"
        "* **Cost:** a known fraction of the shots, 1 - (1 - gamma)^(weight x depth).\n"
        "* **Errors inside the sector:** the filter does not see them; an echo calibration removes part of them (§113).\n"
        "* **Eight classes on eight qubits:** with qang exact; without it the head readout loses about 11 points (§114).\n"
        "* **Unequal T1:** corrected by training with qang under the calibrated rates, which only need the ratios of the T1s (§117).\n"
        "* **Dephasing:** the filter leaves it as the only noise; ZNE of the filtered readout removes most of its bias with enough shots, and training with the filter under the calibrated noise corrects it (§118).\n"
        "* **On IBM device noise models:** the filter gives most of the gain, the echo cuts the decision error, and training under the published T1/T2 or a device calibration adds nothing, since gate errors dominate (§119, §120).\n"
        "* **Seed-to-seed spread:** without the filter the raw logits are K L + (1 - K) v, a trained constant pull that predicts the loss of each seed (§121).\n"
        "* **Limits:** unequal T1 and dephasing are corrected only with calibrated training (or ZNE with enough shots); no quantum advantage.\n\n"
        "Details, pre-registered predictions and failures: `RESEARCH_NOTES.md` §75–§80, §109–§114, scripts in `examples/qnn_*_qg.py`.",
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
        "echo": "## 11. Confusiones de clase dentro del sector: filtro más calibración por eco (§113)\n\n"
        "El filtro elimina los disparos que salieron del sector, pero no los errores que mueven una excitación a otro "
        "qubit *dentro* del sector: en la QNN multiclase esos cambian una clase por otra. Un **eco** (el circuito seguido "
        "de su inverso, una corrida por estado del sector) mide esa mezcla como una matriz de transferencia M; "
        "`unmix_sector` deshace la mitad (M^1/2) después del filtro. Abajo, un dispositivo sintético: T1 más saltos a "
        "vecinos de 6-16% (diagonal del eco 0.74-0.82, como en los modelos de dispositivo de §113), eco medido con 2000 "
        "disparos por estado. Exactitud y, entre paréntesis, decisiones que difieren del modelo sin ruido.\n\n"
        "Medido en §113 sobre cinco modelos de ruido de dispositivo (100 dígitos cada uno, 1000 disparos, sin ruido 0.82 / 0.91):\n\n"
        "| lectura | sin qang | qang | qang + eco |\n|---|---|---|---|\n"
        "| qubit | 0.792 (46 cambios) | 0.818 (19) | 0.836 (15) |\n"
        "| cabeza | 0.858 (51) | 0.890 (24) | 0.900 (15) |\n\n"
        "El filtro hace la mayor parte del trabajo; el eco quita un tercio más de los cambios. Por backend los conteos "
        "son chicos y se mueven en ambos sentidos (E2 y E4 fallaron en uno o dos backends).",
        "multi8": "## 12. Ocho clases en ocho qubits (§114)\n\n"
        "`MultiClassQNN` con 8 qubits, una clase por qubit, dígitos 0-7 (100 por clase, 7 componentes PCA), lectura cabeza, entrenada "
        "sin ruido y ejecutada con T1, más el modelo entrenado con T1 sin qang. En 30 corridas (§114) el filtro mantuvo "
        "cada modelo exactamente en su precisión sin ruido; sin él la lectura cabeza perdió 11.1 puntos con 8 clases "
        "(15.9 con 5, 0.7 con 3; mucha dispersión entre semillas), y el modelo entrenado con el ruido sin qang quedó "
        "unos 3 puntos atrás. La partición de abajo pierde más que ese promedio (con 5 clases las semillas fueron de 2 a 34 puntos). Los dos entrenamientos tardan unos cuatro minutos.",
        "t1fix": "## 13. Corregir el T1 desigual: entrenar con qang con las tasas calibradas (§117)\n\n"
        "Condicionado a que no hubo decaimiento, el estado conservado depende solo de los *cocientes* de las tasas 1 - gamma_q, "
        "así que un modelo entrenado con el filtro con los T1 calibrados del equipo no necesita su nivel absoluto. Peso 2 "
        "(codificación dual, lectura qg_ZZ), decaimiento de 0 a 0.16 entre qubits, calibración con 10 % de error y una deriva "
        "en la que todos los T1 se acortan 1.5 veces. En 40 corridas (§117) este modelo quedó a menos de 0.5 puntos del sin "
        "ruido (el filtro solo: 2.7 puntos abajo) y, en peso 2, perdió 0.7 puntos con la deriva contra 10 del modelo crudo "
        "entrenado con ruido. Abajo: una partición, promedio de tres inicializaciones (una muestra de prueba vale 1.7 puntos). En esta partición el entrenamiento calibrado sube el modelo filtrado de 0.872 a 0.939 (sin ruido 0.944); el efecto de la deriva de §117 no aparece aquí (un conjunto, 60 entradas de prueba). Los nueve entrenamientos tardan unos ocho minutos.",
        "zne": "## 14. Desfase: filtro más extrapolación a ruido cero (§118)\n\n"
        "Con T1 igual, la lectura filtrada es igual a la lectura con desfase solo (exactamente), así que el filtro deja un "
        "único ruido para extrapolar. La ZNE corre el circuito con escalas de ruido 1 y 2 (como lo haría el plegado de "
        "compuertas) y extrapola cada variable a ruido cero. Peso 2 (dual, qg_ZZ), T1 0.08 más desfase 0.03. En 40 corridas "
        "(§118) filtro + ZNE superó a crudo + ZNE por 3-10 puntos y, con probabilidades exactas, quedó a 1.3 puntos del sin "
        "ruido; con 1000 disparos por escala su varianza anuló la ganancia en peso 2. Entrenar con el filtro con el ruido "
        "calibrado recuperó la precisión sin ruido. Con 8 qubits (§122) vale lo mismo. Abajo: una partición, promedio de tres inicializaciones; la primera "
        "columna de la última fila es el modelo entrenado con ruido, comparado con el modelo limpio leído con el filtro. "
        "Los seis entrenamientos tardan unos cuatro minutos.",
        "device": "## 15. Todo el proceso en un modelo de ruido de un equipo IBM (§119)\n\n"
        "El modelo de la sección 2 compilado a Qiskit (una cascada de carga de 4 compuertas RBS y luego las 15 RBS entrenadas) "
        "y ejecutado con el modelo de ruido FakeBrisbane (Aer), 1000 disparos por entrada, con 5 circuitos de eco por modelo. "
        "Un segundo modelo se entrena con qang con los T1 y T2 publicados del equipo (convertidos a decaimiento y desfase por "
        "subcapa). En 189 entradas sobre tres modelos de ruido de IBM (§119) el filtro subió la precisión de 0.958 a 0.972 (sin "
        "ruido 0.979), el eco bajó el error del valor de decisión sin cambiar decisiones, y el entrenamiento calibrado con T1/T2 "
        "no agregó nada (0.968): en estos equipos dominan los errores de compuerta, y el decaimiento publicado es de solo "
        "0.002-0.017 por subcapa. Calibrar con el propio equipo (entrenamiento con el eco, o reajustar la lectura con 40 entradas corridas en el equipo) tampoco mejoró (§120). En esta partición (60 entradas, una entrada = 1.7 puntos) la precisión es la misma con y sin "
        "qang y el eco mueve una decisión en contra, mientras el error del valor de decisión baja de 1.72 (crudo) a 0.95 (filtro) "
        "y 0.50 (filtro + eco). Instala Qiskit; unos cuatro minutos.",
        "pull": "## 16. Por qué la pérdida sin qang varía entre semillas (§121)\n\n"
        "En peso 1 con T1 igual, cada disparo decaído cae en |0...0>, donde todo qg_Z vale +1. Entonces los logits crudos son "
        "exactamente K L + (1 - K) v: los logits sin ruido L, escalados por la fracción conservada K, más un vector constante v "
        "fijado por el entrenamiento (v = W^T 1 + b en la lectura cabeza). Sin el filtro cada entrada es arrastrada hacia la "
        "clase con el v_c más grande. Abajo, el modelo cabeza de la sección 8: las decisiones crudas que predice la fórmula "
        "coinciden con las simuladas, y v muestra el arrastre. En 20 semillas (§121) un índice de arrastre construido con v y "
        "los márgenes de entrenamiento ordenó las semillas por su pérdida sin qang (Spearman 0.65 en cabeza, 0.98 en qubit; "
        "pérdidas de 3 a 42 puntos). Sin entrenamiento.",
        "summary": "## Resumen\n\n"
        "* **Entrenada en simulador, ejecutada con T1:** qang es decisivo, exacto con T1 igual, y más cuanto mayor "
        "es el peso.\n"
        "* **Entrenada con el ruido calibrado:** con y sin qang quedan a menos de una muestra de prueba.\n"
        "* **Costo:** una fracción conocida de disparos, 1 - (1 - gamma)^(peso x profundidad).\n"
        "* **Errores dentro del sector:** el filtro no los ve; una calibración por eco quita una parte (§113).\n"
        "* **Ocho clases en ocho qubits:** con qang exacto; sin él la lectura cabeza pierde unos 11 puntos (§114).\n"
        "* **T1 desigual:** se corrige entrenando con qang con las tasas calibradas, que solo necesitan los cocientes de los T1 (§117).\n"
        "* **Desfase:** el filtro lo deja como único ruido; la ZNE de la lectura filtrada quita la mayor parte de su sesgo con suficientes disparos, y entrenar con el filtro con el ruido calibrado lo corrige (§118).\n"
        "* **En modelos de ruido de equipos IBM:** el filtro da la mayor parte de la ganancia, el eco baja el error del valor de decisión, y entrenar con los T1/T2 publicados o una calibración del equipo no agrega nada, porque dominan los errores de compuerta (§119, §120).\n"
        "* **Dispersión entre semillas:** sin el filtro los logits crudos son K L + (1 - K) v, un arrastre constante entrenado que predice la pérdida de cada semilla (§121).\n"
        "* **Límites:** el T1 desigual y el desfase se corrigen solo con entrenamiento calibrado (o ZNE con suficientes disparos); no hay ventaja cuántica.\n\n"
        "Detalles, predicciones pre-registradas y fallas: `RESEARCH_NOTES.md` §75–§80, §109–§114, scripts en `examples/qnn_*_qg.py`.",
        "cols": ("lectura", "con qang", "sin qang", "diferencia"),
    },
}

CODE = {
    "install": """import importlib, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "qang>=0.6.17", "scikit-learn"], check=False)
importlib.invalidate_caches()
import qang.qml
if not (hasattr(qang.qml, "MultiClassQNN") and hasattr(qang.qml.WeightQNN, "block_unitaries")):  # older qang without the multiclass model: install from GitHub
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
    "echo": """from qang.sectors import sector_states, echo_transfer_matrix, unmix_sector
n = 5; S = sector_states(n, 1)  # the 5 one-excitation states = the 5 classes
rng = np.random.default_rng(0)
eps = rng.uniform(0.06, 0.16, n)  # synthetic in-sector error: the excitation hops to a neighbouring qubit
A = np.eye(n)
for j in range(n):
    for i in (j - 1, j + 1):
        if 0 <= i < n: A[i, j] += eps[j]
A /= A.sum(0, keepdims=True)
def device(P):  # T1 already in P; the hops act inside the sector
    Q = P.copy(); Q[:, S] = P[:, S] @ A.T; return Q
echo = []  # echo = circuit + inverse: twice the hops (A @ A), some leakage, 2000 shots per prepared state
for j in range(n):
    p = np.zeros(2**n); p[0] = 0.15; p[S] = 0.85 * (A @ A)[:, j]
    echo.append(rng.multinomial(2000, p))
M = echo_transfer_matrix(echo, n, 1)
print("echo diagonal", np.round(np.diag(M), 2))
def decide(mc, P, mode):
    if mode == "qang + echo": P = np.array([unmix_sector(p, M, n, 1) for p in P])
    return mc._logits(mc.qg_z(P, mode != "without qang"), mc.params_[mc.n_theta:]).argmax(1)
modes = ("without qang", "qang", "qang + echo")
print(f"{'':<22}" + "".join(f"{m:>16}" for m in modes))
for ro in ("qubit", "head"):
    mc = MultiClassQNN(n_qubits=5, n_classes=5, readout=ro).fit(Ar, br, epochs=120, seed=0)
    ref = mc.predict(Ae)
    P = device(mc.probs(mc.params_[:mc.n_theta], mc.encode(Ae), GAMMA))
    Ps = np.array([rng.multinomial(1000, p / p.sum()) / 1000 for p in P])
    for label, PP in (("exact", P), ("1000 shots", Ps)):
        cells = [decide(mc, PP, m) for m in modes]
        print(f"{ro + ', ' + label:<22}" + "".join(f"{np.mean(c == be):>10.3f} ({int(np.sum(c != ref)):>2})" for c in cells))
    print(f"{ro:<22} noiseless accuracy {np.mean(ref == be):.3f}")""",
    "multi8": """d8 = load_digits(); r8 = np.random.default_rng(0)
k8 = np.concatenate([r8.choice(np.where(d8.target == c)[0], 100, replace=False) for c in range(8)])  # 100 per class
Er, Ee, er, ee = train_test_split(d8.data[k8], d8.target[k8], test_size=0.3, stratify=d8.target[k8], random_state=0)
s8 = StandardScaler().fit(Er); p8 = PCA(7, random_state=0).fit(s8.transform(Er))
Er, Ee = p8.transform(s8.transform(Er)), p8.transform(s8.transform(Ee))
m8s = MinMaxScaler((-1, 1)).fit(Er); Er, Ee = m8s.transform(Er), np.clip(m8s.transform(Ee), -1, 1)
mc8 = MultiClassQNN(n_qubits=8, n_classes=8, readout="head").fit(Er, er, epochs=120, seed=0)
pna8 = MultiClassQNN(n_qubits=8, n_classes=8, readout="head").fit(Er, er, epochs=120, gamma=GAMMA, qang=False, seed=0).params_
print(f"8 classes, head readout: noiseless accuracy {mc8.score(Ee, ee):.3f}")
show([("8 classes, T1, trained clean", mc8.score(Ee, ee, gamma=GAMMA, qang=True), mc8.score(Ee, ee, gamma=GAMMA, qang=False)),
      ("8 classes, unequal T1", mc8.score(Ee, ee, gamma=unequal8, qang=True), mc8.score(Ee, ee, gamma=unequal8, qang=False)),
      ("qang clean vs trained under T1", mc8.score(Ee, ee, gamma=GAMMA, qang=True), mc8.score(Ee, ee, pna8, gamma=GAMMA, qang=False))])""",
    "t1fix": """true = 0.08 * (1 + np.linspace(-1, 1, 5))                      # decay per sublayer, 0 to 0.16
cal = true * (1 + 0.10 * np.random.default_rng(7).normal(size=5))  # calibration with a 10% error
drift = 1 - (1 - true) ** 1.5                                        # every T1 1.5x shorter
mk = lambda: WeightQNN(5, 2, encoding="dual", readout="zz")
acc = {k: [] for k in ("noiseless", "clean q", "clean raw", "cal q", "cal raw", "drift q", "drift raw")}
for seed in (0, 1, 2):                                               # one split, three initializations
    mA = mk().fit(Xtr, ytr, epochs=120, seed=seed)                      # trained without noise
    mC = mk().fit(Xtr, ytr, epochs=120, gamma=cal, qang=True, seed=seed)   # with qang, calibrated rates
    mD = mk().fit(Xtr, ytr, epochs=120, gamma=cal, qang=False, seed=seed)  # without qang, calibrated rates
    for k, v in (("noiseless", mA.score(Xte, yte)),
                 ("clean q", mA.score(Xte, yte, gamma=true, qang=True)), ("clean raw", mA.score(Xte, yte, gamma=true, qang=False)),
                 ("cal q", mC.score(Xte, yte, gamma=true, qang=True)), ("cal raw", mD.score(Xte, yte, gamma=true, qang=False)),
                 ("drift q", mC.score(Xte, yte, gamma=drift, qang=True)), ("drift raw", mD.score(Xte, yte, gamma=drift, qang=False))):
        acc[k].append(v)
a_ = {k: float(np.mean(v)) for k, v in acc.items()}
print(f"noiseless accuracy {a_['noiseless']:.3f} (mean of 3 initializations)")
show([("unequal T1, trained clean", a_["clean q"], a_["clean raw"]),
      ("unequal T1, trained calibrated", a_["cal q"], a_["cal raw"]),
      ("T1 drift 1.5x, trained calibrated", a_["drift q"], a_["drift raw"])])""",
    "zne": """P_DEPH = 0.03
def zne_features(m, th, psi, qang, shots=None, rng=None):
    f = []
    for lam in (1, 2):                                        # noise scale, as by gate folding
        g, p = 1 - (1 - GAMMA) ** lam, (1 - (1 - 2 * P_DEPH) ** lam) / 2
        pr = m.probs(th, psi, g, p)
        if shots:
            pr = np.array([rng.multinomial(shots, q / q.sum()) / shots for q in pr])
        f.append(m.qg_z(pr, qang))
    return f[0], np.clip(2 * f[0] - f[1], -1, 1)             # scale 1, and linear extrapolation to zero
acc = lambda m, P, F: float(np.mean(((F @ P[m.n_theta:-1] + P[-1]) > 0) == yte))
res = {k: [] for k in ("noiseless", "raw", "filter", "raw+ZNE", "filter+ZNE", "raw s", "filter s", "raw+ZNE s", "filter+ZNE s", "aware")}
for seed in (0, 1, 2):
    mz = WeightQNN(5, 2, encoding="dual", readout="zz").fit(Xtr, ytr, epochs=120, seed=seed)
    P = mz.params_; th, psi = P[: mz.n_theta], mz.encode(Xte); rng = np.random.default_rng(seed)
    res["noiseless"].append(mz.score(Xte, yte))
    for qang, tag in ((False, "raw"), (True, "filter")):
        f1, f0 = zne_features(mz, th, psi, qang)
        res[tag].append(acc(mz, P, f1)); res[tag + "+ZNE"].append(acc(mz, P, f0))
        f1, f0 = zne_features(mz, th, psi, qang, 1000, rng)
        res[tag + " s"].append(acc(mz, P, f1)); res[tag + "+ZNE s"].append(acc(mz, P, f0))
    ma = WeightQNN(5, 2, encoding="dual", readout="zz").fit(Xtr, ytr, epochs=120, gamma=GAMMA, dephasing=P_DEPH, qang=True, seed=seed)
    res["aware"].append(ma.score(Xte, yte, gamma=GAMMA, dephasing=P_DEPH, qang=True))
r = {k: float(np.mean(v)) for k, v in res.items()}
print(f"noiseless accuracy {r['noiseless']:.3f} (mean of 3 initializations)")
show([("T1 + dephasing, exact", r["filter"], r["raw"]),
      ("T1 + dephasing + ZNE, exact", r["filter+ZNE"], r["raw+ZNE"]),
      ("T1 + dephasing, 1000 shots", r["filter s"], r["raw s"]),
      ("T1 + dephasing + ZNE, 1000 shots", r["filter+ZNE s"], r["raw+ZNE s"]),
      ("trained with qang under noise vs clean", r["aware"], r["filter"])])""",
    "device": """import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "qiskit", "qiskit-aer", "qiskit-ibm-runtime"], check=False)
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qang.qml import WeightQNN
from qang.sectors import echo_transfer_matrix, unmix_sector

Ir, Ie, jr, je = Xtr, Xte, ytr, yte                                     # the breast-cancer split of section 2

def rbs(t):  # RBS on [a, b] in Qiskit order
    U = np.eye(4); c, s = math.cos(t), math.sin(t); U[1, 1], U[2, 2], U[2, 1], U[1, 2] = c, c, s, -s
    return UnitaryGate(U, label="RBS")
def blocks(m, th, reverse=False):
    g, k = [], 0
    for _ in range(m.layers):
        for pairs in m.sublayers:
            for a, b in pairs:
                g.append((th[k], a, b)); k += 1
    return g
q = lambda i: 4 - i                                                     # qang qubit i = Qiskit qubit 4-i
def circuit(m, x):                                                      # loader cascade + trained RBS block
    qc = QuantumCircuit(5); v = np.append(x, 1.0); v /= np.linalg.norm(v); qc.x(q(0))
    ang = [math.atan2(np.linalg.norm(v[k + 1:]), v[k]) for k in range(3)] + [math.atan2(v[4], v[3])]
    for k, t in enumerate(ang): qc.append(rbs(t), [q(k), q(k + 1)])
    for t, a, b in blocks(m, m.params_[: m.n_theta]): qc.append(rbs(t), [q(a), q(b)])
    qc.measure_all(); return qc
def echo(m, j):                                                         # qubit j excited, block, inverse
    qc = QuantumCircuit(5); qc.x(q(j)); g = blocks(m, m.params_[: m.n_theta])
    for t, a, b in g: qc.append(rbs(t), [q(a), q(b)])
    for t, a, b in reversed(g): qc.append(rbs(-t), [q(a), q(b)])
    qc.measure_all(); return qc
def to_probs(c):
    p = np.zeros(32); tot = sum(c.values())
    for b, n in c.items(): p[int(b.replace(" ", ""), 2)] += n / tot
    return p

dev = FakeBrisbane(); sim = AerSimulator.from_backend(dev)
mA = WeightQNN(5, 1).fit(Ir, jr, epochs=120, seed=0)                   # trained without noise
tc0 = transpile(circuit(mA, Ie[0]), backend=dev, optimization_level=1, seed_transpiler=1)
layout = list(tc0.layout.final_index_layout()); t = tc0.estimate_duration(dev.target, unit="s") / 13
gam = np.zeros(5); deph = []
for vq, ph in enumerate(layout):                                       # published T1/T2 -> qang noise per sublayer
    qp = dev.target.qubit_properties[ph]; gam[4 - vq] = 1 - math.exp(-t / qp.t1)
    deph.append((1 - math.exp(-t * max(1 / qp.t2 - 1 / (2 * qp.t1), 0))) / 2)
print("calibration: gamma per sublayer", np.round(gam, 4), "dephasing", round(float(np.mean(deph)), 4))
mC = WeightQNN(5, 1).fit(Ir, jr, epochs=120, gamma=gam, dephasing=float(np.mean(deph)), qang=True, seed=0)  # calibrated, with qang
def run(m):
    circs = [circuit(m, x) for x in Ie] + [echo(m, j) for j in range(5)]
    tc = transpile(circs, backend=sim, optimization_level=1, seed_transpiler=1, initial_layout=layout)
    pr = [to_probs(sim.run(c, shots=1000 if i < len(Ie) else 4000, seed_simulator=i).result().get_counts()) for i, c in enumerate(tc)]
    M = echo_transfer_matrix(pr[len(Ie):], 5, 1, prepared=[1 << (4 - j) for j in range(5)])
    w, b = m.params_[m.n_theta:-1], m.params_[-1]
    d0 = m.decision(m.params_, Ie)                                       # noiseless decision values
    P = pr[: len(Ie)]
    F = {"raw": [m.qg_z(p, qang=False)[0] for p in P], "filter": [m.qg_z(p, qang=True)[0] for p in P],
         "filter + echo": [m.qg_z(unmix_sector(p, M, 5, 1), qang=False)[0] for p in P]}
    out = {}
    for k, f in F.items():
        d = np.array(f) @ w + b
        out[k] = (float(np.mean((d > 0) == je)), float(np.mean(np.abs(d - d0))))
    return out
rA, rC = run(mA), run(mC)
print(f"noiseless accuracy {mA.score(Ie, je):.3f}; FakeBrisbane noise model, 1000 shots per input")
show([("trained clean: filter", rA["filter"][0], rA["raw"][0]), ("trained clean: filter + echo", rA["filter + echo"][0], rA["raw"][0]),
      ("T1/T2-calibrated: filter", rC["filter"][0], rC["raw"][0]), ("T1/T2-calibrated: filter + echo", rC["filter + echo"][0], rC["raw"][0])])
print("decision error against the noiseless model (lower is better):")
for tag, r in (("trained clean", rA), ("T1/T2-calibrated", rC)):
    print(f"  {tag:<18} raw {r['raw'][1]:.3f}   filter {r['filter'][1]:.3f}   filter + echo {r['filter + echo'][1]:.3f}")""",
    "pull": """from qang.qml import kept_fraction
K = kept_fraction(GAMMA, 1, mc.depth)                    # mc: the head model of section 8 (5-class digits)
head = mc.params_[mc.n_theta:]
W = head[: mc.n * mc.n_classes].reshape(mc.n, mc.n_classes)
v = W.sum(axis=0) + head[mc.n * mc.n_classes:]           # the constant pull of each class
L = mc.logits(mc.params_, Ae)                            # noiseless logits
pred_raw = np.argmax(K * L + (1 - K) * v, axis=1)        # F7: raw logits = K L + (1 - K) v
sim_raw = mc.predict(Ae, gamma=GAMMA, qang=False)
print(f"kept fraction K = {K:.3f}; pull per class v = {np.round(v, 2)} (largest: class {int(np.argmax(v))})")
print(f"raw decisions predicted by K L + (1 - K) v: {np.mean(pred_raw == sim_raw):.3f} of the inputs agree with the simulation")
show([("5 classes, T1", mc.score(Ae, be, gamma=GAMMA, qang=True), mc.score(Ae, be, gamma=GAMMA, qang=False))])
print("raw decisions per class:", np.bincount(sim_raw, minlength=5), " noiseless:", np.bincount(np.argmax(L, axis=1), minlength=5))""",
    "shots": """show([
    ("weight 1, T1, 200 shots", m1.score(Xte, yte, gamma=GAMMA, qang=True, shots=200, seed=1), m1.score(Xte, yte, gamma=GAMMA, qang=False, shots=200, seed=1)),
    ("weight 2, T1, 200 shots", m2.score(Xte, yte, gamma=GAMMA, qang=True, shots=200, seed=1), m2.score(Xte, yte, gamma=GAMMA, qang=False, shots=200, seed=1)),
])""",
}


def build(lang):
    t = T[lang]
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(t["title"])]
    for key in ("install", "data", "w1", "w2", "aware", "real", "shots", "multi", "big", "grad", "echo", "multi8", "t1fix", "zne", "device", "pull"):
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
