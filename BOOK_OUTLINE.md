# BOOK_OUTLINE.md — Mapa Maestro del Tratado Bilingüe (35 Capítulos + Apéndices A–H)

**Título ES**: *Computación, información y física cuántica en el espacio qang* (`qang-libro-es.pdf` / `book/es/`)[cite: 2, 4]
**Título EN**: *Quantum Computing, Information and Physics in qang Space* (`qang-book-en.pdf` / `book/en/`)[cite: 2, 207]
**Web**: `https://viny2030.github.io/qang/` ([`/es/`](https://viny2030.github.io/qang/es/) · [`/en/`](https://viny2030.github.io/qang/en/))

## Estructura por Partes, Capítulos, Módulos y Scripts de Verificación

### Parte I. Fundamentos: la métrica angular-probabilística
- **Cap. 1**: La unidad qang: definición, mili-qang y la ecuación de probabilidad (`qang.probability`, `qang.qga`)[cite: 5, 23, 31]
  - *Scripts*: `tools/make_cap01_figures.py`, `tools/verify_cap01.py`, `tools/verify_qga_lecturas.py`, `examples/unit_circle_qg.py`, `examples/iqm_milliqang_qg.py` (§168)[cite: 1, 2, 44].
- **Cap. 2**: Compuertas nativas, álgebra cerrada y el operador universal (`qang.algebra`, `qang.pulse_area`, `qang.sectors`, `qang.interop`)[cite: 5, 46, 51, 52, 53]
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap02.py`, `examples/pulse_area_qg.py` (§160), `examples/sector_exposure_qg.py` (§68)[cite: 1].
- **Cap. 3**: Reflexiones, Householder y la dinámica de Grover (`qang.algebra`, `qang.probability`)[cite: 6, 56, 65]
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap03.py`, `examples/grover_noise_qg.py` (§63), `examples/amplitude_estimation_chebyshev_qg.py` (§37)[cite: 1].
- **Cap. 4**: Estadística de muestreo, inferencia bayesiana y la distribución de Haar (`qang.statistics`, `qang.proportions`)[cite: 6, 67, 76]
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap04.py`, `examples/few_shot_tomography_qg.py` (§35), `examples/shadows_vs_direct_qg.py` (§62)[cite: 1].

### Parte II. Geometría local, óptica cuántica y fases geométricas
- **Cap. 5**: El radio local de Bloch como medida de pureza y entrelazamiento (`qang.formulation`, `qang.complementarity`)[cite: 7, 79, 87]
  - *Scripts*: `tools/make_cap05_figures.py`, `tools/verify_cap05.py`, `examples/qg_radius_witness_qg.py` (§98), `examples/complementarity_qg.py` (§127)[cite: 1, 87].
- **Cap. 6**: Polarización óptica y parámetros de Stokes en unidades qg (`qang.polarization`)[cite: 7, 89, 95]
  - *Scripts*: `tools/make_cap06_figures.py`, `tools/verify_cap06.py`, `notebooks/qang_optics[_es].ipynb` (§83)[cite: 1, 96].
- **Cap. 7**: Fases geométricas y el teorema de Stokes en la esfera (`qang.geometric`, `qang.phase`)[cite: 7, 97, 98]
  - *Scripts*: `tools/make_cap07_figures.py`, `tools/verify_cap07.py`, `notebooks/qang_optics[_es].ipynb` (§84)[cite: 1, 102].
- **Cap. 8**: Información de Fisher, relojes, gravedad y metrología (`qang.fisher`, `qang.clock_comparison`)[cite: 7, 104, 106]
  - *Scripts*: `tools/verify_cap08.py`, `examples/fisher_natural_gradient_qg.py` (§159), `examples/clock_redshift_qg.py` (§126), `examples/ramsey_qg_operating_point.py` (§36), `examples/ghz_metrology_qg.py` (§58), `examples/thermal_states_qg_tanh.py` (§25)[cite: 1].

### Parte III. Física de neutrinos, oscilaciones y simulación cuántica (`qang.neutrino`)
- **Cap. 9**: Cinemática cuántica de la oscilación de sabor en el vacío (§151, §165 IQM Garnet)[cite: 8, 115, 118, 120]
- **Cap. 10**: Efectos de materia (MSW), neutrinos solares y oscilaciones colectivas (§152, §154, §156, §163)[cite: 8, 127, 128, 129, 131]
- **Cap. 11**: Fases de Majorana, doble beta sin neutrinos, neutrinos estériles y los cuatro números cuánticos (`qang.angular`, §155, §163)[cite: 8, 136, 138, 140]
  - *Notebooks*: `notebooks/qang_neutrinos[_es].ipynb`, `notebooks/qang_quantum_numbers[_es].ipynb`[cite: 1].

### Parte IV. Teleportación cuántica en redes ópticas (`qang.qpon`)
- **Cap. 12**: Teleportación óptica y el límite del par de Bell (§57, §157)[cite: 9, 147, 151]
- **Cap. 13**: Doble carril, filtro de sector y fidelidad a larga distancia (§158)[cite: 9, 155, 157]
  - *Notebooks*: `notebooks/qang_qpon_teleport[_es].ipynb`[cite: 1].

### Parte V. Relatividad, bucles temporales y agujeros negros (`qang.postselection`)
- **Cap. 14**: Curvas temporales cerradas posseleccionadas (P-CTC) vs D-CTC (§164)[cite: 9, 165, 168]
- **Cap. 15**: Termodinámica de agujeros negros, caos cuántico, ley $P\cdot F = 1/4$ y decodificador de Yoshida-Kitaev (§164)[cite: 10, 173, 175]
  - *Notebooks*: `notebooks/qang_methods[_es].ipynb`[cite: 1].

### Parte VI. Ciberseguridad cuántica, generación aleatoria y fronteras post-cuánticas
- **Cap. 16**: Demarcación: criptografía cuántica (QKD/QRNG) frente a PQC (ML-KEM/ML-DSA/SLH-DSA/HQC)[cite: 10, 185, 186]
- **Cap. 17**: Distribución cuántica de claves (QKD) y BB84 en espacio qg (§40, §43, §44, §49)[cite: 11, 195, 198, 200, 203]
- **Cap. 18**: Números aleatorios certificados (QRNG) y auditoría de canales (§31, §41, §42)[cite: 11, 208, 209, 214]
- **Cap. 19**: Baterías cuánticas y redes (§56, §57)[cite: 11, 219, 224]
  - *Notebooks*: `notebooks/qang_criptografia_colab.ipynb`[cite: 1].

### Parte VII. Aprendizaje automático cuántico (QML) y el filtro de simetría (`qang.qml`)
- **Cap. 20**: Redes neuronales cuánticas que conservan el peso (`WeightQNN`, `MultiClassQNN`, `qang.qsp`)[cite: 11, 232, 236, 238]
- **Cap. 21**: El filtro de simetría qg y la exactitud bajo relajación $T_1$ (F5 cocientes $T_1$, F6 desfase + ZNE)[cite: 12, 242, 246, 247]
- **Cap. 22**: Calibración por eco y el sesgo de ruido (*pull effect*, F7, lectura sin arrastre)[cite: 12, 252, 254, 256]
- **Cap. 23**: Geometría de la información, gradiente natural (QNG), núcleos (`qang.kernels`) y mesetas estériles[cite: 12, 13, 261, 266, 267]
  - *Notebooks*: `notebooks/qang_qml[_es].ipynb` (§75–§123)[cite: 1, 2].

### Parte VIII. Dinámica de muchos cuerpos, canales abiertos, materia y química
- **Cap. 24**: Simulación hamiltoniana (XY, XXZ), caminatas cuánticas, canales afines (`qang.channels`), `qang.varqte`, Hubbard y RBM[cite: 13, 277, 280, 281, 283]
- **Cap. 25**: Espectroscopía cuántica, PCA en sector peso 1, química molecular ($H_2$, LiH), solapamientos y QAOA restringido[cite: 13, 288, 290, 291, 292]

### Parte IX. Corrección de errores, síndromes continuos y caracterización
- **Cap. 26**: Corrección de errores (Leung, superficie consciente de $T_1$), qutrits/borrado, *circuit cutting* (`qang.knitting`) y ZNE[cite: 14, 297, 299, 301, 302]
- **Cap. 27**: Caracterización de hardware en qg: protocolo anunciado ($T_1, T_2, T_{\text{eff}}, e_{01}, e_{10}$), XEB/HOP, ZZ estático y cuantización[cite: 14, 306, 308, 309]

### Parte X. Evidencia experimental, hardware real y auditoría de falsación
- **Cap. 28**: Metodología: el protocolo de falsación pre-registrada (commits y hashes SHA-256)[cite: 14, 314, 315]
- **Cap. 29**: Las ventajas empíricas confirmadas[cite: 15, 320]
- **Cap. 30**: Límites estructurales, equivalencias y resultados negativos[cite: 15, 328]
- **Cap. 31**: Primer experimento en hardware real: IQM Resonance / Garnet (`qang.iqm`, §165, §168, §170)[cite: 15, 41, 337, 338]
- **Cap. 32**: Arquitectura de software, ecosistema multi-SDK (`qang.interop`) y ejecución física[cite: 16, 345]

### Partes XI–XIII. Nanocuántica, aplicaciones transversales y ecosistema
- **Cap. 33 (Parte XI)**: Nanocuántica: efecto túnel, fórmula de Landauer, ruido de partición, bloqueo de Coulomb, puntos cuánticos y centros NV[cite: 16, 353, 354, 355, 357, 359]
- **Cap. 34 (Parte XII)**: Arquitectura angular transversal: química (Jordan-Wigner, UCC, Larmor), materia condensada (Ising, Kitaev, Aharonov-Bohm), relatividad/óptica (Wigner, Fresnel, Unruh), Jarzynski y OTOC[cite: 16, 17, 364, 365, 367, 369, 371, 372]
- **Cap. 35 (Parte XIII)**: `qang` como módulo dentro de otras librerías (`qang.integrations`: mitiq, qiskit-experiments, QuTiP, Stim, sklearn, JAX, PyTorch, Qiskit Dynamics, Qibo, pytket, Braket, CUDA-Q)[cite: 1, 17, 379, 380, 382, 383]

### Apéndices (Generados por `tools/make_appendices.py` y `tools/make_apendice_h.py`)
- **Apéndice A**: Catálogo de las 15 compuertas fundamentales en representación qg[cite: 18].
- **Apéndice B**: Fórmulas de transición, matrices de Gell-Mann y polinomios de Chebyshev[cite: 18].
- **Apéndice C**: Protocolo de calibración por eco e inversión en sectores de Hamming (§105–§108)[cite: 18].
- **Apéndice D**: Referencia rápida de la API de `qang` (toma `qang.__version__` automáticamente)[cite: 1, 18].
- **Apéndice E**: Registro de falsación: predicciones pre-registradas verificables (commits y sha256)[cite: 2, 18].
- **Apéndice F**: Fundamento de análisis funcional: teorema de representación de Riesz-Fréchet[cite: 18].
- **Apéndice G**: Energía relativista y antimateria como lecturas qg (propuesta sin verificar + test §170)[cite: 19, 33, 41].
- **Apéndice H**: Catálogo de submedidas qang (52 entradas en 7 familias desde `qang.measures`, generado por `tools/make_apendice_h.py`)[cite: 1, 19].
