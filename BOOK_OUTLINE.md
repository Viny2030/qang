# BOOK_OUTLINE.md — Mapa Maestro del Tratado Bilingüe (35 Capítulos + Apéndices A–H)

**Título ES**: *Computación, información y física cuántica en el espacio qang* (`qang-libro-es.pdf` / `book/es/`)
**Título EN**: *Quantum Computing, Information and Physics in qang Space* (`qang-book-en.pdf` / `book/en/`)
**Web**: `https://viny2030.github.io/qang/` ([`/es/`](https://viny2030.github.io/qang/es/) · [`/en/`](https://viny2030.github.io/qang/en/))

## Estructura por Partes, Capítulos, Módulos y Scripts de Verificación

### Parte I. Fundamentos: la métrica angular-probabilística
- **Cap. 1**: La unidad qang: definición, mili-qang y la ecuación de probabilidad (`qang.probability`, `qang.qga`)
  - *Scripts*: `tools/make_cap01_figures.py`, `tools/verify_cap01.py`, `tools/verify_qga_lecturas.py`, `examples/unit_circle_qg.py`, `examples/iqm_milliqang_qg.py` (§168).
- **Cap. 2**: Compuertas nativas, álgebra cerrada y el operador universal (`qang.algebra`, `qang.pulse_area`, `qang.sectors`, `qang.interop`)
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap02.py`, `examples/pulse_area_qg.py` (§160), `examples/sector_exposure_qg.py` (§68).
- **Cap. 3**: Reflexiones, Householder y la dinámica de Grover (`qang.algebra`, `qang.probability`)
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap03.py`, `examples/grover_noise_qg.py` (§63), `examples/amplitude_estimation_chebyshev_qg.py` (§37).
- **Cap. 4**: Estadística de muestreo, inferencia bayesiana y la distribución de Haar (`qang.statistics`, `qang.proportions`)
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap04.py`, `examples/few_shot_tomography_qg.py` (§35), `examples/shadows_vs_direct_qg.py` (§62).

### Parte II. Geometría local, óptica cuántica y fases geométricas
- **Cap. 5**: El radio local de Bloch como medida de pureza y entrelazamiento (`qang.formulation`, `qang.complementarity`)
  - *Scripts*: `tools/make_cap05_figures.py`, `tools/verify_cap05.py`, `examples/qg_radius_witness_qg.py` (§98), `examples/complementarity_qg.py` (§127).
- **Cap. 6**: Polarización óptica y parámetros de Stokes en unidades qg (`qang.polarization`)
  - *Scripts*: `tools/make_cap06_figures.py`, `tools/verify_cap06.py`, `notebooks/qang_optics[_es].ipynb` (§83).
- **Cap. 7**: Fases geométricas y el teorema de Stokes en la esfera (`qang.geometric`, `qang.phase`)
  - *Scripts*: `tools/make_cap07_figures.py`, `tools/verify_cap07.py`, `notebooks/qang_optics[_es].ipynb` (§84).
- **Cap. 8**: Información de Fisher, relojes, gravedad y metrología (`qang.fisher`, `qang.clock_comparison`)
  - *Scripts*: `tools/verify_cap08.py`, `examples/fisher_natural_gradient_qg.py` (§159), `examples/clock_redshift_qg.py` (§126), `examples/ramsey_qg_operating_point.py` (§36), `examples/ghz_metrology_qg.py` (§58), `examples/thermal_states_qg_tanh.py` (§25).

### Parte III. Física de neutrinos, oscilaciones y simulación cuántica (`qang.neutrino`)
- **Cap. 9**: Cinemática cuántica de la oscilación de sabor en el vacío (§151, §165 IQM Garnet)
- **Cap. 10**: Efectos de materia (MSW), neutrinos solares y oscilaciones colectivas (§152, §154, §156, §163)
- **Cap. 11**: Fases de Majorana, doble beta sin neutrinos, neutrinos estériles y los cuatro números cuánticos (`qang.angular`, §155, §163)
  - *Notebooks*: `notebooks/qang_neutrinos[_es].ipynb`, `notebooks/qang_quantum_numbers[_es].ipynb`.

### Parte IV. Teleportación cuántica en redes ópticas (`qang.qpon`)
- **Cap. 12**: Teleportación óptica y el límite del par de Bell (§57, §157)
- **Cap. 13**: Doble carril, filtro de sector y fidelidad a larga distancia (§158)
  - *Notebooks*: `notebooks/qang_qpon_teleport[_es].ipynb`.

### Parte V. Relatividad, bucles temporales y agujeros negros (`qang.postselection`)
- **Cap. 14**: Curvas temporales cerradas posseleccionadas (P-CTC) vs D-CTC (§164)
- **Cap. 15**: Termodinámica de agujeros negros, caos cuántico, ley $P\cdot F = 1/4$ y decodificador de Yoshida-Kitaev (§164)
  - *Notebooks*: `notebooks/qang_methods[_es].ipynb`.

### Parte VI. Ciberseguridad cuántica, generación aleatoria y fronteras post-cuánticas
- **Cap. 16**: Demarcación: criptografía cuántica (QKD/QRNG) frente a PQC (ML-KEM/ML-DSA/SLH-DSA/HQC)
- **Cap. 17**: Distribución cuántica de claves (QKD) y BB84 en espacio qg (§40, §43, §44, §49)
- **Cap. 18**: Números aleatorios certificados (QRNG) y auditoría de canales (§31, §41, §42)
- **Cap. 19**: Baterías cuánticas y redes (§56, §57)
  - *Notebooks*: `notebooks/qang_criptografia_colab.ipynb`.

### Parte VII. Aprendizaje automático cuántico (QML) y el filtro de simetría (`qang.qml`)
- **Cap. 20**: Redes neuronales cuánticas que conservan el peso (`WeightQNN`, `MultiClassQNN`, `qang.qsp`)
- **Cap. 21**: El filtro de simetría qg y la exactitud bajo relajación $T_1$ (F5 cocientes $T_1$, F6 desfase + ZNE)
- **Cap. 22**: Calibración por eco y el sesgo de ruido (*pull effect*, F7, lectura sin arrastre)
- **Cap. 23**: Geometría de la información, gradiente natural (QNG), núcleos (`qang.kernels`) y mesetas estériles
  - *Notebooks*: `notebooks/qang_qml[_es].ipynb` (§75–§123).

### Parte VIII. Dinámica de muchos cuerpos, canales abiertos, materia y química
- **Cap. 24**: Simulación hamiltoniana (XY, XXZ), caminatas cuánticas, canales afines (`qang.channels`), `qang.varqte`, Hubbard y RBM
- **Cap. 25**: Espectroscopía cuántica, PCA en sector peso 1, química molecular ($H_2$, LiH), solapamientos y QAOA restringido

### Parte IX. Corrección de errores, síndromes continuos y caracterización
- **Cap. 26**: Corrección de errores (Leung, superficie consciente de $T_1$), qutrits/borrado, *circuit cutting* (`qang.knitting`) y ZNE
- **Cap. 27**: Caracterización de hardware en qg: protocolo anunciado ($T_1, T_2, T_{\text{eff}}, e_{01}, e_{10}$), XEB/HOP, ZZ estático y cuantización

### Parte X. Evidencia experimental, hardware real y auditoría de falsación
- **Cap. 28**: Metodología: el protocolo de falsación pre-registrada (commits y hashes SHA-256)
- **Cap. 29**: Las ventajas empíricas confirmadas
- **Cap. 30**: Límites estructurales, equivalencias y resultados negativos
- **Cap. 31**: Primer experimento en hardware real: IQM Resonance / Garnet (`qang.iqm`, §165, §168, §170)
- **Cap. 32**: Arquitectura de software, ecosistema multi-SDK (`qang.interop`) y ejecución física

### Partes XI–XIII. Nanocuántica, aplicaciones transversales y ecosistema
- **Cap. 33 (Parte XI)**: Nanocuántica: efecto túnel, fórmula de Landauer, ruido de partición, bloqueo de Coulomb, puntos cuánticos y centros NV
- **Cap. 34 (Parte XII)**: Arquitectura angular transversal: química (Jordan-Wigner, UCC, Larmor), materia condensada (Ising, Kitaev, Aharonov-Bohm), relatividad/óptica (Wigner, Fresnel, Unruh), Jarzynski y OTOC
- **Cap. 35 (Parte XIII)**: `qang` como módulo dentro de otras librerías (`qang.integrations`: mitiq, qiskit-experiments, QuTiP, Stim, sklearn, JAX, PyTorch, Qiskit Dynamics, Qibo, pytket, Braket, CUDA-Q)

### Apéndices (Generados por `tools/make_appendices.py` y `tools/make_apendice_h.py`)
- **Apéndice A**: Catálogo de las 15 compuertas fundamentales en representación qg.
- **Apéndice B**: Fórmulas de transición, matrices de Gell-Mann y polinomios de Chebyshev.
- **Apéndice C**: Protocolo de calibración por eco e inversión en sectores de Hamming (§105–§108).
- **Apéndice D**: Referencia rápida de la API de `qang` (toma `qang.__version__` automáticamente).
- **Apéndice E**: Registro de falsación: predicciones pre-registradas verificables (commits y sha256).
- **Apéndice F**: Fundamento de análisis funcional: teorema de representación de Riesz-Fréchet.
- **Apéndice G**: Energía relativista y antimateria como lecturas qg (propuesta sin verificar + test §170).
- **Apéndice H**: Catálogo de submedidas qang (52 entradas en 7 familias desde `qang.measures`, generado por `tools/make_apendice_h.py`).
