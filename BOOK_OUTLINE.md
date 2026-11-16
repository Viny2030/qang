# BOOK_OUTLINE.md — Mapa Maestro del Tratado Bilingüe (42 Capítulos + Apéndices A–H)

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
- **Cap. 7**: QAN, información mutua y IQAN: diagnóstico y poda de entrelazamiento en medidas qg (`qang.qan`) — *nuevo*
  - *Scripts*: `tools/verify_cap07.py`, `tests/test_qan.py`, `book/cap07_qan_mi_iqan.md`.
- **Cap. 8**: QAN sobre distribuciones de probabilidad: distancia TV, disparos y sesgo de Miller–Madow (`qang.qdist`) — *nuevo*
  - *Scripts*: `tools/verify_cap08.py`, `tests/test_qdist.py`, `book/cap08_qan_distribuciones.md`.
- **Cap. 9**: Información de Fisher y gradiente natural cuántico (QNG) regularizado en qg (`qang.qfisher`) — *nuevo*
  - *Scripts*: `tools/verify_cap09.py`, `tests/test_qfisher.py`, `book/cap09_fisher_qng.md`.
- **Cap. 10**: Fases geométricas y el teorema de Stokes en la esfera (`qang.geometric`, `qang.phase`)
  - *Scripts*: `tools/make_cap07_figures.py`, `tools/verify_cap07.py`, `notebooks/qang_optics[_es].ipynb` (§84).
- **Cap. 11**: Información de Fisher, relojes, gravedad y metrología (`qang.fisher`, `qang.clock_comparison`)
  - *Scripts*: `tools/verify_cap08.py`, `examples/fisher_natural_gradient_qg.py` (§159), `examples/clock_redshift_qg.py` (§126), `examples/ramsey_qg_operating_point.py` (§36), `examples/ghz_metrology_qg.py` (§58), `examples/thermal_states_qg_tanh.py` (§25).

### Parte III. Física de neutrinos, oscilaciones y simulación cuántica (`qang.neutrino`)
- **Cap. 12**: Cinemática cuántica de la oscilación de sabor en el vacío (§151, §165 IQM Garnet)
- **Cap. 13**: Efectos de materia (MSW), neutrinos solares y oscilaciones colectivas (§152, §154, §156, §163)
- **Cap. 14**: Fases de Majorana, doble beta sin neutrinos, neutrinos estériles y los cuatro números cuánticos (`qang.angular`, §155, §163)
  - *Notebooks*: `notebooks/qang_neutrinos[_es].ipynb`, `notebooks/qang_quantum_numbers[_es].ipynb`.

### Parte IV. Teleportación cuántica en redes ópticas (`qang.qpon`)
- **Cap. 15**: Teleportación óptica y el límite del par de Bell (§57, §157)
- **Cap. 16**: Doble carril, filtro de sector y fidelidad a larga distancia (§158)
  - *Notebooks*: `notebooks/qang_qpon_teleport[_es].ipynb`.

### Parte V. Relatividad, bucles temporales y agujeros negros (`qang.postselection`)
- **Cap. 17**: Curvas temporales cerradas posseleccionadas (P-CTC) vs D-CTC (§164)
- **Cap. 18**: Termodinámica de agujeros negros, caos cuántico, ley $P\cdot F = 1/4$ y decodificador de Yoshida-Kitaev (§164)
  - *Notebooks*: `notebooks/qang_methods[_es].ipynb`.

### Parte VI. Ciberseguridad cuántica, generación aleatoria y fronteras post-cuánticas
- **Cap. 19**: Demarcación: criptografía cuántica (QKD/QRNG) frente a PQC (ML-KEM/ML-DSA/SLH-DSA/HQC)
- **Cap. 20**: Distribución cuántica de claves (QKD) y BB84 en espacio qg (§40, §43, §44, §49)
- **Cap. 21**: Números aleatorios certificados (QRNG) y auditoría de canales (§31, §41, §42)
- **Cap. 22**: Baterías cuánticas y redes (§56, §57)
  - *Notebooks*: `notebooks/qang_criptografia_colab.ipynb`.

### Parte VII. Aprendizaje automático cuántico (QML) y el filtro de simetría (`qang.qml`)
- **Cap. 23**: Redes neuronales cuánticas que conservan el peso (`WeightQNN`, `MultiClassQNN`, `qang.qsp`)
- **Cap. 24**: El filtro de simetría qg y la exactitud bajo relajación $T_1$ (F5 cocientes $T_1$, F6 desfase + ZNE)
- **Cap. 25**: Calibración por eco y el sesgo de ruido (*pull effect*, F7, lectura sin arrastre)
- **Cap. 26**: Geometría de la información, gradiente natural (QNG), núcleos (`qang.kernels`) y mesetas estériles
  - *Notebooks*: `notebooks/qang_qml[_es].ipynb` (§75–§123).

### Parte VIII. Dinámica de muchos cuerpos, canales abiertos, materia y química
- **Cap. 27**: Simulación hamiltoniana (XY, XXZ), caminatas cuánticas, canales afines (`qang.channels`), `qang.varqte`, Hubbard y RBM
- **Cap. 28**: Espectroscopía cuántica, PCA en sector peso 1, química molecular ($H_2$, LiH), solapamientos y QAOA restringido
  - *Reescritura*: `book/cap28_quimica_qaoa.md`, `qang.qchem`, `tools/verify_cap28.py`, `tests/test_qchem.py` (modelo: dímero de Hubbard de 4 qubits; H$_2$/LiH siguen en `RESEARCH_NOTES.md`).

### Parte IX. Corrección de errores, síndromes continuos y caracterización
- **Cap. 29**: Corrección de errores (Leung, superficie consciente de $T_1$), qutrits/borrado, *circuit cutting* (`qang.knitting`) y ZNE
  - *Reescritura*: `book/cap29_correccion_errores.md`, `qang.qecc`, `tools/verify_cap29.py`, `tests/test_qecc.py` (cubre código AD, borrado vs bit, rango de Schmidt del corte y coste ZNE; superficie $T_1$-aware, qutrits y RB sin verificar).
- **Cap. 30**: Caracterización de hardware en qg: protocolo anunciado ($T_1, T_2, T_{\text{eff}}, e_{01}, e_{10}$), XEB/HOP, ZZ estático y cuantización
  - *Reescritura*: `book/cap30_caracterizacion.md`, `qang.qchar`, `tools/verify_cap30.py`, `tests/test_qchar.py` (simulación; hardware sin verificar).

### Parte X. Evidencia experimental, hardware real y auditoría de falsación
- **Cap. 31**: Metodología: el protocolo de falsación pre-registrada (commits y hashes SHA-256)
  - *Reescritura*: `book/cap31_falsacion.md`, `qang.qfalsify`, `tools/verify_cap31.py`, `tests/test_qfalsify.py` (maquinaria del registro; el registro histórico sigue en el Apéndice E).
- **Cap. 32**: Las ventajas empíricas confirmadas
  - *Reescritura*: `book/cap32_libro_de_cuentas.md`, `qang.qledger`, `tools/verify_cap32.py`, `tests/test_qledger.py` (auditoría de 90 submedidas de Cap. 7–31; las ventajas medidas en dispositivos siguen en `RESEARCH_NOTES.md`).
- **Cap. 33**: Teoría de cuerdas: alcance y límites de las medidas qang frente a una CFT de 1+1 dimensiones (cadena XX, $c=1$) (`qang.qcft`) — *adición*
  - *Scripts*: `tools/verify_cap33.py`, `tests/test_qcft.py`, `book/cap33_cuerdas_cft.md`.
- **Cap. 34**: Límites estructurales, equivalencias y resultados negativos
  - *Reescritura*: `book/cap34_limites_resultados_negativos.md`, `qang.qlimits`, `tools/verify_cap34.py`, `tests/test_qlimits.py` (límites exactos; fallos de §157–§164 y LiH citados, `unverified` en el capítulo).
- **Cap. 35**: Primer experimento en hardware real: IQM Resonance / Garnet (`qang.iqm`, §165, §168, §170)
  - *Reescritura*: `book/cap35_hardware_garnet.md`, `qang.qgarnet`, `tools/verify_cap35.py`, `tests/test_qgarnet.py` (`tested on hardware`; recalculado de `examples/data/iqm_*`; pre-registros comprobados contra git; sin réplica).
- **Cap. 36**: Arquitectura de software, ecosistema multi-SDK (`qang.interop`) y ejecución física
  - *Reescritura*: `book/cap36_arquitectura_software.md`, `qang.qarch`, `tools/verify_cap36.py`, `tests/test_qarch.py` (auditoría del repositorio; cifras de IBM §137–§150 citadas, `unverified`).

### Partes XI–XIII. Nanocuántica, aplicaciones transversales y ecosistema
- **Cap. 37 (Parte XI)**: Nanocuántica: efecto túnel, fórmula de Landauer, ruido de partición, bloqueo de Coulomb, puntos cuánticos y centros NV
  - Reescritura (16-nov-2026, Cap. 37): `qang.qnano`, `tools/verify_cap37.py`; túnel, Landauer, ruido binomial estabilizado, Fermi, NV/ODMR (exact; NV con fotones `unverified` en hardware).
- **Cap. 38 (Parte XII)**: Arquitectura angular transversal: química (Jordan-Wigner, UCC, Larmor), materia condensada (Ising, Kitaev, Aharonov-Bohm), relatividad/óptica (Wigner, Fresnel, Unruh), Jarzynski y OTOC
  - Reescritura (16-nov-2026, Cap. 38): `qang.qtransv`, `tools/verify_cap38.py`; Jordan–Wigner, eco de Hahn, TFIM, Aharonov–Bohm, Wigner, Fresnel, Unruh (exact).
- **Cap. 39 (Parte XIII)**: `qang` como módulo dentro de otras librerías (`qang.integrations`: mitiq, qiskit-experiments, QuTiP, Stim, sklearn, JAX, PyTorch, Qiskit Dynamics, Qibo, pytket, Braket, CUDA-Q)
  - Reescritura (16-nov-2026, Cap. 39): `qang.qintegr`, `tools/verify_cap39.py`; auditoría de `qang.integrations` (12/12 guardas, extras y tests) y filtro T1 sin SDK (exact); cifras con SDK `unverified`.
- **Cap. 40 (Parte XIII, ampliación)**: Materia y antimateria: conjugación de carga como mapa antípoda, Zitterbewegung, creación de pares (Landau–Zener), par en singlete, presupuesto de disparos (`qang.qanti`) — *nuevo*
  - *Scripts*: `tools/verify_cap40.py`, `tests/test_qanti.py`, `book/cap40_materia_antimateria.md` (exact; sin hardware; complementa el Apéndice G).
- **Cap. 41 (Parte XIII, ampliación)**: Viajes en el tiempo: balance de información de la postselección (Fisher total = 1), paradoja del abuelo en P-CTC y D-CTC, puntos fijos por ángulo en qga (`qang.qtimeloop`) — *nuevo*
  - *Scripts*: `tools/verify_cap41.py`, `tests/test_qtimeloop.py`, `book/cap41_viajes_en_el_tiempo.md` (exact para los mapas; CTC físicas `unverified`; continúa el Cap. 17).
- **Cap. 42 (Parte XIII, ampliación)**: La teoría de cuerdas leída con qang: conteo exacto de la cuerda bosónica (D=26, 576=299+276+1), el nivel como peso ponderado y lectura Z lineal con su filtro bajo T1, T-dualidad (576 → 676 sin masa en el radio autodual) y el radio de una cadena XXZ leído con una base Z (`qang.qstring`) — *nuevo*
  - *Scripts*: `tools/verify_cap42.py`, `tests/test_qstring.py`, `book/cap42_teoria_de_cuerdas.md` (exact para el modelo; XXZ de tamaño finito `mixed`; física de cuerdas `not adopted`; continúa el Cap. 33).

### Apéndices (Generados por `tools/make_appendices.py` y `tools/make_apendice_h.py`)
- **Apéndice A**: Catálogo de las 15 compuertas fundamentales en representación qg.
- **Apéndice B**: Fórmulas de transición, matrices de Gell-Mann y polinomios de Chebyshev.
- **Apéndice C**: Protocolo de calibración por eco e inversión en sectores de Hamming (§105–§108).
- **Apéndice D**: Referencia rápida de la API de `qang` (toma `qang.__version__` automáticamente).
- **Apéndice E**: Registro de falsación: predicciones pre-registradas verificables (commits y sha256).
- **Apéndice F**: Fundamento de análisis funcional: teorema de representación de Riesz-Fréchet.
- **Apéndice G**: Energía relativista y antimateria como lecturas qg (propuesta sin verificar + test §170).
- **Apéndice H**: Catálogo de submedidas qang (52 entradas en 7 familias desde `qang.measures`, generado por `tools/make_apendice_h.py`).
