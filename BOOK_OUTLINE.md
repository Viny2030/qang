# BOOK_OUTLINE.md — Mapa Maestro del Tratado Bilingüe (44 Capítulos + Apéndices A–H)

**Título ES**: *Computación, información y física cuántica en el espacio qang* (`qang-libro-es.pdf` / `book/es/`)
**Título EN**: *Quantum Computing, Information and Physics in qang Space* (`qang-book-en.pdf` / `book/en/`)
**Web**: `https://viny2030.github.io/qang/` ([`/es/`](https://viny2030.github.io/qang/es/) · [`/en/`](https://viny2030.github.io/qang/en/))

## Estructura por Partes, Capítulos, Módulos y Scripts de Verificación

### Parte I. Fundamentos: la métrica angular-probabilística (la unidad qang)
- **Cap. 1**: La unidad qang: definición, mili-qang y la ecuación de probabilidad (`qang.probability`, `qang.qga`)
  - *Scripts*: `tools/make_cap01_figures.py`, `tools/verify_cap01.py`, `tools/verify_qga_lecturas.py`, `examples/unit_circle_qg.py`, `examples/iqm_milliqang_qg.py` (§168).
- **Cap. 2**: Compuertas nativas, álgebra cerrada y el operador universal (`qang.algebra`, `qang.pulse_area`, `qang.sectors`, `qang.interop`)
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap02.py`, `examples/pulse_area_qg.py` (§160), `examples/sector_exposure_qg.py` (§68).
- **Cap. 3**: Reflexiones, Householder y la dinámica de Grover (`qang.algebra`, `qang.probability`)
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap03.py`, `examples/grover_noise_qg.py` (§63), `examples/amplitude_estimation_chebyshev_qg.py` (§37).
- **Cap. 4**: Estadística de muestreo, inferencia bayesiana y la distribución de Haar (`qang.statistics`, `qang.proportions`)
  - *Scripts*: `tools/make_cap02_04_figures.py`, `tools/verify_cap04.py`, `examples/few_shot_tomography_qg.py` (§35), `examples/shadows_vs_direct_qg.py` (§62).

### Parte II. Geometría local y óptica cuántica
- **Cap. 5**: El radio local de Bloch como medida de pureza y entrelazamiento (`qang.formulation`, `qang.complementarity`)
  - *Scripts*: `tools/make_cap05_figures.py`, `tools/verify_cap05.py`, `examples/qg_radius_witness_qg.py` (§98), `examples/complementarity_qg.py` (§127).
- **Cap. 6**: Polarización óptica y parámetros de Stokes en unidades qg (`qang.polarization`)
  - *Scripts*: `tools/make_cap06_figures.py`, `tools/verify_cap06.py`, `notebooks/qang_optics[_es].ipynb` (§83).

### Parte III. QAN, información y geometría de la información (`qang.qan`, `qang.qdist`, `qang.qfisher`)
- **Cap. 7**: Redes angulares cuánticas (QAN), información mutua en espacio qang e inversión regularizada (IQAN)
  - *Scripts*: `tools/verify_cap07.py`, `tests/test_qan.py`, `book/cap07_qan_mi_iqan.md`.
- **Cap. 8**: Acoplamiento QAN ↔ distribuciones de probabilidad
  - *Scripts*: `tools/verify_cap08.py`, `tests/test_qdist.py`, `book/cap08_qan_distribuciones.md`.
- **Cap. 9**: Información de Fisher en qga
  - *Scripts*: `tools/verify_cap09.py`, `tests/test_qfisher.py`, `book/cap09_fisher_qng.md`.

### Parte IV. Fases geométricas y metrología
- **Cap. 10**: Fases geométricas y el teorema de Stokes en la esfera (`qang.geometric`, `qang.phase`)
  - *Scripts*: `tools/make_cap07_figures.py`, `tools/verify_cap07.py`, `notebooks/qang_optics[_es].ipynb` (§84).
- **Cap. 11**: Metrología en qga (`qang.fisher`, `qang.clock_comparison`)
  - *Scripts*: `tools/verify_cap08.py`, `examples/fisher_natural_gradient_qg.py` (§159), `examples/clock_redshift_qg.py` (§126), `examples/ramsey_qg_operating_point.py` (§36), `examples/ghz_metrology_qg.py` (§58), `examples/thermal_states_qg_tanh.py` (§25).

### Parte V. Neutrinos: oscilación, materia y Majorana
- **Cap. 12**: Oscilación de sabor en el vacío
- **Cap. 13**: Efectos de materia (MSW)
- **Cap. 14**: Fases de Majorana, 0νββ, neutrinos estériles en el sector |11⟩ y proyecciones angulares (`qang.angular`, §155, §163)
  - *Notebooks*: `notebooks/qang_neutrinos[_es].ipynb`, `notebooks/qang_quantum_numbers[_es].ipynb`.

### Parte VI. Teleportación, bucles temporales y agujeros negros
- **Cap. 15**: Teleportación óptica y el límite del par de Bell
- **Cap. 16**: Doble carril, pérdida como borrado anunciado y fidelidad a larga distancia
  - *Notebooks*: `notebooks/qang_qpon_teleport[_es].ipynb`.
- **Cap. 17**: Curvas temporales cerradas por postselección (P-CTC) frente a D-CTC
- **Cap. 18**: Agujeros negros como scramblers
  - *Notebooks*: `notebooks/qang_methods[_es].ipynb`.

### Parte VII. Criptografía y números aleatorios certificados
- **Cap. 19**: Demarcación
- **Cap. 20**: BB84 en espacio qg
- **Cap. 21**: Números aleatorios certificados y auditoría de canal

### Parte VIII. Baterías, aprendizaje cuántico y filtro de simetría
- **Cap. 22**: Baterías cuánticas y redes
  - *Notebooks*: `notebooks/qang_criptografia_colab.ipynb`.
- **Cap. 23**: Redes neuronales cuánticas que conservan el peso (`WeightQNN`, `MultiClassQNN`, `qang.qsp`)
- **Cap. 24**: El filtro de simetría con matrices densidad
- **Cap. 25**: Calibración por eco y el sesgo de ruido
- **Cap. 26**: Geometría de la información aplicada
  - *Notebooks*: `notebooks/qang_qml[_es].ipynb` (§75–§123).

### Parte IX. Dinámica de muchos cuerpos, química y corrección de errores
- **Cap. 27**: Dinámica de muchos cuerpos en el sector de peso 1
- **Cap. 28**: Química y optimización en el sector de peso
  - *Reescritura*: `book/cap28_quimica_qaoa.md`, `qang.qchem`, `tools/verify_cap28.py`, `tests/test_qchem.py` (modelo: dímero de Hubbard de 4 qubits; H$_2$/LiH siguen en `RESEARCH_NOTES.md`).
- **Cap. 29**: Corrección de errores
  - *Reescritura*: `book/cap29_correccion_errores.md`, `qang.qecc`, `tools/verify_cap29.py`, `tests/test_qecc.py` (cubre código AD, borrado vs bit, rango de Schmidt del corte y coste ZNE; superficie $T_1$-aware, qutrits y RB sin verificar).

### Parte X. Caracterización, método y libro de cuentas
- **Cap. 30**: Caracterización de hardware en m-qg
  - *Reescritura*: `book/cap30_caracterizacion.md`, `qang.qchar`, `tools/verify_cap30.py`, `tests/test_qchar.py` (simulación; hardware sin verificar).
- **Cap. 31**: Metodología
  - *Reescritura*: `book/cap31_falsacion.md`, `qang.qfalsify`, `tools/verify_cap31.py`, `tests/test_qfalsify.py` (maquinaria del registro; el registro histórico sigue en el Apéndice E).
- **Cap. 32**: Qué se confirmó y qué no
  - *Reescritura*: `book/cap32_libro_de_cuentas.md`, `qang.qledger`, `tools/verify_cap32.py`, `tests/test_qledger.py` (auditoría de 90 submedidas de Cap. 7–31; las ventajas medidas en dispositivos siguen en `RESEARCH_NOTES.md`).

### Parte XI. Alcance, límites, hardware real y arquitectura
- **Cap. 33**: Teoría de cuerdas
  - *Scripts*: `tools/verify_cap33.py`, `tests/test_qcft.py`, `book/cap33_cuerdas_cft.md`.
- **Cap. 34**: Límites estructurales, equivalencias y resultados negativos
  - *Reescritura*: `book/cap34_limites_resultados_negativos.md`, `qang.qlimits`, `tools/verify_cap34.py`, `tests/test_qlimits.py` (límites exactos; fallos de §157–§164 y LiH citados, `unverified` en el capítulo).
- **Cap. 35**: Primeras corridas en hardware real (`qang.iqm`, §165, §168, §170)
  - *Reescritura*: `book/cap35_hardware_garnet.md`, `qang.qgarnet`, `tools/verify_cap35.py`, `tests/test_qgarnet.py` (`tested on hardware`; recalculado de `examples/data/iqm_*`; pre-registros comprobados contra git; sin réplica).
- **Cap. 36**: Arquitectura de software y ecosistema multi-SDK
  - *Reescritura*: `book/cap36_arquitectura_software.md`, `qang.qarch`, `tools/verify_cap36.py`, `tests/test_qarch.py` (auditoría del repositorio; cifras de IBM §137–§150 citadas, `unverified`).

### Parte XII. Nanocuántica, aplicaciones transversales e integraciones
- **Cap. 37**: Nanocuántica
  - Reescritura (16-nov-2026, Cap. 37): `qang.qnano`, `tools/verify_cap37.py`; túnel, Landauer, ruido binomial estabilizado, Fermi, NV/ODMR (exact; NV con fotones `unverified` en hardware).
- **Cap. 38**: Aplicaciones transversales leídas con qang
  - Reescritura (16-nov-2026, Cap. 38): `qang.qtransv`, `tools/verify_cap38.py`; Jordan–Wigner, eco de Hahn, TFIM, Aharonov–Bohm, Wigner, Fresnel, Unruh (exact).
- **Cap. 39**: `qang.integrations` (`qang.integrations`: mitiq, qiskit-experiments, QuTiP, Stim, sklearn, JAX, PyTorch, Qiskit Dynamics, Qibo, pytket, Braket, CUDA-Q)
  - Reescritura (16-nov-2026, Cap. 39): `qang.qintegr`, `tools/verify_cap39.py`; auditoría de `qang.integrations` (12/12 guardas, extras y tests) y filtro T1 sin SDK (exact); cifras con SDK `unverified`.

### Parte XIII. Física extendida: materia y antimateria, viajes en el tiempo y cuerdas
- **Cap. 40**: Materia y antimateria leídas con qang
  - *Scripts*: `tools/verify_cap40.py`, `tests/test_qanti.py`, `book/cap40_materia_antimateria.md` (exact; sin hardware; complementa el Apéndice G).
- **Cap. 41**: Viajes en el tiempo leídos con qang
  - *Scripts*: `tools/verify_cap41.py`, `tests/test_qtimeloop.py`, `book/cap41_viajes_en_el_tiempo.md` (exact para los mapas; CTC físicas `unverified`; continúa el Cap. 17).
- **Cap. 42**: La teoría de cuerdas leída con qang
  - *Scripts*: `tools/verify_cap42.py`, `tests/test_qstring.py`, `book/cap42_teoria_de_cuerdas.md` (exact para el modelo; XXZ de tamaño finito `mixed`; física de cuerdas `not adopted`; continúa el Cap. 33).

### Parte XIV. Sistemas autónomos: robótica e IA agéntica cuánticas
- **Cap. 43**: Robótica cuántica y control autónomo leídos con qang (`qang.qrobot`)
  - *Scripts*: `tools/verify_cap43.py`, `tests/test_qrobot.py`, `book/cap43_robotica_cuantica.md` (exact; IK variacional y SLAM `unverified`; ventaja en robot real `not adopted`).
- **Cap. 44**: IA agéntica cuántica leída con qang (`qang.qagent`)
  - *Scripts*: `tools/verify_cap44.py`, `tests/test_qagent.py`, `book/cap44_ia_agentica_cuantica.md` (exact; orquestación por LLM y políticas multiqubit `unverified`; ventaja de aprendizaje `not adopted`).

### Apéndices (Generados por `tools/make_appendices.py` y `tools/make_apendice_h.py`)
- **Apéndice A**: Catálogo de las 15 compuertas fundamentales en representación qg.
- **Apéndice B**: Fórmulas de transición, matrices de Gell-Mann y polinomios de Chebyshev.
- **Apéndice C**: Protocolo de calibración por eco e inversión en sectores de Hamming (§105–§108).
- **Apéndice D**: Referencia rápida de la API de `qang` (toma `qang.__version__` automáticamente).
- **Apéndice E**: Registro de falsación: predicciones pre-registradas verificables (commits y sha256).
- **Apéndice F**: Fundamento de análisis funcional: teorema de representación de Riesz-Fréchet.
- **Apéndice G**: Energía relativista y antimateria como lecturas qg (propuesta sin verificar + test §170).
- **Apéndice H**: Catálogo de submedidas qang (52 entradas en 7 familias desde `qang.measures`, generado por `tools/make_apendice_h.py`).
