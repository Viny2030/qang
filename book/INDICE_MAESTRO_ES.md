# Computación, información y física cuántica en el espacio qang (qg)

**Tratado integral — índice maestro consolidado (versión en español).**
Une los dos índices de octubre de 2026 (*Tratado… Índice Maestro Consolidado* e *Índice Maestro… Experimental*)
sin recortar ninguna propuesta, y agrega lo que el repositorio ya tiene y ninguno de los dos índices recogía.
Base: qang 0.12.0, `RESEARCH_NOTES.md` §1–§164 y Apéndice A.

## Cómo leer las marcas

| marca | significado |
|---|---|
| **§N** | respaldado: sección de `RESEARCH_NOTES.md` con código, test y, cuando corresponde, predicciones pre-registradas |
| ⚠️ | está en el repo, pero el índice original lo decía distinto: el texto ya va corregido |
| 🔬 | propuesta del índice original sin respaldo en el repo todavía: hay que demostrarla (o sacarla) antes de escribir el capítulo |
| 🆕 | propuesta nueva para desarrollar dentro del libro |
| 🧪 | candidato a medición en hardware real (IQM Resonance) |

Regla general del libro: todo lo que diga "en dispositivo" es sobre **modelos de ruido simulados** de IBM o IonQ
hasta que exista una corrida real (Parte X).

---

## PARTE I. Fundamentos: la métrica angular-probabilística (la unidad qang)

### Capítulo 1. La unidad qang: definición, mili-qang y la ecuación de probabilidad ✅ escrito
1.1 Definición del qang: regla de Born p₀ = cos²(θ/2), sesgo polar qg_Z = ⟨σ_z⟩ = 2p₀ − 1 = cos θ, ángulo qang y qg_S; el qang como unidad de ingeniería (analogía del decibelio) — §7; figura: el círculo qang con escala en m-qg
1.2 Teoría del mili-qang: definición y escala (−1000 a +1000 m-qg); lineal en probabilidad (1 m-qg = Δp₀ = 0,0005), no lineal en ángulo (1 mrad en el ecuador, 2,56° junto al polo); resolución por disparos σ_m = 1000 sin θ/√N y N = (1000 sin θ/δ)²; presupuesto de error de compuerta en m-qg — §160, §167; figuras: regla del mili-qang, resolución por disparos, presupuesto de pulso
1.3 La ecuación de probabilidad del qang y del mili-qang: θ, qg_Z, qg_S y m como funciones exactas de p₀; tabla de anclajes; generalización a varios resultados por la distancia estadística de Wootters (ángulo de Bhattacharyya) — §167; figura: las lecturas de una misma probabilidad
1.4 Por qué medir en qg: la brecha geometría–probabilidad (singularidad del gradiente y varianza dependiente del estado) — §1, §5
1.5 Inversión θ = arccos(qg_Z) y el signo de rama — §65
1.6 La entropía del resultado qg_S, su invertibilidad por mitades y sus barras de error — §2, §7, §51
1.7 El ángulo qang como transformación arcoseno de una proporción (Bartlett, Anscombe, Freeman–Tukey) — §125
1.8 Estados mixtos y registros de varios qubits — §3, §4, §45
1.9 En código: `qang.probability` como herramienta de análisis (`analyze`) — §167
1.10 Alcance y límites de la base: no es física nueva sino una escala de trabajo; primera prueba en hardware (§168, IQM Garnet): 1,64 m-qg por compuerta X, errores de ángulo de 16–55 m-qg, M2 falló; figura de hardware — §168

### Capítulo 2. Compuertas nativas, álgebra cerrada y el operador universal
2.1 La compuerta elemental RQangGate(qg_Z) y su descomposición nativa en Ry(θ) — §6
2.2 🔬 El operador universal U3_qang(qg_Z, φ, λ): matriz algebraica cerrada y desacople de fases
2.3 🔬 Verificación de U3_qang contra librerías estándar a precisión de máquina (I, H, X, Z)
2.4 🔬 Generalización a qudits (d ≥ 3) por la subálgebra de Cartan de su(d) y matrices de Gell-Mann (punto de partida: la prueba del qutrit, §70–§72)
2.5 El principio de área de pulso mínima (`qang.pulse_area`) y el amortiguamiento cos θ·e^(−θ²σ²/2) — §160
2.6 La compuerta qg con signo y la pantalla sin ruido de cuánto T1 deja pasar el filtro — §68
2.7 Un único orden de qubits para todos los SDK (Qiskit, Cirq, PennyLane, Braket, pytket) — §130
2.8 Sobre la base del capítulo 1: preparar desde p₀ o m-qg; la compuerta como ecuación de probabilidad (un número en m-qg fija su acción, capacidad 1 − qg_S); el área de pulso en m-qg — §160, §167; figura: error de R_x en m-qg, ángulo pedido frente a representante de área mínima

### Capítulo 3. Reflexiones, Householder y la dinámica de Grover
3.1 🔬 Matrices uniestocásticas y estocásticas de rotación cerrada sin funciones trascendentes
3.2 🔬 El reflector de Householder en espacio qang: H(qg_Z) = −qg_Z·σ_z − √(1−qg_Z²)·σ_x
3.3 🔬 El difusor de Grover como reflexión qg pura: D(qg_Z) = −H(qg_Z)
3.4 🔬 Probabilidad de transición entre dos estados qg: P(q1→q2) = [1 + q1q2 + √((1−q1²)(1−q2²))]/2
3.5 La trayectoria de Grover como polinomio de Chebyshev: qg_Z(θ_k) = cos((2k+1)θ_0) — §37
3.6 Límites analíticos, ángulo grueso y sobreoscilación cuando M/N no es despreciable; Grover con ruido: la lectura qg pierde contra el histograma — §37, §63
3.7 Estimación de amplitud de Grover en el simulador ruidoso de IonQ — §39
3.8 Sobre la base del capítulo 1: la probabilidad de transición tiene ángulo qang γ (el ángulo entre los estados) y lectura 1000·r₁·r₂ m-qg; la trayectoria de Grover y la resolución por profundidad en m-qg — §37, §166, §167; figura: trayectoria de Grover en m-qg (a = 0,1 y 0,3)

### Capítulo 4. Estadística de muestreo, inferencia bayesiana y la distribución de Haar
4.1 Propagación del ruido de disparo por el método delta: Var(θ̂) ≈ 1/N — §5, §125
4.2 La distribución de Haar en espacio qg: teorema de la caja de sombrero de Arquímedes, qg_Z ~ U(−1, 1) — §15
4.3 Estimador bayesiano Beta-Binomial: intervalos exactos cerca de los polos frente al colapso del método delta — §5, §125
4.4 Estimadores insesgados de segundo orden para qg² y el radio local r² (`qg2_unbiased`) — §97, §126
4.5 ⚠️ Presupuesto adaptativo de disparos — §143 (planificar disparos para una precisión qg dada). 🔬 La cifra "33,9 % de disparos ahorrados" no está en el repo: medirla o sacarla
4.6 Tomografía de un qubit con pocos disparos; sombras clásicas contra medición directa — §35, §62
4.7 Fundamento de análisis funcional: Riesz–Fréchet, por qué qg_Z es la lectura lineal canónica y qg_S no — Apéndice A
4.8 Sobre la base del capítulo 1: Haar es uniforme en m-qg y en p₀ (½ sin θ en el ángulo); intervalos en m-qg (el método delta colapsa o se sale de la escala); planificar disparos en m-qg; el mili-qang como funcional lineal (Riesz–Fréchet) — §15, §143, §167; figuras: Haar en m-qg y en ángulo, ancho de intervalos con 50 disparos

---

## PARTE II. Geometría local, óptica cuántica y fases geométricas

### Capítulo 5. El radio local de Bloch como medida de pureza y entrelazamiento ✅ escrito
5.0 La ecuación de probabilidad con radio: p₀ = (1 + r cos θ)/2 y r² = (m_X² + m_Y² + m_Z²)/10⁶; en peso definido, 1 − r² = 4p₀(1 − p₀) — §167; figuras: déficit en m-qg, dirección separada del radio, dualidad P² + V² = r²
5.1 r² = qg_X² + qg_Y² + qg_Z² = 2·Tr(ρ²) − 1 y el déficit superficial D = 1 − r² — §97
5.2 Correspondencia con el one-tangle de Coffman–Kundu–Wootters y la medida de Meyer–Wallach — §98
5.3 El radio ante compuertas elementales (rotaciones, CNOT, CZ, iSWAP, Fredkin) — §97
5.4 Desacople de dirección polar y radio: θ = atan2(√(qg_X²+qg_Y²), qg_Z) = arccos(qg_Z/r) bajo ruido despolarizante — §102, §103
5.5 El colapso r = |qg_Z| en estados de peso de Hamming fijo y la lectura de entrelazamiento en base Z — §98, §100, §104
5.6 Dualidad onda–partícula como radio de Bloch e incertidumbre entrópica con qg_S — §127

### Capítulo 6. Polarización óptica y parámetros de Stokes en unidades qg
6.0 La ley de Malus como ecuación de probabilidad: la transmisión de un polarizador es p_0=cos ²Δ, de modo que el ángulo qang es 2Δ y cada parámetro de Stokes normalizado se lee en m-qg (S_1/S_0=qg_Z) — §167
6.0b ✅ escrito: Malus como p₀ = cos²Δ (θ = 2Δ, m = 1000·r·a), Stokes como valores qg, grado de polarización = radio, Mueller = regla de compuertas, polarimetría por conteo con el sesgo de P̂ — §83, §167; figuras: duplicación de ángulos, Malus en m-qg, sesgo del grado de polarización
6.1 Equivalencia Poincaré–Bloch: qg_Z = S1/S0, qg_X = S2/S0, qg_Y = S3/S0 — §83
6.2 Grado de polarización como norma del vector qg y pureza de estados mixtos — §83
6.3 Matrices de Mueller sin pérdidas y la regla qg_P′ = qg{U†PU} — §83
6.4 Ley de Malus unificada: I(θ) = (I0/2)(1 + qg_Z cos 2θ + qg_X sin 2θ) — §83
6.5 Polarimetría de fotón único a partir de conteos e intervalos de credibilidad — §83

### Capítulo 7. Fases geométricas y el teorema de Stokes en la esfera
7.0 La fase geométrica en la ecuación de probabilidad: un bucle a qg_Z constante acumula (qg_Z-1)/2=-p_1 vueltas, es decir, (m-1000)/2000 vueltas en mili-qang — §167
7.0b ✅ escrito: la fase de un bucle a altura constante es p₀ vueltas (mód. 1); el interferómetro convierte la fase en cos²(γ/2) y su ángulo qang es la fase, con 1/√N para cualquier valor; octante de Pancharatnam — §84, §167; figuras: fase contra m-qg con tres cálculos, medición interferométrica
7.1 La conexión de Berry A = i⟨ψ|∇ψ⟩ y su curvatura como campo monopolar — §84
7.2 Stokes en la esfera: γ(C) = −Ω(C)/2 como flujo de curvatura — §84
7.3 Bucles a qg_Z constante: fase φ = (qg_Z − 1)/2 vueltas (mod 1) — §84
7.4 Tres cálculos independientes (producto de solapamientos invariante de gauge, abanico esférico, flujo numérico) — §84
7.5 La fase de Pancharatnam y el octante canónico H → D → R → H — §84

### Capítulo 8. Información de Fisher, relojes, gravedad y metrología 🆕 (no figuraba en los índices)
8.0 Desde la base: el ángulo qang es la coordenada plana de Fisher (información 1 por disparo) y qg_Z la curva (1/(1-qg_Z²)); la resolución en m-qg del capítulo 1 es su cota de Cramér–Rao — §167
8.0b Geometría de la información en qg: información de Fisher, métrica de Fubini–Study (g_ii=1/4 en el ansatz real R_Y+CZ) y el bloque de correladores qg_Y; ⚠️ el gradiente natural en coordenadas qg equivale al descenso euclidiano en θ: la ganancia de 1,3× viene de la métrica de bloque (4,6× con la completa), no de la coordenada — §159
8.1 Dos relojes y la gravedad: la fase diferencial leída de la elipse (corrimiento al rojo gravitatorio a 1 m) — §126
8.2 Sensado Ramsey: dónde operar — §36
8.3 Metrología GHZ: cuándo el entrelazamiento le gana a N qubits independientes — §58
8.4 Estados térmicos: qg_Z = tanh(βh) — §25

---

## PARTE III. Física de neutrinos, oscilaciones y simulación cuántica (`qang.neutrino`)

### Capítulo 9. Cinemática cuántica de la oscilación de sabor en el vacío
9.0 La probabilidad de oscilación P(ν_α→ν_β) como probabilidad qang: su ángulo 2 arcsin √P, su lectura m = 1000(1 − 2P) m-qg y su barra de error en la escala del capítulo 1 — §167
9.1 La oscilación ν_e ↔ ν_μ ↔ ν_τ como rotación geométrica unitaria en la esfera qg — §151
9.2 Probabilidad en el vacío: qg_Z = cos²2θ + sin²2θ·cos Δ (ley de cosenos esférica) y P(ν_α → ν_β) — §151
9.3 El ángulo de mezcla en el máximo de oscilación es un cuarto del ángulo qang: θ_mezcla = θ_qg/4; para θ > π/4 devuelve π/2 − θ (degeneración de octante) — §151
9.4 Barras de error que no dependen de θ: σ = 1/(4√(N+½)) — §151
9.5 Tres sabores en el sector de peso 1 (|100⟩, |010⟩, |001⟩) sobre 3 qubits; PMNS como tres RBS y dos fases — §151
9.6 ⚠️ Supresión del ruido con el filtro qg: el error baja ~3× (3,0–3,8×) **en modelos de ruido simulados** — §151; en hardware real (IQM Garnet) 1,6× y el máximo de oscilación vuelve a 2,6 GeV — §165; figuras: cono de precesión, espectro de tres sabores, corrida en Garnet
9.7 La asimetría CP en el vacío y por qué sobrevive incluso sin filtro — §151

### Capítulo 10. Efectos de materia, neutrinos solares y oscilaciones colectivas
10.0 Las probabilidades de sabor en materia, en el Sol y en oscilaciones colectivas leídas en m-qg: los efectos de materia como corrimientos medidos en la misma escala que el ruido de disparo — §167
10.1 Efecto MSW en la Tierra como circuito de Trotter que conserva el peso; la materia imita violación CP — §152
10.2 Neutrinos solares: conversión MSW adiabática y por qué el Sol entero es una sola rotación (`solar_circuit`) — §156
10.3 Oscilaciones colectivas: el número conservado es el número de masa, no el de sabor (`reweight_sectors`) — §154
10.4 Ordenamiento de masas (normal frente a invertido) leído a través de la materia (NUFIT_IO) — §163
10.5 🔬 Densidades variables y comparación con el Modelo Solar Estándar (perfil de densidad realista; hoy §156 usa un perfil simplificado)

### Capítulo 11. Fases de Majorana, doble beta sin neutrinos y neutrinos estériles
11.0 Las fases de Majorana y el neutrino estéril vistos desde la ecuación de probabilidad: lo que no cambia ninguna probabilidad no cambia ninguna lectura qang — §167
11.1 ⚠️ Invisibilidad de las fases de Majorana en oscilaciones: son fases diagonales que el transpilador elimina (`majorana_invisible`); no es una "eliminación analítica" deliberada — §163
11.2 Las fases de Majorana en la masa efectiva m_ββ de la doble beta sin neutrinos — §163
11.3 El neutrino estéril ν_s como cuarto qubit en el sector de peso 1 (`sterile_circuit`) — §163
11.4 ⚠️ Ganancia de 5,5× del filtro con 4 sabores (3+1) **en modelos de ruido simulados** — §163 🧪
11.5 🔬 Límites fenomenológicos y comparación de modelos 3+1 con datos de experimentos de línea corta
11.6 Los cuatro números cuánticos en términos qg — §155

---

## PARTE IV. Teleportación cuántica en redes ópticas y distribución de entrelazamiento (`qang.qpon`)

### Capítulo 12. Teleportación óptica y el límite del par de Bell
12.1 Fundamentos de Bennett et al. (1993) y el papel de los estados de Bell — §157
12.2 Canales con pérdidas asimétricas en fibra: por qué el par de Bell deja de ser el recurso óptimo — §157
12.3 ⚠️ La regla de diseño polar de una red óptica pasiva (QPON): **tan α* = √η / κ** (el índice original decía √(η/κ)) — §157
12.4 Fidelidad óptima F* = (2 + η/κ)/3; con asistencia F* = 1 en tan α* = √N > 1: un recurso "más que maximalmente entrelazado" — §157
12.5 El estado recurso óptimo leído en su polarización polar: qg_Z* = cos 2α* = (κ² − η)/(κ² + η) — §157
12.6 Pares de Bell en una red cuántica, leídos en qg — §57

### Capítulo 13. Doble carril, filtro de sector y fidelidad a larga distancia
13.1 La fidelidad de teleportación como lectura lineal de tres correladores qg: F = (3 + qg_XX − qg_YY + qg_ZZ)/6 — §158
13.2 El anuncio de pérdidas (heralding) por los usuarios de la red como proyector de filtro de sector — §158
13.3 Codificación en doble carril: detección y descarte de toda pérdida fotónica — §158
13.4 ⚠️ A 20 km el doble carril con filtro supera al óptimo sin ruido de carril simple en Torino y Fez (2 de 3 modelos simulados; hallazgo post hoc, no pre-registrado) — §158
13.5 🔬 Enrutamiento y conmutación cuántica pasiva en topologías metropolitanas multiportadora
13.6 🆕 Repetidores cuánticos e intercambio de entrelazamiento (entanglement swapping) en cadena, leídos con qg_ZZ y el filtro de doble carril: en el libro, álgebra exacta (los correladores de pares diagonales en Bell se multiplican; F = (1 + w^n)/2); la cadena con doble carril en dispositivos sigue pendiente
13.7 🆕 Teleportación de compuertas (gate teleportation) y su lectura en correladores qg

---

## PARTE V. Relatividad, bucles temporales y evaporación de agujeros negros (`qang.postselection`)

> Son **modelos de teoría de la información**, no afirmaciones sobre el espacio-tiempo (§164). Los dos son
> teleportación con posselección sobre un resultado de Bell, es decir, un filtro sobre el sector de Bell.

### Capítulo 14. Curvas temporales cerradas posseleccionadas (P-CTC) y resolución de paradojas
14.1 Modelado cuántico relativista: el formalismo D-CTC de Deutsch frente al P-CTC de Bennett–Svetlichny / Lloyd et al. (2011) — §164 (P-CTC); 🆕 D-CTC a desarrollar
14.2 La teleportación hacia el pasado como proyección de Bell con posselección estricta — §164
14.3 La probabilidad de éxito como ley trigonométrica pura: P_éxito = cos²(φ/2), y su ángulo qang arccos(2P − 1) = φ — §164
14.4 La paradoja del abuelo: P(paradoja) = 0 en φ = π; suprimida hasta el piso de ruido en Brisbane y Fez — §164
14.5 La rotación controlada por un sistema: Tr_C(U) = diag(1, cos(φ/2)) y qg_Z(S) = (1 − c²)/(1 + c²) — §164
14.6 Efectos no lineales de las P-CTC y límites en la discriminación de estados no ortogonales — 🔬 (pendiente: circuito y predicción)
14.7 🆕 D-CTC de Deutsch como punto fijo: comparación numérica con P-CTC en el mismo circuito, y qué ve qg en cada una
14.8 🆕 El costo de la posselección como tasa (cos²(φ/2)): cuántos disparos cuesta "viajar en el tiempo" en un dispositivo 🧪

### Capítulo 15. Termodinámica de agujeros negros, caos cuántico y el decodificador de Yoshida–Kitaev
15.1 La paradoja de la pérdida de información de Hawking y la complementariedad de agujeros negros — §164 (contexto)
15.2 Caos cuántico y dispersión de información (scrambling): un micro-agujero negro de 3 qubits con un Clifford que esparce todo Pauli — §164
15.3 El protocolo de Hayden–Preskill con el decodificador de Yoshida–Kitaev: reconstrucción del mensaje arrojado al horizonte usando la radiación de Hawking — §164
15.4 Formulación en unidades qg y posselección asistida por el filtro de simetría: éxito 1/4, fidelidad 1 sin ruido — §164
15.5 ⚠️ Fidelidad recuperada 0,55–0,74 (> 1/2) **en modelos de ruido de IBM** (Brisbane, Torino, Fez); sin scrambling, 0,25. El índice original citaba IonQ: no hay corrida IonQ de este experimento — §164 🧪
15.6 🆕 Agujeros negros más grandes: scramblers de 5–7 qubits, el costo en compuertas y hasta dónde llega el filtro
15.7 🆕 La curva de Page en juguete: entropía de la radiación frente al tiempo de evaporación, leída con qg_S
15.8 🆕 Agujeros de gusano atravesables como teleportación (protocolo tipo ER = EPR) y la fidelidad leída en correladores qg

---

## PARTE VI. Ciberseguridad cuántica, generación aleatoria y fronteras post-cuánticas

### Capítulo 16. Demarcación: criptografía cuántica frente a criptografía post-cuántica (PQC)
16.1 La naturaleza de la PQC: algoritmos clásicos de retículos (ML-KEM/Kyber, ML-DSA/Dilithium), códigos y hash — §13, §14 (contexto)
16.2 Por qué qg no se aplica a la PQC: no hay operadores hermíticos, qubits ni esfera de Bloch en esos esquemas — §13, §14
16.3 El vector de ataque cuántico (Shor/QPE): el registro de autoestados proyecta a qg_Z = −1 sin resolver la fase — §13
16.4 Conclusión: qang es una herramienta de capa física y cuántica, no de criptografía algebraica clásica — §14
16.5 ⚠️ No existe un módulo `qang.crypto`: la criptografía vive en `examples/` (bb84_*, certified_qrng_*) y en `manuscript/crypto*`

### Capítulo 17. Distribución cuántica de claves (QKD) y BB84 en espacio qg
17.1 QBER en observables qg: QBER_Z = (1 − |qg_Z|)/2, QBER_X = (1 − |qg_X|)/2 — §40
17.2 El testigo de asimetría A_Z = qg(0) + qg(1): relajación T1 del instrumento o espía (Eve) — §40
17.3 Reducción de falsas alarmas ante degradación del hardware manteniendo la detección de intrusión — §40, §43, §44. 🔬 Las cifras "48–96 % → 0,6 %" no están en las notas (el 0,6 % de §40 dice otra cosa: el monitor qg *no* ve a Eve en ese ataque); tomar las cifras reales
17.4 Bloques finitos (cota de Tomamichel–Lim–Gisin–Renner): detección temprana de deriva sin acortar la clave — §43
17.5 Ataques combinados (deriva T1 con intercepción activa), clonación parcial, ataques en una base y deriva T2 — §49
17.6 BB84 en el simulador ruidoso de IonQ — §44

### Capítulo 18. Números aleatorios certificados (QRNG) y auditoría de canales
18.1 El fallo del análisis por sesgo de salida: 50/50 oculta la pérdida de aleatoriedad privada por decoherencia — §41
18.2 Cota cerrada de entropía mínima por coherencia transversal: H_min = −log₂[(1 + √(1 − qg_X² − qg_Y²))/2] — §41
18.3 Seguridad en el 100 % de las corridas con ruido frente al 0 % de los estimadores habituales — §41
18.4 Calibración de lectura asimétrica (protocolo §31): población térmica frente a error del detector — §31, §41
18.5 QRNG en el simulador calibrado de IonQ (Aria-1/Forte-1) con fugas y atenuación — §42

### Capítulo 19. Baterías cuánticas y redes 🆕 (no figuraba en los índices)
19.1 Baterías de qubits en qg: almacenamiento, certificación y carga bloqueada (ergotropía) — §56

---

## PARTE VII. Aprendizaje automático cuántico (QML) y el filtro de simetría

### Capítulo 20. Redes neuronales cuánticas que conservan el peso (WeightQNN)
20.1 Compuertas RBS (Reconfigurable Beam Splitter) como rotaciones de Givens que conservan el peso — §76
20.2 Arquitecturas en peso 1 (modelo E) y peso 2 (modelo W) sobre 5, 6 y 8 qubits — §75, §79, §90, §111, §114
20.3 Codificación de datos: normalización unaria, productos de pares, anillo y codificación dual — §16, §87, §101
20.4 Lecturas: qubit directo, cabezal lineal y correladores ⟨Z_iZ_j⟩; lectura de dos canales — §81, §95
20.5 Simulabilidad clásica y límites de la ventaja cuántica — §75, §80
20.6 La QNN multiclase: cada qubit una clase; 3, 5 y 8 clases a 8 qubits — §110, §114
20.7 QNN dentro de qiskit-machine-learning; filtro diferenciable en PennyLane — §131, §140
20.8 Activaciones QSP ponderadas en una KAN de una capa; procesamiento cuántico de señales en qg — §134, §135, §136

### Capítulo 21. El filtro de simetría qg y la exactitud bajo relajación T1
21.1 Operadores de Kraus del amortiguamiento de amplitud y el decaimiento sin saltos (1 − γ)^(k/2) — §77
21.2 Teorema: la posselección en el sector de peso k restaura exactamente el estado sin ruido bajo T1 homogéneo — §77
21.3 Fracción de disparos conservados: K = (1 − γ)^(k·d); cómo escala con qubits, peso, profundidad y disparos — §94
21.4 Gradiente idéntico: entrenar con ruido filtrado equivale a entrenar sin ruido; entrenar desde disparos — §77, §109, §112
21.5 La propiedad F5: invariancia ante derivas homogéneas; dependencia solo de los cocientes (1 − γ_q)/(1 − γ_r); T1 desigual corregido — §78, §82, §117
21.6 Techo analítico y regla de decisión: MSE ≈ (1 − z²)/(K·S) frente a la lectura cruda; fracción conservada baja — §96
21.7 Desfase (dephasing): el filtro más ZNE, a 5 y 8 qubits; el desfase dentro de las compuertas — §118, §122, §132, §133
21.8 Errores de compuerta y de lectura en el simulador de qang — §124, §129

### Capítulo 22. Calibración por eco y el sesgo de ruido (pull effect)
22.1 Fugas dentro del sector: errores que conservan el peso pero permutan la excitación — §105
22.2 Matrices de transferencia por eco M (5×5, 10×10, 28×28) y desmezcla con M^(−1/2) — §105, §107, §115
22.3 Supresión de errores de clasificación en backends simulados de IBM e IonQ — §106, §113. 🔬 La cifra "43–56 % en el valor de decisión" no está en las notas: verificar
22.4 El sesgo por ruido (pull effect) en clasificadores multiclase: el término K·L + (1 − K)·v_c — §121
22.5 Lecturas sin arrastre (pull-free): robustas incluso sin filtro — §123
22.6 Calibrar con el propio dispositivo; toda la cadena sobre modelos de ruido — §119, §120

### Capítulo 23. Geometría de la información, gradiente natural (QNG) y núcleos
23.1 La singularidad dE/dqg = −(1/sin θ)·dE/dθ y sus regularizaciones (Tikhonov, recorte) — §1
23.2 Información de Fisher cuántica en qg: F_Q = 1/(1 − qg_Z²); el ángulo qang es la coordenada plana de Fisher — §159
23.3 ⚠️ El QNG en coordenadas qg equivale al descenso euclidiano en θ: la ganancia 1,3× viene de la métrica de bloques qg_YY (4,6× con la métrica completa), no de la coordenada — §159
23.4 Descenso amortiguado en los polos: robustez ante tasas de aprendizaje agresivas en VQE — §9
23.5 Núcleos cuánticos filtrados: reparación de la matriz de Gram 2,4–6× — §161
23.6 Mesetas estériles (barren plateaus) en espacio qg; costo global frente a costo local qg — §12, §23

---

## PARTE VIII. Dinámica de muchos cuerpos, canales abiertos, materia y química

### Capítulo 24. Simulación hamiltoniana, caminatas cuánticas y VarQTE
24.1 Trotter de cadenas XY en el sector de peso 1 — §147
24.2 Caminatas cuánticas continuas: propagación balística recuperada (exponente 1,84 frente a 1,94) contra difusión por ruido (0,87) — §147
24.3 Cadenas XXZ con varias excitaciones: el filtro en pesos k = 1, 2, 3 — §60, §61, §148
24.4 Canales cuánticos como transformaciones afines de (qg_X, qg_Y, qg_Z) — §162
24.5 Evolución temporal variacional (VarQTE) a profundidad fija por el principio de McLachlan — §162
24.6 Ecuaciones diferenciales — §22
24.7 Dinámica de Hubbard 1D con filtros N y espín; testigos de la ley de Gauss en una teoría de gauge en red — §29, §66
24.8 Ising en campo transversal: ¿la brecha de entropía es un testigo de coherencia? — §34
24.9 Estados RBM: el plano I–η barrido por los marginales qg — §47

### Capítulo 25. Espectroscopía cuántica, PCA y química molecular
25.1 PCA cuántico de datos en el sector unario de peso 1 — §146
25.2 Rotaciones RBS adyacentes para aislar autovalores en qubits físicos — §146
25.3 Testigo de simetría molecular y número de ocupación: N_e = Σ(1 − qg_Z,i)/2 — §21, §52
25.4 H2 y LiH: precisión recuperada bajo T1 y el límite de los circuitos profundos; por qué el filtro ayuda a H2 y no a LiH — §21, §50, §53, §54
25.5 Filtros resueltos en espín (N↑, N↓), S² total y Λ (el m_l molecular) como tercer filtro — §28, §55, §69, §153
25.6 Solapamientos: inversion test unario con filtro frente al costo en ancillas del swap test — §149
25.7 Optimización con restricciones: QAOA con "exactamente K" — §30, §38

---

## PARTE IX. Corrección de errores, síndromes continuos y caracterización

### Capítulo 26. Corrección de errores y síndromes continuos
26.1 El código de 4 qubits de Leung para canales dominados por T1 y el criterio T2 > T1 — §27, §32
26.2 El síndrome como testigo qg continuo: seguir la deriva y cambiar de código — §33, §88, §92, §93, §99
26.3 Decodificador consciente de T1 con ruido a nivel de circuito; el código de superficie d = 3 en qg — §59, §64
26.4 Qutrits y qubits de borrado: la bandera |2⟩, la regla (1 − h) — §70, §70b, §71, §72, §73, §74
26.5 Corte de circuitos (circuit knitting): multiplicador de disparos γ = 1 + 2√(1 − qg²) para RZZ — §26
26.6 Benchmarking aleatorizado y ZNE en unidades qg; qg_S como señal de benchmarking — §10, §11

### Capítulo 27. Caracterización de hardware en qg
27.1 T1, T2, temperatura efectiva y lectura en un solo barrido — §31, §86
27.2 ⚠️ ¿Qué ruido es? qg_Z medio frente a XEB/HOP — §18. 🔬 La cifra "88–91 % con 100 disparos" no está en el repo: verificarla en §18 o sacarla 🧪
27.3 Errores coherentes de fase condicional y el crosstalk ZZ siempre activo — §145, §150
27.4 Cuantización del control: la grilla pareja en qg baja el error en 43 de 45 distribuciones (mediana 1,56×) — §17

---

## PARTE X. Evidencia experimental, hardware real y auditoría de falsación

### Capítulo 28. Metodología: el protocolo de falsación pre-registrada
28.1 La disciplina del pre-registro: hipótesis y predicciones comprometidas (commit o sha256) antes de correr — §46, §75–§164
28.2 ⚠️ Balance de aciertos y fallos: "123 cumplidas / 53 falladas" corresponde **solo a §75–§123** (línea QML). 🔬 Recontar sobre §1–§164
28.3 Bancos de prueba: modelos de ruido calibrados de IBM (Brisbane, Sherbrooke, Torino, Fez, flota de 21) e IonQ (Aria-1, Forte-1) — §20, §137, §144
28.4 🔬 Ejecución en hardware real: QuTech/Quantum Inspire (Tuna-5, Tuna-9, Tuna-17) no aparece en el repo. IonQ Forte-1 (§128) fue rechazado por el proveedor antes de aceptar el trabajo. **Hoy no hay ningún dato en hardware real**
28.5 🔬 El criterio de falsación para QPU: varianza del eco de inversión Var_θ(F_eco) sobre trayectorias continuas (no está en el repo)
28.6 🆕 🧪 Primera corrida en hardware real: IQM Resonance (Garnet/Emerald) — ver Capítulo 31

### Capítulo 29. Las ventajas empíricas confirmadas (dónde qg aporta un beneficio medible)
29.1 Restauración exacta bajo T1: precisión ideal recuperada en corridas pareadas — §80. 🔬 Las cifras "+2,5 puntos en peso 1, +16,5 en peso 2, 60 corridas" no están en las notas: verificar
29.2 Reducción del error de observables polares en backends ruidosos simulados de IBM e IonQ — §85, §144. 🔬 El rango "2,7–3,6×" no está en las notas: tomar las cifras de §85 y §144
29.3 Fugas dentro del sector mitigadas por eco: decisiones erróneas de 15 a 3 en 945 lecturas — §106–§108 (945 lecturas confirmado; verificar "15 → 3")
29.4 Cuantización del control (ver 27.4) — §17
29.5 Diagnóstico de T1 sin simulación clásica (ver 27.2) — §18
29.6 ⚠️ Mitigación sin circuitos extra: 3× en neutrinos de 3 sabores y 5,5× con estériles (simulado) — §151, §163
29.7 Ciberseguridad operativa: supresión de falsas alarmas en BB84 ante derivas térmicas — §40, §43

### Capítulo 30. Límites estructurales, equivalencias y resultados negativos
30.1 Ceguera a fases relativas y entrelazamiento multipartito en base Z: los estados de grafo — §8
30.2 La no monotonía de qg_S bajo amortiguamiento puro y la solución con el par (qg_S, qg_Z medio) — §10, §14
30.3 La equivalencia del QNG: no hay aceleración intrínseca por cambiar de coordenada — §159
30.4 Fallo del filtro en circuitos profundos: LiH con 60 compuertas entrelazantes y Hubbard con L ≥ 4 — §29, §50
30.5 WeightQNN no es supremacía cuántica frente a modelos cuadráticos clásicos — §75
30.6 Inaplicabilidad en estimación de fase pura (QPE, Shor, HHL) y en PQC — §13
30.7 La paradoja de las P-CTC en Torino, la fidelidad del decodificador y los demás fallos pre-registrados de §157–§164 — §157–§164
30.8 Lista consolidada de limitaciones conocidas — §14

### Capítulo 31. 🆕 🧪 Primer experimento en hardware real: IQM Resonance
31.1 Por qué IQM: compuertas nativas PRX y CZ, red cuadrada; cómo se compilan RBS y XX+YY
31.2 Módulo `qang/iqm.py` (análogo a `qang/ibm.py`): errores de lectura de la calibración, elección de cadena de qubits, filtro qg
31.3 Ensayo gratis con los backends simulados de IQM antes de gastar créditos
31.4 Presupuesto: trabajo de prueba mínimo (≈10 %), experimento principal (≈60 %), reserva (≈30 %); el token solo en variable de entorno
31.5 Experimento principal pre-registrado: neutrinos de 3 sabores (§151) — 3 qubits, ~18 compuertas de dos qubits
31.6 Candidatos siguientes: P-CTC con un qubit (§164), decodificador de Yoshida–Kitaev (§164), H2 (§20/§128)
31.7 Resultado, aciertos y fallos, comparación con los modelos simulados

### Capítulo 32. Arquitectura de software, ecosistema multi-SDK y ejecución física
32.1 El orden unificado `qang.interop` (Qiskit, Cirq, PennyLane, Braket, pytket) — §130
32.2 Primitivas V2 de Qiskit (SamplerV2, EstimatorV2) con observables filtrados — §131
32.3 Compilación consciente de sectores en IBM Runtime y elección de cadenas (`best_chain`); predecir la fracción conservada desde la calibración — §137, §138, §141
32.4 ⚠️ El ZZ estático: `ibm.add_static_zz` **simula** el crosstalk para medir cuánto ve el filtro; no lo mitiga — §150
32.5 ZNE con el filtro y planificación de disparos — §142, §143
32.6 Directrices para hardware superconductor (IBM, IQM) y de iones (IonQ Cloud / AWS Braket) — §128, §137, Cap. 31

## PARTE XI. Nanocuántica 🆕

### Capítulo 33. Nanocuántica: transmisión, ruido de partición y qubits a escala nanométrica
⚠️ Sin módulo ni sección de las notas todavía: cada punto parte de la ecuación de probabilidad del capítulo 1 y requiere desarrollo y verificación propios.
33.1 La probabilidad de transmisión por túnel T como probabilidad qang: θ_T = 2 arccos √T y su lectura en m-qg
33.2 La fórmula de Landauer G = (2e²/h) Σ T_n: cada canal de un contacto puntual cuántico es una lectura qang
33.3 Ruido de partición S = 2eI(1 − T) y factor de Fano F = 1 − T: la varianza binomial T(1 − T), estabilizada por el ángulo qang
33.4 Bloqueo de Coulomb y transistor de un electrón: la ocupación de un nivel como p₀ y qg_Z = −tanh(x/2)
33.5 Puntos cuánticos como qubits de carga y de espín: Rabi y lectura por sensor de carga en qg; fidelidad de lectura en m-qg
33.6 Centros NV en nanodiamante: contraste de ODMR como qg_Z y magnetometría nanométrica en m-qg

---

## Apéndices
A. Catálogo de las 15 compuertas fundamentales en representación qg
B. Fórmulas de transición, matrices de Gell-Mann 🔬 y polinomios de Chebyshev en circuitos paramétricos — §37
C. Protocolo de calibración por eco e inversión matricial en sectores de Hamming — §105–§108
D. Referencia rápida de la API de qang 0.12.0: core, formulation, sectors, qml, ibm, interop, pulse_area, neutrino, postselection, qpon, fisher, kernels, varqte, channels, angular, qsp, knitting, qec, statistics, polarization, geometric, clock_comparison, complementarity, proportions (⚠️ sin `crypto`; 🆕 `iqm`)
E. Declaración de falsación y registro pre-registrado de predicciones (recuento completo §1–§164)
F. Fundamento de análisis funcional: Riesz–Fréchet — Apéndice A de las notas

Bibliografía general y registro DOI de Zenodo (10.5281/zenodo.22832150 y preprints asociados).

---

## Resumen de pendientes antes de escribir

**Por demostrar o sacar (🔬):** analogía del decibelio (1.6); U3_qang y su verificación (2.2–2.3); qudits con Gell-Mann
(2.4); Householder, Grover como reflexión y la fórmula de transición (3.1–3.4); cifra del 33,9 % (4.5); Modelo Solar
Estándar (10.5); fenomenología 3+1 (11.5); enrutamiento metropolitano (13.5); P-CTC no lineales (14.6); cifras de
BB84 (17.3), de clasificación (22.3), de T1 (27.2), de 29.1 y 29.2; Quantum Inspire (28.4); Var_θ(F_eco) (28.5); recuento total (28.2).

**Propuestas nuevas (🆕):** relojes y metrología (Cap. 8); repetidores y teleportación de compuertas (13.6–13.7);
D-CTC (14.7) y costo de la posselección (14.8); agujeros negros más grandes, curva de Page y agujeros de gusano
atravesables (15.6–15.8); baterías (Cap. 19); el experimento IQM (Cap. 31).

**Libro en inglés:** `book/INDICE_MAESTRO_EN.md`, con la misma numeración punto por punto.
