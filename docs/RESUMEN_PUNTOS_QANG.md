# qang: qué aporta en cada línea (resumen al 17 de octubre de 2026)

Cada afirmación remite a una sección de `RESEARCH_NOTES.md` con predicciones registradas antes de correr
(sha256 en `examples/data/*.prereg`) e informadas como aprobadas o fallidas. Todo lo que dice "en dispositivo"
es sobre modelos de ruido de dispositivos IBM simulados, no hardware real.

## La idea en una línea

Un qubit se describe por lo que se mide: qg_Z = ⟨Z⟩ y el ángulo qang arccos(qg_Z). Cuando un circuito conserva
un número (el peso de Hamming, el sabor, la masa, el número de fotones), los disparos que lo violan se
descartan: **el filtro qg**. Casi todos los resultados de abajo son una de esas dos cosas.

## 1. Optimización y entrenamiento de circuitos variacionales

| Afirmación habitual | Qué muestra qang | § |
|---|---|---|
| "Parametrizar en probabilidad acelera el parameter-shift" | **Falso.** Ninguna regla de desplazamiento baja de 2 evaluaciones por parámetro en ninguna coordenada; pasar a qg solo agrega el polo −1/sin θ. | 1, 159 |
| "Parametrizar en probabilidad aplana el paisaje" | **Al revés.** La información de Fisher es 1 en el ángulo y 1/(1−qg²) en qg: el ángulo es la coordenada plana. El descenso en qg no converge nunca (0 de 20). | 15.1, 159 |
| "qg mitiga los barren plateaus" | **No.** La reparametrización solo cambia la escala; el costo local qg ayuda con circuitos poco profundos. | 12, 23 |
| Gradiente natural | **Sí aporta:** en ansätze reales RY+CZ la métrica tiene diagonal 1/4 exacta y la curvatura son correladores qg_YY, medibles con un circuito por capa (1.3× más rápido que el descenso en θ). | 159 |
| VQE / QAOA | QAOA con restricción "exactamente K" y filtro; VQE de LiH con filtros N, S_z y Λ. | 30, 153 |

## 2. Compilación y pulsos

| Tema | Resultado | § |
|---|---|---|
| Ángulo → pulso | Regla del área mínima: RX período 2π, RZZ período π (con Z virtuales). Conserva el unitario exacto. El error sigue al área (4.1× al reducir módulo 2π, 1.14× más con período π). Aplica a compuertas fraccionarias; en IBM las rotaciones comunes tienen duración fija. | 160 |
| Sensibilidad | dqg_Z/dθ = −sin θ: máxima en π/2, nula a primer orden en 0 y π; el ruido de amplitud amortigua como e^(−θ²σ²/2). | 160 |
| T1/T2 | Caracterización de T1, T2, temperatura y lectura en un barrido qg; ZZ siempre activo con duraciones calibradas. | 31, 150 |

## 3. Aprendizaje automático cuántico

| Tema | Resultado | § |
|---|---|---|
| Codificación | Codificación qg (arccos): L capas = polinomio de Chebyshev de grado L exacto. | 16 |
| Kernels (QSVC) | El filtro qg repara la matriz de Gram 2.4–6× y le gana a la normalización diagonal sin sus circuitos extra; la precisión casi no cambia. | 161 |
| QNN | Filtro contra T1, lectura sin sesgo inmune a T1, entrenamiento con ruido calibrado. | 75–124, 139–141 |

## 4. Simulación y sistemas abiertos

| Tema | Resultado | § |
|---|---|---|
| Trotter | El filtro permite circuitos más profundos y mantiene balística una caminata cuántica. | 147, 148 |
| Evolución variacional | Profundidad fija (48 compuertas) le gana al mejor Trotter en t = 2 en tres dispositivos. | 162 |
| Kraus | Todo canal es un mapa afín de (qg_X, qg_Y, qg_Z); T1 lleva qg_Z exponencialmente al equilibrio. | 162 |

## 5. Interoperabilidad y benchmarks

Compuertas qg para Qiskit, Cirq y PennyLane; lecturas de Qiskit, Cirq, PennyLane, Braket y pytket con un solo
orden de qubits; IBM Runtime (pase de transpilador, corrección de lectura, selección de qubits, plan de
disparos); qiskit-machine-learning (§6, §130, §131, §137–§144). Pendiente: anotaciones qg en OpenQASM 3.

## 6. Neutrinos (§151–§156, §163)

La oscilación es una rotación; tres sabores entran en el sector de peso 1 y el filtro aplica (≈3×). En el
máximo, el ángulo de mezcla es un cuarto del ángulo qang, con barra de error independiente de θ. Materia de la
Tierra, gases densos (se conserva el número de ν₂), Sol (todo el Sol es una rotación). La jerarquía de masas se
lee a través de la materia; las fases de Majorana son invisibles (el transpilador las borra); un neutrino
estéril es un cuarto qubit. Artículo: `manuscript/qang_neutrinos.pdf` (español: `qang_neutrinos_es.pdf`).

## 7. Teleportación, tiempo y agujeros negros (§157, §158, §164)

En una red óptica pasiva el recurso óptimo no es un par de Bell (análogo del squeezing óptimo); su qg_Z es
toda la regla de diseño. Que los otros usuarios avisen es un filtro de sector. La fidelidad es una lectura qg
lineal del recurso, y dual-rail con filtro detecta toda pérdida. Los bucles temporales postseleccionados (P-CTC)
y el decodificador de agujeros negros de Yoshida–Kitaev son teleportación con postselección: la paradoja del
abuelo tiene probabilidad 0 y el decodificador recupera el mensaje (fidelidad 0.55–0.74) en tres dispositivos.
Son modelos de información, no afirmaciones sobre el espacio-tiempo.

## Lo que qang no hace

No da ventaja cuántica ni acelera algoritmos; no cura los barren plateaus; el filtro no ve errores dentro del
sector; todos los resultados de dispositivo son simulados (el primer punto en hardware real sigue pendiente).

## Notebooks (Colab)

`qang_start_here`, `qang_qml` (EN/ES), `qang_ibm_recent`, `qang_neutrinos` (EN/ES), `qang_quantum_numbers` (EN/ES),
`qang_qpon_teleport` (EN/ES), `qang_methods` (EN/ES).
