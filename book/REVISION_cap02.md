## Capítulo 2 — Compuertas nativas, álgebra cerrada y el operador universal

Revisado el 2026-10-25 sobre `book/es/cap02.qmd` y `book/en/cap02.qmd` (qang 0.13.1).

**Cómo se comprobó.**

- ES contra EN: cada número del texto en español aparece igual en el inglés (con punto decimal). Sin erratas.
- Script del capítulo: `tools/make_cap02_04_figures.py` sólo escribe figuras y no imprime números; la parte del
  capítulo 2 (`fig_area`) usa `qang.probability.pulse_error_milliqang`. Los números de la sección del área se
  recalcularon con esa función y con los datos guardados de `examples/data/pulse_area_qg.json` (§160).
- Notas citadas: §6, §68, §70, §130, §160, §166 de `RESEARCH_NOTES.md`.
- Verificación independiente: `tools/verify_cap02.py` (`.venv-int`), con Cirq 1.6.1, PennyLane 0.45.1 y QuTiP
  5.3.1. Resultado: **28/28 PASA**.

**Qué se comprobó y contra qué.**

| afirmación | librería | libro | recalculado | estado |
|---|---|---|---|---|
| `u3_qang(q, φ, λ)` = U3(arccos q, φ, λ), 2000 compuertas | PennyLane `qml.U3` (fase incluida) | 3,5·10⁻¹⁴ (contra Qiskit) | 1,8·10⁻¹⁶ | coincide |
| ídem | Cirq `QasmUGate` (salvo fase global e^{−i(φ+λ)/2}, convención de Cirq) | — | 1,8·10⁻¹⁵ | coincide |
| ídem | QuTiP, R_z(φ)R_y(θ)R_z(λ) (salvo fase global) | — | 6,8·10⁻¹⁶ | coincide |
| ídem, repetido en Qiskit 2.6 | Qiskit `UGate` | 3,5·10⁻¹⁴ | 1,8·10⁻¹⁶ | ver pregunta 1 |
| I, X, Z, H en (q, φ, λ) de la @tbl-compuertas; lecturas +1000, −1000, +1000, 0 m-qg | Cirq, PennyLane, QuTiP | tabla | ≤ 1,2·10⁻¹⁶ | coincide |
| matriz de transición \|U_ij\|² = T(q), doblemente estocástica, sin fases | PennyLane | fórmula | 5,6·10⁻¹⁶ | coincide |
| `transition_matrix(0.3)` y p₀ = 0,65 ↔ +300 m-qg | qang | [[0.65, 0.35], …] | igual | coincide |
| p₀ tras U3_qg\|0⟩ = (1+q)/2 | simulador de Cirq | ecuación de probabilidad | 6·10⁻⁸ (float32) | coincide |
| capacidad del canal = 1 − qg_S | información mutua con numpy | fórmula | 1,7·10⁻¹⁶ | coincide |
| p₀ = 0,75 ↔ +500 m-qg ↔ R_y(60°) | PennyLane | 60° | 60,000000° | coincide |
| Gell-Mann diagonales d = 3 | PennyLane `qml.GellMann` (λ₃, λ₈) | fórmula | igual | coincide |
| d = 2: λ₁ = σ_z, ⟨λ₁⟩ = qg_Z | QuTiP | fórmula | igual | coincide |
| Tr(λ_jλ_k) = 2δ_jk y p = 1/d + ½Σ⟨λ_k⟩diag(λ_k), d = 2…8 | QuTiP (`expect`) | identidad | 3,3·10⁻¹⁶ | coincide |
| R_X(θ) = −R_X(θ − 2π) | Cirq, PennyLane | identidad | 1·10⁻¹⁵ | coincide |
| R_ZZ(θ+π) = R_ZZ(θ)(−i Z⊗Z) | Cirq `ZZPowGate`, PennyLane `IsingZZ` | identidad | 5,6·10⁻¹⁶ | coincide |
| `minimal_area_angle(3.5, π)` = (0.358, 1) | PennyLane | (0.358, 1) | (0,3584, 1) | coincide |
| qg_Z = cos θ e^{−θ²σ²/2} | QuTiP + cuadratura de Gauss–Hermite | fórmula | 4,4·10⁻¹⁶ | coincide |
| X con σ = 0,05: qg_Z = −0,988 | Monte Carlo en Cirq (4000 muestras) | −0,988 | −0,98755 ± 0,00028 | coincide |
| corrimiento de una X, σ = 0,05 | QuTiP | 12,3 m-qg | 12,261 | coincide |
| R_X(324°) / representante R_X(−36°) | QuTiP; misma compuerta en Cirq | −31,7 / −0,4 m-qg | −31,700 / −0,399 | coincide |
| resolución de 10⁴ disparos en el ecuador | qang.probability | 10 m-qg | 10,00 | coincide (ver pregunta 3) |
| error fijo ε a primer orden | qang.pulse_area | 1000 ε θ sin θ | −1000 ε θ sin θ | coincide en módulo |
| estado W: qg_Z = 1/3, entropía log₂3/3 = 0,528 | QuTiP | 333 m-qg, 0,528 | 0,3333, 0,5283 | coincide |
| σ_m con 20 000 disparos; desviación de 316 m-qg | qang.probability | 6,7 m-qg, 2,6σ | 6,667, 2,60σ | coincide |
| QAOA: área 4,4×, error 4,1×, período π 1,14× | `examples/data/pulse_area_qg.json` | 4,4 / 4,1 / 1,14 | 4,39 / 4,07 / 1,140 | coincide (ver pregunta 2) |
| tabla de exposición (§68): 0,42; 0,16–0,24; 0,0006–0,047; 0; fugas 16–52, 20–78, 0,4–17, 0,0–2,0 %; ρ = 0,80; ≥ 98 % | §68 | tabla | igual a §68 | coincide con las notas (no se re-corrió) |
| estado W en tres kits: 0,316–0,343 | §130 | igual | igual | coincide con las notas (no se re-corrió) |

**Lo que no coincide o merece una pregunta al autor** (no se corrigió el texto):

1. **3,5·10⁻¹⁴ contra Qiskit.** Con Qiskit 2.6.1 y 2000 compuertas al azar la diferencia máxima es 1,8·10⁻¹⁶, no
   3,5·10⁻¹⁴. Probablemente el valor venga de otra versión o de otro muestreo; no es un error de fondo (la cota del
   libro es más holgada), pero no se reproduce. ¿Se actualiza a "≈ 2·10⁻¹⁶ (Qiskit 2.6)" o se deja como cota?
2. **1,14× mezcla estadísticos.** El 4,4× y el 4,1× son cocientes de medianas (101,34/23,10 y 0,1337/0,0329); el
   1,14× es la mediana de los cocientes por instancia, que es el estadístico de la predicción P4. Con el mismo
   estadístico que los otros dos, el período π baja el error 1,37× (0,0329/0,0240) y el área 1,18× (§160 dice 1,17;
   23,104/19,660 = 1,175). El texto dice "lo bajó 1,14× más" como si fuera comparable con el 4,1×. ¿Se aclara que
   es la mediana de los cocientes?
3. **La X comparada con la resolución del ecuador.** "El corrimiento de una X (12,3 m-qg) es del orden de la
   resolución de 10⁴ disparos en el ecuador (10 m-qg)" compara dos puntos distintos. En la lectura real de la X
   (−987,7 m-qg) la resolución de 10⁴ disparos es 1000·|sin θ|/√N = 1,6 m-qg, así que el corrimiento está a ~8σ:
   es mucho más visible de lo que sugiere la frase. ¿Se agrega el valor en la lectura real?

### Teoría qang, m-qg y probabilidad

1. **Conversiones.** Todos los tríos cumplen m = 1000 qg_Z = 2000 p₀ − 1000 y θ = arccos(m/1000): p₀ = 0,75 ↔ +500 m-qg
   ↔ 60°; p₀ = 0,65 ↔ +300 m-qg (`transition_matrix(0.3)`); I, Z en +1000, X en −1000, H en 0; qg_Z = −0,988 ↔
   −987,7 m-qg; estado W qg_Z = 1/3 ↔ 333 m-qg, 0,316 ↔ 316. Recalculados con `qang.probability`. **Cumple.**
2. **m-qg lineal en la probabilidad.** Ninguna frase trata el m-qg como un ángulo fijo. La sección del área habla de
   "corrimiento en m-qg" y de "sensibilidad −sin θ" por separado, sin convertir m-qg en grados. **Cumple.**
3. **Barras de error.** σ_m = 1000|sin θ|/√N se usa bien en el estado W (6,7 m-qg, 2,6σ) y en el ecuador (10 m-qg con
   10⁴ disparos). Única observación: la comparación de la X con la resolución del ecuador (pregunta 3). **Cumple, con
   una precisión pendiente.**
4. **Base del capítulo 1 aplicada.** Las lecturas de compuertas, el error de pulso y el estado W están en m-qg. La
   sensibilidad d qg_Z/dθ = −sin θ podría darse también en m-qg por radián (−1000 sin θ), pero no hace falta. La
   tabla de exposición y la fuga de T1 quedan en fracciones y porcentajes, que es lo correcto: no son lecturas de un
   qubit. **Cumple.**
5. **Comparación honesta.** La tabla dice "conveniencia" donde sólo cambia la unidad (preparar el estado), atribuye
   la mejora del área al principio físico y no a la unidad, y declara "ninguna demostrada" para qudits. Un matiz:
   la fila de la matriz de transición dice "a favor" para poblaciones, pero ese número es \|U₀₀\|² = p₀, que
   cualquier kit da sin qang; lo que qang aporta es la forma (un número en m-qg), no una información nueva. Podría
   decir "conveniencia" para poblaciones. **Cumple, con un matiz.**
6. **Signos y polos.** +1000 m-qg es \|0⟩ (I y Z no mueven poblaciones), −1000 es \|1⟩ (X las intercambia). El
   corrimiento de pulso es siempre hacia el ecuador: +12,3 para la X (lectura negativa) y −31,7 para R_X(324°)
   (lectura positiva); el texto da los dos signos bien, aunque no dice explícitamente "hacia el ecuador". **Cumple.**
