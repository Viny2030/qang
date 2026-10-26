# Revisión del libro, capítulo por capítulo

Cada sección registra qué se comprobó de un capítulo, contra qué librería, qué coincide y qué no. Los scripts de
verificación (`tools/verify_capNN.py`) recalculan las afirmaciones sin usar qang, salvo para la comparación final,
e imprimen PASA/FALLA por afirmación.

## Protocolo de teoría (revisión reiniciada el 2026-10-26)

La revisión se reinicia desde el capítulo 1 con un criterio más estricto: además de recalcular los números, cada
capítulo se lee contra la teoría del capítulo 1 y cada identidad o razonamiento teórico se prueba, en sympy cuando es
una identidad y en al menos dos librerías cuánticas cuando es una afirmación sobre estados, ruido o muestreo. Los
criterios, que se responden uno por uno en cada capítulo:

- **T1. Conversiones.** $m=1000\,qg_Z=2000p_0-1000$, $p_0=(1000+m)/2000$, $\theta=\arccos(m/1000)$. Todo trío
  (p₀, qg_Z, m, θ) del texto se recalcula.
- **T2. Lineal en la probabilidad, no en el ángulo.** 1 m-qg $=\Delta p_0=0{,}0005$ en toda la escala. Un m-qg no es
  una unidad de ángulo: un error de rotación se da en grados o radianes, o se dice que es un corrimiento de lectura.
- **T3. Barras de error.** $\sigma_m=1000\sqrt{1-qg_Z^2}/\sqrt N$ (igual a $1000\vert\sin\theta\vert/\sqrt N$ sólo con
  $\theta=\arccos qg_Z$); dividida por el contraste $1-e_0-e_1$ si la lectura está corregida; con covarianzas si se
  combinan qubits medidos en los mismos disparos (§45 B).
- **T4. Ángulo qang frente a ángulo de Bloch.** $\theta_{qg}=\arccos qg_Z$ es función de $p_0$ y tiene varianza
  $1/N$ para *cualquier* estado; el ángulo polar de Bloch $\arccos(qg_Z/r)$ de un estado mixto no. El texto debe decir
  cuál de los dos usa.
- **T5. Dominio de validez.** Las fórmulas por método delta son asintóticas: con $Np$ por debajo de ~3, o junto a un
  polo, se usan intervalos exactos o de Wilson (§125, §51).
- **T6. Signos, polos y dirección del ruido.** $+1000$ m-qg es $\vert0\rangle$. El ruido despolarizante y el error
  aleatorio de amplitud llevan la lectura hacia el ecuador; la relajación T1, hacia $+1000$. El signo de cada
  corrimiento del texto debe coincidir con su causa física.
- **T7. La base aplicada.** Las lecturas de un qubit se dan en m-qg o qg; las cantidades que no son lecturas de un
  qubit (fugas, infidelidades, energías) no se fuerzan a la escala.
- **T8. Comparación honesta.** En la tabla "sin qang / con qang", una herramienta clásica (transformación arcoseno,
  intervalo de Wilson, h de Cohen, método delta) no se acredita a qang; donde sólo cambia la unidad, se dice
  "conveniencia".
- **T9. Pruebas.** Cada identidad del capítulo, en sympy; cada afirmación sobre estados o muestreo, en dos librerías
  (Qiskit, Cirq, PennyLane, QuTiP en `.venv-int`); cada cobertura o varianza "exacta", con sumas binomiales exactas.
  Todo queda en `tools/verify_capNN.py`.

Los hallazgos se registran con línea (ES; la EN es la misma salvo indicación) y no se corrige el texto sin la
decisión del autor.

## Capítulo 1 — La unidad qang, el mili-qang y la ecuación de probabilidad

Revisado el 2026-10-25 sobre qang 0.13.1. Script: `tools/verify_cap01.py` (en `.venv-int`: Qiskit 2.6.1, Cirq 1.6.1,
PennyLane 0.45.1, QuTiP 5.3.1, scipy 1.17.1). Resultado: **64 de 65 afirmaciones pasan**.

### Coherencia interna

- **ES y EN.** Todos los números de `book/es/cap01.qmd` aparecen iguales en `book/en/cap01.qmd` (comparación
  automática de los números de ambos textos), y las referencias § coinciden en orden y número.
- **Texto, script y §167/§168.** `tools/make_cap01_figures.py` sólo escribe figuras (no imprime números); regenerarlas
  no cambió ningún PNG. Los números del texto se recalcularon con `qang.probability` y coinciden con los de §167
  (ecuación, escala, resolución por disparos, presupuesto de compuerta, `analyze(750, 1000)`) y con §168 (hardware).

### Verificación independiente

| afirmación | librería | resultado |
|---|---|---|
| $p_0=\cos^2(\theta/2)$, $\theta=2\arccos\sqrt{p_0}$, $qg_Z=\langle Z\rangle=2p_0-1=\cos\theta$, $m=2000p_0-1000$, $\theta=\arccos(m/1000)$, sobre $R_y(\theta)\vert{}0\rangle$ en nueve ángulos (polos incluidos) | Qiskit, Cirq, PennyLane, QuTiP | PASA, error máximo $3\cdot10^{-13}$ |
| tabla de anclajes (`tbl-anclajes`), seis filas | QuTiP, Qiskit | PASA |
| `readings_from_p0(0.75)`: 60°, $qg_Z=0{,}5$, $qg_S=0{,}811$, 500 m-qg | PennyLane | PASA |
| $\sigma_m=1000\vert{}\sin\theta\vert{}/\sqrt N$ por muestreo: 2000 lotes de 1000 disparos a 30°, 60°, 90° y 150° | Qiskit (`StatevectorSampler`), Cirq (`Simulator.run`), PennyLane (`qml.sample`) | PASA en los 12 casos; p. ej. a 60°: 27,62 / 27,36 / 27,50 muestreado frente a 27,39 predicho |
| 27,4 m-qg a 60° con 1000 disparos; 30 000 disparos para 5 m-qg a 60°; $10^4$ y $10^6$ en el ecuador; 0,0447 rad (2,56°) y 8,1° junto al polo; 1 y 2 mrad por m-qg; $\Delta p_0=0{,}0005$ | aritmética de las fórmulas | PASA |
| `analyze(750, 1000)`: m = (500,0; 444,5; 551,7), θ = (1,047; 0,986; 1,110) rad = 60,0°, 56,5°, 63,6°, $qg_S$ = (0,811; 0,768; 0,852) | intervalo de Wilson de scipy (`binomtest(...).proportion_ci`) | PASA; qang usa $z=1{,}96$ y scipy el cuantil exacto, $\vert{}\Delta m\vert{}=0{,}001$ m-qg |
| presupuesto de compuerta: $X$ con $\sigma=0{,}05$ da $qg_Z=-0{,}988$ y 12,3 m-qg; $\pi/2$ no se corre | QuTiP (promedio de $R_x(\theta(1+\varepsilon))$ por cuadratura de Gauss–Hermite) | PASA: −0,9877 y 12,26 m-qg |
| distancia de Wootters 0,113 rad | QuTiP (`fidelity` de estados diagonales) | PASA: 0,11299 |
| mezcla 0,2/0,8: $qg_Z=-0{,}6$, entropía de von Neumann 0,72 | QuTiP | PASA: 0,7219 |
| Bell: $qg_Z$ local 0, entropía local 1 bit, conjunta normalizada 0,5 | Qiskit | PASA |
| $qg_S=H((1+qg_Z)/2)$ para estados mixtos | Cirq (50 matrices densidad al azar) | PASA, error $7\cdot10^{-16}$ |
| §168: pendiente 1,64 m-qg por $X$, $\chi^2$/gl 0,81, 63 m-qg y 8σ a $k=41$, residuos de 16 a 55 m-qg, M1 4 de 5, M2 razón 0,50 con probabilidad 0,013 | numpy y scipy sobre `examples/data/iqm_milliqang_qg_analysis.json` | PASA (1,6447; 0,814; 63,18 ± 7,75 = 8,15σ; −15,6 a −55,3; 0,0132) |

### Lo que no coincide

- **El ejemplo de la sección «Definición y escala»** (ES línea 74, EN línea 74): «Un estado con $qg_Z=0{,}502$ está a
  $501{,}8$ m-qg; su diferencia con el ideal $0{,}5$ es de $1{,}8$ m-qg». Con $m=1000\,qg_Z$, el valor es
  **502,0 m-qg** y la diferencia **2,0 m-qg**. El texto o bien quiso decir $qg_Z=0{,}5018$, o bien los m-qg deberían
  ser 502 y 2. Igual en los dos idiomas, así que no es un error de copia; **no se corrigió**, queda a decisión del
  autor.

### Corregido

- **Tabla de anclajes** (ES y EN): los kets $|0\rangle$, $|1\rangle$, $|+\rangle$, $|{+i}\rangle$ tenían `|` dentro de
  `$...$` en una tabla de Markdown, lo que parte la celda. Se reemplazó por `\vert{}`, en los dos idiomas. Sin
  cambio de contenido.

### No verificado

- Las afirmaciones de la sección del arcoseno con sumas binomiales exactas ($(n+\tfrac12)\operatorname{Var}(\hat\theta)$
  entre 1,000 y 1,001 con $n=100$; cobertura 0,930–0,962) y la cobertura del intervalo de Wilson para $qg_S$
  (0,90–0,99, §51): son resultados de §5, §51 y §125, no de §167, y el texto no fija la grilla de $p$ usada.
- El gradiente $dE/dq$, el óptimo de H₂ en $\theta\approx-0{,}22$ (§65) y la fórmula de $\operatorname{Var}(\hat\theta)$ para
  estados mixtos (§45): se tratan en capítulos posteriores.

### Revisión de teoría (reiniciada, 2026-10-26)

Parte 7 de `tools/verify_cap01.py` (sympy 1.14, Qiskit, Cirq, QuTiP, scipy). Resultado del script completo:
**92 de 93 afirmaciones pasan**; la única falla sigue siendo el ejemplo de 501,8 m-qg.

**Pruebas nuevas.**

| afirmación | herramienta | resultado |
|---|---|---|
| las nueve identidades del capítulo: $qg_Z=2p_0-1=\cos\theta$, $\theta=2\arccos\sqrt{p_0}$, $p_0=(1000+m)/2000$, $dm/d\theta$, $dE/dq$, $d\,qg_S/d\,qg_Z$, $\operatorname{Var}(\hat\theta)=1/N$, $\operatorname{Var}(\hat\phi)=1/N$ del arcoseno, $d(p,\text{polo})=\theta/2$ | sympy | PASA, todas exactas |
| §125 S1 y S2 (antes "no verificado"): $(n+\tfrac12)\operatorname{Var}(\hat\theta)$ con $n=100$ y cobertura de $\hat\theta\pm z/\sqrt{n+1/2}$, $p\in\{0{,}1,\dots,0{,}9\}$, ángulo de Anscombe | sumas binomiales exactas (scipy) | PASA: 1,0000–1,0009 y 0,930–0,962, idénticos al texto |
| §51 (antes "no verificado"): cobertura 0,90–0,99 del Wilson mapeado a $qg_S$ | sumas binomiales exactas, 12 casos | PASA: 0,905–0,988 (el texto venía de 2000 repeticiones) |
| ángulo qang frente a ángulo de Bloch en un estado mixto ($r=0{,}8$) | `DensityMatrix.sample_counts` de Qiskit sobre $\rho$ de QuTiP, $N=10^4$ | $N\operatorname{Var}(\arccos qg_Z)$ = 1,008 y 1,035; $N\operatorname{Var}(\arccos(qg_Z/r))$ = 1,57 y 5,05 frente a 1,56 y 4,84 de la fórmula (θ = π/2, π/8) |
| error de pulso: $\varepsilon_m(k\pi)$ y $\varepsilon_m(3\pi/2)$ | qang, ya validado con QuTiP | +12,3; −48,2; +105,1 m-qg; cero en 3π/2 |
| "lineal y no cuadrático indica error incoherente" (§168) | Cirq (`amplitude_damp` tras cada X) y QuTiP (sobre-rotación de 0,3 %) | PASA: término k² 3 % del faltante con T1 y 99 % con sobre-rotación |
| ídem sobre los datos de Garnet | χ² de $b\,k$ frente a $c\,k^2$ | PASA: $c\,k^2$ rechazado (χ²/gl 3,59, p = 1,8·10⁻⁴); $b\,k$ 0,82 |
| Bell: entropía conjunta normalizada 0,5 | Qiskit (`entropy`, `partial_trace`) | PASA, pero es la de Shannon de los resultados; la de von Neumann del registro es 0 |
| $(qg_Z,\operatorname{sign}qg_X)$ biunívoco en $(-\pi,\pi]$ | Cirq, 37 ángulos | PASA, error 1,3·10⁻¹⁵ |

**Criterios.** T1 cumple (todos los tríos; salvo el ejemplo de 501,8). T3, T5 y T9 cumplen. T2, T4, T6, T7 y T8
tienen los hallazgos de abajo.

**Hallazgos (pendientes de decisión del autor; no se tocó el texto).**

1. **T4 — el ángulo de los estados mixtos** (ES 283–285 y 384, EN 280–282 y 378; tabla ES 337). El capítulo define
   el ángulo qang como $\theta=\arccos qg_Z$, función exacta de $p_0$. Con esa definición, $\operatorname{Var}(\hat\theta)=1/N$
   vale para **todo** estado, puro o mixto: es la transformación arcoseno de una proporción binomial, y la
   probabilidad no sabe si el estado es puro (sympy y Qiskit, tabla de arriba). La fórmula
   $[1+(1-r^2)/(r^2\sin^2\theta)]/N$ que da el texto es la del ángulo polar de Bloch, $\arccos(qg_Z/r)$, como dice
   §45 A, pero el capítulo no lo dice y la presenta como corrección del ángulo qang. Lo mismo en el resumen y en la
   tabla ("para todo estado puro"). Propuesta: decir que el ángulo qang conserva $1/N$ en cualquier estado y que lo
   que pierde la estabilización es la estimación del ángulo geométrico cuando $r<1$.
2. **T6 — "crece con el cuadrado del ángulo"** (ES 135, EN 133). $\varepsilon_m\approx-500\,\sigma^2\theta^2\cos\theta$
   crece como $\theta^2$ sólo en los múltiplos de π, y con signo alterno (+12,3; −48,2; +105,1 m-qg para π, 2π, 3π);
   en los múltiplos impares de π/2 vale cero aunque $\theta^2$ sea mayor. Lo que crece con el cuadrado es la
   dispersión del ángulo, no el corrimiento. Propuesta: "crece con el cuadrado del ángulo en los múltiplos de π, y
   siempre hacia el ecuador".
3. **T2 — "un error de rotación de decenas de m-qg"** (ES 316–317, EN 312). El m-qg no es unidad de ángulo; lo que
   está en m-qg es el corrimiento de la lectura (16–55). Como ángulo, con la lectura corregida por el contraste, son
   sobre-rotaciones de +1,3°, +2,2°, +3,9° y +2,4° (parte 6 del script). Propuesta: "leen entre 16 y 55 m-qg por
   debajo, sobre-rotaciones de 1,3° a 3,9°".
4. **T8 — la barra de error uniforme se acredita a qang** (tabla ES 337, EN 332). "A favor" para una varianza $1/N$
   que es la transformación arcoseno de Bartlett, Anscombe y Freeman–Tukey, que el propio capítulo cita en la sección
   del arcoseno y que existe sin qang. Lo que qang aporta es que esa coordenada es el parámetro nativo de la
   compuerta (§125, veredicto). Propuesta: "conveniencia: la coordenada estabilizada de la estadística clásica
   coincide con el parámetro de la compuerta".
5. **T7 — entropía conjunta del estado de Bell** (ES 287–289, EN 284). "La entropía conjunta normalizada es 0,5. El
   registro es globalmente puro" es correcto sólo si la entropía conjunta es la de Shannon de los resultados en la
   base computacional; la de von Neumann del registro es 0. Unas líneas antes el texto usa la de von Neumann para la
   mezcla 0,2/0,8, así que el lector puede leer las dos como la misma. Propuesta: "la entropía de Shannon conjunta
   de los resultados, normalizada, es 0,5".
6. **T1 — 501,8 m-qg** (ES/EN 74): sin cambios desde la primera revisión; sigue pendiente.
