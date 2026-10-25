# Revisión del libro, capítulo por capítulo

Cada sección registra qué se comprobó de un capítulo, contra qué librería, qué coincide y qué no. Los scripts de
verificación (`tools/verify_capNN.py`) recalculan las afirmaciones sin usar qang, salvo para la comparación final,
e imprimen PASA/FALLA por afirmación.

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
