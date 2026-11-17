# Capítulo 44 — IA agéntica cuántica leída con qang: políticas de Born, el ángulo como gradiente natural, la lectura de decisión y el reparto de disparos entre herramientas
# Chapter 44 — Quantum Agentic AI Read with qang: Born Policies, the Angle as Natural Gradient, the Decision Reading and Shot Allocation Across Tools

**Actualizado:** 17 de noviembre de 2026 · **Updated:** November 17, 2026

> **Resumen.** ¿Qué le aporta qang a un agente autónomo híbrido, y qué no? Se toma el agente mínimo: dos acciones cuya política es la medida de un qubit, $\pi(0)=\cos^2(\theta/2)$. Resultados, todos con verificador: (i) la información de Fisher de la política es 1 para todo $\theta$, y el flujo del gradiente en $\theta$ es *exactamente* el gradiente natural del logit, $\phi_t=\phi_0+\eta\Delta t$ con $\Delta=r_0-r_1$: la acción peor baja a $1\%$ en $4{,}6/(\eta\Delta)$ pasos, frente a $108/(\eta\Delta)$ con el gradiente plano en el logit ($23{,}5\times$); (ii) cerca de un polo el gradiente se anula y la varianza no, de modo que el signo del gradiente cuesta 100 veces más episodios a $\theta=0{,}05$ que en el ecuador y un agente en el polo no se mueve nunca sin un piso de exploración; (iii) la lectura de decisión ($qg_Z$, $qg_S$, $qg_{SNR}$, disparos para el signo) se calcula antes de actuar; (iv) repartir los disparos entre $K$ lecturas con $n_i\propto\sin\theta_i$ baja la suma de varianzas por el factor $K\sum\sin^2\theta_i/(\sum\sin\theta_i)^2$. **Estado: `exact`**; no hay ganancia medida, y una política de un solo qubit es indistinguible de una moneda clásica sesgada. Módulo: `qang.qagent` (NumPy). Verificación: `tools/verify_cap44.py`.
>
> **EN abstract.** What does qang give a hybrid autonomous agent, and what does it not? The minimal agent: two actions whose policy is the measurement of one qubit, $\pi(0)=\cos^2(\theta/2)$. Results, all verified: (i) the policy's Fisher information is 1 for every $\theta$, and the gradient flow in $\theta$ is *exactly* the natural-gradient flow of the logit, $\phi_t=\phi_0+\eta\Delta t$ with $\Delta=r_0-r_1$: the worse action falls to $1\%$ in $4.6/(\eta\Delta)$ steps, against $108/(\eta\Delta)$ for plain gradient in the logit ($23.5\times$); (ii) near a pole the gradient vanishes and its variance does not, so the gradient sign costs 100 times more episodes at $\theta=0.05$ than at the equator, and an agent at the pole never moves without an exploration floor; (iii) the decision reading ($qg_Z$, $qg_S$, $qg_{SNR}$, shots for the sign) is computed before acting; (iv) splitting the shots among $K$ readings with $n_i\propto\sin\theta_i$ lowers the sum of variances by the factor $K\sum\sin^2\theta_i/(\sum\sin\theta_i)^2$. **Status: `exact`**; no measured gain is claimed, and a single-qubit policy is indistinguishable from a classical biased coin.

---

## 44.1 Reglas de lectura

1. Cada resultado entra con su función en `qang.qagent` y su línea en `tools/verify_cap44.py`. Escalas: $m\text{-qg}=1000\,qg_Z$ (lectura); $\theta$ en radianes, con $\pi/2$ rad $=250$ qga y 1 vuelta $=1000$ qga.
2. Se optimiza en $\theta$, nunca en $qg_Z$ sin regularizar: el gradiente plano en $qg_Z$ diverge en los polos (Cap. 34, `THEORY_CORE.md` §2.1). Ese resultado negativo se respeta aquí y se explica con una fórmula (§44.2).
3. Recompensas, pasos y piso de exploración son **parámetros** del modelo, no mediciones. Donde aparece un número, sale de la fórmula con ese parámetro.
4. Ningún estatus `measured gain`: el capítulo no compara contra un agente clásico entrenado en un entorno real, y no repite hipótesis ya refutadas del libro (Cap. 3, Cap. 34).

## 44.2 Fundamentación teórica y formulación matemática

**El agente mínimo.** Un agente de dos acciones prepara $R_Y(\theta)|0\rangle$ y mide en $Z$; la acción es el resultado. Entonces

$$
\pi(0|\theta)=\cos^2\frac\theta2=p_0,\qquad \pi(1|\theta)=\sin^2\frac\theta2,\qquad qg_Z=\cos\theta=\pi(0)-\pi(1),
$$

y el estado de decisión es el punto de la esfera de Bloch; la política es un meridiano. Con recompensas medias $r_0,r_1$ y $\Delta=r_0-r_1$, el retorno esperado es $J(\theta)=r_0\pi(0)+r_1\pi(1)$ y

$$
\partial_\theta J=-\frac{\Delta\sin\theta}{2},\qquad
\partial_\theta\log\pi(0)=-\tan\frac\theta2,\qquad \partial_\theta\log\pi(1)=\cot\frac\theta2 .
$$

**Información de Fisher.** $I(\theta)=\pi(0)\tan^2\tfrac\theta2+\pi(1)\cot^2\tfrac\theta2=\sin^2\tfrac\theta2+\cos^2\tfrac\theta2=1$ para todo $\theta$: es la métrica de Fubini–Study en el meridiano (Cap. 9). En $qg_Z$ la misma información es $1/(1-qg_Z^2)$, que diverge en los polos; por eso el gradiente plano en $qg_Z$ no sirve y el natural en $qg_Z$ coincide con el euclídeo en $\theta$ (Cap. 34).

**El flujo del gradiente.** Con el gradiente exacto, $\dot\theta=\eta\,\partial_\theta J=-\tfrac{\eta\Delta}{2}\sin\theta$, que se integra en forma cerrada:

$$
\tan\frac{\theta_t}{2}=\tan\frac{\theta_0}{2}\,e^{-\eta\Delta t/2}.
$$

Con el logit $\phi=\ln\frac{\pi(0)}{\pi(1)}=2\ln\cot\frac\theta2$ esto es $\phi_t=\phi_0+\eta\Delta t$: **el descenso por gradiente plano en $\theta$ es el gradiente natural en el logit**, porque la información de Fisher del logit es $\pi_0\pi_1$ y $F^{-1}\partial_\phi J=\Delta$. El tiempo para que la acción peor baje a $\pi_\ast$ es

$$
T_\theta=\frac{\ln\frac{1-\pi_\ast}{\pi_\ast}-\phi_0}{\eta\Delta},\qquad
T_{\text{logit plano}}=\frac{F(\phi_\ast)-F(\phi_0)}{\eta\Delta},\quad F(\phi)=2\sinh\phi+2\phi,
$$

y $T_\theta\propto\ln(1/\pi_\ast)$ frente a $T_{\text{logit plano}}\propto1/\pi_\ast$. La diferencia no es una ventaja intrínseca de lo cuántico: es la elección de la métrica (de un precondicionador). El gradiente natural en el logit da la misma curva que el plano en $\theta$.

**El polo.** El estimador REINFORCE [@williams1992] $g=r_a\,\partial_\theta\log\pi(a)$ es insesgado, con

$$
\mathbb E[g]=-\frac{\Delta\sin\theta}{2},\qquad
\mathbb E[g^2]=r_0^2\sin^2\tfrac\theta2+r_1^2\cos^2\tfrac\theta2 ,
$$

de modo que cerca de un polo la media se anula como $\sin\theta$ y la varianza queda en $r_1^2$ (o $r_0^2$): los episodios para resolver el signo, $z^2\mathrm{Var}(g)/\mathbb E[g]^2$, divergen como $1/\sin^2\theta$. Y en $\theta=0$ o $\pi$ el gradiente es exactamente cero: un agente que llega a un polo malo no sale. La salida es un *piso de exploración*: $\theta\in[\theta_{\min},\pi-\theta_{\min}]$ con $\theta_{\min}=2\arcsin\sqrt{p_{\min}}$ garantiza $\pi(a)\ge p_{\min}$.

**La lectura de decisión.** Antes de actuar, las medidas de qang dicen cuán decidida está la política: $qg_Z$ y su lectura $m_Z$ en m-qg, la entropía $qg_S=H(p_0)$ en bits, $qg_{SNR}=|\cot\theta|$ y los disparos para que el signo de $qg_Z$ quede a $z$ desviaciones, $(z/qg_{SNR})^2$ (Cap. 1). En el ecuador $qg_S=1$ bit y el signo no se decide con ningún número de disparos.

**Orquestación de herramientas: el reparto de disparos.** Si un agente consulta $K$ lecturas $qg_{Z,i}$ (una por herramienta o por subrutina) con un presupuesto $B$, la suma de varianzas es $\sum_i\sin^2\theta_i/n_i$, y con $\sum n_i=B$ el mínimo es $n_i\propto\sin\theta_i$:

$$
\frac{V_{\text{uniforme}}}{V_{\text{óptimo}}}=\frac{K\sum_i\sin^2\theta_i}{\big(\sum_i\sin\theta_i\big)^2}\ \ge1\quad\text{(Cauchy–Schwarz)},
$$

con igualdad solo si todos los $\sin\theta_i$ coinciden. El aporte de qang es que $\sin\theta_i=\sqrt{1-qg_{Z,i}^2}$ se conoce de una lectura piloto.

**Qué no hay.** Una política de un qubit tiene la misma distribución que una moneda clásica de sesgo $p_0$: el hardware cuántico no cambia la política ni su aprendizaje. Las ventajas de aprendizaje documentadas para políticas cuánticas [@jerbi2021; @dunjko2016; @skolik2022] se refieren a modelos con varios qubits y entrelazamiento, o a acceso cuántico al entorno, y este capítulo no las reproduce ni las afirma.

## 44.3 Arquitectura híbrida e integración con `qang` v0.16

```text
 entorno ──obs──► [PERCEPCIÓN: qubit(s) de estado → qg_Z, qg_X]            (Qiskit)
                         │ lecturas con intervalos (qang.probability, qang.statistics)
                         ▼
              [RAZONAMIENTO clásico]  ◄── qang.qagent: decision_report, time_to_wrong_probability,
                         │                 episodes_for_gradient_sign, neyman_allocation
                         │ plan: disparos por herramienta, paso η, piso p_min
                         ▼
              [ORQUESTADOR de herramientas]  (planificador determinista; un LLM puede ocupar este lugar [@yao2023react])
                         │ llama a: simulador/QPU · Grover (Cap. 43) · estimador de qg_Z
                         ▼
              [ACCIÓN: un disparo del qubit de política] ──► entorno ──recompensa R──┐
                         ▲                                                           │
                         └──────── actualización REINFORCE en θ (con piso) ◄─────────┘
```

Lo cuántico ocurre en dos recuadros, la percepción (opcional) y la acción; el razonamiento, el orquestador y la actualización son clásicos. Lo que qang agrega al razonamiento es un *chequeo previo*: cuánto confía la política en su acción, cuántos pasos predice el flujo, cuántos episodios cuesta el signo del gradiente y cómo repartir los disparos.

## 44.4 Implementación práctica (`qang` + Qiskit)

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler          # simulador exacto de Qiskit 2.x
from qang.qagent import (decision_report, exploration_floor, score, time_to_wrong_probability,
                         neyman_allocation, uniform_to_neyman_ratio)

MEANS, ETA, P_FLOOR, STEPS = (0.7, 0.3), 0.05, 0.01, 400    # entorno Bernoulli y parámetros del agente
LO = exploration_floor(P_FLOOR)                              # θ_min = 0.2003 rad
sampler = StatevectorSampler(seed=44)
rng = np.random.default_rng(44)

# 1. RAZONAMIENTO previo: qué predice el flujo y cuán decidida está la política
print(round(time_to_wrong_probability(np.pi / 2, ETA, MEANS[0] - MEANS[1], 0.05), 1))   # 147.2 pasos para π(peor) ≤ 5 %
d = decision_report(0.4)
print(round(d["qg_Z"], 3), round(d["m_Z"]), round(d["qg_S"], 3), d["shots_for_sign"])    # 0.921 921 0.24 2.0

# 2. ACCIÓN: un disparo del qubit de política
def act(theta):
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    counts = sampler.run([qc], shots=1).result()[0].data.c.get_counts()
    return int(next(iter(counts)))                           # 0 o 1

# 3. Bucle percepción-razonamiento-acción: REINFORCE en θ con piso de exploración
theta = np.pi / 2
for _ in range(STEPS):
    a = act(theta)
    r = float(rng.random() < MEANS[a])                       # recompensa Bernoulli del entorno
    theta = float(np.clip(theta + ETA * r * score(a, theta), LO, np.pi - LO))
print(round(float(np.sin(theta / 2) ** 2), 3))               # π(acción peor): del orden de 0.01–0.02 (piso 0.01)

# 4. Orquestación: repartir B disparos entre K=4 lecturas piloto
thetas = [0.2, 1.0, np.pi / 2, 2.9]
print(np.round(neyman_allocation(thetas, 4000)).astype(int).tolist(), round(uniform_to_neyman_ratio(thetas), 3))
# [349, 1477, 1755, 420] 1.389
```

**Interpretación.**

- Los valores de los pasos 1 y 4 son deterministas y reproducibles. El flujo predice 147,2 pasos para bajar la acción peor al $5\%$ con $\eta=0{,}05$ y $\Delta=0{,}4$; la política con $\theta=0{,}4$ lee $qg_Z=0{,}921$ (921 m-qg), $qg_S=0{,}240$ bits y necesita 2 disparos para fijar el signo a $3\sigma$. El reparto óptimo para $\theta=(0{,}2;\,1{,}0;\,\pi/2;\,2{,}9)$ da $(349;\,1477;\,1755;\,420)$ y baja la suma de varianzas por $1{,}389$ respecto del reparto uniforme.
- El paso 3 es estocástico. Con el mismo bucle en NumPy (`tools/verify_cap44.py` usa `run_bandit`, que sortea la acción con la misma probabilidad $\cos^2(\theta/2)$), 300 corridas de 400 pasos dan $\pi(\text{peor})$ medio de $0{,}020$ y mediana $0{,}010$ (el piso). Una corrida suelta con Qiskit varía; el valor impreso es un orden de magnitud, no una garantía.
- Qiskit no estaba instalado en el entorno donde se redactó el capítulo: el bloque no se ejecutó, y su salida del paso 3 queda por comprobar en el entorno del lector. Las identidades que lo respaldan se verifican con muestreo de NumPy.
- Concordancia con el flujo (verificada): con $\eta=0{,}02$ y 100 pasos, 400 corridas de Bernoulli dan $\pi(\text{peor})=0{,}316$ frente a $0{,}310$ predicho por el flujo cerrado.

## 44.5 Casos de uso, limitaciones (NISQ frente a simuladores) y líneas futuras

**Casos de uso con respaldo.**

| decisión del agente | cómo ayuda `qang.qagent` | respaldo |
|---|---|---|
| cuántos pasos predice el aprendizaje antes de ejecutarlo | tiempo cerrado del flujo en $\theta$ | `time_to_wrong_probability`, `flow_theta` |
| si la política ya decide su acción | $qg_Z$, $qg_S$, $qg_{SNR}$, disparos para el signo | `decision_report` |
| cuántos episodios cuesta el signo del gradiente (y si hay polo) | $z^2\mathrm{Var}(g)/\mathbb E[g]^2$ | `episodes_for_gradient_sign` |
| cómo evitar el polo malo | piso de exploración $\theta_{\min}$ | `exploration_floor`, `run_bandit` |
| cómo repartir los disparos entre herramientas cuánticas | $n_i\propto\sin\theta_i$ | `neyman_allocation`, `uniform_to_neyman_ratio` |

**Limitaciones.** (i) *Latencia*: una acción es un disparo, y el tiempo por disparo más la cola de un servicio en la nube hacen inviable un lazo interactivo de cientos de pasos en un QPU; en simulador no hay problema, pero tampoco hay ventaja. (ii) *Equivalencia clásica*: la política de un qubit es una moneda sesgada; el beneficio de la parametrización en $\theta$ es una elección de métrica que un gradiente natural clásico reproduce. (iii) *Varios qubits*: las ventajas de la literatura exigen entrelazamiento o acceso cuántico al entorno; qang no tiene un resultado verificado ahí y `THEORY_CORE.md` (§2) recuerda que leer $qg_Z$ por qubit pierde frente al histograma de cadenas (Cap. 3). (iv) *Ruido*: el modelo es ideal; con ruido de lectura hay que calibrar la lectura afín (Cap. 30) antes de usar $qg_Z$ como política.

**Líneas futuras, sin resultado.** *Orquestación por un modelo de lenguaje*: un agente de tipo ReAct [@yao2023react] podría llamar a las funciones de `qang.qagent` como herramientas; no hay experimento en este libro, `unverified`. *Políticas multiqubit y Q-learning cuántico* [@skolik2022]: la métrica de qang para varios qubits (Cap. 5, radio local) es un punto de partida, `unverified`. *Gradiente natural con piso*: probar con una prueba pre-registrada si el piso de exploración $p_{\min}$ óptimo depende de $\Delta$.

## 44.6 Sin qang y con qang

| acción | sin qang | con qang | diferencia |
|---|---|---|---|
| prever la velocidad de aprendizaje | correr y mirar la curva | $T_\theta=(\ln\frac{1-\pi_\ast}{\pi_\ast}-\phi_0)/(\eta\Delta)$ | **cálculo previo**; sin ganancia en el aprendizaje |
| elegir la parametrización | logit con gradiente plano ($\propto1/\pi_\ast$) | ángulo $\theta$ ($\propto\ln(1/\pi_\ast)$) | **elección de métrica**: el gradiente natural clásico da lo mismo |
| detectar el polo | el agente deja de aprender | gradiente nulo, $1/\sin^2\theta$ episodios, piso $\theta_{\min}$ | **diagnóstico** exacto |
| saber si la política ya decide | muestreo hasta que se vea | $qg_Z$, $qg_{SNR}$, disparos para el signo | **chequeo** previo |
| repartir disparos entre $K$ herramientas | uniforme | $n_i\propto\sin\theta_i$ | **hasta $K\sum\sin^2/(\sum\sin)^2$** menos varianza (1,389 en el ejemplo) |

**Balance.** qang no hace más inteligente al agente: le da cuentas exactas de su política (cuánto confía, cuánto tarda en aprender, dónde se atasca, cómo gastar los disparos). La política de un qubit es una moneda sesgada, y la ventaja cuántica de aprendizaje queda fuera de lo que este libro verifica. No hay `measured gain`.

## 44.7 Verificación — `tools/verify_cap44.py`

```bash
python tools/verify_cap44.py          # solo NumPy + qang; ~1 min
pytest tests/test_qagent.py -q        # 4 pruebas, ~1 s
```

Cinco bloques: [1] gradiente, varianza e información de Fisher con $4\cdot10^5$ muestras en cuatro ángulos; [2] flujo cerrado frente a integración de Euler, $\phi_t=\phi_0+\eta\Delta t$, tiempos $4{,}595$ y $108{,}2$ (cociente $23{,}5$; a $0{,}1\%$, $147\times$) y REINFORCE de Bernoulli frente al flujo ($0{,}316$ frente a $0{,}310$); [3] episodios para el signo ($15$ en el ecuador, $1542$ a $\theta=0{,}05$), polo inmóvil, y escape del polo malo con piso en $200$ de $200$ corridas (mediana de $227$ pasos); [4] lectura de decisión frente a `qang.probability`; [5] reparto de disparos ($1{,}389$ calculado, $1{,}376$ simulado).

## 44.8 Apéndice H

| id | familia | fórmula | función | estatus |
|---|---|---|---|---|
| `born_policy_gradient` | agentes | $\partial_\theta J=-\Delta\sin\theta/2$; $\mathbb E[g^2]=r_0^2\sin^2\frac\theta2+r_1^2\cos^2\frac\theta2$ | `qang.qagent.expected_gradient`, `qang.qagent.gradient_variance` | `exact` |
| `policy_fisher_one` | agentes | $I(\theta)=1$ para todo $\theta$ | `qang.qagent.score` | `exact` |
| `policy_flow_closed` | agentes | $\tan\frac{\theta_t}2=\tan\frac{\theta_0}2e^{-\eta\Delta t/2}$ | `qang.qagent.flow_theta` | `exact` |
| `theta_is_natural_logit` | agentes | gradiente plano en $\theta$ ⇔ natural en el logit, $\phi_t=\phi_0+\eta\Delta t$ | `qang.qagent.logit_of`, `qang.qagent.time_to_wrong_probability` | `exact` |
| `logit_vanilla_slowdown` | agentes | $T\propto1/\pi_\ast$ (logit plano) frente a $\ln(1/\pi_\ast)$ | `qang.qagent.logit_vanilla_time` | `exact` |
| `pole_gradient_trap` | agentes | episodios $\propto1/\sin^2\theta$; gradiente nulo en el polo | `qang.qagent.episodes_for_gradient_sign` | `exact` |
| `exploration_floor` | agentes | $\theta_{\min}=2\arcsin\sqrt{p_{\min}}$ | `qang.qagent.exploration_floor`, `qang.qagent.run_bandit` | `exact` |
| `decision_reading` | agentes | $qg_Z$, $qg_S$, $qg_{SNR}$, $(z/qg_{SNR})^2$ | `qang.qagent.decision_report` | `exact` |
| `neyman_shot_allocation` | agentes | $n_i\propto\sin\theta_i$; cociente $K\sum\sin^2/(\sum\sin)^2$ | `qang.qagent.neyman_allocation`, `qang.qagent.uniform_to_neyman_ratio` | `exact` |
| `llm_tool_orchestration` | agentes | un agente de tipo ReAct que llama a `qang.qagent` como herramienta | — | `unverified` |
| `multiqubit_quantum_policy` | agentes | política multiqubit entrelazada con métrica de `qang` | — | `unverified` |
| `agent_learning_advantage` | agentes | ventaja de aprendizaje atribuida a qang | — | `not adopted` |

## 44.9 Referencias bibliográficas clave

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning* 8(3–4), 229–256. [@williams1992]
- Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2.ª ed. MIT Press. [@sutton2018]
- Jerbi, S., Gyurik, C., Marshall, S. C., Briegel, H. J., Dunjko, V. (2021). Parametrized quantum policies for reinforcement learning. *NeurIPS 34*; arXiv:2103.05577. [@jerbi2021]
- Skolik, A., Jerbi, S., Dunjko, V. (2022). Quantum agents in the Gym: a variational quantum algorithm for deep Q-learning. *Quantum* 6, 720. [@skolik2022]
- Dunjko, V., Taylor, J. M., Briegel, H. J. (2016). Quantum-enhanced machine learning. *Phys. Rev. Lett.* 117, 130501. [@dunjko2016]
- Yao, S. et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*; arXiv:2210.03629. [@yao2023react]

## 44.10 CHANGELOG y commit

```markdown
### Added
- `qang.qagent` (Cap. 44): a two-action agent whose policy is one qubit measurement — Born policy, exact gradient and Fisher information 1, closed-form gradient flow (angle = natural gradient of the logit), pole trap and exploration floor, decision reading, Neyman shot allocation across tools; 4 tests in `tests/test_qagent.py`; `tools/verify_cap44.py`; chapter `book/cap44_ia_agentica_cuantica.md` -> `book/es/qjax44.qmd`. New bib keys: jerbi2021, skolik2022, yao2023react, dunjko2016, williams1992, sutton2018.
```

```text
feat(book): add Cap. 44 (quantum agentic AI with qang) with qang.qagent
```
