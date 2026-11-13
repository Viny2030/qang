# THEORY_CORE.md — Núcleo Matemático, Evidencia y Límites (`qang` v0.14.0+)

## 1. Definiciones Canónicas y Escalas
- **Sesgo Polar (`qg_Z`)**: $qg_Z(\theta) = \langle\sigma_z\rangle = p_0 - p_1 = 2p_0 - 1 = \cos\theta \in [-1, 1]$, con $p_0 = \cos^2(\theta/2) = (1 + qg_Z)/2$.
- **Dispersión de Superposición (`qg_S`)**: $qg_S(\theta) = H(p_0) = H((1 + qg_Z)/2) \in [0, 1]$ (entropía de Shannon en bits; cóncava, sin representante de Riesz-Fréchet).
- **Distinción obligatoria `qga` vs `m-qg` (`qang.qga` vs `qang.probability`)**:
  - `qga` (**escala angular**): $1\text{ vuelta} = 2\pi\text{ rad} = 1000\text{ qga} = 10^6\text{ m-qga}$. Polo $|0\rangle = 0\text{ qga}$, ecuador $= 250\text{ qga}$ ($p_0 = 0.5$), polo $|1\rangle = 500\text{ qga}$. Curvatura métrica de Fubini-Study $= 4$; información de Fisher $I_F = 4\pi^2\cdot 10^{-6}\text{ qga}^{-2}$ ($\sigma_{\text{qga}}(N) = 159.15/\sqrt{N}\text{ qga}$ uniforme).
  - `m-qg` (**lectura proyectiva**): $m = 1000 \cdot qg_Z = 2000 p_0 - 1000 \in [-1000, +1000]$. Lineal en $p_0$ ($1\text{ m-qg} = \Delta p_0 = 0.0005$), no lineal en ángulo ($1\text{ mrad}$ en ecuador; $2.56^\circ$ cerca del polo).
- **Magnitudes Derivadas (Cap. 1–3)**:
  - **Q-SNR (`quantum_snr`)**: $qg_{\text{SNR}} = |qg_Z|/\sqrt{1 - qg_Z^2} = |\cot\theta|$. Disparos para confirmar signo a $z\sigma$: $N_{\text{signo}} = (z/qg_{\text{SNR}})^2$.
  - **Curvatura de información (`information_curvature`)**: $qg_\kappa = -\partial^2 qg_S/\partial qg_Z^2 = 1/(\ln 2 \cdot (1 - qg_Z^2)) = I_F(qg_Z)/\ln 2$ (en bits). **No confundir** con la curvatura de pulso $\kappa_{qg} = |\partial^2 m/\partial\theta^2| = 1000|\cos\theta|\text{ m-qg/rad}^2$.
  - **Distancia geodésica (`geodesic_distance`)**: $qg_L = \arccos(\sqrt{p_0 p_0'} + \sqrt{p_1 p_1'}) = |\theta - \theta'|/2 = h_{\text{Cohen}}/2$. Coincide con Fubini-Study solo en un meridiano común.
  - **Déficit disipativo (`dissipative_deficit`)**: $qg_\tau = \sqrt{1 - r^2} = 2\sqrt{\det\rho}$, con radio local $r^2 = qg_X^2 + qg_Y^2 + qg_Z^2 = 2\text{Tr}(\rho^2) - 1$. Requiere medir 3 bases; en peso de Hamming definido $r_q = |qg_Z^{(q)}|$.
- **Compuertas y Área de Pulso (`qang.algebra`, `qang.pulse_area`)**:
  - $U_3^{qg}(q, \varphi, \lambda)$ es algebraica en $q = qg_Z$; su matriz de transición $T(q)$ es doblemente estocástica con capacidad $C = 1 - qg_S$.
  - Área mínima: $R_X(\theta)$ tiene período $2\pi$ ($1000\text{ qga}$); $R_{ZZ}(\theta)$ tiene período $\pi$ ($500\text{ qga}$) vía $R_Z(\pi)$ virtuales. Error aleatorio de amplitud amortigua como $\cos\theta\,e^{-\theta^2\sigma^2/2}$.
- **Estadística (`qang.statistics`)**:
  - Previa de Haar es uniforme en $qg_Z \in [-1, 1]$ ($p_0 \sim \text{Beta}(1,1)$). Usar intervalos Wilson o Bayes-Haar cerca de polos (el método delta colapsa a cobertura $<10\%$).
  - Cuadrados insesgados obligatorios para $r^2$: $\widehat{qg^2} = (N\hat{qg}^2 - 1)/(N - 1)$.

## 2. Hallazgos Negativos, Equivalencias y Prohibiciones (Cap. 30 y §1–§170)
**NO proponer, repetir ni afirmar ventaja en los siguientes casos:**
1. **Optimización / Gradiente Natural en `qg_Z`**: Descenso plano en $qg_Z$ diverge en los polos (0/20 convergencias). El paso amortiguado por $qg_\kappa$ (`curvature_damped_update`) y el gradiente natural en $qg_Z$ equivalen a primer orden al descenso euclidiano en $\theta$. Cambiar unidades a `qga` o `m-qga` solo reescala derivadas por $2\pi/1000$ y **no** mitiga *barren plateaus*.
2. **Calibración Echo (`echo_transfer_matrix`, Cap. 22 / §108)**: **NO usar** cuando el error filtrado esté cerca del ruido de *shots* (márgenes estrechos); añade ruido estadístico.
3. **Búsqueda de Grover con ruido (Cap. 3.6.2 / §63)**: Leer signos/magnitudes $qg_Z$ por qubit pierde frente al histograma de cadenas de bits (66% vs 91% aciertos en $n=5$). En estimación de amplitud (MLAE), la ventaja termina cuando $k \approx 1/p$ ($qg_\tau > 0.93$).
4. **Curva de Page y Estados de Grafo (Cap. 15.8 / Cap. 30.2)**: $qg_S$ depende de la base $Z$ y **no** mide entrelazamiento (crece monótonamente en lugar de bajar en la curva de Page; requiere entropía Rényi-2 con $4^k - 1$ correladores). $qg_S$ tampoco es monótona bajo relajación $T_1$.
5. **Sombras Clásicas (`classical shadows`, Cap. 4.6 / §62)**: Necesitan $3\text{–}41\times$ más disparos que la medición directa en $Z$ para cantidades `qang` y destruyen el filtro disparo a disparo ($3^{-n}$ probabilidad de medir todo en $Z$).
6. **Criptografía Post-Cuántica (PQC, Cap. 16)**: `qang` **no aplica** a ML-KEM/ML-DSA/SLH-DSA (aritmética clásica sin qubits; $\delta \le 2^{-138.8}$ incontable).
7. **Qudits y Fuga a $|2\rangle$ (Cap. 2.5 / §70)**: La subálgebra de Cartan de $\mathfrak{su}(d)$ es una identidad exacta sin ventaja demostrada; descartar fugas a $|2\rangle$ como borrado empeora el decodificador de superficie.
8. **Entrenamiento calibrado en IBM con ruido realista (Cap. 21.6 / §119–§120)**: Entrenar sin ruido + filtro de simetría supera al entrenamiento calibrado con $T_1/T_2$ o datos del dispositivo.
9. **Propuestas explícitamente marcadas como SIN VERIFICAR**: Extensión relativista/antimateria en `qga` (Cap. 1.4.1 / Apéndice G; 3 errores de la fuente original corregidos: $p_0=0.5$ en $250\text{ qga}$, curvatura $K=4$, factor $2$ en $P_{\text{reposo}}$); energía variacional regularizada $E_{\text{total}}$; función de radiación $R(\theta,\varphi)$; compilación por $\kappa_{qg}$ contra mesetas estériles.

## 3. Restricciones Técnicas, Entorno y Hardware (`pyproject.toml`)
- **Core**: Python $\ge 3.9$, únicamente `numpy>=1.24`. Todos los SDKs (`qiskit`, `cirq`, `pennylane`, etc.) se importan bajo demanda.
- **Entornos aislados**: `qang[iqm]` (`iqm-client` fija su versión de Qiskit), `qang[dynamics]` (`qiskit<=1.3`), `jax<0.9` con `pennylane>=0.42`, `amazon-braket-sdk` fija `cloudpickle==2.2.1`, `cudaq` y `chemistry` (`pyscf`) solo Linux/macOS/WSL.
- **Hardware real ejecutado (IQM Garnet)**:
  - §165 (neutrinos 3 sabores, $8\times 300$ shots): filtro baja error $1.59\times$ ($0.062 \to 0.039$) y recupera mínimo en $2.6\text{ GeV}$.
  - §168 (milli-qang, 1 qubit, $26\times 4000$ shots): compuerta X cuesta $1.64\text{ m-qg}$ lineales; ángulos intermedios sesgados $16\text{–}55\text{ m-qg}$; predicción M2 (dispersión sub-binomial en ecuador) fallida/abierta.
  - §170 (Apéndice G, Dirac $E=mc^2$ en 1 qubit): autoestado a $0.6c$ leyó $+759.9\pm 14.9\text{ m-qg}$ (previsto $+800$), Zitterbewegung a $0.44\%$.
