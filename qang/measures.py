"""
qang.measures -- the catalogue of qang sub-measures and what each one is worth (book, appendix H).

The qang unit does not reformulate physics: every sub-measure below is a reading of a probability, or of a few, and is
equal to a quantity that standard physics or statistics already has. What a sub-measure can add is a *decision*: fewer
shots, a shorter pulse, a compilation chosen before running, a reading that noise cannot bias. This registry records,
for each one, its definition, the standard quantity it equals, the decision it improves and the evidence, with one of
these statuses:

    exacta         an exact identity; it names or re-scales something, no gain is claimed
    ganancia       tested (simulation or analysis) with a measured gain over the usual practice
    hardware       tested on a real device
    mixta          tested: a gain in some regimes, none or a loss in others
    sin_ganancia   tested: equivalent to the usual practice, no gain
    en_contra      tested: worse than the usual practice
    sin_verificar  proposed, not tested yet: a candidate for a pre-registered test
    no_adoptada    proposed and rejected, because it is wrong or has no support

``CATALOG`` is the single source of the tables of appendix H in both languages (``tools/make_apendice_h.py``).
``code`` is a dotted path inside qang, resolved by ``resolve``; it may be empty.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

__all__ = ["STATUSES", "FAMILIES", "Measure", "CATALOG", "by_status", "by_family", "find", "resolve", "summary"]

STATUSES: Dict[str, Tuple[str, str]] = {
    "exacta": ("exacta", "exact"),
    "ganancia": ("ganancia medida", "measured gain"),
    "hardware": ("probada en hardware", "tested on hardware"),
    "mixta": ("mixta", "mixed"),
    "sin_ganancia": ("sin ganancia", "no gain"),
    "en_contra": ("en contra", "against"),
    "sin_verificar": ("sin verificar", "unverified"),
    "no_adoptada": ("no adoptada", "not adopted"),
}

FAMILIES: Dict[str, Tuple[str, str]] = {
    "base": ("Lecturas base", "Base readings"),
    "derivadas": ("Magnitudes derivadas", "Derived quantities"),
    "estado": ("Estados y registros", "States and registers"),
    "compuertas": ("Compuertas y compilación", "Gates and compilation"),
    "ruido": ("Ruido y lectura", "Noise and readout"),
    "fisica": ("Lecturas de leyes físicas conocidas", "Readings of known physical laws"),
    "propuestas": ("Propuestas abiertas y descartadas", "Open and rejected proposals"),
}


@dataclass(frozen=True)
class Measure:
    """One qang sub-measure. Text fields are (Spanish, English) pairs; ``symbol`` and ``definition`` are either a
    string shared by both languages (pure LaTeX) or such a pair. ``text(field, lang)`` reads any of them."""

    key: str
    family: str
    symbol: object
    definition: object
    equals: Tuple[str, str]
    decision: Tuple[str, str]
    status: str
    evidence: Tuple[str, str]
    where: str
    code: str = ""

    def text(self, field: str, lang: str = "es") -> str:
        """The value of a text field in "es" or "en"."""
        v = getattr(self, field)
        if isinstance(v, tuple):
            return v[0 if lang == "es" else 1]
        return v


def _m(key, family, symbol, definition, equals, decision, status, evidence, where, code=""):
    return Measure(key, family, symbol, definition, tuple(equals), tuple(decision), status, tuple(evidence), where, code)


CATALOG: Tuple[Measure, ...] = (
    # ------------------------------------------------------------------ base readings
    _m("qg_z", "base", r"$qg_Z$", r"$2p_0-1=\langle\sigma_z\rangle=\cos\theta$",
       ("valor esperado de $\\sigma_z$", "expectation value of $\\sigma_z$"),
       ("coordenada de trabajo medible sin inversión", "working coordinate measured without inversion"),
       "exacta", ("identidad; vale para estados mixtos", "identity; holds for mixed states"),
       "cap. 1, §7", "qang.probability.readings_from_p0"),
    _m("milliqang", "base", "m-qg", r"$m=1000\,qg_Z$",
       ("$\\langle\\sigma_z\\rangle$ por mil, lineal en $p_0$", "$\\langle\\sigma_z\\rangle$ times a thousand, linear in $p_0$"),
       ("presupuestar error de compuerta y ruido de disparo en una sola escala",
        "budget gate error and shot noise on one scale"),
       "hardware",
       ("IQM Garnet: 1,64 m-qg por $X$, sobre-rotaciones de 16–55 m-qg ocultas al modelo de lectura; M2 falló",
        "IQM Garnet: 1.64 m-qg per $X$, over-rotations of 16–55 m-qg hidden from the readout model; M2 failed"),
       "cap. 1, §168", "qang.probability.analyze"),
    _m("theta", "base", r"$\theta$", r"$\arccos qg_Z$",
       ("transformación arcoseno de una proporción", "arcsine transformation of a proportion"),
       ("barra de error $1/\\sqrt N$ igual en todo el círculo", "error bar $1/\\sqrt N$ equal over the whole circle"),
       "exacta", ("sumas binomiales exactas; asintótica (falla con $Np\\approx2{,}5$)",
                  "exact binomial sums; asymptotic (fails at $Np\\approx2.5$)"),
       "cap. 1, §125", "qang.probability.theta_from_milliqang"),
    _m("qg_s", "base", r"$qg_S$", r"$H(p_0)=H\big((1+qg_Z)/2\big)$",
       ("entropía de Shannon del resultado", "Shannon entropy of the outcome"),
       ("certificar aleatoriedad junto con la lectura media", "certify randomness together with the mean reading"),
       "mixta", ("sola se engaña bajo T1; el par con la lectura media no", "alone it is fooled under T1; paired with the mean reading it is not"),
       "caps. 1 y 26, §7, §51", "qang.probability.readings_from_p0"),
    _m("qga", "base", "qga", (r"$1\ \text{vuelta}=1000\ \text{qga}$", r"$1\ \text{turn}=1000\ \text{qga}$"),
       ("ángulo con otra unidad", "an angle in another unit"),
       ("escribir ángulos de compuerta sin factores $\\pi$", "write gate angles without factors of $\\pi$"),
       "exacta", ("conveniencia; no cambia gradientes ni mesetas", "convenience; changes neither gradients nor plateaus"),
       "cap. 1", "qang.qga.p0_from_qga"),
    _m("branch_sign", "base", r"$(qg_Z,s)$", r"$s=\operatorname{sign}(qg_X)$",
       ("el signo del seno del ángulo", "the sign of the angle's sine"),
       ("cruzar un polo durante una optimización", "cross a pole during an optimization"),
       "ganancia", ("H$_2$: con sólo $qg_Z$ se queda en Hartree–Fock; con signo llega a la energía exacta",
                    "H$_2$: with $qg_Z$ alone it stops at Hartree–Fock; with the sign it reaches the exact energy"),
       "cap. 1, §65", "qang.qiskit_gate.SignedRQangGate"),
    # ------------------------------------------------------------------ derived quantities
    _m("qg_kappa", "derivadas", r"$qg_\kappa$", r"$1/[\ln2\,(1-qg_Z^2)]$",
       ("información de Fisher sobre $qg_Z$, en bits", "Fisher information about $qg_Z$, in bits"),
       ("amortiguar el paso de un optimizador junto a los polos", "damp an optimizer's step near the poles"),
       "sin_ganancia", ("evita la divergencia del gradiente en $q$, pero es el descenso en $\\theta$: 103 pasos los dos (cap. 30); el banco de JAX lo repite",
                        "removes the divergence of the gradient in $q$, but it is descent on $\\theta$: 103 steps both (ch. 30); the JAX bench repeats it"),
       "caps. 1, 23 y 30", "qang.probability.information_curvature"),
    _m("qg_snr", "derivadas", r"$qg_{SNR}$", r"$\lvert qg_Z\rvert/\sqrt{1-qg_Z^2}$",
       ("señal a ruido de un disparo, $\\lvert\\cot\\theta\\rvert$", "signal-to-noise of one shot, $\\lvert\\cot\\theta\\rvert$"),
       ("disparos para leer un signo: $(z/qg_{SNR})^2$", "shots to read a sign: $(z/qg_{SNR})^2$"),
       "exacta", ("27 disparos a 60° para 3σ; binomial exacta: error < 1 %", "27 shots at 60° for 3σ; exact binomial: error < 1 %"),
       "caps. 1 y 3", "qang.probability.shots_for_sign"),
    _m("qg_l", "derivadas", r"$qg_L$", r"$\lvert\theta-\theta'\rvert/2$",
       ("ángulo de Bhattacharyya; Fubini–Study en un meridiano común", "Bhattacharyya angle; Fubini–Study on a common meridian"),
       ("distancia entre estados desde dos histogramas en $Z$", "distance between states from two $Z$ histograms"),
       "exacta", ("con azimutes distintos es sólo una cota inferior", "with different azimuths it is only a lower bound"),
       "caps. 1 y 3", "qang.algebra.geodesic_distance"),
    _m("qg_tau", "derivadas", r"$qg_\tau$", r"$\sqrt{1-r^2}=2\sqrt{\det\rho}$",
       ("falta de pureza", "lack of purity"),
       ("ver el área de un pulso donde la lectura en m-qg no se corre", "see a pulse's area where the m-qg reading does not shift"),
       "exacta", ("0,277 para $R_X(324^\\circ)$ frente a 0,031 para $R_X(-36^\\circ)$; necesita tres bases",
                  "0.277 for $R_X(324^\\circ)$ against 0.031 for $R_X(-36^\\circ)$; needs three bases"),
       "caps. 1–3", "qang.probability.dissipative_deficit"),
    _m("kappa_qg", "derivadas", r"$\kappa_{qg}$", r"$1000\,\lvert\cos\theta\rvert$",
       ("curvatura de la lectura respecto del ángulo", "curvature of the reading with respect to the angle"),
       ("predecir el corrimiento por error de amplitud, $\\tfrac12\\theta^2\\sigma^2\\kappa_{qg}$",
        "predict the shift from amplitude error, $\\tfrac12\\theta^2\\sigma^2\\kappa_{qg}$"),
       "exacta", ("QuTiP: coincide a segundo orden", "QuTiP: agrees to second order"),
       "caps. 1 y 2, §160", "qang.probability.pulse_error_milliqang"),
    _m("q_fisher", "derivadas", "qFi", r"$4\pi^2\cdot10^{-6}\ \text{qga}^{-2}$",
       ("información de Fisher del ángulo qang", "Fisher information of the qang angle"),
       ("resolución angular fija: $159{,}15/\\sqrt N$ qga", "fixed angular resolution: $159.15/\\sqrt N$ qga"),
       "exacta", ("no es una variedad plana: la esfera tiene curvatura 4", "not a flat manifold: the sphere has curvature 4"),
       "cap. 1", "qang.qga.angle_sigma_qga"),
    _m("q_hertz", "derivadas", "qHz", r"$\text{qga}/\text{s}$",
       ("frecuencia angular, $2\\pi/1000$ rad/s", "angular frequency, $2\\pi/1000$ rad/s"),
       ("ninguna por ahora", "none so far"),
       "exacta", ("unidad sin uso en el libro", "unit with no use in the book"), "cap. 1"),
    # ------------------------------------------------------------------ states and registers
    _m("purity_one_basis", "estado", r"$1-r^2$", (r"$4p_0(1-p_0)$ en peso definido", r"$4p_0(1-p_0)$ at definite weight"),
       ("pureza local", "local purity"),
       ("medir la pureza con una base en lugar de tres", "measure purity with one basis instead of three"),
       "ganancia", ("vale sólo con peso de Hamming definido", "holds only with a definite Hamming weight"),
       "cap. 5", "qang.mixed.purity"),
    _m("direction_noisy", "estado", r"$\arccos(qg_Z/r)$", r"$qg_Z/r$",
       ("ángulo polar de Bloch de un estado mixto", "Bloch polar angle of a mixed state"),
       ("dirección de un estado con ruido", "direction of a noisy state"),
       "mixta", ("a favor con despolarizante; en contra con desfase puro", "in favour with depolarizing noise; against with pure dephasing"),
       "cap. 5, §45"),
    _m("qg_correlators", "estado", r"$qg_{XX},qg_{YY},qg_{ZZ}$", r"$\langle\sigma_i\otimes\sigma_i\rangle$",
       ("correladores de Pauli", "Pauli correlators"),
       ("diagnosticar un par de Bell y predecir la teleportación sin teleportar",
        "diagnose a Bell pair and predict teleportation without teleporting"),
       "ganancia", ("3 configuraciones en vez de 9; 4–5 compuertas de dos qubits en vez de 10–11, error máximo 0,029",
                    "3 settings instead of 9; 4–5 two-qubit gates instead of 10–11, maximum error 0.029"),
       "caps. 12 y 13"),
    _m("eve_length", "estado", r"$m_E$", r"$\sqrt{1000^2-m_\perp^2}$",
       ("información de Eve (Helstrom con purificación)", "Eve's information (Helstrom with purification)"),
       ("cuántas rondas de prueba hacen falta para certificar bits", "how many test rounds certifying bits needs"),
       "exacta", ("la cuenta es estándar; la lectura la vuelve geométrica", "the computation is standard; the reading makes it geometric"),
       "cap. 18"),
    _m("lost_radius", "estado", r"$(1000-m_r)/2000$", r"$m_r=1000\,r$",
       ("energía del estado pasivo", "energy of the passive state"),
       ("saber cuánta energía nadie puede extraer", "know how much energy nobody can extract"),
       "exacta", ("modesto: el reparto se lee de un vistazo", "modest: the split is read at a glance"), "cap. 19"),
    _m("code_witness", "estado", (r"$qg_Z,qg_X$ de síndromes", r"syndrome $qg_Z,qg_X$"), ("lecturas de los ancillas", "ancilla readings"),
       ("la regla $T_2/T_1$", "the $T_2/T_1$ rule"),
       ("elegir y recalibrar un código corrector sin disparos de prueba", "choose and recalibrate an error-correcting code without test shots"),
       "ganancia", ("6,4 % sobre el oráculo frente a 64–220 % de un código fijo; deriva 13,1 % frente a 20,4 %",
                    "6.4 % over the oracle against 64–220 % for a fixed code; drift 13.1 % against 20.4 %"),
       "cap. 26"),
    _m("cut_gamma", "estado", r"$\gamma$", r"$1+2\sqrt{1-qg^2}$",
       ("sobrecosto de cortar una compuerta", "overhead of cutting a gate"),
       ("elegir qué compuertas cortar", "choose which gates to cut"),
       "mixta", ("hasta 2,9 veces frente a reglas ciegas al ángulo; ninguna frente a qiskit-addon-cutting",
                 "up to 2.9 times against angle-blind rules; none against qiskit-addon-cutting"),
       "cap. 26"),
    _m("graph_states", "estado", (r"$qg_Z$ en estados de grafo", r"$qg_Z$ on graph states"), r"$qg_Z=0,\ qg_S=1$",
       ("la lectura local en $Z$", "the local $Z$ reading"),
       ("detectar entrelazamiento con lecturas locales en $Z$", "detect entanglement with local $Z$ readings"),
       "sin_ganancia", ("no distingue un estado de grafo de un producto: el límite es de la base $Z$",
                        "cannot tell a graph state from a product: the limit belongs to the $Z$ basis"),
       "cap. 30"),
    _m("ghz_parity", "estado", (r"$qg$ de paridad", r"parity $qg$"), r"$(dqg)^2/(1-qg^2)=N^2$",
       ("información de Fisher cuántica de un GHZ", "quantum Fisher information of a GHZ state"),
       ("alcanzar la cota de Heisenberg con una lectura de paridad", "reach the Heisenberg bound with a parity reading"),
       "exacta", ("la fórmula de un disparo del capítulo 1 alcanza la cota", "the one-shot formula of Chapter 1 reaches the bound"),
       "cap. 34"),
    # ------------------------------------------------------------------ gates and compilation
    _m("min_area", "compuertas", ("área mínima", "minimum area"), (r"$\theta\bmod2\pi$; $R_{ZZ}$ de período $\pi$", r"$\theta\bmod2\pi$; $R_{ZZ}$ with period $\pi$"),
       ("representante más corto de la misma compuerta", "shortest representative of the same gate"),
       ("compilar con menos área de pulso: menos error", "compile with less pulse area: less error"),
       "ganancia", ("QAOA de 4 qubits: área 4,4× y error 4,1× menores; la predicción de 1,5× por $R_{ZZ}$ falló (1,14×)",
                    "4-qubit QAOA: area 4.4× and error 4.1× lower; the 1.5× prediction for $R_{ZZ}$ failed (1.14×)"),
       "cap. 2, §160", "qang.pulse_area.minimal_area_circuit"),
    _m("exposure", "compuertas", ("exposición", "exposure"), ("población fuera del sector tras cada compuerta de dos qubits", "population outside the sector after each two-qubit gate"),
       ("cálculo sin ruido del circuito", "noiseless computation on the circuit"),
       ("elegir la compilación antes de gastar disparos", "choose the compilation before spending shots"),
       "ganancia", ("correlación de rangos 0,80 con la fuga de T1; exposición nula quita ≥ 98 % del T1",
                    "rank correlation 0.80 with T1 leakage; zero exposure removes ≥ 98 % of T1"),
       "cap. 2, §68", "qang.sectors.sector_exposure"),
    _m("n_conserving", "compuertas", ("compuertas que conservan $N$", "$N$-conserving gates"), r"$[U,\sum_i Z_i]=0$",
       ("compilación en el sector", "compilation within the sector"),
       ("compilar para que el filtro quite el T1", "compile so that the filter removes T1"),
       "mixta", ("el filtro quita 99,5 % del T1 frente a 48,5 % con CNOT; ninguna en iones",
                 "the filter removes 99.5 % of T1 against 48.5 % with CNOT; none on ions"),
       "cap. 24, §61"),
    _m("qg_grid", "compuertas", ("grilla en qg", "qg grid"), (r"pasos iguales en $qg_Z$", r"equal steps in $qg_Z$"),
       ("cuantización del control", "control quantization"),
       ("cuantizar los ángulos de control", "quantize the control angles"),
       "mixta", ("probabilidades $\\pi/2$ veces mejores, 43 de 45 cargas; en contra para fidelidad junto a los polos",
                 "probabilities $\\pi/2$ times better, 43 of 45 loads; against for fidelity near the poles"),
       "cap. 27"),
    _m("best_chain", "compuertas", "best_chain", (r"cadena de qubits de mayor $K$ predicho", r"qubit chain of highest predicted $K$"),
       ("elección de disposición con datos de calibración", "layout choice from calibration data"),
       ("elegir los qubits físicos antes de correr", "choose the physical qubits before running"),
       "ganancia", ("entre 6 y 17 puntos más de disparos conservados que la disposición por defecto (simulado)",
                    "6 to 17 points more shots kept than the default layout (simulated)"),
       "cap. 32", "qang.ibm.best_chain"),
    # ------------------------------------------------------------------ noise and readout
    _m("weight_filter", "ruido", (r"filtro de peso, $K$", r"weight filter, $K$"), r"$K=(1-\gamma)^{kd}$",
       ("post-selección en el sector de Hamming", "post-selection on the Hamming sector"),
       ("leer, entrenar y derivar bajo T1 sin sesgo; costo en disparos conocido antes de correr",
        "read, train and differentiate under T1 without bias; shot cost known before running"),
       "hardware", ("IQM Resonance: error 0,0615 → 0,0386 (1,59×) con 88 % de los disparos; simulado: exacto a $10^{-12}$ m-qg; "
                    "en contra con $K\\le0{,}14$ (2 de 12) y en LiH profundo (76,6 → 85,0 mHa)",
                    "IQM Resonance: error 0.0615 → 0.0386 (1.59×) with 88 % of the shots; simulated: exact to $10^{-12}$ m-qg; "
                    "against with $K\\le0.14$ (2 of 12) and on deep LiH (76.6 → 85.0 mHa)"),
       "caps. 20, 21, 23, 30 y 31, §112", "qang.sectors.filter_distribution"),
    _m("echo_unmix", "ruido", (r"eco, $M^{1/2}$", r"echo, $M^{1/2}$"), ("matriz de transferencia del eco", "echo transfer matrix"),
       ("corrección de errores dentro del sector", "correction of errors within the sector"),
       ("quitar lo que el filtro no ve", "remove what the filter cannot see"),
       "ganancia", ("−27 a −62 % del error filtrado, con $\\binom nk$ circuitos de eco más",
                    "−27 to −62 % of the filtered error, with $\\binom nk$ more echo circuits"),
       "cap. 22, §106", "qang.sectors.unmix_sector"),
    _m("heralded_readout", "ruido", (r"$qg_{eq}$, $T_\text{ef}$", r"$qg_{eq}$, $T_\text{eff}$"), ("lectura afín con heraldo", "heralded affine readout"),
       ("calibración de lectura con mediciones repetidas", "readout calibration with repeated measurements"),
       ("calibrar la lectura sin confundir temperatura con error", "calibrate readout without mistaking temperature for error"),
       "ganancia", ("temperatura efectiva a 1 mK; evita un certificador inseguro y recupera 10–50 % de bits",
                    "effective temperature to 1 mK; avoids an unsafe certifier and recovers 10–50 % of bits"),
       "caps. 18 y 27, §31"),
    _m("pole_damped", "ruido", (r"paso $\times\lvert\sin\theta\rvert$", r"step $\times\lvert\sin\theta\rvert$"), r"$\theta-\eta\max(\lvert\sin\theta\rvert,\epsilon)\,g$",
       ("región de confianza junto a los polos", "trust region near the poles"),
       ("entrenar con tasas agresivas sin ajuste", "train with aggressive rates without tuning"),
       "ganancia", ("2,3 a 13 veces más éxitos desde tasa 2; poco o nada con tasas seguras",
                    "2.3 to 13 times more successes from rate 2; little or nothing with safe rates"),
       "cap. 23", "qang.integrations.jax.pole_damped_update"),
    _m("local_cost", "ruido", ("costo local", "local cost"), r"$(1-\overline{qg_Z})/2$",
       ("costo local de Cerezo et al.", "local cost of Cerezo et al."),
       ("evitar la meseta estéril de un costo global", "avoid the barren plateau of a global cost"),
       "mixta", ("470 veces más varianza de gradiente con 10 qubits y circuitos cortos; ninguna con circuitos profundos",
                 "470 times more gradient variance with 10 qubits and short circuits; none with deep circuits"),
       "cap. 23"),
    _m("filter_zne", "ruido", (r"ZNE de la lectura filtrada", r"ZNE of the filtered reading"), (r"extrapolación a ruido cero sobre $qg_Z$ filtrado", r"zero-noise extrapolation on the filtered $qg_Z$"),
       ("post-selección y ZNE por separado (mitiq)", "post-selection and ZNE separately (mitiq)"),
       ("mitigar T1 y desfase juntos", "mitigate T1 and dephasing together"),
       "ganancia", ("55,9 → 1,1 m-qg con T1 y desfase; con pocos disparos la ZNE no paga",
                    "55.9 → 1.1 m-qg with T1 and dephasing; with few shots ZNE does not pay"),
       "caps. 21 y 35", "qang.integrations.mitiq.execute_with_filter_zne"),
    _m("t1_diagnosis", "ruido", (r"par $(qg_S,\overline{qg_Z})$", r"pair $(qg_S,\overline{qg_Z})$"), (r"entropía y lectura media del registro", r"entropy and mean reading of the register"),
       ("XEB o HOP con simulación ideal", "XEB or HOP with an ideal simulation"),
       ("distinguir T1 de ruido unital sólo con conteos", "tell T1 from unital noise from counts alone"),
       "mixta", ("0,88–0,91 frente a 0,70–0,76 con ruido fuerte; azar con ruido realista",
                 "0.88–0.91 against 0.70–0.76 with strong noise; chance with realistic noise"),
       "caps. 27, 29 y 30"),
    # ------------------------------------------------------------------ readings of known physical laws
    _m("dirac_beta", "fisica", r"$\langle\beta\rangle$", r"$mc^2/E$",
       ("valor esperado de la matriz de Dirac $\\beta$", "expectation value of the Dirac matrix $\\beta$"),
       ("leer $E=mc^2$ y la conjugación de carga como un $qg_Z$", "read $E=mc^2$ and charge conjugation as a $qg_Z$"),
       "hardware", ("exacta y conocida; la conjugación de carga invierte la lectura; corrida §170 en IQM Garnet",
                    "exact and known; charge conjugation flips the reading; run §170 on IQM Garnet"),
       "ap. G, §170"),
    _m("clock_qg2", "fisica", r"$\widehat{qg^2}$", ("cuadrados insesgados", "unbiased squares"),
       ("corrección estándar de sesgo", "standard bias correction"),
       ("comparar dos relojes con pocos disparos", "compare two clocks with few shots"),
       "ganancia", ("sesgo de +11,8 % a +0,9 % con 100 disparos", "bias from +11.8 % to +0.9 % with 100 shots"),
       "cap. 8"),
    _m("polarization", "fisica", (r"$\sqrt{S_1^2+S_2^2+S_3^2}$ insesgado", r"unbiased $\sqrt{S_1^2+S_2^2+S_3^2}$"), ("cuadrados insesgados de Stokes", "unbiased Stokes squares"),
       ("grado de polarización", "degree of polarization"),
       ("estimar la polarización con pocos fotones", "estimate polarization with few photons"),
       "ganancia", ("el sesgo baja unas tres veces, sin desaparecer", "the bias drops about three times, without vanishing"),
       "cap. 6"),
    _m("angular_momentum", "fisica", r"$\arccos(m/l)$", r"$\langle J_z\rangle/l$",
       ("dirección de $\\langle\\mathbf J\\rangle$", "direction of $\\langle\\mathbf J\\rangle$"),
       ("leer el momento angular como un $qg_Z$", "read angular momentum as a $qg_Z$"),
       "exacta", ("el modelo vectorial $\\arccos(m/\\sqrt{l(l+1)})$ dibuja otra cosa", "the vector model $\\arccos(m/\\sqrt{l(l+1)})$ draws something else"),
       "cap. 11"),
    _m("neutrino_qg", "fisica", (r"$qg_Z$ de sabor", r"flavour $qg_Z$"), r"$1-2P_{\alpha\to\beta}$",
       ("probabilidad de oscilación", "oscillation probability"),
       ("leer el ángulo de mezcla con barra de error fija", "read the mixing angle with a fixed error bar"),
       "exacta", ("identidad sobre la fórmula de dos sabores", "identity on the two-flavour formula"),
       "cap. 9", "qang.neutrino.two_flavour_qg_z"),
    _m("ctc_reading", "fisica", r"$m_Z$, $m_X$", ("lecturas del qubit cronológico", "readings of the chronology qubit"),
       ("comparar matrices densidad", "comparing density matrices"),
       ("distinguir P-CTC de D-CTC con una lectura en $Z$", "tell P-CTC from D-CTC with one $Z$ reading"),
       "ganancia", ("P-CTC mueve $m_Z$, D-CTC no", "P-CTC moves $m_Z$, D-CTC does not"),
       "cap. 14", "qang.postselection.pctc_from_counts"),
    _m("yk_law", "fisica", r"$(1000+m_P)(1000+m_F)$", r"$=10^6$",
       ("ley de Yoshida y Kitaev", "Yoshida–Kitaev law"),
       ("juzgar un decodificador de agujero de gusano", "judge a wormhole decoder"),
       "exacta", ("la ley es conocida; la escala muestra que $m_P=-500$ no prueba nada",
                  "the law is known; the scale shows that $m_P=-500$ proves nothing"),
       "cap. 15", "qang.postselection.yk_from_counts"),
    _m("landauer", "fisica", r"$N/2+\sum_n qg_n/2$", r"$G/G_0=\sum_n T_n$",
       ("fórmula de Landauer", "Landauer formula"),
       ("leer conductancia y ruido de partición en una escala", "read conductance and partition noise on one scale"),
       "exacta", ("conductancia = $qg$ medio, ruido = $qg^2$ medio; la barra de $\\hat\\theta$ es constante (0,985–1,009)",
                  "conductance = mean $qg$, noise = mean $qg^2$; the bar of $\\hat\\theta$ is constant (0.985–1.009)"),
       "cap. 33"),
    _m("thermometry", "fisica", (r"recta en artanh", r"line in artanh"), r"$\operatorname{artanh}qg$",
       ("ajuste de Fermi", "Fermi fit"),
       ("medir temperatura electrónica", "measure electron temperature"),
       "en_contra", ("sesgo de +2,9 % (ajuste de Fermi: −0,14 %)", "bias of +2.9 % (Fermi fit: −0.14 %)"),
       "cap. 33"),
    _m("bogoliubov", "fisica", r"$\theta/2=\operatorname{gd}(r)$", (r"función de Gudermann de la compresión $r$", r"Gudermannian of the squeezing $r$"),
       ("$\\tanh^2r$, $\\bar n$ y la entropía", "$\\tanh^2r$, $\\bar n$ and the entropy"),
       ("leer la compresión como un ángulo qang", "read squeezing as a qang angle"),
       "exacta", ("identidad nueva, exacta", "a new identity, exact"), "cap. 34"),
    _m("aharonov_bohm", "fisica", r"$\theta=2\pi\Phi/\Phi_0$", (r"franjas de interferencia", r"interference fringes"),
       ("fase de Aharonov–Bohm", "Aharonov–Bohm phase"),
       ("estimar el flujo con barra de error pareja", "estimate the flux with an even error bar"),
       "exacta", ("desvío $1/2\\pi\\sqrt N$ constante; el signo pide otra lectura", "deviation $1/2\\pi\\sqrt N$ constant; the sign needs another reading"),
       "cap. 34"),
    # ------------------------------------------------------------------ open and rejected proposals
    _m("kappa_compile", "propuestas", (r"compilación por $\kappa_{qg}$", r"$\kappa_{qg}$ compilation"), ("mínima curvatura de llegada", "minimum arrival curvature"),
       ("—", "—"), ("mitigar mesetas estériles", "mitigate barren plateaus"),
       "sin_verificar", ("sin prueba", "untested"), "caps. 1 y 2"),
    _m("regularized_energy", "propuestas", r"$E_\text{total}$", r"$\langle H\rangle+\lambda_\theta L(\theta)+\lambda_\varphi L(\varphi)$",
       ("—", "—"), ("regularizar un VQE", "regularize a VQE"),
       "sin_verificar", ("$L$ y $\\lambda$ sin definir", "$L$ and $\\lambda$ undefined"), "cap. 1"),
    _m("radiation", "propuestas", r"$R(\theta,\varphi)$", r"$N[p_1+\alpha\cos\varphi]$",
       ("—", "—"), ("modelar emisión desde la población excitada", "model emission from the excited population"),
       "sin_verificar", ("$N$ y $\\alpha$ sin fijar; sin contraste con un modelo de emisión",
                         "$N$ and $\\alpha$ not fixed; not compared with an emission model"), "cap. 1"),
    _m("cpt_shift", "propuestas", "500 qga CPT", (r"$E_-\leftrightarrow$ desplazamiento de 500 qga", r"$E_-\leftrightarrow$ a 500 qga shift"),
       ("—", "—"), ("ubicar la antimateria en la escala", "place antimatter on the scale"),
       "no_adoptada", ("sin respaldo; la lectura correcta es la conjugación de carga sobre $\\langle\\beta\\rangle$",
                       "no support; the correct reading is charge conjugation on $\\langle\\beta\\rangle$"), "cap. 1, ap. G"),
    _m("grover_signs", "propuestas", (r"signos de $qg_{Z,i}$", r"signs of $qg_{Z,i}$"), r"$\operatorname{sign}qg_{Z,i}$",
       ("histograma de cadenas de bits", "bit-string histogram"),
       ("leer el elemento marcado en una búsqueda con ruido", "read the marked item in a noisy search"),
       "en_contra", ("66 % frente a 91 % de la moda del histograma (5 qubits, 100 disparos)",
                     "66 % against 91 % for the histogram mode (5 qubits, 100 shots)"), "cap. 3, §63"),
    _m("grover_stop", "propuestas", (r"$qg_{SNR}$, $qg_\kappa$ de parada", r"stopping $qg_{SNR}$, $qg_\kappa$"), (r"umbral sobre $\lvert qg_k\rvert$", r"threshold on $\lvert qg_k\rvert$"),
       ("la fórmula de $k^\\ast$", "the formula for $k^\\ast$"),
       ("cortar las iteraciones de Grover", "stop Grover iterations"),
       "sin_ganancia", ("no distinguen si la bandera pasó el polo; cuestan una corrida por $k$",
                        "they cannot tell whether the flag passed the pole; they cost one run per $k$"), "cap. 3"),
    _m("qg_cycles", "propuestas", (r"ciclos qg", r"qg cycles"), (r"traducción de ángulo a lectura en Wigner, Kitaev y OTOC", r"angle-to-reading translation in Wigner, Kitaev and OTOC"),
       ("—", "—"), ("proteger estados o medir caos", "protect states or measure chaos"),
       "sin_ganancia", ("no protegen ni miden caos", "they neither protect nor measure chaos"), "cap. 34"),
)


def by_status(status: str) -> List[Measure]:
    """Measures with the given status (a key of STATUSES)."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; one of {sorted(STATUSES)}")
    return [m for m in CATALOG if m.status == status]


def by_family(family: str) -> List[Measure]:
    """Measures of the given family (a key of FAMILIES), in catalogue order."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; one of {sorted(FAMILIES)}")
    return [m for m in CATALOG if m.family == family]


def find(key: str) -> Measure:
    """The measure with this key."""
    for m in CATALOG:
        if m.key == key:
            return m
    raise KeyError(key)


def resolve(measure: Measure):
    """The qang object named by ``measure.code`` (imports its module; may need that module's optional SDK)."""
    if not measure.code:
        raise ValueError(f"{measure.key} has no code path")
    module, _, attr = measure.code.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def summary() -> Dict[str, int]:
    """Number of measures per status, in STATUSES order."""
    return {s: sum(m.status == s for m in CATALOG) for s in STATUSES}
