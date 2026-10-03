"""Builds notebooks/qang_optics.ipynb (English) and notebooks/qang_optics_es.ipynb (Spanish)."""
import sys

import nbformat as nbf

REPO = "https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/"

T = {
    "en": {
        "file": "qang_optics.ipynb",
        "title": "# Polarized light and geometric phase in qg units\n\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](" + REPO + "qang_optics.ipynb)\n\n"
        "A polarization state of light is a qubit: the Poincaré sphere is the Bloch sphere. This notebook uses the "
        "published library (`pip install qang`, modules `qang.polarization` and `qang.geometric`) to show that "
        "the normalized **Stokes parameters are qg values**, that a lossless **Mueller matrix is the qg gate rule**, "
        "**Malus's law** in qg units, polarimetry from photon counts, and the **geometric phase** as Stokes' theorem "
        "on the sphere. Each result is computed twice, with qang and with standard optics or quantum mechanics, and "
        "the two are compared.\n\n"
        "Note: `manuscript/qang_optics.pdf` (RESEARCH_NOTES §83–§84). The physics is standard; what is new is writing it "
        "in qang's units and checking it numerically.",
        "install": "## 1. Install",
        "stokes": "## 2. Stokes parameters are qg values\n\n"
        "With |H⟩ = |0⟩ and |V⟩ = |1⟩: qg_Z = S₁/S₀ (H − V), qg_X = S₂/S₀ (D − A), qg_Y = S₃/S₀ (R − L), with "
        "R = (|H⟩ + i|V⟩)/√2. The qg values from `qang.polarization` are compared with ⟨P⟩ computed by "
        "`qang.formulation` from the state.",
        "partial": "## 3. Partially polarized light is a mixed state\n\n"
        "The degree of polarization is the length of the qg vector; the purity is (1 + P²)/2.",
        "mueller": "## 4. A lossless Mueller matrix is the qg gate rule\n\n"
        "Mueller matrix (optics) against `qang.formulation.apply_gate` (qg'_P = qg_{U†PU}) for wave plates.",
        "malus": "## 5. Malus's law in qg units\n\n"
        "Behind a polarizer at angle θ: I = I₀ (1 + qg_Z cos 2θ + qg_X sin 2θ)/2, for any input. "
        "Lines: the qg formula. Points: the Mueller matrix of the polarizer.",
        "counts": "## 6. Polarimetry from photon counts\n\n"
        "Counts behind the six analyser settings give the three qg values with intervals (`qang.statistics.qg_estimate`).",
        "phase": "## 7. Geometric phase: Stokes' theorem on the sphere\n\n"
        "A loop at constant qg_Z around the Z axis encloses Ω = 2π(1 − qg_Z), and the geometric phase is −Ω/2, i.e. "
        "**(qg_Z − 1)/2 of a turn**. Computed three ways: state overlaps (Pancharatnam), solid angle, and the flux of "
        "the Berry curvature (Stokes' theorem).",
        "octant": "## 8. Pancharatnam's phase for polarized light\n\n"
        "The loop H → D → R → H encloses an octant of the Poincaré sphere (Ω = π/2), so γ = −π/4.",
        "summary": "## Summary\n\n"
        "* Stokes parameters = qg values; degree of polarization = length of the qg vector.\n"
        "* Lossless Mueller matrix = the qg gate rule; Malus's law needs only qg_Z and qg_X.\n"
        "* Geometric phase of a cone loop = (qg_Z − 1)/2 turns, checked three ways (Stokes' theorem).\n\n"
        "Tests: `tests/test_polarization.py`, `tests/test_geometric.py` (26 checks).",
        "labels": ("state", "with qang", "standard", "difference"),
        "plot_malus": ("polarizer angle θ (rad)", "transmitted intensity I/I₀"),
        "plot_phase": ("qg_Z of the loop", "geometric phase (turns)", "(qg_Z − 1)/2", "overlaps", "solid angle", "curvature flux"),
    },
    "es": {
        "file": "qang_optics_es.ipynb",
        "title": "# Luz polarizada y fase geométrica en unidades qg\n\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](" + REPO + "qang_optics_es.ipynb)\n\n"
        "Un estado de polarización de la luz es un qubit: la esfera de Poincaré es la esfera de Bloch. Este cuaderno usa "
        "la librería publicada (`pip install qang`, módulos `qang.polarization` y `qang.geometric`) para mostrar que los "
        "**parámetros de Stokes normalizados son valores qg**, que una **matriz de Mueller sin pérdidas es la regla de "
        "compuertas qg**, la **ley de Malus** en unidades qg, la polarimetría desde conteos de fotones y la **fase "
        "geométrica** como teorema de Stokes en la esfera. Cada resultado se calcula dos veces, con qang y con la óptica "
        "o mecánica cuántica estándar, y se comparan.\n\n"
        "Nota: `manuscript/qang_optics_es.pdf` (RESEARCH_NOTES §83–§84). La física es estándar; lo nuevo es escribirla en "
        "las unidades de qang y verificarla numéricamente.",
        "install": "## 1. Instalación",
        "stokes": "## 2. Los parámetros de Stokes son valores qg\n\n"
        "Con |H⟩ = |0⟩ y |V⟩ = |1⟩: qg_Z = S₁/S₀ (H − V), qg_X = S₂/S₀ (D − A), qg_Y = S₃/S₀ (R − L), con "
        "R = (|H⟩ + i|V⟩)/√2. Los valores qg de `qang.polarization` se comparan con ⟨P⟩ calculado por "
        "`qang.formulation` desde el estado.",
        "partial": "## 3. La luz parcialmente polarizada es un estado mixto\n\n"
        "El grado de polarización es la longitud del vector qg; la pureza es (1 + P²)/2.",
        "mueller": "## 4. Una matriz de Mueller sin pérdidas es la regla de compuertas qg\n\n"
        "Matriz de Mueller (óptica) contra `qang.formulation.apply_gate` (qg'_P = qg_{U†PU}) para láminas de onda.",
        "malus": "## 5. Ley de Malus en unidades qg\n\n"
        "Detrás de un polarizador a ángulo θ: I = I₀ (1 + qg_Z cos 2θ + qg_X sin 2θ)/2, para cualquier entrada. "
        "Líneas: la fórmula qg. Puntos: la matriz de Mueller del polarizador.",
        "counts": "## 6. Polarimetría desde conteos de fotones\n\n"
        "Los conteos detrás de los seis analizadores dan los tres valores qg con intervalos (`qang.statistics.qg_estimate`).",
        "phase": "## 7. Fase geométrica: teorema de Stokes en la esfera\n\n"
        "Un lazo a qg_Z constante alrededor del eje Z encierra Ω = 2π(1 − qg_Z), y la fase geométrica es −Ω/2, es decir "
        "**(qg_Z − 1)/2 de vuelta**. Se calcula de tres maneras: solapamientos de estados (Pancharatnam), ángulo sólido y "
        "flujo de la curvatura de Berry (teorema de Stokes).",
        "octant": "## 8. Fase de Pancharatnam para luz polarizada\n\n"
        "El lazo H → D → R → H encierra un octante de la esfera de Poincaré (Ω = π/2), así que γ = −π/4.",
        "summary": "## Resumen\n\n"
        "* Parámetros de Stokes = valores qg; grado de polarización = longitud del vector qg.\n"
        "* Matriz de Mueller sin pérdidas = regla de compuertas qg; la ley de Malus solo necesita qg_Z y qg_X.\n"
        "* Fase geométrica de un lazo cónico = (qg_Z − 1)/2 vueltas, verificada de tres maneras (teorema de Stokes).\n\n"
        "Pruebas: `tests/test_polarization.py`, `tests/test_geometric.py` (26 verificaciones).",
        "labels": ("estado", "con qang", "estándar", "diferencia"),
        "plot_malus": ("ángulo del polarizador θ (rad)", "intensidad transmitida I/I₀"),
        "plot_phase": ("qg_Z del lazo", "fase geométrica (vueltas)", "(qg_Z − 1)/2", "solapamientos", "ángulo sólido", "flujo de curvatura"),
    },
}

CODE = {
    "install": '!pip install -q -U "qang>=0.6.0" matplotlib\n'
    "import qang\nprint('qang', qang.__version__)",
    "stokes": """import numpy as np
import matplotlib.pyplot as plt
from qang import formulation as F
from qang import geometric as G
from qang import polarization as P

s2 = np.sqrt(2)
states = {"H": [1, 0], "V": [0, 1], "D": [1 / s2, 1 / s2], "A": [1 / s2, -1 / s2],
          "R": [1 / s2, 1j / s2], "L": [1 / s2, -1j / s2], "elliptic": [0.6, 0.8 * np.exp(0.7j)]}
print(f"{LABELS[0]:<10}{LABELS[1]:>34}{LABELS[2]:>34}{LABELS[3]:>12}")
for name, E in states.items():
    q = P.jones_to_qg(E)                                   # with qang: Stokes -> qg
    ref = F.qg_values(np.array(E, dtype=complex))          # standard: <P> from the state
    ref = {k: ref.get(k, 0.0) for k in "XYZ"}
    diff = max(abs(q[k] - ref[k]) for k in "XYZ")
    fmt = lambda d: "(" + ", ".join(f"{d[k]:+.3f}" for k in "XYZ") + ")"
    print(f"{name:<10}{fmt(q):>34}{fmt(ref):>34}{diff:>12.1e}")""",
    "partial": """E = np.array([0.6, 0.8 * np.exp(0.7j)])
for p in (0.0, 0.3, 0.6, 0.9):
    S = P.depolarizer(1 - p) @ P.jones_to_stokes(E)
    q = P.stokes_to_qg(S)
    rho = F.state_from_qg({"I": 1.0, **q}, 1)
    print(f"depolarization {p:.1f}: P = {P.degree_of_polarization(S):.3f}, "
          f"purity (qang) {P.purity_from_stokes(S):.4f}, Tr rho^2 {np.real(np.trace(rho @ rho)):.4f}")""",
    "mueller": """rng = np.random.default_rng(1)
for name, J in {"half-wave 0.3": P.half_wave_plate(0.3), "quarter-wave 1.1": P.quarter_wave_plate(1.1),
                "rotator 0.7": P.rotator(0.7)}.items():
    err = 0.0
    for _ in range(5):
        E = rng.normal(size=2) + 1j * rng.normal(size=2); E /= np.linalg.norm(E)
        q_in = P.jones_to_qg(E)
        q_mueller = P.stokes_to_qg(P.mueller_from_jones(J) @ P.qg_to_stokes(q_in))   # optics
        q_rule = F.apply_gate({"I": 1.0, **q_in}, J)                                   # qang gate rule
        err = max(err, max(abs(q_mueller[k] - q_rule.get(k, 0.0)) for k in "XYZ"))
    print(f"{name:<18} max difference Mueller vs qg rule: {err:.1e}")""",
    "malus": """theta = np.linspace(0, np.pi, 200)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for (name, E), c in zip([("H", [1, 0]), ("D", [1 / s2, 1 / s2]), ("R", [1 / s2, 1j / s2]),
                         ("elliptic", [0.6, 0.8 * np.exp(0.7j)])], ["#2a78d6", "#eb6834", "#1baf7a", "#5f5e58"]):
    q = P.jones_to_qg(E)
    ax.plot(theta, [P.malus_intensity(q, t) for t in theta], color=c, lw=2, label=name)
    pts = np.linspace(0, np.pi, 9)
    ax.plot(pts, [(P.mueller_from_jones(P.linear_polarizer(t)) @ P.qg_to_stokes(q))[0] for t in pts], "o", color=c, ms=5)
ax.set_xlabel(PLOT_MALUS[0]); ax.set_ylabel(PLOT_MALUS[1]); ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center")
ax.set_ylim(-0.05, 1.2); ax.grid(alpha=0.3)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.show()""",
    "counts": """rng = np.random.default_rng(7)
E = np.array([0.6, 0.8 * np.exp(0.7j)]); q_true = P.jones_to_qg(E)
n = 1000
nH = rng.binomial(n, (1 + q_true["Z"]) / 2); nD = rng.binomial(n, (1 + q_true["X"]) / 2); nR = rng.binomial(n, (1 + q_true["Y"]) / 2)
est = P.qg_from_counts(nH, n - nH, nD, n - nD, nR, n - nR, method="wilson")
for k in "XYZ":
    iv = est["intervals"][k]
    print(f"qg_{k}: true {q_true[k]:+.3f}, estimate {est[k]:+.3f}, 95% interval [{iv.low:+.3f}, {iv.high:+.3f}]")
print(f"degree of polarization: estimate {est['P']:.3f} (true 1.000; shot noise can push it above 1)")""",
    "phase": """def cone(qg_z, n):
    s = np.sqrt(1 - qg_z**2); t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([s * np.cos(t), s * np.sin(t), np.full(n, qg_z)], axis=1)

zs = np.linspace(-0.9, 0.9, 7)
over = [G.geometric_phase_qg(cone(z, 2000)).phi for z in zs]                                   # overlaps
solid = [(G.berry_phase_from_solid_angle(G.solid_angle(cone(z, 2000), reference=[0, 0, 1])) / (2 * np.pi)) % 1 for z in zs]
flux = [(-G.curvature_flux(z) / (2 * np.pi)) % 1 for z in zs]                                 # Stokes: flux of the curvature
zz = np.linspace(-1, 0.995, 200)  # (qg_Z - 1)/2 mod 1 jumps to 0 at qg_Z = 1
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(zz, ((zz - 1) / 2) % 1, color="#5f5e58", lw=2, label=PLOT_PHASE[2])
ax.plot(zs, over, "o", color="#2a78d6", ms=8, label=PLOT_PHASE[3])
ax.plot(zs, solid, "s", color="#eb6834", ms=5, label=PLOT_PHASE[4])
ax.plot(zs, flux, "^", color="#1baf7a", ms=5, label=PLOT_PHASE[5])
ax.set_xlabel(PLOT_PHASE[0]); ax.set_ylabel(PLOT_PHASE[1]); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.show()
print("max difference from (qg_Z - 1)/2:", max(abs(((a - (z - 1) / 2) + 0.5) % 1 - 0.5) for a, z in zip(over, zs)))""",
    "octant": """H, D, R = np.array([1, 0]), np.array([1, 1]) / s2, np.array([1, 1j]) / s2
path = [[P.jones_to_qg(E)[k] for k in "XYZ"] for E in (H, D, R)]
print("solid angle:", G.solid_angle(path), " (pi/2 =", np.pi / 2, ")")
print("Pancharatnam phase:", G.pancharatnam_phase([H, D, R]), " (-pi/4 =", -np.pi / 4, ")")""",
}


def build(lang):
    t = T[lang]
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(t["title"])]
    for key in ("install", "stokes", "partial", "mueller", "malus", "counts", "phase", "octant"):
        cells.append(nbf.v4.new_markdown_cell(t[key]))
        code = CODE[key]
        if key == "stokes":
            code = f"LABELS = {t['labels']!r}\nPLOT_MALUS = {t['plot_malus']!r}\nPLOT_PHASE = {t['plot_phase']!r}\n" + code
        cells.append(nbf.v4.new_code_cell(code))
    cells.append(nbf.v4.new_markdown_cell(t["summary"]))
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}
    return nb


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "notebooks"  # run from the repository root: python tools/<this file>
    for lang in T:
        nbf.write(build(lang), f"{out}/{T[lang]['file']}")
        print("wrote", T[lang]["file"])
