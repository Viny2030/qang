"""Add section 5, the local radius (RESEARCH_NOTES §97-§98), to the two
start-here notebooks (English and Spanish) and raise the install cell to
qang >= 0.6.4. Idempotent: an existing radius section is replaced.

python tools/add_radius_section.py
"""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")

INSTALL = '''# In Colab: installs qang from PyPI (0.6.4 or later ships the radius functions).
import importlib, subprocess, sys
def _ok():
    try:
        import qang.formulation as _F
        return hasattr(_F, "radius_profile")
    except ImportError:
        return False
if not _ok():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "qang>=0.6.4"], check=False)
    importlib.invalidate_caches()
    if not _ok():  # fallback: install from GitHub
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/Viny2030/qang.git"], check=True)
        importlib.invalidate_caches()
import numpy as np
import qang
from qang import formulation as F
print("qang", qang.__version__)'''

INSTALL_ES = INSTALL.replace("# In Colab: installs qang from PyPI (0.6.4 or later ships the radius functions).",
                             "# En Colab: instala qang desde PyPI (0.6.4 o posterior trae las funciones del radio).") \
                    .replace("# fallback: install from GitHub", "# si PyPI no la tiene, se instala desde GitHub")

CODE_DEF = '''# r^2 = qg_X^2 + qg_Y^2 + qg_Z^2 for each qubit; area 4 pi r^2
ket0 = np.array([1, 0, 0, 0])
bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
mixed = F.state_from_qg({"I": 1, "Z": 0.6}, 1)
for name, s in [("|00>", ket0), ("Bell", bell), ("qubit with qg_Z = 0.6", mixed)]:
    r2 = F.radius_profile(s)
    print(f"{name:22s} r^2 = {r2}   area / 4pi = {[F.sphere_area(x) / (4 * np.pi) for x in r2]}   "
          f"Meyer-Wallach Q = {F.meyer_wallach(s)}")
# for a pure two-qubit state the deficit 1 - r^2 is the squared concurrence
rng = np.random.default_rng(0)
psi = rng.normal(size=4) + 1j * rng.normal(size=4); psi /= np.linalg.norm(psi)
print("random pure state: 1 - r^2 =", F.radius_deficit(psi), " concurrence^2 =",
      round((2 * abs(psi[0] * psi[3] - psi[1] * psi[2])) ** 2, 12))'''

CODE_GATES = '''# The 15 gates on the radius (product inputs: stabilizer products + random products)
gates = {**F.GATES_1Q, "Rx(0.7)": F.rx(0.7), "Ry(0.7)": F.ry(0.7), "Rz(0.7)": F.rz(0.7), "P(0.7)": F.phase(0.7),
         **F.GATES_MULTI}
for name, U in gates.items():
    c = F.gate_radius_class(U)
    print(f"{name:8s} {c['class']:10s} max deficit {c['max deficit']}")'''

CODE_ALGOS = '''# Radius of the readout qubits for one instance of each algorithm
for name, r2 in F.algorithm_radii().items():
    if isinstance(r2, dict):
        r2 = {k: round(v[0], 3) for k, v in r2.items()}
    elif isinstance(r2, list):
        r2 = [round(x, 3) for x in r2]
    print(f"{name:26s} {r2}")'''

CODE_FILTER = '''# Weight-conserving circuit under T1, with and without qang (RESEARCH_NOTES §98)
from qang.qml import WeightQNN
from qang.sectors import filter_distribution
from qang.statistics import qg2_unbiased
m = WeightQNN(6, 2, layers=8)                      # 6 qubits, weight 2, depth 24
rng = np.random.default_rng(98)
th = rng.uniform(-np.pi, np.pi, m.n_theta)
psi = np.zeros((1, m.dim)); psi[0, m.idx[2]] = rng.normal(size=len(m.idx[2])); psi /= np.linalg.norm(psi)
p_exact = m.probs(th, psi)[0]
p_noisy = m.probs(th, psi, 0.02)[0]                 # equal T1, gamma = 0.02 per qubit per layer
truth = 1 - (p_exact @ m.zsign) ** 2                 # noiseless tangle of each qubit
with_qang = 1 - (filter_distribution(p_noisy, 6, 2)[0] @ m.zsign) ** 2
without = 1 - (p_noisy @ m.zsign) ** 2
print("noiseless tangle :", np.round(truth, 3))
print("with qang        :", np.round(with_qang, 3), " max error", f"{np.max(np.abs(with_qang - truth)):.1e}")
print("without qang     :", np.round(without, 3), " max error", f"{np.max(np.abs(without - truth)):.3f}")
# from 1000 shots, unbiased estimator (N q^2 - 1)/(N - 1) on the kept shots
c = rng.multinomial(1000, p_noisy / p_noisy.sum())
kept = np.where(m.wt == 2, c, 0)
est = [1 - qg2_unbiased(int(kept @ (m.zsign[:, q] > 0)), int(kept.sum())) for q in range(6)]
print("from shots, qang :", np.round(est, 3), f"({kept.sum()} of 1000 shots kept)")'''

EN = [
    ("markdown", "## 5. The local radius: the surface of the sphere (§97)\n\n"
     "Each qubit's local qg values are a point in the unit ball. Its squared radius\n\n"
     "$$r_q^2 = qg_X^2 + qg_Y^2 + qg_Z^2 = 2\\,\\mathrm{Tr}\\rho_q^2 - 1$$\n\n"
     "is 1 on the surface (pure qubit) and 0 at the centre; the qubit lives on a sphere of area $4\\pi r_q^2$, "
     "and the surface deficit is $4\\pi(1-r_q^2)$. If the whole register is pure, $1-r_q^2$ is the known "
     "one-tangle of qubit $q$ with the rest (two qubits: the squared concurrence), and its mean is the "
     "Meyer–Wallach $Q$. If the register is noisy, the deficit mixes entanglement and noise."),
    ("code", CODE_DEF),
    ("markdown", "**The 15 gates.** One-qubit gates rotate the sphere and keep every radius; SWAP exchanges radii; "
     "CNOT, CZ, iSWAP, Toffoli and Fredkin can take a product input to the centre (deficit 1)."),
    ("code", CODE_GATES),
    ("markdown", "**The 14 algorithms.** Product outputs (Bernstein–Vazirani, QFT of a basis state, constant "
     "Deutsch–Jozsa) stay on the surface; Simon and Shor's counting register sit near the centre (the answer is in "
     "correlations); Grover entangles and returns to the surface at the optimum; a Hadamard test has "
     "$r^2 = |\\langle\\psi|U|\\psi\\rangle|^2$, 1 only for an eigenstate."),
    ("code", CODE_ALGOS),
    ("markdown", "**The radius after the qg filter (§98).** In a state of fixed Hamming weight $qg_X = qg_Y = 0$ on "
     "every qubit, so $r_q = |qg_Z|$: the deficit is read from Z-basis shots. Under equal T1 the filter keeps the "
     "noiseless state, so the filtered deficit is the tangle; without the filter, decay adds a false deficit."),
    ("code", CODE_FILTER),
    ("markdown", "**Result of the pre-registered study (§98).** Under equal T1 the filtered deficit equals the "
     "noiseless tangle to $3\\times10^{-15}$; on product states it is exactly 0, while without qang decay shows a "
     "false deficit of 0.60–0.95; from shots the filter has the lower error in 30 of 30 configurations "
     "(24 of 30 with dephasing, which it does not correct). One prediction failed (the half-filling case). "
     "Article: `manuscript/qang_formulation.pdf`, section 6."),
]

ES = [
    ("markdown", "## 5. El radio local: la superficie de la esfera (§97)\n\n"
     "Los valores qg locales de cada qubit son un punto de la bola unidad. Su radio al cuadrado\n\n"
     "$$r_q^2 = qg_X^2 + qg_Y^2 + qg_Z^2 = 2\\,\\mathrm{Tr}\\rho_q^2 - 1$$\n\n"
     "vale 1 en la superficie (qubit puro) y 0 en el centro; el qubit vive sobre una esfera de área "
     "$4\\pi r_q^2$, y el déficit de superficie es $4\\pi(1-r_q^2)$. Si el registro entero es puro, $1-r_q^2$ es "
     "el entrelazamiento conocido del qubit $q$ con el resto (con dos qubits, la concurrencia al cuadrado), y su "
     "promedio es la $Q$ de Meyer–Wallach. Si el registro tiene ruido, el déficit mezcla entrelazamiento y ruido."),
    ("code", CODE_DEF),
    ("markdown", "**Las 15 compuertas.** Las de un qubit giran la esfera y conservan todos los radios; SWAP "
     "intercambia radios; CNOT, CZ, iSWAP, Toffoli y Fredkin pueden llevar una entrada producto al centro "
     "(déficit 1)."),
    ("code", CODE_GATES),
    ("markdown", "**Los 14 algoritmos.** Las salidas producto (Bernstein–Vazirani, QFT de un estado de la base, "
     "Deutsch–Jozsa constante) quedan en la superficie; Simon y el registro de conteo de Shor quedan cerca del "
     "centro (la respuesta está en las correlaciones); Grover entrelaza y vuelve a la superficie en el óptimo; una "
     "prueba de Hadamard tiene $r^2 = |\\langle\\psi|U|\\psi\\rangle|^2$, 1 solo con un autoestado."),
    ("code", CODE_ALGOS),
    ("markdown", "**El radio después del filtro qg (§98).** En un estado de peso de Hamming fijo $qg_X = qg_Y = 0$ "
     "en todos los qubits, así que $r_q = |qg_Z|$: el déficit se lee con disparos en la base Z. Con T1 igual el "
     "filtro conserva el estado sin ruido, y el déficit filtrado es el entrelazamiento; sin filtro, el decaimiento "
     "agrega un déficit falso."),
    ("code", CODE_FILTER),
    ("markdown", "**Resultado del estudio pre-registrado (§98).** Con T1 igual el déficit filtrado coincide con el "
     "entrelazamiento sin ruido a $3\\times10^{-15}$; en estados producto es exactamente 0, mientras que sin qang "
     "el decaimiento muestra un déficit falso de 0.60–0.95; con disparos el filtro tiene el menor error en 30 de "
     "30 configuraciones (24 de 30 con desfase, que no corrige). Una predicción falló (el caso a mitad de "
     "llenado). Artículo: `manuscript/qang_teoria_es.pdf`, sección 5."),
]


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def patch(fname, cells, install, marker):
    path = os.path.join(ROOT, fname)
    nb = json.load(open(path, encoding="utf-8"))
    nb["cells"][2] = cell("code", install)
    keep = []
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and "".join(c["source"]).startswith(marker):
            break
        keep.append(c)
    nb["cells"] = keep + [cell(k, s) for k, s in cells]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    patch("qang_start_here.ipynb", EN, INSTALL, "## 5. The local radius")
    patch("qang_inicio_formulacion.ipynb", ES, INSTALL_ES, "## 5. El radio local")
    print("ok")
