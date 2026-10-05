"""Add section 5, the local radius (RESEARCH_NOTES §97-§98), to the two
start-here notebooks (English and Spanish) and raise the install cell to
qang >= 0.6.4. Idempotent: an existing radius section is replaced.

python tools/add_radius_section.py
"""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")

INSTALL = '''# In Colab: installs qang from PyPI (0.6.15 or later ships the radius, angle and echo functions).
import importlib, subprocess, sys
def _ok():
    try:
        import qang.formulation as _F
        import qang.sectors as _S
        return hasattr(_F, "direction_from_qg") and hasattr(_S, "echo_transfer_matrix")
    except ImportError:
        return False
if not _ok():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "qang>=0.6.15"], check=False)
    importlib.invalidate_caches()
    if not _ok():  # fallback: install from GitHub
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/Viny2030/qang.git"], check=True)
        importlib.invalidate_caches()
import numpy as np
import qang
from qang import formulation as F
print("qang", qang.__version__)'''

INSTALL_ES = INSTALL.replace("# In Colab: installs qang from PyPI (0.6.15 or later ships the radius, angle and echo functions).",
                             "# En Colab: instala qang desde PyPI (0.6.15 o posterior trae el radio, el ángulo y el eco).") \
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

CODE_ANGLE = """# The polar angle in radians, separated from the radius (§102)
from qang.statistics import direction_estimate
theta, phi = 0.5, 0.3
q = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
for p in (0.0, 0.2, 0.4):                       # depolarizing: the vector shrinks by (1 - p)
    qx, qy, qz = (1 - p) * q
    t_qang, _, r = F.direction_from_qg(qx, qy, qz)
    print(f"p = {p:.1f}: r = {r:.2f} | with qang theta = {t_qang:.3f} | without qang arccos(qg_Z) = {np.arccos(qz):.3f}"
          f"   (true {theta})")
# from shots: 1000 per basis (with qang) against 3000 in Z (without qang)
rng = np.random.default_rng(0)
qx, qy, qz = 0.8 * q
k = lambda v, n: (rng.binomial(n, (1 + v) / 2), n)
print("from counts, with qang:", round(direction_estimate(k(qx, 1000), k(qy, 1000), k(qz, 1000))[0], 3),
      "| without qang:", round(float(np.arccos(2 * k(qz, 3000)[0] / 3000 - 1)), 3))"""

CODE_ECHO = """# Echo calibration of the errors the filter keeps (§105-§108), weight 1 on 5 qubits, in NumPy
from qang.sectors import echo_transfer_matrix, unmix_sector, filter_distribution, sector_states
n, k = 5, 1
states = sector_states(n, k)                     # the 5 weight-1 basis states
rng = np.random.default_rng(1)
A = 0.9 * np.eye(5) + rng.uniform(0, 0.025, (5, 5)) * (1 - np.eye(5)); A /= A.sum(0)  # forward in-sector error
M_true = A @ A                                   # an echo has twice the gates
def noisy(x_sector, kept=0.75):                  # in-sector error + decay out of the sector
    p = np.zeros(2**n); p[states] = kept * (A @ x_sector); p[0] = 1 - kept; return p
def echo(j, kept=0.7):
    p = np.zeros(2**n); p[states] = kept * M_true[:, j]; p[0] = 1 - kept; return p
M = echo_transfer_matrix([echo(j) for j in range(5)], n, k)
x = rng.dirichlet(np.ones(5))                    # the noiseless excitation probabilities
p = noisy(x)
zs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)   # qg_Z sign of every outcome, qubit i = bit i
truth = np.zeros(2**n); truth[states] = x
for label, d in (("without qang", p), ("qang (filter)", filter_distribution(p, n, k)[0]),
                 ("qang + echo", unmix_sector(p, M, n, k, power=0.5))):
    print(f"{label:14s} qg_Z error {np.mean(np.abs(d @ zs - truth @ zs)):.4f}")"""

CODE_ECHO_RULE = """# When NOT to use the echo (§108): compare the filtered error with its shot-noise level
shots = 1000
qz_filter = filter_distribution(p, n, k)[0] @ zs
shot_sd = np.sqrt((1 - qz_filter**2) / (0.75 * shots))   # 75% of the shots kept
bias = np.abs(qz_filter - truth @ zs)
print("filtered bias per qubit     :", np.round(bias, 3))
print("shot-noise sd per qubit     :", np.round(shot_sd, 3))
print("use the echo calibration?   :", bool(np.mean(bias) > 2 * np.mean(shot_sd)))"""

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
    ("markdown", "## 6. The angle in radians, separated from the radius (§102, §103)\n\n"
     "$qg_Z = r\\cos\\theta$ mixes the direction with the length of the vector, so $\\arccos(qg_Z)$ drifts towards "
     "$\\pi/2$ when noise shortens it. The radius separates them: "
     "$\\theta = \\mathrm{atan2}(\\sqrt{qg_X^2+qg_Y^2},\\, qg_Z) = \\arccos(qg_Z/r)$. Exact under depolarizing noise, "
     "5-9x more accurate on IBM and IonQ noise models (IonQ in the native gate set); worse under pure dephasing and "
     "when the vector does not shrink (three bases cost shots)."),
    ("code", CODE_ANGLE),
    ("markdown", "## 7. Echo calibration of the errors the filter keeps (§105-§108)\n\n"
     "The filter discards shots that left the sector, but errors that move an excitation inside it pass. An echo "
     "circuit (a sector state, the circuit and its inverse) measures them: its filtered distribution is a column of a "
     "transfer matrix $M$; inverting $M^{1/2}$ on the filtered distribution removes 38-47% of what the filter leaves on "
     "IBM and IonQ noise models (weight 1 and 2), and it halved the classifier's decision error in §106."),
    ("code", CODE_ECHO),
    ("markdown", "**When not to use it (§108).** If the filtered error is already close to the shot noise, the "
     "inversion adds variance and gains nothing: on narrow-margin inputs the filter cut flipped decisions from 47 to "
     "17 and the echo left 19. Compare the two first:"),
    ("code", CODE_ECHO_RULE),
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
    ("markdown", "## 6. El ángulo en radianes, separado del radio (§102, §103)\n\n"
     "$qg_Z = r\\cos\\theta$ mezcla la dirección con la longitud del vector, así que $\\arccos(qg_Z)$ se corre hacia "
     "$\\pi/2$ cuando el ruido lo acorta. El radio los separa: "
     "$\\theta = \\mathrm{atan2}(\\sqrt{qg_X^2+qg_Y^2},\\, qg_Z) = \\arccos(qg_Z/r)$. Exacto con ruido despolarizante, "
     "entre 5 y 9 veces más preciso en los modelos de ruido de IBM e IonQ (IonQ con compuertas nativas); peor con "
     "desfase puro y cuando el vector no se achica (las tres bases cuestan disparos)."),
    ("code", CODE_ANGLE),
    ("markdown", "## 7. Calibración por eco de los errores que deja el filtro (§105-§108)\n\n"
     "El filtro descarta los disparos que salieron del sector, pero los errores que mueven una excitación dentro del "
     "sector pasan. Un circuito de eco (un estado del sector, el circuito y su inverso) los mide: su distribución "
     "filtrada es una columna de una matriz de transferencia $M$; invertir $M^{1/2}$ sobre la distribución filtrada "
     "quita entre el 38 y el 47 % de lo que deja el filtro en los modelos de ruido de IBM e IonQ (peso 1 y 2), y redujo "
     "a la mitad el error de decisión del clasificador en §106."),
    ("code", CODE_ECHO),
    ("markdown", "**Cuándo no usarla (§108).** Si el error filtrado ya está cerca del ruido de disparos, la inversión "
     "agrega varianza y no gana nada: en entradas de margen estrecho el filtro bajó las decisiones cambiadas de 47 a "
     "17 y el eco dejó 19. Conviene comparar primero:"),
    ("code", CODE_ECHO_RULE),
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
