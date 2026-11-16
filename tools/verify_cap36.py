"""tools/verify_cap36.py — Cap. 36: arquitectura de software y ecosistema multi-SDK, auditada con código."""
import os
from qang.qarch import *

ROOT = os.environ.get("QANG_REPO", ".")
QDIR, TDIR, EDIR = (os.path.join(ROOT, d) for d in ("qang", "tests", "examples"))

# [1] ¿de qué depende importar cada módulo? (imports de nivel de módulo, transitivos)
req = requires(QDIR)
groups = group_modules(req)
allroots = set().union(*req.values())
extra_roots = sorted(allroots - {"numpy"})
assert "scipy" not in allroots, "el núcleo no debe exigir SciPy"
assert set(extra_roots) <= {"jax", "qiskit", "cirq", "pennylane"}, extra_roots
assert "qchar" in groups["core"] and "qgarnet" in groups["core"] and "qarch" in groups["core"]
n_mod = len(req)
print(f"[1] {n_mod} módulos de qang; paquetes de terceros exigidos al importar: numpy + {extra_roots}")
for g, mods in groups.items():
    print(f"    {g:10s} {len(mods):3d} módulos" + (f": {', '.join(mods)}" if g != "core" else ""))
assert groups["jax"] and len(groups["core"]) >= 40

# [2] extras de pyproject.toml
with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
    ex = parse_pyproject(fh.read())
chk = check_extras(ex)
assert all(chk.values()), chk
avail = installed_by(ex, "all")
print(f"[2] pyproject: {len(ex) - 1} extras; {chk}; el CI instala «.[all]» = {len(avail)} paquetes (jax y torch no entran)")
assert "jax" not in avail

# [3] pruebas que fallarían al recolectar en el CI (código de salida 2)
bad = unguarded_tests(TDIR, QDIR, avail, [EDIR])
n_tests = len([f for f in os.listdir(TDIR) if f.startswith("test_") and f.endswith(".py")])
assert bad == [], bad
need_jax = [m for m in groups["jax"]]
print(f"[3] {n_tests} archivos de prueba; los que exigen un paquete que «.[all]» no instala y no lo protegen con importorskip: {len(bad)}")

# [4] convención de qubits entre SDKs (qang.interop): leer con la convención equivocada
import itertools
from qang.interop import canonical_counts, qg_z_profile
qg = [0.6, -0.2, 0.0]
p0 = [(1 + q) / 2 for q in qg]
N = 10 ** 6
qiskit_counts, cirq_counts = {}, {}
for bits in itertools.product((0, 1), repeat=3):
    pr = 1.0
    for b, p in zip(bits, p0):
        pr *= p if b == 0 else 1 - p
    c = round(pr * N)
    cirq_counts["".join(map(str, bits))] = c                 # qubit 0 primero
    qiskit_counts["".join(map(str, bits[::-1]))] = c         # qubit 0 al final
ok = qg_z_profile(canonical_counts(qiskit_counts, sdk="qiskit"))
assert all(abs(a[0] - b) < 1e-5 for a, b in zip(ok, qg))
assert canonical_counts(qiskit_counts, sdk="qiskit") == canonical_counts(cirq_counts, sdk="cirq")
err = convention_error_mqg(qiskit_counts, "qiskit", "cirq")
assert abs(err - 600.0) < 0.01
sym = {"000": 5, "111": 5}
assert convention_error_mqg(sym, "qiskit", "cirq") == 0.0
print(f"[4] qg_Z preparado {qg}: qiskit y cirq dan el mismo resultado canónico; leer «qiskit» como «cirq» desvía hasta {err:.1f} m-qg "
      f"(qubit 0 ↔ qubit 2); con un registro simétrico (000/111) el desvío es {convention_error_mqg(sym, 'qiskit', 'cirq'):.1f} m-qg: no avisa")
print("verify_cap36: OK")
