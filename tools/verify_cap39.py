"""tools/verify_cap39.py — Cap. 39: qang.integrations auditado con código (no necesita ningún SDK instalado)."""
import os
from qang.qintegr import *
from qang.qarch import parse_pyproject

ROOT = os.environ.get("QANG_REPO", ".")
QDIR = os.path.join(ROOT, "qang")

# [1] qué importa cada submódulo y si lo protege con require()
mods = integration_modules(QDIR)
assert set(mods) == set(INTEGRATIONS), (set(mods) ^ set(INTEGRATIONS))
rep = lazy_report(QDIR)
rg = require_guard_report(QDIR)
assert all(rg.values()), [m for m, ok in rg.items() if not ok]
eager = sorted(m for m, (t, _) in rep.items() if t)
print(f"[1] {len(mods)} integraciones; las {len(eager)} que importan su SDK a nivel de módulo ({', '.join(eager)}) llaman antes a require(); "
      f"las otras {len(mods) - len(eager)} lo importan de forma diferida; guardas correctas: {sum(rg.values())}/{len(rg)}")

# [2] extras de pyproject.toml
with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
    ex = parse_pyproject(fh.read())
er = extra_report(ex)
assert all(er.values()), [m for m, ok in er.items() if not ok]
bundle = bundle_libraries(ex)
outside = sorted(root for _, root in INTEGRATIONS.values() if root not in bundle)
assert outside == ["cudaq", "qiskit_dynamics"], outside
print(f"[2] un extra por integración, cada uno con su paquete: {len(er)}/{len(er)}; el extra «integrations» reúne {len(INTEGRATIONS) - len(outside)} "
      f"({', '.join(outside)} quedan fuera: entornos aparte)")

# [3] pruebas: cada integración tiene una y cada una está protegida con importorskip
tp = os.path.join(ROOT, "tests", "test_integrations.py")
bad = guard_report(tp)
tm = modules_with_tests(tp)
assert bad == {}, bad
assert tm == set(INTEGRATIONS), set(INTEGRATIONS) - tm
print(f"[3] tests/test_integrations.py ejercita las {len(tm)} integraciones; funciones de prueba con un paquete sin importorskip: {len(bad)}")

# [4] el error dice qué instalar
msgs = {m: missing_library_message(extra) for m, (extra, _) in INTEGRATIONS.items()}
assert all(f'qang[{INTEGRATIONS[m][0]}]' in s for m, s in msgs.items())
print(f"[4] mensaje cuando falta la librería: «{msgs['mitiq']}»")

# [5] filtro bajo T1 en un estado W de 3 qubits, sin SDK (t/T1 = 0.6)
r = w_state_t1_readings(0.6)
assert abs(r["raw_mqg"] - 634.1) < 0.05 and abs(r["filtered_mqg"] - r["ideal_mqg"]) < 1e-9 and abs(r["kept"] - np.exp(-0.6)) < 1e-12
print(f"[5] W de 3 qubits, t/T1 = 0.6: lectura cruda {r['raw_mqg']:+.1f} m-qg, filtrada {r['filtered_mqg']:+.3f} m-qg (ideal {r['ideal_mqg']:+.3f}); "
      f"fracción conservada {r['kept']:.4f} = e^(−0.6)")
print("verify_cap39: OK")
