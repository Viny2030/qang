"""tools/verify_cap32.py — Cap. 32: libro de cuentas de submedidas (Cap. 7–31)."""
import glob
import os
from collections import Counter
from qang.qledger import *

BOOK, QANG, TOOLS = "book", "qang", "tools"
rows = load_ledger(BOOK, 7, 31)
caps = sorted(set(r["cap"] for r in rows))
assert caps == list(range(7, 32)), caps
# [1] estatus válidos
bad_status = [(r["cap"], r["id"], t) for r in rows for t in statuses_of(r["status"]) if t not in STATUSES]
assert not bad_status, bad_status
assert all(primary_status(r["status"]) in STATUSES for r in rows)
# [2] funciones / verificadores existentes
bad = unresolved(rows, QANG, TOOLS)
assert not bad, bad
# [3] cada capítulo tiene verificador y bloque de test
for n in caps:
    assert os.path.exists(f"{TOOLS}/verify_cap{n:02d}.py"), n
    md = glob.glob(f"{BOOK}/cap{n:02d}_*.md")[0]
    assert "`tests/test_" in open(md, encoding="utf-8").read(), md
# [4] ids únicos
ids = Counter(r["id"] for r in rows)
assert max(ids.values()) == 1, [k for k, v in ids.items() if v > 1]
S = summarize(rows)
print(f"[1] {len(rows)} submedidas en {len(caps)} capítulos (7–31); estatus válidos: ", dict(S))
print(f"[2] funciones y verificadores citados existen: {sum(len(function_names(r['functions'])) for r in rows)} referencias, 0 sin resolver")
print(f"[3] 25/25 capítulos con verify_capNN.py y bloque de test; ids únicos: {len(ids)}")
neg = [(r["cap"], r["id"], primary_status(r["status"])) for r in rows if primary_status(r["status"]) in ("against", "no gain", "mixed", "unverified")]
print("[4] hallazgos no positivos (ledger):")
for c, i, s in neg:
    print(f"    Cap. {c:>2}  {i:32s} {s}")
CV, CE = caveats(rows), clean_exact(rows)
assert len(CV) == 19 and len(CE) == 64, (len(CV), len(CE))
print(f"[5] resultados `exact` sin matiz: {len(CE)}; filas con matiz (hardware/mixed/against/no gain): {len(CV)}")
assert S["exact"] + S["against"] + S["no gain"] + S["mixed"] + S["unverified"] == len(rows)
assert sum(S.values()) == len(rows) and S["measured gain"] == 0 and S["tested on hardware"] == 0
print("verify_cap32: OK")
