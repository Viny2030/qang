"""tools/verify_exp32_1.py — Exp. 32/ApH.1: auditoría del libro de 90 submedidas."""
from __future__ import annotations

from collections import Counter

from qang.qledger import (
    STATUSES,
    caveats,
    clean_exact,
    load_ledger,
    statuses_of,
    summarize,
    unresolved,
)


def main() -> None:
    print("Exp. 32/ApH.1 — Libro de 90 submedidas (Caps. 7–31): consistencia y niveles de evidencia")

    rows = load_ledger("book", 7, 31)
    ids = Counter(r["id"] for r in rows)
    s = summarize(rows)
    caveat_rows = caveats(rows)

    assert sorted({r["cap"] for r in rows}) == list(range(7, 32))
    assert not [(r["cap"], r["id"], t) for r in rows for t in statuses_of(r["status"]) if t not in STATUSES]
    assert not unresolved(rows, "qang", "tools")
    assert max(ids.values()) == 1
    assert len(rows) == 90 and len(ids) == 90
    assert s["exact"] == 82 and s["against"] == 3 and s["no gain"] == 3
    assert s["mixed"] == 1 and s["unverified"] == 1
    assert s["measured gain"] == 0 and s["tested on hardware"] == 0
    assert len(caveat_rows) == 19 and len(clean_exact(rows)) == 64
    assert sum(s.values()) == len(rows)

    print(f"  filas={len(rows)} en {len({r['cap'] for r in rows})} capítulos; ids únicos={len(ids)}")
    print(f"  resumen: {dict(s)}")
    print(f"  referencias sin resolver: {len(unresolved(rows, 'qang', 'tools'))}; matices={len(caveat_rows)}; exacto limpio=64")
    print("verify_exp32_1: OK")


if __name__ == "__main__":
    main()
