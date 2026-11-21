# Progreso de revisión teórica y validación — libro qang (ES)

**Repositorio:** `C:\Users\ASUS\qang_repo` · **Rama:** `main` (sin crear ramas nuevas)
**Alcance acordado:** capítulos `cap01`–`cap06` de `book/es` (los 56 archivos ES/EN restantes quedan fuera de este bloque).
**Límite de tiempo:** 3 horas (180 min). **Inicio:** 2026-11-21.

## Checklist por capítulo

| # | Capítulo | Revisión teórica | Fuentes en línea | Validación código | `[x]` y commit | Min |
|---|----------|------------------|------------------|-------------------|----------------|-----|
| 1 | `book/es/cap01.qmd` | [x] | [x] | [x] | [x] | 35 |
| 2 | `book/es/cap02.qmd` | [x] | [x] | [x] | [x] | 22 |
| 3 | `book/es/cap03.qmd` | [x] | [x] | [x] | [x] | 22 |
| 4 | `book/es/cap04.qmd` | [x] | [x] | [x] | [x] | 22 |
| 5 | `book/es/cap05.qmd` | [x] | [x] | [x] | [x] | 22 |
| 6 | `book/es/cap06.qmd` | [x] | [x] | [x] | [x] | 25 |
| 7 | Reporte final y estado de Git | [x] | — | — | [x] | 10 |

**Total estimado:** 158 min (2 h 38 min), con ~22 min de margen.

## Reglas de ejecución

1. Leer el capítulo completo antes de tocar nada; conservar **toda** la información técnica original.
2. Contrastar cada afirmación clave con literatura/documentación en línea; anotar la fuente en el reporte.
3. Ejecutar el script `tools/verify_capNN.py` correspondiente y los tests unitarios relacionados.
4. Marcar `[x]` en este archivo y hacer `git add` + `git commit` **antes** de pasar al siguiente capítulo.
5. No crear ramas, no borrar archivos, no añadir `.venv_clean/` ni otros artefactos al commit.
6. Si se agota el tiempo, entregar el reporte con lo completado y lo pendiente.

## Registro de avance

- [x] **Setup y descubrimiento** (hecho: ruta real del libro, rama, inventario de capítulos y tests).
- [x] **cap01** — commit `docs: revisión teórica y validación cap 1`.
- [x] **cap02** — commit `docs: revisión teórica y validación cap 2`.
- [x] **cap03** — commit `docs: revisión teórica y validación cap 3`.
- [x] **cap04** — commit `docs: revisión teórica y validación cap 4`.
- [x] **cap05** — commit `docs: revisión teórica y validación cap 5`.
- [x] **cap06** — commit `docs: revisión teórica y validación cap 6`.
- [x] **Reporte final**.

## Reporte final

- **Commits:** `e4a2ece` (cap01), `14fbd0a` (cap02), `3a9b9e8` (cap03), `7a7d3e3` (cap04), `52dbb4a` (cap05), `ff3718e` (cap06).
- **Validación pytest:** todas las corridas parciales pasaron (cap01 52, cap02 87, cap03 43, cap04 54, cap05 83, cap06 38 pruebas); la suite consolidada final pasó 213 pruebas en 23,17 s.
- **Verificadores:** `tools/verify_cap06_expansion.py` pasó 12/12; `verify_cap01.py`–`verify_cap06.py` quedan bloqueados por `ModuleNotFoundError: No module named 'qutip'`.
- **No disponible:** render Quarto (`quarto` no instalado).
- **Fuentes en línea contrastadas:** Wootters 1981 (DOI `10.1103/PhysRevD.23.357`), Anscombe 1948 (`10.1093/biomet/35.3-4.246`), Freeman–Tukey 1950 (`10.1214/aoms/1177729756`), Shannon 1948 (`10.1002/j.1538-7305.1948.tb01338.x`), Suzuki 2019 (`10.1007/s11128-019-2565-2`), Berta 2010 (`10.1038/nphys1734`), Coles 2014 (`10.1038/ncomms6814`), Goldberg 2020 (`10.1364/aop.404175`), más documentación de PennyLane, Wikipedia y PDG.
- **Estado Git:** `main` con 7 commits de trabajo; `.venv_clean/` permanece sin trackear y no se incluyó en ningún commit.

---

# Bloque 2 — revisión teórica y validación — capítulos `qjax07`–`qjax12`

**Repositorio:** `C:\Users\ASUS\qang_repo` · **Rama:** `main` (sin crear ramas nuevas).
**Alcance acordado:** capítulos `book/es/qjax07.qmd`–`book/es/qjax12.qmd` (equivalentes a cap07–cap12 en la numeración del usuario).
**Límite de tiempo:** 3 horas (180 min). **Inicio:** 2026-11-21.

## Checklist por capítulo (bloque 2)

| # | Capítulo | Revisión teórica | Fuentes en línea | Validación código | `[x]` y commit | Min |
|---|----------|------------------|------------------|-------------------|----------------|-----|
| 1 | `book/es/qjax07.qmd` | [x] | [x] | [x] | [x] | 30 |
| 2 | `book/es/qjax08.qmd` | [ ] | [ ] | [ ] | [ ] | 30 |
| 3 | `book/es/qjax09.qmd` | [ ] | [ ] | [ ] | [ ] | 30 |
| 4 | `book/es/qjax10.qmd` | [ ] | [ ] | [ ] | [ ] | 30 |
| 5 | `book/es/qjax11.qmd` | [ ] | [ ] | [ ] | [ ] | 30 |
| 6 | `book/es/qjax12.qmd` | [ ] | [ ] | [ ] | [ ] | 30 |
| 7 | Reporte final y estado de Git | [ ] | — | — | [ ] | 10 |

**Total estimado:** 180 min.

## Reglas de ejecución (bloque 2)

1. Leer el capítulo completo antes de tocar nada; conservar **toda** la información técnica original.
2. Contrastar cada afirmación clave con literatura/documentación en línea; anotar la fuente en el reporte.
3. Ejecutar el script `tools/verify_capNN.py` correspondiente y los tests unitarios relacionados.
4. Marcar `[x]` en este archivo y hacer `git add` + `git commit` **antes** de pasar al siguiente capítulo.
5. No crear ramas, no borrar archivos, no añadir `.venv_clean/` ni otros artefactos al commit.
6. Si se agota el tiempo, entregar el reporte con lo completado y lo pendiente.

## Registro de avance (bloque 2)

- [x] **Setup y descubrimiento** (hecho: ruta real del libro, rama, inventario de capítulos y tests).
- [x] **qjax07** (cap07) — revisión, fuentes, `verify_cap07.py`, `tests/test_qan.py` (4 passed) y commit.
- [ ] **qjax08** (cap08).
- [ ] **qjax09** (cap09).
- [ ] **qjax10** (cap10).
- [ ] **qjax11** (cap11).
- [ ] **qjax12** (cap12).
- [ ] **Reporte final (bloque 2)**.
