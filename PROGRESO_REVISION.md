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
| 2 | `book/es/qjax08.qmd` | [x] | [x] | [x] | [x] | 30 |
| 3 | `book/es/qjax09.qmd` | [x] | [x] | [x] | [x] | 30 |
| 4 | `book/es/qjax10.qmd` | [x] | [x] | [x] | [x] | 30 |
| 5 | `book/es/qjax11.qmd` | [x] | [x] | [x] | [x] | 30 |
| 6 | `book/es/qjax12.qmd` | [x] | [x] | [x] | [x] | 30 |
| 7 | Reporte final y estado de Git | [x] | — | — | [x] | 10 |

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
- [x] **qjax08** (cap08) — revisión, fuentes, `verify_cap08.py`, `tests/test_qdist.py` (4 passed) y commit.
- [x] **qjax09** (cap09) — revisión, fuentes, `verify_cap09.py`, `tests/test_qfisher.py` (3 passed) y commit.
- [x] **qjax10** (cap10) — revisión, fuentes, `verify_cap10.py`, `tests/test_qgeom.py` (2 passed) y commit.
- [x] **qjax11** (cap11) — revisión, fuentes, `verify_cap11.py`, `tests/test_qmetro.py` (3 passed) y commit.
- [x] **qjax12** (cap12) — revisión, fuentes, `verify_cap12.py`, `tests/test_qneutrino_vacuum.py` (3 passed) y commit.
- [x] **Reporte final (bloque 2)**.

## Reporte final (bloque 2)

- **Commits:** `684667d` (cap07), `fe1ddf3` (cap08), `2928be3` (cap09), `996501c` (cap10), `d379873` (cap11), `dddd9b0` (cap12); cierre de reporte en `docs: reporte final bloque 2`.
- **Verificadores:** `tools/verify_cap07.py`–`tools/verify_cap12.py` pasan los seis con `verify_capNN: OK`.
- **Pytest del bloque:** `tests/test_qan.py`, `test_qdist.py`, `test_qfisher.py`, `test_qgeom.py`, `test_qmetro.py` y `test_qneutrino_vacuum.py`: `19 passed in 14.45 s`.
- **Pytest global:** la suite completa se interrumpió por tiempo límite (600 s, ~61% recorrido, sin fallos observados hasta ese punto); no se reporta un total global nuevo. La cifra consolidada anterior sigue siendo `213 passed in 23,17 s` del bloque 1.
- **Render:** `quarto` no está disponible en el entorno; no se renderizó el libro.
- **Fuentes contrastadas por capítulo:** cap07 Paninski (NYU PDF), Wu–Poulsen–Mølmer (`10.1103/PhysRevA.80.032319`) y Miller et al. (PMC2677320); cap08 de Wolf (`gs001`), Ozols, MIT OCW y Hellinger; cap09 Braunstein–Caves (`10.1103/PhysRevLett.72.3439`), Barndorff-Nielsen–Gill (`quant-ph/9808009`), Schuld et al. (`10.1103/PhysRevA.99.032331`) y McClean et al. (`10.1038/s41467-018-07090-4`); cap10 Berry (`10.1098/rspa.1984.0543`), revisión de fase geométrica (IAS/IntechOpen), Van Oosterom–Strackee (`10.1109/TBME.1983.325207`) y PRX Quantum 2, 030333; cap11 Huelga (`10.1103/PhysRevLett.79.3865`), Degen (`10.1103/RevModPhys.89.035002`), Kaubruegger (`10.1103/PhysRevX.11.041045`) y Bothwell (`10.1038/s41586-021-04349-7`); cap12 Argüelles–Jones (`10.1103/PhysRevResearch.1.033176`), virtual-Z/DAE 68, PDG 2024 (`10.1103/PhysRevD.110.030001`), NuFIT 6.0 (`10.1007/JHEP12(2024)216`), Nogueira et al. (`1610.05388`) y Chundawat–Li (`2602.07991`).
- **Resultados destacados:** cap10 fija `∬F = 1.295010823 = −γ`, `no gain` y `robustez unverified`; cap11 acota `no gain` de GHZ a desfase markoviano independiente + paridad y documenta `T*=T2/2`; cap12 aclara unidades de `L_osc`, deriva `I_F(θ)`/`I_F(φ)`, la regla de la cadena a `Δm²` y el alcance del filtro de sector.
- **Estado Git:** `main` con los seis commits de revisión del bloque 2 más el cierre; `.venv_clean/` permanece sin trackear y no se incluyó en ningún commit.
