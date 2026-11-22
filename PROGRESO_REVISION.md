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

---

# Bloque 3 — revisión teórica y validación — capítulos `qjax13`–`qjax18`

**Repositorio:** `C:\Users\ASUS\qang_repo` · **Rama:** `main` (sin crear ramas nuevas).
**Alcance acordado:** capítulos `book/es/qjax13.qmd`–`book/es/qjax18.qmd` (equivalentes a cap13–cap18 en la numeración del usuario).
**Límite de tiempo:** 3 horas (180 min). **Inicio:** 2026-11-21.

## Checklist por capítulo (bloque 3)

| # | Capítulo | Revisión teórica | Fuentes en línea | Validación código | `[x]` y commit | Min |
|---|----------|------------------|------------------|-------------------|----------------|-----|
| 1 | `book/es/qjax13.qmd` | [x] | [x] | [x] | [x] | 30 |
| 2 | `book/es/qjax14.qmd` | [x] | [x] | [x] | [x] | 30 |
| 3 | `book/es/qjax15.qmd` | [x] | [x] | [x] | [x] | 30 |
| 4 | `book/es/qjax16.qmd` | [x] | [x] | [x] | [x] | 30 |
| 5 | `book/es/qjax17.qmd` | [x] | [x] | [x] | [x] | 30 |
| 6 | `book/es/qjax18.qmd` | [x] | [x] | [x] | [x] | 30 |
| 7 | Reporte final y estado de Git | [x] | — | — | [x] | 10 |

**Total estimado:** 180 min.

## Reglas de ejecución (bloque 3)

1. Leer el capítulo completo antes de tocar nada; conservar **toda** la información técnica original.
2. Contrastar cada afirmación clave con literatura/documentación en línea; anotar la fuente en el reporte.
3. Ejecutar el script `tools/verify_capNN.py` correspondiente y los tests unitarios relacionados.
4. Marcar `[x]` en este archivo y hacer `git add` + `git commit` **antes** de pasar al siguiente capítulo.
5. No crear ramas, no borrar archivos, no añadir `.venv_clean/` ni otros artefactos al commit.
6. Si se agota el tiempo, entregar el reporte con lo completado y lo pendiente.

## Registro de avance (bloque 3)

- [x] **Setup y descubrimiento** (hecho: ruta real del libro, rama, inventario de capítulos, verificadores y tests).
- [x] **qjax13** (cap13) — revisión, fuentes, `verify_cap13.py`, `tests/test_qneutrino_matter.py` (2 passed) y commit.
- [x] **qjax14** (cap14) — revisión, fuentes, `verify_cap14.py`, `tests/test_qmajorana.py` (3 passed), sección 14.9 y commit.
- [x] **qjax15** (cap15) — revisión, fuentes, `verify_cap15.py`, `tests/test_qtele.py` (3 passed), sección 15.9 y commit.
- [x] **qjax16** (cap16) — revisión, fuentes, `verify_cap16.py`, `tests/test_qdualrail.py` (1 passed), sección 16.7 y commit.
- [x] **qjax17** (cap17) — revisión, fuentes, `verify_cap17.py`, `tests/test_qctc.py` (2 passed), sección 17.7 y commit.
- [x] **qjax18** (cap18) — revisión, fuentes, `verify_cap18.py`, `tests/test_qscramble.py` (1 passed), sección 18.8 y commit.
- [x] **Reporte final (bloque 3)**.

## Reporte final (bloque 3)

- **Commits:** `558a18c` (cap13), `2249b68` (fallback `\tfrac` para PDF), `ec9f79d` (cap14), `7459d98` (cap15), `05456a2` (cap16), `c463ef6` (cap17), `77c1c53` (reemplazo de `\rm` por `\mathrm`), `2ba3bca` (cap18).
- **Verificadores:** `tools/verify_cap13.py`–`tools/verify_cap18.py` pasan los seis con `verify_capNN: OK`.
- **Pytest del bloque:** `test_qneutrino_matter.py` (2 passed), `test_qmajorana.py` (3 passed), `test_qtele.py` (3 passed), `test_qdualrail.py` (1 passed), `test_qctc.py` (2 passed) y `test_qscramble.py` (1 passed): 12 pruebas dirigidas, todas verde; no se reporta una suite global nueva.
- **Render:** `quarto` no está disponible localmente. El CI detectó tres fallos LaTeX: `\tfrac` sin definir, `\rm` obsoleto en `qjax10`, `qjax11` y `qjax12`, y subíndices `_\triangle` sin llaves en `qjax10`. Quedaron corregidos con `\providecommand{\tfrac}[2]{\frac{#1}{#2}}` en la configuración PDF ES/EN, sustitución de `\rm{...}` por `\mathrm{...}` y llaves en `\Phi_{\triangle}=-\Omega_{\triangle}/2`. Queda pendiente el rerun de CI.
- **Warnings no bloqueantes:** crossrefs `@sec-qjax26`, `@sec-qjax23`, `@sec-qjax34`, `@sec-qjax10` y `@sec-qjax31` sin resolver en el render.
- **Fuentes contrastadas por capítulo:** cap13 oscilaciones en materia y MSW; cap14 0νββ, fases de Majorana y codificación 3+1; cap15 teleportación, cota Horodecki y presupuesto de disparos; cap16 doble carril, borrado por pérdida y atenuación de fibra; cap17 P-CTC/D-CTC, Deutsch y Lloyd; cap18 Hayden–Preskill, Yoshida–Kitaev, OTOC y curva de Page.
- **Resultados destacados:** cap13 cierra el filtro de sector; cap14 distingue fase de Majorana y filtro one-hot; cap15 acota $qg_{ZZ}>333$ m-qg bajo Werner; cap16 separa sesgo de lectura y pérdida; cap17 cuantifica $N\propto1/p_{\text{post}}$ y conservación/dephasing de D-CTC; cap18 verifica $P\cdot F=1/4$ y documenta $qg_S$ como coordenada, no entropía.
- **Estado Git:** `main` con los ocho commits del bloque 3; `.venv_clean/` permanece sin trackear y no se incluyó en ningún commit.

---

# Bloque 4 — revisión teórica y validación — capítulos `qjax19`–`qjax24`

**Repositorio:** `C:\Users\ASUS\qang_repo` · **Rama:** `main` (sin crear ramas nuevas).
**Alcance acordado:** capítulos `book/es/qjax19.qmd`–`book/es/qjax24.qmd` (equivalentes a cap19–cap24 en la numeración del usuario).
**Inicio:** 2026-11-21.

## Checklist por capítulo (bloque 4)

| # | Capítulo | Revisión teórica | Fuentes en línea | Validación código | `[x]` y commit | Min |
|---|----------|------------------|------------------|-------------------|----------------|-----|
| 1 | `book/es/qjax19.qmd` | [x] | [x] | [x] | [x] | 30 |
| 2 | `book/es/qjax20.qmd` | [x] | [x] | [x] | [x] | 30 |
| 3 | `book/es/qjax21.qmd` | [x] | [x] | [x] | [x] | 30 |
| 4 | `book/es/qjax22.qmd` | [x] | [x] | [x] | [x] | 30 |
| 5 | `book/es/qjax23.qmd` | [x] | [x] | [x] | [x] | 30 |
| 6 | `book/es/qjax24.qmd` | [x] | [x] | [x] | [x] | 30 |
| 7 | Reporte final y estado de Git | [x] | - | - | [x] | 10 |

## Reglas de ejecución (bloque 4)

1. Leer el capítulo completo antes de tocar nada; conservar **toda** la información técnica original.
2. Contrastar cada afirmación clave con literatura/documentación en línea; anotar la fuente en el reporte.
3. Ejecutar el script `tools/verify_capNN.py` correspondiente y los tests unitarios relacionados.
4. Marcar `[x]` en este archivo y hacer `git add` + `git commit` **antes** de pasar al siguiente capítulo.
5. No crear ramas, no borrar archivos, no añadir `.venv_clean/` ni otros artefactos al commit.
6. Si se agota el tiempo, entregar el reporte con lo completado y lo pendiente.

## Registro de avance (bloque 4)

- [x] **Setup y descubrimiento** (hecho: capítulo activo, verificador, tests y módulo localizados).
- [x] **qjax19** (cap19) — revisión, fuentes, `verify_cap19.py` (`verify_cap19: OK`), `tests/test_qdemarc.py` (3 passed), sección 19.8 y commit.
- [x] **qjax20** (cap20) — revisión, fuentes, `verify_cap20.py` (`verify_cap20: OK`), `tests/test_qkd.py` (2 passed), sección 20.8 y commit.
- [x] **qjax21** (cap21) — revisión, fuentes, `verify_cap21.py` (`verify_cap21: OK`), `tests/test_qrng.py` (2 passed), sección 21.8 y commit; etiqueta `q=0.999` corregida a cola `0.1%`/central `99.8%`.
- [x] **qjax22** (cap22) — revisión, fuentes, `verify_cap22.py` (`verify_cap22: OK`), `tests/test_qbattery.py` (3 passed), sección 22.7 y commit.
- [x] **qjax23** (cap23) - §23.9 (5 precisiones, arXiv:2405.20237, PRX Quantum 2/040203, PRA 100/010302(R)), `verify_cap23.py` (`verify_cap23: OK` con PYTHONPATH), `tests/test_qweightqnn.py` (3 passed), CHANGELOG/PROGRESO actualizados y commit.
- [x] **qjax24** (cap24). Contraste bibliográfico/precisiones §24.8 (Bonet-Monroig 2018 PRA 98/062339/arXiv:1807.10050; Temme-Bravyi-Gambetta 2017 PRL 119/180509/arXiv:1612.02058; contexto Cai 2023 RMP 95/045005). Validación: `tools/verify_cap24.py` => `verify_cap24: OK`; `pytest tests/test_qsymfilter.py -q` => `1 passed`; `git diff --check` limpio.
- [x] **Reporte final (bloque 4)**. Bloque 4 cerrado (qjax19-qjax24 + reporte).

---

# Bloque 5 - revisión teórica y validación - capítulos `qjax25`-`qjax30`

**Repositorio:** `C:\Users\ASUS\PycharmProjects\qang` · **Rama:** `main` (sin crear ramas nuevas).
**Alcance acordado:** capítulos `book/es/qjax25.qmd`-`book/es/qjax30.qmd`.
**Inicio:** 2026-11-22.

## Checklist por capítulo (bloque 5)

| # | Capítulo | Revisión teórica | Fuentes en línea | Validación código | `[x]` y commit | Min |
|---|----------|------------------|------------------|-------------------|----------------|-----|
| 1 | `book/es/qjax25.qmd` | [x] | [x] | [x] | [x] | 30 |
| 2 | `book/es/qjax26.qmd` | [x] | [x] | [x] | [x] | 30 |
| 3 | `book/es/qjax27.qmd` | [x] | [x] | [x] | [x] | 30 |
| 4 | `book/es/qjax28.qmd` | [ ] | [ ] | [ ] | [ ] | 30 |
| 5 | `book/es/qjax29.qmd` | [ ] | [ ] | [ ] | [ ] | 30 |
| 6 | `book/es/qjax30.qmd` | [ ] | [ ] | [ ] | [ ] | 30 |
| 7 | Reporte final y estado de Git | [ ] | - | - | [ ] | 10 |

## Registro de avance (bloque 5)

- [x] **Setup y descubrimiento** (Bloque 5: qjax25-qjax30, verificadores/tests localizados).
- [x] **qjax25** (cap25). Reestructurado 2026-11-22: eliminada §25.7 anexa; demostraciones integradas en §§25.1-25.3; contraste "Sin qang vs. Con qang" con m-qg/MSE y estatus `exact`/`hardware unverified`; ES/EN actualizados. Validación: `verify_cap25.py` => `verify_cap25: OK`; `pytest tests/test_qecho.py -q` => `3 passed`; `git diff --check` limpio.
- [x] **qjax26** (cap26). §26.7 contraste/precisiones (Havlíček 2019 Nature 567:209 DOI 10.1038/s41586-019-0980-2; Thanasilp 2024 Nat. Commun. 15:5200 DOI 10.1038/s41467-024-49287-w; contexto Schuld-Killoran 2019 PRL 122:040504 DOI 10.1103/PhysRevLett.122.040504). Validación: `tools/verify_cap26.py` => `verify_cap26: OK`; `pytest tests/test_qkernels.py -q` => `2 passed`; `git diff --check` limpio.
- [x] **qjax27** (cap27). §27.7 contraste/precisiones (Childs-Goldstone 2004 PRA 70:022314 DOI 10.1103/PhysRevA.70.022314; Ruskai-Szarek-Werner 2002 LAA 347:159 DOI 10.1016/S0024-3795(01)00547-X; King-Ruskai 2001 IEEE TIT 47:192 DOI 10.1109/18.904522; McArdle 2019 npjQI 5:75 DOI 10.1038/s41534-019-0187-2). Validación: `tools/verify_cap27.py` => `verify_cap27: OK`; `pytest tests/test_qdynamics.py -q` => `3 passed`; `git diff --check` limpio.
- [ ] **qjax28** (cap28).
- [ ] **qjax29** (cap29).
- [ ] **qjax30** (cap30).
- [ ] **Reporte final (bloque 5)**.
