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
| 5 | `book/es/cap05.qmd` | [ ] | [ ] | [ ] | [ ] | 22 |
| 6 | `book/es/cap06.qmd` | [ ] | [ ] | [ ] | [ ] | 25 |
| 7 | Reporte final y estado de Git | [ ] | — | — | [ ] | 10 |

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
- [ ] **cap05** — commit `docs: revisión teórica y validación cap 5`.
- [ ] **cap06** — commit `docs: revisión teórica y validación cap 6`.
- [ ] **Reporte final**.
