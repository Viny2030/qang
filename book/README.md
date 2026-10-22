# The qang book (Quarto)

Two Quarto books with the same chapter numbering:

- `book/es/` — *Computación, información y física cuántica en el espacio qang* (Spanish)
- `book/en/` — *Quantum Computing, Information and Physics in qang Space* (English)

The master indices are `INDICE_MAESTRO_ES.md` and `INDICE_MAESTRO_EN.md`. `tools/make_book_skeleton.py` rebuilds each
`_quarto.yml` from them and creates any missing chapter as a draft; it never overwrites a chapter that exists.

## Publishing (automatic)

`.github/workflows/book.yml` runs on every push to `main` that touches `book/`: it renders both books as HTML, PDF and
Word with Quarto and TinyTeX, adds the bilingual landing page `book/index.html`, and replaces the `gh-pages` branch with
that site only (`force_orphan`). The site is https://viny2030.github.io/qang/ (Spanish at `es/`, English at `en/`);
each book has a download button for PDF and Word (`qang-libro-es.pdf/.docx`, `qang-book-en.pdf/.docx`).

Write the chapters in `main` only. Never commit to `gh-pages` by hand: the workflow overwrites it, and a manual commit
can publish files that must stay private.

## Rendering locally or on Posit Cloud

On Posit Cloud (Quarto is preinstalled), after cloning the repository:

```bash
cd book/es && quarto render
```

The site is written to `book/es/_book/`. The PDF needs a LaTeX installation (`quarto install tinytex`).

The chapters contain no executable cells (code is shown, not run), so rendering needs Quarto only, not Python.
