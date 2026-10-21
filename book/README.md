# The qang book (Quarto)

Two Quarto books with the same chapter numbering:

- `book/es/` — *Computación, información y física cuántica en el espacio qang* (Spanish)
- `book/en/` — *Quantum Computing, Information and Physics in qang Space* (English)

The master indices are `INDICE_MAESTRO_ES.md` and `INDICE_MAESTRO_EN.md`. `tools/make_book_skeleton.py` rebuilds each
`_quarto.yml` from them and creates any missing chapter as a draft; it never overwrites a chapter that exists.

## Render and publish on Posit Cloud

1. New Project → **New Project from Git Repository** → `https://github.com/Viny2030/qang`.
2. In the Terminal tab (Quarto is preinstalled):

   ```bash
   cd book/es && quarto render
   cd ../en && quarto render
   ```

   The sites are written to `book/es/_book/` and `book/en/_book/`; open `index.html` with **View in Web Browser**.
3. Publish online, for example to Quarto Pub (free; asks you to log in once):

   ```bash
   cd book/es && quarto publish quarto-pub
   ```

   or to GitHub Pages with `quarto publish gh-pages` from the repository root of a clone.

The chapters contain no executable cells (code is shown, not run), so rendering needs only Quarto, not Python.
