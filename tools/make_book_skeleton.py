"""
Builds the Quarto book skeleton from book/INDICE_MAESTRO_ES.md and book/INDICE_MAESTRO_EN.md:
book/es/_quarto.yml, book/en/_quarto.yml and one .qmd per chapter and appendix. A chapter file that already
exists is never overwritten, so written chapters are safe; only _quarto.yml is regenerated.

python tools/make_book_skeleton.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.join(HERE, "..", "book")

LANG = {
    "es": {"index": "INDICE_MAESTRO_ES.md", "title": "Computación, información y física cuántica en el espacio qang",
           "subtitle": "Un tratado sobre la unidad qg, el filtro de simetría y sus aplicaciones",
           "part_re": r"^## PARTE ([IVX]+)\. (.+)$", "chap_re": r"^### Capítulo (\d+)\. (.+)$",
           "app_head": "## Apéndices", "draft": "Capítulo en preparación. Contenido previsto:",
           "app_draft": "Apéndice en preparación.", "preface": "Prefacio", "lang": "es",
           "other": "English edition", "other_url": "../en/"},
    "en": {"index": "INDICE_MAESTRO_EN.md", "title": "Quantum Computing, Information and Physics in qang Space",
           "subtitle": "A treatise on the qg unit, the symmetry filter and their applications",
           "part_re": r"^## PART ([IVX]+)\. (.+)$", "chap_re": r"^### Chapter (\d+)\. (.+)$",
           "app_head": "## Appendices", "draft": "Chapter in preparation. Planned contents:",
           "app_draft": "Appendix in preparation.", "preface": "Preface", "lang": "en",
           "other": "Edición en español", "other_url": "../es/"},
}

MARKS = ("🔬", "🆕", "🧪", "⚠️")


EDITORIAL = (r"\s*\((?:el índice original[^)]*|the original index[^)]*)\)",
             r"\.?\s*El índice original citaba[^.]*\.?[^—]*", r"\.?\s*The original index cited[^.]*\.?[^—]*",
             r"\s*\((?:2 de 3 modelos simulados; hallazgo post hoc, no pre-registrado|2 of 3 simulated models; a post hoc "
             r"finding, not pre-registered)\)")


def clean(text):
    # an editorial note (a figure to verify, an aside about the source index) is cut at its marker
    if "🔬" in text and not text.lstrip().startswith("🔬"):
        text = text[:text.index("🔬")].rstrip(" .;")
    for pat in EDITORIAL:
        text = re.sub(pat, "", text)
    for m in MARKS:
        text = text.replace(m, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse(path, cfg):
    parts, apps, cur_part, cur_chap, in_app = [], [], None, None, False
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith(cfg["app_head"]):
            in_app = True
            continue
        if in_app:
            m = re.match(r"^([A-F])\. (.+)$", line)
            if m:
                apps.append((m.group(1), clean(m.group(2))))
            continue
        m = re.match(cfg["part_re"], line)
        if m:
            cur_part = {"num": m.group(1), "title": clean(m.group(2)), "chapters": []}
            parts.append(cur_part)
            continue
        m = re.match(cfg["chap_re"], line)
        if m and cur_part is not None:
            cur_chap = {"num": int(m.group(1)), "title": clean(m.group(2)), "items": []}
            cur_part["chapters"].append(cur_chap)
            continue
        m = re.match(r"^(\d+\.\d+) (.+)$", line)
        if m and cur_chap is not None:
            cur_chap["items"].append((m.group(1), clean(m.group(2))))
    return parts, apps


def chapter_title(title):
    # drop "(not in either index)" style remarks from the heading
    return re.sub(r"\s*\((?:no figuraba en los índices|not in either index)\)", "", title)


def write_if_missing(path, text):
    if os.path.exists(path):
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return True


def build(lang):
    cfg = LANG[lang]
    parts, apps = parse(os.path.join(BOOK, cfg["index"]), cfg)
    root = os.path.join(BOOK, lang)
    os.makedirs(root, exist_ok=True)
    created = 0
    yml_parts = []
    for p in parts:
        files = []
        for c in p["chapters"]:
            fn = f"cap{c['num']:02d}.qmd"
            files.append(fn)
            body = "\n".join(f"- **{n}** {t}" for n, t in c["items"])
            text = (f"# {chapter_title(c['title'])}\n\n::: {{.callout-note}}\n{cfg['draft']}\n:::\n\n{body}\n")
            created += write_if_missing(os.path.join(root, fn), text)
        yml_parts.append((p, files))
    app_files = []
    for letter, title in apps:
        fn = f"apendice_{letter.lower()}.qmd" if lang == "es" else f"appendix_{letter.lower()}.qmd"
        app_files.append(fn)
        created += write_if_missing(os.path.join(root, fn),
                                    f"# {title}\n\n::: {{.callout-note}}\n{cfg['app_draft']}\n:::\n")
    created += write_if_missing(os.path.join(root, "index.qmd"), f"# {cfg['preface']} {{.unnumbered}}\n")
    created += write_if_missing(os.path.join(root, "references.bib"), "")
    lines = ["project:", "  type: book", "  output-dir: _book", "", "lang: " + cfg["lang"], "",
             "book:", f'  title: "{cfg["title"]}"', f'  subtitle: "{cfg["subtitle"]}"',
             '  author: "Vicente Humberto Monteverde"', '  date: "2026"',
             "  repo-url: https://github.com/Viny2030/qang", "  repo-subdir: book/" + lang,
             "  repo-actions: [issue]", "  search: true",
             "  sidebar:", "    tools:", f'      - text: "{cfg["other"]}"', f"        href: {cfg['other_url']}",
             "  chapters:", "    - index.qmd"]
    for p, files in yml_parts:
        lines.append(f'    - part: "{p["num"]}. {p["title"]}"')
        lines.append("      chapters:")
        lines += [f"        - {f}" for f in files]
    lines.append("  appendices:")
    lines += [f"    - {f}" for f in app_files]
    lines += ["", "bibliography: references.bib", "", "format:", "  html:", "    theme: cosmo", "    toc: true",
              "    number-sections: true", "    html-math-method: mathjax", "    code-copy: true", ""]
    with open(os.path.join(root, "_quarto.yml"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    n_ch = sum(len(f) for _, f in yml_parts)
    print(f"{lang}: {len(parts)} parts, {n_ch} chapters, {len(app_files)} appendices, {created} new files")


if __name__ == "__main__":
    for lang in ("es", "en"):
        build(lang)
