"""Zenodo metadata for the qang preprints and the software (docs/zenodo/).

Reads each article's title and abstract from manuscript/<dir>/main.tex, turns
the LaTeX into plain text, and writes one Zenodo deposit JSON per record plus
docs/zenodo/README.md with the upload steps. Rerun after changing an abstract.

python tools/make_zenodo_pack.py
"""

import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "zenodo")
CREATOR = [{"name": "Monteverde, Vicente Humberto", "affiliation": "Universidad del Museo Social Argentino (UMSA), Argentina",
            "orcid": "0000-0001-8884-4811"}]
REPO = "https://github.com/Viny2030/qang"
SOFTWARE_DOI = "10.5281/zenodo.22832150"  # the qang paper; the software record DOI is filled in after upload

PAPERS = [
    # key, tex dir, pdf (EN), pdf (ES or None), existing DOI (new version) or None, keywords
    ("qang", ".", "preprint_qang_revisado.pdf", None, "10.5281/zenodo.22832150",
     ["qang", "polar bias", "Z-basis metrics", "parameterized quantum circuits", "barren plateaus", "entropy"]),
    ("witness", "witness", "preprint_testigo_simetria.pdf", None, "10.5281/zenodo.23036963",
     ["symmetry verification", "Hamming weight", "error mitigation", "amplitude damping", "VQE", "qang"]),
    ("leakage", "leakage", "qang_fuga.pdf", "qang_fuga_es.pdf", None,
     ["leakage", "erasure qubits", "three-level readout", "T1-aware decoding", "quantum error correction", "qang"]),
    ("formulation", "formulation", "qang_formulation.pdf", "qang_teoria_es.pdf", None,
     ["Pauli representation", "quantum gates", "quantum algorithms", "Hamming weight", "one-tangle",
      "Meyer-Wallach entanglement", "Bloch radius", "qang"]),
    ("qml", "qml", "qang_qml.pdf", "qang_qml_es.pdf", None,
     ["quantum neural networks", "quantum machine learning", "symmetry filter", "T1 noise", "pre-registration",
      "reconfigurable beam splitter", "qang"]),
    ("radius", "radius", "qang_radius.pdf", "qang_radio_es.pdf", None,
     ["Bloch radius", "one-tangle", "Meyer-Wallach entanglement", "symmetry verification", "Hamming weight",
      "amplitude damping", "pre-registration", "qang"]),
    ("optics", "optics", "qang_optics.pdf", "qang_optics_es.pdf", None,
     ["polarization", "Stokes parameters", "Mueller matrices", "Pancharatnam phase", "Berry phase", "qang"]),
    ("crypto", "crypto", "qang_criptografia.pdf", "qang_criptografia_es.pdf", None,
     ["BB84", "quantum key distribution", "eavesdropping detection", "certified randomness", "qang"]),
    ("rbm_comment", "rbm_comment", "qang_rbm_comment.pdf", None, None,
     ["restricted Boltzmann machines", "mutual information", "comment", "qang"]),
    ("battery_network", "battery_network", "qang_baterias_redes.pdf", None, None,
     ["quantum batteries", "entanglement distillation", "quantum networks", "qang"]),
]

REPL = [
    (r"\\(?:big|Big|bigg|left|right)(?![a-zA-Z])", ""), (r"\\'e", "é"), (r"\\'\{e\}", "é"),
    (r"\\sqrt\{([^}]*)\}", r"√(\1)"), (r"\\mathbb\{Z\}", "ℤ"), (r"\\pm", "±"),
    (r"\\qg([XYZS])(?![a-zA-Z])", r"qg_\1"),
    (r"\\(alpha)(?![a-zA-Z])", "α"), (r"\\(beta)(?![a-zA-Z])", "β"), (r"\\(eta)(?![a-zA-Z])", "η"),
    (r"\\(omega)(?![a-zA-Z])", "ω"), (r"\\(lambda)(?![a-zA-Z])", "λ"), (r"\\(mu)(?![a-zA-Z])", "μ"),
    (r"\\le(?![a-zA-Z])", "≤"), (r"\\ge(?![a-zA-Z])", "≥"),
    (r"\\q\{([^}]*)\}", r"qg_\1"), (r"\\qZ", "qg_Z"), (r"\\qX", "qg_X"), (r"\\qY", "qg_Y"),
    (r"~?\\cite[pt]?\{[^}]*\}", ""), (r"\\(?:texttt|emph|textbf|mathrm|text)\{([^}]*)\}", r"\1"),
    (r"\\times\s*10\^\{([^}]*)\}", r"×10^\1"), (r"\\times", "×"), (r"\\sim", "~"), (r"\\pi", "π"), (r"\\gamma", "γ"), (r"\\langle", "⟨"), (r"\\rangle", "⟩"),
    (r"\\sigma_z", "σ_z"), (r"\\dagger", "†"), (r"\\rho", "ρ"), (r"\\varphi", "φ"), (r"\\theta", "θ"),
    (r"\\,", " "), (r"\\%", "%"), (r"---", "—"), (r"--", "–"), (r"``", "“"), (r"''", "”"), (r"~", " "),
    (r"\^\{([^{}]{2,})\}", r"^(\1)"), (r"_\{([^{}]{2,})\}", r"_(\1)"),
    (r"\$", ""), (r"\\\\", " "), (r"[{}]", ""), (r"\\([a-zA-Z]+)", r"\1"), (r"\s+", " "),
]


def plain(tex):
    for a, b in REPL:
        tex = re.sub(a, b, tex)
    return tex.strip()


def read(tex_dir):
    s = open(os.path.join(ROOT, "manuscript", tex_dir, "main.tex"), encoding="utf-8").read()
    title = re.search(r"\\title\{(.*?)\}\s*\n\\author", s, re.S).group(1)
    title = re.sub(r"\\\\\s*", " ", title)
    title = re.sub(r"\\large.*$", "", title, flags=re.S)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S).group(1)
    return plain(title), plain(abstract)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for key, d, pdf, pdf_es, doi, kw in PAPERS:
        title, abstract = read(d)
        files = [f"manuscript/{pdf}"] + ([f"manuscript/{pdf_es}"] if pdf_es else [])
        meta = {
            "upload_type": "publication", "publication_type": "preprint", "title": title, "creators": CREATOR,
            "description": abstract, "keywords": kw, "license": "cc-by-4.0", "access_right": "open",
            "language": "eng", "version": "v1" if doi is None else "v2",
            "related_identifiers": [
                {"identifier": REPO, "relation": "isSupplementedBy", "resource_type": "software"},
                {"identifier": SOFTWARE_DOI, "relation": "references", "scheme": "doi"},
            ],
            "notes": "Every numerical claim is reproduced by the open-source library qang (pip install qang, version "
                     "0.6.13) and pinned by its tests; predictions were committed before each run.",
        }
        if key == "qang":
            meta["related_identifiers"] = [{"identifier": REPO, "relation": "isSupplementedBy", "resource_type": "software"}]
        json.dump({"metadata": meta, "files": files, "new_version_of": doi}, open(os.path.join(OUT, f"{key}.json"), "w",
                  encoding="utf-8"), ensure_ascii=False, indent=1)
        rows.append((key, title, doi, files))
    soft = {
        "upload_type": "software", "title": "qang: the Qang (qg) unit and metric for parametric quantum circuit design",
        "creators": CREATOR, "license": "MIT", "access_right": "open", "version": "0.6.13",
        "description": "Python library for the qang (qg) framework: qubits described by measurement-level quantities "
                       "(the polar bias qg_Z = ⟨σ_z⟩ and its companions), native qg-parameterized gates for Qiskit, Cirq "
                       "and PennyLane, few-shot qg estimators, the qg symmetry witness and filter, weight-conserving QNN "
                       "classifiers read with and without the filter, the qg formulation of 15 gates and 14 algorithms "
                       "with the local Bloch radius, and polarization optics and geometric phase in qg units. 1236 tests.",
        "keywords": ["quantum computing", "qang", "polar bias", "symmetry verification", "error mitigation", "qiskit",
                     "cirq", "pennylane", "quantum machine learning"],
        "related_identifiers": [{"identifier": REPO, "relation": "isSupplementTo", "resource_type": "software"},
                                {"identifier": "https://pypi.org/project/qang/0.6.13/", "relation": "isIdenticalTo"}],
    }
    json.dump(soft, open(os.path.join(ROOT, ".zenodo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lines = ["# Zenodo metadata pack", "",
             "One JSON per record (`metadata` is the Zenodo deposit metadata; `files` are the PDFs to upload;",
             "`new_version_of` is the DOI of an existing record to version, or null for a new record).",
             "`.zenodo.json` at the repository root is the software record, read by the GitHub–Zenodo integration",
             "when a GitHub release is made. Generated by `tools/make_zenodo_pack.py`.", "",
             "| record | title | action | files |", "|---|---|---|---|"]
    for key, title, doi, files in rows:
        act = f"new version of {doi}" if doi else "new record"
        lines.append(f"| `{key}.json` | {title} | {act} | {', '.join(files)} |")
    lines += ["| `.zenodo.json` | qang software 0.6.13 | GitHub release (or upload the wheel and sdist) | dist/ |", "",
              "Upload steps (zenodo.org, logged in with ORCID):", "",
              "1. New record: Upload → New upload → drag the PDFs → copy title, description, keywords, license from the",
              "   JSON → Publish. Write the DOI back into `manuscript/refs.bib` and the README.",
              "2. New version: open the existing record → New version → replace the PDF → update the fields → Publish.",
              "3. Software: enable the repository at zenodo.org/account/settings/github and make a GitHub release",
              "   `v0.6.13`; Zenodo archives it with `.zenodo.json`.", ""]
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", len(rows), "records")


if __name__ == "__main__":
    main()
