"""Usage: python tools/build_book_bib.py book

Build book/{es,en}/references.bib from DOIs (doi.org content negotiation) plus hand entries."""
import json, re, sys, tempfile, time, urllib.request, pathlib

SCR = pathlib.Path(tempfile.gettempdir())
CACHE = SCR / "qang_book_doi_cache.json"
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

DOIS = {
    # base
    "nielsen2010": "10.1017/CBO9780511976667",
    "wootters1981": "10.1103/PhysRevD.23.357",
    "bartlett1947": "10.2307/3001536",
    "anscombe1948": "10.1093/biomet/35.3-4.246",
    "freeman1950": "10.1214/aoms/1177729756",
    "bloch1946": "10.1103/PhysRev.70.460",
    "born1926": "10.1007/BF01397477",
    "shannon1948": "10.1002/j.1538-7305.1948.tb01338.x",
    "braunstein1994": "10.1103/PhysRevLett.72.3439",
    # gates
    "barenco1995": "10.1103/PhysRevA.52.3457",
    "khaneja2001": "10.1016/S0301-0104(01)00318-4",
    "javadiabhari2024": "10.48550/arXiv.2405.08810",
    "bergholm2018": "10.48550/arXiv.1811.04968",
    # grover
    "householder1958": "10.1145/320941.320947",
    "grover1996": "10.1145/237814.237866",
    "brassard2002": "10.1090/conm/305/05215",
    "suzuki2020": "10.1007/s11128-019-2565-2",
    # statistics
    "jeffreys1946": "10.1098/rspa.1946.0056",
    "wilson1927": "10.1080/01621459.1927.10502953",
    "cirq": "10.5281/zenodo.4062499",
    "mezzadri2007": "10.48550/arXiv.math-ph/0609050",
    "huang2020": "10.1038/s41567-020-0932-7",
    # radius
    "coffman2000": "10.1103/PhysRevA.61.052306",
    "meyer2002": "10.1063/1.1497700",
    "greenberger1988": "10.1016/0375-9601(88)90114-4",
    "englert1996": "10.1103/PhysRevLett.77.2154",
    "maassen1988": "10.1103/PhysRevLett.60.1103",
    "berta2010": "10.1038/nphys1734",
    "deutsch1992": "10.1098/rspa.1992.0167",
    "bernstein1997": "10.1137/S0097539796300921",
    "simon1997": "10.1137/S0097539796298637",
    # optics, phases
    "bornwolf1999": "10.1017/CBO9781139644181",
    "berry1984": "10.1098/rspa.1984.0023",
    "pancharatnam1956": "10.1007/BF03046050",
    "aharonov1987": "10.1103/PhysRevLett.58.1593",
    "samuel1988": "10.1103/PhysRevLett.60.2339",
    # metrology
    "amari1998": "10.1162/089976698300017746",
    "ramsey1950": "10.1103/PhysRev.78.695",
    "giovannetti2004": "10.1126/science.1104149",
    "huelga1997": "10.1103/PhysRevLett.79.3865",
    "matsuzaki2011": "10.1103/PhysRevA.84.012103",
    "chin2012": "10.1103/PhysRevLett.109.233601",
    "correa2015": "10.1103/PhysRevLett.114.220405",
    "chou2010": "10.1126/science.1192720",
    # neutrinos
    "maki1962": "10.1143/PTP.28.870",
    "pdg2024": "10.1103/PhysRevD.110.030001",
    "esteban2020": "10.1007/JHEP09(2020)178",
    "wolfenstein1978": "10.1103/PhysRevD.17.2369",
    "duan2010": "10.1146/annurev.nucl.012809.104524",
    "majorana1937": "10.1007/BF02961314",
    "schechter1980": "10.1103/PhysRevD.22.2227",
    "bilenky1980": "10.1016/0370-2693(80)90927-2",
    "dolinski2019": "10.1146/annurev-nucl-101918-023407",
    "aguilar2001": "10.1103/PhysRevD.64.112007",
    # teleportation, networks
    "bennett1993": "10.1103/PhysRevLett.70.1895",
    "yaghoobianzadeh2026": "10.20935/AcadQuant8460",
    "bell1964": "10.1103/PhysicsPhysiqueFizika.1.195",
    "knill2001": "10.1038/35051009",
    "zukowski1993": "10.1103/PhysRevLett.71.4287",
    "briegel1998": "10.1103/PhysRevLett.81.5932",
    # CTC, black holes
    "deutsch1991": "10.1103/PhysRevD.44.3197",
    "lloyd2011": "10.1103/PhysRevLett.106.040403",
    "svetlichny2011": "10.1007/s10773-011-0973-x",
    "hawking1976": "10.1103/PhysRevD.14.2460",
    "susskind1993": "10.1103/PhysRevD.48.3743",
    "page1993": "10.1103/PhysRevLett.71.3743",
    "hayden2007": "10.1088/1126-6708/2007/09/120",
    "yoshida2017": "10.48550/arXiv.1710.03363",
    "gao2017": "10.1007/JHEP12(2017)151",
    "maldacena2017": "10.1002/prop.201700034",
    "maldacena2013": "10.1002/prop.201300020",
    "jafferis2022": "10.1038/s41586-022-05424-3",
    "kobrin2023": "10.48550/arXiv.2302.07897",
    # crypto, randomness
    "bennett2014": "10.1016/j.tcs.2014.05.025",
    "shor1997": "10.1137/S0097539795293172",
    "kitaev1995": "10.48550/arXiv.quant-ph/9511026",
    "fips203": "10.6028/NIST.FIPS.203",
    "fips204": "10.6028/NIST.FIPS.204",
    "fips205": "10.6028/NIST.FIPS.205",
    "nistir8545": "10.6028/NIST.IR.8545",
    "shor2000": "10.1103/PhysRevLett.85.441",
    "tomamichel2012": "10.1038/ncomms1631",
    "konig2009": "10.1109/TIT.2009.2025545",
    "herrero2017": "10.1103/RevModPhys.89.015004",
    "ma2016": "10.1038/npjqi.2016.21",
    # batteries
    "pusz1978": "10.1007/BF01614224",
    "allahverdyan2004": "10.1209/epl/i2004-10101-2",
    "alicki2013": "10.1103/PhysRevE.87.042123",
    "campaioli2024": "10.1103/RevModPhys.96.031001",
    # QML
    "landman2022": "10.22331/q-2022-12-22-881",
    "havlicek2019": "10.1038/s41586-019-0980-2",
    "schuld2019": "10.1103/PhysRevLett.122.040504",
    "low2017": "10.1103/PhysRevLett.118.010501",
    "martyn2021": "10.1103/PRXQuantum.2.040203",
    "cerezo2023": "10.48550/arXiv.2312.09121",
    "bonetmonroig2018": "10.1103/PhysRevA.98.062339",
    "mcardle2019": "10.1103/PhysRevLett.122.180501",
    "temme2017": "10.1103/PhysRevLett.119.180509",
    "hahn1950": "10.1103/PhysRev.80.580",
    "bravyi2021": "10.1103/PhysRevA.103.042605",
    "stokes2020": "10.22331/q-2020-05-25-269",
    "mcclean2018": "10.1038/s41467-018-07090-4",
    "cerezo2021": "10.1038/s41467-021-21728-w",
    "levenberg1944": "10.1090/qam/10666",
    "marquardt1963": "10.1137/0111030",
    # many-body, chemistry
    "trotter1959": "10.1090/S0002-9939-1959-0108732-6",
    "suzuki1991": "10.1063/1.529425",
    "lloyd1996": "10.1126/science.273.5278.1073",
    "aharonov1993": "10.1103/PhysRevA.48.1687",
    "mclachlan1964": "10.1080/00268976400100041",
    "yuan2019": "10.22331/q-2019-10-07-191",
    "lindblad1976": "10.1007/BF01608499",
    "gorini1976": "10.1063/1.522979",
    "hubbard1963": "10.1098/rspa.1963.0204",
    "baumgratz2014": "10.1103/PhysRevLett.113.140401",
    "singh2026": "10.20935/AcadQuant8243",
    "lloyd2014": "10.1038/nphys3029",
    "jordan1928": "10.1007/BF01331938",
    "peruzzo2014": "10.1038/ncomms5213",
    "omalley2016": "10.1103/PhysRevX.6.031007",
    "kandala2017": "10.1038/nature23879",
    "buhrman2001": "10.1103/PhysRevLett.87.167902",
    "farhi2014": "10.48550/arXiv.1411.4028",
    "hadfield2019": "10.3390/a12020034",
    "sun2018": "10.1002/wcms.1340",
    # error correction, characterization
    "shor1995": "10.1103/PhysRevA.52.R2493",
    "leung1997": "10.1103/PhysRevA.56.2567",
    "kitaev2003": "10.1016/S0003-4916(02)00018-0",
    "dennis2002": "10.1063/1.1499754",
    "fowler2012": "10.1103/PhysRevA.86.032324",
    "wu2022": "10.1038/s41467-022-32094-6",
    "peng2020": "10.1103/PhysRevLett.125.150504",
    "magesan2011": "10.1103/PhysRevLett.106.180504",
    "krantz2019": "10.1063/1.5089550",
    "arute2019": "10.1038/s41586-019-1666-5",
    "cross2019": "10.1103/PhysRevA.100.032328",
    "hein2004": "10.1103/PhysRevA.69.062311",
    # method, hardware
    "nosek2018": "10.1073/pnas.1708274114",
    "abdurakhimov2024": "10.48550/arXiv.2408.12433",
    # nano
    "landauer1957": "10.1147/rd.13.0223",
    "buttiker1986": "10.1103/PhysRevLett.57.1761",
    "blanter2000": "10.1016/S0370-1573(99)00123-4",
    "fano1947": "10.1103/PhysRev.72.26",
    "datta1995": "10.1017/CBO9780511805776",
    "beenakker1991": "10.1103/PhysRevB.44.1646",
    "hanson2007": "10.1103/RevModPhys.79.1217",
    "elzerman2004": "10.1038/nature02693",
    "doherty2013": "10.1016/j.physrep.2013.02.001",
    "dreau2011": "10.1103/PhysRevB.84.195204",
    "monteverde2026": "10.5281/zenodo.22832150",
}

HAND = r"""
@book{cramer1946,
  title = {Mathematical Methods of Statistics},
  author = {Cram{\'e}r, Harald},
  publisher = {Princeton University Press},
  address = {Princeton},
  year = {1946}
}

@article{rao1945,
  title = {Information and the accuracy attainable in the estimation of statistical parameters},
  author = {Rao, C. Radhakrishna},
  journal = {Bulletin of the Calcutta Mathematical Society},
  volume = {37},
  pages = {81--91},
  year = {1945}
}

@article{bhattacharyya1943,
  title = {On a measure of divergence between two statistical populations defined by their probability distributions},
  author = {Bhattacharyya, Anil Kumar},
  journal = {Bulletin of the Calcutta Mathematical Society},
  volume = {35},
  pages = {99--109},
  year = {1943}
}

@article{riesz1907,
  title = {Sur une esp{\`e}ce de g{\'e}om{\'e}trie analytique des syst{\`e}mes de fonctions sommables},
  author = {Riesz, Frigyes},
  journal = {Comptes rendus de l'Acad{\'e}mie des sciences},
  volume = {144},
  pages = {1409--1411},
  year = {1907}
}

@article{frechet1907,
  title = {Sur les ensembles de fonctions et les op{\'e}rations lin{\'e}aires},
  author = {Fr{\'e}chet, Maurice},
  journal = {Comptes rendus de l'Acad{\'e}mie des sciences},
  volume = {144},
  pages = {1414--1416},
  year = {1907}
}

@article{stokes1852,
  title = {On the composition and resolution of streams of polarized light from different sources},
  author = {Stokes, George Gabriel},
  journal = {Transactions of the Cambridge Philosophical Society},
  volume = {9},
  pages = {399--416},
  year = {1852}
}

@article{pontecorvo1968,
  title = {Neutrino experiments and the problem of conservation of leptonic charge},
  author = {Pontecorvo, Bruno},
  journal = {Soviet Physics JETP},
  volume = {26},
  pages = {984--988},
  year = {1968}
}

@article{mikheyev1985,
  title = {Resonance amplification of oscillations in matter and spectroscopy of solar neutrinos},
  author = {Mikheyev, Stanislav P. and Smirnov, Alexei Yu.},
  journal = {Soviet Journal of Nuclear Physics},
  volume = {42},
  pages = {913--917},
  year = {1985}
}

@book{helstrom1976,
  title = {Quantum Detection and Estimation Theory},
  author = {Helstrom, Carl W.},
  publisher = {Academic Press},
  address = {New York},
  year = {1976}
}

@book{popper1959,
  title = {The Logic of Scientific Discovery},
  author = {Popper, Karl R.},
  publisher = {Hutchinson},
  address = {London},
  year = {1959}
}

@misc{jax2018,
  title = {{JAX}: composable transformations of {Python+NumPy} programs},
  author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao},
  howpublished = {\url{https://github.com/jax-ml/jax}},
  year = {2018}
}

@misc{iqmresonance,
  title = {{IQM Resonance}: cloud access to {IQM} quantum computers},
  author = {{IQM Quantum Computers}},
  howpublished = {\url{https://resonance.meetiqm.com}},
  note = {Accessed 2026-10},
  year = {2026}
}

@misc{braket,
  title = {Amazon Braket Python SDK},
  author = {{Amazon Web Services}},
  howpublished = {\url{https://github.com/amazon-braket/amazon-braket-sdk-python}},
  year = {2020}
}

@misc{qangrepo,
  title = {qang: the qang (qg) unit, the symmetry filter and their applications. Source code and research notes},
  author = {Monteverde, Vicente Humberto},
  howpublished = {\url{https://github.com/Viny2030/qang}},
  year = {2026}
}
"""


def fetch(doi):
    if doi in cache:
        return cache[doi]
    req = urllib.request.Request("https://doi.org/" + doi, headers={"Accept": "application/x-bibtex; charset=utf-8"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                txt = r.read().decode("utf-8").strip()
            if txt.startswith("@"):
                cache[doi] = txt
                CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
                return txt
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2)
    print("FAILED", doi, err, file=sys.stderr)
    return None


KEEP = ["author", "title", "journal", "booktitle", "institution", "type", "number", "volume", "pages",
        "edition", "publisher", "address", "year", "doi"]


def parse(txt):
    m = re.match(r"\s*@(\w+)\s*\{\s*([^,\s]*)\s*,", txt)
    typ, i, fields = m.group(1).lower(), m.end(), {}
    while True:
        m = re.compile(r"\s*(\w+)\s*=\s*").match(txt, i)
        if not m:
            break
        name, i = m.group(1).lower(), m.end()
        if txt[i] == "{":
            depth, j = 0, i
            while True:
                depth += {"{": 1, "}": -1}.get(txt[j], 0)
                if depth == 0:
                    break
                j += 1
            val, i = txt[i + 1:j], j + 1
        else:
            m = re.compile(r"[^,}\s]+").match(txt, i)
            val, i = m.group(0), m.end()
        fields[name] = val
        m = re.compile(r"\s*,").match(txt, i)
        if not m:
            break
        i = m.end()
    return typ, fields


def clean(key, txt):
    typ, f = parse(txt)
    for k, v in list(f.items()):
        v = re.sub(r"<[^>]+>", "", v)
        f[k] = re.sub(r"\s+", " ", v).strip()
    if "pages" in f:
        f["pages"] = f["pages"].replace("–", "--")
    return key, typ, f


OVERRIDES = {
    "nielsen2010": {"title": "Quantum Computation and Quantum Information", "year": "2010",
                    "publisher": "Cambridge University Press", "edition": "10th anniversary"},
    "born1926": {"title": "Zur Quantenmechanik der Sto{\\ss}vorg{\\\"a}nge", "journal": "Zeitschrift f{\\\"u}r Physik"},
    "jordan1928": {"title": "{\\\"U}ber das Paulische {\\\"A}quivalenzverbot", "journal": "Zeitschrift f{\\\"u}r Physik"},
    "anscombe1948": {"title": "The transformation of Poisson, binomial and negative-binomial data",
                     "author": "Anscombe, F. J."},
    "clopper1934": {"title": "The use of confidence or fiducial limits illustrated in the case of the binomial",
                    "author": "Clopper, C. J. and Pearson, E. S."},
    "aguilar2001": {"title": "Evidence for neutrino oscillations from the observation of $\\bar\\nu_e$ appearance in a $\\bar\\nu_\\mu$ beam",
                    "author": "Aguilar, A. and others and {LSND Collaboration}"},
    "campaioli2024": {"title": "Colloquium: Quantum batteries"},
    "gorini1976": {"title": "Completely positive dynamical semigroups of {N}-level systems"},
    "sun2018": {"title": "{PySCF}: the {Python}-based simulations of chemistry framework", "year": "2018"},
    "fips203": {"title": "Module-Lattice-Based Key-Encapsulation Mechanism Standard", "author": "{National Institute of Standards and Technology}",
                "number": "FIPS 203", "type": "Federal Information Processing Standards Publication"},
    "fips204": {"title": "Module-Lattice-Based Digital Signature Standard", "author": "{National Institute of Standards and Technology}",
                "number": "FIPS 204", "type": "Federal Information Processing Standards Publication"},
    "fips205": {"title": "Stateless Hash-Based Digital Signature Standard", "author": "{National Institute of Standards and Technology}",
                "number": "FIPS 205", "type": "Federal Information Processing Standards Publication"},
    "zukowski1993": {"title": "``Event-ready-detectors'' {Bell} experiment via entanglement swapping"},
    "cirq": {"author": "{Cirq Developers}", "title": "Cirq", "publisher": "Zenodo"},
    "monteverde2026": {"author": "Monteverde, Vicente Humberto", "publisher": "Zenodo"},
}


def override(key, typ, f):
    if key in ("fips203", "fips204", "fips205"):
        typ = "techreport"
    f.update(OVERRIDES.get(key, {}))
    body = ",\n".join("  %s = {%s}" % (k, f[k]) for k in KEEP if f.get(k))
    return "@%s{%s,\n%s\n}" % (typ, key, body)


out, failed = [], []
for key, doi in DOIS.items():
    t = fetch(doi)
    if t is None:
        failed.append(key)
        continue
    out.append(override(*clean(key, t)))
bib = "% Generated from doi.org metadata by tools/build_book_bib.py; hand entries for works without a DOI.\n\n" + "\n\n".join(out) + "\n" + HAND
for lang in ("es", "en"):
    pathlib.Path(sys.argv[1], lang, "references.bib").write_text(bib, encoding="utf-8")
print("entries:", len(out), "failed:", failed)
