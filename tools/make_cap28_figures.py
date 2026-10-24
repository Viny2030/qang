"""Figure of chapter 28 of the book (the pre-registered falsification protocol), Spanish and English.

    python tools/make_cap28_figures.py

writes book/es/figuras/cap28_*.png and book/en/figures/cap28_*.png and prints the numbers the chapter quotes. Nothing
here is simulated: it is an audit of the repository itself.

* Stored verdicts: every examples/data/*.json with a "verdict" dict of booleans is a pre-registered run whose
  predictions were scored by code; passed and failed are counted per run.
* sha256 pre-registrations: each examples/data/*.prereg holds the sha256 of a frozen script (LF-normalized bytes).
  For each one: does the hash match the script today, does it match the script at the commit that added the
  .prereg, and was that commit earlier than the one that added the results.
* Commit pre-registrations: every "committed ... (abcdef1)" cited in RESEARCH_NOTES.md must exist in the history, and
  at that commit the notes must not yet contain the section's results.
* The two IQM Garnet runs (§165, §168) are summarized from their stored files, including the per-gate cost of the X
  chains as a one-slope weighted fit.
"""

import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from make_cap01_figures import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402

DATA = os.path.join(ROOT, "examples", "data")
RED = "#8c2d04"

TEXT = {
    "es": {
        "dir": os.path.join(ROOT, "book", "es", "figuras"),
        "v_x": "predicciones",
        "v_pass": "cumplidas",
        "v_fail": "falladas",
        "v_title": "Veredictos guardados por el código: 33 corridas",
        "a_y": "pre-registros",
        "a_cats": ("por commit,\ncitados en las notas", "por sha256,\nverificables en git", "por sha256,\nsólo con fecha propia"),
        "a_ok": "consistentes",
        "a_title": "Auditoría de la anterioridad",
    },
    "en": {
        "dir": os.path.join(ROOT, "book", "en", "figures"),
        "v_x": "predictions",
        "v_pass": "passed",
        "v_fail": "failed",
        "v_title": "Verdicts stored by the code: 33 runs",
        "a_y": "pre-registrations",
        "a_cats": ("by commit,\ncited in the notes", "by sha256,\ncheckable in git", "by sha256,\nown timestamp only"),
        "a_ok": "consistent",
        "a_title": "Audit of the 'before'",
    },
}


def git(*args, binary=False):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    return out.stdout if binary else out.stdout.decode("utf-8", "replace")


def stored_verdicts():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and isinstance(d.get("verdict"), dict):
            v = [x for x in d["verdict"].values() if isinstance(x, bool)]
            if v:
                rows.append((os.path.basename(f)[:-5], sum(v), len(v) - sum(v)))
    return rows


def sha_audit():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "*.prereg"))):
        h, path = open(f, encoding="utf-8").read().split()[:2]
        rel = os.path.relpath(f, ROOT).replace("\\", "/")
        now = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read().replace(b"\r\n", b"\n")).hexdigest() == h
        added = git("log", "--diff-filter=A", "--format=%h %ct", "--", rel).split()
        pre_c, pre_t = added[-2], int(added[-1])
        at_pre = hashlib.sha256(git("show", f"{pre_c}:{path}", binary=True).replace(b"\r\n", b"\n")).hexdigest() == h
        name = os.path.basename(f)[:-7]
        res_t = None
        for cand in (f"{name}.json", f"{name}_run.json", f"{name}_analysis.json"):
            a = git("log", "--diff-filter=A", "--format=%ct", "--", f"examples/data/{cand}").split()
            if a:
                res_t = int(a[-1])
                break
        out.append({"name": name, "match_now": now, "match_at_prereg": at_pre,
                    "before_results": res_t is not None and pre_t < res_t})
    return out


def commit_audit():
    txt = open(os.path.join(ROOT, "RESEARCH_NOTES.md"), encoding="utf-8").read()
    pairs = []
    for s in re.split(r"\n(?=## )", txt):
        m = re.match(r"## (\S+)\.", s)
        if m:
            pairs += [(m.group(1), c) for c in re.findall(r"(?:committed|commit)[^\n]{0,60}?\(([0-9a-f]{7,8})\)", s)]
    ok = 0
    for sec, c in pairs:
        exists = subprocess.run(["git", "cat-file", "-e", c + "^{commit}"], cwd=ROOT).returncode == 0
        earlier = not re.search(rf"\n## {re.escape(sec)}\. ", git("show", f"{c}:RESEARCH_NOTES.md")) if exists else False
        ok += exists and earlier
    return len(pairs), ok, sorted({s for s, _ in pairs}, key=lambda x: float(re.sub(r"[^0-9.]", "", x)))


def iqm_runs():
    with open(os.path.join(DATA, "iqm_neutrino_qg_run.json"), encoding="utf-8") as fh:
        nu = json.load(fh)
    # the §165 checks, scored from the stored run (I3: filtered P(mu -> mu) lowest at 2.6 GeV)
    surv = [row[1] for row in nu["filtered"]]
    nu_v = {"I1": nu["filtered_error"] < nu["raw_error"], "I2": nu["raw_error"] / nu["filtered_error"] >= 1.5,
            "I3": abs(nu["energies"][int(np.argmin(surv))] - 2.6) < 1e-9, "I4": nu["kept"] >= 0.5}
    with open(os.path.join(DATA, "iqm_milliqang_qg_analysis.json"), encoding="utf-8") as fh:
        mq = json.load(fh)
    ks = np.array([int(k) for k in mq["chains"]], float)
    s = np.array([v["s"] for v in mq["chains"].values()])
    w = 1 / np.array([v["sigma"] for v in mq["chains"].values()]) ** 2
    slope = float(np.sum(w * s * (ks - 1)) / np.sum(w * (ks - 1) ** 2))
    chi2 = float(np.sum(w * (s - slope * (ks - 1)) ** 2) / (len(ks) - 1))
    return nu, nu_v, mq, slope, chi2


def figure(t, rows, sha, n_commit, ok_commit):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    rows = sorted(rows, key=lambda r: (r[1] + r[2], r[1]))
    y = np.arange(len(rows))
    ax1.barh(y, [r[1] for r in rows], color=GREEN, label=t["v_pass"])
    ax1.barh(y, [r[2] for r in rows], left=[r[1] for r in rows], color=ORANGE, label=t["v_fail"])
    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0].replace("_qg", "").replace("_", " ") for r in rows], fontsize=5.5)
    ax1.set_xlabel(t["v_x"])
    ax1.set_title(t["v_title"], fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="lower right")
    ax1.grid(axis="x", alpha=0.3)
    ver = sum(1 for r in sha if r["match_at_prereg"] and r["before_results"])
    counts = (n_commit, ver, len(sha) - ver)
    good = (ok_commit, ver, 0)
    x = np.arange(3)
    ax2.bar(x, counts, color=GREY, width=0.6)
    ax2.bar(x, good, color=BLUE, width=0.6, label=t["a_ok"])
    for xi, c in zip(x, counts):
        ax2.text(xi, c + 0.8, str(c), ha="center", fontsize=9, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(t["a_cats"], fontsize=8)
    ax2.set_ylabel(t["a_y"])
    ax2.set_title(t["a_title"], fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    rows = stored_verdicts()
    p, f = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"stored verdicts: {len(rows)} runs, {p + f} predictions, {p} passed ({p / (p + f):.0%}), {f} failed")
    print("  runs with every prediction failed:", [r[0] for r in rows if r[1] == 0])
    print("  runs with every prediction passed:", [r[0] for r in rows if r[2] == 0])

    sha = sha_audit()
    print(f"\nsha256 pre-registrations: {len(sha)}")
    for r in sha:
        print(f"  {r['name']:40s} matches today {r['match_now']!s:5s}  matches at its commit {r['match_at_prereg']!s:5s}"
              f"  committed before the results {r['before_results']}")
    ver = [r["name"] for r in sha if r["match_at_prereg"] and r["before_results"]]
    print(f"  verifiable in git: {len(ver)}; own timestamp only: {len(sha) - len(ver)}")

    n_commit, ok_commit, secs = commit_audit()
    print(f"\ncommit pre-registrations cited in the notes: {n_commit}, consistent {ok_commit} (sections {secs[0]}-{secs[-1]})")

    nu, nu_v, mq, slope, chi2 = iqm_runs()
    print(f"\n§165 IQM Garnet neutrinos: error raw {nu['raw_error']:.4f} -> filtered {nu['filtered_error']:.4f}"
          f" ({nu['raw_error'] / nu['filtered_error']:.2f}x), kept {nu['kept']:.3f}, survival minimum at {nu['survival_min_E']} GeV,"
          f" {nu['shots']} shots, verdict {nu_v}")
    print(f"§168 IQM Garnet milli-qang: e0 {mq['e0']:.4f}, e1 {mq['e1']:.3f}, M2 ratio {mq['M2_ratio']:.2f},"
          f" X-gate cost (one slope) {slope:.2f} m-qg per gate, chi2/dof {chi2:.2f}, verdict {mq['verdict']}")

    for lang, tx in TEXT.items():
        os.makedirs(tx["dir"], exist_ok=True)
        name = {"es": "cap28_auditoria", "en": "cap28_audit"}[lang]
        fig = figure(tx, rows, sha, n_commit, ok_commit)
        path = os.path.join(tx["dir"], name + ".png")
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
