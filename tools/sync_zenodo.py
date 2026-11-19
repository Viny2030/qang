"""tools/sync_zenodo.py — keeps .zenodo.json in sync with the version in pyproject.toml.

Zenodo's GitHub integration reads .zenodo.json from the tagged release, so this
file is the landing metadata of the DOI (10.5281/zenodo.22832150) cited by the
README, CITATION.cff and paper.md. It drifted from 0.12.0 while the package moved
to 0.16.0 because nothing regenerated it at release time.

This script rewrites exactly two things: the `version` field and the PyPI
`related_identifiers` entry. Every other byte is left untouched, so the diff
stays readable. Standard library only.

    python tools/sync_zenodo.py           # rewrite in place
    python tools/sync_zenodo.py --check   # exit 1 if out of sync (for CI)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
ZENODO = ROOT / ".zenodo.json"


def package_version() -> str:
    """The version declared in [project] of pyproject.toml."""
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit(f"no version found in {PYPROJECT}")
    return match.group(1)


def sync(version: str) -> bool:
    """Point .zenodo.json at `version`. Returns True if the file changed."""
    text = ZENODO.read_text(encoding="utf-8")
    updated = re.sub(r'("version"\s*:\s*")[^"]*(")', rf'\g<1>{version}\g<2>', text, count=1)
    updated = re.sub(
        r'(https://pypi\.org/project/qang/)[^/]*/', rf'\g<1>{version}/', updated
    )
    if updated == text:
        return False
    ZENODO.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if .zenodo.json is out of sync with pyproject.toml",
    )
    args = parser.parse_args(argv)

    version = package_version()
    changed = sync(version)

    if args.check:
        if changed:
            print(f".zenodo.json is out of sync with pyproject.toml ({version})")
            return 1
        print(f".zenodo.json in sync ({version})")
        return 0

    print(f".zenodo.json {'updated' if changed else 'already in sync'} ({version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
