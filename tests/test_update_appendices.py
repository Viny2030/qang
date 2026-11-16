import importlib.util
import os
import sys

SPEC = importlib.util.spec_from_file_location("update_appendices", os.path.join(os.path.dirname(__file__), "..", "tools", "update_appendices.py"))


def _load(tmp_root):
    sys.argv = ["x", str(tmp_root)]
    m = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(m)
    return m


def test_put_section_idempotent(tmp_path):
    m = _load(os.path.join(os.path.dirname(__file__), ".."))
    p = tmp_path / "a.qmd"
    p.write_text("# Título\n\ntexto\n", encoding="utf-8")
    assert m.put_section(str(p), "## Nueva\n\ncuerpo") is True
    first = p.read_text(encoding="utf-8")
    assert first.count(m.BEGIN) == 1 and first.startswith("# Título")
    assert m.put_section(str(p), "## Nueva\n\ncuerpo") is False
    assert m.put_section(str(p), "## Nueva\n\notro") is True
    assert p.read_text(encoding="utf-8").count(m.BEGIN) == 1


def test_facts_hold():
    m = _load(os.path.join(os.path.dirname(__file__), ".."))
    f = m.facts()
    assert f["multi_keep"] == ["CZ", "Fredkin", "SWAP", "iSWAP"] and f["z_mono"] and not f["qgs_mono"]
