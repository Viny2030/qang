import importlib.util
import os
import sys
import textwrap

_SPEC = importlib.util.spec_from_file_location("check_book_api", os.path.join(os.path.dirname(__file__), "..", "tools", "check_book_api.py"))
cba = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cba)


def _repo(tmp_path, code):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "book" / "es").mkdir(parents=True)
    return tmp_path


def test_code_blocks_and_aliases():
    t = "x\n```python\nimport qang.probability as pb\nr = pb.analyze(1, 2)\n```\n```{python}\na=1\n```\n"
    b = cba.code_blocks(t)
    assert len(b) == 2 and b[0][0] == 3


def test_detects_wrong_keyword_and_missing_name(tmp_path):
    os.makedirs(tmp_path / "qang")
    (tmp_path / "qang" / "__init__.py").write_text("")
    (tmp_path / "qang" / "m.py").write_text("def f(a, b=1):\n    return a\n")
    os.makedirs(tmp_path / "book" / "es")
    (tmp_path / "book" / "es" / "c.qmd").write_text(textwrap.dedent("""\
        ```python
        import qang.m as m
        m.f(1, b=2)
        m.f(1, c=3)
        m.g(1)
        ```
        """))
    for k in [k for k in sys.modules if k == "qang" or k.startswith("qang.")]:
        del sys.modules[k]
    problems, unimp, nb, nc = cba.scan(str(tmp_path))
    assert nb == 1 and len(problems) == 2
    assert any("c=3" in p or "unexpected keyword" in p for p in problems) and any("no existe" in p for p in problems)
    for k in [k for k in sys.modules if k == "qang" or k.startswith("qang.")]:
        del sys.modules[k]
