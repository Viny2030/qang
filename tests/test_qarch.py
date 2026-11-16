import os
import textwrap
from qang.qarch import (requires, group_modules, parse_pyproject, check_extras, installed_by, unguarded_tests, requirement_root,
                        third_party_roots)


def _write(d, name, src):
    p = d / name
    p.write_text(textwrap.dedent(src))
    return p


def test_import_levels(tmp_path):
    p = _write(tmp_path, "m.py", """
        import numpy as np
        try:
            import jax
        except ImportError:
            jax = None
        def f():
            import scipy
    """)
    assert third_party_roots(str(p)) == {"numpy", "jax"}
    assert third_party_roots(str(p), top_only=False) == {"numpy", "jax", "scipy"}


def test_transitive_requirements(tmp_path):
    q = tmp_path / "qang"; q.mkdir()
    _write(q, "__init__.py", "")
    _write(q, "a.py", "import numpy as np\n")
    _write(q, "b.py", "import jax\n")
    _write(q, "c.py", "from .b import x\n")
    r = requires(str(q))
    assert r["a"] == set() or r["a"] == {"numpy"}
    assert r["c"] == {"jax"}
    g = group_modules(r)
    assert "jax" in g and "c" in g["jax"]


def test_pyproject_and_requirement_roots():
    t = '[project]\ndependencies = [\n "numpy>=1",\n]\n[project.optional-dependencies]\njax = [\'jax[cpu]>=0.4,<0.9; python_version >= "3.10"\']\n' \
        'iqm = ["iqm-client[qiskit]>=36"]\nall = ["scikit-learn>=1.2", "qiskit>=1"]\n[project.urls]\nx = "y"\n'
    e = parse_pyproject(t)
    assert e["jax"][0].startswith("jax[cpu]") and e["dependencies"] == ["numpy>=1"]
    assert requirement_root("scikit-learn>=1.2") == "sklearn" and requirement_root('jax[cpu]>=1; python_version>="3"') == "jax"
    assert installed_by(e, "all") == {"numpy", "sklearn", "qiskit"}
    assert all(check_extras(e).values())


def test_unguarded_test_detected(tmp_path):
    q = tmp_path / "qang"; q.mkdir(); t = tmp_path / "tests"; t.mkdir()
    _write(q, "__init__.py", ""); _write(q, "j.py", "import jax\n")
    _write(t, "test_bad.py", "from qang.j import x\n")
    _write(t, "test_ok.py", "import pytest\npytest.importorskip('jax')\nfrom qang.j import x\n")
    bad = unguarded_tests(str(t), str(q), {"numpy"})
    assert bad == [("test_bad.py", ["jax"])]
