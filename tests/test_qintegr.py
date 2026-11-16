import math
import textwrap
import numpy as np
from qang.qintegr import (INTEGRATIONS, integration_modules, import_levels, extra_report, guard_report, modules_with_tests,
                          missing_library_message, w_state_t1_readings)
from qang.qarch import parse_pyproject


def test_import_levels(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(textwrap.dedent("""
        import numpy as np
        def f():
            import mitiq
    """))
    assert import_levels(str(p)) == (set(), {"mitiq"})


def test_extra_report_and_message():
    ex = parse_pyproject('[project]\ndependencies = ["numpy"]\n[project.optional-dependencies]\nmitiq = ["mitiq>=1"]\nexperiments = ["qiskit-experiments>=0.6"]\n[project.urls]\nx="y"\n')
    r = extra_report(ex)
    assert r["mitiq"] and r["qiskit_experiments"] and not r["stim"]
    assert 'qang[mitiq]' in missing_library_message("mitiq")


def test_guard_report(tmp_path):
    t = tmp_path / "test_x.py"
    t.write_text(textwrap.dedent("""
        import pytest
        def test_ok():
            pytest.importorskip("stim")
            from qang.integrations import stim as S
        def test_bad():
            from qang.integrations import qutip as Q
    """))
    assert guard_report(str(t)) == {"test_bad": ["qutip"]}
    assert modules_with_tests(str(t)) == {"stim", "qutip"}


def test_w_state_filter_exact():
    r = w_state_t1_readings(0.6)
    assert abs(r["raw_mqg"] - 634.1) < 0.05 and abs(r["filtered_mqg"] - 1000 / 3) < 1e-9 and abs(r["kept"] - math.exp(-0.6)) < 1e-12


def test_require_guard(tmp_path):
    from qang.qintegr import require_guard_report
    d = tmp_path / "integrations"
    d.mkdir()
    (d / "mitiq.py").write_text("from . import require\nrequire('mitiq', 'mitiq')\nfrom mitiq import zne\n")
    (d / "stim.py").write_text("from . import require\nfrom stim import X\nrequire('stim', 'stim')\n")
    (d / "qutip.py").write_text("def f():\n    import qutip\n")
    assert require_guard_report(str(tmp_path)) == {"mitiq": True, "qutip": True, "stim": False}


def test_require_guard_assignment(tmp_path):
    from qang.qintegr import require_guard_report
    d = tmp_path / "integrations"
    d.mkdir()
    (d / "mitiq.py").write_text("from . import require\nmitiq = require('mitiq', 'mitiq')\nfrom mitiq import zne\n")
    assert require_guard_report(str(tmp_path)) == {"mitiq": True}


def test_guard_via_helper(tmp_path):
    t = tmp_path / "test_y.py"
    t.write_text(textwrap.dedent("""
        import pytest
        def _jax64():
            return pytest.importorskip("jax")
        def test_a():
            _jax64()
            from qang.integrations import jax as J
    """))
    assert guard_report(str(t)) == {}
