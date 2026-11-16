from qang.qfalsify import register, verify_registration, binom_upper, decide, holm
def test_registry(): p = {"a": 1, "b": "x"}; d = register(p); assert verify_registration({"b": "x", "a": 1}, d) and not verify_registration({"a": 2, "b": "x"}, d)
def test_pvalue(): assert abs(binom_upper(60, 100, 0.5) - 0.02844) < 1e-5
def test_decide(): assert decide(60, 100, 0.5) == "confirmada" and decide(40, 100, 0.5) == "refutada" and decide(55, 100, 0.5) == "no concluyente"
def test_holm(): assert holm((0.001, 0.03, 0.04)) == (True, False, False)
