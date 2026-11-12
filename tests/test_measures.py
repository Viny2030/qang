"""Tests for qang.measures: the catalogue of sub-measures (book, appendix H)."""

import importlib.util

import pytest

from qang import measures as M


def test_catalogue_is_well_formed():
    keys = [m.key for m in M.CATALOG]
    assert len(keys) == len(set(keys))
    for m in M.CATALOG:
        assert m.status in M.STATUSES, m.key
        assert m.family in M.FAMILIES, m.key
        for pair in (m.equals, m.decision, m.evidence):
            assert len(pair) == 2 and all(isinstance(s, str) and s for s in pair), m.key
        assert m.symbol and m.definition and m.where, m.key
        for lang in ("es", "en"):
            assert "\t" not in m.text("definition", lang) + m.text("symbol", lang), m.key
        assert "\t" not in "".join(m.equals + m.decision + m.evidence), m.key


def test_english_symbols_and_definitions_have_no_spanish():
    spanish = (" de ", " en ", " del ", "sobre", "pasos", "lecturas", "compuertas", "población", "insesgad",
               "vuelta", "período", "grilla", "filtro", "costo", "signos", "parada", "compilación")
    for m in M.CATALOG:
        text = " " + m.text("symbol", "en") + " " + m.text("definition", "en") + " "
        assert not any(w in text for w in spanish), (m.key, text)


def test_every_family_and_the_main_statuses_are_used():
    assert all(M.by_family(f) for f in M.FAMILIES)
    counts = M.summary()
    assert sum(counts.values()) == len(M.CATALOG)
    for s in ("exacta", "ganancia", "hardware", "sin_verificar"):
        assert counts[s] > 0


def test_code_paths_resolve():
    for m in M.CATALOG:
        if not m.code:
            continue
        top = m.code.split(".")[1] if m.code.startswith("qang.") else ""
        if top == "qiskit_gate" and importlib.util.find_spec("qiskit") is None:
            continue
        if top == "integrations" and importlib.util.find_spec("jax") is None:
            continue
        assert callable(M.resolve(m)) or M.resolve(m) is not None, m.code


def test_lookups():
    assert M.find("qg_kappa").status == "sin_ganancia"
    assert M.find("min_area").status == "ganancia"
    with pytest.raises(KeyError):
        M.find("nope")
    with pytest.raises(ValueError):
        M.by_status("nope")
    with pytest.raises(ValueError):
        M.resolve(M.find("q_hertz"))


def test_registry_values_match_the_library():
    from qang import algebra, probability

    assert probability.shots_for_sign(0.5) == 27
    assert round(algebra.geodesic_distance(0.3, -0.5), 3) == 0.414
    assert round(probability.dissipative_deficit(0.0, 0.0, -0.6), 6) == 0.8
