"""Tests for qang.sectors.reweight_sectors (§154)."""

import numpy as np
import pytest


def test_reweight_sectors_generalises_the_filter():
    from qang.sectors import filter_distribution, reweight_sectors

    rng = np.random.default_rng(3)
    p = rng.random(16)
    f, kept = filter_distribution(p, 4, 2)
    r, kept_r = reweight_sectors(p, 4, [0, 0, 1, 0, 0])
    assert np.allclose(f, r) and kept == pytest.approx(kept_r)
    target = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    r, _ = reweight_sectors(p, 4, target)
    w = np.array([bin(k).count("1") for k in range(16)])
    assert np.allclose(np.bincount(w, weights=r, minlength=5), target)
