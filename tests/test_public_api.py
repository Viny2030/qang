import math

import pytest

import qang
from qang import (
    amplitudes_from_p0,
    azimuth_qga,
    filter_distribution,
    hamming_weights,
    p0_from_qga,
    qg_s_from_qg_z,
    readings_from_p0,
)


def test_public_api_exports_helpers():
    for name in qang.__all__:
        assert hasattr(qang, name), name

    assert qang.probability is not None
    assert qang.qga is not None
    assert qang.sectors is not None

    assert amplitudes_from_p0(0.25) == pytest.approx((0.5, math.sqrt(0.75)))
    assert readings_from_p0(0.25)["qg_z"] == pytest.approx(-0.5)
    assert qg_s_from_qg_z(0.0) == pytest.approx(1.0)
    assert p0_from_qga(250.0) == pytest.approx(0.5)
    assert azimuth_qga(1.0, 0.0) == pytest.approx(0.0)
    assert hamming_weights(2).tolist() == [0, 1, 1, 2]
    assert filter_distribution([1.0, 0.0, 0.0, 0.0], 2, 0)[0].tolist() == [1.0, 0.0, 0.0, 0.0]
