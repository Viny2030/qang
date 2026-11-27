import math

import pytest

import qang
from qang import (
    REPORT_FIELDS,
    amplitudes_from_p0,
    azimuth_qga,
    filter_distribution,
    hamming_weights,
    p0_from_qga,
    qang_report,
    qg_s_from_qg_z,
    readings_from_p0,
)


def test_public_api_exports_the_canonical_report_and_helpers():
    for name in qang.__all__:
        assert hasattr(qang, name), name

    assert qang.REPORT_FIELDS == REPORT_FIELDS
    assert REPORT_FIELDS[:3] == ("p0", "amplitude_0", "amplitude_1")
    assert "theta_rad" in REPORT_FIELDS and "phi_qga" in REPORT_FIELDS

    assert qang.probability is not None
    assert qang.qga is not None
    assert qang.report is not None
    assert qang.sectors is not None

    assert amplitudes_from_p0(0.25) == pytest.approx((0.5, math.sqrt(0.75)))
    assert readings_from_p0(0.25)["qg_z"] == pytest.approx(-0.5)
    assert qg_s_from_qg_z(0.0) == pytest.approx(1.0)
    assert p0_from_qga(250.0) == pytest.approx(0.5)
    assert azimuth_qga(1.0, 0.0) == pytest.approx(0.0)
    assert hamming_weights(2).tolist() == [0, 1, 1, 2]
    assert filter_distribution([1.0, 0.0, 0.0, 0.0], 2, 0)[0].tolist() == [1.0, 0.0, 0.0, 0.0]

    report = qang_report({"0": 750, "1": 250}, 250.0, 1000, phi_qga=125.0)
    assert tuple(report) == REPORT_FIELDS
    assert report["phi_qga"] == pytest.approx(125.0)
