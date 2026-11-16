from qang.qrng import certified_min_entropy, point_min_entropy
def test_cert_below_point(): assert certified_min_entropy(5000, 10000) < point_min_entropy(5000, 10000)
def test_cert_grows_with_N(): assert certified_min_entropy(500000, 1000000) > certified_min_entropy(5000, 10000)
