"""tools/verify_cap49.py — Cap. 49: filtro de simetría, exposición/eco y presupuesto de estabilizadores.
Solo NumPy + qang; las comprobaciones que requieren Qiskit o SciPy se omiten si no están instalados.
Uso: python tools/verify_cap49.py   (si algo cambia, el script se detiene)."""
from itertools import combinations

import numpy as np

from qang import measures
from qang.sectors import (
    echo_transfer_matrix,
    filter_distribution,
    reweight_sectors,
    sector_exposure,
    stabilizer_shot_budget,
    unmix_sector,
)

rng = np.random.default_rng(49)

filtered, kept = filter_distribution([1.0, 0.0, 0.0, 0.0], 2, 0)
assert np.allclose(filtered, [1.0, 0.0, 0.0, 0.0]) and kept == 1.0
p = np.array([0.5, 0.1, 0.2, 0.2])
filtered, kept = filter_distribution(p, 2, 1)
assert np.allclose(filtered, [0.0, 1 / 3, 2 / 3, 0.0]) and abs(kept - 0.3) < 1e-15
assert abs(sum(filtered) - 1.0) < 1e-15
print("[1] filter_distribution: peso fijo, distribución normalizada y fracción retenida (1,0 y 0,3)")

delta, measured = reweight_sectors([1.0, 0.0, 0.0, 0.0], 2, [0.25, 0.5, 0.25])
assert np.allclose(delta, [1.0, 0.0, 0.0, 0.0]) and measured == 1.0
mixed, measured = reweight_sectors(p, 2, [0.25, 0.5, 0.25])
assert abs(mixed[1] + mixed[2] - 0.5) < 1e-15 and abs(mixed[0] + mixed[3] - 0.5) < 1e-15
assert abs(sum(mixed) - 1.0) < 1e-15
print("[2] reweight_sectors: con delta recupera el filtro; conmezcla reparte 0,25/0,5/0,25 por sector")

chosen, total = stabilizer_shot_budget([0.2, 0.5, 0.1], [1, 2, 1], 3)
assert chosen == [0, 1] and abs(total - 0.7) < 1e-15
worst = 0.0
for _ in range(300):
    w = rng.uniform(0, 5, int(rng.integers(1, 10)))
    c = rng.integers(0, 6, len(w))
    budget = int(rng.integers(0, 15))
    _, got = stabilizer_shot_budget(w, c, budget)
    want = max(
        (sum(w[i] for i in combo) for k in range(len(w) + 1) for combo in combinations(range(len(w)), k)
         if sum(c[i] for i in combo) <= budget),
        default=0.0,
    )
    worst = max(worst, abs(got - want))
assert worst < 1e-9
print(f"[3] stabilizer_shot_budget: mochila exacta (300 instancias, peor diferencia {worst:.1e}); sin evidencia de hardware")

record = measures.find("stabilizer_budget")
assert record.status == "sin_verificar"
assert "sin comparar" in record.evidence[0]
assert "measured gain" not in record.evidence[0].lower()
print("[4] prerregistro/hardware: sin corrida de dispositivo; el catálogo mantiene `sin_verificar` y ninguna ganancia medida")

try:
    from qiskit import QuantumCircuit

    conserve = QuantumCircuit(2)
    conserve.x(1)
    conserve.cx(0, 1)
    exposed = QuantumCircuit(2)
    exposed.h(0)
    exposed.cx(0, 1)
    zero = sector_exposure(conserve, 1)
    nonzero = sector_exposure(exposed, 1)
    assert zero["max"] == 0.0 and zero["mean"] == 0.0
    assert nonzero["max"] > 0.99
    print(f"[5] exposición (Qiskit): CX conserva el peso (0); H+CX sale del sector ({nonzero['max']:.3f})")
except ImportError:
    print("[5] exposición (Qiskit): omitida, Qiskit no instalado")

try:
    M = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    full = np.zeros(8)
    full[[1, 2, 4]] = M @ np.array([0.5, 0.3, 0.2])
    echo = []
    for j in range(3):
        run = np.zeros(8)
        run[[1, 2, 4]] = M[:, j]
        echo.append(run)
    measured_M = echo_transfer_matrix(echo, 3, 1)
    assert np.allclose(measured_M, M)
    restored = unmix_sector(full, measured_M, 3, 1, power=1.0)
    assert np.allclose(restored[[1, 2, 4]], [0.5, 0.3, 0.2], atol=1e-9)
    print("[6] eco (SciPy): matriz de transferencia recuperada y mezcla invertida sin ganancia física nueva")
except ImportError:
    print("[6] eco (SciPy): omitida, SciPy no instalado")

print("verify_cap49: OK")
