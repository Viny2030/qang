"""tools/verify_cap22.py — Baterías cuánticas (ergotropía en qg) y redes (V^k). python tools/verify_cap22.py"""
import numpy as np
import jax.numpy as jnp
from qang.qbattery import energy_qg, ergotropy_qg, passive_fraction, swap_visibility, max_links, chain_two_links_numeric

# 1. Ergotropía: estados puros (r=1) W = E; estados diagonales mezclados W = max(0, −qg_Z)·... = (|qg_Z| − qg_Z)/2
for qz in (-1.0, -0.6, 0.0, 0.6, 1.0):
    E = float(energy_qg(qz)); Wp = float(ergotropy_qg(1.0, qz)); Wd = float(ergotropy_qg(abs(qz), qz))
    assert abs(Wp - E) < 1e-15 and abs(Wd - max(0.0, -qz)) < 1e-15
    print(f"[1] qg_Z={1000*qz:+5.0f} m-qg: E/ω={E:.2f} | W/ω puro={Wp:.2f} | W/ω diagonal={Wd:.2f} | fracción pasiva (r=|qg_Z|)={float(passive_fraction(abs(qz), qz)) if qz<1 else 0:.2f}")
# 2. Coherencia como carga extraíble: qg_Z = 0 con r = 1 (|+>) tiene W = ω/2; con r = 0 (I/2) W = 0 — misma lectura Z
assert abs(float(ergotropy_qg(1.0, 0.0)) - 0.5) < 1e-15 and float(ergotropy_qg(0.0, 0.0)) == 0.0
print("[2] misma lectura 0 m-qg: |+> W=0.5ω vs I/2 W=0 ⇒ la ergotropía exige r (3 bases), no sólo qg_Z")
# 3. Redes: V^k y k_max con umbral 1/3
for V in (0.99, 0.95, 0.9, 0.8):
    print(f"[3] V={V}: k_max={max_links(V)} enlaces (V^k>1/3); V^k_max={swap_visibility(V, max_links(V)):.3f} ⇒ "
          f"{1000*swap_visibility(V, max_links(V)):.0f} m-qg; k_max+1 → {1000*swap_visibility(V, max_links(V)+1):.0f} m-qg")
assert max_links(0.9) == 10 and max_links(0.95) == 21
# 4. Verificación numérica de V1·V2 por composición de canales
qxx, qyy, qzz = chain_two_links_numeric(0.9, 0.8)
assert abs(float(qzz) - 0.72) < 1e-13 and abs(float(qxx) - 0.72) < 1e-13 and abs(float(qyy) + 0.72) < 1e-13
print(f"[4] dos enlaces 0.9 y 0.8: qg_ZZ extremo = {float(qzz):.3f} = V1·V2 (720 m-qg)")
print("verify_cap22: OK")
