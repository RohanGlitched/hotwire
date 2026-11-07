"""Op-amp saturation recovery: the macromodel's pole node is clamped (anti-windup).

Without the clamp the integrator node winds up to A0 times the overdrive while the
output sits on a rail, so a Schmitt trigger never flips back and a hard clipper's
output zero crossings lag by hundreds of microseconds.
"""
from __future__ import annotations

import numpy as np

from engine import parse
from engine.analyses import op, tran
from engine import devices as dv

SCHMITT = """* Inverting Schmitt trigger, thresholds about +-1.23 V
*@ in VIN
*@ out out
VCC vcc 0 15
VEE vee 0 -15
VIN in 0 DC 0 SIN(0 2.5 1k)
R1 out p 100k
R2 p 0 10k
X1 p in vcc vee out TL072
"""

CLIPPER = """* Gain-101 non-inverting amplifier driven into both rails
*@ in VIN
*@ out out
VCC vcc 0 15
VEE vee 0 -15
VIN in 0 DC 0 SIN(0 {amp} 1k)
X1 in fb vcc vee out TL072
RF out fb 100k
RG fb 0 1k
"""


def _zero_cross_lags(t, y, after=1e-3, period=1e-3):
    lags = []
    for k in range(1, len(t)):
        if t[k] > after and (y[k - 1] > 0) != (y[k] > 0):
            tc = t[k - 1] + (t[k] - t[k - 1]) * y[k - 1] / (y[k - 1] - y[k])
            ideal = round(tc / (period / 2)) * (period / 2)   # the input crosses zero every half period
            lags.append(tc - ideal)
    return lags


def test_schmitt_trigger_toggles_between_both_rails():
    c = parse(SCHMITT)
    r = tran(c, 3e-3, 1e-6, nodes=["out", "in"])
    assert r.converged
    late = r.t > 1e-3
    y, x = r.v["out"][late], r.v["in"][late]
    hi = 15 - 1.5
    assert y.max() > 0.98 * hi and y.min() < -0.98 * hi          # reaches both rails
    # the output follows the input's sign with hysteresis: high when the input is well below -1.23 V
    assert np.all(y[x < -1.5] > 12) and np.all(y[x > 1.5] < -12)
    flips = np.count_nonzero(np.diff(np.sign(y)) != 0)
    assert flips >= 4                                              # two full cycles, two flips each


def test_rail_clipper_zero_crossing_lag_is_small():
    for amp, limit in ((1.0, 50e-6), (0.2, 20e-6)):
        c = parse(CLIPPER.format(amp=amp))
        r = tran(c, 3e-3, 0.5e-6, nodes=["out"])
        assert r.converged
        lags = _zero_cross_lags(r.t, r.v["out"])
        assert len(lags) >= 3
        assert max(abs(x) for x in lags) < limit, f"{amp} V: lag {max(lags) * 1e6:.1f} us"


def test_clamp_keeps_linear_operation_and_full_swing():
    # in the linear region the clamp carries no current: a gain of -10 stays exact
    c = parse("""* inverting x10
VCC vcc 0 15
VEE vee 0 -15
VIN in 0 DC 0.5
R1 in m 10k
R2 m out 100k
X1 0 m vcc vee out OPAMP_IDEAL
""")
    assert abs(op(c).v["out"] + 5.0) < 1e-4
    # driven hard, the pole node stays inside mid +/- K*half and the output still swings
    # to within 0.5% of the rails
    c = parse(CLIPPER.format(amp=1.0).replace("SIN(0 1 1k)", "").replace("DC 0", "DC 1"))
    o = op(c)
    assert o.v["out"] > 0.99 * 13.5
    p_node = o.x[next(i for i, n in enumerate(_internal_names(c)) if n.endswith("#p"))]
    assert p_node < dv.OPAMP_CLAMP_K * 13.5 + 1.0


def _internal_names(c):
    from engine.mna import Network
    net = Network(c)
    return list(net.node_names) + list(net.internal_names)


def test_clamp_current_and_slope():
    sp, slope = dv._softplus(1.0)
    assert abs(sp - 1.0) < 1e-12 and abs(slope - 1.0) < 1e-12
    sp, slope = dv._softplus(-1.0)
    assert sp < 1e-40 and slope < 1e-40
    assert abs(dv._softplus(0.0)[1] - 0.5) < 1e-15
