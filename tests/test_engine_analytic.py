"""Engine checks against hand calculations and closed-form results."""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine import NetlistError, SimulationError, ac, op, parse, process_audio, tran
from engine import measure as ms
from engine.devices import VT

OA = ".model OA OPAMP(A0=2e5 GBW=1e6 RIN=1e12 ROUT=75 VSAT_HI=1.5 VSAT_LO=1.5 IQ=1m)\n"


def test_divider_dc():
    c = parse("* divider\nV1 in 0 9\nR1 in mid 10k\nR2 mid 0 20k\n")
    r = op(c)
    assert r.converged and r.method == "newton"
    assert r.v["mid"] == pytest.approx(6.0, abs=1e-6)
    # SPICE sign: a source delivering power carries negative current
    assert r.i["V1"] == pytest.approx(-9.0 / 30e3, rel=1e-6)
    assert r.power["R2"] == pytest.approx(6.0 ** 2 / 20e3, rel=1e-6)


def test_rc_lowpass_corner():
    R, C = 1e3, 100e-9
    c = parse(f"* rc\n*@ in VIN\n*@ out out\nVIN in 0 0\nR1 in out {R}\nC1 out 0 {C}\n")
    a = ac(c, 10, 1e6, 200)
    f0 = 1 / (2 * math.pi * R * C)
    assert ms.fc(a) == pytest.approx(f0, rel=1e-3)
    assert ms.gain_db(a, f0) == pytest.approx(-3.0103, abs=0.01)
    assert ms.phase_deg(a, f0) == pytest.approx(-45.0, abs=0.2)


def test_rc_step_response():
    R, C = 1e3, 1e-6
    c = parse(f"* rc step\n*@ in VIN\n*@ out out\nVIN in 0 DC 0 PULSE(0 1 0 1n 1n 1 2)\n"
              f"R1 in out {R}\nC1 out 0 {C}\n")
    tr = tran(c, 5e-3, 1e-6, nodes=["out"])
    tau = R * C
    k = int(round(tau / 1e-6))
    assert tr.v["out"][k] == pytest.approx(1 - math.exp(-1), abs=2e-3)


def test_ce_bias_matches_hand_calculation():
    IS, BF = 1e-14, 200.0
    c = parse(f"""* CE bias
VCC vcc 0 9
R1 vcc b 100k
R2 b 0 22k
RC vcc c 4.7k
RE e 0 1k
Q1 c b e QX
.model QX NPN(IS={IS} BF={BF})
""")
    r = op(c)
    # hand calculation: Thevenin base network, Ebers-Moll with BR = 1, iterate on VBE
    vth, rth = 9 * 22 / 122, 100e3 * 22e3 / 122e3
    vbe = 0.65
    for _ in range(50):
        ib = (vth - vbe) / (rth + (BF + 1) * 1e3)
        ic = BF * ib
        vbe = VT * math.log(ic / IS + 1)
    q = r.devices["Q1"]
    assert q["ic"] == pytest.approx(ic, rel=2e-3)
    assert q["vbe"] == pytest.approx(vbe, abs=1e-3)
    assert r.v["c"] == pytest.approx(9 - 4.7e3 * ic, rel=2e-3)
    assert q["region"] == "active"


def test_bjt_regions_and_pnp_mirror():
    c = parse("""* saturated and cut-off transistors
VCC vcc 0 5
RB1 vcc b1 10k
RC1 vcc c1 1k
Q1 c1 b1 0 QN
RC2 vcc c2 1k
Q2 c2 0 0 QN
VEE vee 0 -5
RB3 vee b3 10k
RC3 vee c3 1k
Q3 c3 b3 0 QP
.model QN NPN(IS=1e-14 BF=100)
.model QP PNP(IS=1e-14 BF=100)
""")
    r = op(c)
    assert r.devices["Q1"]["region"] == "saturation"
    assert r.devices["Q2"]["region"] == "cutoff"
    assert r.devices["Q3"]["region"] == "saturation"
    # PNP mirrors NPN: same magnitudes, opposite signs
    assert r.v["c3"] == pytest.approx(-r.v["c1"], abs=1e-9)
    assert r.devices["Q3"]["ic"] == pytest.approx(-r.devices["Q1"]["ic"], rel=1e-9)


@pytest.mark.parametrize("kind", ["inverting", "noninverting"])
def test_opamp_gain(kind):
    A0 = 2e5
    if kind == "inverting":
        body = "X1 0 m vcc vee out OA\nR1 in m 10k\nR2 m out 100k\n"
        ideal, beta = -10.0, 10e3 / 110e3
    else:
        body = "X1 in m vcc vee out OA\nR1 m 0 10k\nR2 m out 90k\n"
        ideal, beta = 10.0, 0.1
    c = parse(f"* opamp\n*@ in VIN\n*@ out out\nVCC vcc 0 15\nVEE vee 0 -15\nVIN in 0 0\n{body}{OA}")
    a = ac(c, 10, 1e6, 40)
    expected = abs(ideal) / (1 + 1 / (A0 * beta))
    assert abs(a.h[0]) == pytest.approx(expected, rel=1e-4)
    # one dominant pole: closed-loop bandwidth ~ GBW * beta
    assert ms.f3db_high(a) == pytest.approx(1e6 * beta, rel=0.03)
    # DC transfer of a 0.5 V input, and clean rail limiting at 15 - 1.5 V
    tr = tran(c, 2e-3, 2e-6, input_wave=2.0 * np.sin(2 * np.pi * 1e3 * np.arange(1001) * 2e-6))
    assert np.max(np.abs(tr.v["out"])) == pytest.approx(13.5, abs=0.05)


def test_lc_resonance():
    L, C, R = 10e-3, 100e-9, 1e3
    c = parse(f"* tank\n*@ in VIN\n*@ out out\nVIN in 0 0\nR1 in out {R}\nL1 out 0 {L}\nC1 out 0 {C}\n")
    a = ac(c, 1e3, 20e3, 2000)
    f0 = 1 / (2 * math.pi * math.sqrt(L * C))
    lo, hi, peak, pf = ms.corners(a)
    assert pf == pytest.approx(f0, rel=2e-3)
    assert peak == pytest.approx(0.0, abs=1e-3)
    assert ms.q_factor(a) == pytest.approx(R * math.sqrt(C / L), rel=0.01)


def test_lc_ringing_frequency_in_transient():
    L, C = 1e-3, 1e-6
    c = parse(f"* ring\n*@ in VIN\n*@ out out\nVIN in 0 DC 0 PULSE(0 1 0 1n 1n 1 2)\n"
              f"R1 in out 100\nL1 out 0 {L}\nC1 out 0 {C}\n")
    tr = tran(c, 3e-3, 1e-7, nodes=["out"])
    y = tr.v["out"]
    z = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    f = 1 / (np.mean(np.diff(z)) * 1e-7)
    # the 100 ohm source damps the tank: Q = R sqrt(C/L), ring at w0 sqrt(1 - 1/(4 Q^2))
    q = 100 * math.sqrt(C / L)
    fd = math.sqrt(1 - 1 / (4 * q * q)) / (2 * math.pi * math.sqrt(L * C))
    assert f == pytest.approx(fd, rel=0.002)


def test_diode_forward_drop():
    c = parse("* diode\nV1 a 0 5\nR1 a k 4.3k\nD1 k 0 DX\n.model DX D(IS=1e-14 N=1)\n")
    r = op(c)
    i = r.devices["D1"]["id"]
    assert r.v["k"] == pytest.approx(VT * math.log(i / 1e-14 + 1), abs=1e-6)
    assert i == pytest.approx((5 - r.v["k"]) / 4.3e3, rel=1e-6)


def test_jfet_saturation_current():
    c = parse("* idss\nVD d 0 10\nJ1 d 0 0 JX\n.model JX NJF(VTO=-2 BETA=1m LAMBDA=0)\n")
    r = op(c)
    assert r.devices["J1"]["id"] == pytest.approx(1e-3 * 4, rel=1e-6)
    assert r.devices["J1"]["region"] == "saturation"


def test_process_audio_is_ac_coupled_and_linear_for_small_signals():
    c = parse("* buffer\n*@ in VIN\n*@ out out\nVIN in 0 1\nR1 in out 1k\nR2 out 0 1k\n")
    x = 0.1 * np.sin(2 * np.pi * 440 * np.arange(4800) / 48000)
    y = process_audio(c, x, 48000)
    assert len(y) == len(x)
    assert np.allclose(y, 0.5 * x, atol=1e-9)
    assert np.allclose(process_audio(c, np.zeros(100), 48000), 0.0)


def test_capacitor_ic_starts_an_oscillator():
    c = parse("""* Wien bridge
*@ out out
VCC vcc 0 12
VEE vee 0 -12
R1 out a 10k
C1 a p 10n ic=0.1
R2 p 0 10k
C2 p 0 10n
RF out n 22k
RG n 0 10k
X1 p n vcc vee out OA
""" + OA)
    tr = tran(c, 20e-3, 2e-6, nodes=["out"])
    assert np.ptp(tr.v["out"][-2000:]) > 10.0


def test_thd_measure_on_known_signal():
    n = 4096
    t = np.arange(n) / n * 8
    x = np.sin(2 * np.pi * t) + 0.1 * np.sin(2 * np.pi * 3 * t)
    assert ms.thd(x, cycles=8) == pytest.approx(10.0, rel=1e-6)
    assert ms.thd(x, fs=n / 8, f0=1.0) == pytest.approx(10.0, rel=1e-3)


def test_errors_are_clear():
    with pytest.raises(NetlistError):
        parse("* bad\nQ1 c b e NOPE\n")
    c = parse("* no io\nV1 a 0 1\nR1 a 0 1k\n")
    with pytest.raises(NetlistError, match="output"):
        ac(c)
    with pytest.raises(SimulationError):
        tran(c, -1, 1e-6)


def test_dc_fallback_reports_method():
    # a high-gain comparator loop that plain Newton from zero finds hard
    c = parse("""* latch
VCC vcc 0 9
VEE vee 0 -9
X1 p n vcc vee out OA
R1 out p 100k
R2 p 0 10k
RN n 0 1k
""" + OA)
    r = op(c)
    assert r.converged
    assert r.method in ("newton", "gmin", "source")
