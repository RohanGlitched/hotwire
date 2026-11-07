"""MOSFETs (SPICE level 1) against closed-form Shichman-Hodges, plus the .subckt / .param parser."""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine import parse
from engine.analyses import ac, op, tran
from engine.netlist import NetlistError, eval_expr

NMOS = ".model NX NMOS(VTO=1 KP=2m LAMBDA=0.02)\n"


def _id(text: str, name: str = "M1") -> dict:
    o = op(parse(text))
    assert o.converged
    return o.devices[name]


# --------------------------------------------------------------------------- analytic device checks

@pytest.mark.parametrize("vgs,vds", [(2.0, 5.0), (3.0, 10.0), (4.5, 6.0)])
def test_saturation_current(vgs, vds):
    d = _id(f"* sat\nVD d 0 {vds}\nVG g 0 {vgs}\nM1 d g 0 NX\n{NMOS}")
    expected = 2e-3 / 2 * (vgs - 1) ** 2 * (1 + 0.02 * vds)
    assert d["region"] == "saturation"
    # plus the reverse-biased bulk-drain diode: IS + GMIN * VDS (about 5 pA)
    assert d["id"] == pytest.approx(expected + 1e-14 + 1e-12 * vds, rel=1e-12, abs=1e-15)
    assert d["gm"] == pytest.approx(2e-3 * (vgs - 1) * (1 + 0.02 * vds), rel=1e-9)


@pytest.mark.parametrize("vgs,vds", [(3.0, 0.5), (5.0, 1.0)])
def test_triode_current(vgs, vds):
    d = _id(f"* triode\nVD d 0 {vds}\nVG g 0 {vgs}\nM1 d g 0 NX\n{NMOS}")
    expected = 2e-3 * (1 + 0.02 * vds) * vds * ((vgs - 1) - vds / 2)
    assert d["region"] == "triode"
    assert d["id"] == pytest.approx(expected + 1e-14 + 1e-12 * vds, rel=1e-12, abs=1e-15)


def test_cutoff_and_reverse_mode():
    d = _id(f"* off\nVD d 0 5\nVG g 0 0.9\nM1 d g 0 NX\n{NMOS}")
    assert d["region"] == "cutoff" and abs(d["id"]) < 1e-11
    # drain below source: source and drain swap roles, the current reverses
    d = _id(f"* reverse\nVD d 0 -0.3\nVG g 0 3\nM1 d g 0 0 NX\n{NMOS}")
    vgd = 3 + 0.3
    expected = -2e-3 * (1 + 0.02 * 0.3) * 0.3 * ((vgd - 1) - 0.15)
    # plus the forward-biased bulk-drain diode (IS = 1e-14) carrying current out of the drain
    diode = 1e-14 * (math.exp(0.3 / 0.025864186) - 1)
    assert d["id"] == pytest.approx(expected - diode, rel=1e-6)


def test_body_effect_raises_threshold():
    gamma, phi, vto = 0.5, 0.7, 1.0
    for vsb in (0.0, 1.0, 3.0):
        txt = (f"* body\nVD d 0 8\nVG g 0 {3 + vsb}\nVS s 0 {vsb}\nM1 d g s 0 NB\n"
               f".model NB NMOS(VTO={vto} KP=1m GAMMA={gamma} PHI={phi})\n")
        d = _id(txt)
        vth = vto + gamma * (math.sqrt(phi + vsb) - math.sqrt(phi))
        assert d["vth"] == pytest.approx(vth, abs=1e-12)
        assert d["id"] == pytest.approx(1e-3 / 2 * (3 - vth) ** 2, rel=1e-8)
        # gmbs = gm * gamma / (2 sqrt(phi + vsb))
        assert d["gmbs"] == pytest.approx(d["gm"] * gamma / (2 * math.sqrt(phi + vsb)), rel=1e-9)


def test_pmos_mirrors_nmos():
    n = _id(f"* n\nVD d 0 6\nVG g 0 3\nM1 d g 0 NX\n{NMOS}")
    p = _id("* p\nVD d 0 -6\nVG g 0 -3\nM1 d g 0 PX\n.model PX PMOS(VTO=-1 KP=2m LAMBDA=0.02)\n")
    assert p["id"] == pytest.approx(-n["id"], rel=1e-12)
    assert p["region"] == n["region"] == "saturation"


def test_geometry_and_multiplier_scale_beta():
    base = _id(f"* g\nVD d 0 5\nVG g 0 2\nM1 d g 0 NX\n{NMOS}")["id"]
    wide = _id(f"* g\nVD d 0 5\nVG g 0 2\nM1 d g 0 NX W=200u L=100u\n{NMOS}")["id"]
    multi = _id(f"* g\nVD d 0 5\nVG g 0 2\nM1 d g 0 NX M=3\n{NMOS}")["id"]
    assert wide == pytest.approx(2 * base, rel=1e-8)
    assert multi == pytest.approx(3 * base, rel=1e-8)


def test_built_in_parts_and_three_node_form():
    c = parse("* 2N7000\nVD d 0 10\nVG g 0 2.2\nM1 d g s 2N7000\nRS s 0 1\n")
    m = c.element("M1")
    assert m.nodes == ["d", "g", "s", "s"] and c.models["2N7000"].type == "NMOS"
    d = op(c).devices["M1"]
    assert d["region"] == "saturation" and 1e-3 < d["id"] < 20e-3
    for part in ("BS170", "BS250", "IRF510", "IRF9510"):
        parse(f"* p\nVD d 0 -5\nM1 d g 0 {part}\nVG g 0 0\n")


def test_overlap_capacitance_sets_the_input_pole():
    # gate driven through 1 Meg into CGS = 100 p (to an AC ground): pole at 1/(2 pi R C)
    c = parse("""* gate RC
*@ in VIN
*@ out g
VIN in 0 DC 0 AC 1
RG in g 1meg
VD d 0 0
M1 d g 0 NC
.model NC NMOS(VTO=5 KP=1m CGS=100p)
""")
    a = ac(c, 10, 1e5, 40)
    from engine import measure as ms
    assert ms.f3db_high(a) == pytest.approx(1 / (2 * math.pi * 1e6 * 100e-12), rel=2e-3)


def test_led_switch_transient_switches_cleanly():
    c = parse("""* LED switch
*@ in VIN
*@ out d
VCC vcc 0 9
VIN g 0 DC 0 PULSE(0 5 0.1m 1u 1u 0.2m 0.4m)
RL vcc a 330
D1 a d LED_RED
M1 d g 0 2N7000
""")
    r = tran(c, 0.8e-3, 1e-6, nodes=["d"])
    assert r.converged
    on = r.v["d"][(r.t > 0.15e-3) & (r.t < 0.28e-3)]
    off = r.v["d"][(r.t > 0.35e-3) & (r.t < 0.48e-3)]
    assert on.max() < 0.1 and off.min() > 5.0


# --------------------------------------------------------------------------- parser: .param, .subckt

def test_param_arithmetic():
    assert eval_expr("2*4.7k") == pytest.approx(9400)
    assert eval_expr("{1/(2*pi*f*c)}", {"f": 1e3, "c": 10e-9}) == pytest.approx(1 / (2 * math.pi * 1e3 * 10e-9))
    assert eval_expr("sqrt(16)+2^3+max(1,2)") == pytest.approx(14)
    c = parse("""* params
.param r1=4.7k c1=10n
.param fc={1/(2*pi*r1*c1)} r2={2*r1}
V1 in 0 AC 1
R1 in out {r2}
C1 out 0 {c1}
""")
    assert c.element("R1").value == pytest.approx(9400)


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "(1).__class__", "[1,2]", "lambda: 1",
                                  "a if b else c", "x.y"])
def test_param_evaluator_refuses_code(expr):
    with pytest.raises(NetlistError):
        eval_expr(expr, {"a": 1, "b": 1, "c": 1, "x": 1})


def test_param_errors_are_plain():
    with pytest.raises(NetlistError, match="isn't defined"):
        parse("* p\nV1 a 0 1\nR1 a 0 {rx}\n")
    with pytest.raises(NetlistError, match="division by zero"):
        parse("* p\n.param z=0\nV1 a 0 1\nR1 a 0 {1/z}\n")


SUBS = """* nested subcircuits
.param rbase=1k
.subckt HALF a b params: r=1k
R1 a m {r}
R2 m b {r}
.ends HALF
.subckt QUAD x y params: rq={2*rbase}
X1 x mid HALF r={rq}
X2 mid y HALF
C1 mid 0 1n
.ends
V1 in 0 8
XA in 0 QUAD
XB in xa.mid QUAD rq=500
"""


def test_subckt_flatten_with_nesting():
    c = parse(SUBS)
    names = [e.name for e in c.elements]
    assert "XA.X1.R1" in names and "XB.X2.R2" in names and "XA.C1" in names
    r = c.element("XA.X1.R1")
    assert r.kind == "R" and r.value == pytest.approx(2000) and r.params["sub"] == "XA.X1"
    assert c.element("XB.X1.R1").value == pytest.approx(500)        # instance override
    assert c.element("XA.X2.R1").value == pytest.approx(1000)       # inner default
    assert c.element("XA.X1.R1").nodes == ["in", "xa.x1.m"]
    assert c.element("XA.X2.R2").nodes == ["xa.x2.m", "0"]
    assert "xa.mid" in c.nodes()
    o = op(c)
    # XA: 2k+2k from in to xa.mid, 1k+1k on to ground; XB (500+500, then 1k+1k = 3k) also
    # runs from in to xa.mid, in parallel with XA's upper 4k
    upper = 1 / (1 / 4e3 + 1 / 3e3)
    assert o.v["xa.mid"] == pytest.approx(8 * 2e3 / (2e3 + upper), rel=1e-8)


def test_subckt_with_models_mosfets_and_opamps():
    c = parse("""* buffered boost
.subckt BOOST in out vcc
.model NL NMOS(VTO=2 KP=0.2)
RG in 0 1meg
M1 out in s NL
RS s 0 1k
RD vcc out 4.7k
.ends
.subckt BUF in out vcc vee
X1 in out vcc vee out TL072
.ends
VCC vcc 0 9
VEE vee 0 -9
VIN g 0 3
X1 g d vcc BOOST
X2 d o vcc vee BUF
""")
    assert c.element("X1.M1").model == "X1.NL" and "X1.NL" in c.models
    assert c.element("X2.X1").kind == "X" and c.element("X2.X1").model == "TL072"
    o = op(c)
    assert o.converged and o.devices["X1.M1"]["region"] in ("saturation", "triode")
    assert o.v["o"] == pytest.approx(o.v["d"], abs=1e-3)


def test_opamp_x_lines_still_work_and_error_names_the_subckt():
    parse("* oa\nV1 vcc 0 9\nV2 vee 0 -9\nV3 a 0 1\nX1 a o vcc vee o TL072\n")
    with pytest.raises(NetlistError, match="isn't a .subckt"):
        parse("* oa\nV1 a 0 1\nX1 a b NOPE\n")


@pytest.mark.parametrize("text,msg", [
    ("* s\n.subckt S a b\nR1 a b 1k\nV1 x 0 1\nX1 x 0 S\n", "has no .ends"),
    ("* s\n.subckt S a b\nR1 a b 1k\n.ends\nV1 x 0 1\nX1 x S\n", "connects 1 node but .subckt S has 2 pins"),
    ("* s\n.subckt S a b\nX1 a b S\n.ends\nV1 x 0 1\nX1 x 0 S\n", "uses itself"),
    ("* s\n.subckt S a b params: r=1\nR1 a b {r}\n.ends\nV1 x 0 1\nX1 x 0 S q=2\n", "has no parameter 'q'"),
    ("* s\n.subckt S a b\n.subckt T c\n.ends\n.ends\nV1 x 0 1\n", "can't be defined inside another"),
    ("* s\n.subckt S a 0\nR1 a 0 1\n.ends\nV1 x 0 1\n", "ground (0) can't be a pin"),
    ("* s\n.include models.lib\nV1 a 0 1\nR1 a 0 1k\n", "Paste the model or subcircuit text inline"),
    ("* s\n.lib parts.lib tt\nV1 a 0 1\nR1 a 0 1k\n", "never reads files"),
    ("* m\nM1 d g 2N7000\nV1 d 0 1\n", "needs drain, gate, source"),
    ("* m\nM1 d g 0 0 NQ\n.model NQ NMOS(LEVEL=3 VTO=1)\nV1 d 0 1\n", "LEVEL=3 isn't simulated"),
    ("* m\nM1 d g 0 0 NQ\n.model NQ VDMOS(Vto=3 Kp=1)\nV1 d 0 1\n", "VDMOS"),
    ("* m\nM1 d g 0 0 NQ FOO=2\n.model NQ NMOS(VTO=1)\nV1 d 0 1\n", "unknown MOSFET instance parameter"),
    ("* m\nM1 d g 0 0 Q2N3904\nV1 d 0 1\n.model Q2N3904 NPN(BF=100)\n", "doesn't fit a M element"),
])
def test_parser_error_messages(text, msg):
    with pytest.raises(NetlistError, match=msg.replace("(", r"\(").replace(")", r"\)").replace(".", r"\.")):
        parse(text)


def test_control_blocks_are_skipped_and_nested_depth_is_capped():
    c = parse("* ctl\nV1 a 0 1\nR1 a 0 1k\n.control\nop\nprint all\n.endc\n.end\n")
    assert [e.name for e in c.elements] == ["V1", "R1"]
    deep = ["* deep"]
    for i in range(6):
        deep += [f".subckt S{i} a b", f"X1 a b S{i + 1}" if i < 5 else "R1 a b 1k", ".ends"]
    deep += ["V1 x 0 1", "X1 x 0 S0"]
    with pytest.raises(NetlistError, match="nest deeper than 4"):
        parse("\n".join(deep) + "\n")
    ok = ["* ok"]
    for i in range(4):
        ok += [f".subckt S{i} a b", f"X1 a b S{i + 1}" if i < 3 else "R1 a b 1k", ".ends"]
    ok += ["V1 x 0 1", "X1 x 0 S0"]
    assert parse("\n".join(ok) + "\n").element("X1.X1.X1.X1.R1").value == 1000


def test_tox_derives_kp_like_ngspice():
    c = parse("* t\nV1 d 0 5\nM1 d d 0 0 NT\n.model NT NMOS(VTO=1 TOX=20n UO=500)\n")
    kp = 500e-4 * 3.9 * 8.854214871e-12 / 20e-9
    assert c.models["NT"].params["KP"] == pytest.approx(kp)
    assert np.isfinite(op(c).v["d"])


# --------------------------------------------------------------------------- MOSFETs through every module

BOOST = (__import__("pathlib").Path(__file__).resolve().parents[1] / "library" / "mosfet_boost.cir").read_text(
    encoding="utf-8")
SWITCH = """* High-side switch and an overdriven gate
*@ supply VCC
VCC vcc 0 12
VG g 0 0
M1 load g vcc vcc IRF9510
RL load 0 100
M2 x y 0 0 IRF510
VY y 0 25
RX vcc x 10k
"""


def test_erc_floating_gate_is_an_error():
    from engine import erc
    issues = erc.check(parse("* fg\nVD d 0 5\nM1 d g 0 2N7000\nC1 g x 1n\nVX x 0 1\n"))
    err = [i for i in issues if i["code"] == "no_dc_path"]
    assert err and "gate has no DC path" in err[0]["message"] and err[0]["level"] == "error"
    # bulk tied to source counts as one lead: a source node used nowhere else is dangling
    assert not erc.check(parse(BOOST))


def test_stress_checks_vgs_and_power_against_ratings():
    from engine import stress
    c = parse(SWITCH)
    rows = stress.check(c, op(c))
    msgs = " ".join(r["message"] for r in rows)
    assert "M2 has 25.0 V gate to source" in msgs
    assert not any(r["part"] == "M1" and r["level"] == "error" for r in rows)
    hot = parse(SWITCH.replace("RL load 0 100", "RL load 0 4"))
    assert any(r["part"] == "M1" and r["what"] == "power" and r["level"] == "error"
               for r in stress.check(hot, op(hot)))


def test_bom_kicad_deck_and_compiled_json():
    from engine import export, parts
    c = parse(SWITCH)
    rows = [r for r in parts.bom(c) if r["kind"] == "MOSFET"]
    assert {r["value"] for r in rows} == {"IRF9510", "IRF510"} and all(r["package"] == "TO-220" for r in rows)
    net = export.kicad_netlist(c)
    assert 'part "Q_PMOS_GDS"' in net and "TO-220-3_Vertical" in net
    assert '(node (ref "M1") (pin "1"))' in net        # gate is pin 1 of a GDS part
    deck = export.spice_deck(parse(BOOST))
    assert "M1 d g s s BS170 W=0.0001 L=0.0001" in deck and "NMOS(LEVEL=1" in deck
    assert "CGSO=1.7e-07" in deck                         # 17 pF over the default W = 100 um
    cj = export.compile_json(parse(BOOST))
    m = next(e for e in cj["elements"] if e["name"] == "M1")
    assert m["type"] == "nmos" and len(m["n"]) == 4 and m["p"]["VTO"] == 2.0
    # a flattened subcircuit exports with names SPICE reads by their first letter
    sub = parse("* s\n.subckt ST a b\nR1 a b 1k\n.ends\nV1 x 0 1\nX1 x 0 ST\n")
    assert "RX1_R1 x 0 1k" in export.spice_deck(sub)


def test_exported_opamp_carries_the_anti_windup_clamp():
    from engine import export
    deck = export.spice_deck(parse("* oa\nV1 vcc 0 9\nV2 vee 0 -9\nV3 a 0 1\nX1 a o vcc vee o TL072\n"))
    assert "Bcl p 0 I=" in deck and "tanh(" in deck and ".subckt TL072 inp inn vcc vee out" in deck


def test_schematic_and_stripboard_draw_mosfets():
    from engine import schematic, stripboard
    for text in (BOOST, SWITCH):
        svg = schematic.svg(parse(text))
        assert "schematic unavailable" not in svg and "M1" in svg
    b = stripboard.layout(parse(BOOST))
    assert stripboard.check(b) == []
    steps = " ".join(stripboard.build_steps(b))
    assert "M1 BS170 (TO-92, pins DGS left to right)" in steps
    b2 = stripboard.layout(parse("* sw\nVCC vcc 0 12\nVIN g 0 0\n*@ in VIN\nM1 load g vcc IRF9510\nRL load 0 100\n"))
    assert "metal tab" in " ".join(stripboard.build_steps(b2)) and stripboard.check(b2) == []


def test_montecarlo_varies_threshold_and_kp():
    from engine import montecarlo
    r = montecarlo.run(parse(BOOST), specs=["v(d) in 3..7 V"], n=40, seed=3)
    params = {s["param"] for s in r["sensitivity"]}
    assert "M1.VTO" in params
    assert r["yield_pct"] >= 90


def test_region_spec_accepts_active_for_a_mosfet():
    from engine.specs import Evaluator, evaluate
    c = parse(BOOST)
    want = ["region(M1) == active", "region(M1) == saturation", "region(M1) == triode"]
    got = {r.spec: r.passed for r in evaluate(c, want, Evaluator(c))}
    assert [got[s] for s in want] == [True, True, False]
