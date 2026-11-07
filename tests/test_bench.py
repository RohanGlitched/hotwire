"""Tests for the analysis layer: specs, ERC, stress, E-series, tuning, yield, exports, the bench."""
import json
import math

import pytest

from engine import parse
from engine import erc, stress
from engine.export import compile_json, kicad_netlist, netlist_text, spice_deck
from engine.parts import bom, neighbours, snap
from engine.specs import Evaluator, evaluate, parse_spec

RC = """* RC low-pass
*@ in VIN
*@ out out
*@ spec fc in 1.5k..1.7k
*@ spec gain_db(100) > -0.5
VIN in 0 DC 0 AC 1
R1 in out 10k
C1 out 0 10n
"""

CE = """* CE amp
*@ in VIN
*@ out out
*@ supply VCC
*@ spec gain_db(1k) in 20..30
*@ spec region(Q1) == active
*@ spec supply_current < 2m
VIN in 0 DC 0 AC 1
VCC vcc 0 DC 9
C1 in b 1u
R1 vcc b 47k
R2 b 0 10k
Q1 c b e 2N3904
RC vcc c 4.7k
RE1 e e2 220
RE2 e2 0 820
CE e2 0 100u
C2 c out 1u
RL out 0 100k
"""


def test_spec_parse_forms():
    s = parse_spec("fc in 2.8k..3.2k")
    assert (s.metric, s.op, s.lo, s.hi) == ("fc", "in", 2800.0, 3200.0)
    s = parse_spec("thd(1k, 0.2) > 5%")
    assert s.metric == "thd" and s.args == ["1k", "0.2"] and s.lo == 5.0
    s = parse_spec("gain_db(1000) in 24..27 dB")
    assert s.lo == 24 and s.hi == 27
    assert parse_spec("bias(Q1) == active").metric == "region"
    assert parse_spec("supply_current < 2mA").lo == pytest.approx(0.002)


def test_rc_corner_matches_theory():
    c = parse(RC)
    res = {r.metric: r for r in evaluate(c)}
    fc = 1 / (2 * math.pi * 10e3 * 10e-9)
    assert res["fc"].measured == pytest.approx(fc, rel=0.01)
    assert all(r.passed for r in res.values())


def test_failing_spec_explains_itself():
    c = parse(RC)
    r = evaluate(c, ["fc < 1k"])[-1]
    assert not r.passed and r.margin < 0
    assert "corner frequency" in r.explanation and "target" in r.explanation


def test_ce_amp_bias_and_current():
    c = parse(CE)
    res = evaluate(c)
    assert all(r.passed for r in res), [r.explanation for r in res if not r.passed]


def test_erc_finds_floating_and_no_dc_path():
    bad = """* bad
VIN in 0 DC 1
R1 in a 1k
C1 a b 1u
R2 b c 1k
X1 b c vcc 0 out TL072
VCC vcc 0 DC 9
R3 out 0 10k
"""
    issues = erc.check(parse(bad))
    codes = {i["code"] for i in issues}
    assert "no_dc_path" in codes


def test_erc_clean_circuit():
    assert not [i for i in erc.check(parse(CE)) if i["level"] == "error"]


def test_stress_flags_overloaded_resistor():
    hot = """* hot
VCC vcc 0 DC 12
R1 vcc 0 220
"""
    c = parse(hot)
    ev = Evaluator(c)
    rows = stress.check(c, ev.op)
    assert any(r["part"] == "R1" and r["level"] == "error" for r in rows)


def test_e_series():
    assert snap(4650) == 4700
    assert snap(1.03e-7, "E12") == pytest.approx(1e-7)
    assert neighbours(4700, "E12", 1) == [3900.0, 4700.0, 5600.0]


def test_tune_hits_corner():
    from engine.tune import tune
    c = parse(RC.replace("*@ spec fc in 1.5k..1.7k", "*@ spec fc in 3k..3.4k"))
    r = tune(c, ["R1"])
    assert r["all_pass"]
    assert r["values"]["R1"] in (4700.0, 5100.0)


def test_yield_is_deterministic():
    from engine.montecarlo import run
    c = parse(RC)
    a = run(c, n=40, seed=3)
    b = run(c, n=40, seed=3)
    assert a["yield_pct"] == b["yield_pct"]
    assert 0 <= a["yield_pct"] <= 100
    assert a["sensitivity"] and a["sensitivity"][0]["part"] in ("R1", "C1")


def test_netlist_text_roundtrip_with_changes():
    c = parse(RC).with_values({"R1": 4700})
    text = netlist_text(c)
    assert "R1 in out 4.7k" in text
    assert parse(text).element("R1").value == 4700


def test_exports():
    c = parse(CE)
    deck = spice_deck(c, ".op")
    assert ".model 2N3904 NPN" in deck and deck.strip().endswith(".end")
    net = kicad_netlist(c)
    assert '(ref "Q1")' in net and "GND" in net
    j = compile_json(c)
    assert j["version"] == 1 and j["output"] == j["nodes"].index("out") + 1
    json.dumps(j)


def test_bom_groups_and_hardware():
    rows = bom(parse(CE))
    kinds = {r["kind"] for r in rows}
    assert {"Resistor", "Capacitor", "Transistor", "Hardware"} <= kinds
    jack = [r for r in rows if r["value"] == "9 V DC jack"]
    assert jack


def test_bench_check_and_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTWIRE_DATA", str(tmp_path))
    from engine.store import Store
    from engine.workbench import Bench
    b = Bench(Store(tmp_path))
    r1 = b.check(netlist=RC.replace("in 1.5k..1.7k", "< 1k"), name="rc test", note="first try")
    assert r1["verdict"] == "fail" and r1["version"] == 1
    r2 = b.check(netlist=RC, design=r1["design"], note="fixed")
    assert r2["verdict"] in ("pass", "pass_with_warnings") and r2["version"] == 2
    meta = b.store.meta(r1["design"])
    assert [v["verdict"] for v in meta["versions"]][:2] == ["fail", r2["verdict"]]


def test_explain_ranks_the_gain_resistor(tmp_path):
    from engine.store import Store
    from engine.workbench import Bench
    r = Bench(Store(tmp_path)).explain(netlist=CE)
    top = r["most_influential"]["gain_db(1k) in 20..30"]
    assert top and top[0]["part"] in ("RC", "RE1")
    assert "C2" in r["barely_matter"]


def test_guitar_model_and_cleanup():
    from engine.guitar import with_guitar
    c = parse(open("library/silicon_fuzz.cir", encoding="utf-8").read())
    g = with_guitar(c, 0.25)
    assert g.has("LGTR_PU") and g.element(c.input).nodes[0] == "gtr_emf"
    full, low = evaluate(c, ["thd_gv(300, 0.1, 1.0) > 30%", "thd_gv(300, 0.1, 0.25) < 5%"])[-2:]
    assert full.passed and low.passed, (full.explanation, low.explanation)


def test_jfet_active_alias():
    c = parse(open("library/jfet_buffer.cir", encoding="utf-8").read())
    r = evaluate(c, ["region(J1) == active"])[-1]
    assert r.passed
