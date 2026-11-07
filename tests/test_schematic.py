"""Schematic layout: every library and gallery circuit renders cleanly (no overlaps, well-formed SVG)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from engine.netlist import parse
from engine.schematic import layout, svg

ROOT = Path(__file__).resolve().parents[1]
GALLERY = sorted((ROOT / "library").glob("*.cir")) + sorted((ROOT / "docs" / "gallery" / "extra").glob("*.cir"))
NS = "{http://www.w3.org/2000/svg}"


def _parse_svg(text):
    root = ET.fromstring(text)
    assert root.tag == NS + "svg"
    return root


@pytest.mark.parametrize("path", GALLERY, ids=lambda p: p.stem)
def test_gallery_circuit_is_clean(path):
    c = parse(path.read_text(encoding="utf-8"))
    L = layout(c)
    assert L.overlaps() == [], L.overlaps()
    assert L.failed == [], f"unrouted nets: {L.failed}"
    # every part in the netlist is drawn exactly once (pots count once)
    drawn = {p["name"] for p in L.parts}
    for e in c.elements:
        if e.pot:
            assert e.pot in drawn
        elif e.kind in "VI" and (e.name == (c.input or "").upper() or e.name in c.supplies):
            continue
        elif e.kind == "V" and not e.params:
            continue
        else:
            assert e.name in drawn, f"{e.name} missing"
    # labels: reference and value for every drawn component
    texts = " ".join(lb["text"] for lb in L.labels)
    for p in L.parts:
        if p["kind"] not in ("IN", "OUT", "NET", "X"):
            assert p["name"] in texts
    text = svg(c)
    root = _parse_svg(text)
    assert root.find(NS + "title") is not None
    vb = [float(v) for v in root.get("viewBox").split()]
    assert vb[2] > 100 and vb[3] > 60
    assert "currentColor" in text and "var(--hotwire-ink, #1d1d1f)" in text
    if c.input and "0" in c.element(c.input).nodes:      # a floating input source is drawn as a source
        assert ">IN<" in text
    if c.output:
        assert ">OUT<" in text


def test_label_boxes_never_overlap_wires_or_each_other():
    c = parse((ROOT / "library" / "ce_amplifier.cir").read_text(encoding="utf-8"))
    L = layout(c)
    labels = [lb["box"] for lb in L.labels]
    assert labels
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
    assert L.crossings == 0


def test_signal_flows_left_to_right():
    c = parse((ROOT / "library" / "ce_amplifier.cir").read_text(encoding="utf-8"))
    L = layout(c)
    pos = {p["name"]: p for p in L.parts}
    assert pos["IN"]["x"] < pos["C1"]["x"] < pos["Q1"]["x"] < pos["C2"]["x"] < pos["OUT"]["x"]
    # collector resistor above the transistor, emitter leg below
    assert pos["RC"]["y"] < pos["Q1"]["y"] < pos["RE1"]["y"]


def test_supplies_are_flags_and_grounds_are_local():
    c = parse((ROOT / "library" / "ce_amplifier.cir").read_text(encoding="utf-8"))
    text = svg(c)
    assert "+9V" in text
    assert "VCC" not in {p["name"] for p in layout(c).parts}


def test_positive_ground_rail_on_top():
    c = parse((ROOT / "docs" / "gallery" / "extra" / "pnp_fuzz_positive_ground.cir").read_text(encoding="utf-8"))
    L = layout(c)
    flags = [t for t in L.terms if t.kind == "railup"]
    assert flags and all("−" in t.label for t in flags)


def test_electrolytic_polarity_follows_dc():
    c = parse("""*@ title polarity
*@ in VIN
*@ out out
VIN in 0 AC 1
VCC vcc 0 9
C1 in b 10u
R1 vcc b 47k
R2 b 0 10k
Q1 c b 0 QN
RC vcc c 4.7k
C2 out c 10u
RL out 0 100k
.model QN NPN(IS=1e-14 BF=200)
""")
    L = layout(c)
    pos = {P.name: P for P in L.placed}
    # p1 is the + plate: C1 + toward the base (biased), C2 + toward the collector
    assert pos["C1"].part.pins["p1"] == "b"
    assert pos["C2"].part.pins["p1"] == "c"


ODD = {
    "single resistor": "R1 a 0 1k\nV1 a 0 1\n",
    "floating pair": "*@ title floating\nR1 a b 1k\nR2 b 0 1k\nC1 c d 1n\nR3 d 0 1k\n",
    "sources and controlled": """*@ title controlled
*@ in VIN
*@ out o
VIN in 0 AC 1
E1 o 0 in 0 10
G1 x 0 in 0 1m
R1 x 0 1k
S1 o y on
R2 y 0 1k
I1 0 y 1m
""",
    "parallel bundle": """*@ title bundle
*@ in V1
*@ out b
V1 a 0 AC 1
R1 a b 1k
R2 a b 2k
C1 a b 1n
D1 a b DX
D2 b a DX
L1 b 0 10m
.model DX D(IS=1e-14)
""",
    "jfet p-channel": """*@ in V1
*@ out s
V1 g 0 AC 1
VN vn 0 -9
J1 vn g s PJ
RS s 0 10k
.model PJ PJF(VTO=1)
""",
}


@pytest.mark.parametrize("name", list(ODD))
def test_odd_circuits_never_throw(name):
    c = parse(ODD[name])
    text = svg(c)
    _parse_svg(text)
    L = layout(c)
    assert L.overlaps() == []


def test_forty_part_ladder():
    lines = ["*@ title RC ladder", "*@ in VIN", "*@ out n20", "VIN n0 0 AC 1"]
    for i in range(20):
        lines.append(f"R{i} n{i} n{i + 1} 1k")
        lines.append(f"C{i} n{i + 1} 0 10n")
    c = parse("\n".join(lines))
    L = layout(c)
    assert len(L.parts) >= 40
    assert L.overlaps() == [] and L.failed == []
    _parse_svg(svg(c, title=False))


def test_no_title_option_and_theming():
    c = parse((ROOT / "library" / "rc_lowpass.cir").read_text(encoding="utf-8"))
    with_title = svg(c)
    without = svg(c, title=False)
    assert ">RC low-pass" in re.sub(r"<title>.*?</title>", "", with_title)
    assert ">RC low-pass" not in re.sub(r"<title>.*?</title>", "", without)
    assert "var(--hotwire-accent, #c2410c)" in without
    assert "font-family=\"ui-monospace, 'JetBrains Mono', monospace\"" in without
    assert "background" not in without and "<rect" not in without
