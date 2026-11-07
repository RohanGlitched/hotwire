"""Stripboard layouts: every example board must rebuild to exactly the netlist's connectivity.

The connectivity check here is independent of the layout engine's own bookkeeping: it maps every
lead back to circuit terminals ("R1:0" = R1's first node), groups terminals by what the copper,
cuts and links actually join, and compares that partition with the circuit's nodes.
"""
from __future__ import annotations

import copy
import glob
import os
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.netlist import parse  # noqa: E402
from engine.stripboard import (LayoutError, build_steps, check, connectivity, layout,  # noqa: E402
                               offboard, resistor_bands, svg)

EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "docs", "stripboard", "examples", "*.cir")))
_CACHE: dict = {}


def board_for(path):
    if path not in _CACHE:
        circuit = parse(open(path, encoding="utf-8").read())
        _CACHE[path] = (circuit, layout(circuit))
    return _CACHE[path]


def ids(paths):
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def terminal_node(circuit, term):
    name, idx = term.rsplit(":", 1)
    return circuit.element(name).nodes[int(idx)]


def piece_map(board):
    piece = {}
    for i, s in enumerate(board.strips):
        for c in range(s["start"], s["end"] + 1):
            piece[(s["row"], c)] = i
    return piece


def copper_groups(board):
    """Union-find over copper pieces joined by links; returns hole -> group id."""
    piece = piece_map(board)
    parent = list(range(len(board.strips)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k in board.links:
        a, b = piece[tuple(k["a"])], piece[tuple(k["b"])]
        parent[find(a)] = find(b)
    return lambda cell: find(piece[cell]) if cell in piece else None


# --------------------------------------------------------------------------- examples

def test_at_least_eight_examples():
    assert len(EXAMPLES) >= 8


@pytest.mark.parametrize("path", EXAMPLES, ids=ids(EXAMPLES))
def test_connectivity_equals_netlist(path):
    circuit, board = board_for(path)
    group_of = copper_groups(board)
    realised: dict = {}          # circuit terminal -> copper group
    for p in board.parts + board.pads:
        for lead in p.leads:
            g = group_of((lead.row, lead.col))
            assert g is not None, f"{p.ref}.{lead.name} is not on copper"
            for t in lead.terms:
                assert t not in realised, f"terminal {t} appears twice on the board"
                assert terminal_node(circuit, t) == lead.net, f"{t} landed on net {lead.net}"
                realised[t] = g
    # partition of realised terminals by copper == partition by circuit node
    by_copper: dict = {}
    by_node: dict = {}
    for t, g in realised.items():
        by_copper.setdefault(g, set()).add(t)
        by_node.setdefault(terminal_node(circuit, t), set()).add(t)
    assert sorted(map(sorted, by_copper.values())) == sorted(map(sorted, by_node.values()))
    # board-only nets (op-amp ties, NC pins) never touch circuit nets
    for p in board.parts:
        for lead in p.leads:
            if lead.net.startswith("~"):
                g = group_of((lead.row, lead.col))
                assert all(realised[t] != g for t in realised), f"{p.ref}.{lead.name} shorts to a circuit net"
    assert check(board) == []


@pytest.mark.parametrize("path", EXAMPLES, ids=ids(EXAMPLES))
def test_every_onboard_part_placed(path):
    circuit, board = board_for(path)
    off_kinds = set("VIS")
    expected = set()
    for e in circuit.elements:
        if e.kind in off_kinds or e.pot:
            continue
        if e.kind == "X":
            continue
        expected.add(e.name)
    on_board = {p.ref for p in board.parts if p.kind != "DIP"}
    realised = {t.split(":")[0] for p in board.parts + board.pads for l in p.leads for t in l.terms}
    # simulation-only source/load resistors may legitimately move off the board
    for name in expected - on_board:
        assert name in realised and any(name in n for n in board.notes), f"{name} missing from the board"
    # every op-amp half sits in a DIP
    halves = {t.split(":")[0] for p in board.parts if p.kind == "DIP" for l in p.leads for t in l.terms}
    assert halves == {e.name for e in circuit.elements if e.kind == "X"}
    # every terminal of every on-board part is realised exactly once
    for e in circuit.elements:
        if e.name in on_board or e.name in halves:
            for i in range(len(e.nodes)):
                assert f"{e.name}:{i}" in {t for p in board.parts for l in p.leads for t in l.terms}


@pytest.mark.parametrize("path", EXAMPLES, ids=ids(EXAMPLES))
def test_no_shared_holes_and_no_body_overlap(path):
    _, board = board_for(path)
    used = {}
    for p in board.parts + board.pads:
        for l in p.leads:
            cell = (l.row, l.col)
            assert cell not in used, f"{p.ref} and {used[cell]} share {board.hole(*cell)}"
            assert 0 <= l.row < board.height and 0 <= l.col < board.width
            used[cell] = p.ref
    for k in board.links:
        for end in ("a", "b"):
            cell = tuple(k[end])
            assert cell not in used, f"link end and {used.get(cell)} share {board.hole(*cell)}"
            used[cell] = "link"
    for cell in board.cuts:
        assert cell not in used, f"track cut {board.hole(*cell)} has a lead in it"
    owner = {}
    for p in board.parts + board.pads:
        for cell in p.body:
            assert cell not in owner, f"{p.ref} overlaps {owner[cell]} at {board.hole(*cell)}"
            owner[cell] = p.ref
    for k in board.links:
        (r1, c), (r2, _) = k["a"], k["b"]
        for y in range(min(r1, r2), max(r1, r2) + 1):
            assert (y, c) not in owner, f"link crosses {owner.get((y, c))}"
            owner[(y, c)] = "link"


@pytest.mark.parametrize("path", EXAMPLES, ids=ids(EXAMPLES))
def test_package_geometry(path):
    _, board = board_for(path)
    for p in board.parts:
        rows = [l.row for l in p.leads]
        cols = [l.col for l in p.leads]
        if p.kind in ("R", "D", "L"):
            assert abs(rows[0] - rows[1]) + abs(cols[0] - cols[1]) >= 3
            assert rows[0] == rows[1] or cols[0] == cols[1]
        elif p.kind in ("C", "LED"):
            assert abs(rows[0] - rows[1]) + abs(cols[0] - cols[1]) >= 2
        elif p.kind in ("Q", "J"):
            assert len(set(cols)) == 1
            assert sorted(rows) == list(range(min(rows), min(rows) + 3))
            names = "".join(l.name for l in sorted(p.leads, key=lambda l: l.row))
            pins = p.style["pins"]
            # pin 1 at the top means the flat face points left (side +1), else the order is mirrored
            assert names == (pins if p.style["side"] == 1 else pins[::-1])
        elif p.kind == "DIP":
            left = sorted((l for l in p.leads if l.col == min(cols)), key=lambda l: l.row)
            right = sorted((l for l in p.leads if l.col == max(cols)), key=lambda l: l.row)
            assert max(cols) - min(cols) == 3 and len(left) == len(right) == 4
            assert [l.row for l in left] == [l.row for l in right]
            if p.style["rot"] == 0:
                assert [l.name for l in left] == ["1", "2", "3", "4"]
                assert [l.name for l in right] == ["8", "7", "6", "5"]
            else:
                assert [l.name for l in left] == ["5", "6", "7", "8"]
                assert [l.name for l in right] == ["4", "3", "2", "1"]


@pytest.mark.parametrize("path", EXAMPLES, ids=ids(EXAMPLES))
def test_svg_is_valid(path):
    _, board = board_for(path)
    text = svg(board)
    root = ET.fromstring(text)
    assert root.tag.endswith("svg")
    assert root.get("viewBox")
    assert "nan" not in text.lower().replace("nanf", "")
    assert "currentColor" in text
    assert "var(--hotwire-copper" in text and "var(--hotwire-board" in text
    # transparent: no full-canvas background rectangle
    w, h = root.get("viewBox").split()[2:]
    assert f'width="{w}" height="{h}"' not in text.split("<defs>")[1]
    for p in board.parts:
        assert p.ref in text


@pytest.mark.parametrize("path", EXAMPLES, ids=ids(EXAMPLES))
def test_build_steps(path):
    _, board = board_for(path)
    steps = build_steps(board)
    assert steps[0].startswith("1. Cut a piece of stripboard")
    joined = "\n".join(steps)
    for r, c in board.cuts:
        assert board.hole(r, c) in joined
    for p in board.parts:
        assert p.ref in joined
        if p.kind in ("D", "LED", "Q", "J") or p.style.get("cap") == "electrolytic":
            line = next(s for s in steps if s.split(". ", 1)[1].startswith(p.ref + " "))
            assert "Polarity check" in line
    assert any("short" in s for s in steps)
    numbers = [int(s.split(".")[0]) for s in steps]
    assert numbers == list(range(1, len(steps) + 1))
    # cuts come first, then links, then parts low to tall, then off-board wiring
    kinds = [s for s in steps]
    i_cut = next(i for i, s in enumerate(kinds) if "Cut the copper" in s or "No track cuts" in s)
    i_wire = next((i for i, s in enumerate(kinds) if s.split(". ", 1)[1].startswith("Off-board")), len(kinds))
    first_part = min(i for i, s in enumerate(kinds) for p in board.parts
                     if s.split(". ", 1)[1].startswith(p.ref + " "))
    assert i_cut < first_part < i_wire


# --------------------------------------------------------------------------- the checker itself

def test_checker_catches_a_missing_cut_and_a_missing_link():
    path = next(p for p in EXAMPLES if "opamp_clipper" in p)
    _, board = board_for(path)
    assert board.cuts and board.links
    broken = copy.deepcopy(board)
    r, c = broken.cuts[0]
    broken.cuts = broken.cuts[1:]
    left = [s for s in broken.strips if s["row"] == r and s["end"] == c - 1]
    right = [s for s in broken.strips if s["row"] == r and s["start"] == c + 1]
    left[0]["end"] = right[0]["end"]
    broken.strips.remove(right[0])
    assert any("short" in x for x in check(broken))
    broken = copy.deepcopy(board)
    broken.links = broken.links[1:]
    assert any("split" in x for x in check(broken))


def test_connectivity_groups_cover_every_lead():
    _, board = board_for(EXAMPLES[0])
    groups = connectivity(board)
    leads = sum(len(p.leads) for p in board.parts + board.pads)
    assert sum(len(g) for g in groups) == leads


# --------------------------------------------------------------------------- behaviours

def test_op_amp_unused_half_is_tied_off():
    path = next(p for p in EXAMPLES if "opamp_clipper" in p)
    circuit, board = board_for(path)
    u1 = next(p for p in board.parts if p.kind == "DIP")
    pin = {l.name: l for l in u1.leads}
    assert pin["6"].net == pin["7"].net and pin["6"].net.startswith("~")
    assert pin["5"].net == "vref"          # the 4.5 V bias found by name and divider shape
    group_of = copper_groups(board)
    assert group_of((pin["6"].row, pin["6"].col)) == group_of((pin["7"].row, pin["7"].col))
    assert any("tied off" in n for n in board.notes)


def test_single_op_amp_pinout_and_nc_pins_isolated():
    path = next(p for p in EXAMPLES if "sallen_key" in p)
    circuit, board = board_for(path)
    u1 = next(p for p in board.parts if p.kind == "DIP")
    pin = {l.name: l for l in u1.leads}
    x = circuit.element("X1")
    assert pin["3"].net == x.nodes[0] and pin["2"].net == x.nodes[1]
    assert pin["7"].net == x.nodes[2] and pin["4"].net == x.nodes[3] and pin["6"].net == x.nodes[4]
    group_of = copper_groups(board)
    for n in ("1", "5", "8"):
        g = group_of((pin[n].row, pin[n].col))
        others = [l for p in board.parts + board.pads for l in p.leads
                  if l is not pin[n] and group_of((l.row, l.col)) == g]
        assert others == [], f"NC pin {n} touches {others}"


def test_dual_op_amp_both_halves_share_one_package():
    path = next(p for p in EXAMPLES if "overdrive_full" in p)
    circuit, board = board_for(path)
    dips = [p for p in board.parts if p.kind == "DIP"]
    assert len(dips) == 1
    pin = {l.name: l for l in dips[0].leads}
    assert set(pin["8"].terms) == {"X1:2", "X2:2"} and set(pin["4"].terms) == {"X1:3", "X2:3"}


def test_ideal_elements_are_refused():
    text = "*@ in VIN\nVIN in 0 AC 1\nE1 out 0 in 0 10\nR1 out 0 1k\n"
    with pytest.raises(LayoutError, match="ideal elements"):
        layout(parse(text))
    text = "I1 0 a DC 1m\nR1 a 0 1k\n"
    with pytest.raises(LayoutError, match="ideal elements"):
        layout(parse(text))


def test_explicit_model_defaults_and_note():
    text = """*@ title Custom transistor
*@ in VIN
*@ out out
*@ supply VCC
VCC vcc 0 9
VIN in 0 AC 1
C1 in b 100n
R1 vcc b 1meg
Q1 c b 0 MYQ
R2 vcc c 10k
C2 c out 1u
R3 out 0 1meg
.model MYQ NPN(IS=1e-14 BF=200)
"""
    circuit = parse(text)
    board = layout(circuit)
    q = next(p for p in board.parts if p.ref == "Q1")
    assert q.style["pins"] == "EBC"
    assert any("EBC" in n and "MYQ" in n for n in board.notes)
    assert check(board) == []


def test_led_can_be_panel_mounted():
    path = next(p for p in EXAMPLES if "power_led" in p)
    circuit = parse(open(path, encoding="utf-8").read())
    board = layout(circuit, led="pads")
    assert not any(p.kind == "LED" for p in board.parts)
    labels = {p.label for p in board.pads}
    assert {"D2+", "D2−"} <= labels
    assert check(board) == []


def test_offboard_lists_jacks_supply_and_pots():
    path = next(p for p in EXAMPLES if "tone_stack" in p)
    circuit, board = board_for(path)
    items = offboard(circuit)
    kinds = [i["kind"] for i in items]
    assert kinds.count("pot") == 3 and "input jack" in kinds and "output jack" in kinds
    treble = next(i for i in items if i["ref"] == "RV1")
    assert treble["nets"] == ["t", "out", "bt"] and treble["value"] == "A250k"
    # every pot lug net that touches the board has a pad
    pads = {p.label for p in board.pads}
    assert any(l.startswith("RV1") for l in pads)


def test_deterministic():
    circuit = parse(open(EXAMPLES[1], encoding="utf-8").read())
    a, b = layout(circuit), layout(circuit)
    assert a.to_dict() == b.to_dict()


@pytest.mark.parametrize("ohms,bands", [
    (4700, ["yellow", "violet", "red", "gold"]),
    (10e3, ["brown", "black", "orange", "gold"]),
    (1e6, ["brown", "black", "green", "gold"]),
    (470, ["yellow", "violet", "brown", "gold"]),
    (2.2, ["red", "red", "gold", "gold"]),
    (100, ["brown", "black", "brown", "gold"]),
    (82e3, ["grey", "red", "orange", "gold"]),
    (4990, ["yellow", "white", "white", "brown", "brown"]),
])
def test_resistor_colour_code(ohms, bands):
    assert resistor_bands(ohms) == bands
