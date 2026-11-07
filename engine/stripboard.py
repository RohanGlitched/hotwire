"""Stripboard (veroboard) layouts from a proven circuit.

    from engine import parse
    from engine.stripboard import layout, svg, build_steps, offboard

    board = layout(parse(text))        # automatic placement on a 0.1-inch grid
    open("board.svg", "w").write(svg(board))
    for step in build_steps(board): print(step)

The board model follows how pedal builders draw "vero layouts":

* Copper strips run left to right (rows A, B, C...); holes are numbered from 1
  at the left.  A hole name is row letter + column number: ``C12``.
* Every net sits on one or more contiguous **segments** of strip.  Two nets can
  share a strip when a **track cut** (a hole drilled through the copper)
  separates them.  Cuts always sit in a hole with no lead in it.
* When a net needs more than one segment, insulated **wire links** join them.
* Parts stand across strips: two-lead parts span two strips in one column,
  TO-92 transistors use three adjacent holes in one column (pin order from the
  part library), DIP-8 op-amps straddle cuts on four strips.
* Off-board parts (jacks, DC supply, pots, switches) become labelled **pads**.

How the layout is found:

1. Order the parts by signal flow (breadth-first from the input, not through the
   supply rails); power-only parts go first, next to the DC pads.
2. Decode an ordering with a left-to-right sweep.  Each part takes the cheapest
   free spot within a few columns of the cursor: strips are assigned on the fly
   (a strip is reused after a cut once its net has no parts left to come), a net
   that has to appear on a second strip gets a wire link, and a lookahead charges
   strips that its later parts could not reach.  TO-92 pins need three adjacent
   strips holding their nets in pin order (or mirrored, which turns the flat face
   round); a DIP-8 straddles cuts on four strips.  Candidates are tried in order of
   a cheap lower bound, so most are never simulated.
3. Search: randomised orderings over a range of board heights, then a local search
   (move or swap parts in the ordering, keep any change that does not make the
   board worse).  Score = holes + 1.5 per cut + 12 per link + 1.2 per strip of link
   length.  GND stays on the bottom strip and +V on the top one where it pays off.
4. Verify: rebuild connectivity from the copper pieces, cuts, links and leads alone
   and compare it with the netlist.  A board that fails is never returned.
"""
from __future__ import annotations

import html as _html
import math
import random
import re
from dataclasses import dataclass, field

from .netlist import Circuit, format_value

__all__ = [
    "layout", "svg", "build_steps", "offboard", "check", "connectivity",
    "resistor_bands", "Board", "Placed", "Lead", "LayoutError",
]


class LayoutError(ValueError):
    """The circuit can't be turned into a stripboard (ideal elements, no room...)."""


# =========================================================================== part knowledge

# Used only when engine.parts_data has no entry (or no pin data) for a part.
_FALLBACK = {
    "2N3904": ("TO-92", "EBC"), "2N3906": ("TO-92", "EBC"), "2N2222": ("TO-92", "EBC"),
    "PN2222": ("TO-92", "EBC"), "2N5088": ("TO-92", "EBC"), "2N5089": ("TO-92", "EBC"),
    "MPSA18": ("TO-92", "EBC"), "MPSA13": ("TO-92", "EBC"), "2N4401": ("TO-92", "EBC"),
    "BC549C": ("TO-92", "CBE"), "BC549": ("TO-92", "CBE"), "BC547": ("TO-92", "CBE"),
    "BC548": ("TO-92", "CBE"), "BC550": ("TO-92", "CBE"), "BC559C": ("TO-92", "CBE"),
    "BC557": ("TO-92", "CBE"), "BC560": ("TO-92", "CBE"), "AC128": ("TO-1 (metal can)", "EBC"),
    "J201": ("TO-92", "DSG"), "2N5457": ("TO-92", "DSG"), "2N5458": ("TO-92", "DSG"),
    "MPF102": ("TO-92", "DSG"), "J113": ("TO-92", "DSG"), "2N5952": ("TO-92", "DSG"),
    "1N4148": ("DO-35", "AK"), "1N914": ("DO-35", "AK"), "BAT41": ("DO-35", "AK"),
    "1N34A": ("DO-7 (glass)", "AK"), "1N60": ("DO-7 (glass)", "AK"),
    "1N5817": ("DO-41", "AK"), "1N5818": ("DO-41", "AK"), "1N4001": ("DO-41", "AK"),
    "1N4007": ("DO-41", "AK"),
}
_SINGLE_OPAMPS = {"TL071", "TL081", "LM741", "UA741", "NE5534", "OPA134", "TL061", "LF356", "CA3140"}
_DUAL_OPAMPS = {"TL072", "TL082", "NE5532", "LM358", "RC4558", "JRC4558", "OPA2134", "TL062",
                "LM1458", "MC1458", "LM833", "OPAMP_IDEAL"}


def _parts_table() -> dict:
    try:
        from .parts_data import PARTS
        return PARTS
    except Exception:  # pragma: no cover - parts table missing
        return {}


def _part_entry(name: str) -> dict | None:
    table = _parts_table()
    key = (name or "").upper()
    if key in table:
        return table[key]
    alias = {"LED": "LED_RED", "OPAMP": "OPAMP_IDEAL", "IDEAL": "OPAMP_IDEAL"}.get(key)
    return table.get(alias) if alias else None


def _three_pin(model: str, kind: str, notes: list) -> tuple:
    """(package, pins) for a transistor; pins is the physical order, e.g. 'EBC'."""
    entry = _part_entry(model) or {}
    pins = str(entry.get("pins") or "").upper()
    pkg = entry.get("package")
    want = set("CBE") if kind == "Q" else set("DGS")
    if len(pins) == 3 and set(pins) == want:
        return pkg or "TO-92", pins
    if model.upper() in _FALLBACK and set(_FALLBACK[model.upper()][1]) == want:
        return _FALLBACK[model.upper()]
    default = "EBC" if kind == "Q" else "DGS"
    notes.append(f"{model} has no known pinout (explicit .model part), so the layout assumes "
                 f"{default} (left to right, flat face toward you, leads down). Check the datasheet "
                 f"and flip the part if yours differs.")
    return "TO-92", default


def _diode_package(model: str) -> str:
    entry = _part_entry(model) or {}
    if entry.get("package"):
        return entry["package"]
    if model.upper() in _FALLBACK:
        return _FALLBACK[model.upper()][0]
    return "DO-35"


def _is_led(model: str) -> bool:
    m = (model or "").upper()
    if m.startswith("LED"):
        return True
    entry = _part_entry(m) or {}
    return "LED" in str(entry.get("package", "")).upper()


def _led_colour(model: str) -> str:
    m = (model or "").upper()
    for c in ("GREEN", "YELLOW", "BLUE", "WHITE", "ORANGE"):
        if c in m:
            return c.lower()
    return "red"


def _opamp_kind(model: str) -> tuple:
    """('dual'|'single', part shown on the board, note or None)."""
    m = (model or "").upper()
    entry = _part_entry(m) or {}
    pins = str(entry.get("pins", "")).upper()
    if m == "OPAMP_IDEAL" or entry.get("package") == "virtual":
        return "dual", "TL072", (f"{m} is an ideal op-amp in the simulation; the board uses a TL072 "
                                 "(any dual op-amp with the standard pinout fits)")
    if "7 OUT B" in pins or m in _DUAL_OPAMPS:
        return "dual", m, None
    if "6 OUT" in pins or m in _SINGLE_OPAMPS:
        return "single", m, None
    return "dual", m, (f"{m} is an explicit .model op-amp, so the board assumes the standard dual "
                       "DIP-8 pinout (TL072 style)")


# =========================================================================== resistor colours

_COLOURS = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white"]
_HEX = {"black": "#1b1b1b", "brown": "#7a4a25", "red": "#cf3328", "orange": "#ee8a22",
        "yellow": "#f3d13a", "green": "#3b9a4c", "blue": "#2e62c2", "violet": "#8b4fc2",
        "grey": "#8e8e8e", "white": "#f5f5f5", "gold": "#c9a447", "silver": "#c2c6cb"}


def _mult_colour(exp: int) -> str | None:
    if exp == -1:
        return "gold"
    if exp == -2:
        return "silver"
    if 0 <= exp <= 9:
        return _COLOURS[exp]
    return None


def resistor_bands(ohms: float) -> list:
    """Colour bands for a resistor value: 4 bands (5% gold) when two significant digits are
    exact (all E12/E24 values), else 5 bands (1% brown) for E96-style values."""
    if not ohms or ohms <= 0 or not math.isfinite(ohms):
        return []
    e = math.floor(math.log10(ohms) + 1e-9)
    for digits, tol in ((2, "gold"), (3, "brown")):
        mult = e - (digits - 1)
        sig = round(ohms / 10 ** mult)
        if sig >= 10 ** digits:
            sig //= 10
            mult += 1
        if abs(sig * 10 ** mult - ohms) <= ohms * 2e-3 or digits == 3:
            mc = _mult_colour(mult)
            if mc is None:
                return []
            ds = [int(ch) for ch in str(sig).zfill(digits)]
            return [_COLOURS[d] for d in ds] + [mc, tol]
    return []


# =========================================================================== data model

@dataclass
class Lead:
    name: str          # "1", "2", "A", "K", "+", "-", "E", "B", "C", "D", "G", "S", "1".."8", "" (pad)
    net: str           # net name ("0" is ground; "~..." are board-only tie nets)
    row: int
    col: int
    terms: list = field(default_factory=list)   # circuit terminals realised: "R1:0", "XU1:2", ...


@dataclass
class Placed:
    ref: str
    kind: str          # R C L D LED Q J DIP PAD
    value: str
    part: str
    package: str
    leads: list
    body: list         # (row, col) cells the part covers on the component side
    style: dict = field(default_factory=dict)
    label: str = ""    # pads: text on the tag
    group: str = ""    # pads: in/out/pos/neg/gnd/vext/pot/sw/led


@dataclass
class Board:
    title: str
    width: int
    height: int
    parts: list
    pads: list
    links: list        # {"net", "a": (r, c), "b": (r, c), "colour"}
    cuts: list         # (row, col)
    strips: list       # {"row", "start", "end", "net"}: copper pieces between cuts
    notes: list
    offboard: list
    stats: dict

    def hole(self, r: int, c: int) -> str:
        return f"{_row_name(r)}{c + 1}"

    def to_dict(self) -> dict:
        return {
            "title": self.title, "width": self.width, "height": self.height,
            "size_in": [round(self.width * 0.1, 1), round(self.height * 0.1, 1)],
            "cuts": [self.hole(r, c) for r, c in self.cuts],
            "links": [{"net": _net_name(k["net"]), "from": self.hole(*k["a"]), "to": self.hole(*k["b"])}
                      for k in self.links],
            "parts": [{"ref": p.ref, "kind": p.kind, "value": p.value, "part": p.part,
                       "package": p.package,
                       "leads": {(l.name or str(i + 1)): self.hole(l.row, l.col) for i, l in enumerate(p.leads)}}
                      for p in self.parts],
            "pads": [{"label": p.label, "net": _net_name(p.leads[0].net),
                      "hole": self.hole(p.leads[0].row, p.leads[0].col), "group": p.group}
                     for p in self.pads],
            "offboard": self.offboard, "notes": self.notes, "stats": self.stats,
        }


def _row_name(r: int) -> str:
    s = ""
    r += 1
    while r:
        r, m = divmod(r - 1, 26)
        s = chr(65 + m) + s
    return s


def _net_name(n: str) -> str:
    return "GND" if n == "0" else n


# =========================================================================== preparation

@dataclass
class _Item:
    ref: str
    kind: str
    nets: list
    names: list
    terms: list
    value: str = ""
    part: str = ""
    package: str = ""
    spans: tuple = (0, 0)
    style: dict = field(default_factory=dict)
    label: str = ""
    group: str = ""
    dist: float = 0.0


_RS = re.compile(r"^R(S|SRC|SOURCE|SIG|GTR|GUITAR|PICKUP)\d*$")
_RL = re.compile(r"^R(L|LOAD|AMP)\d*$")


def _cap_value(v: float) -> str:
    return format_value(v).replace("u", "µ")


def _dc_estimate(circuit: Circuit) -> dict:
    """DC node voltages for electrolytic polarity: the real operating point when the
    simulator converges, else a linear estimate (semis as weak conductances)."""
    try:
        from .analyses import op
        r = op(circuit)
        if r.converged:
            return {k: float(v) for k, v in r.v.items()}
    except Exception:
        pass
    try:
        import numpy as np
        nodes = circuit.nodes()
        idx = {n: i for i, n in enumerate(nodes)}
        vs = [e for e in circuit.elements if e.kind == "V"]
        n, m = len(nodes), len(vs)
        A = np.zeros((n + m, n + m))
        b = np.zeros(n + m)

        def g(a, bb, val):
            ia, ib = idx.get(a), idx.get(bb)
            if ia is not None:
                A[ia, ia] += val
            if ib is not None:
                A[ib, ib] += val
            if ia is not None and ib is not None:
                A[ia, ib] -= val
                A[ib, ia] -= val
        for i in range(n):
            A[i, i] += 1e-9
        for e in circuit.elements:
            k = e.kind
            if k == "R":
                g(e.nodes[0], e.nodes[1], 1.0 / max(e.value, 1e-3))
            elif k == "L":
                g(e.nodes[0], e.nodes[1], 1e3)
            elif k == "C":
                g(e.nodes[0], e.nodes[1], 1e-12)
            elif k == "D":
                g(e.nodes[0], e.nodes[1], 1e-7)
            elif k == "Q":
                g(e.nodes[0], e.nodes[2], 1e-6)
                g(e.nodes[1], e.nodes[2], 1e-5)
            elif k in ("J", "M"):
                g(e.nodes[0], e.nodes[2], 1e-4)
            elif k == "S":
                g(e.nodes[0], e.nodes[1], 10.0 if e.params.get("on") else 1e-9)
            elif k == "X":
                o, p = idx.get(e.nodes[4]), idx.get(e.nodes[0])
                if o is not None:
                    A[o, o] += 1.0
                    if p is not None:
                        A[o, p] -= 1.0
        for j, e in enumerate(vs):
            row = n + j
            ia, ib = idx.get(e.nodes[0]), idx.get(e.nodes[1])
            if ia is not None:
                A[ia, row] += 1
                A[row, ia] += 1
            if ib is not None:
                A[ib, row] -= 1
                A[row, ib] -= 1
            b[row] = 0.0 if e.name == circuit.input else (e.value or 0.0)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        return {nd: float(x[i]) for nd, i in idx.items()}
    except Exception:
        return {}


def _find_bias(circuit: Circuit, vcc: str, gnd: str = "0") -> str | None:
    best, score = None, -1.0
    for node in circuit.nodes():
        if node in (vcc, gnd):
            continue
        rs = [e for e in circuit.elements if e.kind == "R" and node in e.nodes and not e.pot]
        to_v = any(set(e.nodes) == {node, vcc} for e in rs)
        to_g = any(set(e.nodes) == {node, gnd} for e in rs)
        if not (to_v and to_g):
            continue
        s = 1.0
        if any(e.kind == "C" and set(e.nodes) == {node, gnd} for e in circuit.elements):
            s += 1.0
        if re.search(r"ref|bias|vb|vr|half|mid", node):
            s += 2.0
        if s > score:
            best, score = node, s
    return best


@dataclass
class _Prep:
    items: list
    pads: list
    notes: list
    rails_top: list
    rails_bottom: list
    remaining: dict
    offboard: list
    title: str
    net_items: dict = field(default_factory=dict)


def _prepare(circuit: Circuit, led: str = "board") -> _Prep:
    notes: list = []
    els = circuit.elements
    for e in els:
        if e.kind in "EG":
            what = "voltage-controlled voltage source" if e.kind == "E" else "voltage-controlled current source"
            raise LayoutError(
                f"{e.name} is an ideal {what}, and the circuit has ideal elements that can't be "
                f"soldered. Replace it with a real stage (an op-amp or transistor) and re-check first.")
        if e.kind == "I":
            raise LayoutError(
                f"{e.name} is an ideal current source, and the circuit has ideal elements that can't "
                f"be soldered. Use a resistor from a supply, or a transistor current source, instead.")

    off: set = set()
    pads: dict = {}
    order: list = []

    def pad(net, label, group, terms):
        key = (net, label)
        if key not in pads:
            pads[key] = _Item("PAD:" + label, "PAD", [net], [""], [[]], label=label, group=group)
            order.append(key)
        pads[key].terms[0].extend(terms)

    # ---- the input source (and a simulation-only source resistor)
    in_node = None
    if circuit.input:
        vin = circuit.element(circuit.input)
        off.add(vin.name)
        p, n = vin.nodes
        terms = [f"{vin.name}:0"]
        in_node = p
        if p != "0":
            attached = [e for e in els if p in e.nodes and e is not vin]
            if (len(attached) == 1 and attached[0].kind == "R" and not attached[0].pot
                    and _RS.match(attached[0].name)):
                r = attached[0]
                k = 1 if r.nodes[0] == p else 0
                in_node = r.nodes[k]
                terms = [f"{r.name}:{k}"]
                off.add(r.name)
                notes.append(f"{r.name} models the guitar's source resistance in the simulation, so it "
                             f"stays off the board (the pickup supplies it).")
        if in_node != "0":
            pad(in_node, "IN", "in", terms)
        if n == "0":
            pad("0", "GND", "gnd", [f"{vin.name}:1"])
        else:
            pad(n, "IN-", "in", [f"{vin.name}:1"])

    # ---- supplies and any other voltage source become power pads
    def vlabel(v):
        s = f"{abs(v):g}V"
        return ("+" if v >= 0 else "−") + s

    for e in els:
        if e.kind != "V" or e.name == circuit.input:
            continue
        off.add(e.name)
        a, b = e.nodes
        v = e.value or 0.0
        if b == "0" or a == "0":
            node = a if b == "0" else b
            val = v if b == "0" else -v
            pad(node, vlabel(val), "pos" if val >= 0 else "neg", [f"{e.name}:{0 if b == '0' else 1}"])
            pad("0", "GND", "gnd", [f"{e.name}:{1 if b == '0' else 0}"])
        else:
            pad(a, f"{e.name}+", "vext", [f"{e.name}:0"])
            pad(b, f"{e.name}−", "vext", [f"{e.name}:1"])
        if e.name not in circuit.supplies:
            notes.append(f"{e.name} is an ideal voltage source in the netlist; on the board it is just a "
                         f"pair of pads for an external {vlabel(v)} supply.")

    # ---- simulation-only load resistor at the output
    out = circuit.output
    for e in els:
        if (e.kind == "R" and not e.pot and out and out != "0" and _RL.match(e.name)
                and set(e.nodes) == {out, "0"}):
            off.add(e.name)
            k = e.nodes.index(out)
            pad(out, "OUT", "out", [f"{e.name}:{k}"])
            pad("0", "GND", "gnd", [f"{e.name}:{1 - k}"])
            notes.append(f"{e.name} models the amplifier input that loads the pedal in the simulation, "
                         f"so it stays off the board.")
    if out and out != "0":
        pad(out, "OUT", "out", [])

    # ---- pots and switches mount on the enclosure
    for name, p in circuit.pots.items():
        off.update({name + "_A", name + "_B"})
        lugs: dict = {}
        for net, k, terms in ((p.a, 1, [f"{name}_A:0"]), (p.w, 2, [f"{name}_A:1", f"{name}_B:0"]),
                              (p.b, 3, [f"{name}_B:1"])):
            ks, ts = lugs.setdefault(net, ([], []))
            ks.append(k)
            ts.extend(terms)
        for net, (ks, ts) in lugs.items():
            label = f"{name}·" + "+".join(str(k) for k in ks)
            pad(net, label, "pot", ts)
            pads[(net, label)].style = {"pot": name, "lugs": ks}
            if len(ks) > 1:
                notes.append(f"{name} lugs {' and '.join(str(k) for k in ks)} share a net: join them on the "
                             f"pot with a short wire and run one wire to the {label} pad.")
    for e in els:
        if e.kind == "S":
            off.add(e.name)
            pad(e.nodes[0], f"{e.name}·1", "sw", [f"{e.name}:0"])
            pad(e.nodes[1], f"{e.name}·2", "sw", [f"{e.name}:1"])

    # ---- on-board parts
    items: list = []
    dc = None
    for e in els:
        if e.name in off or e.kind in "VXS":
            continue
        t0, t1 = f"{e.name}:0", f"{e.name}:1"
        if e.kind == "R":
            items.append(_Item(e.name, "R", list(e.nodes), ["1", "2"], [[t0], [t1]],
                               value=format_value(e.value), part="resistor", package="1/4 W axial",
                               spans=(3, 7), style={"bands": resistor_bands(e.value)}))
        elif e.kind == "L":
            items.append(_Item(e.name, "L", list(e.nodes), ["1", "2"], [[t0], [t1]],
                               value=format_value(e.value) + "H", part="inductor", package="axial choke",
                               spans=(3, 6)))
        elif e.kind == "C":
            v = e.value
            if v < 1e-9:
                items.append(_Item(e.name, "C", list(e.nodes), ["1", "2"], [[t0], [t1]],
                                   value=_cap_value(v), part="ceramic", package="2.5 mm disc",
                                   spans=(2, 4), style={"cap": "ceramic"}))
            elif v < 1e-6:
                items.append(_Item(e.name, "C", list(e.nodes), ["1", "2"], [[t0], [t1]],
                                   value=_cap_value(v), part="film", package="5 mm box",
                                   spans=(2, 5), style={"cap": "film"}))
            else:
                if dc is None:
                    dc = _dc_estimate(circuit)
                a, b = e.nodes
                va, vb = dc.get(a, 0.0) if a != "0" else 0.0, dc.get(b, 0.0) if b != "0" else 0.0
                if abs(va - vb) < 0.02:
                    pos_first = b == "0" or (a != "0" and a in circuit.supplies)
                    notes.append(f"{e.name} sits with almost no DC across it, so its polarity hardly "
                                 f"matters; the layout puts + toward {_net_name(a if pos_first else b)}.")
                else:
                    pos_first = va > vb
                nets = [a, b] if pos_first else [b, a]
                tt = [[t0], [t1]] if pos_first else [[t1], [t0]]
                items.append(_Item(e.name, "C", nets, ["+", "-"], tt, value=_cap_value(v),
                                   part="electrolytic", package="radial can", spans=(2, 4),
                                   style={"cap": "electrolytic", "dv": round(abs(va - vb), 2)}))
        elif e.kind == "D":
            if _is_led(e.model):
                colour = _led_colour(e.model)
                if led == "pads":
                    off.add(e.name)
                    pad(e.nodes[0], f"{e.name}+", "led", [t0])
                    pad(e.nodes[1], f"{e.name}−", "led", [t1])
                    continue
                items.append(_Item(e.name, "LED", list(e.nodes), ["A", "K"], [[t0], [t1]],
                                   value=f"{colour} LED", part=e.model, package="5 mm LED",
                                   spans=(2, 3), style={"colour": colour}))
            else:
                items.append(_Item(e.name, "D", list(e.nodes), ["A", "K"], [[t0], [t1]],
                                   value=e.model, part=e.model, package=_diode_package(e.model),
                                   spans=(3, 6)))
        elif e.kind in "QJM":
            # a MOSFET (TO-92 or TO-220) is a three-lead row like a JFET: D, G, S
            pk = "J" if e.kind == "M" else e.kind
            pkg, pins = _three_pin(e.model, pk, notes)
            idx = {"C": 0, "B": 1, "E": 2} if e.kind == "Q" else {"D": 0, "G": 1, "S": 2}
            nets = [e.nodes[idx[ch]] for ch in pins]
            terms = [[f"{e.name}:{idx[ch]}"] for ch in pins]
            mtype = circuit.models[e.model].type if e.model in circuit.models else ""
            if e.kind == "M" and e.nodes[3] != e.nodes[2]:
                notes.append(f"{e.name}'s body goes to {_net_name(e.nodes[3])} in the netlist, but a discrete "
                             f"MOSFET's body is tied to its source inside the package; the board can only "
                             f"connect the source.")
            items.append(_Item(e.name, pk, nets, list(pins), terms, value=e.model, part=e.model,
                               package=pkg, style={"pins": pins, "type": mtype}))

    # ---- op-amps: pair halves of dual parts into DIP-8 packages
    xs = [e for e in els if e.kind == "X"]
    groups: list = []
    waiting: dict = {}
    for e in xs:
        kind, shown, note = _opamp_kind(e.model)
        if note and note not in notes:
            notes.append(note)
        if kind == "single":
            groups.append(("single", shown, [e]))
            continue
        key = (shown, e.nodes[2], e.nodes[3])
        if key in waiting:
            groups.append(("dual", shown, [waiting.pop(key), e]))
        else:
            waiting[key] = e
    for key, e in waiting.items():
        groups.append(("dual", key[0], [e]))
    groups.sort(key=lambda g: els.index(g[2][0]))
    for i, (kind, shown, halves) in enumerate(groups, 1):
        ref = f"U{i}"
        while any(e.name == ref for e in els):
            ref += "'"
        A = halves[0]
        ta = lambda k, x=A: f"{x.name}:{k}"
        if kind == "single":
            nets = [f"~nc:{ref}.1", A.nodes[1], A.nodes[0], A.nodes[3], f"~nc:{ref}.5", A.nodes[4],
                    A.nodes[2], f"~nc:{ref}.8"]
            terms = [[], [ta(1)], [ta(0)], [ta(3)], [], [ta(4)], [ta(2)], []]
            what = f"{ref} {shown} ({A.name})"
        else:
            if len(halves) == 2:
                B = halves[1]
                tb = lambda k, x=B: f"{x.name}:{k}"
                nets = [A.nodes[4], A.nodes[1], A.nodes[0], A.nodes[3], B.nodes[0], B.nodes[1],
                        B.nodes[4], A.nodes[2]]
                terms = [[ta(4)], [ta(1)], [ta(0)], [ta(3), tb(3)], [tb(0)], [tb(1)], [tb(4)],
                         [ta(2), tb(2)]]
                what = f"{ref} {shown} (A = {A.name}, B = {B.name})"
            else:
                if A.nodes[3] != "0":
                    bias = "0"
                else:
                    bias = _find_bias(circuit, A.nodes[2]) or A.nodes[0]
                tie = f"~{ref}B"
                nets = [A.nodes[4], A.nodes[1], A.nodes[0], A.nodes[3], bias, tie, tie, A.nodes[2]]
                terms = [[ta(4)], [ta(1)], [ta(0)], [ta(3)], [], [], [], [ta(2)]]
                what = f"{ref} {shown} (A = {A.name}, B unused)"
                notes.append(f"Only one half of {ref} ({shown}) is used, so the spare half is tied off "
                             f"safely: pin 5 (IN+ B) to {_net_name(bias)} and pin 7 (OUT B) to pin 6 "
                             f"(IN- B) with a link, making it an idle follower.")
        items.append(_Item(ref, "DIP", nets, [str(k) for k in range(1, 9)], terms, value=shown,
                           part=shown, package="DIP-8", style={"single": kind == "single", "what": what}))

    # ---- drop pads whose net has nothing on the board (wired directly off-board)
    onboard = {n for it in items for n in it.nets}
    pad_items = []
    dropped: dict = {}
    for key in order:
        it = pads[key]
        if it.nets[0] in onboard:
            pad_items.append(it)
        else:
            dropped.setdefault(it.nets[0], []).append(it.label)
    def _who(label):
        if label in ("IN", "OUT"):
            return f"the {'input' if label == 'IN' else 'output'} jack tip"
        if "·" in label:
            ref, lugs = label.split("·", 1)
            return f"{ref} lug {lugs.replace('+', ' and ')}"
        return label
    for net, labels in dropped.items():
        if len(labels) > 1:
            notes.append(f"Nothing on the board touches net {_net_name(net)}: wire "
                         f"{' and '.join(_who(l) for l in labels)} straight to each other.")

    # ---- rails: keep +V at the top, GND at the bottom
    deg: dict = {}
    for it in items + pad_items:
        for n in it.nets:
            deg[n] = deg.get(n, 0) + 1
    top: list = []
    for it in pad_items:
        if it.group == "pos":
            s = it.nets[0]
            cands = [s]
            for e in els:
                if e.name in off:
                    continue
                if e.kind == "D" and e.nodes[0] == s:
                    cands.append(e.nodes[1])
                if e.kind == "R" and s in e.nodes and e.value <= 1000:
                    cands.append(e.nodes[1] if e.nodes[0] == s else e.nodes[0])
            r = max(cands, key=lambda n: (deg.get(n, 0), n == s))
            if r not in top and r != "0":
                top.append(r)
    bottom = ["0"] + [it.nets[0] for it in pad_items if it.group == "neg"]

    remaining: dict = {}
    for it in items + pad_items:
        for n in it.nets:
            remaining[n] = remaining.get(n, 0) + 1

    # ---- signal-flow distances (breadth-first from the input, not through rails)
    rails = set(top) | set(bottom) | {it.nets[0] for it in pad_items if it.group in ("pos", "neg", "vext")}
    adj: dict = {}
    for it in items + pad_items:
        for n in set(it.nets):
            adj.setdefault(n, []).append(it)
    dist = {}
    if in_node and in_node in adj:
        dist[in_node] = 0
        queue = [in_node]
        while queue:
            n = queue.pop(0)
            for it in adj.get(n, []):
                for m in it.nets:
                    if m not in dist and m not in rails:
                        dist[m] = dist[n] + 1
                        queue.append(m)
    for k, it in enumerate(items + pad_items):
        ds = [dist[n] for n in it.nets if n in dist]
        if ds:
            it.dist = min(ds) + 0.01 * (sum(ds) / len(ds))
        elif all(n in rails or n.startswith("~") for n in it.nets):
            it.dist = -1.0
        else:
            it.dist = 50.0 + k * 0.01

    net_items: dict = {}
    for it in items:
        for n in set(it.nets):
            net_items.setdefault(n, []).append(it)
    return _Prep(items, pad_items, notes, top, bottom, remaining, offboard(circuit, led=led),
                 circuit.title, net_items)


# =========================================================================== off-board list

def offboard(circuit: Circuit, led: str = "board") -> list:
    """Parts that mount off the board (jacks, supply, pots, switches) and how they wire up."""
    out: list = []
    if circuit.input:
        vin = circuit.element(circuit.input)
        out.append({"ref": vin.name, "kind": "input jack", "label": "IN", "nets": [vin.nodes[0]],
                    "wiring": "input jack tip to the IN pad, sleeve to GND (through the footswitch "
                              "if the pedal has true bypass)"})
    for e in circuit.elements:
        if e.kind == "V" and e.name != circuit.input:
            v = e.value or 0.0
            out.append({"ref": e.name, "kind": "supply", "label": (f"+{v:g}V" if v >= 0 else f"−{-v:g}V"),
                        "nets": list(e.nodes),
                        "wiring": (f"{v:g} V DC jack or battery: + to the {('+' if v >= 0 else chr(0x2212))}{abs(v):g}V pad, "
                                   "- to GND (a 2.1 mm centre-negative jack has + on the sleeve lug)")})
    if circuit.output and circuit.output != "0":
        out.append({"ref": "OUT", "kind": "output jack", "label": "OUT", "nets": [circuit.output],
                    "wiring": "output jack tip to the OUT pad, sleeve to GND"})
    for name, p in circuit.pots.items():
        taper = {"log": "A", "lin": "B", "revlog": "C"}.get(p.taper, "B")
        out.append({"ref": name, "kind": "pot", "label": p.label,
                    "value": f"{taper}{format_value(p.value)}", "nets": [p.a, p.w, p.b],
                    "wiring": f"{p.label} pot {taper}{format_value(p.value)}: lug 1 = {_net_name(p.a)}, "
                              f"lug 2 (wiper) = {_net_name(p.w)}, lug 3 = {_net_name(p.b)} "
                              "(shaft toward you, lugs down, lug 1 on the left)"})
    for e in circuit.elements:
        if e.kind == "S":
            out.append({"ref": e.name, "kind": "switch", "label": e.name, "nets": list(e.nodes),
                        "wiring": f"SPST switch between {_net_name(e.nodes[0])} and {_net_name(e.nodes[1])}"})
        if e.kind == "D" and led == "pads" and _is_led(e.model):
            out.append({"ref": e.name, "kind": "panel LED", "label": e.name, "nets": list(e.nodes),
                        "wiring": "panel LED: long leg (anode) to the + pad, short leg to the - pad"})
    return out


# =========================================================================== placement engine

class _Sim:
    """Board state for the sweep, with an undo log so candidates can be tried cheaply."""

    def __init__(self, H: int, remaining: dict, prefs: dict, net_items: dict | None = None):
        self.H = H
        self.net_items = net_items or {}
        self.done: set = set()
        self.current = None
        self.rows = [[] for _ in range(H)]       # per row: segments [net, first, last]
        self.holes: dict = {}
        self.body: dict = {}
        self.remaining = dict(remaining)
        self.prefs = prefs                        # net -> preferred row
        self.pref_rows = {r: n for n, r in prefs.items()}
        self.net_rows: dict = {}                  # net -> {row: segment count}
        self.log: list = []
        self.links: list = []
        self.placed: list = []
        self.k = {"maxcol": -1, "cuts": 0, "links": 0, "linklen": 0, "pen": 0.0}
        self.uid = 0

    # ---- undo helpers
    def mark(self) -> int:
        return len(self.log)

    def rollback(self, mark: int) -> None:
        log = self.log
        while len(log) > mark:
            log.pop()()

    def _put(self, d: dict, key, val) -> None:
        d[key] = val
        self.log.append(lambda d=d, key=key: d.pop(key))

    def _inc(self, name: str, delta) -> None:
        old = self.k[name]
        self.k[name] = old + delta
        self.log.append(lambda name=name, old=old: self.k.__setitem__(name, old))

    def _grow(self, c: int) -> None:
        if c > self.k["maxcol"]:
            self._inc("maxcol", c - self.k["maxcol"])

    def _extend(self, seg: list, c: int) -> None:
        if c < seg[1] or c > seg[2]:
            old = (seg[1], seg[2])
            seg[1], seg[2] = min(seg[1], c), max(seg[2], c)
            self.log.append(lambda seg=seg, old=old: seg.__setitem__(slice(1, 3), list(old)))

    def _rem(self, net: str) -> None:
        old = self.remaining[net]
        self.remaining[net] = old - 1
        self.log.append(lambda net=net, old=old: self.remaining.__setitem__(net, old))

    def _append_seg(self, r: int, net: str, c: int) -> None:
        self.rows[r].append([net, c, c])
        rows = self.net_rows.setdefault(net, {})
        rows[r] = rows.get(r, 0) + 1

        def undo(r=r, net=net):
            self.rows[r].pop()
            rr = self.net_rows[net]
            rr[r] -= 1
            if not rr[r]:
                del rr[r]
        self.log.append(undo)

    # ---- queries
    @staticmethod
    def _seg_for(segs: list, c: int, net: str):
        n = len(segs)
        for i in range(n - 1, -1, -1):
            s = segs[i]
            if s[0] != net:
                continue
            if i > 0 and segs[i - 1][2] > c - 2:
                continue
            if i < n - 1 and segs[i + 1][1] < c + 2:
                continue
            return i
        return None

    def _other_open(self, net: str, r: int) -> bool:
        for q in self.net_rows.get(net, ()):
            if q != r and self.rows[q][-1][0] == net:
                return True
        return False

    # ---- mutations (return False on failure; caller rolls back)
    def attach(self, r: int, c: int, net: str, owner, term: bool = True) -> bool:
        if not 0 <= r < self.H or c < 0:
            return False
        key = (r, c)
        if key in self.holes:
            return False
        b = self.body.get(key)
        if b is not None and b != owner:
            return False
        self._put(self.holes, key, owner)
        segs = self.rows[r]
        i = self._seg_for(segs, c, net)
        if i is not None:
            self._extend(segs[i], c)
        elif not self._open(r, c, net, owner):
            return False
        if term:
            self._rem(net)
        self._grow(c)
        return True

    def _open(self, r: int, c: int, net: str, owner) -> bool:
        segs = self.rows[r]
        if segs:
            tail = segs[-1]
            if tail[0] == net or tail[2] > c - 2:
                return False
            m = tail[0]
            if self.remaining[m] > 0 and not self._other_open(m, r):
                return False
            self._inc("cuts", 1)
            if self.prefs.get(m) == r:
                self._inc("pen", 3.0)
        p = self.prefs.get(net)
        if p is not None and p != r:
            self._inc("pen", _W["rail_off"])
        elif p is None and r in self.pref_rows and self.pref_rows[r] != net:
            self._inc("pen", _W["rail_steal"])
        seen = bool(self.net_rows.get(net))
        if not seen and self.net_items:
            self._lookahead(r, net)
        self._append_seg(r, net, c)
        if seen:
            return self._link(r, c, net)
        return True

    def _lookahead(self, r: int, net: str) -> None:
        """Opening a new net on strip r: cost the parts still to come that will need to reach
        an already-open net from here (a span they can't make means a link later)."""
        pen = 0.0
        rows = self.rows
        for it in self.net_items.get(net, ()):
            if id(it) in self.done or it is self.current:
                continue
            k = it.kind
            if k in ("R", "C", "L", "D", "LED"):
                lo, hi = it.spans
                reach = range(lo, hi + 1)
            elif k in ("Q", "J"):
                reach = range(1, 3)
            else:
                continue
            for m in set(it.nets):
                if m == net:
                    continue
                qs = [q for q in self.net_rows.get(m, ()) if rows[q][-1][0] == m]
                if qs and not any(abs(q - r) in reach for q in qs):
                    pen += 0.8
        if pen:
            self._inc("pen", pen)

    def _link(self, r: int, c: int, net: str) -> bool:
        cands = sorted((q for q in self.net_rows.get(net, ()) if q != r), key=lambda q: abs(q - r))
        for q in cands[:4]:
            lo, hi = min(q, r), max(q, r)
            for dt in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
                t = c + dt
                if t < 0 or (r, t) in self.holes or (q, t) in self.holes:
                    continue
                body = self.body
                if any((y, t) in body for y in range(lo, hi + 1)):
                    continue
                if any((y, t) in self.holes for y in range(lo + 1, hi)):
                    continue
                ir = self._seg_for(self.rows[r], t, net)
                iq = self._seg_for(self.rows[q], t, net)
                if ir is None or iq is None:
                    continue
                self.uid += 1
                lid = ("link", self.uid)
                cells = [(y, t) for y in range(lo, hi + 1)]
                for x in cells:
                    body[x] = lid
                self.log.append(lambda cells=cells, body=body: [body.pop(x) for x in cells])
                self._put(self.holes, (r, t), lid)
                self._put(self.holes, (q, t), lid)
                self._extend(self.rows[r][ir], t)
                self._extend(self.rows[q][iq], t)
                self.links.append((net, (q, t), (r, t)))
                self.log.append(self.links.pop)
                self._inc("links", 1)
                self._inc("linklen", hi - lo)
                self._grow(t)
                return True
        return False

    def free(self, cells) -> bool:
        H, body, holes = self.H, self.body, self.holes
        for (r, c) in cells:
            if r < 0 or r >= H or c < 0 or (r, c) in body or (r, c) in holes:
                return False
        return True

    def bound(self, item: _Item, cand):
        """Lower bound on the cost of a candidate, or None when it can't work (no state change)."""
        leads, order = cand[1], cand[3]
        nets = item.nets
        rows, rem, net_rows = self.rows, self.remaining, self.net_rows
        dip = item.kind == "DIP"
        lb = 4.0 * max(0, cand[6] - self.k["maxcol"]) + cand[5]
        for i in order:
            r, c = leads[i]
            net = nets[i]
            segs = rows[r]
            if self._seg_for(segs, c, net) is not None:
                continue
            if segs and not dip:
                tail = segs[-1]
                if tail[0] == net or tail[2] > c - 2:
                    return None
                m = tail[0]
                if rem[m] - nets.count(m) > 0 and not self._other_open(m, r):
                    return None
                lb += 1.2
            if net_rows.get(net) and not dip:
                lb += _W["step_link"]
        return lb

    def place(self, item: _Item, cand) -> bool:
        col, leads, body, order, orient, extra = cand[:6]
        self.uid += 1
        owner = ("part", self.uid)
        self.current = item
        bd = self.body
        for cell in body:
            bd[cell] = owner
        self.log.append(lambda body=body, bd=bd: [bd.pop(x) for x in body])
        self._grow(max(x[1] for x in body))
        for i in order:
            r, c = leads[i]
            if not self.attach(r, c, item.nets[i], owner):
                return False
        self.placed.append((item, cand))
        self.log.append(self.placed.pop)
        self.done.add(id(item))
        self.log.append(lambda i=id(item): self.done.discard(i))
        return True


def _cands(item: _Item, c: int, H: int, sim: "_Sim"):
    """Free candidate placements at column ``c``:
    (col, leads, body cells, attach order, orientation, extra cost, rightmost column)."""
    body_d, holes = sim.body, sim.holes

    def free(r, cc):
        return (r, cc) not in body_d and (r, cc) not in holes

    k = item.kind
    if k == "PAD":
        for r in range(H):
            if not free(r, c):
                continue
            if c == 0:
                yield (c, [(r, c)], [(r, c)], (0,), {}, 0.0, c)
            elif free(r, c + 1):
                yield (c, [(r, c)], [(r, c), (r, c + 1)], (0,), {}, 0.0, c + 1)
    elif k in ("R", "C", "L", "D", "LED"):
        lo, hi = item.spans
        run = [0] * (H + 1)                      # free holes from row r downward in this column
        for r in range(H - 1, -1, -1):
            run[r] = run[r + 1] + 1 if free(r, c) else 0
        for span in range(lo, hi + 1):
            extra = 0.25 * (span - lo)
            for r0 in range(H - span):
                if run[r0] <= span:
                    continue
                r1 = r0 + span
                body = [(y, c) for y in range(r0, r1 + 1)]
                yield (c, [(r0, c), (r1, c)], body, (0, 1), {}, extra, c)
                yield (c, [(r1, c), (r0, c)], body, (0, 1), {}, extra, c)
        # lying along one strip, with a track cut between the leads
        for span in range(lo, min(hi, lo + 1) + 1):
            extra = 0.6 + 0.25 * (span - lo)
            for r in range(H):
                if not all(free(r, c + x) for x in range(span + 1)):
                    continue
                body = [(r, c + x) for x in range(span + 1)]
                yield (c, [(r, c), (r, c + span)], body, (0, 1), {}, extra, c + span)
                yield (c, [(r, c + span), (r, c)], body, (1, 0), {}, extra, c + span)
    elif k in ("Q", "J"):
        run = [0] * (H + 1)
        for r in range(H - 1, -1, -1):
            run[r] = run[r + 1] + 1 if free(r, c) else 0
        for r0 in range(H - 2):
            if run[r0] < 3:
                continue
            for flip in (False, True):
                side = -1 if flip else 1           # pin 1 at the top: flat face left, round side right
                if c + side < 0 or not all(free(y, c + side) for y in range(r0, r0 + 3)):
                    continue
                rows = [r0 + 2, r0 + 1, r0] if flip else [r0, r0 + 1, r0 + 2]
                body = [(y, c) for y in range(r0, r0 + 3)] + [(y, c + side) for y in range(r0, r0 + 3)]
                yield (c, [(rows[i], c) for i in range(3)], body, (0, 1, 2), {"side": side}, 0.0,
                       max(c, c + side))
    elif k == "DIP":
        for r0 in range(H - 3):
            body = [(r0 + y, c + x) for y in range(4) for x in range(4)]
            if not all(free(*cell) for cell in body):
                continue
            for rot in (0, 1):
                leads = [None] * 8
                for p in range(1, 9):
                    if rot == 0:
                        leads[p - 1] = (r0 + p - 1, c) if p <= 4 else (r0 + 8 - p, c + 3)
                    else:
                        leads[p - 1] = (r0 + 4 - p, c + 3) if p <= 4 else (r0 + p - 5, c)
                left = sorted((i for i in range(8) if leads[i][1] == c), key=lambda i: leads[i][0])
                right = sorted((i for i in range(8) if leads[i][1] != c), key=lambda i: leads[i][0])
                yield (c, leads, body, tuple(left + right), {"rot": rot}, 0.0, c + 3)


def _prefs_for(prep: _Prep, H: int) -> dict:
    prefs = {}
    for i, n in enumerate(prep.rails_top):
        prefs[n] = i
    # GND on the bottom strip; a negative rail two strips above it so a decoupling
    # capacitor can bridge the two
    for i, n in enumerate(prep.rails_bottom):
        prefs.setdefault(n, H - 1 if i == 0 else H - 1 - 2 * i)
    return prefs


def _sequence(prep: _Prep, rng: random.Random, noise: float) -> list:
    rank = {"gnd": 0, "pos": 1, "neg": 2, "vext": 3}
    head = sorted((p for p in prep.pads if p.group in rank), key=lambda p: rank[p.group])
    ins = [p for p in prep.pads if p.group == "in"]
    outs = [p for p in prep.pads if p.group == "out"]
    rest = [p for p in prep.pads if p.group not in rank and p.group not in ("in", "out")]
    body = prep.items + rest
    keyed = [(it.dist + (rng.uniform(-1.2, 1.2) if noise else 0.0), i, it) for i, it in enumerate(body)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return head + ins + [t[2] for t in keyed] + outs


def _run(prep: _Prep, seq: list, H: int, rng: random.Random, noise: float):
    sim = _Sim(H, prep.remaining, _prefs_for(prep, H), prep.net_items)
    cursor = 0
    for it in seq:
        best, best_cost = None, math.inf
        back = _W["back"]
        for lo, hi in ((cursor - back, cursor + 2), (cursor + 3, cursor + 7), (cursor + 8, cursor + 16)):
            pool = []
            for c in range(max(0, lo), hi + 1):
                near = 0.04 * abs(c - cursor)
                for cand in _cands(it, c, H, sim):
                    lb = sim.bound(it, cand)
                    if lb is not None:
                        pool.append((lb + near, near, len(pool), cand))
            pool.sort(key=lambda t: (t[0], t[2]))
            for lb, near, _, cand in pool:
                if lb >= best_cost:
                    break
                k0 = dict(sim.k)
                m = sim.mark()
                if sim.place(it, cand):
                    k1 = sim.k
                    cost = (4.0 * (k1["maxcol"] - k0["maxcol"]) + 1.2 * (k1["cuts"] - k0["cuts"])
                            + _W["step_link"] * (k1["links"] - k0["links"]) + 0.4 * (k1["linklen"] - k0["linklen"])
                            + (k1["pen"] - k0["pen"]) + cand[5] + near)
                    if noise:
                        cost += noise * rng.random()
                    if cost < best_cost:
                        best, best_cost = cand, cost
                sim.rollback(m)
            if best is not None:
                break
        if best is None:
            return None
        if not sim.place(it, best):  # pragma: no cover - same move succeeded a moment ago
            return None
        cursor = max(cursor, best[0])
    return sim


# Search weights.  step_*: per-move costs in the sweep; cut/link/linklen: board score on top
# of its area (holes); back: how many columns behind the cursor a part may drop into.
_W = {"step_link": 5.0, "cut": 1.5, "link": 12.0, "linklen": 1.2, "back": 5, "rail_off": 0.8,
      "rail_steal": 1.0}


def _score(sim: _Sim, max_width: int) -> tuple:
    used = [r for r in range(sim.H) if sim.rows[r]]
    h = (max(used) - min(used) + 1) if used else 0
    cols = [c for (_, c) in sim.body] + [c for (_, c) in sim.holes]
    w = (max(cols) - min(cols) + 1) if cols else 0
    cuts = sum(max(0, len(s) - 1) for s in sim.rows)
    score = w * h + _W["cut"] * cuts + _W["link"] * sim.k["links"] + _W["linklen"] * sim.k["linklen"]
    if w > max_width:
        score += 1000 + 50 * (w - max_width)
    return score, w, h, cuts


def layout(circuit: Circuit, max_width: int = 30, *, led: str = "board", seed: int = 1,
           effort: float = 1.0) -> Board:
    """Place ``circuit`` on stripboard and return a verified :class:`Board`.

    ``max_width`` is the widest board (in holes) the search aims for.  ``led="pads"`` puts
    LEDs off the board (panel-mounted) instead of on it.  Deterministic for a given ``seed``.
    """
    if not isinstance(circuit, Circuit):
        raise LayoutError("layout() needs a parsed Circuit (engine.netlist.parse)")
    prep = _prepare(circuit, led=led)
    if not prep.items:
        raise LayoutError("nothing to put on the board: every part is off-board")
    n = len(prep.items) + len(prep.pads)
    need = 4 if any(it.kind == "DIP" for it in prep.items) else 3
    h_lo = max(need, len(prep.rails_top) + 2 * len(prep.rails_bottom), 4)
    attempts = max(4, int(round(effort * min(14, 200 / max(n, 1) + 4))))
    # stage 1: randomised signal-flow orderings over a range of board heights
    pool: list = []
    best_sc = best_h = best_w = None
    for H in range(h_lo, h_lo + 24):
        found_here = False
        for a in range(attempts):
            rng = random.Random(seed * 100003 + H * 1009 + a)
            seq = _sequence(prep, rng, 0.0 if a == 0 else 1.0)
            sim = _run(prep, seq, H, rng, 0.0)
            if sim is None:
                continue
            found_here = True
            sc = _score(sim, max_width)
            pool.append((sc, H, seq, sim))
            if best_sc is None or sc[0] < best_sc:
                best_sc, best_h, best_w = sc[0], sc[2], sc[1]
        # stop a few strips past the best height, unless the best board is still too wide
        if pool and found_here and H >= best_h + 3 and (best_w <= max_width or H >= h_lo + 23):
            break
    best = None
    if pool:
        pool.sort(key=lambda t: t[0][0])
        # stage 2: local search on the best orderings (move / swap items, keep improvements)
        rng = random.Random(seed * 7919 + 17)
        iters = max(10, int(effort * min(90, 2400 / max(n, 1))))
        starts = []
        for t in pool:
            if all(t[1] != u[1] or abs(t[0][0] - u[0][0]) > 1e-9 for u in starts):
                starts.append(t)
            if len(starts) == 3:
                break
        best = pool[0]
        for sc, H, seq, sim in starts:
            cur = (sc, H, seq, sim)
            for _ in range(iters // len(starts)):
                s2 = list(cur[2])
                i, j = rng.randrange(len(s2)), rng.randrange(len(s2))
                if rng.random() < 0.5:
                    s2.insert(j, s2.pop(i))
                else:
                    s2[i], s2[j] = s2[j], s2[i]
                H2 = cur[1]
                if rng.random() < 0.12:
                    H2 = max(h_lo, H2 + rng.choice((-1, 1)))
                sim2 = _run(prep, s2, H2, rng, 0.0)
                if sim2 is None:
                    continue
                sc2 = _score(sim2, max_width)
                if sc2[0] <= cur[0][0]:
                    cur = (sc2, H2, s2, sim2)
            if cur[0][0] < best[0][0]:
                best = cur
        best = (best[0], best[3])
    if best is None:
        raise LayoutError("could not place this circuit on a stripboard (too many nets at once); "
                          "split it into smaller boards")
    board = _finalize(best[1], prep, max_width)
    problems = check(board)
    if problems:  # pragma: no cover - the engine never returns a board that fails its own check
        raise LayoutError("internal layout check failed: " + "; ".join(problems[:5]))
    return board


_LINK_COLOURS = ["#2f6db3", "#3a9a5b", "#c23b3b", "#d39b1d", "#7a4fb0", "#2b2b2b"]


def _finalize(sim: _Sim, prep: _Prep, max_width: int) -> Board:
    used = [r for r in range(sim.H) if sim.rows[r]]
    r_off = min(used)
    H = max(used) - r_off + 1
    cols = [c for (_, c) in sim.body] + [c for (_, c) in sim.holes]
    c_off = min(cols)
    W = max(cols) - c_off + 1

    def rc(cell):
        return (cell[0] - r_off, cell[1] - c_off)

    body = {rc(k): v for k, v in sim.body.items()}
    cuts: list = []
    strips: list = []
    for r in range(sim.H):
        segs = sim.rows[r]
        if r < r_off or r > r_off + H - 1:
            continue
        rr = r - r_off
        if not segs:
            strips.append({"row": rr, "start": 0, "end": W - 1, "net": None})
            continue
        bounds = []
        for a, b in zip(segs, segs[1:]):
            lo, hi = a[2] + 1 - c_off, b[1] - 1 - c_off
            pick = next((x for x in range(lo, hi + 1) if (rr, x) not in body), lo)
            bounds.append(pick)
            cuts.append((rr, pick))
        edges = [-1] + bounds + [W]
        for i, s in enumerate(segs):
            strips.append({"row": rr, "start": edges[i] + 1, "end": edges[i + 1] - 1, "net": s[0]})

    parts, pads = [], []
    for item, cand in sim.placed:
        col, leads, cells, order, orient = cand[:5]
        lead_objs = [Lead(item.names[i], item.nets[i], *rc(leads[i]), terms=list(item.terms[i]))
                     for i in range(len(leads))]
        style = dict(item.style)
        style.update(orient)
        p = Placed(item.ref, item.kind, item.value, item.part, item.package, lead_objs,
                   [rc(x) for x in cells], style, item.label, item.group)
        (pads if item.kind == "PAD" else parts).append(p)
    links = []
    for i, (net, a, b) in enumerate(sim.links):
        links.append({"net": net, "a": rc(a), "b": rc(b), "colour": _LINK_COLOURS[i % len(_LINK_COLOURS)]})

    notes = list(prep.notes)
    if W > max_width:
        notes.append(f"The board is {W} holes wide, wider than the {max_width}-hole target: the circuit "
                     f"needs that many columns at this height.")
    stats = {"width": W, "height": H, "holes": W * H, "cuts": len(cuts), "links": len(links),
             "parts": len(parts), "pads": len(pads),
             "size_in": f"{W * 0.1:.1f} x {H * 0.1:.1f} in",
             "size_mm": f"{W * 2.54:.1f} x {H * 2.54:.1f} mm"}
    return Board(prep.title, W, H, parts, pads, links, sorted(cuts), strips, notes, prep.offboard, stats)


# =========================================================================== verification

def _find(parent: dict, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def connectivity(board: Board) -> list:
    """Rebuild electrical groups from the board alone (copper pieces, cuts, links, leads).

    Returns a list of groups; each group is a set of (owner ref, lead name, declared net).
    """
    piece_of: dict = {}
    for i, s in enumerate(board.strips):
        for c in range(s["start"], s["end"] + 1):
            piece_of[(s["row"], c)] = i
    parent = {i: i for i in range(len(board.strips))}
    for k in board.links:
        a, b = piece_of.get(tuple(k["a"])), piece_of.get(tuple(k["b"]))
        if a is None or b is None:
            continue
        ra, rb = _find(parent, a), _find(parent, b)
        parent[ra] = rb
    groups: dict = {}
    for p in board.parts + board.pads:
        for lead in p.leads:
            i = piece_of.get((lead.row, lead.col))
            key = ("void", p.ref, lead.name) if i is None else _find(parent, i)
            groups.setdefault(key, set()).add((p.ref, lead.name, lead.net))
    return list(groups.values())


def check(board: Board) -> list:
    """Every physical and electrical rule a builder relies on; returns a list of problems."""
    problems: list = []
    W, H = board.width, board.height
    holes: dict = {}
    cutset = set(board.cuts)

    def use(cell, who):
        r, c = cell
        if not (0 <= r < H and 0 <= c < W):
            problems.append(f"{who} is off the board at {cell}")
        if cell in holes:
            problems.append(f"{who} and {holes[cell]} share hole {board.hole(*cell)}")
        holes[cell] = who
        if cell in cutset:
            problems.append(f"{who} sits in the track cut at {board.hole(*cell)}")

    for p in board.parts + board.pads:
        for lead in p.leads:
            use((lead.row, lead.col), f"{p.ref}.{lead.name or 'pad'}")
    for k in board.links:
        use(tuple(k["a"]), f"link {_net_name(k['net'])}")
        use(tuple(k["b"]), f"link {_net_name(k['net'])}")

    # bodies never overlap (links count as bodies along their length)
    owner: dict = {}
    for p in board.parts + board.pads:
        for cell in p.body:
            if cell in owner and owner[cell] != p.ref:
                problems.append(f"{p.ref} overlaps {owner[cell]} at {board.hole(*cell)}")
            owner[cell] = p.ref
    for i, k in enumerate(board.links):
        (r1, c1), (r2, c2) = k["a"], k["b"]
        if c1 != c2:
            problems.append(f"link {i} is not vertical")
        for y in range(min(r1, r2), max(r1, r2) + 1):
            if (y, c1) in owner:
                problems.append(f"link {_net_name(k['net'])} crosses {owner[(y, c1)]} at {board.hole(y, c1)}")
            owner[(y, c1)] = f"link{i}"
        for y in range(min(r1, r2) + 1, max(r1, r2)):
            if (y, c1) in holes:
                problems.append(f"link {_net_name(k['net'])} runs over a used hole at {board.hole(y, c1)}")

    # geometry of each package
    for p in board.parts:
        rows = [l.row for l in p.leads]
        cols = [l.col for l in p.leads]
        if p.kind in ("R", "C", "L", "D", "LED"):
            span = abs(rows[0] - rows[1]) + abs(cols[0] - cols[1])
            need = {"R": 3, "L": 3, "D": 3, "C": 2, "LED": 2}[p.kind]
            if (cols[0] != cols[1] and rows[0] != rows[1]) or span < need:
                problems.append(f"{p.ref} lead spacing {span} is under {need} holes")
        elif p.kind in ("Q", "J"):
            if len(set(cols)) != 1 or sorted(rows) != list(range(min(rows), min(rows) + 3)):
                problems.append(f"{p.ref} pins are not three adjacent holes in one column")
        elif p.kind == "DIP":
            left = [l for l in p.leads if l.col == min(cols)]
            if len(left) != 4 or max(cols) - min(cols) != 3:
                problems.append(f"{p.ref} is not a 0.3-inch DIP-8 footprint")

    # electrical: each group carries exactly one net, each net forms exactly one group
    seen: dict = {}
    for g in connectivity(board):
        nets = {n for (_, _, n) in g}
        if len(nets) != 1:
            problems.append("short between nets " + ", ".join(sorted(_net_name(n) for n in nets)))
        for n in nets:
            seen[n] = seen.get(n, 0) + 1
    for n, k in seen.items():
        if k > 1:
            problems.append(f"net {_net_name(n)} is split into {k} unconnected pieces")
    return problems


# =========================================================================== build steps

def _lead(board: Board, p: Placed, name: str) -> str:
    for l in p.leads:
        if l.name == name:
            return board.hole(l.row, l.col)
    return "?"


def _flat_dir(p: Placed) -> str:
    return "left" if p.style.get("side", 1) == 1 else "right"


def build_steps(board: Board) -> list:
    """A numbered build order a first-time builder can follow: cuts, links, parts from low
    to tall with a polarity check for each polarised part, off-board wiring, then a short test."""
    b = board

    def h(r, c):
        return f"hole {b.hole(r, c)}"

    def at(p, name):
        l = next(l for l in p.leads if l.name == name)
        return h(l.row, l.col)

    steps: list = []
    last = _row_name(b.height - 1)
    steps.append(f"Cut a piece of stripboard {b.width} holes wide by {b.height} strips "
                 f"({b.stats['size_in']}), strips running left to right. Strips are lettered A to {last} "
                 f"from the top and holes numbered 1 to {b.width} from the left, so hole C12 is strip C, "
                 "hole 12. The layout is drawn from the component side.")
    if b.cuts:
        steps.append(f"Cut the copper at {len(b.cuts)} hole{'s' if len(b.cuts) != 1 else ''}: "
                     + ", ".join(b.hole(r, c) for r, c in b.cuts)
                     + ". On the copper side left and right are mirrored, so count from the same corner. "
                       "Twist a 3.5 mm drill bit by hand until the copper ring is gone, then hold the board "
                       "to a light to make sure no sliver bridges the cut.")
    else:
        steps.append("No track cuts are needed.")
    if b.links:
        steps.append(f"Fit {len(b.links)} wire link{'s' if len(b.links) != 1 else ''} (insulated solid-core "
                     "wire, lying flat): " + ", ".join(f"{b.hole(*k['a'])} to {b.hole(*k['b'])}"
                                                       for k in b.links) + ".")
    order = {"R": 0, "L": 1, "D": 2, "DIP": 3, "Cc": 4, "Cf": 5, "Q": 6, "J": 6, "Ce": 7, "LED": 8}

    def key(p):
        k = p.kind
        if k == "C":
            k = {"ceramic": "Cc", "film": "Cf", "electrolytic": "Ce"}[p.style.get("cap", "film")]
        return (order.get(k, 9), min(l.col for l in p.leads), _natural(p.ref))

    for p in sorted(b.parts, key=key):
        a = h(p.leads[0].row, p.leads[0].col)
        z = h(p.leads[-1].row, p.leads[-1].col)
        flat = p.leads[0].row == p.leads[-1].row and p.kind in ("R", "L", "D", "C", "LED")
        lie = " lying along the strip over the cut" if flat else ""
        if p.kind == "R":
            bands = " ".join(p.style.get("bands", []))
            steps.append(f"{p.ref} {p.value} resistor ({bands}) from {a} to {z}{lie}.")
        elif p.kind == "L":
            steps.append(f"{p.ref} {p.value} inductor from {a} to {z}{lie}.")
        elif p.kind == "D":
            steps.append(f"{p.ref} {p.part} diode{lie}: anode at {at(p, 'A')}, cathode at {at(p, 'K')}. "
                         f"Polarity check: the band goes at {at(p, 'K')}.")
        elif p.kind == "DIP":
            pin1 = next(l for l in p.leads if l.name == "1")
            notch = "top" if p.style.get("rot", 0) == 0 else "bottom"
            steps.append(f"{p.ref} 8-pin DIP socket with pin 1 at {h(pin1.row, pin1.col)}, notch toward the "
                         f"{notch} of the board. The chip goes in last.")
        elif p.kind == "C":
            cap = p.style.get("cap")
            if cap == "electrolytic":
                steps.append(f"{p.ref} {p.value}F electrolytic, 16 V or more{lie}: + at {at(p, '+')}, "
                             f"- at {at(p, '-')}. Polarity check: the stripe (-) faces {at(p, '-')}.")
            else:
                steps.append(f"{p.ref} {p.value}F {cap} capacitor from {a} to {z}{lie}.")
        elif p.kind in ("Q", "J"):
            pins = ", ".join(f"{l.name} at {b.hole(l.row, l.col)}" for l in p.leads)
            pkg = str(p.package or "")
            if "TO-220" in pkg:
                check = (f"Polarity check: printed face toward the {_flat_dir(p)}, metal tab away from it "
                         "(the tab is the drain: keep it clear of other parts, or bolt on a heatsink with "
                         "an insulating pad).")
            elif "TO-1" in pkg or "TO-18" in pkg or "TO-5" in pkg or "can" in pkg.lower():
                check = ("Polarity check: a metal can has no flat face; find the collector by the red dot "
                         "or the wider gap between leads (or the tab, which marks the emitter on TO-18), and "
                         "match the pins to the holes above with a meter in diode mode before soldering.")
            else:
                check = f"Polarity check: flat face toward the {_flat_dir(p)}."
            steps.append(f"{p.ref} {p.part} ({pkg}, pins {p.style.get('pins')} left to right): {pins}. {check}")
        elif p.kind == "LED":
            steps.append(f"{p.ref} {p.value}{lie}: anode (long leg) at {at(p, 'A')}, cathode (short leg, "
                         f"flat side) at {at(p, 'K')}. Polarity check: flat side toward {at(p, 'K')}.")
    wires = []
    for p in b.pads:
        if p.group == "pot":
            continue
        where = h(p.leads[0].row, p.leads[0].col)
        what = {"in": "input jack tip (through the footswitch for true bypass)",
                "out": "output jack tip (through the footswitch for true bypass)",
                "gnd": "jack sleeves and the DC jack ground",
                "pos": "DC jack + (the sleeve lug on a 2.1 mm centre-negative jack) or the battery clip +",
                "neg": "negative supply", "vext": "external supply", "sw": "switch lug"}.get(p.group, "wire")
        if p.group == "led":
            what = "panel LED anode (long leg)" if p.label.endswith("+") else "panel LED cathode (short leg)"
        wires.append(f"{p.label} pad at {where} to the {what}")
    for item in b.offboard:
        if item["kind"] == "pot":
            lugs = []
            for i in (1, 2, 3):
                hit = next((p for p in b.pads if p.style.get("pot") == item["ref"]
                            and i in p.style.get("lugs", ())), None)
                net = item["nets"][i - 1]
                jack = next((o for o in b.offboard if o["kind"] in ("input jack", "output jack")
                             and o["nets"] == [net]), None)
                lugs.append(f"lug {i} to {b.hole(hit.leads[0].row, hit.leads[0].col)}" if hit
                            else f"lug {i} to the {jack['kind']} tip" if jack
                            else f"lug {i} to {_net_name(net)} off the board")
            wires.append(f"{item['ref']} {item['label']} pot ({item['value']}, shaft toward you, lugs down, "
                         f"lug 1 on the left): " + ", ".join(lugs))
    if wires:
        steps.append("Off-board wiring, stranded wire about 10 cm long: " + "; ".join(wires) + ".")
    pos = [p for p in b.pads if p.group == "pos"]
    gnd = [p for p in b.pads if p.group == "gnd"]
    if pos and gnd:
        steps.append(f"Before power: with a meter on ohms, check for shorts between {pos[0].label} "
                     f"({b.hole(pos[0].leads[0].row, pos[0].leads[0].col)}) and GND "
                     f"({b.hole(gnd[0].leads[0].row, gnd[0].leads[0].col)}). It must not read near 0 ohms. "
                     "Then check the neighbouring strips at every cut for solder bridges.")
    else:
        steps.append("Before connecting anything: with a meter on ohms, check for shorts between neighbouring "
                     "strips at every cut, and that IN and OUT do not read near 0 ohms to GND.")
    ics = [p for p in b.parts if p.kind == "DIP"]
    if ics:
        steps.append("Fit the chip" + ("s" if len(ics) > 1 else "") + ": "
                     + ", ".join(f"{p.part} into the {p.ref} socket, notch matching" for p in ics)
                     + ". Power up and measure the supply at the V+ pin ("
                     + ", ".join(f"{p.ref} pin {7 if p.style.get('single') else 8}" for p in ics)
                     + ") before playing.")
    return [f"{i}. {s}" for i, s in enumerate(steps, 1)]


# =========================================================================== SVG

P = 30.0
_FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
_INK = "#2a2118"
_BOARD = "var(--hotwire-board, #e9dcc0)"
_COPPER = "var(--hotwire-copper, #c48a5a)"
_PAD_COLOURS = {"in": "#2e8b57", "out": "#2e8b57", "pos": "#c0392b", "neg": "#2c5da8",
                "gnd": "#3a3f45", "vext": "#b0651e", "pot": "#7a52b3", "sw": "#a97a16",
                "led": "#c0392b"}


def _f(x: float) -> str:
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _esc(s: str) -> str:
    return _html.escape(str(s), quote=True)


def _text_w(s: str, size: float) -> float:
    w = 0.0
    for ch in s:
        if ch in "il.,:;|'!·":
            w += 0.28
        elif ch in "mwMW":
            w += 0.86
        elif ch.isupper() or ch.isdigit():
            w += 0.62
        else:
            w += 0.54
    return w * size


class _Svg:
    def __init__(self, board: Board):
        self.b = board
        self.out: list = []
        left_tags = [p for p in board.pads if p.leads[0].col == 0]
        tagw = max((_text_w(p.label, 9) + 12 for p in left_tags), default=0)
        self.X0 = 22 + (tagw + 22 if left_tags else 12)
        self.Y0 = 84.0
        self.rects: list = []
        self.labels: list = []
        self.joints: list = []

    def cx(self, c: float) -> float:
        return self.X0 + c * P + P / 2

    def cy(self, r: float) -> float:
        return self.Y0 + r * P + P / 2

    def add(self, s: str) -> None:
        self.out.append(s)


def svg(board: Board) -> str:
    """Component-side view of the board as a standalone, theme-aware SVG."""
    b = board
    s = _Svg(b)
    W, H = b.width, b.height
    bw, bh = W * P, H * P
    X0, Y0 = s.X0, s.Y0

    # ---- legend content first (decides the canvas width)
    entries = _legend_entries(b)
    right = X0 + bw + 34
    width = max(right + 12, X0 + 600.0)
    ncols = max(2, int((width - X0 - 10) // 228))
    col_w = (width - X0 - 10) / ncols
    rows_per_col = math.ceil(len(entries) / ncols) if entries else 0
    leg_y = Y0 + bh + 40
    legend_h = 64 + rows_per_col * 17
    height = leg_y + legend_h + 10

    a = s.add
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(width)} {_f(height)}" '
      f'width="{_f(width)}" height="{_f(height)}" role="img" aria-labelledby="hw-sb-title" '
      f'font-family="{_FONT}" style="color: var(--hotwire-ink, currentColor)">')
    a(f'<title id="hw-sb-title">{_esc(b.title)}: stripboard layout, {W} x {H} holes</title>')
    a('<defs>'
      '<linearGradient id="hwCu" x1="0" y1="0" x2="0" y2="1">'
      '<stop offset="0" stop-color="#ffffff" stop-opacity="0.22"/>'
      '<stop offset="0.5" stop-color="#ffffff" stop-opacity="0"/>'
      '<stop offset="1" stop-color="#000000" stop-opacity="0.12"/></linearGradient>'
      '<linearGradient id="hwBodyV" x1="0" y1="0" x2="1" y2="0">'
      '<stop offset="0" stop-color="#ffffff" stop-opacity="0.35"/>'
      '<stop offset="0.45" stop-color="#ffffff" stop-opacity="0"/>'
      '<stop offset="1" stop-color="#000000" stop-opacity="0.22"/></linearGradient>'
      '<radialGradient id="hwLed" cx="0.38" cy="0.35" r="0.7">'
      '<stop offset="0" stop-color="#ffffff" stop-opacity="0.75"/>'
      '<stop offset="0.35" stop-color="#ffffff" stop-opacity="0.08"/>'
      '<stop offset="1" stop-color="#000000" stop-opacity="0.25"/></radialGradient>'
      '<filter id="hwShadow" x="-20%" y="-20%" width="140%" height="140%">'
      '<feDropShadow dx="0.8" dy="1.4" stdDeviation="1.1" flood-color="#000" flood-opacity="0.35"/></filter>'
      '</defs>')
    # ---- title
    st = b.stats
    a(f'<text x="{_f(X0)}" y="30" font-size="17" font-weight="650" fill="currentColor">{_esc(b.title)}</text>')
    sub = (f"Stripboard {W} × {H} holes ({st['size_in']}) · {len(b.parts)} parts on board · "
           f"{len(b.cuts)} track cut{'s' if len(b.cuts) != 1 else ''} · {len(b.links)} link{'s' if len(b.links) != 1 else ''} · "
           f"{len(b.pads)} off-board pad{'s' if len(b.pads) != 1 else ''}")
    a(f'<text x="{_f(X0)}" y="50" font-size="11.5" fill="currentColor" opacity="0.72">{_esc(sub)}</text>')

    # ---- board and copper
    a(f'<rect x="{_f(X0 - 6)}" y="{_f(Y0 - 6)}" width="{_f(bw + 12)}" height="{_f(bh + 12)}" rx="7" '
      f'fill="{_BOARD}" stroke="#000" stroke-opacity="0.18" filter="url(#hwShadow)"/>')
    nets_seen = {}
    for strip in b.strips:
        x1 = s.cx(strip["start"]) - P / 2 + 2
        x2 = s.cx(strip["end"]) + P / 2 - 2
        y = s.cy(strip["row"]) - 0.36 * P
        tip = f"<title>{_esc(_net_name(strip['net']))}</title>" if strip["net"] else ""
        a(f'<g><rect x="{_f(x1)}" y="{_f(y)}" width="{_f(x2 - x1)}" height="{_f(0.72 * P)}" rx="2.5" '
          f'fill="{_COPPER}"/><rect x="{_f(x1)}" y="{_f(y)}" width="{_f(x2 - x1)}" height="{_f(0.72 * P)}" '
          f'rx="2.5" fill="url(#hwCu)"/>{tip}</g>')
        if strip["net"]:
            nets_seen[strip["net"]] = 1
    holes = []
    for r in range(H):
        for c in range(W):
            holes.append(f'<circle cx="{_f(s.cx(c))}" cy="{_f(s.cy(r))}" r="{_f(0.12 * P)}"/>')
    a('<g fill="#3a2a1d">' + "".join(holes) + "</g>")
    # cut spots: bare board around the drilled hole
    for r, c in b.cuts:
        a(f'<circle cx="{_f(s.cx(c))}" cy="{_f(s.cy(r))}" r="{_f(0.3 * P)}" fill="{_BOARD}"/>'
          f'<circle cx="{_f(s.cx(c))}" cy="{_f(s.cy(r))}" r="{_f(0.13 * P)}" fill="#3a2a1d"/>')

    # ---- row letters and column numbers
    lab = []
    for r in range(H):
        lab.append(f'<text x="{_f(X0 + bw + 14)}" y="{_f(s.cy(r) + 3.5)}" text-anchor="middle">{_row_name(r)}</text>')
    step = 1 if W <= 40 else 2
    for c in range(0, W, step):
        lab.append(f'<text x="{_f(s.cx(c))}" y="{_f(Y0 - 12)}" text-anchor="middle">{c + 1}</text>')
        lab.append(f'<text x="{_f(s.cx(c))}" y="{_f(Y0 + bh + 20)}" text-anchor="middle">{c + 1}</text>')
    a('<g font-size="9.5" fill="currentColor" opacity="0.62">' + "".join(lab) + "</g>")

    # ---- links (below parts)
    for k in b.links:
        (r1, c1), (r2, c2) = k["a"], k["b"]
        x, y1, y2 = s.cx(c1), s.cy(min(r1, r2)), s.cy(max(r1, r2))
        a(f'<g><title>link {_esc(_net_name(k["net"]))}: {b.hole(*k["a"])} to {b.hole(*k["b"])}</title>'
          f'<line x1="{_f(x)}" y1="{_f(y1)}" x2="{_f(x)}" y2="{_f(y2)}" stroke="#b9bec4" stroke-width="2"/>'
          f'<line x1="{_f(x)}" y1="{_f(y1 + 0.3 * P)}" x2="{_f(x)}" y2="{_f(y2 - 0.3 * P)}" '
          f'stroke="{k["colour"]}" stroke-width="{_f(0.24 * P)}" stroke-linecap="round"/>'
          f'<line x1="{_f(x - 0.04 * P)}" y1="{_f(y1 + 0.32 * P)}" x2="{_f(x - 0.04 * P)}" y2="{_f(y2 - 0.32 * P)}" '
          f'stroke="#fff" stroke-opacity="0.35" stroke-width="1.6" stroke-linecap="round"/>'
          + _joint(x, y1) + _joint(x, y2) + "</g>")
        s.rects.append((x - 0.14 * P, y1 + 0.2 * P, x + 0.14 * P, y2 - 0.2 * P))

    # ---- parts and pads (cuts and lead joints are obstacles for pad tags and labels)
    for r, c in b.cuts:
        s.rects.append((s.cx(c) - 0.24 * P, s.cy(r) - 0.24 * P, s.cx(c) + 0.24 * P, s.cy(r) + 0.24 * P))
    for p in b.parts:
        _draw_part(s, p)
    s.joints = [(s.cx(l.col) - 4, s.cy(l.row) - 4, s.cx(l.col) + 4, s.cy(l.row) + 4)
                for p in b.parts + b.pads for l in p.leads]
    s.joints += [(s.cx(k[e][1]) - 4, s.cy(k[e][0]) - 4, s.cx(k[e][1]) + 4, s.cy(k[e][0]) + 4)
                 for k in b.links for e in ("a", "b")]
    for p in b.pads:
        _draw_pad(s, p)
    # ---- cut marks on top of everything
    part_rects = s.rects[len(b.cuts):]
    for r, c in b.cuts:
        x, y = s.cx(c), s.cy(r)
        under = any(a0 < x < a1 and b0 < y < b1 for (a0, b0, a1, b1) in part_rects)
        d = 0.17 * P if under else 0.22 * P
        path = f"M{_f(x - d)} {_f(y - d)}L{_f(x + d)} {_f(y + d)}M{_f(x + d)} {_f(y - d)}L{_f(x - d)} {_f(y + d)}"
        a(f'<g><title>track cut {b.hole(r, c)}</title><path d="{path}" stroke="#fff" stroke-opacity="0.85" '
          f'stroke-width="5.4" stroke-linecap="round"/><path d="{path}" stroke="var(--hotwire-cut, #e2552b)" '
          f'stroke-width="3" stroke-linecap="round"/></g>')
    # ---- labels
    _draw_labels(s)

    # ---- legend
    _draw_legend(s, entries, leg_y, ncols, col_w, width)
    a("</svg>")
    return "\n".join(s.out)


def _joint(x: float, y: float) -> str:
    return (f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(0.13 * P)}" fill="#cfd3d7" '
            f'stroke="#80878e" stroke-width="0.8"/>')


def _label_text(p: Placed) -> str:
    if p.kind == "LED":
        return f"{p.ref} {p.style.get('colour', 'red')}"
    if p.kind in ("D", "Q", "J", "DIP"):
        return f"{p.ref} {p.part}"
    return f"{p.ref} {p.value}"


def _draw_part(s: _Svg, p: Placed) -> None:
    if p.kind in ("R", "C", "L", "D", "LED"):
        _draw_two(s, p)
    elif p.kind in ("Q", "J"):
        _draw_to92(s, p)
    elif p.kind == "DIP":
        _draw_dip(s, p)


def _draw_two(s: _Svg, p: Placed) -> None:
    """Two-lead part drawn in a vertical local frame, rotated when it lies along a strip."""
    la, lb = p.leads
    horiz = la.row == lb.row
    if horiz:
        ta, tb = la.col, lb.col
        xm, ym = (s.cx(la.col) + s.cx(lb.col)) / 2, s.cy(la.row)
    else:
        ta, tb = la.row, lb.row
        xm, ym = s.cx(la.col), (s.cy(la.row) + s.cy(lb.row)) / 2
    tm = (ta + tb) / 2
    t_top = min(ta, tb)

    def Y(t):
        return ym + (t - tm) * P

    def pos(lead):
        return lead.col if horiz else lead.row

    def glyph(text, gx, gy, size, colour, weight="700"):
        rot = f' transform="rotate(90 {_f(gx)} {_f(gy - size * 0.35)})"' if horiz else ""
        return (f'<text x="{_f(gx)}" y="{_f(gy)}" text-anchor="middle" font-size="{size}" '
                f'font-weight="{weight}" fill="{colour}"{rot}>{text}</text>')

    y1, y2 = Y(t_top), Y(max(ta, tb))
    x = xm
    avail = (y2 - y1) - 0.5 * P
    g = [f'<line x1="{_f(x)}" y1="{_f(y1)}" x2="{_f(x)}" y2="{_f(y2)}" stroke="#a9afb5" stroke-width="1.9"/>',
         _joint(x, y1), _joint(x, y2)]
    hw, hl = 0.2 * P, 0.5 * P
    if p.kind in ("R", "L"):
        L = min(2.3 * P, avail)
        wdt = 0.42 * P
        bands = p.style.get("bands", [])
        fill = "#dcc59b" if len(bands) != 5 else "#86afd6"
        if p.kind == "L":
            fill = "#3f7a54"
        top = ym - L / 2
        g.append(f'<rect x="{_f(x - wdt / 2)}" y="{_f(top)}" width="{_f(wdt)}" height="{_f(L)}" '
                 f'rx="{_f(wdt / 2)}" fill="{fill}" stroke="#000" stroke-opacity="0.25" stroke-width="0.7"/>')
        if p.kind == "R" and bands:
            fr = [0.2, 0.32, 0.44, 0.82] if len(bands) == 4 else [0.17, 0.28, 0.39, 0.5, 0.84]
            for col, f in zip(bands, fr):
                g.append(f'<rect x="{_f(x - wdt / 2 + 0.6)}" y="{_f(top + f * L - 0.04 * L)}" '
                         f'width="{_f(wdt - 1.2)}" height="{_f(0.08 * L)}" fill="{_HEX[col]}"/>')
        if p.kind == "L":
            for f in (0.3, 0.5, 0.7):
                g.append(f'<rect x="{_f(x - wdt / 2 + 0.6)}" y="{_f(top + f * L - 0.03 * L)}" '
                         f'width="{_f(wdt - 1.2)}" height="{_f(0.06 * L)}" fill="#d9b26f"/>')
        g.append(f'<rect x="{_f(x - wdt / 2)}" y="{_f(top)}" width="{_f(wdt)}" height="{_f(L)}" '
                 f'rx="{_f(wdt / 2)}" fill="url(#hwBodyV)"/>')
        hw, hl = wdt / 2, L / 2
    elif p.kind == "C":
        cap = p.style.get("cap")
        if cap == "ceramic":
            ry = min(0.42 * P, avail / 2 + 4)
            g.append(f'<ellipse cx="{_f(x)}" cy="{_f(ym)}" rx="{_f(0.3 * P)}" ry="{_f(ry)}" '
                     f'fill="#e2a23a" stroke="#9a6514" stroke-width="0.8"/>'
                     f'<ellipse cx="{_f(x - 2)}" cy="{_f(ym - 3)}" rx="{_f(0.12 * P)}" ry="{_f(0.18 * P)}" '
                     f'fill="#fff" opacity="0.28"/>')
            hw, hl = 0.3 * P, ry
        elif cap == "film":
            L = min(1.5 * P, avail + 6)
            wdt = 0.52 * P
            g.append(f'<rect x="{_f(x - wdt / 2)}" y="{_f(ym - L / 2)}" width="{_f(wdt)}" height="{_f(L)}" '
                     f'rx="2.5" fill="#b8443a" stroke="#6e2019" stroke-width="0.7"/>'
                     f'<rect x="{_f(x - wdt / 2)}" y="{_f(ym - L / 2)}" width="{_f(wdt)}" height="{_f(L)}" '
                     f'rx="2.5" fill="url(#hwBodyV)"/>')
            hw, hl = wdt / 2, L / 2
        else:
            L = min(1.8 * P, avail + 8)
            wdt = 0.58 * P
            top = ym - L / 2
            plus = next(l for l in p.leads if l.name == "+")
            neg_top = pos(plus) != t_top
            sy = top if neg_top else ym + L / 2 - 0.3 * L
            g.append(f'<rect x="{_f(x - wdt / 2)}" y="{_f(top)}" width="{_f(wdt)}" height="{_f(L)}" '
                     f'rx="{_f(0.2 * P)}" fill="#24446d" stroke="#0f2340" stroke-width="0.8"/>'
                     f'<rect x="{_f(x - wdt / 2 + 0.8)}" y="{_f(sy + (0.8 if neg_top else 0))}" '
                     f'width="{_f(wdt - 1.6)}" height="{_f(0.3 * L - 0.8)}" fill="#c7d3e6"/>'
                     f'<rect x="{_f(x - wdt / 2)}" y="{_f(top)}" width="{_f(wdt)}" height="{_f(L)}" '
                     f'rx="{_f(0.2 * P)}" fill="url(#hwBodyV)"/>')
            g.append(glyph("−", x, sy + 0.15 * L + 3.8, 11, "#24446d"))
            py = Y(pos(plus)) + (0.4 * P if pos(plus) == t_top else -0.2 * P)
            g.append(glyph("+", x + 0.32 * P, py, 10, "#b3261e"))
            hw, hl = wdt / 2, L / 2
    elif p.kind == "D":
        pkg = p.package.upper()
        k_top = pos(next(l for l in p.leads if l.name == "K")) == t_top
        if "DO-41" in pkg:
            L, wdt, fill, band = min(1.45 * P, avail + 4), 0.42 * P, "#1f1f1f", "#c9c9c9"
        elif "DO-7" in pkg:
            L, wdt, fill, band = min(1.35 * P, avail + 4), 0.34 * P, "#c7d6a0", "#1b1b1b"
        else:
            L, wdt, fill, band = min(1.25 * P, avail + 4), 0.32 * P, "#e07a3e", "#1b1b1b"
        top = ym - L / 2
        by = top + 0.08 * L if k_top else top + 0.74 * L
        g.append(f'<rect x="{_f(x - wdt / 2)}" y="{_f(top)}" width="{_f(wdt)}" height="{_f(L)}" '
                 f'rx="{_f(wdt / 2)}" fill="{fill}" stroke="#000" stroke-opacity="0.3" stroke-width="0.6"/>'
                 f'<rect x="{_f(x - wdt / 2 + 0.4)}" y="{_f(by)}" width="{_f(wdt - 0.8)}" height="{_f(0.18 * L)}" '
                 f'fill="{band}"/>'
                 f'<rect x="{_f(x - wdt / 2)}" y="{_f(top)}" width="{_f(wdt)}" height="{_f(L)}" '
                 f'rx="{_f(wdt / 2)}" fill="url(#hwBodyV)"/>')
        hw, hl = wdt / 2, L / 2
    elif p.kind == "LED":
        colour = {"red": "#e0443a", "green": "#3cb45a", "yellow": "#f1c232", "blue": "#3d7fe0",
                  "white": "#f2f2ee", "orange": "#f08a2a"}.get(p.style.get("colour"), "#e0443a")
        k_top = pos(next(l for l in p.leads if l.name == "K")) == t_top
        d = -1 if k_top else 1
        rr = 0.46 * P
        off = 0.36 * P
        hc = math.sqrt(rr * rr - off * off)
        yc = ym + d * off
        sweep = 1 if d == 1 else 0
        path = f"M{_f(x - hc)} {_f(yc)} A{_f(rr)} {_f(rr)} 0 1 {sweep} {_f(x + hc)} {_f(yc)} Z"
        g.append(f'<path d="{path}" fill="{colour}" stroke="#000" stroke-opacity="0.35" stroke-width="0.8"/>'
                 f'<path d="{path}" fill="url(#hwLed)"/>')
        hw, hl = rr, rr
    tip = f"<title>{_esc(p.ref)} {_esc(p.value)} ({_esc(p.package)})</title>"
    wrap = f'<g transform="rotate(-90 {_f(xm)} {_f(ym)})">' if horiz else "<g>"
    s.add(wrap + tip + "".join(g) + "</g>")
    if horiz:
        s.rects.append((xm - hl, ym - hw, xm + hl, ym + hw))
    else:
        s.rects.append((xm - hw, ym - hl, xm + hw, ym + hl))
    s.labels.append((p, horiz, xm, ym, abs(tb - ta)))


def _draw_to92(s: _Svg, p: Placed) -> None:
    rows = [l.row for l in p.leads]
    c = p.leads[0].col
    side = p.style.get("side", 1)          # +1: round side to the right, flat face left
    r0 = min(rows)
    x = s.cx(c)
    ytop, ybot = s.cy(r0) - 0.46 * P, s.cy(r0 + 2) + 0.46 * P
    xf = x - side * 0.26 * P
    depth = 1.12 * P
    metal = "TO-1" in p.package or "TO-18" in p.package
    to220 = "TO-220" in p.package
    g = [f"<g><title>{_esc(p.ref)} {_esc(p.part)} ({_esc(p.package)}, {_esc(p.style.get('pins', ''))}, "
         f"{'printed' if to220 else 'flat'} face {_flat_dir(p)})</title>"]
    for l in p.leads:
        g.append(_joint(s.cx(l.col), s.cy(l.row)))
    if to220:
        # standing TO-220: plastic body on the printed side, metal tab behind it
        body = 0.55 * P
        tab = 0.22 * P
        xb = min(xf, xf + side * body)
        xt = xf + body if side == 1 else xf - body - tab
        g.append(f'<rect x="{_f(xb)}" y="{_f(ytop - 0.1 * P)}" width="{_f(body)}" '
                 f'height="{_f(ybot - ytop + 0.2 * P)}" fill="#262626" stroke="#000" stroke-width="0.8" '
                 f'filter="url(#hwShadow)"/>'
                 f'<rect x="{_f(xt)}" y="{_f(ytop - 0.25 * P)}" width="{_f(tab)}" '
                 f'height="{_f(ybot - ytop + 0.5 * P)}" fill="#c3c8cd" stroke="#6d747b" stroke-width="0.8"/>')
        depth = body + tab
        txt = "#2a2f35"     # pin letters land on the tab
    elif metal:
        g.append(f'<rect x="{_f(min(xf, xf + side * depth))}" y="{_f(ytop + 2)}" width="{_f(depth)}" '
                 f'height="{_f(ybot - ytop - 4)}" rx="{_f(0.5 * P)}" fill="#c3c8cd" stroke="#6d747b" '
                 f'stroke-width="1" filter="url(#hwShadow)"/>')
        txt = "#2a2f35"
    else:
        sweep = 1 if side == 1 else 0
        path = f"M{_f(xf)} {_f(ytop)} A{_f(depth)} {_f((ybot - ytop) / 2)} 0 0 {sweep} {_f(xf)} {_f(ybot)} Z"
        g.append(f'<path d="{path}" fill="#262626" stroke="#000" stroke-width="0.8" filter="url(#hwShadow)"/>'
                 f'<path d="{path}" fill="url(#hwBodyV)" opacity="0.5"/>'
                 f'<line x1="{_f(xf)}" y1="{_f(ytop + 1)}" x2="{_f(xf)}" y2="{_f(ybot - 1)}" stroke="#777" '
                 f'stroke-width="1.6"/>')
        txt = "#f2f2f2"
    for l in p.leads:
        g.append(f'<circle cx="{_f(x)}" cy="{_f(s.cy(l.row))}" r="{_f(0.11 * P)}" fill="#c9cdd1"/>')
        g.append(f'<text x="{_f(x + side * 0.4 * P)}" y="{_f(s.cy(l.row) + 3.4)}" text-anchor="middle" '
                 f'font-size="9" font-weight="700" fill="{txt}">{_esc(l.name)}</text>')
    g.append("</g>")
    s.add("".join(g))
    xa, xb = sorted((xf, xf + side * depth))
    s.rects.append((xa, ytop, xb, ybot))
    s.labels.append((p, False, x, (ytop + ybot) / 2, 3))


def _draw_dip(s: _Svg, p: Placed) -> None:
    rows = [l.row for l in p.leads]
    cols = [l.col for l in p.leads]
    c0, c3 = min(cols), max(cols)
    r0, r3 = min(rows), max(rows)
    x1, x2 = s.cx(c0) + 0.24 * P, s.cx(c3) - 0.24 * P
    y1, y2 = s.cy(r0) - 0.46 * P, s.cy(r3) + 0.46 * P
    rot = p.style.get("rot", 0)
    g = [f"<g><title>{_esc(p.ref)} {_esc(p.part)} ({_esc(p.style.get('what', ''))})</title>"]
    for l in p.leads:
        xl = s.cx(l.col)
        xe = x1 if l.col == c0 else x2
        g.append(f'<rect x="{_f(min(xl, xe) - (0.12 * P if l.col == c0 else 0))}" y="{_f(s.cy(l.row) - 0.13 * P)}" '
                 f'width="{_f(abs(xe - xl) + 0.12 * P)}" height="{_f(0.26 * P)}" fill="#c3c8cd" stroke="#7d848b" '
                 f'stroke-width="0.6"/>')
    g.append(f'<rect x="{_f(x1)}" y="{_f(y1)}" width="{_f(x2 - x1)}" height="{_f(y2 - y1)}" rx="3" '
             f'fill="#25282c" stroke="#0d0e10" stroke-width="0.8" filter="url(#hwShadow)"/>')
    nx = (x1 + x2) / 2
    ny = y1 if rot == 0 else y2
    g.append(f'<path d="M{_f(nx - 0.24 * P)} {_f(ny)} A{_f(0.24 * P)} {_f(0.24 * P)} 0 0 {0 if rot == 0 else 1} '
             f'{_f(nx + 0.24 * P)} {_f(ny)} Z" fill="#555b62"/>')
    pin1 = next(l for l in p.leads if l.name == "1")
    dx = s.cx(pin1.col) + (0.5 * P if pin1.col == c0 else -0.5 * P)
    dy = s.cy(pin1.row) + (-0.26 * P if rot == 0 else 0.26 * P)
    g.append(f'<circle cx="{_f(dx)}" cy="{_f(dy)}" r="{_f(0.09 * P)}" fill="#9aa1a8"/>')
    for l in p.leads:
        px = s.cx(l.col) + (0.5 * P if l.col == c0 else -0.5 * P)
        g.append(f'<text x="{_f(px)}" y="{_f(s.cy(l.row) + (6.5 if rot == 0 else 1.0))}" text-anchor="middle" '
                 f'font-size="7" fill="#aab1b8">{_esc(l.name)}</text>')
    tx = s.cx(c0 + 2)
    g.append(f'<text transform="translate({_f(tx)} {_f((y1 + y2) / 2)}) rotate(-90)" text-anchor="middle" '
             f'dy="4" font-size="11" fill="#f4f4f4"><tspan font-weight="700">{_esc(p.ref)}</tspan> '
             f'<tspan fill="#c9ced4" font-size="9.5">{_esc(p.part)}</tspan></text>')
    g.append("</g>")
    s.add("".join(g))
    s.rects.append((s.cx(c0) - 0.25 * P, y1, s.cx(c3) + 0.25 * P, y2))


def _overlaps(r, rects, pad=1.0) -> bool:
    x0, y0, x1, y1 = r
    for (a0, b0, a1, b1) in rects:
        if x0 < a1 + pad and x1 > a0 - pad and y0 < b1 + pad and y1 > b0 - pad:
            return True
    return False


def _draw_labels(s: _Svg) -> None:
    size = 8.6
    half = 4.6
    out = []
    bx0, by0 = s.X0 + 1, s.Y0 + 1
    bx1, by1 = s.X0 + s.b.width * P - 1, s.Y0 + s.b.height * P - 1
    placed_rects: list = []
    shifts = (0, -0.35, 0.35, -0.7, 0.7, -1.05, 1.05, -1.4, 1.4, -1.8, 1.8)
    for p, horiz, xm, ym, span in sorted(s.labels, key=lambda t: -t[4]):
        text = _label_text(p)
        tl = _text_w(text, size) + 4
        choice = None
        c = p.leads[0].col
        sides = (s.X0 + (c + 1) * P, s.X0 + c * P)
        if p.kind in ("Q", "J") and p.style.get("side", 1) == 1:
            sides = (s.X0 + c * P, s.X0 + (c + 1) * P)
        if horiz:
            r = p.leads[0].row
            for yl in (s.Y0 + r * P, s.Y0 + (r + 1) * P):
                for sh in shifts:
                    x0 = min(max(xm - tl / 2 + sh * P, bx0), bx1 - tl)
                    rect = (x0, yl - half, x0 + tl, yl + half)
                    if not _overlaps(rect, s.rects) and not _overlaps(rect, placed_rects):
                        choice = (False, rect)
                        break
                if choice:
                    break
        else:
            for xl in sides:
                for sh in shifts:
                    y0 = min(max(ym - tl / 2 + sh * P, by0), by1 - tl)
                    rect = (xl - half, y0, xl + half, y0 + tl)
                    if not _overlaps(rect, s.rects) and not _overlaps(rect, placed_rects):
                        choice = (True, rect)
                        break
                if choice:
                    break
        if choice is None:
            if horiz:
                yl = s.Y0 + (p.leads[0].row + 1) * P
                choice = (False, (xm - tl / 2, yl - half, xm + tl / 2, yl + half))
            else:
                xl = s.X0 + (p.leads[0].col + 1) * P
                choice = (True, (xl - half, ym - tl / 2, xl + half, ym + tl / 2))
        vertical, rect = choice
        placed_rects.append(rect)
        cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        rot = " rotate(-90)" if vertical else ""
        out.append(f'<text transform="translate({_f(cx)} {_f(cy)}){rot}" text-anchor="middle" dy="3.1">'
                   f'<tspan font-weight="700">{_esc(p.ref)}</tspan> {_esc(text[len(p.ref) + 1:])}</text>')
    s.add(f'<g font-size="{size}" fill="{_INK}" stroke="#efe5cc" stroke-width="3.2" paint-order="stroke" '
          f'stroke-linejoin="round">' + "".join(out) + "</g>")


def _draw_pad(s: _Svg, p: Placed) -> None:
    l = p.leads[0]
    x, y = s.cx(l.col), s.cy(l.row)
    colour = _PAD_COLOURS.get(p.group, "#555")
    tw = _text_w(p.label, 9) + 10
    g = [f'<g><title>{_esc(p.label)} pad ({_esc(_net_name(l.net))}): {s.b.hole(l.row, l.col)}</title>']
    if l.col == 0:
        tx2 = s.X0 - 14
        tx1 = tx2 - tw
        g.append(f'<path d="M{_f(x)} {_f(y)} C{_f(x - 0.6 * P)} {_f(y)} {_f(tx2 + 10)} {_f(y)} {_f(tx2)} {_f(y)}" '
                 f'fill="none" stroke="{colour}" stroke-width="2.6" stroke-linecap="round"/>')
    else:
        options = [(x + 0.36 * P, x + 0.36 * P + tw), (x - 0.36 * P - tw, x - 0.36 * P)]
        tx1, tx2 = options[0]
        for o in options:
            rect = (o[0], y - 7.5, o[1], y + 7.5)
            if o[0] > s.X0 and not _overlaps(rect, s.rects, 0.5) and not _overlaps(rect, s.joints, 0.5):
                tx1, tx2 = o
                break
        s.rects.append((min(tx1, x - 0.3 * P), y - 8, max(tx2, x + 0.3 * P), y + 8))
    g.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(0.27 * P)}" fill="#dfe3e6" stroke="#80878e" stroke-width="1"/>'
             f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(0.11 * P)}" fill="{colour}"/>')
    g.append(f'<rect x="{_f(tx1)}" y="{_f(y - 7.5)}" width="{_f(tw)}" height="15" rx="4" fill="{colour}" '
             f'stroke="#fff" stroke-opacity="0.7" stroke-width="0.8"/>'
             f'<text x="{_f((tx1 + tx2) / 2)}" y="{_f(y + 3.3)}" text-anchor="middle" font-size="9" '
             f'font-weight="700" fill="#fff">{_esc(p.label)}</text>')
    g.append("</g>")
    s.add("".join(g))


def _legend_entries(b: Board) -> list:
    out = []
    for p in sorted(b.parts, key=lambda p: (p.kind, _natural(p.ref))):
        if p.kind == "R":
            out.append((p.ref, f"{p.value}Ω", " ".join(p.style.get("bands", [])), p.style.get("bands", [])))
        elif p.kind == "C":
            out.append((p.ref, f"{p.value}F", p.style.get("cap", ""), None))
        elif p.kind == "L":
            out.append((p.ref, p.value, "inductor", None))
        elif p.kind == "D":
            out.append((p.ref, p.part, f"diode, {p.package}", None))
        elif p.kind == "LED":
            out.append((p.ref, p.value, "flat side = cathode", None))
        elif p.kind in ("Q", "J"):
            out.append((p.ref, p.part, f"{p.package} {p.style.get('pins')}, flat {_flat_dir(p)}", None))
        elif p.kind == "DIP":
            out.append((p.ref, p.part, "DIP-8 in a socket", None))
    for item in b.offboard:
        if item["kind"] == "pot":
            out.append((item["ref"], item["value"], f"{item['label']} pot, off-board", None))
    return out


def _natural(ref: str) -> tuple:
    m = re.match(r"([A-Za-z_]+)(\d*)", ref)
    return (m.group(1), int(m.group(2) or 0)) if m else (ref, 0)


def _draw_legend(s: _Svg, entries: list, y: float, ncols: int, col_w: float, width: float) -> None:
    a = s.add
    x = s.X0
    g = ['<g font-size="10.5" fill="currentColor">']
    # symbol key
    kx = x
    g.append(f'<rect x="{_f(kx)}" y="{_f(y - 6)}" width="30" height="12" rx="2" fill="{_COPPER}"/>'
             f'<text x="{_f(kx + 38)}" y="{_f(y + 4)}">copper strip</text>')
    kx += 128
    d = 5
    g.append(f'<path d="M{_f(kx + 6 - d)} {_f(y - d)}L{_f(kx + 6 + d)} {_f(y + d)}M{_f(kx + 6 + d)} {_f(y - d)}'
             f'L{_f(kx + 6 - d)} {_f(y + d)}" stroke="var(--hotwire-cut, #e2552b)" stroke-width="3" '
             f'stroke-linecap="round"/><text x="{_f(kx + 20)}" y="{_f(y + 4)}">track cut (drill the hole)</text>')
    kx += 182
    g.append(f'<line x1="{_f(kx)}" y1="{_f(y)}" x2="{_f(kx + 30)}" y2="{_f(y)}" stroke="#2f6db3" stroke-width="7" '
             f'stroke-linecap="round"/><text x="{_f(kx + 40)}" y="{_f(y + 4)}">wire link</text>')
    kx += 112
    g.append(f'<rect x="{_f(kx)}" y="{_f(y - 7)}" width="30" height="14" rx="4" fill="#2e8b57"/>'
             f'<text x="{_f(kx + 38)}" y="{_f(y + 4)}">off-board wire pad</text>')
    y0 = y + 30
    g.append(f'<text x="{_f(x)}" y="{_f(y0 - 4)}" font-weight="650" font-size="11">Parts</text>'
             f'<text x="{_f(width - 12)}" y="{_f(y0 - 4)}" text-anchor="end" opacity="0.6">'
             f'component side, 0.1 in pitch</text>')
    per = math.ceil(len(entries) / ncols) if entries else 0
    for i, (ref, val, desc, bands) in enumerate(entries):
        cx = x + (i // per) * col_w if per else x
        cy = y0 + 14 + (i % per) * 17 if per else y0
        bx = cx
        if bands:
            for j, col in enumerate(bands):
                g.append(f'<rect x="{_f(bx + j * 5)}" y="{_f(cy - 8)}" width="4" height="10" rx="0.8" '
                         f'fill="{_HEX[col]}" stroke="#000" stroke-opacity="0.25" stroke-width="0.5"/>')
        g.append(f'<text x="{_f(cx + 28)}" y="{_f(cy)}"><tspan font-weight="700">{_esc(ref)}</tspan> '
                 f'{_esc(val)}<tspan opacity="0.62"> {_esc(desc)}</tspan></text>')
    g.append("</g>")
    a("".join(g))
