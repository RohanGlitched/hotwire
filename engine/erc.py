"""Electrical rule check: the mistakes that make a simulator (or a real board) fail.

Runs on the netlist alone, before any simulation, and reports problems the way an
experienced builder would point them out.
"""
from __future__ import annotations

from collections import defaultdict

from .netlist import Circuit

__all__ = ["check"]


def _issue(level: str, code: str, message: str, parts=None, nodes=None) -> dict:
    return {"level": level, "code": code, "message": message,
            "parts": list(parts or []), "nodes": list(nodes or [])}


class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def _dc_pairs(e):
    """Pairs of nodes an element connects at DC (for 'is there a DC path')."""
    n = e.nodes
    k = e.kind
    if k in ("R", "L", "V", "D", "I"):
        return [(n[0], n[1])] if k != "I" else []
    if k == "S":
        return [(n[0], n[1])] if e.params.get("on") else []
    if k == "Q":   # c b e all share DC paths through the junctions
        return [(n[0], n[1]), (n[1], n[2])]
    if k == "J":   # channel d-s conducts; gate only through a junction (treat as path)
        return [(n[0], n[2]), (n[1], n[2])]
    if k == "M":   # MOSFET: channel d-s and the bulk junctions conduct; the gate is insulated
        return [(n[0], n[2]), (n[3], n[2])]
    if k == "X":   # op-amp: output driven from the supplies; inputs are high impedance
        return [(n[4], n[2]), (n[4], n[3]), (n[2], n[3])]
    if k in ("E",):
        return [(n[0], n[1])]
    return []      # C, G: no DC path


def check(c: Circuit) -> list:
    issues: list = []
    pins = defaultdict(list)        # node -> [(element, pin index)]
    for e in c.elements:
        for i, n in enumerate(e.nodes):
            if e.kind == "M" and i == 3 and n == e.nodes[2]:
                continue  # bulk tied to source is one lead, not two
            pins[n].append((e, i))

    # 1. shorted / degenerate two-terminal parts
    for e in c.elements:
        if e.kind == "R" and e.value is not None and e.value < 0.5 and not e.pot:
            if e.value >= 1e-4:
                msg = f"{e.name} is {e.value * 1e3:g} mΩ. In SPICE 'M' means milli: write 1meg for 1 MΩ"
            else:
                msg = f"{e.name} is {e.value:g} Ω, far below any real resistor; check the value"
            issues.append(_issue("warning", "tiny_resistor", msg, [e.name], e.nodes))
        if e.pot and e.nodes[0] == e.nodes[1]:
            continue  # the unused half of a pot wired as a variable resistor
        if e.kind in ("R", "C", "L", "V", "I", "D", "S") and e.nodes[0] == e.nodes[1]:
            lvl = "error" if e.kind == "V" else "warning"
            issues.append(_issue(lvl, "shorted", f"{e.name} has both ends on node '{e.nodes[0]}', so it does nothing"
                                 + (" and shorts the source" if e.kind == "V" else ""), [e.name], [e.nodes[0]]))
        if e.kind == "X":
            if e.nodes[2] == e.nodes[3]:
                issues.append(_issue("error", "opamp_supply", f"{e.name}'s V+ and V− pins are the same node, so it has no supply",
                                     [e.name], [e.nodes[2]]))
            if e.nodes[0] == e.nodes[1]:
                issues.append(_issue("warning", "opamp_inputs", f"{e.name}'s + and − inputs are tied together",
                                     [e.name], [e.nodes[0]]))

    # 2. floating nodes: only one pin connects to them
    for node, lst in pins.items():
        if node == "0":
            continue
        if len(lst) == 1:
            e, _ = lst[0]
            if c.output == node:
                continue  # an output node left open is normal (it goes to a jack)
            issues.append(_issue("warning", "dangling", f"node '{node}' only connects to {e.name}; that pin goes nowhere",
                                 [e.name], [node]))

    # 3. DC path to ground (otherwise the operating point is undefined)
    dsu = _DSU()
    dsu.find("0")
    for e in c.elements:
        for a, b in _dc_pairs(e):
            dsu.union(a, b)
    ground = dsu.find("0")
    for node in c.nodes():
        if dsu.find(node) != ground:
            parts = sorted({e.name for e, _ in pins[node]})
            op_in = [e.name for e, i in pins[node] if e.kind == "X" and i in (0, 1)]
            gates = [e.name for e, i in pins[node] if e.kind == "M" and i == 1]
            if gates:
                msg = (f"node '{node}' has no DC path to ground: the gate has no DC path, because "
                       f"{', '.join(gates)}'s gate is insulated and nothing else sets its voltage. Add a gate "
                       "resistor (e.g. 1M to ground, or a divider from the supply)")
            elif op_in:
                msg = (f"node '{node}' has no DC path to ground: {', '.join(op_in)}'s input would float. "
                       "Add a bias resistor (e.g. 1M to the bias rail or ground)")
            else:
                msg = (f"node '{node}' has no DC path to ground (it's isolated by capacitors). "
                       "Add a resistor to ground or to a bias point")
            issues.append(_issue("error", "no_dc_path", msg, parts, [node]))

    # 4. loops of voltage sources / inductors (ideal loop = infinite current)
    vl = _DSU()
    for e in c.elements:
        if e.kind in ("V", "L") or (e.kind == "S" and e.params.get("on")):
            if not vl.union(e.nodes[0], e.nodes[1]):
                kinds = "voltage sources and inductors" if e.kind != "S" else "a closed switch and sources"
                issues.append(_issue("error", "source_loop",
                                     f"{e.name} closes a loop of {kinds} with nothing to limit the current",
                                     [e.name], e.nodes))

    # 5. annotation sanity
    if c.input:
        src = c.element(c.input)
        dc_nodes = {e.nodes[0] for e in c.elements if e.kind == "V" and e.name != c.input and e.nodes[1] == "0"}
        if "0" not in src.nodes and src.nodes[1] not in dc_nodes:
            issues.append(_issue("warning", "input_ref", f"input source {c.input} isn't referenced to ground", [c.input]))
    if c.output and c.input:
        src = c.element(c.input)
        if c.output in src.nodes:
            issues.append(_issue("warning", "out_is_in", "the output node is the input source's own node"))
    for s in c.supplies:
        e = c.element(s)
        if e.kind == "V" and not e.value:
            issues.append(_issue("warning", "supply_zero", f"supply {s} is 0 V", [s]))
    for e in c.elements:
        if e.kind == "V":
            peak = abs(e.value or 0.0) + (abs(e.params["sin"][1]) if "sin" in e.params else 0.0)
            if peak > 50:
                issues.append(_issue("warning", "high_voltage",
                                     f"{e.name} reaches {peak:g} V. Hotwire is for low-voltage circuits; anything above "
                                     "about 50 V (and anything mains-connected) needs a qualified designer and a certified supply",
                                     [e.name]))

    # 6. electrolytic polarity hint is a stress check (needs the operating point)
    return issues


def check_op(c: Circuit, op) -> list:
    """Rules that need the DC operating point."""
    issues = []
    if not getattr(op, "converged", True) or getattr(op, "method", "") == "failed":
        issues.append(_issue("error", "no_dc_solution",
                             "the DC operating point didn't converge, so no measurement can be trusted. "
                             "Check model parameters, a missing bias path, or a circuit that can't settle"))
    v = op.v
    for e in c.elements:
        if e.kind == "X":
            vp = 0.0 if e.nodes[2] == "0" else float(v.get(e.nodes[2], 0.0))
            vn = 0.0 if e.nodes[3] == "0" else float(v.get(e.nodes[3], 0.0))
            if vp < vn - 0.1:
                issues.append(_issue("error", "opamp_supply_reversed",
                                     f"{e.name}'s V+ pin ({e.nodes[2]}, {vp:.2f} V) is below its V− pin ({e.nodes[3]}, {vn:.2f} V): "
                                     "the supply pins are swapped (pin order is in+ in- V+ V- out)", [e.name]))
            elif vp - vn < 0.5:
                issues.append(_issue("error", "opamp_unpowered",
                                     f"{e.name} has {vp - vn:.2f} V across its supply pins, so it can't work", [e.name]))
    return issues
