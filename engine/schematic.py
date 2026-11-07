"""Netlist -> SVG schematic with an automatic signal-flow layout.

    from engine.schematic import svg, layout
    text = svg(circuit)          # themeable SVG string (light and dark)
    lay = layout(circuit)        # placed parts, label boxes and wires (used by the tests)
    lay.overlaps()               # [] when no symbol, label or wire collides

Layout approach
---------------
1. Model. A ``*@ pot`` becomes one three-pin part. Supply sources become rail
   flags (``+9V``), the ``*@ in`` source becomes an IN jack and every ground
   pin gets its own local ground symbol. Bias nodes such as a 4.5 V divider
   that feed several parts are drawn once as a small divider and referenced
   by ``VREF`` flags elsewhere, the way pedal schematics do it.
2. Spine. The signal path from the input to the ``*@ out`` node that passes
   through the most active stages (and the fewest passive parts) is laid out
   left to right. Transistors enter at the base and step the spine to the
   collector or emitter, op-amps enter at the non-feedback input, and a
   volume pot hangs down with its wiper carrying the signal on.
3. Hangers. Parts and short chains of parts that run from a spine node to
   ground or to a rail hang vertically below or above that node, each in its
   own column; emitter legs and op-amp gain legs hang from the device pin.
4. The rest (feedback loops, bridges, side branches) is placed by a
   free-space search close to the nets it joins, trying each label position
   in turn so nothing overlaps.
5. A grid maze router (A* with bend, crossing and hugging costs) draws each
   net with orthogonal wires that never pass through symbols, labels or
   other nets; wires of different nets only cross at right angles, without a
   dot. Junction dots mark every point where three wires meet.
6. A few strategies are tried and the one with the fewest overlaps,
   crossings and bends wins; a plain ladder layout is the last resort.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from xml.sax.saxutils import escape

import numpy as np

from .netlist import Circuit, format_value

__all__ = ["svg", "layout", "Layout"]

G = 10                      # routing grid (px)
FS = 11                     # label font size
CW = FS * 0.6               # monospace advance
LH = 13                     # label line height
MONO = "ui-monospace, 'JetBrains Mono', monospace"
SANS = "ui-sans-serif, system-ui, sans-serif"

# orientation matrices (a, b, c, d): x' = a*x + b*y, y' = c*x + d*y (screen y points down)
ID = (1, 0, 0, 1)
R90 = (0, -1, 1, 0)          # left pin -> top
R180 = (-1, 0, 0, -1)
R270 = (0, 1, -1, 0)         # left pin -> bottom
FY = (1, 0, 0, -1)           # mirror top/bottom
FX = (-1, 0, 0, 1)           # mirror left/right
T1 = (0, 1, 1, 0)
T2 = (0, -1, -1, 0)
ORIENTS = (ID, R90, R180, R270, FY, FX, T1, T2)

_P2 = {"p1": (-30, 0, -1, 0), "p2": (30, 0, 1, 0)}
# kind -> (pins {name: (x, y, dx, dy)}, core box without leads)
_DEFS = {
    "R": (_P2, (-18, -6, 18, 6)),
    "C": (_P2, (-5, -10, 5, 10)),
    "CP": (_P2, (-14, -11, 9, 10)),
    "L": (_P2, (-20, -6, 20, 1)),
    "D": (_P2, (-7, -8, 7, 8)),
    "SCH": (_P2, (-7, -8, 10, 8)),
    "LED": (_P2, (-7, -18, 13, 8)),
    "S": (_P2, (-17, -13, 17, 3)),
    "V": (_P2, (-13, -13, 13, 13)),
    "I": (_P2, (-13, -13, 13, 13)),
    "Q": ({"b": (-30, 0, -1, 0), "c": (10, -30, 0, -1), "e": (10, 30, 0, 1)}, (-15, -17, 19, 17)),
    "J": ({"g": (-30, 0, -1, 0), "d": (10, -30, 0, -1), "s": (10, 30, 0, 1)}, (-13, -16, 19, 16)),
    "X": ({"inA": (-60, -20, -1, 0), "inB": (-60, 20, -1, 0), "out": (60, 0, 1, 0),
           "vcc": (0, -40, 0, -1), "vee": (0, 40, 0, 1)}, (-40, -40, 40, 40)),
    "POT": ({"a": (-30, 0, -1, 0), "b": (30, 0, 1, 0), "w": (0, -30, 0, -1)}, (-18, -16, 18, 6)),
    "E": ({"p": (0, -30, 0, -1), "n": (0, 30, 0, 1), "cp": (-30, -10, -1, 0), "cn": (-30, 10, -1, 0)},
          (-20, -14, 13, 14)),
    "IN": ({"p": (10, 0, 1, 0)}, (-24, -7, 4, 7)),
    "OUT": ({"p": (-10, 0, -1, 0)}, (-4, -7, 31, 7)),
    "NET": ({"p": (-10, 0, -1, 0)}, (-4, -7, 31, 7)),
}
_DEFS["G"] = _DEFS["E"]
_TWO = ("R", "C", "CP", "L", "D", "SCH", "LED", "S", "V", "I")
_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))     # right, down, left, up


def _dir_index(dx, dy):
    return _DIRS.index((int(dx), int(dy)))


def _xf(m, x, y):
    return (m[0] * x + m[1] * y, m[2] * x + m[3] * y)


def _box_xf(m, b, ox, oy):
    xs, ys = [], []
    for x, y in ((b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])):
        X, Y = _xf(m, x, y)
        xs.append(X)
        ys.append(Y)
    return (ox + min(xs), oy + min(ys), ox + max(xs), oy + max(ys))


def _hit(a, b, pad=0.0):
    return a[0] - pad < b[2] and b[0] < a[2] + pad and a[1] - pad < b[3] and b[1] < a[3] + pad


def _infl(b, d):
    return (b[0] - d, b[1] - d, b[2] + d, b[3] + d)


def _seg_box(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def _seg_hits_box(a, b, box, tol=0.5):
    """Does the axis-aligned segment a-b pass through the interior of box?"""
    x0, y0, x1, y1 = _seg_box(a, b)
    return max(x0, box[0] + tol) <= min(x1, box[2] - tol) and max(y0, box[1] + tol) <= min(y1, box[3] - tol)


def _orient_for(kind, want):
    pins = _DEFS[kind][0]
    for m in ORIENTS:
        if all(_xf(m, pins[p][2], pins[p][3]) == tuple(d) for p, d in want.items()):
            return m
    return ID


def _text_w(s, size=FS):
    return len(s) * size * 0.6


# =========================================================================== model

class _Part:
    __slots__ = ("name", "ref", "kind", "pins", "lines", "flags", "order")

    def __init__(self, name, ref, kind, pins, lines, flags=None, order=0):
        self.name, self.ref, self.kind, self.pins = name, ref, kind, pins
        self.lines, self.flags, self.order = lines, flags or {}, order

    def __repr__(self):
        return f"<{self.name} {self.kind} {self.pins}>"


def _fmt_volts(v):
    s = format_value(abs(v))
    return ("+" if v > 0 else "−") + s + "V"


def _uval(v, unit=""):
    return format_value(v).replace("u", "µ") + unit


def _diode_kind(model):
    name = (model or "").upper()
    desc = ""
    try:
        from .parts_data import PARTS
        desc = str(PARTS.get(name, {}).get("desc", "")).lower()
    except Exception:
        pass
    if "LED" in name or " led" in (" " + desc):
        return "LED"
    if "schottky" in desc or name.startswith(("1N58", "BAT", "SS1", "SB1")):
        return "SCH"
    return "D"


def _model_label(model):
    m = (model or "").upper()
    if m == "OPAMP_IDEAL":
        return "ideal"
    if m.startswith("LED_"):
        return m[4:].lower() + " LED"
    return model or ""


class _Model:
    """Parts, rails and nets derived from a Circuit, ready for layout."""

    def __init__(self, c: Circuit, node_voltages=None):
        self.c = c
        self.title = c.title or "Untitled circuit"
        self.parts: list = []
        self.rails: dict = {}          # net -> volts (fixed, drawn as flags)
        self.rail_label: dict = {}
        self.in_net = None
        self.in_term = False
        self.out_net = c.output
        self.first_parts: set = set()  # parts drawn first at the start net (floating input source)
        inp = (c.input or "").upper() or None
        signal_srcs = []
        pot_done = set()
        for idx, e in enumerate(c.elements):
            if e.pot:
                if e.pot in pot_done:
                    continue
                pot_done.add(e.pot)
                p = c.pots[e.pot]
                val = f"{format_value(p.value)} {p.taper}"
                plab = p.label if (p.label and p.label.upper() != p.name.upper()) else ""
                self.parts.append(_Part(p.name, p.name, "POT", {"a": p.a, "w": p.w, "b": p.b},
                                        [p.name, val + (f" · {plab}" if plab else "")],
                                        {"val": val, "plabel": plab}, idx))
                continue
            k = e.kind
            n = e.nodes
            if k in "VI":
                signal = any(key in e.params for key in ("ac", "sin", "pulse"))
                if e.name == inp and k in "VI":
                    if n[1] == "0" and n[0] != "0":
                        self.in_net, self.in_term = n[0], True
                        continue
                    if n[0] == "0" and n[1] != "0":
                        self.in_net, self.in_term = n[1], True
                        continue
                    self.in_net = n[0]
                elif k == "V" and not signal and (n[0] == "0") != (n[1] == "0") and (e.value or 0) != 0:
                    net = n[0] if n[1] == "0" else n[1]
                    v = e.value if n[1] == "0" else -e.value
                    if net not in self.rails:
                        self.rails[net] = float(v)
                        self.rail_label[net] = _fmt_volts(v)
                        continue
                if n[0] == n[1]:
                    continue
                lines = [e.name, ("AC" if signal else _uval(e.value or 0, "V" if k == "V" else "A"))]
                part = _Part(e.name, e.name, k, {"p1": n[0], "p2": n[1]}, lines, {"ac": signal}, idx)
                self.parts.append(part)
                if e.name == inp:
                    self.first_parts.add(part.name)
                else:
                    signal_srcs.append(part)
                continue
            ref = e.name
            if k == "R":
                part = _Part(e.name, ref, "R", {"p1": n[0], "p2": n[1]}, [ref, format_value(e.value)])
            elif k == "C":
                kind = "CP" if (e.value or 0) >= 1e-6 * 0.999 else "C"
                part = _Part(e.name, ref, kind, {"p1": n[0], "p2": n[1]}, [ref, _uval(e.value, "F")])
            elif k == "L":
                part = _Part(e.name, ref, "L", {"p1": n[0], "p2": n[1]}, [ref, _uval(e.value, "H")])
            elif k == "D":
                part = _Part(e.name, ref, _diode_kind(e.model), {"p1": n[0], "p2": n[1]},
                             [ref, _model_label(e.model)])
            elif k == "Q":
                mt = c.models.get(e.model).type if e.model in c.models else "NPN"
                part = _Part(e.name, ref, "Q", {"c": n[0], "b": n[1], "e": n[2]},
                             [ref, _model_label(e.model)], {"pnp": mt == "PNP"})
            elif k == "J":
                mt = c.models.get(e.model).type if e.model in c.models else "NJF"
                part = _Part(e.name, ref, "J", {"d": n[0], "g": n[1], "s": n[2]},
                             [ref, _model_label(e.model)], {"pch": mt == "PJF"})
            elif k == "M":
                # drawn on the 3-pin FET frame; the body arrow joins the source inside the
                # symbol, and a body tied elsewhere is named on the label
                mt = c.models.get(e.model).type if e.model in c.models else "NMOS"
                lines = [ref, _model_label(e.model)]
                if n[3] != n[2]:
                    lines.append("body " + ("GND" if n[3] == "0" else n[3]))
                part = _Part(e.name, ref, "J", {"d": n[0], "g": n[1], "s": n[2]}, lines,
                             {"pch": mt == "PMOS", "mos": True})
            elif k == "X":
                if len(ref) > 1 and ref[0] == "X" and ref[1].isalpha():
                    ref = ref[1:]
                part = _Part(e.name, ref, "X", {"inp": n[0], "inn": n[1], "vcc": n[2], "vee": n[3], "out": n[4]},
                             [ref, _model_label(e.model)])
            elif k in "EG":
                val = f"×{format_value(e.value)}" if k == "E" else f"{format_value(e.value)}S"
                part = _Part(e.name, ref, k, {"p": n[0], "n": n[1], "cp": n[2], "cn": n[3]}, [ref, val])
            elif k == "S":
                part = _Part(e.name, ref, "S", {"p1": n[0], "p2": n[1]},
                             [ref, "on" if e.params.get("on") else "off"], {"on": bool(e.params.get("on"))})
            else:
                continue
            part.order = idx
            self.parts.append(part)
        if self.out_net in self.rails or self.out_net == "0":
            self.out_net = None
        # battery mode: nothing but one supply drives the circuit -> draw it as a source on the left
        self.battery = None
        has_active = any(p.kind in ("Q", "J", "X", "E", "G") for p in self.parts)
        if self.in_net is None and not signal_srcs and len(self.rails) == 1 and not has_active:
            net, v = next(iter(self.rails.items()))
            src = next((e for e in c.elements if e.kind == "V" and net in e.nodes), None)
            if src is not None:
                del self.rails[net]
                part = _Part(src.name, src.name, "V", {"p1": net if v > 0 else "0", "p2": "0" if v > 0 else net},
                             [src.name, _uval(abs(v), "V")], {"ac": False}, -1)
                self.parts.insert(0, part)
                self.first_parts.add(part.name)
                self.battery = part.name
                self.in_net = net
        self.volts = self._estimate(node_voltages)
        pos = [v for v in self.rails.values() if v > 0]
        neg = [v for v in self.rails.values() if v < 0]
        self.top_positive = bool(pos) or not neg
        self.rail_up = {n: ((v > 0) if self.top_positive else (v < 0)) for n, v in self.rails.items()}
        # electrolytic polarity: + toward the higher DC estimate
        for p in self.parts:
            if p.kind == "CP":
                a, b = p.pins["p1"], p.pins["p2"]
                if self.volts.get(b, 0.0) > self.volts.get(a, 0.0) + 0.05:
                    p.pins = {"p1": b, "p2": a}
        self.refs = self._find_refs()
        self.fixed = {"0"} | set(self.rails) | set(self.refs)
        self.nets = defaultdict(list)
        for p in self.parts:
            for lp, n in p.pins.items():
                self.nets[n].append((p, lp))

    # -- rough DC estimate (polarity of electrolytics, ref-node detection) -------
    def _estimate(self, given):
        c = self.c
        known = {"0": 0.0}
        known.update(self.rails)
        nodes = [n for n in c.nodes() if n not in known]
        idx = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)
        out = dict(known)
        if N == 0:
            return out
        Gm = np.zeros((N, N))
        I = np.zeros(N)

        def g(a, b, cond):
            ia, ib = idx.get(a), idx.get(b)
            if ia is not None:
                Gm[ia, ia] += cond
                if ib is None and b in known:
                    I[ia] += cond * known[b]
            if ib is not None:
                Gm[ib, ib] += cond
                if ia is None and a in known:
                    I[ib] += cond * known[a]
            if ia is not None and ib is not None:
                Gm[ia, ib] -= cond
                Gm[ib, ia] -= cond

        def pin_to(a, v, cond):
            ia = idx.get(a)
            if ia is not None:
                Gm[ia, ia] += cond
                I[ia] += cond * v

        try:
            for e in c.elements:
                k, n = e.kind, e.nodes
                if k == "R":
                    g(n[0], n[1], 1.0 / max(e.value or 1.0, 1e-3))
                elif k == "L":
                    g(n[0], n[1], 1.0)
                elif k == "D":
                    g(n[0], n[1], 1e-5)
                elif k == "Q":
                    g(n[0], n[2], 1e-4)
                    g(n[1], n[2], 1e-5)
                elif k in ("J", "M"):
                    g(n[0], n[2], 2e-4)
                elif k == "S":
                    g(n[0], n[1], 10.0 if e.params.get("on") else 1e-9)
                elif k == "V" and not (n[0] in known and n[1] in known):
                    g(n[0], n[1], 1.0)
                elif k == "X":
                    vt = (known.get(n[2], 0.0) + known.get(n[3], 0.0)) / 2
                    pin_to(n[4], vt, 1.0)
            for i in range(N):
                Gm[i, i] += 1e-9
            v = np.linalg.solve(Gm, I)
            for n, i in idx.items():
                out[n] = float(v[i])
        except Exception:
            for n in nodes:
                out[n] = 0.0
        if given:
            for n, v in given.items():
                try:
                    out[str(n).lower()] = float(v)
                except (TypeError, ValueError):
                    pass
        return out

    def _find_refs(self):
        """Bias nodes (resistor to a rail + R/C to ground) feeding several parts -> VREF flags."""
        refs = {}
        by_net = defaultdict(list)
        for p in self.parts:
            for lp, n in p.pins.items():
                by_net[n].append((p, lp))
        for net, conns in by_net.items():
            if net in self.rails or net == "0" or net in (self.in_net, self.out_net):
                continue
            if any(p.kind in ("Q", "J", "D", "SCH", "LED", "E", "G") or (p.kind == "X" and lp in ("out", "vcc", "vee"))
                   or p.kind == "POT" for p, lp in conns):
                continue
            to_rail = to_gnd = 0
            consumers = 0
            for p, lp in conns:
                others = [n for q, n in p.pins.items() if q != lp]
                if p.kind == "R" and others and others[0] in self.rails:
                    to_rail += 1
                elif p.kind in ("R", "C", "CP") and others and others[0] == "0":
                    to_gnd += 1
                else:
                    consumers += 1
            if to_rail >= 1 and to_gnd >= 1 and consumers >= 1 and len(conns) >= 4:
                lab = net.upper() if len(net) <= 5 and net.isalnum() else "VREF"
                refs[net] = lab
        return refs

    def label_of(self, net):
        if net in self.rails:
            return self.rail_label[net]
        return self.refs.get(net, net.upper())

    def up_pin(self, part):
        if part.kind == "Q":
            npn = not part.flags.get("pnp")
            return "c" if npn == self.top_positive else "e"
        if part.kind == "J":
            nch = not part.flags.get("pch")
            return "d" if nch == self.top_positive else "s"
        return None

    def fixed_up(self, net):
        """Is a flag for this fixed net drawn pointing up?"""
        if net in self.rails:
            return self.rail_up[net]
        return False


# =========================================================================== spine

def _traversals(p):
    k = p.kind
    if k in _TWO:
        return (("p1", "p2", 0), ("p2", "p1", 0))
    if k == "Q":
        return (("b", "c", 3), ("b", "e", 2.5), ("e", "c", 2))
    if k == "J":
        return (("g", "d", 3), ("g", "s", 2.5), ("s", "d", 2))
    if k == "X":
        return (("inp", "out", 3), ("inn", "out", 3))
    if k == "POT":
        return (("a", "w", 0), ("b", "w", 0), ("a", "b", 0), ("b", "a", 0))
    if k in ("E", "G"):
        return (("cp", "p", 2),)
    return ()


def _spines(M, budget=60000):
    """Return candidate spines [(nets, steps)], best first. steps: (part, enter, exit)."""
    adj = defaultdict(list)
    for p in M.parts:
        for a, b, w in _traversals(p):
            u, v = p.pins[a], p.pins[b]
            if u in M.fixed or v in M.fixed or u == v:
                continue
            adj[u].append((v, p, a, b, w))
    for u in adj:
        adj[u].sort(key=lambda t: t[1].order)
    goal = M.out_net if M.out_net and M.out_net not in M.fixed else None
    if M.in_net and M.in_net not in M.fixed:
        starts = [M.in_net]
    else:
        starts = [n for n in M.c.nodes() if n not in M.fixed and n in adj]
    found = []
    count = [0]

    def key(act, npas):
        return (act, -npas)

    for s in starts:
        def dfs(net, path, visited, act, npas):
            count[0] += 1
            if count[0] > budget:
                return
            if goal is None or net == goal:
                if path:
                    found.append((key(act, npas), list(path)))
                if goal is not None:
                    return
            for v, p, a, b, w in adj.get(net, ()):
                if v in visited or any(st[0] is p for st in path):
                    continue
                visited.add(v)
                path.append((p, a, b, net, v))
                dfs(v, path, visited, act + w, npas + (0 if w else 1))
                path.pop()
                visited.discard(v)

        dfs(s, [], {s}, 0.0, 0)
    found.sort(key=lambda t: t[0], reverse=True)
    out, seen = [], set()
    for _, path in found:
        sig = tuple(st[0].name for st in path)
        if sig in seen:
            continue
        seen.add(sig)
        nets = [path[0][3]] + [st[4] for st in path]
        out.append((nets, [(st[0], st[1], st[2]) for st in path]))
        if len(out) >= 3:
            break
    if not out:
        s = starts[0] if starts else (M.in_net or next((n for n in M.c.nodes() if n not in M.fixed), None))
        out.append(([s] if s else [], []))
    return out


# =========================================================================== placement objects

class _Label:
    __slots__ = ("lines", "x", "y", "anchor", "box", "size")

    def __init__(self, lines, x, y, anchor, size=FS):
        self.lines, self.x, self.y, self.anchor, self.size = list(lines), x, y, anchor, size
        w = max((_text_w(s, size) for s in self.lines), default=0)
        h = len(self.lines) * LH
        x0 = x if anchor == "start" else (x - w / 2 if anchor == "middle" else x - w)
        self.box = (x0, y, x0 + w, y + h)

    def moved(self, dx, dy):
        return _Label(self.lines, self.x + dx, self.y + dy, self.anchor, self.size)


class _Placed:
    def __init__(self, part, x, y, m, pinmap=None):
        self.part, self.x, self.y, self.m = part, x, y, m
        self.pinmap = pinmap or {}
        self.labels = []
        self.terms = []
        d = _DEFS[part.kind]
        self.core = _box_xf(m, d[1], x, y)
        xs = [self.core[0], self.core[2]]
        ys = [self.core[1], self.core[3]]
        for lp in part.pins:
            px, py, _, _ = self.pin(lp)
            xs.append(px)
            ys.append(py)
        self.box = (min(xs), min(ys), max(xs), max(ys))

    @property
    def name(self):
        return self.part.name

    def pin(self, lp):
        sp = self.pinmap.get(lp, lp)
        px, py, dx, dy = _DEFS[self.part.kind][0][sp]
        X, Y = _xf(self.m, px, py)
        DX, DY = _xf(self.m, dx, dy)
        return (self.x + X, self.y + Y, DX, DY)


class _Term:
    """Ground symbol, rail flag or reference flag attached to a pin by a short stub."""
    __slots__ = ("kind", "x", "y", "label", "stub", "box", "net", "lbox", "owner")

    def __init__(self, kind, x, y, label, stub, net, owner):
        self.kind, self.x, self.y, self.label, self.stub, self.net, self.owner = kind, x, y, label, stub, net, owner
        self.lbox = None
        if kind == "gnd":
            self.box = (x - 9, y - 1, x + 9, y + 9)
        elif kind == "gndup":
            self.box = (x - 9, y - 9, x + 9, y + 1)
        else:
            w = _text_w(label) / 2
            if kind == "railup":
                self.lbox = (x - w, y - 4 - LH, x + w, y - 4)
                self.box = (min(x - 8, x - w), y - 4 - LH, max(x + 8, x + w), y + 1)
            else:
                self.lbox = (x - w, y + 4, x + w, y + 4 + LH)
                self.box = (min(x - 8, x - w), y - 1, max(x + 8, x + w), y + 4 + LH)

    def moved_to(self, ny):
        stub = list(self.stub)
        stub[-1] = (stub[-1][0], ny)
        return _Term(self.kind, self.x, ny, self.label, stub, self.net, self.owner)


def _label_options(P):
    part, k = P.part, P.part.kind
    lines = part.lines
    cb = P.core
    cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
    if k in ("X", "IN", "OUT", "NET"):
        return [[]]

    def above(ls, gap=3):
        return _Label(ls, cx, cb[1] - gap - len(ls) * LH, "middle")

    def below(ls, gap=3):
        return _Label(ls, cx, cb[3] + gap, "middle")

    def right(ls, gap=6):
        return _Label(ls, cb[2] + gap, cy - len(ls) * LH / 2, "start")

    def left(ls, gap=6):
        return _Label(ls, cb[0] - gap, cy - len(ls) * LH / 2, "end")

    if k in _TWO:
        horiz = abs(P.pin("p1")[1] - P.pin("p2")[1]) < 1
        if horiz:
            return [[above(lines[:1]), below(lines[1:])], [above(lines)], [below(lines)]]
        return [[right(lines)], [left(lines)]]
    if k in ("Q", "J"):
        bx = P.pin("b" if k == "Q" else "g")[0]
        if bx < cx:
            return [[right(lines, 5)], [left(lines, 22)], [above(lines)], [below(lines)]]
        return [[left(lines, 5)], [right(lines, 22)], [above(lines)], [below(lines)]]
    if k == "POT":
        _, _, dx, dy = P.pin("w")
        if dx != 0:
            lines = [part.ref, part.flags.get("val", "")] + ([part.flags["plabel"]] if part.flags.get("plabel") else [])
        l3 = [part.ref, part.flags.get("val", "")] + ([part.flags["plabel"]] if part.flags.get("plabel") else [])
        if dy < 0:
            return [[below(lines)], [below(lines, 14)], [above(lines, 18)], [right(l3, 18)], [left(l3, 18)],
                    [_Label(l3, cb[2] + 18, cb[3] + 4, "start")], [_Label(l3, cb[0] - 18, cb[3] + 4, "end")]]
        if dy > 0:
            return [[above(lines)], [above(lines, 14)], [below(lines, 18)], [right(l3, 18)], [left(l3, 18)],
                    [_Label(l3, cb[2] + 18, cb[1] - 4 - len(l3) * LH, "start")],
                    [_Label(l3, cb[0] - 18, cb[1] - 4 - len(l3) * LH, "end")]]
        wy = P.pin("w")[1]
        if dx > 0:
            rb = _Label(lines, cb[2] + 6, wy + 7, "start")
            return [[left(lines)], [rb], [right(lines, 26)], [above(lines)], [below(lines)]]
        lb_ = _Label(lines, cb[0] - 6, wy + 7, "end")
        return [[right(lines)], [lb_], [left(lines, 26)], [above(lines)], [below(lines)]]
    return [[right(lines)], [left(lines)], [above(lines)], [below(lines)]]


# =========================================================================== router

class _Router:
    BEND = 4
    CROSS = 40
    HUG = 3
    JUNC = 2

    def __init__(self):
        self.block = defaultdict(int)
        self.eblock = defaultdict(int)
        self.pins = {}
        self.reserve = {}
        self.edges = {}
        self.use = {}
        self.tree = defaultdict(set)
        self.crossings = 0
        self.bounds = (-100, -100, 100, 100)

    @staticmethod
    def ekey(p, q):
        if p[0] != q[0]:
            return (min(p[0], q[0]), p[1], 0)
        return (p[0], min(p[1], q[1]), 1)

    def add_box(self, b, pad):
        x0, y0, x1, y1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
        i0, i1 = math.ceil(x0 / G), math.floor(x1 / G)
        j0, j1 = math.ceil(y0 / G), math.floor(y1 / G)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.block[(i, j)] += 1
        for i in range(math.ceil(x0 / G - 0.5), math.floor(x1 / G - 0.5) + 1):
            for j in range(j0, j1 + 1):
                self.eblock[(i, j, 0)] += 1
        for i in range(i0, i1 + 1):
            for j in range(math.ceil(y0 / G - 0.5), math.floor(y1 / G - 0.5) + 1):
                self.eblock[(i, j, 1)] += 1

    def add_pin(self, x, y, dx, dy, net):
        p = (int(round(x / G)), int(round(y / G)))
        self.pins[p] = (net, _dir_index(dx, dy))
        q = (p[0] + int(dx), p[1] + int(dy))
        self.reserve.setdefault(q, net)
        return p

    def deg(self, net, p):
        i, j = p
        e = self.edges
        return sum(1 for k in ((i, j, 0), (i - 1, j, 0), (i, j, 1), (i, j - 1, 1)) if e.get(k) == net)

    def commit(self, net, pts):
        for a, b in zip(pts, pts[1:]):
            k = self.ekey(a, b)
            self.edges[k] = net
            for p in (a, b):
                self.use.setdefault(p, {}).setdefault(net, set()).add(k[2])
        for p in pts[1:-1]:
            u = self.use.get(p)
            if u and len(u) > 1:
                self.crossings += 1
        self.tree[net].update(pts)

    def _step(self, net, p, q, di, tree, target, arrive):
        k = self.ekey(p, q)
        if k in self.edges:
            return None
        if q == target:
            return (1, False) if di == arrive else None
        if self.eblock.get(k):
            return None
        if q in self.pins or self.block.get(q):
            return None
        r = self.reserve.get(q)
        if r is not None and r != net:
            return None
        if q in tree:
            return None
        ax = 0 if di in (0, 2) else 1
        cost = 1
        cross = False
        u = self.use.get(q)
        if u:
            if len(u) > 1:
                return None
            (m, axes), = u.items()
            if m == net or axes != {1 - ax} or self.deg(m, q) != 2:
                return None
            cross = True
            cost += self.CROSS
        for ox, oy in (((0, 1), (0, -1)) if ax == 0 else ((1, 0), (-1, 0))):
            n = self.use.get((q[0] + ox, q[1] + oy))
            if n:
                for mm, a in n.items():
                    if mm != net and ax in a:
                        cost += self.HUG
                        break
            if self.block.get((q[0] + ox, q[1] + oy)):
                cost += 0.3
        return cost, cross

    def search(self, net, target, tdir, limit=40000):
        tree = self.tree[net]
        arrive = (tdir + 2) % 4
        i0, j0, i1, j1 = self.bounds
        tx, ty = target
        heap = []
        best = {}
        parent = {}
        cnt = 0
        for s in tree:
            pin = self.pins.get(s)
            if pin is not None:
                if pin[0] != net or self.deg(net, s) > 0:
                    continue
                dirs = (pin[1],)
                extra = 0
            else:
                u = self.use.get(s)
                if u and any(m != net for m in u):
                    continue
                d = self.deg(net, s)
                if d >= 3:
                    continue
                dirs = range(4)
                extra = self.JUNC if d >= 2 else 0
            for di in dirs:
                dx, dy = _DIRS[di]
                q = (s[0] + dx, s[1] + dy)
                if not (i0 <= q[0] <= i1 and j0 <= q[1] <= j1):
                    continue
                r = self._step(net, s, q, di, tree, target, arrive)
                if r is None:
                    continue
                g = r[0] + extra
                key = (q, di, r[1])
                if g < best.get(key, 1e18):
                    best[key] = g
                    parent[key] = ("S", s)
                    hh = abs(q[0] - tx) + abs(q[1] - ty)
                    heapq.heappush(heap, (g + hh, g, cnt, q, di, r[1]))
                    cnt += 1
        n_exp = 0
        while heap:
            f, g, _, p, d, cross = heapq.heappop(heap)
            key = (p, d, cross)
            if g > best.get(key, 1e18):
                continue
            if p == target:
                pts = [p]
                k = key
                while True:
                    par = parent[k]
                    if par[0] == "S":
                        pts.append(par[1])
                        break
                    k = par
                    pts.append(k[0])
                pts.reverse()
                return pts
            n_exp += 1
            if n_exp > limit:
                break
            for di in ((d,) if cross else range(4)):
                if di == (d + 2) % 4:
                    continue
                dx, dy = _DIRS[di]
                q = (p[0] + dx, p[1] + dy)
                if not (i0 <= q[0] <= i1 and j0 <= q[1] <= j1):
                    continue
                r = self._step(net, p, q, di, tree, target, arrive)
                if r is None:
                    continue
                ng = g + r[0] + (self.BEND if di != d else 0)
                nk = (q, di, r[1])
                if ng < best.get(nk, 1e18):
                    best[nk] = ng
                    parent[nk] = key
                    hh = abs(q[0] - tx) + abs(q[1] - ty)
                    if q[0] != tx and q[1] != ty:
                        hh += self.BEND
                    heapq.heappush(heap, (ng + hh, ng, cnt, q, di, r[1]))
                    cnt += 1
        return None

    def route(self, net, pinlist):
        """Connect every (point, dir) in pinlist to the net's tree. Returns failed pins."""
        tree = self.tree[net]
        todo = []
        for p, d in pinlist:
            if p not in tree and all(p != t[0] for t in todo):
                todo.append((p, d))
        if not tree:
            if not todo:
                return []
            first = todo.pop(0)
            tree.add(first[0])
        failed = []
        # the signal row first: left/right-facing pins in x order, then the rest nearest-first
        rows = [t for t in todo if t[1] in (0, 2)]
        if len(rows) >= 2 and len(tree) == 1:
            rows.sort(key=lambda t: t[0][0])
            for t in rows:
                if t[0] in tree:
                    continue
                todo.remove(t)
                path = self.search(net, t[0], t[1])
                if path is None:
                    failed.append(t)
                else:
                    self.commit(net, path)
        while todo:
            tl = list(tree)

            def dist(t):
                return min(abs(t[0][0] - a[0]) + abs(t[0][1] - a[1]) for a in tl)
            nxt = min(todo, key=dist)
            todo.remove(nxt)
            path = self.search(net, nxt[0], nxt[1])
            if path is None:
                failed.append(nxt)
                continue
            self.commit(net, path)
        return failed


# =========================================================================== layout engine

class _Hang:
    __slots__ = ("part", "near", "far", "dir")

    def __init__(self, part, near):
        self.part, self.near, self.far, self.dir = part, near, [], 1


class _Engine:
    def __init__(self, M, opts):
        self.M = M
        self.o = opts
        self.placed = {}
        self.terms = []
        self.obst = []            # (box, tag)
        self.R = _Router()
        self.netpins = defaultdict(list)
        self.fixed_wires = []     # (id, [points px])
        self.buses = []           # (net, [points px]) rail buses drawn as wires
        self.bus_dots = []
        self.failed = []
        self.tags = []
        self.uid = 0
        self.claimed = set()
        self.hang_up = defaultdict(list)
        self.hang_down = defaultdict(list)
        self.anchored = set()
        self.island_parts = set()

    # -- obstacle bookkeeping ------------------------------------------------
    def _hits(self, box, pad):
        b = _infl(box, pad)
        for o, _ in self.obst:
            if _hit(b, o):
                return True
        return False

    def _stub_hits(self, stub, pad=3):
        for a, b in zip(stub, stub[1:]):
            if self._hits(_seg_box(a, b), pad):
                return True
        return False

    def _term_plans(self, P):
        out = []
        M = self.M
        for lp, net in P.part.pins.items():
            if net not in M.fixed:
                continue
            px, py, dx, dy = P.pin(lp)
            if net == "0":
                if dy > 0:
                    out.append(_Term("gnd", px, py + 10, "", [(px, py), (px, py + 10)], net, P.name))
                elif dy < 0:
                    out.append(_Term("gndup", px, py - 10, "", [(px, py), (px, py - 10)], net, P.name))
                else:
                    x2 = px + 20 * dx
                    out.append(_Term("gnd", x2, py + 10, "", [(px, py), (x2, py), (x2, py + 10)], net, P.name))
            else:
                lab = M.label_of(net)
                up = M.fixed_up(net)
                if dy < 0:
                    out.append(_Term("railup", px, py - 10, lab, [(px, py), (px, py - 10)], net, P.name))
                elif dy > 0:
                    out.append(_Term("raildn", px, py + 10, lab, [(px, py), (px, py + 10)], net, P.name))
                else:
                    x2 = px + 20 * dx
                    if up:
                        out.append(_Term("railup", x2, py - 10, lab, [(px, py), (x2, py), (x2, py - 10)], net, P.name))
                    else:
                        out.append(_Term("raildn", x2, py + 10, lab, [(px, py), (x2, py), (x2, py + 10)], net, P.name))
        return out

    def plan(self, part, x, y, m, pinmap=None, check=True, prefer=None, avoid=None):
        x, y = int(round(x / G)) * G, int(round(y / G)) * G
        P = _Placed(part, x, y, m, pinmap)
        if check and self._hits(P.box, 6):
            return None
        terms = self._term_plans(P)
        if check:
            for t in terms:
                if self._hits(t.box, 4) or self._stub_hits(t.stub):
                    return None
        if avoid is not None and avoid(P):
            return None
        if check:
            for ab in self._approach_boxes(P):
                if self._hits(ab, 5):
                    return None
        opts = _label_options(P)
        appr = self._approach_boxes(P)
        opts = [o for o in opts if not any(_hit(lb.box, ab, 2) for lb in o for ab in appr)] or opts
        if prefer is not None:
            opts = sorted(opts, key=lambda o: 0 if (o and o[0].anchor == prefer) else 1)
        for opt in opts:
            ok = True
            if check:
                for lb in opt:
                    if self._hits(lb.box, 3) or any(_hit(lb.box, t.box, 2) for t in terms):
                        ok = False
                        break
            if ok:
                P.labels = opt
                P.terms = terms
                return P
        if not check:
            P.labels = opts[0]
            P.terms = terms
            return P
        return None

    def _approach_boxes(self, P):
        out = []
        for lp, net in P.part.pins.items():
            if net in self.M.fixed:
                continue
            px, py, dx, dy = P.pin(lp)
            ax, ay = px + 10 * dx, py + 10 * dy
            out.append((min(px + 2 * dx, ax), min(py + 2 * dy, ay), max(px + 2 * dx, ax), max(py + 2 * dy, ay)))
        return out

    def push_right(self, part, x, y, m, pinmap=None, prefer=None, avoid=None, steps=120):
        for k in range(steps):
            P = self.plan(part, x + k * G, y, m, pinmap, prefer=prefer, avoid=avoid)
            if P is not None:
                return P
        return self.plan(part, x + steps * G, y, m, pinmap, check=False, prefer=prefer)

    def commit(self, P):
        self.placed[P.name] = P
        self.claimed.add(P.name)
        self.obst.append((P.box, "body"))
        self.R.add_box(P.box, 4)
        for lb in P.labels:
            self.obst.append((lb.box, "label"))
            self.R.add_box(lb.box, 2)
        for t in P.terms:
            self._add_term(t)
        for lp, net in P.part.pins.items():
            px, py, dx, dy = P.pin(lp)
            if net in self.M.fixed:
                continue
            g = self.R.add_pin(px, py, dx, dy, net)
            self.netpins[net].append((g, _dir_index(dx, dy)))
            self.anchored.add(net)
        for t in P.terms:
            px, py = t.stub[0]
            self.R.pins[(int(round(px / G)), int(round(py / G)))] = (f"#t{id(t)}", 0)

    def _add_term(self, t):
        self.terms.append(t)
        self.obst.append((t.box, "term"))
        self.R.add_box(t.box, 3)
        for a, b in zip(t.stub, t.stub[1:]):
            self.obst.append((_seg_box(a, b), f"stub{id(t)}"))
        self._commit_fixed(f"#t{id(t)}", t.stub)

    def _commit_fixed(self, nid, pts):
        gp = [(int(round(x / G)), int(round(y / G))) for x, y in pts]
        full = [gp[0]]
        for q in gp[1:]:
            p = full[-1]
            while p != q:
                p = (p[0] + (q[0] > p[0]) - (q[0] < p[0]), p[1] + (q[1] > p[1]) - (q[1] < p[1]))
                full.append(p)
        for a, b in zip(full, full[1:]):
            k = self.R.ekey(a, b)
            self.R.edges[k] = nid
            for p in (a, b):
                self.R.use.setdefault(p, {}).setdefault(nid, set()).add(k[2])
        self.R.tree[nid].update(full)

    def _remove_term(self, t):
        self.terms.remove(t)
        self.obst = [o for o in self.obst if not ((o[1] == "term" and o[0] == t.box) or o[1] == f"stub{id(t)}")]
        nid = f"#t{id(t)}"
        for k in [k for k, v in self.R.edges.items() if v == nid]:
            del self.R.edges[k]
        for p in list(self.R.tree.get(nid, ())):
            u = self.R.use.get(p)
            if u and nid in u:
                del u[nid]
                if not u:
                    del self.R.use[p]
        self.R.tree.pop(nid, None)
        self._rebuild_blocks()

    def _rebuild_blocks(self):
        self.R.block = defaultdict(int)
        self.R.eblock = defaultdict(int)
        for P in self.placed.values():
            self.R.add_box(P.box, 4)
            for lb in P.labels:
                self.R.add_box(lb.box, 2)
        for t in self.terms:
            self.R.add_box(t.box, 3)

    # -- classification --------------------------------------------------------
    def _hang_tree(self, part, near, depth, seen_nets, used):
        if part.kind not in _TWO and part.kind != "POT":
            return None
        if part.name in self.claimed or part.name in used:
            return None
        M = self.M
        used.add(part.name)
        node = _Hang(part, near)
        near_net = part.pins[near]
        for lp, n in part.pins.items():
            if lp == near:
                continue
            if n in M.fixed or n == near_net:
                node.far.append((lp, n, []))
                continue
            if n in self.anchored or n in seen_nets or depth >= 4:
                used.discard(part.name)
                return None
            seen_nets.add(n)
            kids = []
            for q, qp in M.nets[n]:
                if q is part or q.name in used:
                    if q is part:
                        continue
                    continue
                k = self._hang_tree(q, qp, depth + 1, seen_nets, used)
                if k is None:
                    if self.o.get("strict"):
                        used.discard(part.name)
                        return None
                    continue        # a branch that leads elsewhere is placed later on its own
                kids.append(k)
            node.far.append((lp, n, kids))
        return node

    @staticmethod
    def _tree_nets(h, acc):
        for lp, n, kids in h.far:
            acc.add(n)
            for k in kids:
                _Engine._tree_nets(k, acc)
        return acc

    @staticmethod
    def _tree_parts(h, acc):
        acc.add(h.part.name)
        for _, _, kids in h.far:
            for k in kids:
                _Engine._tree_parts(k, acc)
        return acc

    def _set_dir(self, h, d):
        h.dir = d
        for _, _, kids in h.far:
            for k in kids:
                self._set_dir(k, d)

    def classify(self, nets_in_order, steps):
        M = self.M
        spine_parts = [st[0].name for st in steps]
        order = list(nets_in_order)
        for part, ent, ext in steps:
            if part.kind in ("Q", "J", "X") or (part.kind == "POT" and ext == "w"):
                for lp, n in part.pins.items():
                    if lp in (ent, ext) or (part.kind == "X" and lp in ("vcc", "vee")):
                        continue
                    if n not in M.fixed and n not in order:
                        order.append(n)
        self.anchored = set(order)
        for name in spine_parts:
            self.claimed.add(name)
        for net in order:
            for part, lp in M.nets.get(net, []):
                if part.name in self.claimed or part.name in self.island_parts:
                    continue
                if part.kind == "POT" and lp == "w" and any(
                        n not in M.fixed and n != net for q, n in part.pins.items() if q != "w"):
                    continue
                used = set()
                h = self._hang_tree(part, lp, 0, set(), used)
                if h is None:
                    continue
                touched = self._tree_nets(h, set())
                up = any(M.fixed_up(n) for n in touched if n in M.rails)
                down = "0" in touched or any(n in M.fixed and not M.fixed_up(n) for n in touched)
                d = -1 if (up and not down) else 1
                self._set_dir(h, d)
                (self.hang_up if d < 0 else self.hang_down)[net].append(h)
                for nm in self._tree_parts(h, set()):
                    self.claimed.add(nm)
                self.anchored |= touched - M.fixed
        for lst in list(self.hang_up.values()) + list(self.hang_down.values()):
            lst.sort(key=lambda h: (0 if h.part.name in M.first_parts else 1, h.part.order))
        # spine parts are not "claimed" for placement purposes until placed
        for name in spine_parts:
            self.claimed.discard(name)

    # -- hanger placement --------------------------------------------------------
    def push_near(self, part, x, y, m, d, prefer=None, steps=40):
        """Nearest free spot going right and/or away from the parent (for chain kids)."""
        for k in range(steps):
            for j in range(0, k + 1, 2):
                P = self.plan(part, x + (k - j) * G, y + j * G * d, m, prefer=prefer)
                if P is not None:
                    return P
        return self.push_right(part, x, y, m, prefer=prefer)

    def place_hang(self, h, x, near_y, prefer=None, avoid=None, kid=False):
        part = h.part
        d = h.dir
        want = {h.near: (0, -d)}
        if part.kind == "POT" and h.near != "w":
            want["w"] = (1, 0)
        m = _orient_for(part.kind, want)
        lx, ly, _, _ = _DEFS[part.kind][0][h.near]
        X, Y = _xf(m, lx, ly)
        if part.kind == "POT" and prefer is None and any(kids for _, _, kids in h.far):
            prefer = "start"
        if kid:
            P = self.push_near(part, x - X, near_y - Y, m, d, prefer=prefer)
        else:
            P = self.push_right(part, x - X, near_y - Y, m, prefer=prefer, avoid=avoid)
        self.commit(P)
        for lp, n, kids in h.far:
            if not kids:
                continue
            px, py, dx, dy = P.pin(lp)
            if dx == 0:
                cx, cy = px, py + 20 * d
            else:
                cx, cy = px + 20 * dx, py + 20 * d
            for k in kids:
                Pk = self.place_hang(k, cx, cy, kid=True)
                cx = Pk.pin(k.near)[0] + 20
        return P

    def place_row(self, net, y, x_up, x_down):
        """Hang the up/down hanger trees of a net from its row at height y."""
        ups = self.hang_up.pop(net, [])
        downs = self.hang_down.pop(net, [])
        cols = set()
        right = max(x_up, x_down)
        xu = x_up
        for h in ups:
            P = self.place_hang(h, xu, y - 20)
            cols.add(P.x if h.part.kind not in ("POT",) else P.pin(h.near)[0])
            xu = P.pin(h.near)[0] + 20
            right = max(right, xu)
        xd = x_down + (20 if (ups and x_down == x_up) else 0)

        for h in downs:
            near = h.near

            def avoid_cols(P, near=near):
                return P.pin(near)[0] in cols
            P = self.place_hang(h, xd, y + 20, avoid=avoid_cols)
            xd = P.pin(near)[0] + 20
            right = max(right, xd)
        return right

    # -- spine placement -------------------------------------------------------------
    def place_spine(self, nets, steps):
        M = self.M
        y = 0
        x_enter = 0
        if nets and M.in_term and nets[0] == M.in_net:
            P = self.plan(_Part("IN", "IN", "IN", {"p": M.in_net}, []), -10, 0, ID)
            self.commit(P)
            x_enter = 0
        x_up = x_down = x_enter + 20
        for i, net in enumerate(nets):
            cur = self.place_row(net, y, x_up, x_down)
            if i >= len(steps):
                break
            part, ent, ext = steps[i]
            cur, y, x_up, x_down = self.place_step(part, ent, ext, cur, y)
        if nets and M.out_net == nets[-1]:
            P = self.push_right(_Part("OUT", "OUT", "OUT", {"p": M.out_net}, []), cur + 10, y, ID)
            self.commit(P)

    def place_step(self, part, ent, ext, x, y):
        M = self.M
        k = part.kind
        if k in _TWO:
            m = ID if ent == "p1" else R180
            P = self.push_right(part, x + 30, y, m)
            self.commit(P)
            ex = P.pin(ext)[0]
            return ex + 20, y, ex + 20, ex + 20
        if k in ("Q", "J") and ent in ("b", "g"):
            up = M.up_pin(part)
            cpin = "c" if k == "Q" else "d"
            m = ID if up == cpin else FY
            side = [p for p in part.pins if p not in (ent, ext)][0]
            P = self.push_right(part, x + 30, y, m)
            self.commit(P)
            sx, sy, sdx, sdy = P.pin(side)
            snet = part.pins[side]
            if snet not in M.fixed:
                self.place_side(snet, sx, sy, sdy)
            ex, ey, edx, edy = P.pin(ext)
            ny = ey + 10 * edy
            right = max(P.box[2], max((lb.box[2] for lb in P.labels), default=P.box[2])) + 20
            if edy < 0:
                return ex + 20, ny, ex, right
            return ex + 20, ny, right, ex
        if k == "X" and ent in ("inp", "inn"):
            return self.place_opamp(part, ent, ext, x, y)
        if k == "POT" and ext == "w" and part.pins[[q for q in part.pins if q not in (ent, ext)][0]] not in M.fixed                 and not self.o.get("vpot"):
            # tone-control style: horizontal pot, the wiper drops to carry the signal on
            m = _orient_for("POT", {ent: (-1, 0), "w": (0, 1)})
            lx, ly, _, _ = _DEFS["POT"][0][ent]
            X, Y = _xf(m, lx, ly)
            P = self.push_right(part, x - X, y - Y, m)
            self.commit(P)
            wx, wy, _, _ = P.pin("w")
            return wx + 20, wy + 10, wx + 20, wx + 20
        if k == "POT" and ext == "w":
            m = _orient_for("POT", {ent: (0, -1), "w": (1, 0)})
            lx, ly, _, _ = _DEFS["POT"][0][ent]
            X, Y = _xf(m, lx, ly)
            P = self.push_right(part, x - X, y + 20 - Y, m)
            self.commit(P)
            third = [p for p in part.pins if p not in (ent, ext)][0]
            tnet = part.pins[third]
            tx, ty, tdx, tdy = P.pin(third)
            if tnet not in M.fixed and tnet != part.pins[ent]:
                self.place_side(tnet, tx, ty, tdy)
            wx, wy, _, _ = P.pin("w")
            return wx + 20, wy, wx + 20, wx + 20
        if k == "POT":
            wnet = part.pins["w"]
            want = {ent: (-1, 0), "w": (0, 1) if wnet == "0" or (wnet in M.fixed and not M.fixed_up(wnet)) else (0, -1)}
            m = _orient_for("POT", want)
            lx, ly, _, _ = _DEFS["POT"][0][ent]
            X, Y = _xf(m, lx, ly)
            P = self.push_right(part, x - X, y - Y, m)
            self.commit(P)
            ex, ey, _, _ = P.pin(ext)
            return ex + 20, ey, ex + 20, ex + 20
        # generic: put the entering pin on the spine
        pin = _DEFS[k][0].get(part_pin(k, ent), (0, 0, -1, 0))
        P = self.push_right(part, x - pin[0] + 10, y - pin[1], ID, self._pinmap_default(part))
        self.commit(P)
        ex, ey, edx, edy = P.pin(ext)
        ny = ey + 10 * edy
        return ex + 20, ny, ex + 20, ex + 20

    def _lane_groups(self, fnet, onet):
        """Unclaimed parts (or two-part series chains) joining fnet and onet: one lane each."""
        M = self.M
        lanes = []
        used = set()
        for part, lp in M.nets.get(fnet, []):
            if part.name in self.claimed or part.name in used or part.name in self.island_parts:
                continue
            nets = set(part.pins.values())
            if part.kind in _TWO or part.kind == "POT":
                if nets == {fnet, onet}:
                    lanes.append([(part, fnet)])
                    used.add(part.name)
                    continue
                others = nets - {fnet}
                if len(others) == 1:
                    mid = next(iter(others))
                    if mid in M.fixed or mid in self.anchored:
                        continue
                    conns = [(q, qp) for q, qp in M.nets.get(mid, []) if q is not part]
                    if len(conns) == 1:
                        q = conns[0][0]
                        if (q.kind in _TWO or q.kind == "POT") and set(q.pins.values()) == {mid, onet} \
                                and q.name not in self.claimed:
                            lanes.append([(part, fnet), (q, mid)])
                            used.update((part.name, q.name))
        lanes.sort(key=lambda ln: (len(ln), ln[0][0].order))
        return lanes

    def _lane_orient(self, part, left_net):
        if part.kind == "POT":
            ends = [lp for lp in ("a", "b") if part.pins[lp] == left_net]
            lp = ends[0] if ends else "w"
            if lp == "w":
                return _orient_for("POT", {"w": (-1, 0)}), "w"
            return _orient_for("POT", {lp: (-1, 0), "w": (0, -1)}), lp
        lp = next((q for q in ("p1", "p2") if part.pins[q] == left_net), "p1")
        return (ID if lp == "p1" else R180), lp

    def place_opamp(self, part, ent, ext, x, y):
        M = self.M
        other = "inn" if ent == "inp" else "inp"
        flip = part.name in self.o.get("flip", ())
        pinmap = {ent: "inB", other: "inA"} if flip else {ent: "inA", other: "inB"}
        snet, onet, enet = part.pins[other], part.pins["out"], part.pins[ent]
        d = -1 if flip else 1                      # side of the other (non-spine) input
        # feedback lanes: parts from either input to the output
        fnet, fside = None, 0
        lanes = []
        if snet not in M.fixed and snet != onet:
            lanes = self._lane_groups(snet, onet)
            fnet, fside = snet, d
        if not lanes and enet not in M.fixed:
            lanes = self._lane_groups(enet, onet)
            fnet, fside = enet, -d
        width = 0
        for ln in lanes:
            width = max(width, sum(60 for _ in ln) + 20 * (len(ln) - 1))
        xb = x + 10
        ox = xb + max(80, width - 40)
        P = self.push_right(part, ox, y + 20 * d, ID, pinmap)
        self.commit(P)
        ox, oy = P.x, P.y
        xo = ox + 70
        for ln in lanes:
            for q, _ in ln:
                self.claimed.add(q.name)
        # lanes, nearest first, stepping away from the op-amp
        if lanes:
            yl = oy + fside * 95
            span = (xo - 10) - (xb + 10)
            for ln in lanes:
                lw = 60 * len(ln) + 20 * (len(ln) - 1)
                for tries in range(12):
                    cx = xb + 10 + max(0, (span - lw) // 2 // G * G)
                    plans = []
                    ok = True
                    for q, lnet in ln:
                        m, lp = self._lane_orient(q, lnet)
                        T = _Placed(q, 0, 0, m)
                        px, py, _, _ = T.pin(lp)
                        Pq = self.plan(q, cx - px, yl - py, m)
                        if Pq is None or any(_hit(_infl(Pq.box, 4), o.box) for o in plans):
                            ok = False
                            break
                        plans.append(Pq)
                        far = [r for r in q.pins if r != lp and q.pins[r] != lnet]
                        cx = (Pq.pin(far[0])[0] if far else Pq.box[2]) + 20
                    if ok and cx - 20 <= xo - 10:
                        for Pq in plans:
                            self.commit(Pq)
                        break
                    yl += fside * 10
                else:
                    for q, _ in ln:
                        self.claimed.discard(q.name)
                yl += fside * 60
        # gain leg (side hangers) below the lanes, on the bus column
        if not flip and snet not in M.fixed and (self.hang_down.get(snet) or self.hang_up.get(snet)):
            hs = self.hang_down.pop(snet, []) + self.hang_up.pop(snet, [])
            if lanes and fnet == snet:
                ny = (max(Pq.box[3] for Pq in self.placed.values() if Pq.part.name in
                          {q.name for ln in lanes for q, _ in ln}) + 40) if lanes else oy + 60
                xs = xb
            else:
                ny, xs = oy + 60, x
            for h in hs:
                self._set_dir(h, 1)
                Ph = self.place_hang(h, xs, ny)
                xs = Ph.pin(h.near)[0] + 20
            self.anchored.add(snet)
        ex, ey, _, _ = P.pin("out")
        return xo + 20, ey, xo + 20, xo + 20

    def _pinmap_default(self, part):
        if part.kind == "X":
            return {"inp": "inA", "inn": "inB"}
        return None

    def place_side(self, net, px, py, dy):
        """Hang a device side net (emitter, source, pot end) from its pin."""
        if net in self.anchored and not (self.hang_down.get(net) or self.hang_up.get(net)):
            return
        self.anchored.add(net)
        d = 1 if dy >= 0 else -1
        same = self.hang_down.pop(net, []) if d > 0 else self.hang_up.pop(net, [])
        opp = self.hang_up.pop(net, []) if d > 0 else self.hang_down.pop(net, [])
        x = px
        for h in same:
            self._set_dir(h, d)
            P = self.place_hang(h, x, py + 20 * d)
            x = P.pin(h.near)[0] + 20
        # hangers going the other way (e.g. a drain decoupling cap to ground) would cut across the
        # signal row; they are placed later by the free-space search, once the rows are routed
        del opp

    # -- terminals alignment ------------------------------------------------------------
    def align_terms(self):
        groups = defaultdict(list)
        for t in self.terms:
            own = self.placed.get(t.owner)
            if own is not None and own.part.kind == "X":
                continue
            if t.kind == "gnd" and len(t.stub) == 2:
                groups["gnd"].append(t)
            elif t.kind == "railup" and len(t.stub) == 2:
                groups["up:" + t.net].append(t)
        for key, ts in groups.items():
            down = key == "gnd"
            ts.sort(key=lambda t: -t.y if down else t.y)
            reach = 40 if down else 10 ** 6
            while ts:
                seed = ts[0].y
                cl = [t for t in ts if abs(t.y - seed) <= reach]
                ts = [t for t in ts if abs(t.y - seed) > reach]
                for t in cl:
                    if t.y == seed:
                        continue
                    nt = t.moved_to(seed)
                    self._remove_term(t)
                    ok = not self._hits(nt.box, 3)
                    if ok:
                        a, b = nt.stub[-2], nt.stub[-1]
                        seg = _seg_box((a[0], a[1] + (5 if down else -5)), b)
                        ok = not self._hits(seg, 2)
                    self._add_term(nt if ok else t)

    def rail_buses(self):
        by = defaultdict(list)
        for t in self.terms:
            own = self.placed.get(t.owner)
            if t.kind == "railup" and not (own is not None and own.part.kind == "X"):
                by[(t.net, t.y)].append(t)
        for (net, y), ts in by.items():
            if len(ts) < 2:
                continue
            ts.sort(key=lambda t: t.x)
            groups = [[ts[0]]]
            for t in ts[1:]:
                a = groups[-1][-1]
                seg = (a.x, y - 1, t.x, y + 1)
                blocked = False
                own = (f"stub{id(a)}", f"stub{id(t)}")
                for o, tag in self.obst:
                    if o in (a.box, t.box) or tag in own:
                        continue
                    if _hit(seg, o, 1):
                        blocked = True
                        break
                if blocked:
                    groups.append([t])
                else:
                    groups[-1].append(t)
            for gr in groups:
                if len(gr) < 2:
                    continue
                pts = [(gr[0].x, y), (gr[-1].x, y)]
                self.buses.append((net, pts))
                for t in gr[1:]:
                    nt = _Term("bar", t.x, t.y, "", t.stub, net, t.owner)
                    nt.box = (t.x - 1, t.y - 1, t.x + 1, t.y + 1)
                    self.terms[self.terms.index(t)] = nt
                    if t is not gr[-1]:
                        self.bus_dots.append((t.x, y))
                self.obst.append(((gr[0].x, y - 1, gr[-1].x, y + 1), "wire"))
                self._commit_fixed(f"#bus{len(self.buses)}", pts)
        self._rebuild_blocks()

    # -- routing ------------------------------------------------------------------
    def set_bounds(self):
        xs, ys = [], []
        for o, _ in self.obst:
            xs += [o[0], o[2]]
            ys += [o[1], o[3]]
        if not xs:
            xs, ys = [0, 100], [0, 100]
        m = 120
        self.R.bounds = (int(math.floor((min(xs) - m) / G)), int(math.floor((min(ys) - m) / G)),
                         int(math.ceil((max(xs) + m) / G)), int(math.ceil((max(ys) + m) / G)))

    def route_net(self, net):
        pins = self.netpins.get(net, [])
        if len(pins) < 2 and not self.R.tree.get(net):
            return
        before = set(self.R.edges)
        failed = self.R.route(net, sorted(pins, key=lambda t: (t[0][0], t[0][1])))
        new = [k for k in self.R.edges if k not in before and self.R.edges[k] == net]
        for k in new:
            i, j, ax = k
            a = (i * G, j * G)
            b = ((i + 1) * G, j * G) if ax == 0 else (i * G, (j + 1) * G)
            self.obst.append((_seg_box(a, b), "wire"))
        for f in failed:
            self.failed.append((net, f))

    def route_all(self, order):
        self.set_bounds()
        done = set()
        for n in order:
            if n in done or n in self.M.fixed:
                continue
            done.add(n)
            self.route_net(n)
        for n in list(self.netpins):
            if n not in done and n not in self.M.fixed:
                done.add(n)
                self.route_net(n)

    # -- leftovers -------------------------------------------------------------------
    def _orient_candidates(self, part):
        k = part.kind
        if k in _TWO:
            return [(m, None) for m in (ID, R180, R90, R270)]
        if k in ("Q", "J"):
            return [(m, None) for m in (ID, FY, FX, R180)]
        if k == "X":
            return [(ID, {"inp": "inA", "inn": "inB"}), (ID, {"inp": "inB", "inn": "inA"})]
        if k == "POT":
            return [(m, None) for m in ORIENTS]
        if k in ("E", "G"):
            return [(ID, None), (FX, None)]
        return [(ID, None)]

    def free_spot(self, part, window=320, trials=5):
        M = self.M
        anchors = {}
        for lp, n in part.pins.items():
            if n in M.fixed:
                continue
            pts = set(self.R.tree.get(n, ())) | {p for p, _ in self.netpins.get(n, [])}
            if pts:
                anchors[lp] = np.array([(p[0] * G, p[1] * G) for p in pts], float)
        bx = [o for o, _ in self.obst] or [(0.0, 0.0, 0.0, 0.0)]
        B = np.array(bx, float).reshape(-1, 4)
        if anchors:
            allp = np.vstack(list(anchors.values()))
            cx, cy = float(np.mean(allp[:, 0])), float(np.mean(allp[:, 1]))
        else:
            cx, cy = float(B[:, 0].min()) + 60, float(B[:, 3].max()) + 120
        cx, cy = round(cx / G) * G, round(cy / G) * G
        cands = []
        for win in (window, window * 2, window * 4):
            xs = np.arange(cx - win, cx + win + 1, G, dtype=float)
            ys = np.arange(cy - win, cy + win + 1, G, dtype=float)
            X, Y = np.meshgrid(xs, ys)
            PX, PY = X.ravel(), Y.ravel()
            for m, pinmap in self._orient_candidates(part):
                T = _Placed(part, 0, 0, m, pinmap)
                terms = self._term_plans(T)
                base_cost = np.zeros(PX.shape)
                for lp, A in anchors.items():
                    px, py, dx, dy = T.pin(lp)
                    qx, qy = PX + px, PY + py
                    DX = A[None, :, 0] - qx[:, None]
                    DY = A[None, :, 1] - qy[:, None]
                    dist = np.abs(DX) + np.abs(DY)
                    along = DX * dx + DY * dy
                    perp = np.abs(DX * dy) + np.abs(DY * dx)
                    pen = np.where(along < 0, 60.0, np.where(along == 0, 14.0, np.where(perp == 0, -8.0, 5.0)))
                    base_cost += (dist + pen).min(axis=1)
                for lp, n in part.pins.items():
                    if n not in M.fixed:
                        continue
                    _, _, dx, dy = T.pin(lp)
                    want_down = n == "0" or not M.fixed_up(n)
                    if (want_down and dy <= 0) or (not want_down and dy >= 0):
                        base_cost += 25.0
                vb = self.o.get("vbias", 0)
                if vb:
                    base_cost += np.where(PY < cy, 0.0, 15.0) if vb < 0 else np.where(PY > cy, 0.0, 15.0)
                fixed_shapes = [_infl(T.box, 6)] + [_infl(t.box, 4) for t in terms] + \
                    [_infl(b, 5) for b in self._approach_boxes(T)]
                for t in terms:
                    for a, b in zip(t.stub, t.stub[1:]):
                        fixed_shapes.append(_infl(_seg_box(a, b), 3))
                free_fixed = np.ones(PX.shape, bool)
                for sh in fixed_shapes:
                    free_fixed &= ~self._coll(PX, PY, sh, B)
                if not free_fixed.any():
                    continue
                appr = self._approach_boxes(T)
                for opt in _label_options(T):
                    if any(_hit(lb.box, ab, 2) for lb in opt for ab in appr):
                        continue
                    free = free_fixed.copy()
                    for lb in opt:
                        free &= ~self._coll(PX, PY, _infl(lb.box, 3), B)
                        for t in terms:
                            if _hit(lb.box, t.box, 2):
                                free[:] = False
                    if not free.any():
                        continue
                    cost = np.where(free, base_cost, np.inf)
                    # best few positions above and below the anchors, for diversity
                    for half in (PY <= cy, PY > cy):
                        cc = np.where(half, cost, np.inf)
                        for i in np.argsort(cc)[:3]:
                            if np.isfinite(cc[i]):
                                cands.append((float(cc[i]), m, pinmap, opt, float(PX[i]), float(PY[i])))
            if cands:
                break
        if not cands:
            P = self.plan(part, float(B[:, 2].max()) + 100, float(B[:, 3].max()) + 100, ID,
                          self._pinmap_default(part), check=False)
            self.commit(P)
            return P
        cands.sort(key=lambda t: t[0])
        picked = []
        for cnd in cands:
            if all(not (cnd[1] == q[1] and cnd[2] == q[2] and abs(cnd[4] - q[4]) + abs(cnd[5] - q[5]) < 40)
                   for q in picked):
                picked.append(cnd)
            if len(picked) >= trials:
                break
        if len(anchors) < 2 or len(picked) == 1:
            best = picked[0]
        else:
            best, best_cost = None, None
            for cnd in picked:
                P = self._make(part, cnd)
                c = self._trial_cost(P) + 0.3 * cnd[0]
                if best_cost is None or c < best_cost:
                    best, best_cost = cnd, c
        P = self._make(part, best)
        self.commit(P)
        return P

    def _make(self, part, cnd):
        _, m, pinmap, opt, px, py = cnd
        P = _Placed(part, px, py, m, pinmap)
        P.labels = [lb.moved(px, py) for lb in opt]
        P.terms = self._term_plans(P)
        return P

    def _snapshot(self):
        R = self.R
        return (dict(R.block), dict(R.eblock), dict(R.pins), dict(R.reserve), dict(R.edges),
                {p: {n: set(a) for n, a in u.items()} for p, u in R.use.items()},
                {n: set(t) for n, t in R.tree.items()}, R.crossings, R.bounds,
                len(self.obst), dict(self.placed), list(self.terms),
                {n: list(v) for n, v in self.netpins.items()}, set(self.anchored), set(self.claimed),
                list(self.failed))

    def _restore(self, s):
        R = self.R
        R.block = defaultdict(int, s[0])
        R.eblock = defaultdict(int, s[1])
        R.pins, R.reserve, R.edges, R.use = s[2], s[3], s[4], s[5]
        R.tree = defaultdict(set, s[6])
        R.crossings, R.bounds = s[7], s[8]
        del self.obst[s[9]:]
        self.placed, self.terms = s[10], s[11]
        self.netpins = defaultdict(list, s[12])
        self.anchored, self.claimed, self.failed = s[13], s[14], s[15]

    def _trial_cost(self, P):
        snap = self._snapshot()
        before = set(self.R.edges)
        cross0, fail0 = self.R.crossings, len(self.failed)
        try:
            self.commit(P)
            self.set_bounds()
            for lp, n in P.part.pins.items():
                if n not in self.M.fixed:
                    self.route_net(n)
            new = [k for k in self.R.edges if k not in before]
            runs = defaultdict(list)
            for i, j, ax in new:
                runs[(ax, j if ax == 0 else i)].append(i if ax == 0 else j)
            nruns = sum(len(_runs(sorted(v))) for v in runs.values())
            cost = len(new) + 5 * nruns + 60 * (self.R.crossings - cross0) + 400 * (len(self.failed) - fail0)
        finally:
            self._restore(snap)
        return cost

    @staticmethod
    def _coll(PX, PY, s, B):
        x0 = PX[:, None] + s[0]
        y0 = PY[:, None] + s[1]
        x1 = PX[:, None] + s[2]
        y1 = PY[:, None] + s[3]
        return ((x0 < B[None, :, 2]) & (B[None, :, 0] < x1) & (y0 < B[None, :, 3]) & (B[None, :, 1] < y1)).any(axis=1)

    def place_leftovers(self, pending):
        M = self.M
        pending = list(pending)
        while pending:
            def score(p):
                anch = sum(1 for n in p.pins.values() if n not in M.fixed and n in self.anchored)
                return (-anch, p.order)
            pending.sort(key=score)
            part = pending.pop(0)
            self.free_spot(part)
            self.set_bounds()
            for lp, n in part.pins.items():
                if n not in M.fixed:
                    self.route_net(n)

    # -- reference-node islands --------------------------------------------------------
    def island_members(self):
        M = self.M
        isl = defaultdict(list)
        for net in M.refs:
            for p, lp in M.nets.get(net, []):
                others = [n for q, n in p.pins.items() if q != lp]
                if p.kind in ("R", "C", "CP") and others and others[0] in M.fixed and others[0] not in M.refs:
                    isl[net].append(p)
        for lst in isl.values():
            for p in lst:
                self.island_parts.add(p.name)
        return isl

    def place_islands(self, isl):
        M = self.M
        if not isl:
            return
        for net, parts in isl.items():
            bb = self.bbox()
            inet = "@" + net
            y = bb[3] + 150
            x0 = bb[0] + 30
            for p in parts:
                q = _Part(p.name, p.ref, p.kind, {k: (inet if n == net else n) for k, n in p.pins.items()},
                          p.lines, p.flags, p.order)
                lp = next(k for k, n in q.pins.items() if n == inet)
                h = _Hang(q, lp)
                far = [n for k, n in q.pins.items() if k != lp][0]
                h.far.append(([k for k in q.pins if k != lp][0], far, []))
                h.dir = -1 if (far in M.rails and M.fixed_up(far)) else 1
                (self.hang_up if h.dir < 0 else self.hang_down)[inet].append(h)
            self.anchored.add(inet)
            right = self.place_row(inet, y, x0, x0)
            P = self.push_right(_Part("@tag" + net, "", "NET", {"p": inet}, [M.label_of(net)]), right + 10, y, ID)
            self.commit(P)
            self.set_bounds()
            self.route_net(inet)

    def bbox(self):
        xs, ys = [], []
        for o, _ in self.obst:
            xs += [o[0], o[2]]
            ys += [o[1], o[3]]
        if not xs:
            return (0, 0, 0, 0)
        return (min(xs), min(ys), max(xs), max(ys))

    # -- tags for nets the router could not finish ---------------------------------------
    def tag_failures(self):
        for net, (p, d) in self.failed:
            self.tags.append((net, p, d))

    # -- full build --------------------------------------------------------------------
    def build(self, spine):
        M = self.M
        nets, steps = spine
        isl = self.island_members()
        self.classify(nets, steps)
        self.place_spine(nets, steps)
        self.align_terms()
        self.rail_buses()
        self.route_all(nets)
        pend = [p for p in M.parts if p.name not in self.placed and p.name not in self.island_parts]
        # unplaced hanger trees (nets never reached) are placed as leftovers part by part
        if M.out_net and not any(P.part.kind == "OUT" for P in self.placed.values()):
            pend.append(_Part("OUT", "OUT", "OUT", {"p": M.out_net}, [], {}, 10 ** 6))
        if M.in_net and M.in_term and not any(P.part.kind == "IN" for P in self.placed.values()):
            pend.append(_Part("IN", "IN", "IN", {"p": M.in_net}, [], {}, -1))
        self.hang_up.clear()
        self.hang_down.clear()
        self.place_leftovers(pend)
        self.place_islands(isl)
        self.tag_failures()

    def build_ladder(self):
        """Last resort: one row of parts, every net routed."""
        M = self.M
        x = 0
        y = 0
        isl = self.island_members()
        parts = [p for p in M.parts if p.name not in self.island_parts]
        if M.in_net and M.in_term:
            parts.insert(0, _Part("IN", "IN", "IN", {"p": M.in_net}, []))
        if M.out_net:
            parts.append(_Part("OUT", "OUT", "OUT", {"p": M.out_net}, []))
        for i, p in enumerate(parts):
            m, pm = self._orient_candidates(p)[0]
            P = self.push_right(p, x, y + (i % 2) * 140, m, pm)
            self.commit(P)
            x = P.box[2] + 60
        self.set_bounds()
        for n in list(self.netpins):
            if n not in M.fixed:
                self.route_net(n)
        self.place_islands(isl)
        self.tag_failures()


def part_pin(kind, lp):
    if kind == "X":
        return {"inp": "inA", "inn": "inB"}.get(lp, lp)
    return lp


# =========================================================================== result

class Layout:
    """A finished schematic layout: placed parts, labels, terminals and routed wires."""

    def __init__(self, E: _Engine, strategy=""):
        self.title = E.M.title
        self.strategy = strategy
        self._E = E
        self.placed = list(E.placed.values())
        self.terms = list(E.terms)
        self.crossings = E.R.crossings
        self.failed = [n for n, _ in E.failed]
        # wires per net: merge grid edges into straight runs
        by_net = defaultdict(list)
        for (i, j, ax), net in E.R.edges.items():
            if net.startswith("#"):
                continue
            by_net[net].append((i, j, ax))
        self.wires = {}
        for net, edges in by_net.items():
            segs = []
            h = defaultdict(list)
            v = defaultdict(list)
            for i, j, ax in edges:
                (h if ax == 0 else v)[j if ax == 0 else i].append(i if ax == 0 else j)
            for j, lst in h.items():
                for a, b in _runs(sorted(lst)):
                    segs.append(((a * G, j * G), ((b + 1) * G, j * G)))
            for i, lst in v.items():
                for a, b in _runs(sorted(lst)):
                    segs.append(((i * G, a * G), (i * G, (b + 1) * G)))
            self.wires[net] = segs
        self.buses = list(E.buses)
        # junction dots: grid points where three or more wire ends meet on one net
        self.dots = []
        pinpts = defaultdict(set)
        for net, pins in E.netpins.items():
            for p, _ in pins:
                pinpts[net].add(p)
        for net in by_net:
            cnt = defaultdict(int)
            for i, j, ax in by_net[net]:
                cnt[(i, j)] += 1
                cnt[(i + 1, j) if ax == 0 else (i, j + 1)] += 1
            for p, c in cnt.items():
                c2 = c + (1 if p in pinpts[net] else 0)
                if c2 >= 3:
                    self.dots.append((p[0] * G, p[1] * G))
        self.dots += list(E.bus_dots)
        self.tags = list(E.tags)

    # -- inspection helpers (tests) ---------------------------------------------------
    @property
    def parts(self):
        return [{"name": P.name, "kind": P.part.kind, "x": P.x, "y": P.y, "box": P.core, "full": P.box}
                for P in self.placed]

    @property
    def labels(self):
        out = []
        for P in self.placed:
            for lb in P.labels:
                out.append({"owner": P.name, "text": " ".join(lb.lines), "box": lb.box})
        for t in self.terms:
            if t.lbox:
                out.append({"owner": t.owner, "text": t.label, "box": t.lbox})
        return out

    def boxes(self):
        """(kind, owner, box) for every symbol body, label and terminal."""
        items = []
        for P in self.placed:
            items.append(("body", P.name, P.core))
            for lb in P.labels:
                items.append(("label", P.name, lb.box))
        for t in self.terms:
            if t.kind != "bar":
                items.append(("term", t.owner, t.box))
        return items

    def overlaps(self):
        """List of human-readable collisions (empty when the drawing is clean)."""
        items = self.boxes()
        out = []
        for a in range(len(items)):
            ka, oa, ba = items[a]
            for b in range(a + 1, len(items)):
                kb, ob, bb = items[b]
                if oa == ob and not (ka == "label" and kb == "label"):
                    continue
                if oa == ob and ka == kb == "label":
                    if _hit(_infl(ba, -0.5), bb):
                        out.append(f"{ka} {oa} overlaps {kb} {ob}")
                    continue
                if _hit(_infl(ba, -0.5), bb):
                    out.append(f"{ka} {oa} overlaps {kb} {ob}")
        segs = []
        for net, ss in self.wires.items():
            for a, b in ss:
                segs.append((net, a, b))
        for net, pts in self.buses:
            segs.append(("bus:" + net, pts[0], pts[1]))
        for t in self.terms:
            for a, b in zip(t.stub, t.stub[1:]):
                segs.append(("stub:" + t.owner, a, b))
        for net, a, b in segs:
            for k, o, box in items:
                if net == "stub:" + o and k == "term":
                    continue
                if k == "term" and net.startswith("bus:"):
                    continue
                if _seg_hits_box(a, b, box):
                    out.append(f"wire {net} crosses {k} {o}")
        return out

    def score(self):
        bends = 0
        length = 0
        for ss in self.wires.values():
            for a, b in ss:
                length += abs(a[0] - b[0]) + abs(a[1] - b[1])
            bends += len(ss)
        bb = self.bbox()
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        return (1000 * len(self.overlaps()) + 300 * len(self.failed) + 40 * self.crossings
                + 3 * bends + 0.02 * length + 0.0004 * area)

    def bbox(self):
        xs, ys = [], []
        for _, _, b in self.boxes():
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
        for P in self.placed:
            xs += [P.box[0], P.box[2]]
            ys += [P.box[1], P.box[3]]
        for ss in self.wires.values():
            for a, b in ss:
                xs += [a[0], b[0]]
                ys += [a[1], b[1]]
        for net, p, d in self.tags:
            xs.append(p[0] * G + 60)
            ys.append(p[1] * G)
        if not xs:
            return (0, 0, 10, 10)
        return (min(xs), min(ys), max(xs), max(ys))


def _runs(lst):
    out = []
    start = prev = None
    for v in lst:
        if start is None:
            start = prev = v
        elif v == prev + 1:
            prev = v
        elif v == prev:
            continue
        else:
            out.append((start, prev))
            start = prev = v
    if start is not None:
        out.append((start, prev))
    return out


# =========================================================================== public API

_CACHE: dict = {}


def _cache_key(circuit, node_voltages):
    try:
        vals = tuple((e.name, e.value, tuple(e.nodes), e.model) for e in circuit.elements)
        return (circuit.source, circuit.title, vals, repr(sorted((node_voltages or {}).items())))
    except Exception:
        return None


def layout(circuit: Circuit, node_voltages=None) -> Layout:
    """Lay the circuit out; tries a few strategies and keeps the cleanest (cached per netlist)."""
    key = _cache_key(circuit, node_voltages)
    if key is not None and key in _CACHE:
        return _CACHE[key]
    L = _layout(circuit, node_voltages)
    if key is not None:
        if len(_CACHE) > 32:
            _CACHE.clear()
        _CACHE[key] = L
    return L


def _layout(circuit, node_voltages=None):
    M = _Model(circuit, node_voltages)
    try:
        spines = _spines(M)
    except Exception:
        spines = [([M.in_net] if M.in_net else [], [])]
    strategies = []
    for i, sp in enumerate(spines[:2]):
        strategies.append((f"spine{i}", sp, {}))
        if i == 0:
            strategies.append(("spine0-above", sp, {"vbias": -1}))
            strategies.append(("spine0-below", sp, {"vbias": 1}))
            amps = [st[0].name for st in sp[1] if st[0].kind == "X"]
            for nm in amps:
                strategies.append((f"spine0-flip-{nm}", sp, {"flip": {nm}}))
    best = None
    for name, sp, opts in strategies:
        try:
            E = _Engine(M, opts)
            E.build(sp)
            L = Layout(E, name)
            s = L.score() + (150 if name.startswith("spine1") else 0)
        except Exception:
            continue
        if best is None or s < best[0]:
            best = (s, L)
        if s < 1000 and not L.failed and L.crossings == 0 and name == "spine0" and len(spines) == 1:
            break
    if best is None:
        try:
            E = _Engine(M, {})
            E.build_ladder()
            return Layout(E, "ladder")
        except Exception:
            E = _Engine(M, {})
            return Layout(E, "empty")
    return best[1]


def svg(circuit: Circuit, title: bool = True, node_voltages=None) -> str:
    """Render the circuit as a themeable SVG schematic."""
    name = getattr(circuit, "title", None) or "Untitled circuit"
    try:
        L = layout(circuit, node_voltages)
        return _render(L, name, title)
    except Exception:
        try:
            E = _Engine(_Model(circuit, node_voltages), {})
            E.build_ladder()
            return _render(Layout(E, "ladder"), name, title)
        except Exception:
            return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 60" width="320" height="60" '
                    f'role="img" style="color: var(--hotwire-ink, #1d1d1f)"><title>{escape(name)} (schematic)</title>'
                    f'<text x="10" y="34" fill="currentColor" font-family="{SANS}" font-size="13">'
                    f'schematic unavailable for this netlist</text></svg>')


# =========================================================================== rendering

def _arrow(tip, d, length=6.0, half=3.0):
    ux, uy = d
    n = math.hypot(ux, uy) or 1.0
    ux, uy = ux / n, uy / n
    bx, by = tip[0] - ux * length, tip[1] - uy * length
    return (f"M{tip[0]:.2f} {tip[1]:.2f}L{bx - uy * half:.2f} {by + ux * half:.2f}"
            f"L{bx + uy * half:.2f} {by - ux * half:.2f}Z")


def _shape(part):
    """Local-coordinate drawing: (strokes, thick strokes, fills, circles, accent strokes, accent fills, upright marks)."""
    k = part.kind
    s, th, f, circ, acs, acf, marks = [], [], [], [], [], [], []
    if k == "R":
        s.append("M-30 0H-18L-15 -5L-9 5L-3 -5L3 5L9 -5L15 5L18 0H30")
    elif k == "C":
        s.append("M-30 0H-4M4 0H30")
        th.append("M-4 -10V10M4 -10V10")
    elif k == "CP":
        s.append("M-30 0H-4M5 0H30M-14 -8H-8M-11 -11V-5")
        th.append("M-4 -10V10M8 -10Q2 0 8 10")
    elif k == "L":
        s.append("M-30 0H-20" + "a5 5 0 0 1 10 0" * 4 + "H30")
    elif k in ("D", "SCH", "LED"):
        s.append("M-30 0H-7M7 0H30")
        f.append("M-7 -8L7 0L-7 8Z")
        if k == "SCH":
            s.append("M10 -5V-8H7V8H4V5")
        else:
            s.append("M7 -8V8")
        if k == "LED":
            s.append("M-1 -11L4.5 -16.5M5 -11L10.5 -16.5")
            f.append(_arrow((7, -19), (1, -1), 5, 2.4))
            f.append(_arrow((13, -19), (1, -1), 5, 2.4))
    elif k == "S":
        circ += [(-14, 0, 2.5), (14, 0, 2.5)]
        s.append("M-30 0H-16.5M16.5 0H30")
        s.append("M-12 -1.5L13 -4.5" if part.flags.get("on") else "M-12 -1.5L12 -12")
    elif k == "V":
        circ.append((0, 0, 13))
        s.append("M-30 0H-13M13 0H30")
        if part.flags.get("ac"):
            marks.append((0, 0, "~"))
        else:
            marks.append((-6, 0, "+"))
            marks.append((6, 0, "-"))
    elif k == "I":
        circ.append((0, 0, 13))
        s.append("M-30 0H-13M13 0H30M-7 0H3")
        f.append(_arrow((8, 0), (1, 0), 6, 3))
    elif k == "Q":
        s.append("M-30 0H-6M-6 -5L10 -15V-30M-6 5L10 15V30")
        th.append("M-6 -11V11")
        circ.append((2, 0, 17))
        if part.flags.get("pnp"):
            f.append(_arrow((-6 + 0.25 * 16, 5 + 0.25 * 10), (-16, -10), 6, 3))
        else:
            f.append(_arrow((-6 + 0.85 * 16, 5 + 0.85 * 10), (16, 10), 6, 3))
    elif k == "J" and part.flags.get("mos"):
        # enhancement MOSFET: insulated gate plate, three-segment channel, body line to the
        # source; the arrow points in for N-channel, out for P-channel
        s.append("M-30 0H-9M-3 -8H10V-30M-3 8H10V30M-3 0H10V8")
        th.append("M-9 -11V11M-3 -12V-5M-3 -3.5V3.5M-3 5V12")
        circ.append((3, 0, 16))
        if part.flags.get("pch"):
            f.append(_arrow((9, 0), (1, 0), 6, 3))
        else:
            f.append(_arrow((-2, 0), (-1, 0), 6, 3))
    elif k == "J":
        s.append("M-30 0H-4M-4 -8H10V-30M-4 8H10V30")
        th.append("M-4 -12V12")
        circ.append((3, 0, 16))
        if part.flags.get("pch"):
            f.append(_arrow((-17, 0), (-1, 0), 6, 3))
        else:
            f.append(_arrow((-5, 0), (1, 0), 6, 3))
    elif k == "X":
        s.append("M-40 -40L40 0L-40 40ZM-60 -20H-40M-60 20H-40M40 0H60M0 -20V-40M0 20V40")
    elif k == "POT":
        s.append("M-30 0H-18L-15 -5L-9 5L-3 -5L3 5L9 -5L15 5L18 0H30")
        acs.append("M0 -30V-15")
        acf.append(_arrow((0, -7), (0, 1), 8, 3.5))
    elif k in ("E", "G"):
        s.append("M0 -30V-14M0 14V30M0 -14L13 0L0 14L-13 0ZM-30 -10H-20M-30 10H-20")
        circ += [(-18, -10, 2), (-18, 10, 2)]
        if k == "E":
            marks += [(0, -6, "+"), (0, 6, "-")]
        else:
            s.append("M0 -7V3")
            f.append(_arrow((0, 8), (0, 1), 5, 3))
        marks += [(-24, -16, "+"), (-24, 16, "-")]
    return s, th, f, circ, acs, acf, marks


def _mark(x, y, g):
    if g == "+":
        return f"M{x - 3:.1f} {y:.1f}H{x + 3:.1f}M{x:.1f} {y - 3:.1f}V{y + 3:.1f}"
    if g == "-":
        return f"M{x - 3:.1f} {y:.1f}H{x + 3:.1f}"
    return f"M{x - 6:.1f} {y:.1f}C{x - 3:.1f} {y - 7:.1f} {x:.1f} {y - 7:.1f} {x:.1f} {y:.1f}S{x + 3:.1f} {y + 7:.1f} {x + 6:.1f} {y:.1f}"


def _n(v):
    return f"{v:.1f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _text(lb, cls=""):
    out = []
    for i, line in enumerate(lb.lines):
        y = lb.y + (i + 1) * LH - 3.5
        out.append(f'<text x="{_n(float(lb.x))}" y="{_n(float(y))}" text-anchor="{lb.anchor}">{escape(line)}</text>')
    return "".join(out)


def _render(L: Layout, title_text: str, show_title: bool) -> str:
    bb = L.bbox()
    pad = 18
    top_extra = 34 if show_title else 0
    x0, y0 = bb[0] - pad, bb[1] - pad - top_extra
    w, h = (bb[2] - bb[0]) + 2 * pad, (bb[3] - bb[1]) + 2 * pad + top_extra
    if show_title:
        w = max(w, _text_w(title_text, 14) + 2 * pad)
    x0, y0, w, h = math.floor(x0), math.floor(y0), math.ceil(w), math.ceil(h)
    ink, fills, texts, acc = [], [], [], []
    # wires
    d = []
    for net, segs in L.wires.items():
        for a, b in segs:
            d.append(f"M{_n(float(a[0]))} {_n(float(a[1]))}L{_n(float(b[0]))} {_n(float(b[1]))}")
    for net, pts in L.buses:
        d.append(f"M{_n(float(pts[0][0]))} {_n(float(pts[0][1]))}L{_n(float(pts[1][0]))} {_n(float(pts[1][1]))}")
    for t in L.terms:
        d.append("M" + "L".join(f"{_n(float(x))} {_n(float(y))}" for x, y in t.stub))
    if d:
        ink.append(f'<path d="{"".join(d)}"/>')
    # symbols
    for P in L.placed:
        k = P.part.kind
        m = P.m
        tf = f'matrix({m[0]} {m[2]} {m[1]} {m[3]} {_n(float(P.x))} {_n(float(P.y))})'
        if k in ("IN", "OUT", "NET"):
            px, py, dx, dy = P.pin("p")
            cx = P.x
            tgt = acc if k != "NET" else ink
            if k == "NET":
                tgt.append(f'<path d="M{_n(float(px))} {_n(float(py))}H{_n(float(cx - 5))}L{_n(float(cx))} {_n(float(py - 5))}'
                           f'H{_n(float(cx + _text_w(P.part.lines[0]) + 8))}V{_n(float(py + 5))}H{_n(float(cx))}Z"/>')
                texts.append(f'<text x="{_n(float(cx + 4))}" y="{_n(float(py + 3.5))}" font-size="10">{escape(P.part.lines[0])}</text>')
            else:
                tgt.append(f'<circle cx="{_n(float(cx))}" cy="{_n(float(py))}" r="4" fill="none" stroke="currentColor"/>')
                tgt.append(f'<path d="M{_n(float(cx + 4 * (1 if k == "IN" else -1)))} {_n(float(py))}L{_n(float(px))} {_n(float(py))}" stroke="currentColor"/>')
                if k == "IN":
                    acc.append(f'<text x="{_n(float(cx - 9))}" y="{_n(float(py + 4))}" text-anchor="end" stroke="none" fill="currentColor" font-family="{MONO}" font-size="11" font-weight="600">IN</text>')
                else:
                    acc.append(f'<text x="{_n(float(cx + 9))}" y="{_n(float(py + 4))}" stroke="none" fill="currentColor" font-family="{MONO}" font-size="11" font-weight="600">OUT</text>')
            continue
        s, th, f, circ, acs, acf, marks = _shape(P.part)
        body = "".join(f'<path d="{p}"/>' for p in s)
        body += "".join(f'<path d="{p}" stroke-width="2.2"/>' for p in th)
        body += "".join(f'<circle cx="{c[0]}" cy="{c[1]}" r="{c[2]}"/>' for c in circ)
        if k == "X":
            pm = P.pinmap
            plus_top = pm.get("inp", "inA") == "inA"
            yp, yn = (-20, 20) if plus_top else (20, -20)
            body += f'<path d="M-36 {yp}H-29M-32.5 {yp - 3.5}V{yp + 3.5}M-36 {yn}H-29"/>'
        ink.append(f'<g transform="{tf}">{body}</g>')
        if f:
            fills.append(f'<g transform="{tf}">' + "".join(f'<path d="{p}"/>' for p in f) + "</g>")
        if acs or acf:
            acc.append(f'<g transform="{tf}">' + "".join(f'<path d="{p}" fill="none" stroke="currentColor"/>' for p in acs)
                       + "".join(f'<path d="{p}" fill="currentColor" stroke="none"/>' for p in acf) + "</g>")
        if marks:
            md = []
            for lx, ly, g in marks:
                X, Y = _xf(m, lx, ly)
                md.append(_mark(P.x + X, P.y + Y, g))
            ink.append(f'<path d="{"".join(md)}" stroke-width="1.2"/>')
        if k == "X":
            ref, model = P.part.lines[0], P.part.lines[1]
            fs = 9.5 if max(len(ref), len(model)) <= 6 else 8
            texts.append(f'<text x="{_n(float(P.x - 8))}" y="{_n(float(P.y - 2))}" text-anchor="middle" font-size="{fs}">{escape(ref)}</text>')
            texts.append(f'<text x="{_n(float(P.x - 8))}" y="{_n(float(P.y + 10))}" text-anchor="middle" font-size="{fs}">{escape(model)}</text>')
        for lb in P.labels:
            texts.append(_text(lb))
    # terminals
    for t in L.terms:
        x, y = t.x, t.y
        if t.kind == "gnd":
            ink.append(f'<path d="M{x - 8} {y}H{x + 8}M{x - 5} {y + 4}H{x + 5}M{x - 2} {y + 8}H{x + 2}"/>')
        elif t.kind == "gndup":
            ink.append(f'<path d="M{x - 8} {y}H{x + 8}M{x - 5} {y - 4}H{x + 5}M{x - 2} {y - 8}H{x + 2}"/>')
        elif t.kind in ("railup", "raildn"):
            ink.append(f'<path d="M{x - 7} {y}H{x + 7}" stroke-width="2"/>')
            ty = y - 6.5 if t.kind == "railup" else y + 4 + LH - 3.5
            texts.append(f'<text x="{_n(float(x))}" y="{_n(float(ty))}" text-anchor="middle">{escape(t.label)}</text>')
    # tags for anything the router could not reach
    for net, p, dd in L.tags:
        px, py = p[0] * G, p[1] * G
        dx, dy = _DIRS[dd]
        ex, ey = px + dx * 14, py + dy * 14
        ink.append(f'<path d="M{px} {py}L{ex} {ey}"/>')
        anchor = "start" if dx >= 0 else "end"
        texts.append(f'<text x="{ex + (3 if dx >= 0 else -3)}" y="{ey + (4 if dy == 0 else (12 if dy > 0 else -4))}" '
                     f'text-anchor="{anchor}" font-size="10">{escape(net.upper())}</text>')
    for x, y in L.dots:
        fills.append(f'<circle cx="{x}" cy="{y}" r="2.6"/>')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {w} {h}" width="{w}" height="{h}" '
           f'role="img" aria-label="{escape(title_text)} schematic" class="hotwire-schematic" '
           f'style="color: var(--hotwire-ink, #1d1d1f)">',
           f"<title>{escape(title_text)} (schematic)</title>",
           "<style>@media (prefers-color-scheme: dark){.hw-ink{color:var(--hotwire-ink,#e8e8ed)}}</style>",
           '<g class="hw-ink">']
    if show_title:
        out.append(f'<text x="{bb[0]}" y="{y0 + 22}" font-family="{SANS}" font-size="14" font-weight="600" '
                   f'fill="currentColor">{escape(title_text)}</text>')
    out.append('<g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">')
    out += ink
    out.append("</g>")
    out.append('<g fill="currentColor" stroke="none">')
    out += fills
    out.append("</g>")
    out.append(f'<g fill="currentColor" font-family="{MONO}" font-size="{FS}" style="font-variant-numeric: tabular-nums">')
    out += texts
    out.append("</g>")
    out.append('<g class="hw-acc" style="color: var(--hotwire-accent, #c2410c)" stroke-width="1.5" '
               'stroke-linecap="round" stroke-linejoin="round">')
    out += acc
    out.append("</g></g></svg>")
    return "".join(out)
