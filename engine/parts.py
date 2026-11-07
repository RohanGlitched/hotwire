"""Preferred values (E-series) and a bill of materials a builder can order from."""
from __future__ import annotations

import math
from collections import OrderedDict

from .netlist import Circuit, format_value

__all__ = ["SERIES", "snap", "neighbours", "bom", "bom_markdown"]

SERIES = {
    "E6": [1.0, 1.5, 2.2, 3.3, 4.7, 6.8],
    "E12": [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
    "E24": [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3,
            4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1],
}


def _e_series(n: int) -> list:
    return [round(10 ** (i / n), 2 if n <= 48 else 2) for i in range(n)]


SERIES["E48"] = [1.00, 1.05, 1.10, 1.15, 1.21, 1.27, 1.33, 1.40, 1.47, 1.54, 1.62, 1.69, 1.78, 1.87,
                 1.96, 2.05, 2.15, 2.26, 2.37, 2.49, 2.61, 2.74, 2.87, 3.01, 3.16, 3.32, 3.48, 3.65,
                 3.83, 4.02, 4.22, 4.42, 4.64, 4.87, 5.11, 5.36, 5.62, 5.90, 6.19, 6.49, 6.81, 7.15,
                 7.50, 7.87, 8.25, 8.66, 9.09, 9.53]
SERIES["E96"] = [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37,
                 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91,
                 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67,
                 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74,
                 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23,
                 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
                 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76]


def _ladder(series: str):
    base = SERIES[series]
    for dec in range(-13, 8):
        for b in base:
            yield b * 10 ** dec


def snap(value: float, series: str = "E24") -> float:
    """Nearest preferred value (in log space)."""
    if value <= 0:
        return value
    exp = math.floor(math.log10(value))
    best, err = value, float("inf")
    for d in (exp - 1, exp, exp + 1):
        for b in SERIES[series]:
            cand = b * 10 ** d
            e = abs(math.log10(cand / value))
            if e < err:
                best, err = cand, e
    return float(f"{best:.3g}")


def neighbours(value: float, series: str = "E24", steps: int = 6) -> list:
    """Preferred values around ``value``: ``steps`` below and above, sorted."""
    centre = snap(value, series)
    base = SERIES[series]
    n = len(base)
    exp = math.floor(math.log10(centre) + 1e-9)
    mant = centre / 10 ** exp
    idx = min(range(n), key=lambda i: abs(base[i] - mant))
    out = []
    for k in range(-steps, steps + 1):
        j = idx + k
        d, j = exp + j // n, j % n
        out.append(float(f"{base[j] * 10 ** d:.3g}"))
    return out


def _cap_kind(value: float) -> tuple:
    if value < 1e-9:
        return "ceramic C0G/NP0", "2.5 mm radial"
    if value < 1e-6:
        return "film (box)", "5 mm radial"
    return "electrolytic, 16 V or more (25 V is common)", "radial can, 5 mm"


POT_VALUES = [500, 1e3, 2e3, 5e3, 10e3, 20e3, 25e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6]


def snap_pot(value: float) -> float:
    """Nearest value pots are actually sold in (A500K, B100K, C25K…)."""
    return min(POT_VALUES, key=lambda v: abs(math.log(v / value)))


def _pot_code(taper: str) -> str:
    return {"log": "A", "lin": "B", "revlog": "C"}.get(taper, "B")


def bom(c: Circuit, series: str = "E24") -> list:
    series = str(series or "E24").upper()
    if series not in SERIES:
        from .netlist import NetlistError
        raise NetlistError(0, f"series must be one of {', '.join(SERIES)}")
    """Grouped bill of materials. Each row: refs, qty, kind, value, snapped, part, package, note."""
    try:
        from .parts_data import PARTS
    except ImportError:
        PARTS = {}
    groups: "OrderedDict[tuple, dict]" = OrderedDict()

    def add(key, ref, row):
        g = groups.get(key)
        if g:
            g["refs"].append(ref)
            g["qty"] += 1
        else:
            row = dict(row)
            row["refs"] = [ref]
            row["qty"] = 1
            groups[key] = row

    precise = float(c.tol.get("R", 0.05)) <= 0.011 or series in ("E48", "E96")
    r_grade = "metal film 1% (5-band code)" if precise else "carbon or metal film 5% (4-band code)"
    for e in c.elements:
        if e.pot or e.name == c.input or e.name in c.supplies:
            continue
        k = e.kind
        if k == "R":
            s = snap(e.value, series)
            nonstd = abs(s - e.value) / e.value > 0.005
            add(("R", s), e.name, {
                "kind": "Resistor", "value": format_value(e.value, "Ω"), "snapped": format_value(s, "Ω"),
                "part": f"{format_value(s, 'Ω')} {r_grade}", "package": "1/4 W axial, 0.4 in lead spacing",
                "note": f"{format_value(e.value, 'Ω')} isn't a standard value; {format_value(s, 'Ω')} is the nearest {series}" if nonstd else "",
            })
        elif k == "C":
            s = snap(e.value, "E12" if e.value < 1e-6 else "E6")
            kind, pkg = _cap_kind(e.value)
            rating = c.ratings.get(e.name)
            if rating and rating[1] == "V":
                kind = kind.split(",")[0] + f", {rating[0]:g} V"
            nonstd = abs(s - e.value) / e.value > 0.005
            add(("C", s, kind), e.name, {
                "kind": "Capacitor", "value": format_value(e.value, "F"), "snapped": format_value(s, "F"),
                "part": f"{format_value(s, 'F')} {kind}", "package": pkg,
                "note": ("polarised: circuit_check and the build steps say which lead is +" if e.value >= 1e-6 else "")
                        + (f"; nearest standard value {format_value(s, 'F')}" if nonstd else ""),
            })
        elif k == "L":
            add(("L", e.value), e.name, {
                "kind": "Inductor", "value": format_value(e.value, "H"), "snapped": format_value(e.value, "H"),
                "part": f"{format_value(e.value, 'H')} inductor", "package": "radial / toroid",
                "note": "large audio inductors (100 mH+) are sold as wah or filter inductors",
            })
        elif k == "S":
            add(("S",), e.name, {
                "kind": "Switch", "value": "SPST", "snapped": "SPST",
                "part": "SPST toggle (or one pole of a DPDT mini toggle)", "package": "panel",
                "note": "simulated in the position written in the netlist",
            })
        elif k in ("D", "Q", "J", "X", "M"):
            base = (e.model or "").split("__")[0]
            info = PARTS.get(base, {})
            kind = {"D": "Diode", "Q": "Transistor", "J": "JFET", "X": "Op-amp", "M": "MOSFET"}[k]
            if base.startswith("LED"):
                kind = "LED"
            pins = info.get("pins")
            note = (f"pinout {pins}" if pins and k in ("Q", "J", "M") else "")
            if k == "M" and "TO-220" in str(info.get("package", "")):
                note += "; bolt to a heatsink (insulating pad) above about 1 W"
            if k == "M" and info:
                note += "; static-sensitive: handle by the case"
            add((k, base), e.name, {
                "kind": kind, "value": base, "snapped": base,
                "part": info.get("desc", base), "package": info.get("package", ""),
                "note": note + ("custom .model: pick a real part with these parameters" if not info else ""),
            })
    rows = list(groups.values())
    # op-amp packages hold two amps: count packages, not amplifiers
    for r in rows:
        if r["kind"] == "Op-amp":
            info = PARTS.get(r["value"], {})
            per = 2 if any(t in (info.get("desc", "") + info.get("package", "")).lower() for t in ("dual",)) else 1
            if per == 2:
                pk = math.ceil(r["qty"] / 2)
                r["note"] = (r["note"] + "; " if r["note"] else "") + f"{r['qty']} amplifiers = {pk} dual package(s)"
                r["qty"] = pk
    for p in c.pots.values():
        rows.append({"refs": [p.name], "qty": 1, "kind": "Potentiometer", "value": format_value(p.value, "Ω"),
                     "snapped": format_value(snap_pot(p.value), "Ω"),
                     "part": f"{_pot_code(p.taper)}{format_value(snap_pot(p.value), '').upper()} ({p.taper}) 16 mm pot",
                     "package": "16 mm, PCB or solder lugs", "note": f"knob: {p.label}"})
    # off-board hardware a pedal / audio build needs
    ics = sum(r["qty"] for r in rows if r["kind"] == "Op-amp")
    if ics:
        rows.append({"refs": ["-"], "qty": ics, "kind": "Hardware", "value": "DIP-8 socket", "snapped": "",
                     "part": "8-pin DIP IC socket", "package": "DIP-8", "note": "so the op-amp can be swapped"})
    if c.input:
        rows.append({"refs": ["J1"], "qty": 1, "kind": "Hardware", "value": "input jack", "snapped": "",
                     "part": "6.35 mm (1/4\") mono jack", "package": "panel", "note": "IN"})
    if c.output:
        rows.append({"refs": ["J2"], "qty": 1, "kind": "Hardware", "value": "output jack", "snapped": "",
                     "part": "6.35 mm (1/4\") mono jack", "package": "panel", "note": "OUT"})
    pedal = False
    for s in c.supplies:
        e = c.element(s)
        if e.kind == "V" and e.value and abs(abs(e.value) - 9) < 0.6:
            pedal = bool(c.input and c.output)
            rail = e.value if e.nodes[1] == "0" else -e.value
            if rail < 0:
                rows.append({"refs": ["J3"], "qty": 1, "kind": "Hardware", "value": "9 V supply (positive ground)",
                             "snapped": "", "part": "9 V battery snap, or an isolated supply output", "package": "",
                             "note": "positive-ground circuit: never share a daisy-chain supply with negative-ground "
                                     "pedals; a standard centre-negative jack would short the supply"})
            else:
                rows.append({"refs": ["J3"], "qty": 1, "kind": "Hardware", "value": "9 V DC jack", "snapped": "",
                             "part": "2.1 mm DC jack, centre negative", "package": "panel",
                             "note": "pedal-standard polarity; add a 1N5817 in series if the circuit has no reverse-polarity protection"})
            break
    if pedal:
        rows += [
            {"refs": ["SW1"], "qty": 1, "kind": "Hardware", "value": "3PDT footswitch", "snapped": "",
             "part": "3PDT latching footswitch", "package": "panel",
             "note": "true bypass: two poles switch in/out, the third drives the LED"},
            {"refs": ["LED1", "R_LED"], "qty": 1, "kind": "Hardware", "value": "status LED", "snapped": "",
             "part": "3 mm or 5 mm LED with a 4.7 kΩ series resistor and a bezel", "package": "panel",
             "note": "about 1.5 mA from 9 V"},
            {"refs": ["-"], "qty": 1, "kind": "Hardware", "value": "enclosure", "snapped": "",
             "part": "125B (or 1590B) aluminium enclosure", "package": "",
             "note": "pick the size that fits the board, pots and jacks"},
            {"refs": ["-"], "qty": 1, "kind": "Hardware", "value": "battery snap", "snapped": "",
             "part": "9 V battery snap (optional)", "package": "",
             "note": "wire through the input jack's stereo ring so unplugging saves the battery"},
        ]
    return rows


def bom_markdown(rows: list) -> str:
    out = ["| Qty | Ref | Part | Value | Package | Note |", "|---:|---|---|---|---|---|"]
    for r in rows:
        refs = ", ".join(r["refs"])
        out.append(f"| {r['qty']} | {refs} | {r['part']} | {r['snapped'] or r['value']} | {r['package']} | {r['note']} |")
    return "\n".join(out)
