"""A guitar in front of the circuit: pickup, volume pot and cable.

Effects pedals don't see an ideal voltage source. A magnetic pickup is an
inductor with resistance and capacitance, the guitar's volume pot sits between
it and the jack, and the cable adds capacitance. That source impedance is why a
low-input-impedance fuzz "cleans up" when you roll the guitar volume back, and
why a treble booster sounds different on a humbucker. ``with_guitar`` inserts
that chain between the circuit's ``*@ in`` source and the node it drove.

Defaults are a typical single-coil passive pickup: 2.5 H, 6.5 kΩ, 120 pF,
a 500 kΩ audio-taper volume pot and 3 m of cable (about 300 pF).
"""
from __future__ import annotations

import copy

from .netlist import Circuit, Element, NetlistError, pot_fraction

__all__ = ["with_guitar", "PICKUPS"]

PICKUPS = {
    "single_coil": {"L": 2.5, "R": 6.5e3, "C": 120e-12},
    "humbucker": {"L": 4.5, "R": 8.5e3, "C": 150e-12},
    "p90": {"L": 7.0, "R": 8.0e3, "C": 130e-12},
}


def with_guitar(c: Circuit, volume: float = 1.0, pickup: str = "single_coil",
                pot_ohms: float = 500e3, cable_f: float = 300e-12) -> Circuit:
    """A copy of ``c`` driven through a guitar pickup + volume pot + cable."""
    if not c.input:
        raise NetlistError(0, "the guitar model needs '*@ in <source>'")
    p = PICKUPS.get(pickup, PICKUPS["single_coil"])
    g = copy.deepcopy(c)
    src = g.element(g.input)
    if src.kind != "V":
        raise NetlistError(0, "the guitar model needs a voltage source as the input")
    if abs(float(src.value or 0.0)) > 1e-9:
        raise NetlistError(0, f"{src.name} carries a {src.value:g} V DC bias, but a guitar doesn't: put the bias in "
                              "the circuit (a divider after the input cap) and give the source DC 0")
    if any(n.startswith("gtr_") for n in g.nodes()):
        raise NetlistError(0, "node names starting with 'gtr_' are reserved for the guitar model; rename them")
    hot, ref = src.nodes
    if hot == "0":
        hot, ref = ref, hot
    src.nodes = ["gtr_emf", ref]
    frac = pot_fraction(min(1.0, max(0.0, float(volume))), "log")
    r_wiper_gnd = max(1.0, pot_ohms * frac)
    r_hot_wiper = max(1.0, pot_ohms * (1 - frac))
    add = [
        Element("LGTR_PU", "L", ["gtr_emf", "gtr_l"], p["L"], line=src.line),
        Element("RGTR_PU", "R", ["gtr_l", "gtr_hot"], p["R"], line=src.line),
        Element("CGTR_PU", "C", ["gtr_hot", ref], p["C"], line=src.line),
        Element("RGTR_VOLA", "R", ["gtr_hot", hot], r_hot_wiper, line=src.line),
        Element("RGTR_VOLB", "R", [hot, ref], r_wiper_gnd, line=src.line),
        Element("CGTR_CABLE", "C", [hot, ref], cable_f, line=src.line),
    ]
    idx = g.elements.index(src) + 1
    g.elements[idx:idx] = add
    g.title = f"{c.title} (guitar volume {volume:.2f})"
    return g
