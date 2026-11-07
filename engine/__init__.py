"""Hotwire's analog circuit simulator (pure Python + numpy).

    from engine import parse, op, ac, tran, process_audio
    c = parse(netlist_text)
    op(c).v["out"]                      # DC operating point
    ac(c).gain_db                       # small-signal transfer *@ in -> *@ out
    tran(c, 5e-3, 1e-6).v["out"]        # trapezoidal transient
    process_audio(c, samples, 48000)    # audio through the circuit (AC-coupled output)
"""
from __future__ import annotations

from .analyses import (AcResult, OpResult, TranResult, ac, headroom_at, op, process_audio,
                       sine_response, tran, thd_at)
from .mna import SimulationError
from .netlist import Circuit, Element, Model, NetlistError, Pot, parse

__all__ = [
    "parse", "Circuit", "Element", "Model", "Pot", "NetlistError", "SimulationError",
    "op", "ac", "tran", "process_audio", "sine_response", "thd_at", "headroom_at",
    "OpResult", "AcResult", "TranResult",
]
