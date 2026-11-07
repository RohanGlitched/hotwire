"""Exports: the netlist rewritten with new values, ngspice decks, KiCad netlists and
the compiled JSON the Desktop live-audio engine runs."""
from __future__ import annotations

import math
import re
from collections import defaultdict

from .netlist import Circuit, format_value, parse

__all__ = ["netlist_text", "spice_deck", "kicad_netlist", "compile_json"]


def _fmt_spice(v: float) -> str:
    s = format_value(v)
    return s.replace("M", "meg") if s.endswith("M") else s


def netlist_text(c: Circuit) -> str:
    """The original netlist text with changed values and pot positions written back."""
    lines = c.source.splitlines()
    by_line = defaultdict(list)
    for e in c.elements:
        by_line[e.line].append(e)
    out = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.lower().startswith("*@ pot"):
            toks = stripped.split()
            if len(toks) >= 3:
                p = c.pots.get(toks[2].upper())
                if p:
                    if re.search(r"\bpos=", line):
                        line = re.sub(r"\bpos=[0-9.]+", f"pos={p.pos:.3g}", line)
                    else:
                        line = line.rstrip() + f" pos={p.pos:.3g}"
            out.append(line)
            continue
        # elements flattened from a .subckt share its text: never rewrite those lines
        elems = [e for e in by_line.get(i, []) if not e.pot and not e.params.get("sub")]
        if elems and not stripped.startswith("*"):
            e = elems[0]
            toks = line.split()
            idx = {"R": 3, "C": 3, "L": 3, "E": 5, "G": 5}.get(e.kind)
            if idx is not None and len(toks) > idx and e.value is not None:
                try:
                    old = parse(f"Rx a 0 {toks[idx]}\n").elements[0].value if e.kind in "RCL" else float(toks[idx])
                except Exception:
                    old = None
                if old is None or abs(old - e.value) > 1e-9 * max(abs(e.value), 1e-30):
                    toks[idx] = _fmt_spice(e.value)
                    indent = line[: len(line) - len(line.lstrip())]
                    line = indent + " ".join(toks)
        out.append(line)
    return "\n".join(out) + ("\n" if not out or out[-1] else "")


# --------------------------------------------------------------------------- ngspice

def _n(x: float) -> str:
    return f"{x:.12g}"


def opamp_subckt_lines(name: str, params: dict) -> list:
    """The Hotwire op-amp macromodel as an ngspice subcircuit (pins in+ in- vcc vee out).

    Exactly the equations engine/mna.py and engine/devices.py simulate: RIN across the
    inputs; a VCCS A0/RP into RP || CP (one pole at GBW/A0); the anti-windup clamp, a
    softplus conductance that holds the pole node within mid +/- K*half; the tanh output
    limiter between VEE+VSAT_LO and VCC-VSAT_HI behind ROUT; IQ from VCC to VEE.
    """
    from . import devices as dv

    p = dv.model_params("OPAMP", params)
    a0 = max(p["A0"], 1.0)
    rp = dv.OPAMP_RP
    cp = 1.0 / (2 * math.pi * (max(p["GBW"], 1.0) / a0) * rp)
    hi = f"(v(vcc)-{_n(p['VSAT_HI'])})"
    lo = f"(v(vee)+{_n(p['VSAT_LO'])})"
    mid = f"(({hi}+{lo})/2)"
    hh = f"max(({hi}-{lo})/2,{_n(dv.OPAMP_HALF_MIN)})"
    eps = _n(dv.OPAMP_CLAMP_EPS)
    k = _n(dv.OPAMP_CLAMP_K)

    def softplus(x: str) -> str:
        return f"(({x})>0 ? ({x})+{eps}*ln(1+exp(-({x})/{eps})) : {eps}*ln(1+exp(({x})/{eps})))"

    lines = [f".subckt {name} inp inn vcc vee out"]
    if p["RIN"] > 0 and math.isfinite(p["RIN"]):
        lines.append(f"Rin inp inn {_n(p['RIN'])}")
    lines += [
        f"G1 0 p inp inn {_n(a0 / rp)}",
        f"Rp p 0 {_n(rp)}",
        f"Cp p 0 {_n(cp)}",
        f"Bcl p 0 I={_n(dv.opamp_clamp_g(a0))}*({softplus(f'v(p)-{mid}-{k}*{hh}')}"
        f"-{softplus(f'{mid}-{k}*{hh}-v(p)')})",
        f"Bo o 0 V={mid}+{hh}*tanh((v(p)-{mid})/{hh})",
        f"Ro o out {_n(max(p['ROUT'], 1e-3))}",
    ]
    if p["IQ"]:
        lines.append(f"Iq vcc vee {_n(p['IQ'])}")
    lines.append(".ends")
    return lines


def _opamp_subckt(name: str, p: dict) -> str:
    return "\n".join(opamp_subckt_lines(name, p)) + "\n"


def _deck_name(e) -> str:
    """SPICE reads the kind from the first letter: X1.R1 (from a .subckt) becomes RX1_R1."""
    if "." not in e.name:
        return e.name
    return e.kind + e.name.replace(".", "_")


def _deck_model(name: str) -> str:
    return name.replace(".", "_")


def spice_deck(c: Circuit, analysis: str = "") -> str:
    """A self-contained ngspice/LTspice deck: pots expanded, models and op-amp subcircuits included."""
    from . import devices as dv

    lines = [f"* {c.title}", "* exported by Hotwire"]
    used_models = set()
    mos_models: dict = {}          # (model, W) -> deck model name
    mos_lines: list = []
    for e in c.elements:
        n = " ".join(e.nodes)
        nm = _deck_name(e)
        if e.kind == "M":
            mod = c.models[e.model]
            mp = dv.model_params(mod.type, mod.params)
            for key, dst in (("W", "W"), ("L", "L"), ("M", "MULT")):
                if key in e.params:
                    mp[dst] = float(e.params[key])
            mparams, inst = dv.mos_ngspice_params(mp)
            key = (e.model, inst["W"])
            mname = mos_models.get(key)
            if mname is None:
                mname = e.model if not any(k[0] == e.model for k in mos_models) else f"{e.model}_{len(mos_models)}"
                mname = mname.replace(".", "_")
                mos_models[key] = mname
                body = " ".join(f"{k}={v:.12g}" for k, v in mparams.items())
                mos_lines.append(f".model {mname} {mod.type}(LEVEL=1 {body})")
            extra = " ".join(f"{k}={v:.12g}" for k, v in inst.items())
            lines.append(f"{nm} {n} {mname} {extra}")
        elif e.kind in "RCL":
            extra = f" ic={e.params['ic']:g}" if "ic" in e.params else ""
            lines.append(f"{nm} {n} {_fmt_spice(e.value)}{extra}")
        elif e.kind in "VI":
            parts = [f"{nm} {n} DC {e.value or 0:g}"]
            if "ac" in e.params:
                parts.append(f"AC {e.params['ac']:g}")
            if "sin" in e.params:
                parts.append("SIN(" + " ".join(f"{x:g}" for x in e.params["sin"]) + ")")
            if "pulse" in e.params:
                parts.append("PULSE(" + " ".join(f"{x:g}" for x in e.params["pulse"]) + ")")
            lines.append(" ".join(parts))
        elif e.kind in "DQJ":
            lines.append(f"{nm} {n} {_deck_model(e.model)}")
            used_models.add(e.model)
        elif e.kind == "X":
            lines.append(f"{nm} {n} {_deck_model(e.model)}")
            used_models.add(e.model)
        elif e.kind in "EG":
            lines.append(f"{nm} {n} {e.value:g}")
        elif e.kind == "S":
            lines.append(f"R{nm} {n} {0.1 if e.params.get('on') else 1e9:g}")
    for m in sorted(used_models):
        mod = c.models[m]
        if mod.type == "OPAMP":
            lines.append(_opamp_subckt(_deck_model(m), mod.params).rstrip())
        else:
            params = " ".join(f"{k}={v:g}" for k, v in mod.params.items())
            lines.append(f".model {_deck_model(m)} {mod.type}({params})")
    lines += mos_lines
    if analysis:
        lines.append(analysis)
    lines.append(".end")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- KiCad

_KICAD = {
    "R": ("Device", "R", "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal", ["1", "2"]),
    "C": ("Device", "C", "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm", ["1", "2"]),
    "CP": ("Device", "C_Polarized", "Capacitor_THT:CP_Radial_D5.0mm_P2.00mm", ["1", "2"]),
    "L": ("Device", "L", "Inductor_THT:L_Radial_D10.0mm_P5.00mm", ["1", "2"]),
    "D": ("Device", "D", "Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal", ["2", "1"]),       # A, K
    "LED": ("Device", "LED", "LED_THT:LED_D5.0mm", ["2", "1"]),                           # A, K
    "NPN": ("Device", "Q_NPN_EBC", "Package_TO_SOT_THT:TO-92_Inline", ["3", "2", "1"]),   # C B E
    "PNP": ("Device", "Q_PNP_EBC", "Package_TO_SOT_THT:TO-92_Inline", ["3", "2", "1"]),
    "NJF": ("Device", "Q_NJFET_DSG", "Package_TO_SOT_THT:TO-92_Inline", ["1", "3", "2"]),  # D G S
    "PJF": ("Device", "Q_PJFET_DSG", "Package_TO_SOT_THT:TO-92_Inline", ["1", "3", "2"]),
}
_DUAL_UNITS = [("3", "2", "1"), ("5", "6", "7")]   # (+, -, out) for unit A and B


def kicad_netlist(c: Circuit) -> str:
    comps, pins = [], defaultdict(list)    # node -> [(ref, pin)]
    opamp_slot = {}
    ic_count = 0
    for e in c.elements:
        if e.name == c.input or e.name in c.supplies or e.kind in ("V", "I", "E", "G"):
            for n in e.nodes[:2]:
                pins[n]  # keep the net
            continue
        if e.pot:
            continue
        if e.kind in "RCL":
            key = "CP" if e.kind == "C" and e.value >= 1e-6 else e.kind
            lib, part, fp, pinmap = _KICAD[key]
            comps.append((e.name, format_value(e.value), lib, part, fp))
            for n, p in zip(e.nodes, pinmap):
                pins[n].append((e.name, p))
        elif e.kind in "DQJ":
            mtype = c.models[e.model].type
            key = "LED" if e.model.startswith("LED") else ("D" if mtype == "D" else mtype)
            lib, part, fp, pinmap = _KICAD[key]
            comps.append((e.name, e.model.split("__")[0], lib, part, fp))
            for n, p in zip(e.nodes, pinmap):
                pins[n].append((e.name, p))
        elif e.kind == "M":
            # KiCad's Q_NMOS_xyz / Q_PMOS_xyz symbols number the pins in physical order;
            # the body is tied to the source inside a discrete MOSFET (no bulk pin)
            from .parts_data import PARTS
            base = e.model.split("__")[0]
            info = PARTS.get(base, {})
            order = str(info.get("pins") or "DGS").upper()
            if sorted(order) != ["D", "G", "S"]:
                order = "DGS"
            pol = "P" if c.models[e.model].type == "PMOS" else "N"
            fp = ("Package_TO_SOT_THT:TO-220-3_Vertical" if "TO-220" in str(info.get("package", ""))
                  else "Package_TO_SOT_THT:TO-92_Inline")
            comps.append((e.name, base, "Device", f"Q_{pol}MOS_{order}", fp))
            for n, lead in zip(e.nodes[:3], "DGS"):
                pins[n].append((e.name, str(order.index(lead) + 1)))
        elif e.kind == "X":
            model = e.model.split("__")[0]
            slot = opamp_slot.get(model, 0)
            if slot % 2 == 0:
                ic_count += 1
                ref = f"U{ic_count}"
                comps.append((ref, model, "Amplifier_Operational", model if model != "OPAMP_IDEAL" else "TL072",
                              "Package_DIP:DIP-8_W7.62mm_Socket"))
                pins[e.nodes[2]].append((ref, "8"))
                pins[e.nodes[3]].append((ref, "4"))
            ref = f"U{ic_count}"
            plus, minus, out = _DUAL_UNITS[slot % 2]
            pins[e.nodes[0]].append((ref, plus))
            pins[e.nodes[1]].append((ref, minus))
            pins[e.nodes[4]].append((ref, out))
            opamp_slot[model] = slot + 1
        elif e.kind == "S":
            comps.append((e.name, "SW", "Switch", "SW_SPST", "Button_Switch_THT:SW_PUSH_6mm"))
            pins[e.nodes[0]].append((e.name, "1"))
            pins[e.nodes[1]].append((e.name, "2"))
    for p in c.pots.values():
        comps.append((p.name, format_value(p.value), "Device", "R_Potentiometer",
                      "Potentiometer_THT:Potentiometer_Alpha_RD901F-40-00D_Single_Vertical"))
        pins[p.a].append((p.name, "1"))
        pins[p.w].append((p.name, "2"))
        pins[p.b].append((p.name, "3"))
    out = ['(export (version "E")',
           f'  (design (source "{c.title}") (tool "Hotwire"))',
           "  (components"]
    for ref, val, lib, part, fp in comps:
        out.append(f'    (comp (ref "{ref}") (value "{val}") (footprint "{fp}") '
                   f'(libsource (lib "{lib}") (part "{part}")))')
    out.append("  )")
    out.append("  (nets")
    for code, (node, plist) in enumerate(sorted(pins.items(), key=lambda kv: (kv[0] != "0", kv[0])), 1):
        name = "GND" if node == "0" else f"/{node}"
        nodes = " ".join(f'(node (ref "{r}") (pin "{p}"))' for r, p in plist)
        out.append(f'    (net (code "{code}") (name "{name}") {nodes})')
    out.append("  )")
    out.append(")")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- live engine JSON

def compile_json(c: Circuit, op_result=None, sr_hint: int = 48000) -> dict:
    nodes = c.nodes()
    idx = {n: i + 1 for i, n in enumerate(nodes)}
    idx["0"] = 0
    elems = []
    for e in c.elements:
        d = {"name": e.name, "kind": e.kind, "n": [idx[n] for n in e.nodes]}
        if e.kind in "RCLEG":
            d["value"] = float(e.value)
        elif e.kind in "VI":
            d["dc"] = float(e.value or 0.0)
            if e.name == c.input:
                d["input"] = True
        elif e.kind in "DQJXM":
            m = c.models[e.model]
            d["p"] = {k: float(v) for k, v in m.params.items()}
            if e.kind in "QJM":
                d["type"] = m.type.lower()
            if e.kind == "M":       # instance geometry rides along with the model
                for key, dst in (("W", "W"), ("L", "L"), ("M", "MULT")):
                    if key in e.params:
                        d["p"][dst] = float(e.params[key])
        elif e.kind == "S":
            d["on"] = bool(e.params.get("on"))
        elems.append(d)
    pots = [{"name": p.name, "a": idx[p.a], "w": idx[p.w], "b": idx[p.b], "value": p.value,
             "taper": p.taper, "pos": p.pos, "label": p.label, "halves": [p.name + "_A", p.name + "_B"]}
            for p in c.pots.values()]
    op_v = None
    if op_result is not None:
        op_v = [float(op_result.v.get(n, 0.0)) for n in nodes]
    return {"version": 1, "title": c.title, "nodes": nodes, "input": c.input,
            "output": idx.get(c.output) if c.output else None, "sr_hint": sr_hint,
            "elements": elems, "pots": pots, "op": op_v}
