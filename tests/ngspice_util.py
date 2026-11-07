"""Run ngspice-42 (installed in WSL) on a Hotwire circuit and read back the results.

The deck is generated from the parsed :class:`engine.netlist.Circuit`, so
ngspice sees exactly the parameters the Hotwire engine uses:

* every model referenced by the circuit is written as an explicit ``.model``;
* op-amps become a subcircuit with the same macromodel (RIN, gain stage
  A0/RP into RP||CP, a B-source tanh limiter, ROUT, IQ);
* Hotwire's 1e-12 S node shunt is reproduced with ``.options rshunt=1e12``.

Results are cached in ``tests/_ngspice/<name>/`` keyed by a hash of the deck,
so the suite only calls WSL when a circuit or analysis changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from engine import devices as dv
from engine.netlist import Circuit

ROOT = Path(__file__).resolve().parent / "_ngspice"
NGSPICE = ("LD_LIBRARY_PATH=~/ngs/root/usr/lib/x86_64-linux-gnu "
           "~/ngs/root/usr/bin/ngspice -b")


def wsl_path(p: Path) -> str:
    """I:\\a\\b -> /mnt/i/a/b"""
    s = str(Path(p).resolve())
    if len(s) > 1 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")


def available() -> bool:
    """True when WSL and the ngspice build are reachable (or every result is cached)."""
    if shutil.which("wsl") is None:
        return False
    try:
        r = subprocess.run(["wsl", "-e", "bash", "-lc", NGSPICE.replace(" -b", "") + " -v"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "ngspice-42" in (r.stdout + r.stderr)


def _num(x: float) -> str:
    return f"{x:.12g}"


_SKIP_PARAMS = {"NBV"}       # Hotwire-only (0 = use N)


def _model_line(name: str, mtype: str, params: dict) -> str:
    keep = {k: v for k, v in params.items() if k not in _SKIP_PARAMS and math.isfinite(float(v))}
    if mtype in ("NPN", "PNP"):
        keep = {k: v for k, v in keep.items()}
    body = " ".join(f"{k}={_num(float(v))}" for k, v in keep.items())
    return f".model {name} {mtype}({body})"


def _opamp_subckt(name: str, p: dict) -> list[str]:
    """The engine's own macromodel (with the anti-windup clamp), from engine/export.py."""
    from engine.export import opamp_subckt_lines
    return opamp_subckt_lines(name, p)


def deck(c: Circuit, *, out_dir: Path, input_name: str | None = None, nodes: list | None = None,
         ac: tuple | None = None, tran: tuple | None = None) -> str:
    """ngspice deck for circuit ``c``.  ``ac=(fstart, fstop, ppd)``, ``tran=(tstep, tstop)``."""
    inp = (input_name or c.input or "").upper()
    lines = [f"* {c.title} (generated for the Hotwire cross-check)"]
    sub_done: dict[str, str] = {}
    sub_lines: list[str] = []
    used_models = set()
    for e in c.elements:
        n = " ".join(e.nodes)
        k = e.kind
        if k in "RCL":
            lines.append(f"{e.name} {n} {_num(e.value)}")
        elif k in "VI":
            parts = [f"{e.name} {n} DC {_num(e.value or 0.0)}"]
            if e.name == inp and ac:
                parts.append("AC 1")
            if "sin" in e.params:
                parts.append("SIN(" + " ".join(_num(x) for x in e.params["sin"]) + ")")
            elif "pulse" in e.params:
                parts.append("PULSE(" + " ".join(_num(x) for x in e.params["pulse"]) + ")")
            lines.append(" ".join(parts))
        elif k in "EG":
            lines.append(f"{e.name} {n} {_num(e.value)}")
        elif k == "S":
            on = bool(e.params.get("on"))
            lines.append(f"R{e.name} {n} {_num(0.1 if on else 1e9)}")
        elif k in "DQJ":
            lines.append(f"{e.name} {n} {e.model}")
            used_models.add(e.model)
        elif k == "M":
            # one private model per instance: CGS/CGD farads fold into CGSO/CGDO per metre of W
            m = c.models[e.model]
            mp = dv.model_params(m.type, m.params)
            for key, dst in (("W", "W"), ("L", "L"), ("M", "MULT")):
                if key in e.params:
                    mp[dst] = float(e.params[key])
            mparams, inst = dv.mos_ngspice_params(mp)
            mname = f"{e.model}_{e.name}".replace(".", "_")
            lines.append(f"{e.name} {n} {mname} " + " ".join(f"{k2}={_num(v)}" for k2, v in inst.items()))
            sub_lines.append(f".model {mname} {m.type}(LEVEL=1 "
                             + " ".join(f"{k2}={_num(v)}" for k2, v in mparams.items()) + ")")
        elif k == "X":
            m = c.models[e.model]
            p = dv.model_params("OPAMP", m.params)
            sname = sub_done.get(e.model)
            if sname is None:
                sname = f"OA_{len(sub_done)}"
                sub_done[e.model] = sname
                sub_lines += _opamp_subckt(sname, p)
            lines.append(f"{e.name} {n} {sname}")
    for name in sorted(used_models):
        m = c.models[name]
        lines.append(_model_line(name, m.type, m.params))
    lines += sub_lines
    lines.append(".options rshunt=1e12 reltol=1e-6 abstol=1e-15 vntol=1e-9 itl1=500 itl4=200")
    od = wsl_path(out_dir)
    node_list = nodes or c.nodes()
    vecs = " ".join(f"v({x})" for x in node_list)
    lines += [".control", "set wr_singlescale", "set wr_vecnames", "option numdgt=12",
              "op", f"wrdata {od}/op.txt {vecs}"]
    if ac:
        fstart, fstop, ppd = ac
        lines += [f"ac dec {int(ppd)} {_num(fstart)} {_num(fstop)}", f"wrdata {od}/ac.txt v({c.output})"]
    if tran:
        tstep, tstop = tran
        tv = " ".join(f"v({x})" for x in node_list)
        lines += [f"tran {_num(tstep)} {_num(tstop)} 0 {_num(tstep)}", f"linearize {tv}",
                  f"wrdata {od}/tran.txt {tv}"]
    lines += [".endc", ".end"]
    return "\n".join(lines) + "\n"


def _read(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().split()
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return header, data


def run(c: Circuit, name: str, *, input_name: str | None = None, ac: tuple | None = None,
        tran: tuple | None = None, nodes: list | None = None, force: bool = False) -> dict:
    """Simulate ``c`` in ngspice; returns {'op': {node: V}, 'ac': (f, h), 'tran': (t, {node: v})}."""
    out_dir = ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    text = deck(c, out_dir=out_dir, input_name=input_name, ac=ac, tran=tran, nodes=nodes)
    # the deck names its own output folder; hash it without that path so cached results
    # shipped in the repo stay valid on any machine
    digest = hashlib.sha256(text.replace(wsl_path(out_dir), "<out>").encode()).hexdigest()[:16]
    stamp = out_dir / "deck.hash"
    cir = out_dir / "deck.cir"
    if force or not stamp.exists() or stamp.read_text().strip() != digest:
        for f in ("op.txt", "ac.txt", "tran.txt"):
            (out_dir / f).unlink(missing_ok=True)
        cir.write_text(text, encoding="utf-8")
        r = subprocess.run(["wsl", "-e", "bash", "-lc", f"{NGSPICE} {wsl_path(cir)}"],
                           capture_output=True, text=True, timeout=600)
        (out_dir / "ngspice.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        if not (out_dir / "op.txt").exists():
            raise RuntimeError(f"ngspice failed for {name}; see {out_dir / 'ngspice.log'}")
        stamp.write_text(digest)
    res: dict = {}
    node_list = nodes or c.nodes()
    _, d = _read(out_dir / "op.txt")
    res["op"] = {n: float(d[0, 1 + j]) for j, n in enumerate(node_list)}
    if ac:
        _, d = _read(out_dir / "ac.txt")
        res["ac"] = (d[:, 0], d[:, 1] + 1j * d[:, 2])
    if tran:
        _, d = _read(out_dir / "tran.txt")
        res["tran"] = (d[:, 0], {n: d[:, 1 + j] for j, n in enumerate(node_list)})
    return res


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")




# --------------------------------------------------------------------------- cross-check suite

MODELS = """
.model D1N4148 D(IS=2.52n N=1.752 RS=0.568 CJO=4p M=0.4 TT=11.5n)
.model D1N34A D(IS=2e-7 N=1.3 RS=7 CJO=0.5p M=0.5)
.model LEDR D(IS=93.2p N=3.73 RS=2 CJO=20p)
.model Q2N3904 NPN(IS=6.734f BF=416.4 BR=0.7371 NF=1 NR=1 VAF=74.03 IKF=66.78m ISE=6.734f NE=1.259
+ RB=10 RC=1 CJE=4.493p MJE=0.2593 VJE=0.75 CJC=3.638p MJC=0.3085 VJC=0.75 TF=301.2p TR=239.5n)
.model Q2N3906 PNP(IS=1.41f BF=180.7 BR=4.977 NF=1 NR=1 VAF=18.7 IKF=80m ISE=0 RB=10 RC=2.5
+ CJE=8.063p MJE=0.3677 VJE=0.75 CJC=4.017p MJC=0.5 VJC=0.75 TF=513.3p TR=33.42n)
.model QBC549 NPN(IS=7.049f BF=381.7 BR=2.359 VAF=62.79 NF=1.008 NR=1.005 RB=10 RC=0.9
+ CJE=11.5p MJE=0.6721 VJE=0.5 CJC=5.25p MJC=0.3147 VJC=0.5683 TF=410.2p TR=10n)
.model AC128 PNP(IS=5u BF=90 BR=4 NF=1.15 NR=1.15 VAF=40 RB=200 CJE=40p CJC=25p)
.model J201 NJF(VTO=-0.6 BETA=1.304m LAMBDA=2.4m RD=1 RS=1 IS=114.5f CGD=4.06p CGS=4.06p PB=0.5)
.model OA1 OPAMP(A0=2e5 GBW=3e6 RIN=1e12 ROUT=75 VSAT_HI=1.5 VSAT_LO=1.5 IQ=1.4m)
.model NMG NMOS(LEVEL=1 VTO=1.8 KP=0.12 LAMBDA=0.01 GAMMA=0.45 PHI=0.7 RD=1.5 RS=0.8 IS=1e-14
+ CBD=35p CBS=30p PB=0.8 MJ=0.5 CGSO=1.6e-7 CGDO=4e-8)
.model NMC NMOS(VTO=1 KP=200u LAMBDA=0.02 GAMMA=0.4 PHI=0.65 CBD=20p CBS=20p CGSO=5e-8 CGDO=2e-8 CGBO=1e-8)
.model PMC PMOS(VTO=-1 KP=100u LAMBDA=0.02 GAMMA=0.4 PHI=0.65 CBD=25p CBS=25p CGSO=5e-8 CGDO=2e-8 CGBO=1e-8)
"""

# name -> (netlist, {"ac": (fstart, fstop, points/decade) or None, "tran": (tstep, tstop) or None})
CIRCUITS: dict[str, tuple[str, dict]] = {
    "diode_clipper_sym": ("""* Symmetric diode clipper (1N4148 pair)
*@ in VIN
*@ out out
VIN in 0 DC 0 SIN(0 2 1k)
R1 in out 4.7k
D1 out 0 D1N4148
D2 0 out D1N4148
C1 out 0 10n
""", {"ac": (10, 100e3, 20), "tran": (5e-6, 4e-3)}),

    "diode_clipper_asym": ("""* Asymmetric clipper: two silicon diodes one way, one germanium the other
*@ in VIN
*@ out out
VIN in 0 DC 0 SIN(0 3 500)
R1 in out 10k
D1 out m D1N4148
D2 m 0 D1N4148
D3 0 out D1N34A
C1 out 0 4.7n
""", {"ac": (10, 100e3, 20), "tran": (5e-6, 6e-3)}),

    "ce_amp": ("""* Common-emitter amplifier, bypassed emitter (2N3904, full GP terms)
*@ in VIN
*@ out out
*@ supply VCC
VCC vcc 0 9
VIN in 0 DC 0 SIN(0 20m 1k)
C1 in b 1u
R1 vcc b 100k
R2 b 0 22k
RC vcc c 4.7k
RE e 0 1k
CE e 0 100u
Q1 c b e Q2N3904
C2 c out 1u
RL out 0 100k
""", {"ac": (10, 1e6, 20), "tran": (5e-6, 5e-3)}),

    "fuzz_two_transistor": ("""* Two-transistor fuzz (silicon NPN, fuzz-face topology)
*@ in VIN
*@ out out
*@ supply VCC
VCC vcc 0 9
VIN in 0 DC 0 SIN(0 0.1 200)
RS in g 10k
C1 g b1 2.2u
Q1 c1 b1 0 QBC549
R1 vcc c1 33k
Q2 c2 c1 e2 QBC549
R3 vcc c2 8.2k
R5 e2 b1 100k
RE e2 0 1k
CE e2 0 22u
R4 c2 o1 470
C3 o1 out 10n
RL out 0 500k
""", {"ac": (10, 100e3, 20), "tran": (5e-6, 15e-3)}),

    "jfet_booster": ("""* JFET common-source booster (J201)
*@ in VIN
*@ out out
*@ supply VCC
VCC vcc 0 9
VIN in 0 DC 0 SIN(0 0.3 1k)
C1 in g 100n
RG g 0 1meg
J1 d g s J201
RD vcc d 6.8k
RSRC s 0 680
CS s 0 22u
C2 d out 1u
RL out 0 100k
""", {"ac": (10, 1e6, 20), "tran": (5e-6, 5e-3)}),

    "sallen_key_lp": ("""* Unity-gain Sallen-Key low-pass, about 1.07 kHz
*@ in VIN
*@ out out
*@ supply VCC VEE
VCC vcc 0 9
VEE vee 0 -9
VIN in 0 DC 0 SIN(0 0.5 1k)
R1 in a 10k
R2 a p 10k
C1 a out 22n
C2 p 0 10n
X1 p out vcc vee out OA1
""", {"ac": (10, 1e6, 20), "tran": (5e-6, 4e-3)}),

    "opamp_soft_clipper": ("""* Non-inverting op-amp with diodes in the feedback loop
*@ in VIN
*@ out out
*@ supply VCC VEE
VCC vcc 0 9
VEE vee 0 -9
VIN in 0 DC 0 SIN(0 0.2 500)
X1 in fb vcc vee out OA1
RF out fb 220k
CF out fb 51p
D1 out fb D1N4148
D2 fb out D1N4148
RG fb g 4.7k
CG g 0 47n
""", {"ac": (10, 1e6, 20), "tran": (5e-6, 6e-3)}),

    "led_driver": ("""* Transistor LED driver switched by a pulse
*@ in VIN
*@ out c
*@ supply VCC
VCC vcc 0 9
VIN in 0 DC 5 PULSE(0 5 0.2m 1u 1u 0.5m 1m)
RB in b 10k
Q1 c b 0 Q2N3904
RL vcc a 390
D1 a c LEDR
""", {"ac": None, "tran": (2e-6, 2e-3)}),

    "germanium_pnp_stage": ("""* Germanium PNP common-emitter stage on a negative supply
*@ in VIN
*@ out out
*@ supply VNEG
VNEG vneg 0 -9
VIN in 0 DC 0 SIN(0 10m 1k)
C1 in b 10u
R1 b vneg 68k
R2 b 0 10k
RC c vneg 3.3k
RE e 0 1k
CE e 0 47u
Q1 c b e AC128
C2 c out 10u
RL out 0 100k
""", {"ac": (10, 1e6, 20), "tran": (5e-6, 5e-3)}),

    "rlc_bandpass": ("""* RLC band-pass: resistor feeding a parallel LC tank (f0 about 5 kHz)
*@ in VIN
*@ out out
VIN in 0 DC 0 SIN(0 0.5 5k)
R1 in out 1k
L1 out m 10m
RL m 0 10
C1 out 0 100n
""", {"ac": (100, 100e3, 40), "tran": (1e-6, 2e-3)}),

    "baxandall_tone_stack": ("""* Passive Baxandall bass/treble tone stack with pots
*@ in VIN
*@ out out
VIN in 0 DC 0 SIN(0 1 200)
R1 in n1 10k
*@ pot RVB n1 wb n2 100k taper=log pos=0.7 label="Bass"
R2 n2 0 1k
C1 n1 wb 22n
C2 n2 wb 220n
R3 wb out 10k
C3 in n3 2.2n
*@ pot RVT n3 wt n4 100k taper=log pos=0.3 label="Treble"
C4 n4 0 22n
R4 wt out 1k
RL out 0 1meg
""", {"ac": (10, 100e3, 20), "tran": (5e-6, 10e-3)}),

    "opamp_inverting_clip": ("""* Inverting op-amp, gain -10, driven into the rails
*@ in VIN
*@ out out
*@ supply VCC VEE
VCC vcc 0 9
VEE vee 0 -9
VIN in 0 DC 0 SIN(0 1 1k)
R1 in m 10k
R2 m out 100k
X1 0 m vcc vee out OA1
RL out 0 10k
""", {"ac": (10, 1e6, 20), "tran": (2e-6, 3e-3)}),

    "push_pull_follower": ("""* Class-B complementary emitter follower (crossover distortion)
*@ in VIN
*@ out out
*@ supply VCC VEE
VCC vcc 0 9
VEE vee 0 -9
VIN in 0 DC 0 SIN(0 2 1k)
RIN in b 100
Q1 vcc b out Q2N3904
Q2 vee b out Q2N3906
RL out 0 1k
""", {"ac": None, "tran": (2e-6, 3e-3)}),

    "half_wave_rectifier": ("""* Half-wave rectifier with reservoir cap (stiff diode switching)
*@ in VIN
*@ out out
VIN in 0 DC 0 SIN(0 5 50)
D1 in out D1N4148
C1 out 0 100u
RL out 0 1k
""", {"ac": None, "tran": (20e-6, 60e-3)}),

    "npn_pnp_cascade": ("""* NPN emitter follower driving a PNP common-emitter stage
*@ in VIN
*@ out out
*@ supply VCC
VCC vcc 0 12
VIN in 0 DC 0 SIN(0 0.05 2k)
C1 in b1 1u
R1 vcc b1 150k
R2 b1 0 100k
Q1 vcc b1 e1 Q2N3904
RE1 e1 0 4.7k
Q2 c2 bp e2 Q2N3906
RB2 vcc bp 47k
CB2 e1 bp 1u
RB3 bp 0 100k
RE2 vcc e2 3.3k
CE2 vcc e2 10u
RC2 c2 0 4.7k
C2 c2 out 1u
RL out 0 47k
""", {"ac": (10, 1e6, 20), "tran": (5e-6, 3e-3)}),

    "mosfet_cs_amp": ("""* MOSFET common-source amplifier, bulk to ground (body effect, RD/RS, junction caps)
*@ in VIN
*@ out out
*@ supply VCC
VCC vcc 0 12
VIN in 0 DC 0 SIN(0 10m 1k)
C1 in g 1u
R1 vcc g 2.2meg
R2 g 0 1meg
M1 d g s 0 NMG W=200u L=100u
RD vcc d 4.7k
RS s 0 1k
CS s 0 100u
C2 d out 1u
RL out 0 100k
""", {"ac": (10, 1e6, 20), "tran": (5e-6, 5e-3)}),

    "mosfet_source_follower": ("""* NMOS source follower (BS170, bulk tied to source)
*@ in VIN
*@ out out
*@ supply VCC
VCC vcc 0 9
VIN in 0 DC 0 SIN(0 1 1k)
C1 in g 1u
R1 vcc g 1meg
R2 g 0 1meg
M1 vcc g s BS170
RS s 0 2.2k
C2 s out 10u
RL out 0 10k
""", {"ac": (10, 1e6, 20), "tran": (5e-6, 5e-3)}),

    "cmos_inverter": ("""* CMOS inverter switching a 1 nF load
*@ in VIN
*@ out out
*@ supply VDD
VDD vdd 0 5
VIN in 0 DC 0 PULSE(0 5 20u 10u 10u 40u 100u)
MP out in vdd vdd PMC W=200u L=100u
MN out in 0 0 NMC
CL out 0 1n
""", {"ac": None, "tran": (0.2e-6, 200e-6)}),

    "pmos_high_side_switch": ("""* P-channel high-side switch (IRF9510) driven by an NPN level shifter
*@ in VIN
*@ out load
*@ supply VCC
VCC vcc 0 12
VIN in 0 DC 0 PULSE(0 5 50u 2u 2u 200u 400u)
RB in b 10k
Q1 g b 0 Q2N3904
RG vcc g 10k
M1 load g vcc vcc IRF9510
RL load 0 100
CL load 0 100n
""", {"ac": None, "tran": (0.5e-6, 800e-6)}),
}


def circuit_text(name: str) -> str:
    """Netlist text of a cross-check circuit, with the shared .model block."""
    return CIRCUITS[name][0] + MODELS


def compare(name: str, *, force: bool = False) -> dict:
    """Run one cross-check circuit in both simulators and return the error metrics."""
    from engine import ac as hw_ac, op as hw_op, parse, tran as hw_tran
    from engine import measure as ms

    _, cfg = CIRCUITS[name]
    c = parse(circuit_text(name))
    ref = run(c, name, ac=cfg.get("ac"), tran=cfg.get("tran"), force=force)
    out: dict = {"name": name, "title": c.title}

    # DC: worst node error, absolute and as a fraction of the 1% / 1 mV allowance
    o = hw_op(c)
    worst, worst_ratio, worst_node = 0.0, 0.0, None
    for n, vref in ref["op"].items():
        err = abs(o.v[n] - vref)
        allowed = max(0.01 * abs(vref), 1e-3)
        if err / allowed >= worst_ratio:
            worst_ratio, worst, worst_node = err / allowed, err, n
    out["dc_max_err_mv"] = worst * 1e3
    out["dc_worst_node"] = worst_node
    out["dc_ratio"] = worst_ratio
    out["dc_converged"] = o.converged
    out["dc_method"] = o.method

    if cfg.get("ac"):
        fstart, fstop, ppd = cfg["ac"]
        a = hw_ac(c, fstart, fstop, ppd)
        f_ng, h_ng = ref["ac"]
        h_hw = (np.interp(np.log10(f_ng), np.log10(a.f), a.h.real)
                + 1j * np.interp(np.log10(f_ng), np.log10(a.f), a.h.imag))
        g_hw, g_ng = ms.db(h_hw), ms.db(h_ng)
        band = g_ng > g_ng.max() - 60.0          # ignore the deep stop band
        out["ac_max_err_db"] = float(np.max(np.abs(g_hw - g_ng)[band]))
        lo_n, hi_n, _, _ = ms.corners((f_ng, h_ng))
        lo_h, hi_h, _, _ = ms.corners((a.f, a.h))
        errs = [abs(x / y - 1) * 100 for x, y in ((lo_h, lo_n), (hi_h, hi_n))
                if x is not None and y is not None]
        out["f3db_err_pct"] = max(errs) if errs else None
    if cfg.get("tran"):
        tstep, tstop = cfg["tran"]
        tr = hw_tran(c, tstop, tstep, nodes=[c.output])
        t_ng, v_ng = ref["tran"]
        y_ng = v_ng[c.output]
        y_hw = np.interp(t_ng, tr.t, tr.v[c.output])
        out["tran_rms_err_pct"] = ms.rms_error(y_hw, y_ng) * 100
        out["tran_signal_rms"] = ms.rms(y_ng - y_ng.mean())
        out["tran_converged"] = tr.converged
        out["tran_stats"] = tr.stats
    return out


def compare_audio(name: str, sr: int = 48000) -> dict:
    """process_audio at ``sr`` against the ngspice transient (input = the circuit's SIN source)."""
    from engine import parse, process_audio
    from engine import measure as ms

    _, cfg = CIRCUITS[name]
    c = parse(circuit_text(name))
    ref = run(c, name, ac=cfg.get("ac"), tran=cfg.get("tran"))
    t_ng, v_ng = ref["tran"]
    vin = c.element(c.input)
    vo, va, f, td, theta = vin.params["sin"][:5]
    n = int(t_ng[-1] * sr)
    t = np.arange(n) / sr
    x = va * np.sin(2 * np.pi * f * t)
    y = process_audio(c, x, sr)
    y_ng = np.interp(t, t_ng, v_ng[c.output]) - ref["op"][c.output]
    return {"name": name, "audio_rms_err_pct": ms.rms_error(y, y_ng) * 100}
