"""ngspice-42 cross-check of the Hotwire reference library.

For every ``library/*.cir`` this module

1. parses the netlist with its own small reader (independent of engine/netlist.py),
2. writes an ngspice deck to ``tests/ngspice_library/``: ``*@ pot`` lines become two
   resistors (SPEC taper law), built-in parts become ``.model`` lines from
   ``engine/parts_data.py``, and every op-amp becomes the Hotwire macromodel
   subcircuit (see ``opamp_subckt``),
3. runs ``.op`` and ``.ac`` (1 Hz to 200 kHz, 100 points per decade) and, where a spec
   needs it, ``.tran`` with a sine on the ``*@ in`` source,
4. measures every ``*@ spec`` with the numpy code below and asserts it passes.

``python tests/test_library_ngspice.py --write-index`` also rewrites
``library/index.json`` (the ground truth the Python simulator is compared against).
``python tests/test_library_ngspice.py [id ...]`` prints a report without writing.

ngspice is found as ``$HOTWIRE_NGSPICE`` (a full command prefix), a native ``ngspice`` on
PATH, or (Windows) inside WSL at ``~/ngs/root/usr/bin/ngspice`` or ``ngspice``.  The
tests skip when none is available.

Measurement conventions (the Python engine must use the same ones to compare):
* gain_db(f), phase(f): out/in transfer, dB and phase linearly interpolated over log f
  on the 100-points-per-decade grid.
* f3db_low/f3db_high: first -3 dB crossings below/above the gain maximum of the
  1 Hz-200 kHz sweep (log-f interpolation); null if the response does not fall 3 dB
  inside the sweep.  fc: f3db_high if the gain at 1 Hz >= the gain at 200 kHz, else
  f3db_low.  peak_db / peak_f: maximum gain on the grid and its frequency.
  (Same rules as engine/specs.py.)
* tran specs (same timing as engine/specs.py): sine starting at the DC operating point
  with a 5 ms linear fade-in, settle = max(60 ms, 8 cycles), then a window of
  max(4, round(0.02 f)) whole cycles; trapezoidal integration, max step T/200.
  thd = sqrt(sum H2..H15^2) / H1 in %, Hk by Fourier projection (256 samples/cycle) of the
  window minus its mean.  vout_peak = max |v(out) - mean| over the window.
  gain_tran = 20 log10(vout_peak / vin_peak).
* headroom(f): largest vin_peak with thd <= 1 %: geometric bisection between 5 mV and
  5 V, 9 steps (same as engine/specs.py).
* supply_current: sum of the DC currents delivered by the ``*@ supply`` sources.
* BJT region: cutoff if |Ic| < 1 uA, saturation if |Vce| < 0.2 V, else active.
  JFET region (SPICE names): cutoff if Vgs < VTO, triode if Vds < Vgs - VTO, else
  saturation (the amplifying region), so JFET specs read ``region(J1) == saturation``.
  MOSFETs (level 1, bulk on the source) use the same rule in device polarity.
* MOSFET built-ins carry CGS/CGD in farads (a Hotwire convenience); the deck writes them as
  CGSO/CGDO per metre of the default W = 100 um, which ngspice multiplies back.
* ngspice runs at its default 27 C (Vt = 25.865 mV), GMIN 1e-12.
"""
from __future__ import annotations

import argparse
import cmath
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # allows running the script without pytest installed
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
DECK_DIR = ROOT / "tests" / "ngspice_library"
INDEX_PATH = LIB_DIR / "index.json"


def _load_parts():
    spec = importlib.util.spec_from_file_location("hotwire_parts_data", ROOT / "engine" / "parts_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PARTS


PARTS = _load_parts()

# Library metadata that is not part of the netlist format: category and tags.
CATALOG = {
    "rc_lowpass": ("filter", ["textbook", "passive", "low-pass", "first-order"]),
    "sallen_key_lowpass": ("filter", ["textbook", "op-amp", "low-pass", "second-order", "butterworth"]),
    "ce_amplifier": ("amp", ["textbook", "bjt", "common-emitter", "degeneration"]),
    "jfet_buffer": ("utility", ["jfet", "buffer", "source-follower", "pedal"]),
    "jfet_boost": ("amp", ["jfet", "boost", "clean", "pedal"]),
    "diode_hard_clipper": ("drive", ["op-amp", "distortion", "hard-clip", "silicon-diodes", "pedal"]),
    "softclip_overdrive": ("drive", ["op-amp", "overdrive", "soft-clip", "mid-hump", "tone", "pedal"]),
    "silicon_fuzz": ("fuzz", ["bjt", "fuzz", "silicon", "two-transistor", "pedal"]),
    "germanium_fuzz": ("fuzz", ["bjt", "fuzz", "germanium", "two-transistor", "pnp", "pedal"]),
    "passive_tone_stack": ("tone", ["passive", "tone-stack", "amp", "bass-mid-treble"]),
    "baxandall_tone": ("tone", ["op-amp", "active", "tone-control", "bass-treble"]),
    "treble_booster": ("drive", ["bjt", "germanium", "pnp", "treble", "boost", "pedal"]),
    "inductor_wah": ("filter", ["bjt", "inductor", "band-pass", "resonant", "wah", "pedal"]),
    "octave_fuzz": ("fuzz", ["bjt", "octave", "rectifier", "germanium-diodes", "pedal"]),
    "power_filter_led": ("utility", ["power", "led", "reverse-polarity", "ripple-filter", "pedal"]),
    "mosfet_boost": ("amp", ["mosfet", "boost", "clean", "common-source", "pedal"]),
    "wien_oscillator": ("oscillator", ["textbook", "op-amp", "oscillator", "sine", "1khz"]),
}

AC_FSTART, AC_FSTOP, AC_PPD = 1.0, 200e3, 100
REPORT_FREQS = (100.0, 1000.0, 5000.0)

# --------------------------------------------------------------------------- values

_SUFFIX = [("meg", 1e6), ("mil", 25.4e-6), ("f", 1e-15), ("p", 1e-12), ("n", 1e-9), ("u", 1e-6),
           ("m", 1e-3), ("k", 1e3), ("g", 1e9), ("t", 1e12)]
_NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(.*)$")


def parse_value(tok: str) -> float:
    m = _NUM.match(tok.strip())
    if not m:
        raise ValueError(f"not a number: {tok!r}")
    x = float(m.group(1))
    rest = m.group(2).lower()
    for suf, mult in _SUFFIX:
        if rest.startswith(suf):
            return x * mult
    return x


def taper_fraction(taper: str, pos: float) -> float:
    pos = min(1.0, max(0.0, pos))
    if taper == "log":
        return (11.0 ** pos - 1.0) / 10.0
    if taper == "revlog":
        return 1.0 - (11.0 ** (1.0 - pos) - 1.0) / 10.0
    return pos


# --------------------------------------------------------------------------- netlist reader

class Netlist:
    def __init__(self, path: Path):
        self.path = path
        self.id = path.stem
        self.text = path.read_text(encoding="utf-8")
        self.title = self.id
        self.description = ""
        self.input = None
        self.output = None
        self.supplies: list[str] = []
        self.pots: dict[str, dict] = {}
        self.specs: list[str] = []
        self.lines: list[tuple[str, object]] = []   # ('el', tokens) | ('pot', name) | ('model', text)
        self.elements: dict[str, list[str]] = {}
        self._parse()

    def _logical_lines(self):
        out = []
        for raw in self.text.splitlines():
            s = raw.rstrip()
            if s.startswith("+") and out:
                out[-1] = out[-1] + " " + s[1:].strip()
            else:
                out.append(s)
        return out

    def _parse(self):
        seen_title = False
        for line in self._logical_lines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("*@"):
                body = s[2:].strip()
                key, _, rest = body.partition(" ")
                key = key.lower()
                rest = rest.strip()
                if key == "title":
                    self.title = rest
                    seen_title = True
                elif key == "in":
                    self.input = rest.upper()
                elif key == "out":
                    self.output = rest.lower()
                elif key == "supply":
                    self.supplies.append(rest.upper())
                elif key == "spec":
                    self.specs.append(rest)
                elif key == "pot":
                    toks = shlex.split(rest)
                    name, a, w, b, val = toks[:5]
                    pot = {"name": name.upper(), "a": a.lower(), "w": w.lower(), "b": b.lower(),
                           "value": parse_value(val), "taper": "lin", "pos": 0.5, "label": name.upper()}
                    for kv in toks[5:]:
                        k, _, v = kv.partition("=")
                        k = k.lower()
                        if k == "taper":
                            pot["taper"] = v.lower()
                        elif k == "pos":
                            pot["pos"] = float(v)
                        elif k == "label":
                            pot["label"] = v
                    self.pots[pot["name"]] = pot
                    self.lines.append(("pot", pot["name"]))
                continue
            if s.startswith("*"):
                if seen_title and not self.description:
                    self.description = s.lstrip("*").strip()
                continue
            low = s.lower()
            if low.startswith(".model"):
                self.lines.append(("model", s))
                continue
            if low.startswith("."):
                continue  # .op/.ac/.tran/.end are chosen by the harness
            toks = s.split()
            name = toks[0].upper()
            self.elements[name] = toks
            self.lines.append(("el", toks))

    @property
    def explicit_models(self):
        names = set()
        for kind, item in self.lines:
            if kind == "model":
                names.add(item.split()[1].upper())
        return names

    def input_nodes(self):
        toks = self.elements[self.input]
        return toks[1].lower(), toks[2].lower()


# --------------------------------------------------------------------------- deck writer

def _fmt(x: float) -> str:
    return f"{x:.10g}"


DEFAULT_MOS_W = 100e-6      # ngspice DEFW


def model_line(name: str) -> str:
    p = PARTS[name]
    prm = dict(p["params"])
    level = ""
    if p["type"] in ("NMOS", "PMOS"):
        prm["CGSO"] = prm.get("CGSO", 0.0) + prm.pop("CGS", 0.0) / DEFAULT_MOS_W
        prm["CGDO"] = prm.get("CGDO", 0.0) + prm.pop("CGD", 0.0) / DEFAULT_MOS_W
        level = "LEVEL=1 "
    params = " ".join(f"{k}={_fmt(v)}" for k, v in prm.items())
    return f".model {name} {p['type']}({level}{params})"


def _mos_model_index(toks) -> int:
    """Token index of a MOSFET line's model: the last token that isn't name=value."""
    return max(i for i, x in enumerate(toks) if i > 0 and "=" not in x)


def opamp_subckt(name: str) -> str:
    """Hotwire op-amp macromodel (per amplifier), pins in+ in- vcc vee out.

    * RIN across the inputs.
    * Gain stage: VCCS gm = A0/1Meg into 1 Meg || C, C sets the single pole at GBW/A0.
    * Anti-windup clamp: a smooth one-sided conductance (1000 S) holds the gain node
      between (VEE + VSAT_LO) and (VCC - VSAT_HI); the corner is rounded with a
      softplus of width 10 mV.  The node is referenced to ground.
    * Output: unity VCVS of the gain node behind ROUT.
    * IQ: constant current source from vcc to vee.
    """
    p = PARTS[name]["params"]
    a0, gbw = p["A0"], p["GBW"]
    fp = gbw / a0
    r1 = 1e6
    gm = a0 / r1
    c1 = 1.0 / (2 * math.pi * fp * r1)
    rout = max(p["ROUT"], 1e-3)
    eps = 0.01
    gc = 1000.0
    hi = f"(v(vcc)-{_fmt(p['VSAT_HI'])})"
    lo = f"(v(vee)+{_fmt(p['VSAT_LO'])})"

    def softplus(x):
        return f"(({x})>0 ? ({x})+{eps}*ln(1+exp(-({x})/{eps})) : {eps}*ln(1+exp(({x})/{eps})))"

    lines = [
        f".subckt HWOA_{name} inp inn vcc vee out",
        f"Rin inp inn {_fmt(p['RIN'])}",
        f"G1 0 n1 inp inn {_fmt(gm)}",
        f"R1 n1 0 {_fmt(r1)}",
        f"C1 n1 0 {_fmt(c1)}",
        f"Bcl n1 0 I={gc}*({softplus('v(n1)-' + hi)}-{softplus(lo + '-v(n1)')})",
        "E1 ob 0 n1 0 1",
        f"Ro ob out {_fmt(rout)}",
    ]
    if p["IQ"] > 0:
        lines.append(f"Iq vcc vee DC {_fmt(p['IQ'])}")
    lines.append(".ends")
    return "\n".join(lines)


def circuit_body(nl: Netlist, pots=None, input_line=None) -> list[str]:
    """Element lines for ngspice with pots expanded and parts resolved."""
    pos = {k: v["pos"] for k, v in nl.pots.items()}
    if pots:
        pos.update({k.upper(): v for k, v in pots.items()})
    out = []
    used_parts, used_opamps = set(), set()
    for kind, item in nl.lines:
        if kind == "pot":
            p = nl.pots[item]
            frac = taper_fraction(p["taper"], pos[item])
            ra = max(1.0, frac * p["value"])
            rb = max(1.0, (1.0 - frac) * p["value"])
            out.append(f"{item}_A {p['a']} {p['w']} {_fmt(ra)}")
            out.append(f"{item}_B {p['w']} {p['b']} {_fmt(rb)}")
            continue
        if kind == "model":
            out.append(item)
            continue
        toks = list(item)
        name = toks[0].upper()
        c = name[0]
        if name == nl.input and input_line is not None:
            out.append(input_line)
            continue
        if c == "M" and _mos_model_index(toks) == 4:
            toks.insert(4, toks[3])          # 3-node form: ngspice needs the bulk, tied to the source
        if c in "DQJM":
            mi = 3 if c == "D" else (_mos_model_index(toks) if c == "M" else 4)
            m = toks[mi].upper()
            if m not in nl.explicit_models:
                if m not in PARTS:
                    raise KeyError(f"{nl.id}: unknown part {m}")
                used_parts.add(m)
            toks[mi] = m
        elif c == "X":
            m = toks[-1].upper()
            if PARTS.get(m, {}).get("type") != "OPAMP":
                raise KeyError(f"{nl.id}: unknown op-amp {m}")
            used_opamps.add(m)
            toks[-1] = f"HWOA_{m}"
        elif c == "S":
            state = toks[3].lower()
            toks = [f"R_S_{name}", toks[1], toks[2], "0.1" if state == "on" else "1e9"]
        out.append(" ".join(toks))
    for m in sorted(used_parts):
        out.append(model_line(m))
    for m in sorted(used_opamps):
        out.append(opamp_subckt(m))
    return out


def device_print_list(nl: Netlist) -> list[str]:
    vecs = []
    for name, toks in nl.elements.items():
        n = name.lower()
        if name[0] == "Q":
            vecs += [f"@{n}[ic]", f"@{n}[ib]", f"@{n}[vbe]", f"@{n}[vbc]"]
        elif name[0] == "J":
            vecs += [f"@{n}[id]", f"@{n}[vgs]", f"@{n}[vgd]"]
        elif name[0] == "M":
            vecs += [f"@{n}[id]", f"@{n}[vgs]", f"@{n}[vds]"]
        elif name[0] == "D":
            vecs += [f"@{n}[id]"]
    return vecs


def deck_header(nl: Netlist, what: str) -> list[str]:
    return [f"* Hotwire ngspice deck ({what}) for library/{nl.path.name}: {nl.title}",
            "* Generated by tests/test_library_ngspice.py; do not edit by hand."]


def ac_expr(nl: Netlist) -> str:
    np_, nm = nl.input_nodes()
    vin = f"v({np_})" if nm in ("0", "gnd") else f"(v({np_})-v({nm}))"
    return f"v({nl.output})/{vin}"


def build_op_ac_deck(nl: Netlist, pots=None, ac=True) -> str:
    lines = deck_header(nl, "op + ac") + circuit_body(nl, pots)
    ctl = [".control", "set noaskquit", "set filetype=ascii", "set wr_vecnames", "set wr_singlescale",
           "set numdgt=12", "op", "print all"]
    dev = device_print_list(nl)
    for i in range(0, len(dev), 8):
        ctl.append("print " + " ".join(dev[i:i + 8]))
    if ac and nl.input and nl.output:
        ctl += [f"ac dec {AC_PPD} {_fmt(AC_FSTART)} {_fmt(AC_FSTOP)}",
                f"let hr = real({ac_expr(nl)})", f"let hi = imag({ac_expr(nl)})",
                "wrdata ac.txt hr hi"]
    ctl += [".endc", ".end"]
    return "\n".join(lines + ctl) + "\n"


def tran_times(f: float):
    """(max step, tstop, window cycles) matching engine/specs.py sine_run."""
    T = 1.0 / f
    settle = max(0.06, 8 * T)
    ncyc = max(4, int(round(0.02 * f)))
    return T / 200.0, settle + ncyc * T, ncyc


def build_tran_deck(nl: Netlist, f: float, amp: float, pots=None) -> str:
    np_, nm = nl.input_nodes()
    vin = (f"B{nl.input} {np_} {nm} V={_fmt(amp)}*min(1,time/0.005)"
           f"*sin({_fmt(2 * math.pi * f)}*time)")
    step, tstop, _ = tran_times(f)
    lines = deck_header(nl, f"tran {f:g} Hz, {amp:g} V peak") + circuit_body(nl, pots, input_line=vin)
    ctl = [".control", "set noaskquit", "set filetype=ascii", "set wr_vecnames", "set wr_singlescale",
           "set numdgt=12", f"tran {_fmt(step)} {_fmt(tstop)} 0 {_fmt(step)}",
           f"wrdata tran.txt v({nl.output})", ".endc", ".end"]
    return "\n".join(lines + ctl) + "\n"


def build_osc_deck(nl: Netlist, tstop=0.2, step=2e-6) -> str:
    lines = deck_header(nl, "free-running tran") + circuit_body(nl)
    ctl = [".control", "set noaskquit", "set filetype=ascii", "set wr_vecnames", "set wr_singlescale",
           "set numdgt=12", f"tran {_fmt(step)} {_fmt(tstop)} 0 {_fmt(step)} uic",
           f"wrdata tran.txt v({nl.output})", ".endc", ".end"]
    return "\n".join(lines + ctl) + "\n"


# --------------------------------------------------------------------------- running ngspice

def _win_to_wsl(p: Path) -> str:
    s = str(p.resolve())
    if len(s) > 1 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")


_RUNNER = None


def find_ngspice():
    """Return a function (workdir: Path) -> CompletedProcess running deck.cir, or None."""
    global _RUNNER
    if _RUNNER is not None:
        return _RUNNER or None
    env = os.environ.get("HOTWIRE_NGSPICE")
    if env:
        prefix = shlex.split(env)
        _RUNNER = lambda wd: subprocess.run(prefix + ["-b", "deck.cir"], cwd=wd, capture_output=True,
                                            text=True, timeout=600)
        return _RUNNER
    if os.name != "nt" and shutil.which("ngspice"):
        _RUNNER = lambda wd: subprocess.run(["ngspice", "-b", "deck.cir"], cwd=wd, capture_output=True,
                                            text=True, timeout=600)
        return _RUNNER
    if os.name == "nt" and shutil.which("wsl"):
        probe = ('if [ -x "$HOME/ngs/root/usr/bin/ngspice" ]; then echo local; '
                 'elif command -v ngspice >/dev/null; then echo path; else echo none; fi')
        try:
            r = subprocess.run(["wsl", "-e", "bash", "-c", probe], capture_output=True, text=True, timeout=120)
            kind = r.stdout.strip()
        except Exception:
            kind = "none"
        if kind == "local":
            cmd = ('LD_LIBRARY_PATH=$HOME/ngs/root/usr/lib/x86_64-linux-gnu '
                   '$HOME/ngs/root/usr/bin/ngspice -b deck.cir')
        elif kind == "path":
            cmd = "ngspice -b deck.cir"
        else:
            _RUNNER = False
            return None
        _RUNNER = lambda wd: subprocess.run(["wsl", "-e", "bash", "-c", f"cd '{_win_to_wsl(wd)}' && {cmd}"],
                                            capture_output=True, text=True, timeout=600)
        return _RUNNER
    _RUNNER = False
    return None


class NgspiceError(RuntimeError):
    pass


def run_deck(deck: str, outputs=("ac.txt", "tran.txt")):
    runner = find_ngspice()
    if runner is None:
        raise NgspiceError("ngspice not available")
    wd = Path(tempfile.mkdtemp(prefix="hotwire_ngs_"))
    try:
        (wd / "deck.cir").write_text(deck, encoding="utf-8", newline="\n")
        r = runner(wd)
        text = r.stdout + "\n" + r.stderr
        low = text.lower()
        for bad in ("singular matrix", "timestep too small", "no convergence", "simulation(s) aborted",
                    "error on line", "unknown model", "could not find"):
            if bad in low:
                raise NgspiceError(f"ngspice reported '{bad}':\n{text[-3000:]}")
        files = {}
        for name in outputs:
            fp = wd / name
            if fp.exists():
                files[name] = np.loadtxt(fp, skiprows=1, ndmin=2)
        return text, files
    finally:
        shutil.rmtree(wd, ignore_errors=True)


_PRINT_RE = re.compile(r"^\s*(\S+)\s*=\s*([-+0-9.eE]+)\s*$")


def parse_prints(text: str) -> dict:
    vals = {}
    for line in text.splitlines():
        m = _PRINT_RE.match(line)
        if m:
            try:
                vals[m.group(1).lower()] = float(m.group(2))
            except ValueError:
                pass
    return vals


# --------------------------------------------------------------------------- measurements

def interp_logf(f, y, f0):
    lf = np.log10(f)
    return float(np.interp(math.log10(f0), lf, y))


class Measured:
    """Lazy ngspice measurements for one circuit at one set of pot positions."""

    def __init__(self, nl: Netlist, pots=None, save_decks=False):
        self.nl = nl
        self.pots = pots
        self.save_decks = save_decks
        self._op = None
        self._ac = None
        self._tran = {}
        self.decks = {}

    def _save(self, fname, deck):
        self.decks[fname] = deck
        if self.save_decks:
            DECK_DIR.mkdir(parents=True, exist_ok=True)
            (DECK_DIR / fname).write_text(deck, encoding="utf-8", newline="\n")

    # -- op + ac
    def _run_op_ac(self):
        deck = build_op_ac_deck(self.nl, self.pots)
        self._save(f"{self.nl.id}.cir", deck)
        text, files = run_deck(deck)
        vals = parse_prints(text)
        self._op = vals
        if "ac.txt" in files:
            d = files["ac.txt"]
            f = d[:, 0]
            h = d[:, 1] + 1j * d[:, 2]
            self._ac = (f, h)

    @property
    def op(self):
        if self._op is None:
            self._run_op_ac()
        return self._op

    @property
    def ac_data(self):
        if self._op is None:
            self._run_op_ac()
        if self._ac is None:
            raise ValueError("no AC data (circuit has no *@ in / *@ out)")
        return self._ac

    def v(self, node):
        node = node.lower()
        if node in ("0", "gnd"):
            return 0.0
        op = self.op
        if node in op:
            return op[node]
        if f"v({node})" in op:
            return op[f"v({node})"]
        raise KeyError(f"no node {node}")

    def branch(self, src):
        return self.op[f"{src.lower()}#branch"]

    def supply_current(self):
        return sum(-self.branch(s) for s in self.nl.supplies)

    def element_current(self, name):
        name = name.upper()
        toks = self.nl.elements.get(name)
        c = name[0]
        low = name.lower()
        if c in "VLE" and f"{low}#branch" in self.op:
            return self.op[f"{low}#branch"]
        if c == "D":
            return self.op[f"@{low}[id]"]
        if c == "Q":
            return self.op[f"@{low}[ic]"]
        if c in "JM":
            return self.op[f"@{low}[id]"]
        if c == "R" and toks:
            return (self.v(toks[1]) - self.v(toks[2])) / parse_value(toks[3])
        raise KeyError(f"cannot measure current of {name}")

    def element_power(self, name):
        name = name.upper()
        toks = self.nl.elements[name]
        c = name[0]
        if c == "R":
            vr = self.v(toks[1]) - self.v(toks[2])
            return vr * vr / parse_value(toks[3])
        if c == "D":
            return self.element_current(name) * (self.v(toks[1]) - self.v(toks[2]))
        if c == "Q":
            ic = self.op[f"@{name.lower()}[ic]"]
            ib = self.op[f"@{name.lower()}[ib]"]
            vc, vb, ve = (self.v(t) for t in toks[1:4])
            return ic * (vc - ve) + ib * (vb - ve)
        if c in "JM":
            vd, vg, vs = (self.v(t) for t in toks[1:4])
            return self.element_current(name) * (vd - vs)
        raise KeyError(f"cannot measure power of {name}")

    def device_info(self):
        info = {}
        for name, toks in self.nl.elements.items():
            low = name.lower()
            c = name[0]
            if c == "Q":
                model = toks[4].upper()
                typ = PARTS.get(model, {}).get("type", "NPN")
                ic = self.op[f"@{low}[ic]"]
                ib = self.op[f"@{low}[ib]"]
                vc, vb, ve = (self.v(t) for t in toks[1:4])
                vce = vc - ve
                sgn = 1 if typ == "NPN" else -1
                if abs(ic) < 1e-6:
                    region = "cutoff"
                elif sgn * vce < 0.2:
                    region = "saturation"
                else:
                    region = "active"
                info[name] = {"type": typ, "ic": ic, "ib": ib, "vbe": vb - ve, "vbc": vb - vc,
                              "vce": vce, "region": region}
            elif c == "J":
                model = toks[4].upper()
                vto = PARTS[model]["params"]["VTO"] if model in PARTS else -2.0
                vd, vg, vs = (self.v(t) for t in toks[1:4])
                vgs, vds = vg - vs, vd - vs
                if vgs < vto:
                    region = "cutoff"
                elif vds < vgs - vto:
                    region = "triode"
                else:
                    region = "saturation"
                info[name] = {"type": "NJF", "id": self.op[f"@{low}[id]"], "vgs": vgs, "vds": vds,
                              "region": region}
            elif c == "M":
                model = toks[_mos_model_index(toks)].upper()
                typ = PARTS.get(model, {}).get("type", "NMOS")
                vto = PARTS[model]["params"]["VTO"] if model in PARTS else 0.0
                tp = 1.0 if typ == "NMOS" else -1.0
                vd, vg, vs = (self.v(t) for t in toks[1:4])
                vgs, vds = vg - vs, vd - vs
                vgst = tp * (vgs - vto)
                if vgst <= 0:
                    region = "cutoff"
                elif tp * vds < vgst:
                    region = "triode"
                else:
                    region = "saturation"
                info[name] = {"type": typ, "id": self.op[f"@{low}[id]"], "vgs": vgs, "vds": vds,
                              "region": region}
            elif c == "D":
                info[name] = {"id": self.op[f"@{low}[id]"], "vd": self.v(toks[1]) - self.v(toks[2])}
            elif c == "X":
                model = toks[-1].upper()
                p = PARTS[model]["params"]
                vout = self.v(toks[5])
                vcc, vee = self.v(toks[3]), self.v(toks[4])
                sat = vout > vcc - p["VSAT_HI"] - 0.05 or vout < vee + p["VSAT_LO"] + 0.05
                info[name] = {"vout": vout, "saturated": bool(sat)}
        return info

    # -- ac helpers
    def gain_db(self, f0):
        f, h = self.ac_data
        return interp_logf(f, 20 * np.log10(np.abs(h)), f0)

    def phase(self, f0):
        f, h = self.ac_data
        ph = np.degrees(np.unwrap(np.angle(h)))
        return interp_logf(f, ph, f0)

    def _crossings(self):
        f, h = self.ac_data
        g = 20 * np.log10(np.abs(h))
        k = int(np.argmax(g))
        gmax = g[k]
        thr = gmax - 3.0
        lf = np.log10(f)
        lo = hi = None
        for i in range(k, 0, -1):
            if g[i - 1] < thr <= g[i]:
                t = (thr - g[i - 1]) / (g[i] - g[i - 1])
                lo = 10 ** (lf[i - 1] + t * (lf[i] - lf[i - 1]))
                break
        for i in range(k, len(g) - 1):
            if g[i] >= thr > g[i + 1]:
                t = (g[i] - thr) / (g[i] - g[i + 1])
                hi = 10 ** (lf[i] + t * (lf[i + 1] - lf[i]))
                break
        return lo, hi, float(gmax), float(f[k])

    def f3db_low(self):
        return self._crossings()[0]

    def f3db_high(self):
        return self._crossings()[1]

    def fc(self):
        lo, hi, _, _ = self._crossings()
        f, h = self.ac_data
        g = np.abs(h)
        return hi if g[0] >= g[-1] else lo

    def peak_db(self):
        return self._crossings()[2]

    def f_peak(self):
        return self._crossings()[3]

    def q(self):
        lo, hi, _, fpk = self._crossings()
        if lo is None or hi is None:
            return None
        return fpk / (hi - lo)

    # -- tran helpers
    def tran(self, f, amp):
        key = (float(f), float(amp))
        if key not in self._tran:
            deck = build_tran_deck(self.nl, f, amp, self.pots)
            fname = f"{self.nl.id}_tran_{f:g}Hz_{amp:g}V.cir"
            if self.save_decks and len(self._tran) < 6:
                self._save(fname, deck)
            _, files = run_deck(deck)
            d = files["tran.txt"]
            self._tran[key] = analyse_periodic(d[:, 0], d[:, 1], f, ncyc=tran_times(f)[2])
        return self._tran[key]

    def thd(self, f, amp):
        return self.tran(f, amp)["thd_pct"]

    def vout_peak(self, f, amp):
        return self.tran(f, amp)["vout_peak"]

    def gain_tran(self, f, amp):
        return 20 * math.log10(max(self.tran(f, amp)["vout_peak"], 1e-15) / amp)

    def headroom(self, f, lo=0.005, hi=5.0, steps=9):
        if self.thd(f, lo) > 1.0:
            return lo
        if self.thd(f, hi) <= 1.0:
            return hi
        for _ in range(steps):
            mid = math.sqrt(lo * hi)
            if self.thd(f, mid) > 1.0:
                hi = mid
            else:
                lo = mid
        return lo


def analyse_periodic(t, v, f, ncyc=10, nh=15, n_per_cycle=256):
    T = 1.0 / f
    t1 = t[-1]
    t0 = t1 - ncyc * T
    n = ncyc * n_per_cycle
    ts = t0 + np.arange(n) * (ncyc * T / n)
    vs = np.interp(ts, t, v)
    mean = float(np.mean(vs))
    w = 2 * math.pi * f * (ts - t0)
    amps = []
    for k in range(1, nh + 1):
        c = 2.0 / n * np.sum((vs - mean) * np.exp(-1j * k * w))
        amps.append(abs(c))
    h1 = amps[0]
    thd = 100.0 * math.sqrt(sum(a * a for a in amps[1:])) / h1 if h1 > 0 else float("inf")
    return {"thd_pct": thd, "vout_peak": float(np.max(np.abs(vs - mean))), "h": amps,
            "mean": mean, "rms": float(np.sqrt(np.mean((vs - mean) ** 2)))}


def analyse_oscillator(t, v, settle_frac=0.5):
    m = t >= t[-1] * settle_frac
    tt, vv = t[m], v[m]
    mean = float(np.mean(vv))
    x = vv - mean
    idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    tc = tt[idx] - x[idx] * (tt[idx + 1] - tt[idx]) / (x[idx + 1] - x[idx])
    if len(tc) < 3:
        return {"oscillates": False, "freq_hz": None, "vpeak": float(np.max(np.abs(x))), "thd_pct": None}
    period = float(np.mean(np.diff(tc)))
    fosc = 1.0 / period
    k = min(10, len(tc) - 1)
    tw0, tw1 = tc[-1 - k], tc[-1]
    res = analyse_periodic(t[t <= tw1], v[t <= tw1], fosc, ncyc=k)
    return {"oscillates": True, "freq_hz": fosc, "vpeak": res["vout_peak"], "thd_pct": res["thd_pct"],
            "dc": mean}


# --------------------------------------------------------------------------- specs

_SPEC_RE = re.compile(
    r"^\s*(?P<metric>[a-z_0-9]+)\s*(?:\((?P<args>[^)]*)\))?\s*"
    r"(?:(?P<op><=|>=|==~|==|<|>)\s*(?P<val>\S+)|in\s+(?P<lo>[^.\s][^\s]*?)\.\.(?P<hi>\S+))\s*(?P<unit>\S*)\s*$",
    re.I)


def parse_spec(s: str):
    m = _SPEC_RE.match(s)
    if not m:
        raise ValueError(f"cannot parse spec: {s!r}")
    d = m.groupdict()
    args = [a.strip() for a in (d["args"] or "").split(",") if a.strip()]
    return d["metric"].lower(), args, d["op"], d["val"], d["lo"], d["hi"], d["unit"]


SPEC_UNITS = {"gain_db": "dB", "phase": "deg", "f3db_low": "Hz", "f3db_high": "Hz", "fc": "Hz",
              "peak_f": "Hz", "peak_db": "dB", "v": "V", "i": "A", "p": "W", "supply_current": "A",
              "thd": "%", "vout_peak": "V", "gain_tran": "dB", "headroom": "V", "bias": "",
              "region": "", "battery_hours": "h", "thd_gv": "%", "gain_gv": "dB"}


def guitar_variant(nl: "Netlist", vol: float) -> "Netlist":
    """The same netlist driven through the guitar model of engine/guitar.py
    (single coil 2.5 H / 6.5 k / 120 p, 500 k log volume pot, 300 p cable)."""
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from engine.netlist import pot_fraction
    frac = pot_fraction(vol, "log")
    out = []
    for line in nl.text.splitlines():
        toks = line.split()
        if toks and nl.input and toks[0].upper() == nl.input.upper():
            hot, ref = toks[1], toks[2]
            out.append(" ".join([toks[0], "gtr_emf", ref] + toks[3:]))
            out += [f"LGTR_PU gtr_emf gtr_l 2.5", f"RGTR_PU gtr_l gtr_hot 6.5k",
                    f"CGTR_PU gtr_hot {ref} 120p",
                    f"RGTR_VOLA gtr_hot {hot} {max(1.0, 500e3 * (1 - frac)):.6g}",
                    f"RGTR_VOLB {hot} {ref} {max(1.0, 500e3 * frac):.6g}",
                    f"CGTR_CABLE {hot} {ref} 300p"]
        else:
            out.append(line)
    DECK_DIR.mkdir(parents=True, exist_ok=True)
    path = DECK_DIR / f"_gv_{nl.id}_{vol:g}.cir"
    path.write_text(chr(10).join(out) + chr(10), encoding="utf-8", newline=chr(10))
    g = Netlist(path)
    g.id = f"{nl.id}_gv{vol:g}"
    return g


def measure_metric(meas: Measured, metric, args):
    a = args
    if metric == "gain_db":
        return meas.gain_db(parse_value(a[0]))
    if metric == "phase":
        return meas.phase(parse_value(a[0]))
    if metric == "f3db_low":
        return meas.f3db_low()
    if metric == "f3db_high":
        return meas.f3db_high()
    if metric == "fc":
        return meas.fc()
    if metric == "peak_db":
        return meas.peak_db()
    if metric == "peak_f":
        return meas.f_peak()
    if metric == "v":
        return meas.v(a[0])
    if metric == "i":
        return meas.element_current(a[0])
    if metric == "p":
        return meas.element_power(a[0])
    if metric == "supply_current":
        return meas.supply_current()
    if metric == "battery_hours":
        return parse_value(a[0]) * 1e-3 / meas.supply_current()
    if metric == "thd":
        return meas.thd(parse_value(a[0]), parse_value(a[1]))
    if metric == "vout_peak":
        return meas.vout_peak(parse_value(a[0]), parse_value(a[1]))
    if metric == "gain_tran":
        return meas.gain_tran(parse_value(a[0]), parse_value(a[1]))
    if metric == "headroom":
        return meas.headroom(parse_value(a[0]))
    if metric in ("thd_gv", "gain_gv"):
        vol = float(a[2]) if len(a) > 2 else 1.0
        sub = Measured(guitar_variant(meas.nl, vol), meas.pots)
        f, amp = parse_value(a[0]), parse_value(a[1])
        return sub.thd(f, amp) if metric == "thd_gv" else sub.gain_tran(f, amp)
    if metric in ("bias", "region"):  # bias is an alias of region
        name = a[0].upper()
        return meas.device_info()[name]["region"]
    raise ValueError(f"unknown metric {metric}")


def evaluate_spec(meas: Measured, s: str) -> dict:
    metric, args, op, val, lo, hi, unit = parse_spec(s)
    measured = measure_metric(meas, metric, args)
    if isinstance(measured, str):
        target = val.lower()
        ok = measured == target
        return {"spec": s, "measured": measured, "target": target, "pass": bool(ok), "unit": ""}
    if measured is None:
        return {"spec": s, "measured": None, "pass": False, "unit": SPEC_UNITS.get(metric, "")}
    if op:
        t = parse_value(val)
        ok = {"<": measured < t, "<=": measured <= t, ">": measured > t, ">=": measured >= t,
              "==~": abs(measured - t) <= 0.05 * abs(t), "==": measured == t}[op]
        margin = {"<": t - measured, "<=": t - measured, ">": measured - t, ">=": measured - t,
                  "==~": 0.05 * abs(t) - abs(measured - t), "==": -abs(measured - t)}[op]
    else:
        l, h = parse_value(lo), parse_value(hi)
        ok = l <= measured <= h
        margin = min(measured - l, h - measured)
    return {"spec": s, "measured": _round(measured), "pass": bool(ok), "margin": _round(margin),
            "unit": SPEC_UNITS.get(metric, "")}


def _round(x, sig=6):
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_round(v, sig) for v in x]
    if isinstance(x, dict):
        return {k: _round(v, sig) for k, v in x.items()}
    if x == 0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{sig}g}")


# --------------------------------------------------------------------------- whole-circuit record

def library_files():
    return sorted(LIB_DIR.glob("*.cir"))


def measure_circuit(nl: Netlist, save_decks=False, sweeps=True) -> dict:
    meas = Measured(nl, save_decks=save_decks)
    specs = [evaluate_spec(meas, s) for s in nl.specs]
    rec = {
        "id": nl.id,
        "file": f"library/{nl.path.name}",
        "title": nl.title,
        "description": nl.description,
        "category": CATALOG.get(nl.id, ("misc", []))[0],
        "tags": CATALOG.get(nl.id, ("misc", []))[1],
        "input": nl.input,
        "output": nl.output,
        "supplies": nl.supplies,
        "pots": [{k: p[k] for k in ("name", "a", "w", "b", "value", "taper", "pos", "label")}
                 for p in nl.pots.values()],
        "specs": specs,
    }
    m = {}
    nodes = {k: v for k, v in meas.op.items()
             if "#" not in k and not k.startswith("@") and not k.startswith("i(") and "." not in k}
    m["op"] = {"nodes": _round(nodes), "devices": _round(meas.device_info())}
    if nl.supplies:
        m["op"]["supply_current_a"] = _round(meas.supply_current())
    if nl.input and nl.output:
        lo, hi, gmax, fpk = meas._crossings()
        m["ac"] = {
            "gain_db": {f"{f:g}": _round(meas.gain_db(f), 5) for f in REPORT_FREQS},
            "phase_deg": {f"{f:g}": _round(meas.phase(f), 5) for f in REPORT_FREQS},
            "peak_db": _round(gmax, 5), "f_peak_hz": _round(fpk, 5),
            "f3db_low_hz": _round(lo, 5), "f3db_high_hz": _round(hi, 5),
        }
        trans = []
        for (f, a), r in sorted(meas._tran.items()):
            trans.append({"f_hz": f, "vin_peak": a, "thd_pct": _round(r["thd_pct"], 5),
                          "vout_peak": _round(r["vout_peak"], 5),
                          "gain_tran_db": _round(20 * math.log10(max(r["vout_peak"], 1e-15) / a), 5),
                          "h2_over_h1": _round(r["h"][1] / r["h"][0], 5),
                          "tstop_s": _round(tran_times(f)[1]), "window_cycles": tran_times(f)[2]})
        if trans:
            m["tran"] = trans
        if sweeps and nl.pots:
            sw = {}
            for pname in nl.pots:
                sw[pname] = {}
                for pos in (0.0, 1.0):
                    mm = Measured(nl, pots={pname: pos})
                    sw[pname][f"{pos:g}"] = {"gain_db_100": _round(mm.gain_db(100.0), 5),
                                             "gain_db_1k": _round(mm.gain_db(1000.0), 5),
                                             "gain_db_10k": _round(mm.gain_db(10000.0), 5),
                                             "peak_db": _round(mm.peak_db(), 5),
                                             "f_peak_hz": _round(mm.f_peak(), 5)}
            m["pot_sweep"] = sw
    else:
        deck = build_osc_deck(nl)
        meas._save(f"{nl.id}_osc.cir", deck)
        _, files = run_deck(deck)
        d = files["tran.txt"]
        m["oscillator"] = _round(analyse_oscillator(d[:, 0], d[:, 1]), 6)
        m["oscillator"]["tstop_s"] = 0.2
    rec["measured"] = m
    rec["all_specs_pass"] = all(s["pass"] for s in specs)
    return rec


def write_index(records):
    index = {
        "version": 1,
        "generated_by": "tests/test_library_ngspice.py",
        "simulator": "ngspice-42 (batch mode, 27 C, trapezoidal)",
        "conventions": {
            "pot_taper": "fraction a->w: lin=pos, log=(11**pos-1)/10, revlog=1-(11**(1-pos)-1)/10; halves >= 1 ohm",
            "ac_sweep": f"{AC_FSTART:g} Hz to {AC_FSTOP:g} Hz, {AC_PPD} points/decade, transfer v(out)/v(in source)",
            "f3db": "first -3 dB crossings below/above the sweep maximum; null if none inside the sweep",
            "tran": "sine from the DC operating point with 5 ms linear fade-in, max step T/200, "
                    "settle max(60 ms, 8 cycles), window max(4, round(0.02 f)) cycles",
            "thd": "100*sqrt(H2^2+...+H15^2)/H1 (Fourier projection of the window minus its mean)",
            "vout_peak": "max |v(out) - window mean|",
            "headroom": "largest vin_peak with THD <= 1 %, geometric bisection 5 mV..5 V, 9 steps",
            "supply_current": "sum of DC currents delivered by the *@ supply sources",
            "opamp_macromodel": "RIN; VCCS A0/1Meg into 1Meg||C (pole GBW/A0); gain node clamped to "
                                "[VEE+VSAT_LO, VCC-VSAT_HI] by a 1000 S softplus clamp (10 mV); "
                                "unity VCVS + ROUT; IQ from VCC to VEE",
        },
        "circuits": records,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- pytest

if pytest is not None:
    _needs_ngspice = pytest.mark.skipif(find_ngspice() is None, reason="ngspice not available")

    @_needs_ngspice
    @pytest.mark.parametrize("path", library_files(), ids=lambda p: p.stem)
    def test_library_circuit_passes_specs_in_ngspice(path):
        nl = Netlist(path)
        assert nl.title and nl.title != nl.id, "missing *@ title"
        assert nl.description, "missing description comment line"
        assert nl.specs, "no *@ spec lines"
        meas = Measured(nl)
        assert meas.op, "operating point did not converge"
        failures = [r for r in (evaluate_spec(meas, s) for s in nl.specs) if not r["pass"]]
        assert not failures, f"{nl.id}: specs failing in ngspice: {failures}"

    @_needs_ngspice
    @pytest.mark.parametrize("path", library_files(), ids=lambda p: p.stem)
    def test_index_matches_ngspice(path):
        """index.json op voltages and AC gains must match a fresh ngspice run."""
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        rec = next(c for c in index["circuits"] if c["id"] == path.stem)
        nl = Netlist(path)
        meas = Measured(nl)
        for node, v in rec["measured"]["op"]["nodes"].items():
            assert abs(meas.v(node) - v) <= max(1e-6, 1e-4 * abs(v)), node
        if "ac" in rec["measured"]:
            for f, g in rec["measured"]["ac"]["gain_db"].items():
                assert abs(meas.gain_db(float(f)) - g) < 1e-3, f

    def test_library_names_are_generic():
        banned = ["tube screamer", "screamer", "fuzz face", "big muff", "muff", "rat ", "klon", "cry baby",
                  "crybaby", "dunlop", "ibanez", "boss ", "electro-harmonix", "proco", "vox", "marshall",
                  "fender", "rangemaster", "octavia", "dallas", "arbiter", "centaur", "mxr", "jhs"]
        for p in library_files():
            text = p.read_text(encoding="utf-8").lower()
            for b in banned:
                assert b not in text, f"{p.name} mentions {b!r}"

    def test_parts_data_complete():
        need = ["1N4148", "1N914", "1N34A", "1N5817", "1N4001", "LED_RED", "LED_GREEN", "BAT41",
                "2N3904", "2N2222", "BC549C", "2N5088", "MPSA18", "2N3906", "BC559C", "AC128",
                "J201", "2N5457", "MPF102", "TL072", "TL071", "NE5532", "LM358", "RC4558", "OPAMP_IDEAL",
                "2N7000", "BS170", "BS250", "IRF510", "IRF9510"]
        for n in need:
            p = PARTS[n]
            for k in ("type", "params", "rating", "desc", "package", "pins", "source"):
                assert k in p, (n, k)
            if p["type"] == "OPAMP":
                assert set(p["params"]) == {"A0", "GBW", "RIN", "ROUT", "VSAT_HI", "VSAT_LO", "IQ"}, n


# --------------------------------------------------------------------------- CLI

def _report(rec):
    m = rec["measured"]
    print(f"\n== {rec['id']}: {rec['title']}  [{'PASS' if rec['all_specs_pass'] else 'FAIL'}]")
    op = m["op"]
    print("   op:", ", ".join(f"{k}={v:.4g}" for k, v in op["nodes"].items()))
    for d, info in op["devices"].items():
        print(f"   {d}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in info.items()))
    if "supply_current_a" in op:
        print(f"   supply current {op['supply_current_a'] * 1e3:.3f} mA")
    if "ac" in m:
        a = m["ac"]
        print("   gain_db:", a["gain_db"], f" peak {a['peak_db']:.2f} dB @ {a['f_peak_hz']:.0f} Hz,"
              f" f3db {a['f3db_low_hz']} .. {a['f3db_high_hz']}")
    for t in m.get("tran", []):
        print(f"   tran {t['f_hz']:g} Hz {t['vin_peak']:g} V: THD {t['thd_pct']:.2f} %, "
              f"vout_peak {t['vout_peak']:.3f} V, gain {t['gain_tran_db']:.2f} dB, H2/H1 {t['h2_over_h1']:.3f}")
    if "pot_sweep" in m:
        for p, s in m["pot_sweep"].items():
            print(f"   pot {p}: " + "; ".join(f"pos {k}: {v['gain_db_100']:.1f}/{v['gain_db_1k']:.1f}/{v['gain_db_10k']:.1f} dB@100/1k/10k, peak {v['peak_db']:.2f} dB"
                                               f"@{v['f_peak_hz']:.0f}" for k, v in s.items()))
    if "oscillator" in m:
        print("   osc:", m["oscillator"])
    for s in rec["specs"]:
        print(f"   [{'ok' if s['pass'] else 'XX'}] {s['spec']}  -> {s['measured']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--write-index", action="store_true")
    ap.add_argument("--no-sweeps", action="store_true")
    args = ap.parse_args(argv)
    if find_ngspice() is None:
        print("ngspice not available", file=sys.stderr)
        return 2
    files = [p for p in library_files() if not args.ids or p.stem in args.ids]
    records = []
    if args.write_index and args.ids:
        print("--write-index needs the whole library (no ids)", file=sys.stderr)
        return 1
    if args.write_index:
        DECK_DIR.mkdir(parents=True, exist_ok=True)
        for old in DECK_DIR.glob("*.cir"):
            old.unlink()
    for p in files:
        nl = Netlist(p)
        rec = measure_circuit(nl, save_decks=args.write_index, sweeps=not args.no_sweeps)
        records.append(rec)
        _report(rec)
    if args.write_index:
        write_index(records)
        print(f"\nwrote {INDEX_PATH}")
    bad = [r["id"] for r in records if not r["all_specs_pass"]]
    print("\nFAILING:", bad if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
