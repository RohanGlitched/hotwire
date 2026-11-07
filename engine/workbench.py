"""The bench: high-level operations the tools, the slash command and the Desktop API share.

Every operation takes a netlist (text), a stored design id, or a library id / file
path, does its work, saves artifacts under the design's version folder, and returns
a plain dict (JSON-safe) with a short ``summary`` line and a ``media`` list of files.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import erc, montecarlo, parts, stress
from .export import compile_json, kicad_netlist, netlist_text, spice_deck
from .netlist import Circuit, NetlistError, format_value, parse
from .specs import Evaluator, evaluate, parse_spec
from .store import Store, slugify

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "library"

__all__ = ["Bench", "resolve_source", "library_index"]


# --------------------------------------------------------------------------- helpers

def library_index() -> list:
    idx = LIBRARY_DIR / "index.json"
    if idx.exists():
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
            return data["circuits"] if isinstance(data, dict) and "circuits" in data else data
        except Exception:
            pass
    out = []
    for f in sorted(LIBRARY_DIR.glob("*.cir")):
        try:
            c = parse(f.read_text(encoding="utf-8"))
            out.append({"id": f.stem, "file": f.name, "title": c.title, "pots": list(c.pots), "specs": c.specs})
        except Exception:
            continue
    return out


def _library_file(key: str) -> Path | None:
    def norm(s):
        return "".join(ch for ch in s.lower() if ch.isalnum())
    k = norm(key.removesuffix(".cir"))
    if not k:
        return None
    for f in LIBRARY_DIR.glob("*.cir"):
        if norm(f.stem) == k:
            return f
    return None


def resolve_source(store: Store, netlist=None, design=None, path=None):
    """Return (netlist_text, design_id_or_None, origin_label)."""
    if netlist and str(netlist).strip():
        return str(netlist), None, "netlist"
    if design:
        if not store.exists(design):
            raise NetlistError(0, f"no design named '{design}' (call circuit_check with a netlist first)")
        return store.latest_netlist(design), design, f"design {design}"
    if path:
        lib = _library_file(path)
        if lib:
            return lib.read_text(encoding="utf-8"), None, f"library {lib.stem}"
        p = Path(path).expanduser()
        if p.suffix.lower() in (".cir", ".net", ".sp", ".spice") and p.is_file():
            if p.stat().st_size > 512_000:
                raise NetlistError(0, f"{p.name} is too large to be a netlist")
            return p.read_text(encoding="utf-8", errors="replace"), None, f"file {p.name}"
        raise NetlistError(0, f"'{path}' is neither a library id nor a readable .cir file")
    raise NetlistError(0, "give a netlist, a design id, or a library id/path")


def _r(x, sig=4):
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not math.isfinite(x) else float(f"{x:.{sig}g}")
    return x


def _audio_source(audio, name: str, sr: int, seconds: float, raw: bool = False) -> np.ndarray:
    """A named demo source or a user WAV, mono at ``sr``, ``seconds`` long, peak 1."""
    name = str(name or "guitar_riff")
    if name.lower().endswith(".wav"):
        p = Path(name).expanduser()
        if not p.is_file():
            raise NetlistError(0, f"can't find the WAV file {name}")
        x, file_sr = audio.read_wav(p)
        x = np.asarray(x, dtype=float)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if file_sr != sr:
            n = int(round(len(x) * sr / file_sr))
            x = np.interp(np.arange(n) / sr, np.arange(len(x)) / file_sr, x)
        n = int(seconds * sr)
        x = np.tile(x, int(np.ceil(n / max(len(x), 1))))[:n] if len(x) < n else x[:n]
    elif hasattr(audio, "source"):
        try:
            x = np.asarray(audio.source(name, sr, seconds, raw=raw), dtype=float)
        except TypeError:
            x = np.asarray(audio.source(name, sr, seconds), dtype=float)
    else:
        known = ("guitar_riff", "guitar_chords", "bass_line", "drum_loop", "sine_sweep")
        if name not in known:
            raise NetlistError(0, f"unknown source '{name}' (use {', '.join(known)} or a .wav path)")
        x = np.asarray(audio.generate(name, seconds=seconds, sr=sr), dtype=float)
    pk = float(np.max(np.abs(x))) or 1.0
    return x / pk


def _try_png(svg_path: Path) -> Path | None:
    """PNG twin of an SVG for chat apps that can't show SVG (uses resvg when present)."""
    try:
        import resvg_py  # bundled with Hermes
    except Exception:
        return None
    try:
        import re
        svg = svg_path.read_text(encoding="utf-8")
        # resvg doesn't resolve CSS variables: use each var()'s fallback colour
        for _ in range(3):
            svg = re.sub(r"var\(\s*--[\w-]+\s*,\s*([^()]+?|[a-z]+\([^()]*\))\s*\)", r"\1", svg)
        svg = svg.replace("currentColor", "#1d1d1f")
        png = resvg_py.svg_to_bytes(svg_string=svg, width=1400, background="white")
        out = svg_path.with_suffix(".png")
        out.write_bytes(bytes(png))
        return out
    except Exception:
        return None


# --------------------------------------------------------------------------- bench

class Bench:
    def __init__(self, store: Store | None = None, sample_rate: int = 48000, emit=None):
        self.store = store or Store()
        self.sr = sample_rate
        self.emit = emit or (lambda event, payload: None)

    # -- shared ------------------------------------------------------------
    def _load(self, netlist=None, design=None, path=None, pots=None):
        text, did, origin = resolve_source(self.store, netlist, design, path)
        if did is None and design and self.store.exists(design):
            did = design  # a new netlist for an existing design: becomes its next version
        c = parse(text)
        if pots:
            c = c.with_pots(pots)
        return text, c, did, origin

    def _save(self, c: Circuit, text: str, did, name, note) -> tuple:
        if did is not None and self.store.exists(did):
            try:
                if self.store.latest_netlist(did).strip() == text.strip() and note in ("audition", "build") or                         (note.startswith("export") and self.store.latest_netlist(did).strip() == text.strip()):
                    v = self.store.meta(did)["latest"]
                    return did, v, self.store.version_dir(did, v), False
            except Exception:
                pass
        if did is None:
            found = self.store.find_by_netlist(text)
            if found:
                did, v = found
                return did, v, self.store.version_dir(did, v), False
            # same name as an existing design: it's the next version of that design
            for key in (name, c.title):
                if key and self.store.exists(slugify(key)):
                    did = slugify(key)
                    break
        new_text = netlist_text(c) if c.source == text else text
        did, v, vdir = self.store.save_version(did, name or c.title, new_text, note or "")
        return did, v, vdir, True

    def _op_summary(self, op, c: Circuit) -> dict:
        nodes = {n: _r(op.v.get(n)) for n in c.nodes()}
        devs = {}
        for name, d in (op.devices or {}).items():
            devs[name] = {k: (_r(v) if not isinstance(v, str) else v) for k, v in d.items()}
        return {"converged": bool(getattr(op, "converged", True)), "method": getattr(op, "method", ""),
                "nodes": nodes, "devices": devs}

    # -- check -------------------------------------------------------------
    def check(self, netlist=None, design=None, path=None, name=None, pots=None, specs=None,
              note="", save=True) -> dict:
        t0 = time.time()
        text, c, did, origin = self._load(netlist, design, path, pots)
        issues = erc.check(c)
        errors = [i for i in issues if i["level"] == "error"]
        result = {"title": c.title, "origin": origin, "erc": issues, "media": []}
        if save:
            did, v, vdir, fresh = self._save(c, text, did, name, note)
            result.update(design=did, version=v)
        else:
            vdir = None
        # schematic first: useful even when the circuit can't be solved
        if vdir is not None:
            self._schematic(c, vdir, result)
        if errors:
            result["verdict"] = "fail"
            result["summary"] = f"{len(errors)} wiring error(s) before simulation: " + "; ".join(e["message"] for e in errors[:3])
            self._finish(result, vdir, t0)
            return result
        ev = Evaluator(c)
        try:
            op = ev.op
        except Exception as e:
            result["verdict"] = "fail"
            result["summary"] = f"the DC operating point didn't solve: {e}"
            self._finish(result, vdir, t0)
            return result
        result["op"] = self._op_summary(op, c)
        op_issues = erc.check_op(c, op)
        if op_issues:
            issues = issues + op_issues
            result["erc"] = issues
        if any(i["level"] == "error" for i in op_issues):
            result["verdict"] = "fail"
            result["summary"] = "; ".join(i["message"] for i in op_issues if i["level"] == "error")
            result["summary"] = result["summary"][:1].upper() + result["summary"][1:]
            self._finish(result, vdir, t0)
            return result
        stab = self._stability(c)
        result["stability"] = stab
        ss_specs = [t for t in list(c.specs) + list(specs or []) if t.split("(")[0].strip().split(" ")[0].lower()
                    in ("gain_db", "phase", "fc", "f3db_low", "f3db_high", "peak_db", "peak_f", "zin")]
        if stab.get("state") in ("latches", "oscillates"):
            lvl = "error" if (stab["state"] == "latches" and ss_specs) else "warning"
            issues = issues + [{"level": lvl, "code": stab["state"], "message": stab["message"], "parts": [], "nodes": []}]
            result["erc"] = issues
        spec_res = [r.as_dict() for r in evaluate(c, specs, ev)]
        if stab.get("state") in ("latches", "oscillates"):
            # small-signal numbers describe an operating point the real circuit doesn't stay at
            for r in spec_res:
                if r.get("metric") in ("gain_db", "phase", "fc", "f3db_low", "f3db_high", "peak_db", "peak_f") and r["pass"]:
                    r["pass"] = False
                    r["explanation"] = (f"{r['explanation']}, but the circuit {stab['state']}, so this small-signal "
                                        "number doesn't describe what it actually does")
        result["specs"] = spec_res
        result["stress"] = stress.check(c, op)
        powered = c.supplies or any(e.kind == "V" and e.name != c.input and (e.value or 0) != 0 for e in c.elements)
        result["supply_ma"] = _r(ev.supply_current() * 1000, 3) if powered else None
        bias_warn = []
        for nm, d in (op.devices or {}).items():
            reg = str(d.get("region", "")).lower()
            dk = nm.rsplit(".", 1)[-1][:1]      # X1.Q1 from a .subckt is still a Q; MOSFETs amplify in saturation
            if (reg in ("saturation", "cutoff") and dk == "Q") or (reg in ("triode", "cutoff") and dk == "M"):
                bias_warn.append({"part": nm, "region": reg,
                                  "message": f"{nm} sits in {reg} at idle; fine for a switch or a fuzz, wrong for a linear amplifier"})
            elif reg == "reverse" and dk == "Q":
                issues = issues + [{"level": "error", "code": "transistor_reversed", "parts": [nm], "nodes": [],
                                    "message": f"{nm} runs in reverse-active mode: it's wired backwards (collector and "
                                               "emitter swapped, or an NPN used where a PNP belongs)"}]
                result["erc"] = issues
        result["bias"] = bias_warn
        # plots
        if vdir is not None and c.input and c.output:
            try:
                from .plot import bode_svg
                acr = ev.acr
                marks = []
                for r in spec_res:
                    if r["metric"] in ("gain_db", "fc", "f3db_low", "f3db_high") and r.get("measured") is not None:
                        s = parse_spec(r["spec"])
                        marks.append({"kind": r["metric"], "f": (float(parts_freq(s)) if r["metric"] == "gain_db" else r["measured"]),
                                      "lo": s.lo if s.op in (">", ">=", "in") else None,
                                      "hi": (s.hi if s.op == "in" else (s.lo if s.op in ("<", "<=") else None)),
                                      "measured": r["measured"], "pass": r["pass"]})
                svg = bode_svg(acr.f, acr.h, specs=marks, title=c.title)
                p = vdir / "bode.svg"
                p.write_text(svg, encoding="utf-8")
                result["media"].append(str(p))
            except Exception as e:
                result["plot_error"] = str(e)
        if not spec_res:
            issues = issues + [{"level": "warning", "code": "no_specs", "parts": [], "nodes": [],
                                "message": "no specs: the circuit solved, but nothing was measured against a target. "
                                           "Add '*@ spec' lines (gain, corner, current…) so a pass means something"}]
            result["erc"] = issues
        failed = [r for r in spec_res if not r["pass"]]
        if any(i["level"] == "error" for i in issues) and not failed:
            failed = [{"explanation": next(i["message"] for i in issues if i["level"] == "error")}]
        stress_err = [s for s in result["stress"] if s["level"] == "error"]
        stress_warn = [s for s in result["stress"] if s["level"] == "warning"]
        warn = [i for i in issues if i["level"] == "warning"]
        if failed or stress_err:
            result["verdict"] = "fail"
        elif stress_warn or warn or bias_warn:
            result["verdict"] = "pass_with_warnings"
        else:
            result["verdict"] = "pass"
        n_ok = sum(1 for r in spec_res if r["pass"])
        bits = [f"{n_ok}/{len(spec_res)} specs pass" if spec_res else "no specs given (add '*@ spec' lines)"]
        if failed:
            bits.append("failing: " + "; ".join(r["explanation"] for r in failed[:3]))
        if stress_err:
            bits.append("overstressed: " + "; ".join(s["message"] for s in stress_err[:2]))
        if result.get("supply_ma") is not None:
            bits.append(f"draws {result['supply_ma']:g} mA")
        result["summary"] = ". ".join(b[:1].upper() + b[1:] for b in bits)
        self._finish(result, vdir, t0)
        return result

    def _stability(self, c: Circuit) -> dict:
        """Kick the circuit with a tiny impulse and watch for 0.3 s: does it settle back to its
        operating point, run off to a rail and stay (latch), or keep growing (oscillate)?
        Capacitor initial conditions and the circuit's own time-varying sources are removed
        first, so start-up transients and LFOs aren't mistaken for instability."""
        if not c.input or not c.output:
            return {"state": "not_checked", "message": "no *@ in / *@ out to kick"}
        import copy
        from .analyses import process_audio
        q = copy.deepcopy(c)
        for e in q.elements:
            e.params.pop("ic", None)
            if e.kind in ("V", "I") and e.name != q.input:
                e.params.pop("sin", None)
                e.params.pop("pulse", None)
        sr = 24000
        x = np.zeros(int(0.3 * sr))
        x[1] = 1e-5
        try:
            y = np.asarray(process_audio(q, x, sr), dtype=float)
        except Exception as e:
            return {"state": "unknown", "message": f"stability run failed: {e}"}
        if not np.all(np.isfinite(y)):
            return {"state": "oscillates", "message": "the output runs away (non-finite) after a tiny kick"}
        # trapezoidal integration can leave a tiny sample-to-sample sign flip (numerical ringing at
        # Nyquist, not physics): average neighbouring samples so only real behaviour is judged
        y = 0.5 * (y[1:] + y[:-1])
        win = int(0.05 * sr)
        rms = [float(np.sqrt(np.mean(y[i:i + win] ** 2))) for i in range(0, len(y) - win + 1, win)]
        tail = y[-int(0.02 * sr):]
        peak = float(np.max(np.abs(tail)))
        if peak >= 0.3 and float(np.std(tail)) < 0.05 * peak:
            return {"state": "latches", "peak_v": round(peak, 3),
                    "message": f"the circuit latches: after a 10 µV kick its output runs {peak:.2f} V away from the "
                               "operating point and stays there (positive feedback): expected for a Schmitt trigger "
                               "or flip-flop, a fault in an amplifier"}
        growing = rms[-1] > 1e-6 and rms[-1] > 3.0 * max(rms[1], 1e-12) and rms[-1] >= rms[-2] * 0.9
        if peak >= 0.3 or growing:
            return {"state": "oscillates", "peak_v": round(float(np.max(np.abs(y))), 4),
                    "message": "the circuit oscillates: a 10 µV kick keeps growing instead of dying away "
                               "(expected for an oscillator, a fault in an amplifier)"}
        return {"state": "stable", "message": "settles back to its operating point after a small kick"}

    def _schematic(self, c, vdir, result):
        try:
            from .schematic import svg as schematic_svg
            p = vdir / "schematic.svg"
            p.write_text(schematic_svg(c), encoding="utf-8")
            result["media"].insert(0, str(p))
        except Exception as e:
            result["schematic_error"] = str(e)

    def _finish(self, result, vdir, t0):
        result["seconds"] = round(time.time() - t0, 2)
        if vdir is not None:
            pngs = []
            for m in list(result["media"]):
                if m.endswith(".svg"):
                    png = _try_png(Path(m))
                    if png:
                        pngs.append(str(png))
            result["png"] = pngs
            (vdir / "check.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
            self.store.update_version(result["design"], result["version"], verdict=result.get("verdict"),
                                      specs_pass=sum(1 for r in result.get("specs", []) if r["pass"]),
                                      specs_total=len(result.get("specs", [])))
            self.emit("design.updated", {"design": result["design"], "version": result["version"],
                                         "verdict": result.get("verdict"), "title": result.get("title")})

    # -- simulate ----------------------------------------------------------
    def simulate(self, analysis, netlist=None, design=None, path=None, pots=None, fstart=10, fstop=100e3,
                 freq=1000, amplitude=0.2, cycles=4, nodes=None) -> dict:
        text, c, did, origin = self._load(netlist, design, path, pots)
        out = {"title": c.title, "analysis": analysis, "media": []}
        outdir = self._scratch(did, c)
        if analysis == "op":
            from .analyses import op
            r = op(c)
            out["op"] = self._op_summary(r, c)
            out["currents"] = {k: _r(v) for k, v in r.i.items()}
            out["power"] = {k: _r(v) for k, v in r.power.items()}
            out["summary"] = ", ".join(f"v({n})={_r(v, 3)} V" for n, v in list(out["op"]["nodes"].items())[:8])
        elif analysis == "ac":
            from .analyses import ac
            from .plot import bode_svg
            r = ac(c, fstart=float(fstart), fstop=float(fstop), points_per_decade=40)
            mag = 20 * np.log10(np.maximum(np.abs(r.h), 1e-30))
            k = (np.unique(np.round(np.geomspace(1, len(r.f), 25)).astype(int) - 1)
                 if len(r.f) > 25 else range(len(r.f)))
            out["points"] = [{"f": _r(r.f[i]), "gain_db": _r(mag[i], 3), "phase": _r(float(np.angle(r.h[i], deg=True)), 3)} for i in k]
            ev = Evaluator(c)
            ev._ac = r
            lo, hi, peak, pf = ev.corners()
            out.update(peak_db=_r(peak, 3), peak_f=_r(pf), f3db_low=_r(lo), f3db_high=_r(hi))
            p = outdir / "ac.svg"
            p.write_text(bode_svg(r.f, r.h, title=c.title), encoding="utf-8")
            out["media"].append(str(p))
            out["summary"] = f"peak {peak:.1f} dB at {format_value(pf, 'Hz')}" + (f", -3 dB at {format_value(lo, 'Hz')}" if lo else "") + (f" and {format_value(hi, 'Hz')}" if hi else "")
        elif analysis == "tran":
            from .analyses import process_audio
            from .plot import scope_svg
            if not c.input or not c.output:
                raise NetlistError(0, "a transient run needs '*@ in' and '*@ out'")
            freq = float(freq or 1000)
            cycles = float(cycles or 4)
            if not 1.0 <= freq <= 50e3:
                raise NetlistError(0, f"freq must be between 1 Hz and 50 kHz for a transient run (got {freq:g} Hz)")
            if not 0.5 <= cycles <= 200:
                raise NetlistError(0, f"cycles must be between 0.5 and 200 (got {cycles:g})")
            sr = 192000 if freq > 2000 else (96000 if freq >= 50 else 48000)
            n = int(sr * max(cycles, 1) / freq)
            settle = int(sr * min(max(0.05, 6 / freq), 2.0))
            if n + settle > 1_200_000:
                raise NetlistError(0, "that run would need over a million time steps; use fewer cycles or a higher frequency")
            t = np.arange(settle + n) / sr
            x = amplitude * np.sin(2 * math.pi * freq * t) * np.minimum(1, t / 0.005)
            y = process_audio(c, x, sr)
            tt = t[settle:] - t[settle]
            p = outdir / "tran.svg"
            traces = {"in": x[settle:], "out": y[settle:]}
            if nodes:
                from .analyses import tran as _tran
                want = [n_.lower() for n_ in nodes if n_.lower() in c.nodes()][:3]
                if want:
                    tr = _tran(c, tstop=(settle + n) / sr, tstep=1 / sr, input_wave=x, nodes=want)
                    for n_ in want:
                        seg = np.asarray(tr.v[n_])[settle:settle + n]
                        traces[n_] = seg - float(np.mean(seg))
            p.write_text(scope_svg(tt, traces, title=c.title), encoding="utf-8")
            out["media"].append(str(p))
            ys = y[settle:]
            out.update(vin_peak=_r(amplitude), vout_peak=_r(float(np.max(np.abs(ys - np.mean(ys))))),
                       vout_max=_r(float(np.max(ys))), vout_min=_r(float(np.min(ys))))
            out["summary"] = f"{format_value(amplitude, 'V')} peak in → {format_value(out['vout_peak'], 'V')} peak out at {format_value(freq, 'Hz')}"
        else:
            raise NetlistError(0, "analysis must be op, ac or tran")
        return out

    def _scratch(self, did, c):
        if did:
            d = self.store.version_dir(did) / "runs"
        else:
            d = self.store.root.parent / "runs" / slugify(c.title)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- audition ----------------------------------------------------------
    def audition(self, netlist=None, design=None, path=None, name=None, pots=None, source="guitar_riff",
                 level=0.2, seconds=4.0, sweep=None, guitar_volume=None, pickup="single_coil") -> dict:
        from . import audio
        from .analyses import process_audio
        text, c, did, origin = self._load(netlist, design, path, pots)
        if not c.input or not c.output:
            raise NetlistError(0, "to hear a circuit it needs '*@ in <source>' and '*@ out <node>'")
        if any(i["level"] == "error" for i in erc.check(c)):
            raise NetlistError(0, "fix the wiring errors first (run circuit_check)")
        did, v, vdir, _ = self._save(c, text, did, name, "audition")
        # named guitar sources go through the modelled pickup; a user's WAV already contains
        # their pickup, so it only gets the guitar model when they ask for it
        guitarish = str(source).startswith("guitar") and c.element(c.input).kind == "V"
        sweep_guitar = bool(sweep and str(sweep.get("pot", "")).lower() in ("guitar", "guitar_volume", "volume_guitar"))
        if guitar_volume is None and (guitarish or sweep_guitar):
            guitar_volume = 1.0
        base_c = c
        if guitar_volume is not None:
            from .guitar import with_guitar
            c = with_guitar(c, float(guitar_volume), pickup or "single_coil")
        seconds = float(min(max(seconds or 4.0, 0.5), 8.0))
        sr = self.sr
        raw = guitar_volume is not None and guitarish
        dry = _audio_source(audio, source, sr, seconds, raw=raw)
        dry = np.asarray(dry, dtype=float)
        pk = float(np.max(np.abs(dry))) or 1.0
        x = dry / pk * float(level or 0.2)
        if guitar_volume is not None:
            # "dry" = what bypass sounds like: the same guitar through its cable into a 1 MΩ amp input
            from .guitar import with_guitar
            amp = parse("*@ in VIN\n*@ out amp\nVIN amp 0 DC 0 AC 1\nRAMP amp 0 1meg\n")
            bypass = np.asarray(process_audio(with_guitar(amp, float(guitar_volume), pickup or "single_coil"), x, sr), dtype=float)
        else:
            bypass = x
        t0 = time.time()
        if sweep_guitar:
            from .guitar import with_guitar
            y = self._sweep_render(base_c, x, sr, sweep,
                                   make=lambda pos: with_guitar(base_c, pos, pickup or "single_coil"))
        elif sweep and sweep.get("pot"):
            y = self._sweep_render(c, x, sr, sweep)
        else:
            y = process_audio(c, x, sr)
        render_s = time.time() - t0
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):
            raise NetlistError(0, "the simulation produced invalid samples (the circuit may oscillate or not converge)")
        # loudness-matched A/B so the comparison is about tone, not level
        def rms(a):
            return float(np.sqrt(np.mean(a * a))) or 1e-12
        out_rms, in_rms = rms(y), rms(bypass)
        target = 0.12
        wet = y / rms(y) * target
        dryn = bypass / rms(bypass) * target
        for arr in (wet, dryn):
            m = float(np.max(np.abs(arr)))
            if m > 0.98:
                arr *= 0.98 / m
        tag = "sweep-" + slugify(str(sweep.get("pot", ""))) if sweep else (Path(source).stem if str(source).endswith(".wav") else source)
        if guitar_volume is not None and not sweep_guitar and float(guitar_volume) != 1.0:
            tag += f"-gv{int(round(float(guitar_volume) * 100)):02d}"
        if pots:
            tag += "-" + "-".join(f"{k.lower()}{int(round(float(v) * 100)):02d}" for k, v in sorted(pots.items()))
        pd, pw = vdir / f"dry-{tag}.wav", vdir / f"circuit-{tag}.wav"
        audio.write_wav(pd, dryn, sr)
        audio.write_wav(pw, wet, sr)
        media = [str(pw), str(pd)]
        try:
            from .plot import spectrum_svg
            ps = vdir / f"spectrum-{tag}.svg"
            ps.write_text(spectrum_svg(y, sr, ref=bypass * (rms(y) / rms(bypass))), encoding="utf-8")
            media.append(str(ps))
        except Exception:
            pass
        gain_db = 20 * math.log10(out_rms / in_rms)
        # crude character readout
        clip = float(np.mean(np.abs(y - np.mean(y)) > 0.9 * np.max(np.abs(y - np.mean(y)))))
        result = {
            "design": did, "version": v, "title": c.title, "source": source, "level_v": level,
            "seconds": seconds, "sample_rate": sr, "render_seconds": round(render_s, 2),
            "gain_db_rms": round(gain_db, 2), "output_peak_v": _r(float(np.max(np.abs(y)))),
            "media": media, "wet": str(pw), "dry": str(pd),
            "guitar": ({"volume": guitar_volume, "pickup": pickup} if guitar_volume is not None else None),
            "summary": (f"rendered {seconds:g} s of {source} at {level:g} V through '{c.title}' in {render_s:.1f} s; "
                        f"level change {gain_db:+.1f} dB RMS (A/B files are loudness-matched)"),
        }
        (vdir / f"audition-{tag}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        self.emit("design.audition", {"design": did, "version": v, "wet": str(pw), "dry": str(pd)})
        return result

    def _sweep_render(self, c: Circuit, x, sr, sweep, make=None):
        """Render with a pot moving: re-render in short blocks with crossfades."""
        from .analyses import process_audio
        pot = sweep["pot"].upper()
        if make is None:
            if pot not in c.pots:
                raise NetlistError(0, f"no pot named {sweep['pot']} (pots: {', '.join(c.pots) or 'none'}; or use 'guitar_volume')")
            make = lambda pos: c.with_pots({pot: pos})
        a, b = float(sweep.get("from", 0.0)), float(sweep.get("to", 1.0))
        blocks = 8
        n = len(x)
        edges = np.linspace(0, n, blocks + 1).astype(int)
        out = np.zeros(n)
        xf = int(0.02 * sr)
        for i in range(blocks):
            pos = a + (b - a) * (i + 0.5) / blocks
            ci = make(pos)
            s, e = edges[i], edges[i + 1]
            pre = max(0, s - int(0.3 * sr))           # pre-roll to settle the circuit
            y = np.asarray(process_audio(ci, x[pre:min(n, e + xf)], sr))[s - pre:]
            seg = y[: e - s + (xf if e < n else 0)]
            if i == 0:
                out[s:s + len(seg)] = seg
            else:
                ramp = np.linspace(0, 1, xf)
                out[s:s + xf] = out[s:s + xf] * (1 - ramp) + seg[:xf] * ramp
                out[s + xf:s + len(seg)] = seg[xf:]
        return out

    # -- tune / yield / bom / build / export ----------------------------------
    def tune(self, vary, netlist=None, design=None, path=None, name=None, pots=None, specs=None, series=None) -> dict:
        from .tune import tune
        text, c, did, origin = self._load(netlist, design, path, pots)
        r = tune(c, vary, specs, series)
        newc = r.pop("circuit")
        before_full = [x.as_dict() for x in evaluate(c, specs)]
        r["before"] = before_full
        before_pass = sum(1 for x in before_full if x["pass"])
        # the search skips slow large-signal specs; the verdict must not
        final = [x.as_dict() for x in evaluate(newc, specs)]
        r["after"] = final
        r["all_pass"] = all(x["pass"] for x in final)
        blockers = self._blockers(newc)
        if blockers:
            r["all_pass"] = False
            r["blockers"] = blockers
        after_pass = sum(1 for x in final if x["pass"])
        if after_pass < before_pass and not r["all_pass"]:
            # never replace a design with one that passes fewer specs
            r["changes"] = []
            r["kept_original"] = True
            r["summary"] = (f"no better values: tuning {', '.join(vary)} only traded specs ({before_pass} passed "
                            f"before, {after_pass} after), so the design is unchanged. Run circuit_explain to see "
                            "which parts set the failing spec, and tune those")
            r["design"] = did
            return r
        new_text = netlist_text(newc)
        if did is None:
            found = self.store.find_by_netlist(text)
            did = found[0] if found else None
        note = "tuned " + ", ".join(f"{ch['part']} {ch['from']}→{ch['to']}" for ch in r["changes"]) if r["changes"] else "tune: no better values"
        did, v, vdir = self.store.save_version(did, name or c.title, new_text, note)
        r.update(design=did, version=v, netlist=new_text, summary=(
            ("all specs pass after tuning: " if r["all_pass"] else
             ("tuned, but the design can't pass: " + "; ".join(r["blockers"][:1]) + ". Values: " if r.get("blockers")
              else "best found, some specs still fail: "))
            + (", ".join(f"{ch['part']} {ch['from']} → {ch['to']}" for ch in r["changes"]) or "no value changes helped")))
        self.store.update_version(did, v, verdict="pass" if r["all_pass"] else "fail")
        self.emit("design.updated", {"design": did, "version": v, "verdict": "pass" if r["all_pass"] else "fail", "title": c.title})
        return r

    def _blockers(self, c: Circuit) -> list:
        """Reasons a design can't pass however its specs read: wiring errors, a DC solve that
        failed, swapped op-amp supplies, a reversed transistor, latching or oscillation."""
        out = [i["message"] for i in erc.check(c) if i["level"] == "error"]
        if out:
            return out
        try:
            from .analyses import op as _op
            o = _op(c)
        except Exception as e:
            return [f"the DC operating point didn't solve: {e}"]
        out += [i["message"] for i in erc.check_op(c, o) if i["level"] == "error"]
        out += [f"{n} is wired backwards (reverse-active)" for n, d in (o.devices or {}).items()
                if str(d.get("region", "")).lower() == "reverse" and n.startswith("Q")]
        st = self._stability(c)
        if st.get("state") in ("latches", "oscillates") and any(
                t.split("(")[0].strip().lower() in ("gain_db", "phase", "fc", "f3db_low", "f3db_high", "peak_db", "zin")
                for t in c.specs):
            out.append(st["message"])
        return out

    def yield_(self, netlist=None, design=None, path=None, pots=None, n=200, seed=1) -> dict:
        text, c, did, origin = self._load(netlist, design, path, pots)
        blockers = self._blockers(c)
        if blockers:
            return {"error": "fix the nominal design first: " + "; ".join(blockers[:2]), "yield_pct": None,
                    "title": c.title}
        r = montecarlo.run(c, None, n=int(200 if n is None else n), seed=int(1 if seed is None else seed))
        if r.get("yield_pct") is not None:
            top = r["sensitivity"][0] if r["sensitivity"] else None
            r["summary"] = (f"{r['yield_pct']:g}% of {r['builds']} virtual builds meet every spec "
                            f"(95% interval {r['ci95_pct'][0]:g}–{r['ci95_pct'][1]:g}%)"
                            + (f"; {top['param']} drives {top['share_pct']:g}% of the variation" if top else ""))
        r["title"] = c.title
        if did:
            (self.store.version_dir(did) / "yield.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            r["design"] = did
        return r

    def bom(self, netlist=None, design=None, path=None, pots=None, series="E24") -> dict:
        text, c, did, origin = self._load(netlist, design, path, pots)
        rows = parts.bom(c, series or "E24")
        md = parts.bom_markdown(rows)
        out = {"title": c.title, "rows": rows, "markdown": md,
               "summary": f"{sum(r['qty'] for r in rows)} parts in {len(rows)} lines"}
        if did:
            p = self.store.version_dir(did) / "bom.md"
            p.write_text(f"# {c.title}: parts list\n\n{md}\n", encoding="utf-8")
            out["file"] = str(p)
            out["design"] = did
        return out

    def build(self, netlist=None, design=None, path=None, pots=None, name=None, max_width=30) -> dict:
        from . import stripboard
        text, c, did, origin = self._load(netlist, design, path, pots)
        did, v, vdir, _ = self._save(c, text, did, name, "build")
        board = stripboard.layout(c, max_width=int(max_width or 30))
        p = vdir / "stripboard.svg"
        p.write_text(stripboard.svg(board), encoding="utf-8")
        steps = stripboard.build_steps(board)
        off = stripboard.offboard(c)
        info = {"cols": board.width, "rows": board.height, "cuts": len(board.cuts), "links": len(board.links),
                "size_in": [round(board.width * 0.1, 1), round(board.height * 0.1, 1)],
                "parts_on_board": len(board.parts), "notes": list(board.notes)}
        try:
            info["layout"] = board.to_dict()
        except Exception:
            pass
        media = [str(p)]
        png = _try_png(p)
        out = {"design": did, "version": v, "title": c.title, "board": info, "steps": steps, "offboard": off,
               "media": media, "png": [str(png)] if png else [],
               "summary": f"stripboard {info['cols']} holes × {info['rows']} strips ({info['size_in'][0]} × {info['size_in'][1]} in), "
                          f"{info.get('cuts', 0)} track cut{'s' if info.get('cuts', 0) != 1 else ''}, "
                          f"{info.get('links', 0)} wire link{'s' if info.get('links', 0) != 1 else ''}, {len(steps)} build steps"}
        (vdir / "build.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        return out

    def export(self, fmt, netlist=None, design=None, path=None, pots=None, name=None) -> dict:
        text, c, did, origin = self._load(netlist, design, path, pots)
        did, v, vdir, _ = self._save(c, text, did, name, f"export {fmt}")
        slug = slugify(c.title)
        if fmt == "kicad":
            p = vdir / f"{slug}.net"
            p.write_text(kicad_netlist(c), encoding="utf-8")
        elif fmt == "spice":
            p = vdir / f"{slug}.cir"
            p.write_text(spice_deck(c, ".op"), encoding="utf-8")
        elif fmt == "svg":
            from .schematic import svg as schematic_svg
            p = vdir / f"{slug}-schematic.svg"
            p.write_text(schematic_svg(c), encoding="utf-8")
        elif fmt == "json":
            from .analyses import op
            p = vdir / f"{slug}.json"
            p.write_text(json.dumps(compile_json(c, op(c), self.sr), indent=1), encoding="utf-8")
        else:
            raise NetlistError(0, "format must be kicad, spice, svg or json")
        return {"design": did, "version": v, "format": fmt, "file": str(p), "media": [str(p)],
                "summary": f"wrote {p.name}"}

    def explain(self, netlist=None, design=None, path=None, pots=None, specs=None, step=0.10) -> dict:
        """What each part does, measured: nudge every part (and pot) and record how each
        small-signal and DC spec moves. Large-signal specs are skipped for speed."""
        text, c, did, origin = self._load(netlist, design, path, pots)
        texts = list(c.specs) + [s for s in (specs or []) if s not in c.specs]
        parsed, skipped = [], []
        for t in texts:
            try:
                s = parse_spec(t)
            except Exception:
                continue
            if s.needs != "tran" and s.metric != "region":
                parsed.append(s)
            elif s.needs == "tran":
                skipped.append(s.text)
        if not parsed and c.input and c.output:
            parsed = [parse_spec("gain_db(1k) > -1000"), parse_spec("f3db_low < 1e9"), parse_spec("f3db_high < 1e9")]
        if not parsed:
            raise NetlistError(0, "add '*@ spec' lines (gain, corner, voltages) so there is something to explain")
        base = Evaluator(c)
        base_vals = {s.text: base.measure(s) for s in parsed}
        rows = []
        movers = [e for e in c.elements if e.kind in ("R", "C", "L") and not e.pot and e.value]
        for e in movers:
            try:
                ev = Evaluator(c.with_values({e.name: e.value * (1 + step)}))
                deltas = {}
                for s in parsed:
                    b0, b1 = base_vals[s.text], ev.measure(s)
                    if b0 is None or b1 is None:
                        continue
                    deltas[s.text] = _delta(s, b0, b1)
                rows.append({"part": e.name, "value": format_value(e.value), "change": f"+{int(step * 100)}%", "effects": deltas})
            except Exception:
                continue
        for p in c.pots.values():
            new = min(1.0, p.pos + 0.1) if p.pos <= 0.9 else p.pos - 0.1
            try:
                ev = Evaluator(c.with_pots({p.name: new}))
                deltas = {}
                for s in parsed:
                    b0, b1 = base_vals[s.text], ev.measure(s)
                    if b0 is None or b1 is None:
                        continue
                    deltas[s.text] = _delta(s, b0, b1)
                rows.append({"part": p.name, "value": f"{p.label} {p.pos:.2f}→{new:.2f}", "change": "knob", "effects": deltas})
            except Exception:
                continue
        # who matters most for each spec
        top = {}
        for s in parsed:
            ranked = sorted((r for r in rows if s.text in r["effects"]),
                            key=lambda r: -abs(r["effects"][s.text]["norm"]))[:3]
            top[s.text] = [{"part": r["part"], "effect": r["effects"][s.text]["text"]} for r in ranked
                           if abs(r["effects"][s.text]["norm"]) > 1e-4]
        inert = [r["part"] for r in rows if all(abs(d["norm"]) < 1e-3 for d in r["effects"].values())]
        lines = []
        for s in parsed:
            if top[s.text]:
                lines.append(f"{s.text}: " + ", ".join(f"{t['part']} ({t['effect']})" for t in top[s.text]))
        out = {"title": c.title, "baseline": {k: _r(v) for k, v in base_vals.items()},
               "per_part": rows, "most_influential": top, "barely_matter": inert,
               "summary": " | ".join(lines) or "no part moves these specs measurably"}
        if skipped:
            out["not_explained"] = skipped
            out["summary"] += (f" | not covered (large-signal specs need a time simulation per part; "
                               f"use circuit_check after a change): {', '.join(skipped)}")
        return out

    def match(self, dry, wet, netlist=None, design=None, path=None, level=0.2) -> dict:
        """Find knob settings that reproduce a recorded tone; with no circuit, rank the library."""
        from . import match as m
        x, y, sr = m.load_pair(dry, wet)
        if netlist or design or path:
            text, c, did, origin = self._load(netlist, design, path)
            if not c.input or not c.output:
                raise NetlistError(0, "matching needs '*@ in' and '*@ out'")
            r = m.fit_pots(c, x, y, sr, level=level)
            r["title"] = c.title
            knobs = ", ".join(f"{r['labels'][p]} {v:.2f}" for p, v in r["positions"].items()) or "no knobs"
            r["summary"] = f"closest settings for '{c.title}': {knobs} (distance {r['distance']:g}; under 3 is close, over 8 is a different sound)"
            return r
        ranked = []
        import time
        t0 = time.time()
        target = m.tone_features(y, sr)
        for item in library_index():
            f = _library_file(item.get("id") or "")
            if not f:
                continue
            try:
                c = parse(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not c.input or not c.output or item.get("category") in ("utility", "oscillator"):
                continue
            best = None
            main = next(iter(c.pots), None)
            for pos in ((0.2, 0.5, 0.9) if main else (None,)):
                ci = c.with_pots({main: pos}) if main else c
                try:
                    out = m._render(ci, x, sr, level)
                    d = m.distance(m.tone_features(out, sr), target)
                except Exception:
                    continue
                if best is None or d < best[0]:
                    best = (d, pos)
            if best:
                ranked.append({"id": f.stem, "title": c.title, "distance": round(best[0], 2),
                               "knob": (f"{c.pots[main].label} {best[1]:.1f}" if main else "")})
            if time.time() - t0 > 90:
                break
        ranked.sort(key=lambda r: r["distance"])
        top = ranked[0] if ranked else None
        return {"ranking": ranked[:8], "summary": (f"closest library circuit: {top['title']} ({top['knob']}), distance {top['distance']:g}"
                                                   if top else "no audio circuits to compare")}

    def library(self, query=None) -> dict:
        items = library_index()
        if query:
            f = _library_file(query)
            if f:
                c = parse(f.read_text(encoding="utf-8"))
                meta = next((i for i in items if i.get("id") == f.stem), {})
                return {"id": f.stem, "title": c.title, "netlist": f.read_text(encoding="utf-8"),
                        "pots": {p.name: {"label": p.label, "value": format_value(p.value), "taper": p.taper} for p in c.pots.values()},
                        "specs": c.specs,
                        "measured_in_ngspice": meta.get("specs") or meta.get("measured") or meta.get("ngspice"),
                        "description": meta.get("description", ""),
                        "summary": f"{c.title}: {len(c.elements)} parts, {len(c.pots)} knobs, {len(c.specs)} specs"}
            words = [w for w in query.lower().split() if w]
            found = [i for i in items if all(w in json.dumps(i).lower() for w in words)]
            if not found:
                return {"circuits": [], "summary": f"no reference circuit matches '{query}'. "
                        "Call circuit_library with no query to list all of them"}
            items = found
        brief = [{"id": i.get("id"), "title": i.get("title"), "category": i.get("category"),
                  "description": i.get("description", "")} for i in items]
        return {"circuits": brief, "summary": f"{len(brief)} reference circuits"}


def _delta(s, b0, b1) -> dict:
    """Change of a measurement, in its own unit, plus a normalised size for ranking."""
    unit = s.unit
    if unit in ("dB", "°"):
        d = b1 - b0
        return {"delta": _r(d, 3), "norm": d / 3.0, "text": f"{d:+.2f} {unit}"}
    if b0 == 0:
        d = b1 - b0
        return {"delta": _r(d), "norm": d, "text": f"{format_value(d, unit)} change"}
    rel = (b1 - b0) / abs(b0)
    return {"delta": _r(b1 - b0), "norm": rel, "text": f"{rel * 100:+.1f}%"}


def parts_freq(s) -> float:
    from .netlist import parse_value
    return parse_value(s.args[0]) if s.args else 1000.0
