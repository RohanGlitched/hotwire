"""Hotwire backend routes for the Desktop Bench pane, mounted at /api/plugins/hotwire/.

Hermes imports this file on its own (``spec_from_file_location``), not as part of
the plugin package, so the engine is loaded here under a private package name
that points at ``<plugin root>/engine``. Every route validates its input and
answers ``{"error": "..."}`` with a 4xx status instead of a traceback.
"""
from __future__ import annotations

import base64
import importlib
import json
import logging
import math
import os
import re
import sys
import threading
import types
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter()

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
_PKG = "hotwire_bench_engine"
_ID_RE = re.compile(r"[a-z0-9-]{1,48}")
_POT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}")
_SOURCES = ("guitar_riff", "bass_line", "sine_sweep", "drum_loop")
_MIME = {".svg": "image/svg+xml", ".wav": "audio/wav", ".png": "image/png", ".json": "application/json",
         ".cir": "text/plain; charset=utf-8", ".md": "text/markdown; charset=utf-8",
         ".net": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8"}
_INLINE_SVG_MAX = 1_500_000
_AUDIO_MAX = 12_000_000

_lock = threading.Lock()
_bench_cache: dict = {}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


# --------------------------------------------------------------------------- engine access

def _engine(name: str):
    """Import ``engine.<name>`` from the plugin folder under a private package name."""
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(PLUGIN_ROOT / "engine")]
        pkg.__package__ = _PKG
        pkg.__file__ = str(PLUGIN_ROOT / "engine" / "__init__.py")
        sys.modules[_PKG] = pkg
    return importlib.import_module(f"{_PKG}.{name}")


def _emit(event: str, payload: dict):
    try:
        from hermes_cli.plugin_events import broadcast_plugin_event
        broadcast_plugin_event("hotwire", event, payload)
    except Exception:
        pass


def bench():
    """One Bench per data directory (tests point HOTWIRE_DATA at a temp folder)."""
    store_mod = _engine("store")
    root = str(store_mod.data_dir())
    with _lock:
        b = _bench_cache.get(root)
        if b is None:
            wb = _engine("workbench")
            b = wb.Bench(store=store_mod.Store(Path(root)), emit=_emit)
            _bench_cache[root] = b
        return b


# --------------------------------------------------------------------------- helpers

def _clean(x):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return _clean(x.item())
        if isinstance(x, np.ndarray):
            return _clean(x.tolist())
    except Exception:
        pass
    return str(x)


def _ok(data) -> JSONResponse:
    return JSONResponse(_clean(data))


def _fail(e: Exception) -> JSONResponse:
    if isinstance(e, ApiError):
        return JSONResponse({"error": e.message}, status_code=e.status)
    if isinstance(e, ImportError):
        logger.warning("hotwire api: missing engine module", exc_info=True)
        return JSONResponse({"error": "that part of the bench is not installed yet"}, status_code=503)
    msg = getattr(e, "message", None)
    if msg:  # NetlistError / SimulationError carry a plain-English message
        line = getattr(e, "line", None)
        return JSONResponse({"error": str(msg), **({"line": line} if line else {})}, status_code=422)
    if e.__class__.__name__ in ("SimulationError", "NetlistError"):
        return JSONResponse({"error": str(e)}, status_code=422)
    logger.warning("hotwire api error", exc_info=True)
    return JSONResponse({"error": "the bench could not finish that request"}, status_code=500)


def _guard(fn, *a, **kw):
    try:
        return _ok(fn(*a, **kw))
    except Exception as e:  # never leak a traceback to the client
        return _fail(e)


def _design_id(did) -> str:
    if not isinstance(did, str) or not _ID_RE.fullmatch(did):
        raise ApiError("design must be a design id like 'two-transistor-fuzz'")
    if not bench().store.exists(did):
        raise ApiError(f"no design named '{did}'", 404)
    return did


def _version(did: str, v) -> int:
    meta = bench().store.meta(did)
    if v in (None, "", "latest"):
        return int(meta["latest"])
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ApiError("version must be a whole number")
    if not any(int(r.get("v", -1)) == n for r in meta.get("versions", [])):
        raise ApiError(f"{did} has no version {n}", 404)
    return n


def _pots_arg(pots) -> dict:
    """{"RV1": 0.5} from a dict, or from 'RV1:0.5,RV2:0.3'."""
    if pots in (None, "", {}):
        return {}
    if isinstance(pots, str):
        out = {}
        for part in pots.split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ApiError("pots must look like RV1:0.5,RV2:0.3")
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
        pots = out
    if not isinstance(pots, dict) or len(pots) > 16:
        raise ApiError("pots must be an object like {\"RV1\": 0.5}")
    clean = {}
    for k, v in pots.items():
        if not isinstance(k, str) or not _POT_RE.fullmatch(k):
            raise ApiError(f"'{k}' is not a pot name")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ApiError(f"pot {k} needs a position between 0 and 1")
        if not math.isfinite(f):
            raise ApiError(f"pot {k} needs a position between 0 and 1")
        clean[k.upper()] = min(1.0, max(0.0, f))
    return clean


def _vdir(did: str, v: int) -> Path:
    return bench().store.root / did / f"v{v}"


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_svg(p: Path):
    try:
        if p.is_file() and p.stat().st_size <= _INLINE_SVG_MAX:
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return None


def _parse(text: str):
    return _engine("netlist").parse(text)


def _pots_of(text: str) -> list:
    try:
        c = _parse(text)
    except Exception:
        return []
    fmt = _engine("netlist").format_value
    return [{"name": p.name, "label": p.label or p.name, "value": p.value, "value_text": fmt(p.value, "Ω"),
             "taper": p.taper, "pos": p.pos, "a": p.a, "w": p.w, "b": p.b} for p in c.pots.values()]


# --------------------------------------------------------------------------- display helpers
# The engine is gaining plain-English fields (spec ``label``, ``measured_text``,
# ``target_text``). Checks saved by an older engine lack them, so fill them in
# here when the engine can, and leave the record alone when it can't.

_OUTDATED_RE = re.compile(r"unknown spec metric", re.I)


def _fmt_measure(v, unit: str) -> str:
    specs_mod = _engine("specs")
    f = getattr(specs_mod, "_fmt", None)
    if f is not None:
        try:
            return f(v, unit or "")
        except Exception:
            pass
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if unit in ("%", "°"):
        return f"{v:.3g}{unit}"
    if unit in ("dB", "h"):
        return f"{v:.3g} {unit}"
    return _engine("netlist").format_value(float(v), unit or "")


def _fill_specs(specs) -> list:
    """Spec rows with a plain-English label and display strings, whatever engine wrote them."""
    out = []
    if not isinstance(specs, list):
        return out
    try:
        specs_mod = _engine("specs")
    except Exception:
        specs_mod = None
    for s in specs:
        if not isinstance(s, dict):
            continue
        s = dict(s)
        text = str(s.get("spec") or "")
        parsed = None
        if specs_mod is not None and text:
            try:
                parsed = specs_mod.parse_spec(text)
            except Exception:
                parsed = None
        if (not s.get("label") or s.get("label") == text) and parsed is not None:
            lab = getattr(specs_mod, "spec_label", None)
            if lab is not None:
                try:
                    s["label"] = lab(parsed)
                except Exception:
                    pass
        if not s.get("unit") and parsed is not None:
            try:
                s["unit"] = parsed.unit
            except Exception:
                pass
        if not s.get("measured_text"):
            try:
                s["measured_text"] = _fmt_measure(s.get("measured"), s.get("unit") or "")
            except Exception:
                s["measured_text"] = "—" if s.get("measured") is None else str(s.get("measured"))
        if not s.get("target_text") or s.get("target_text") == "—":
            tt = getattr(specs_mod, "_target_text", None) if specs_mod is not None else None
            if tt is not None and parsed is not None:
                try:
                    s["target_text"] = tt(parsed)
                except Exception:
                    s["target_text"] = s.get("target") or "—"
            else:
                s["target_text"] = s.get("target") or "—"
        why = f"{s.get('error') or ''} {s.get('explanation') or ''}"
        if _OUTDATED_RE.search(why):
            # the spec is fine; the engine that checked it was older than the spec
            s["outdated"] = True
        out.append(s)
    return out


_PIN_NAMES = {"Q": ("collector", "base", "emitter"), "J": ("drain", "gate", "source"),
              "D": ("anode", "cathode"), "X": ("+ input", "− input", "V+ pin", "V− pin", "output")}


def _node_phrase(c, node: str, skip: str, volts: dict) -> str:
    """Plain words for where a node goes: ground, the supply, Q1's base, the input…"""
    if node == "0":
        return "ground"
    elements = list(getattr(c, "elements", []) or [])
    by_name = {e.name: e for e in elements}
    for sname in getattr(c, "supplies", []) or []:
        e = by_name.get(sname)
        if e is not None and e.nodes and e.nodes[0] == node:
            v = volts.get(node)
            return f"the +{v:g} V supply" if isinstance(v, (int, float)) and v > 0 else "the supply"
    src = by_name.get(getattr(c, "input", None) or "")
    if src is not None and src.nodes and src.nodes[0] == node:
        return "the input"
    pins = []
    for e in elements:
        if e.name == skip:
            continue
        for i, n in enumerate(e.nodes or []):
            if n == node:
                pins.append((e, i))
    rank = {"Q": 0, "J": 0, "X": 0, "D": 1}
    pins.sort(key=lambda p: rank.get(p[0].kind, 5 if not p[0].pot else 3))
    for e, i in pins:
        names = _PIN_NAMES.get(e.kind)
        if names and i < len(names):
            return f"{e.name}'s {names[i]}"
        if e.pot:
            pot = (getattr(c, "pots", {}) or {}).get(e.pot)
            lab = f" ({pot.label})" if pot is not None and pot.label and pot.label != pot.name else ""
            if pot is not None and node == pot.w:
                return f"{pot.name}'s wiper{lab}"
            return f"{e.pot}{lab}"
    if node == getattr(c, "output", None):
        return "the output"
    if pins:
        return pins[0][0].name
    return f"node {node}"


def _ratio_phrase(what: str) -> str:
    return {"power": "power rating", "voltage": "voltage rating", "current": "current rating",
            "supply": "supply rating"}.get(what, f"{what} rating")


def _stress_view(text: str, chk, pots=None):
    """A headline and plain rows for the Stress card, built from a check record."""
    if not isinstance(chk, dict) or not isinstance(chk.get("op"), dict):
        return None
    try:
        c = _parse(text)
        if pots:
            c = c.with_pots(pots)
    except Exception:
        return None
    volts = {str(k): v for k, v in (chk["op"].get("nodes") or {}).items()}
    devs = chk["op"].get("devices") or {}
    stress_mod = _engine("stress")
    part_rating = getattr(stress_mod, "_part_rating", None)

    def vn(n):
        if n == "0":
            return 0.0
        v = volts.get(n)
        return float(v) if isinstance(v, (int, float)) else 0.0

    def rating(name, unit, default):
        r = (getattr(c, "ratings", {}) or {}).get(name)
        return r[0] if r and r[1] == unit else default

    loads = []  # (pct, part, what)
    parts = set()
    for e in c.elements:
        k = e.kind
        if k in ("V", "I", "E", "G", "S"):
            continue
        parts.add(e.pot or e.name)
        try:
            if k == "R" and e.value:
                p = (vn(e.nodes[0]) - vn(e.nodes[1])) ** 2 / float(e.value)
                lim = rating(e.pot or e.name, "W", 0.25)
                loads.append((100 * p / lim, e.pot or e.name, "power"))
            elif k == "C" and e.value:
                dv = abs(vn(e.nodes[0]) - vn(e.nodes[1]))
                lim = rating(e.name, "V", 25.0 if e.value >= 1e-6 else 50.0)
                loads.append((100 * dv / lim, e.name, "voltage"))
            elif part_rating is not None and k in ("Q", "D", "J", "X"):
                r = part_rating(c, e) or {}
                d = devs.get(e.name) or {}
                if k == "Q" and r.get("pd_w"):
                    pd = d.get("power")
                    if not isinstance(pd, (int, float)):
                        pd = abs(float(d.get("vce", 0) or 0)) * abs(float(d.get("ic", 0) or 0))
                    loads.append((100 * abs(pd) / r["pd_w"], e.name, "power"))
                elif k == "D" and r.get("imax_a"):
                    loads.append((100 * abs(float(d.get("id", 0) or 0)) / r["imax_a"], e.name, "current"))
                elif k == "J" and r.get("vmax_v"):
                    loads.append((100 * abs(float(d.get("vds", 0) or 0)) / r["vmax_v"], e.name, "voltage"))
                elif k == "X" and r.get("vmax_v") and len(e.nodes) > 3:
                    loads.append((100 * abs(vn(e.nodes[2]) - vn(e.nodes[3])) / r["vmax_v"], e.name, "supply"))
        except Exception:
            continue
    rows = []
    for i in chk.get("erc") or []:
        if isinstance(i, dict):
            rows.append({"level": i.get("level") or "warning", "text": str(i.get("message") or ""), "part": i.get("part")})
    for s in chk.get("stress") or []:
        if not isinstance(s, dict):
            continue
        msg = str(s.get("message") or "")
        if s.get("what") == "polarity" and s.get("level") == "info":
            e = next((x for x in c.elements if x.name == s.get("part")), None)
            if e is not None and len(e.nodes) >= 2:
                dv = vn(e.nodes[0]) - vn(e.nodes[1])
                if abs(dv) >= 0.05:
                    hi, lo = (e.nodes[0], e.nodes[1]) if dv > 0 else (e.nodes[1], e.nodes[0])
                    msg = (f"{e.name} (electrolytic): + side toward {_node_phrase(c, hi, e.name, volts)}, "
                           f"− side toward {_node_phrase(c, lo, e.name, volts)}")
                else:
                    msg = f"{e.name} (electrolytic) has almost no DC across it: use a non-polarised type"
        rows.append({"level": s.get("level") or "info", "text": msg, "part": s.get("part")})
    for b in chk.get("bias") or []:
        if isinstance(b, dict):
            rows.append({"level": "warning", "text": str(b.get("message") or ""), "part": b.get("part")})
    order = {"error": 0, "warning": 1}
    rows.sort(key=lambda r: order.get(r["level"], 2))
    n = len(parts)
    worst = max(loads, key=lambda x: x[0]) if loads else None
    over = sorted({s.get("part") for s in (chk.get("stress") or []) if isinstance(s, dict) and s.get("level") == "error"} - {None})
    warm = sorted({s.get("part") for s in (chk.get("stress") or []) if isinstance(s, dict) and s.get("level") == "warning"} - {None})

    def pct(x):
        return "under 1%" if x < 1 else f"{x:.0f}%"
    worst_txt = f"worst: {worst[1]} at {pct(worst[0])} of its {_ratio_phrase(worst[2])}" if worst else ""
    word = "part" if n == 1 else "parts"
    if over:
        head = (f"{len(over)} of {n} {word} over {'its rating' if len(over) == 1 else 'their ratings'}: "
                f"{', '.join(over[:3])}")
        tone = "fail"
    elif warm:
        head = f"All {n} {word} within ratings, {len(warm)} running warm"
        tone = "warn"
    else:
        head = f"All {n} {word} within ratings" if n != 1 else "The one part is within its rating"
        tone = "pass"
    if worst_txt:
        head += f" · {worst_txt}"
    return {"headline": head, "tone": tone, "parts": n, "rows": rows,
            "worst": ({"part": worst[1], "pct": round(worst[0], 2), "what": worst[2]} if worst else None)}


def _decorate_check(text: str, chk, pots=None):
    """The check record as the Bench shows it: filled spec rows and a stress view."""
    if not isinstance(chk, dict):
        return chk
    chk = dict(chk)
    if isinstance(chk.get("specs"), list):
        chk["specs"] = _fill_specs(chk["specs"])
    try:
        sv = _stress_view(text, chk, pots)
    except Exception:
        logger.debug("hotwire stress view failed", exc_info=True)
        sv = None
    if sv:
        chk["stress_view"] = sv
    chk["outdated"] = any(s.get("outdated") for s in chk.get("specs") or [] if isinstance(s, dict))
    return chk


_title_cache: dict = {}


def _design_title(did: str, meta: dict) -> str:
    """The circuit's own title from its newest check (cached by file time), else the design name."""
    for r in reversed(meta.get("versions") or []):
        p = _vdir(did, int(r.get("v", 0))) / "check.json"
        try:
            st = p.stat().st_mtime
        except OSError:
            continue
        hit = _title_cache.get(str(p))
        if hit and hit[0] == st:
            return hit[1] or meta.get("name") or did
        chk = _read_json(p) or {}
        t = chk.get("title") if isinstance(chk, dict) else None
        if len(_title_cache) > 500:
            _title_cache.clear()
        _title_cache[str(p)] = (st, t)
        return t or meta.get("name") or did
    # never checked: the netlist's own '*@ title' line
    try:
        p = _vdir(did, int(meta.get("latest") or 0)) / "netlist.cir"
        with p.open(encoding="utf-8", errors="replace") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                m = re.match(r"\s*\*@\s*title\s+(.+)", line)
                if m:
                    return m.group(1).strip()[:120]
    except Exception:
        pass
    return meta.get("name") or did


def _check_brief(chk) -> dict:
    if not isinstance(chk, dict):
        return {}
    specs = chk.get("specs") or []
    return {"verdict": chk.get("verdict"), "summary": chk.get("summary"),
            "specs_pass": sum(1 for s in specs if s.get("pass")), "specs_total": len(specs),
            "specs": [{"spec": s.get("spec"), "pass": bool(s.get("pass"))} for s in specs]}


def _auditions(vdir: Path) -> list:
    out = []
    for f in sorted(vdir.glob("audition-*.json")):
        a = _read_json(f)
        if not isinstance(a, dict):
            continue
        wet, dry = a.get("wet"), a.get("dry")
        if not wet or not dry:
            continue
        out.append({"tag": f.stem[len("audition-"):], "wet": wet, "dry": dry, "source": a.get("source"),
                    "seconds": a.get("seconds"), "gain_db_rms": a.get("gain_db_rms"), "guitar": a.get("guitar"),
                    "summary": a.get("summary"), "version": a.get("version"),
                    "spectrum": str(vdir / f"spectrum-{f.stem[len('audition-'):]}.svg")
                    if (vdir / f"spectrum-{f.stem[len('audition-'):]}.svg").exists() else None,
                    "mtime": f.stat().st_mtime})
    out.sort(key=lambda a: -a["mtime"])
    return out


def _version_payload(did: str, v: int) -> dict:
    vdir = _vdir(did, v)
    try:
        netlist = (vdir / "netlist.cir").read_text(encoding="utf-8")
    except Exception:
        raise ApiError(f"{did} v{v} has no netlist on disk", 404)
    chk = _decorate_check(netlist, _read_json(vdir / "check.json"))
    svg = {}
    for key, fname in (("schematic", "schematic.svg"), ("bode", "bode.svg"), ("stripboard", "stripboard.svg")):
        s = _read_svg(vdir / fname)
        if s:
            svg[key] = s
    meta = bench().store.meta(did)
    row = next((r for r in meta.get("versions", []) if int(r.get("v", -1)) == v), {})
    build = _read_json(vdir / "build.json")
    bom_md = None
    if (vdir / "bom.md").is_file():
        bom_md = (vdir / "bom.md").read_text(encoding="utf-8", errors="replace")
    return {"design": did, "version": v, "note": row.get("note", ""), "time": row.get("time"),
            "title": (chk or {}).get("title") or meta.get("name") or did,
            "netlist": netlist, "pots": _pots_of(netlist), "check": chk, "svg": svg,
            "auditions": _auditions(vdir), "build": build, "bom_md": bom_md,
            "yield": _read_json(vdir / "yield.json"), "dir": str(vdir)}


def _versions(did: str) -> list:
    meta = bench().store.meta(did)
    out = []
    for r in meta.get("versions", []):
        v = int(r.get("v", 0))
        brief = _check_brief(_read_json(_vdir(did, v) / "check.json"))
        out.append({"v": v, "note": r.get("note", ""), "time": r.get("time"),
                    "verdict": r.get("verdict") or brief.get("verdict"),
                    "specs_pass": r.get("specs_pass", brief.get("specs_pass")),
                    "specs_total": r.get("specs_total", brief.get("specs_total")),
                    "specs": brief.get("specs", []), "checked": bool(brief),
                    "has_audio": any(_vdir(did, v).glob("audition-*.json"))})
    return out


def _netlist_for(did: str, v: int) -> str:
    return (_vdir(did, v) / "netlist.cir").read_text(encoding="utf-8")


def _bode_inline(c, spec_res) -> str | None:
    """The Bode plot for a re-check that was not saved (same marks as Bench.check)."""
    if not (c.input and c.output):
        return None
    try:
        plot = _engine("plot")
        specs_mod = _engine("specs")
        wb = _engine("workbench")
        ev = specs_mod.Evaluator(c)
        acr = ev.acr
        marks = []
        for r in spec_res or []:
            if r.get("metric") in ("gain_db", "fc", "f3db_low", "f3db_high") and r.get("measured") is not None:
                s = specs_mod.parse_spec(r["spec"])
                marks.append({"kind": r["metric"],
                              "f": float(wb.parts_freq(s)) if r["metric"] == "gain_db" else r["measured"],
                              "lo": s.lo if s.op in (">", ">=", "in") else None,
                              "hi": (s.hi if s.op == "in" else (s.lo if s.op in ("<", "<=") else None)),
                              "measured": r["measured"], "pass": r["pass"]})
        return plot.bode_svg(acr.f, acr.h, specs=marks, title=c.title)
    except Exception:
        logger.debug("hotwire inline bode failed", exc_info=True)
        return None


async def _body(request: Request) -> dict:
    try:
        raw = await request.body()
        if len(raw) > 600_000:
            raise ApiError("request body is too large", 413)
        data = json.loads(raw or b"{}")
    except ApiError:
        raise
    except Exception:
        raise ApiError("body must be JSON")
    if not isinstance(data, dict):
        raise ApiError("body must be a JSON object")
    return data


# --------------------------------------------------------------------------- routes

@router.get("/health")
def health():
    def run():
        return {"ok": True, "data_dir": str(bench().store.root.parent)}
    return _guard(run)


@router.get("/designs")
def list_designs(limit: int = 50):
    def run():
        n = max(1, min(int(limit), 200))
        rows = []
        for m in bench().store.list(n):
            vers = m.get("versions") or []
            last = vers[-1] if vers else {}
            checked = next((r for r in reversed(vers) if r.get("verdict")), {})
            rows.append({"id": m.get("id"), "name": m.get("name"), "title": _design_title(m.get("id"), m),
                         "latest": m.get("latest"),
                         "updated": m.get("updated", m.get("created")), "created": m.get("created"),
                         "verdict": checked.get("verdict"), "verdict_v": checked.get("v"),
                         "specs_pass": checked.get("specs_pass"), "specs_total": checked.get("specs_total"),
                         "last_note": last.get("note", ""), "versions": len(vers)})
        return {"designs": rows}
    return _guard(run)


@router.get("/designs/{did}")
def get_design(did: str):
    def run():
        d = _design_id(did)
        meta = bench().store.meta(d)
        versions = _versions(d)
        latest = int(meta["latest"])
        # the newest version that carries a check is what the bench shows first
        shown = next((r["v"] for r in reversed(versions) if r["checked"]), latest)
        payload = _version_payload(d, shown)
        audio = None
        for r in reversed(versions):
            a = _auditions(_vdir(d, r["v"]))
            if a:
                audio = {**a[0], "version": r["v"]}
                break
        return {"id": d, "name": meta.get("name"), "title": _design_title(d, meta), "latest": latest,
                "created": meta.get("created"),
                "updated": meta.get("updated"), "versions": versions, "current": payload,
                "latest_audition": audio}
    return _guard(run)


@router.get("/designs/{did}/v/{n}")
def get_version(did: str, n: str):
    def run():
        d = _design_id(did)
        return _version_payload(d, _version(d, n))
    return _guard(run)


@router.post("/check")
async def check(request: Request):
    try:
        body = await _body(request)
    except Exception as e:
        return _fail(e)

    def run():
        pots = _pots_arg(body.get("pots"))
        netlist = body.get("netlist")
        did = body.get("design")
        if body.get("save") is True:
            # re-check a stored version in place (a design checked by an older engine)
            if netlist is not None or pots:
                raise ApiError("save re-checks a stored version: send design and version only")
            d = _design_id(did)
            return _recheck(d, _version(d, body.get("version")))
        if netlist is not None:
            if not isinstance(netlist, str) or not netlist.strip():
                raise ApiError("netlist must be SPICE text")
            if len(netlist) > 200_000:
                raise ApiError("netlist is too large", 413)
            text = netlist
        else:
            d = _design_id(did)
            text = _netlist_for(d, _version(d, body.get("version")))
        c = _parse(text)
        known = {k for k in c.pots}
        bad = [k for k in pots if k not in known]
        if bad:
            raise ApiError(f"no pot named {bad[0]} (pots: {', '.join(known) or 'none'})")
        r = bench().check(netlist=text, pots=pots or None, save=False)
        cp = c.with_pots(pots) if pots else c
        bode = _bode_inline(cp, r.get("specs")) if r.get("specs") is not None else None
        r["svg"] = {"bode": bode} if bode else {}
        r["pots"] = pots
        r.pop("media", None)
        r.pop("png", None)
        return _decorate_check(text, r, pots or None)
    return _guard(run)


def _recheck(did: str, v: int) -> dict:
    """Check one stored version again with this engine and save the result into that version."""
    text = _netlist_for(did, v)
    vdir = _vdir(did, v)
    r = bench().check(netlist=text, save=False)
    c = _parse(text)
    r.update(design=did, version=v)
    media = []
    try:
        s = _engine("schematic").svg(c)
        (vdir / "schematic.svg").write_text(s, encoding="utf-8")
        media.append(str(vdir / "schematic.svg"))
    except Exception:
        logger.debug("hotwire re-check schematic failed", exc_info=True)
    bode = _bode_inline(c, r.get("specs")) if r.get("specs") is not None else None
    if bode:
        (vdir / "bode.svg").write_text(bode, encoding="utf-8")
        media.append(str(vdir / "bode.svg"))
    r["media"] = media
    r.pop("png", None)
    (vdir / "check.json").write_text(json.dumps(_clean(r), indent=1, default=str), encoding="utf-8")
    specs = r.get("specs") or []
    bench().store.update_version(did, v, verdict=r.get("verdict"),
                                 specs_pass=sum(1 for s in specs if s.get("pass")), specs_total=len(specs))
    _emit("design.updated", {"design": did, "version": v, "verdict": r.get("verdict"), "title": r.get("title")})
    out = _decorate_check(text, r)
    out["svg"] = {"bode": bode} if bode else {}
    out["saved"] = True
    return out


@router.post("/audition")
async def audition(request: Request):
    try:
        body = await _body(request)
    except Exception as e:
        return _fail(e)

    def run():
        d = _design_id(body.get("design"))
        pots = _pots_arg(body.get("pots"))
        source = body.get("source") or "guitar_riff"
        if source not in _SOURCES:
            raise ApiError(f"source must be one of {', '.join(_SOURCES)}")
        try:
            seconds = float(body.get("seconds") or 4.0)
            level = float(body.get("level") or 0.2)
        except (TypeError, ValueError):
            raise ApiError("seconds and level must be numbers")
        if not (0.5 <= seconds <= 8.0):
            raise ApiError("seconds must be between 0.5 and 8")
        if not (0.001 <= level <= 5.0):
            raise ApiError("level must be between 0.001 and 5 volts")
        gv = body.get("guitar_volume")
        if gv is not None:
            if isinstance(gv, bool):
                raise ApiError("guitar_volume must be a number between 0 and 1")
            try:
                gv = float(gv)
            except (TypeError, ValueError):
                raise ApiError("guitar_volume must be a number between 0 and 1")
            if not (math.isfinite(gv) and 0.0 <= gv <= 1.0):
                raise ApiError("guitar_volume must be a number between 0 and 1")
            if not source.startswith("guitar"):
                gv = None  # the guitar's volume knob only exists for guitar sources
        meta = bench().store.meta(d)
        # the latest netlist text matches the stored design, so no new version is forked
        text = _netlist_for(d, int(meta["latest"]))
        c = _parse(text)
        bad = [k for k in pots if k not in c.pots]
        if bad:
            raise ApiError(f"no pot named {bad[0]} (pots: {', '.join(c.pots) or 'none'})")
        kw = {"guitar_volume": gv} if gv is not None else {}
        r = bench().audition(netlist=text, pots=pots or None, source=source, seconds=seconds, level=level, **kw)
        r["spectrum"] = next((m for m in r.get("media") or [] if str(m).endswith(".svg")), None)
        _emit("design.audition", {"design": r.get("design"), "version": r.get("version"),
                                  "wet": r.get("wet"), "dry": r.get("dry")})
        return r
    return _guard(run)


@router.get("/compile/{did}")
def compile_design(did: str, pots: str = "", v: str = ""):
    def run():
        d = _design_id(did)
        text = _netlist_for(d, _version(d, v or None))
        c = _parse(text)
        p = _pots_arg(pots)
        bad = [k for k in p if k not in c.pots]
        if bad:
            raise ApiError(f"no pot named {bad[0]}")
        if p:
            c = c.with_pots(p)
        analyses = _engine("analyses")
        export = _engine("export")
        opr = analyses.op(c)
        out = export.compile_json(c, opr if getattr(opr, "converged", True) else None)
        out["design"] = d
        return out
    return _guard(run)


@router.get("/library")
def library(q: str = ""):
    def run():
        if q and len(q) > 80:
            raise ApiError("query is too long")
        _warm_library()
        return bench().library(q or None)
    return _guard(run)


_lib_svg_cache: dict = {}
_lib_warm = {"started": False}


def _lib_svg(f: Path) -> str:
    """Schematic SVG (no title) of a library file: memory, then disk, then a fresh layout."""
    st = f.stat().st_mtime
    hit = _lib_svg_cache.get(f.stem)
    if hit and hit[0] == st:
        return hit[1]
    disk = bench().store.root.parent / "cache" / "library" / f"{f.stem}-{int(st)}.svg"
    s = None
    try:
        if disk.is_file():
            s = disk.read_text(encoding="utf-8")
    except Exception:
        s = None
    if not s:
        c = _parse(f.read_text(encoding="utf-8"))
        sch = _engine("schematic")
        try:
            s = sch.svg(c, title=False)
        except TypeError:
            s = sch.svg(c)
        try:
            disk.parent.mkdir(parents=True, exist_ok=True)
            for old in disk.parent.glob(f"{f.stem}-*.svg"):
                old.unlink(missing_ok=True)
            disk.write_text(s, encoding="utf-8")
        except Exception:
            logger.debug("hotwire library thumbnail not cached on disk", exc_info=True)
    with _lock:
        _lib_svg_cache[f.stem] = (st, s)
    return s


def _warm_library():
    """Lay out every library schematic once in the background, so the cards fill in fast."""
    if _lib_warm["started"] or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    _lib_warm["started"] = True

    def go():
        try:
            lib_dir = PLUGIN_ROOT / "library"
            for f in sorted(lib_dir.glob("*.cir")):
                try:
                    _lib_svg(f)
                except Exception:
                    continue
        except Exception:
            pass
    threading.Thread(target=go, name="hotwire-library-warm", daemon=True).start()


@router.get("/library/{lid}/schematic")
def library_schematic(lid: str):
    """The schematic of a library circuit (no title), for the library cards. Cached by file time."""
    def run():
        if not re.fullmatch(r"[a-z0-9_-]{1,64}", lid or ""):
            raise ApiError("library must be a library id")
        f = _engine("workbench")._library_file(lid)
        if f is None:
            raise ApiError(f"no library circuit '{lid}'", 404)
        return {"id": f.stem, "svg": _lib_svg(f)}
    return _guard(run)


@router.post("/open")
async def open_library(request: Request):
    """Check a library circuit, which saves it as a design the bench can show."""
    try:
        body = await _body(request)
    except Exception as e:
        return _fail(e)

    def run():
        lid = body.get("library")
        if not isinstance(lid, str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", lid):
            raise ApiError("library must be a library id")
        wb = _engine("workbench")
        if wb._library_file(lid) is None:
            raise ApiError(f"no library circuit '{lid}'", 404)
        r = bench().check(path=lid, note="opened from the library")
        return {"design": r.get("design"), "version": r.get("version"), "verdict": r.get("verdict"),
                "summary": r.get("summary")}
    return _guard(run)


@router.get("/bom/{did}")
def bom(did: str, v: str = "", series: str = "E24"):
    def run():
        d = _design_id(did)
        if series not in ("E12", "E24", "E48", "E96"):
            raise ApiError("series must be E12, E24, E48 or E96")
        text = _netlist_for(d, _version(d, v or None))
        r = bench().bom(netlist=text, series=series)
        r.pop("file", None)
        return r
    return _guard(run)


@router.post("/build")
async def build(request: Request):
    try:
        body = await _body(request)
    except Exception as e:
        return _fail(e)

    def run():
        d = _design_id(body.get("design"))
        meta = bench().store.meta(d)
        text = _netlist_for(d, int(meta["latest"]))
        r = bench().build(netlist=text)
        p = Path(r["media"][0]) if r.get("media") else None
        r["svg"] = _read_svg(p) if p else None
        return r
    return _guard(run)


def _safe_file(path: str) -> Path:
    if not isinstance(path, str) or not path or len(path) > 1024 or "\x00" in path:
        raise ApiError("path is required")
    p = Path(path)
    if not p.is_absolute():
        p = bench().store.root.parent / p
    if not bench().store.contains(p):
        raise ApiError("that file is outside the Hotwire data folder", 403)
    p = p.resolve()
    if not p.is_file():
        raise ApiError("file not found", 404)
    if p.suffix.lower() not in _MIME:
        raise ApiError("that file type is not served", 403)
    return p


@router.get("/file")
def get_file(path: str = ""):
    try:
        p = _safe_file(path)
        return Response(p.read_bytes(), media_type=_MIME[p.suffix.lower()],
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except Exception as e:
        return _fail(e)


@router.get("/audio")
def get_audio(path: str = ""):
    """A wav as base64 JSON, for clients whose REST door only speaks JSON."""
    def run():
        p = _safe_file(path)
        if p.suffix.lower() != ".wav":
            raise ApiError("only .wav files are served here")
        if p.stat().st_size > _AUDIO_MAX:
            raise ApiError("that audio file is too large", 413)
        return {"path": str(p), "mime": "audio/wav", "b64": base64.b64encode(p.read_bytes()).decode("ascii")}
    return _guard(run)


@router.get("/svg")
def get_svg(path: str = ""):
    def run():
        p = _safe_file(path)
        if p.suffix.lower() != ".svg":
            raise ApiError("only .svg files are served here")
        s = _read_svg(p)
        if s is None:
            raise ApiError("that drawing is too large", 413)
        return {"path": str(p), "svg": s}
    return _guard(run)
