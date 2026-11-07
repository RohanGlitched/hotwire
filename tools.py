"""Tool handlers. Each returns a JSON string and never raises.

Results carry a ``summary`` line, the structured data, and ``media`` paths. The
``show`` field lists ready-made ``MEDIA:`` lines for the model to paste into its
reply, which is how Hermes delivers files (Desktop renders them, messaging
platforms attach them).
"""
from __future__ import annotations

import json
import logging
import re
import traceback

logger = logging.getLogger(__name__)

_BENCH = None
_SAMPLE_RATE = 48000


def configure(sample_rate=None):
    """Called from register(ctx) with the user's plugin settings."""
    global _SAMPLE_RATE, _BENCH
    try:
        sr = int(sample_rate or 48000)
    except (TypeError, ValueError):
        sr = 48000
    _SAMPLE_RATE = sr if sr in (44100, 48000) else 48000
    _BENCH = None


def _emit(event: str, payload: dict):
    try:
        from hermes_cli.plugin_events import broadcast_plugin_event
        broadcast_plugin_event("hotwire", event, payload)
    except Exception:
        pass


def bench():
    global _BENCH
    if _BENCH is None:
        from .engine.workbench import Bench
        _BENCH = Bench(emit=_emit, sample_rate=_SAMPLE_RATE)
    return _BENCH


def _media_lines(result: dict) -> list:
    files = list(result.get("media") or [])
    # chat apps can't show SVG: offer the PNG twins there; Desktop gets the SVG
    pngs = result.get("png") or []
    lines = [f"MEDIA:{f}" for f in files]
    if pngs:
        result["png_for_chat_apps"] = [f"MEDIA:{p}" for p in pngs]
    return lines


def _run(fn, args: dict, **call):
    try:
        result = fn(**call)
        if isinstance(result, dict) and result.get("error"):
            return json.dumps({"ok": False, **result}, default=str)
        if isinstance(result, dict):
            show = _media_lines(result)
            if show:
                result["show"] = show
                result["show_hint"] = ("Put these MEDIA: lines, each on its own line, in your reply so the user "
                                       "sees/hears them (on Telegram/Slack/Discord use png_for_chat_apps for images).")
        return json.dumps({"ok": True, **result}, default=str)
    except FileNotFoundError as e:
        return json.dumps({"ok": False, "error": f"can't find the file {e.filename}"})
    except Exception as e:  # NetlistError / ValueError / LayoutError carry plain-English messages
        msg = getattr(e, "message", None) or str(e)
        line = getattr(e, "line", None)
        expected = getattr(e, "message", None) or isinstance(e, ValueError) or type(e).__name__.endswith("Error") and msg
        if not expected:
            logger.warning("hotwire tool failed: %s", traceback.format_exc())
        return json.dumps({"ok": False, "error": msg, **({"line": line} if line else {})})


def _num(v, default, name):
    if v is None or v == "":
        return default
    try:
        return float(str(v).strip().rstrip("VvsS% ").strip()) if isinstance(v, str) else float(v)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {v!r}")


def _list(v, words=False):
    """Accept a list, a JSON list sent as a string, or a comma/newline separated string."""
    if v is None:
        return None
    if isinstance(v, str):
        st = v.strip()
        if st.startswith("["):
            try:
                got = json.loads(st)
                if isinstance(got, list):
                    return [str(x) for x in got] or None
            except ValueError:
                pass
        sep = r"[\n;\s]+|,(?![^()]*\))" if words else r"[\n;]|,(?![^()]*\))"
        parts = [x.strip() for x in re.split(sep, st) if x and x.strip()]
        return parts or None
    return list(v) or None


def _fraction(v, name):
    """0..1 from 0.7, "0.7", "70%" or 70."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().endswith("%"):
        x = _num(v.strip()[:-1], None, name) / 100.0
    else:
        x = _num(v, None, name)
        if x is not None and 1.0 < x <= 100.0:
            x = x / 100.0
    if x is None or not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 (or a percentage), got {v!r}")
    return x


def _src(args: dict) -> dict:
    return {"netlist": args.get("netlist"), "design": args.get("design"), "path": args.get("path"),
            "pots": args.get("pots") or None}


def circuit_check(args: dict, **kwargs) -> str:
    return _run(lambda **kw: bench().check(**kw), args, **_src(args), name=args.get("name"),
                specs=_list(args.get("specs")), note=args.get("note") or "")


def circuit_simulate(args: dict, **kwargs) -> str:
    def go(**kw):
        extra = {k: _num(args[k], None, k) for k in ("fstart", "fstop", "freq", "amplitude", "cycles")
                 if args.get(k) is not None}
        if args.get("nodes"):
            extra["nodes"] = _list(args.get("nodes"))
        return bench().simulate(**kw, **extra)
    return _run(go, args, analysis=str(args.get("analysis") or "op").lower(), **_src(args))


def circuit_audition(args: dict, **kwargs) -> str:
    def go(**kw):
        level = _num(args.get("level"), 0.2, "level")
        if level <= 0 or level > 20:
            raise ValueError(f"level is the input peak in volts and must be above 0 and at most 20 (got {level:g})")
        pickup = str(args.get("pickup") or "single_coil").lower().replace("-", "_").replace(" ", "_")
        pickup = {"p90": "p90", "p_90": "p90", "single": "single_coil", "singlecoil": "single_coil",
                  "hb": "humbucker"}.get(pickup, pickup)
        if pickup not in ("single_coil", "humbucker", "p90"):
            raise ValueError("pickup must be single_coil, humbucker or p90")
        return bench().audition(level=level, seconds=_num(args.get("seconds"), 4.0, "seconds"),
                                guitar_volume=_fraction(args.get("guitar_volume"), "guitar_volume"),
                                pickup=pickup, **kw)
    return _run(go, args, **_src(args), name=args.get("name"),
                source=args.get("source") or "guitar_riff", sweep=args.get("sweep") or None)


def circuit_tune(args: dict, **kwargs) -> str:
    return _run(lambda **kw: bench().tune(**kw), args, vary=_list(args.get("vary"), words=True) or [], **_src(args),
                name=args.get("name"), specs=_list(args.get("specs")), series=args.get("series"))


def circuit_explain(args: dict, **kwargs) -> str:
    return _run(lambda **kw: bench().explain(**kw), args, **_src(args), specs=_list(args.get("specs")))


def circuit_match(args: dict, **kwargs) -> str:
    return _run(lambda **kw: bench().match(level=_num(args.get("level"), 0.2, "level"), **kw), args,
                dry=args.get("dry"), wet=args.get("wet"),
                netlist=args.get("netlist"), design=args.get("design"), path=args.get("path"))


def circuit_yield(args: dict, **kwargs) -> str:
    return _run(lambda **kw: bench().yield_(n=int(_num(args.get("n"), 200, "n")),
                                            seed=int(_num(args.get("seed"), 1, "seed")), **kw), args, **_src(args))


def circuit_bom(args: dict, **kwargs) -> str:
    return _run(bench().bom, args, **_src(args), series=args.get("series") or "E24")


def circuit_build(args: dict, **kwargs) -> str:
    def go(**kw):
        mw = int(_num(args.get("max_width"), 30, "max_width"))
        if not 8 <= mw <= 80:
            raise ValueError(f"max_width must be between 8 and 80 holes (got {mw})")
        return bench().build(max_width=mw, **kw)
    return _run(go, args, **_src(args), name=args.get("name"))


def circuit_export(args: dict, **kwargs) -> str:
    return _run(bench().export, args, fmt=(args.get("format") or "spice").lower(), **_src(args), name=args.get("name"))


def circuit_library(args: dict, **kwargs) -> str:
    return _run(bench().library, args, query=args.get("query"))


# --------------------------------------------------------------------------- /hotwire

_HELP = """Hotwire: analog circuit bench
  /hotwire                  this help + recent designs
  /hotwire library          reference circuits you can start from
  /hotwire designs          your recent designs and their verdicts
  /hotwire check <file.cir> check a netlist file (or a library id)
  /hotwire <library id>     check a reference circuit, e.g. /hotwire softclip_overdrive
Or just ask: "design a 9 V fuzz that cleans up with the guitar volume, and let me hear it"."""


def slash_command(raw_args: str = "", **kwargs) -> str:
    try:
        args = (raw_args or "").strip()
        b = bench()
        if args in ("", "help"):
            recent = b.store.list(5)
            lines = [_HELP]
            if recent:
                lines.append("\nRecent designs:")
                for m in recent:
                    last = m["versions"][-1] if m.get("versions") else {}
                    lines.append(f"  {m['id']}  v{m['latest']}  {last.get('verdict', '')}  {m.get('name', '')}")
            return "\n".join(lines)
        if args == "library":
            lib = b.library()
            return "Reference circuits:\n" + "\n".join(
                f"  {c['id']:<28} {c.get('title') or ''}" for c in lib["circuits"])
        if args == "designs":
            rows = b.store.list(20)
            if not rows:
                return "No designs yet. Ask Hermes to design a circuit, or run /hotwire <library id>."
            return "Designs:\n" + "\n".join(
                f"  {m['id']:<28} v{m['latest']:<3} {(m['versions'][-1].get('verdict') or ''):<20} {m.get('name', '')}"
                for m in rows)
        target = args[6:].strip() if args.startswith("check ") else args
        r = b.check(path=target)
        out = [f"{r['title']}: {r.get('verdict', '?').upper()}  ({r.get('design')} v{r.get('version')})", r.get("summary", "")]
        for s in r.get("specs", []):
            out.append(f"  {'PASS' if s['pass'] else 'FAIL'}  {s['spec']:<28} {s['explanation']}")
        for s in r.get("stress", []):
            out.append(f"  {s['level'].upper():<7} {s['message']}")
        for m in r.get("media", []):
            out.append(f"  file: {m}")
        return "\n".join(out)
    except Exception as e:
        return f"Hotwire: {getattr(e, 'message', None) or e}"
