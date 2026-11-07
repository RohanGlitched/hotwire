"""Part stress: is any part being asked to do more than its rating at idle?

Defaults when the netlist gives no ``*@ rating``:
- resistors: 0.25 W metal film, flagged above 50% (derating) and failed above 100%;
- capacitors: electrolytic (>= 1 uF) 25 V, film/ceramic 50 V; flagged above 80%;
  for electrolytics it reports which lead is + (SPICE capacitors carry no polarity,
  so the operating point decides how the part must go in);
- semiconductors: absolute maximums from the part library.
"""
from __future__ import annotations

from .netlist import Circuit, format_value

__all__ = ["check"]


def _rating_for(c: Circuit, name: str, unit: str, default: float) -> float:
    r = c.ratings.get(name)
    if r and r[1] == unit:
        return r[0]
    return default


def _part_rating(c: Circuit, e) -> dict:
    try:
        from .parts_data import PARTS
    except ImportError:
        return {}
    base = (e.model or "").split("__")[0]
    return dict((PARTS.get(base) or {}).get("rating") or {})


def _package(e) -> str:
    try:
        from .parts_data import PARTS
    except ImportError:
        return ""
    return str((PARTS.get((e.model or "").split("__")[0]) or {}).get("package", ""))


def _row(e, what, value, limit, unit, level, message):
    return {"part": e.name, "what": what, "value": float(value), "limit": float(limit),
            "unit": unit, "load_pct": round(100 * abs(value) / limit, 1) if limit else None,
            "level": level, "message": message}


def check(c: Circuit, op) -> list:
    rows = []
    v = op.v

    def vn(n):
        return 0.0 if n == "0" else float(v.get(n, 0.0))

    for e in c.elements:
        k = e.kind
        if k == "R" and e.value:
            p = abs(float(op.power.get(e.name, (vn(e.nodes[0]) - vn(e.nodes[1])) ** 2 / e.value)))
            lim = _rating_for(c, e.name, "W", 0.25)
            if p > lim:
                rows.append(_row(e, "power", p, lim, "W", "error",
                                 f"{e.name} dissipates {format_value(p, 'W')}, over its {format_value(lim, 'W')} rating: it will burn"))
            elif p > 0.5 * lim:
                rows.append(_row(e, "power", p, lim, "W", "warning",
                                 f"{e.name} runs at {100 * p / lim:.0f}% of its {format_value(lim, 'W')} rating; use a bigger resistor"))
        elif k == "C" and e.value:
            dv = vn(e.nodes[0]) - vn(e.nodes[1])
            electro = e.value >= 1e-6
            lim = _rating_for(c, e.name, "V", 25.0 if electro else 50.0)
            if abs(dv) > lim:
                rows.append(_row(e, "voltage", dv, lim, "V", "error",
                                 f"{e.name} sees {abs(dv):.2f} V, over its {lim:g} V rating"))
            elif abs(dv) > 0.8 * lim:
                rows.append(_row(e, "voltage", dv, lim, "V", "warning",
                                 f"{e.name} sees {abs(dv):.2f} V, {100 * abs(dv) / lim:.0f}% of its {lim:g} V rating"))
            if electro and abs(dv) >= 0.05:
                hi, lo = (e.nodes[0], e.nodes[1]) if dv > 0 else (e.nodes[1], e.nodes[0])
                rows.append(_row(e, "polarity", abs(dv), 0.0, "V", "info",
                                 f"{e.name} (electrolytic): + lead to '{hi}' ({vn(hi):.2f} V), − lead to '{lo}' ({vn(lo):.2f} V)"))
            elif electro and abs(dv) < 0.05 and e.value >= 1e-6:
                rows.append(_row(e, "polarity", abs(dv), 0.0, "V", "info",
                                 f"{e.name} (electrolytic) has almost no DC across it: use a bipolar (non-polarised) type"))
        elif k == "D":
            dev = op.devices.get(e.name, {})
            i = float(dev.get("id", 0.0))
            r = _part_rating(c, e)
            imax = r.get("imax_a")
            if imax and i > imax:
                rows.append(_row(e, "current", i, imax, "A", "error",
                                 f"{e.name} carries {format_value(i, 'A')}, over its {format_value(imax, 'A')} rating"))
            elif imax and i > 0.8 * imax:
                rows.append(_row(e, "current", i, imax, "A", "warning",
                                 f"{e.name} carries {format_value(i, 'A')}, {100 * i / imax:.0f}% of its rating"))
            vr = -float(dev.get("vd", 0.0))
            vmax = r.get("vmax_v")
            if vmax and vr > vmax:
                rows.append(_row(e, "reverse voltage", vr, vmax, "V", "error",
                                 f"{e.name} has {vr:.1f} V reverse across it, over its {vmax:g} V rating"))
        elif k == "Q":
            dev = op.devices.get(e.name, {})
            r = _part_rating(c, e)
            vce = abs(float(dev.get("vce", 0.0)))
            ic = abs(float(dev.get("ic", 0.0)))
            pd = vce * ic
            if r.get("vceo_v") and vce > r["vceo_v"]:
                rows.append(_row(e, "VCE", vce, r["vceo_v"], "V", "error",
                                 f"{e.name} has {vce:.1f} V across it, over its {r['vceo_v']:g} V VCEO"))
            if r.get("ic_a") and ic > r["ic_a"]:
                rows.append(_row(e, "IC", ic, r["ic_a"], "A", "error",
                                 f"{e.name} carries {format_value(ic, 'A')}, over its {format_value(r['ic_a'], 'A')} rating"))
            if r.get("pd_w"):
                if pd > r["pd_w"]:
                    rows.append(_row(e, "power", pd, r["pd_w"], "W", "error",
                                     f"{e.name} dissipates {format_value(pd, 'W')}, over its {format_value(r['pd_w'], 'W')} rating"))
                elif pd > 0.5 * r["pd_w"]:
                    rows.append(_row(e, "power", pd, r["pd_w"], "W", "warning",
                                     f"{e.name} dissipates {format_value(pd, 'W')}, {100 * pd / r['pd_w']:.0f}% of its rating; it will run hot"))
        elif k == "J":
            dev = op.devices.get(e.name, {})
            r = _part_rating(c, e)
            vds = abs(float(dev.get("vds", 0.0)))
            if r.get("vmax_v") and vds > r["vmax_v"]:
                rows.append(_row(e, "VDS", vds, r["vmax_v"], "V", "error",
                                 f"{e.name} has {vds:.1f} V drain to source, over its {r['vmax_v']:g} V rating"))
        elif k == "M":
            dev = op.devices.get(e.name, {})
            r = _part_rating(c, e)
            vds = abs(float(dev.get("vds", 0.0)))
            vgs = abs(float(dev.get("vgs", 0.0)))
            idd = abs(float(dev.get("id", 0.0)))
            pd = abs(float(dev.get("power", vds * idd)))
            if r.get("vds_v") and vds > r["vds_v"]:
                rows.append(_row(e, "VDS", vds, r["vds_v"], "V", "error",
                                 f"{e.name} has {vds:.1f} V drain to source, over its {r['vds_v']:g} V VDS rating"))
            if r.get("vgs_v") and vgs > r["vgs_v"]:
                rows.append(_row(e, "VGS", vgs, r["vgs_v"], "V", "error",
                                 f"{e.name} has {vgs:.1f} V gate to source, over its ±{r['vgs_v']:g} V rating: "
                                 "the gate oxide breaks down"))
            if r.get("id_a") and idd > r["id_a"]:
                rows.append(_row(e, "ID", idd, r["id_a"], "A", "error",
                                 f"{e.name} carries {format_value(idd, 'A')}, over its {format_value(r['id_a'], 'A')} rating"))
            pmax = _rating_for(c, e.name, "W", r.get("pd_w") or 0.0)
            if pmax:
                if pd > pmax:
                    rows.append(_row(e, "power", pd, pmax, "W", "error",
                                     f"{e.name} dissipates {format_value(pd, 'W')}, over its {format_value(pmax, 'W')} rating"
                                     + (" (free air; add '*@ rating' with a heatsink)" if "TO-220" in str(_package(e)) else "")))
                elif pd > 0.5 * pmax:
                    rows.append(_row(e, "power", pd, pmax, "W", "warning",
                                     f"{e.name} dissipates {format_value(pd, 'W')}, {100 * pd / pmax:.0f}% of its rating; it will run hot"))
        elif k == "X":
            r = _part_rating(c, e)
            span = abs(vn(e.nodes[2]) - vn(e.nodes[3]))
            if r.get("vmax_v") and span > r["vmax_v"]:
                rows.append(_row(e, "supply", span, r["vmax_v"], "V", "error",
                                 f"{e.name}'s supply span is {span:.1f} V, over its {r['vmax_v']:g} V maximum"))
            lo = r.get("vmin_v")
            if lo and span < lo:
                rows.append(_row(e, "supply", span, lo, "V", "warning",
                                 f"{e.name} gets {span:.1f} V, under its {lo:g} V minimum supply"))
    return rows
