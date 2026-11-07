"""Monte Carlo yield: build the circuit a few hundred times with real-world part spread.

Each virtual build draws every resistor, capacitor and inductor uniformly inside its
tolerance, transistor current gain (BF) and saturation current (IS) from clipped
normal / log-normal spreads, and JFET pinch-off (VTO) within an absolute window.
The result says what fraction of builds meet every spec and which parts drive the
failures, so a designer knows what to tighten.
"""
from __future__ import annotations

import math

import numpy as np

from .netlist import Circuit
from .specs import Evaluator, parse_spec

__all__ = ["run"]


def _draws(c: Circuit, rng, tol: dict):
    """One build: (changes dict for Circuit.with_values, relative deviations per param)."""
    changes, dev = {}, {}
    for e in c.elements:
        if e.kind in ("R", "C", "L") and e.value and not e.pot:
            t = tol.get(e.kind, 0.05)
            u = rng.uniform(-1, 1)
            changes[e.name] = e.value * (1 + t * u)
            dev[e.name] = u
    seen = set()
    for e in c.elements:
        if e.kind not in ("Q", "D", "J", "M") or not e.model:
            continue
        m = c.models.get(e.model)
        if not m:
            continue
        if e.kind == "Q" and "BF" in m.params:
            t = tol.get("BF", 0.4)
            z = float(np.clip(rng.normal(0, 0.5), -1, 1))
            changes[f"{e.name}.BF"] = max(5.0, m.params["BF"] * (1 + t * z))
            dev[f"{e.name}.BF"] = z
        if e.kind in ("Q", "D") and "IS" in m.params:
            t = tol.get("IS", 0.5)
            z = float(np.clip(rng.normal(0, 0.5), -1, 1))
            changes[f"{e.name}.IS"] = m.params["IS"] * math.exp(math.log1p(t) * z)
            dev[f"{e.name}.IS"] = z
        if e.kind == "J":
            t = tol.get("VTO", 0.5)
            z = float(np.clip(rng.normal(0, 0.5), -1, 1))
            vto = m.params.get("VTO", -2.0)
            changes[f"{e.name}.VTO"] = vto + t * z
            dev[f"{e.name}.VTO"] = z
        if e.kind == "M":   # MOSFET: threshold +/-0.3 V absolute (*@ tol VTH=), KP +/-20% (*@ tol KP=)
            t = tol.get("VTH", 0.3)
            z = float(np.clip(rng.normal(0, 0.5), -1, 1))
            changes[f"{e.name}.VTO"] = m.params.get("VTO", 0.0) + t * z
            dev[f"{e.name}.VTO"] = z
            t = tol.get("KP", 0.2)
            z = float(np.clip(rng.normal(0, 0.5), -1, 1))
            changes[f"{e.name}.KP"] = m.params.get("KP", 2e-5) * (1 + t * z)
            dev[f"{e.name}.KP"] = z
        seen.add(e.model)
    return changes, dev


def run(c: Circuit, specs=None, n: int = 200, seed: int = 1, tol: dict | None = None,
        max_seconds: float = 60.0) -> dict:
    import time

    texts = list(c.specs) + [s for s in (specs or []) if s not in c.specs]
    parsed = []
    for t in texts:
        try:
            parsed.append(parse_spec(t))
        except Exception:
            continue
    if not parsed:
        return {"error": "no specs to test: add '*@ spec' lines or pass specs", "yield_pct": None}
    heavy = any(s.needs == "tran" for s in parsed)
    if int(n) < 10:
        return {"error": "n must be at least 10 virtual builds (200 is a good default)", "yield_pct": None}
    n = int(min(n, 1000))
    if heavy:
        n = min(n, 60)
    tol = dict(c.tol, **(tol or {}))
    rng = np.random.default_rng(seed)

    passes = 0
    fail_counts = {s.text: 0 for s in parsed}
    margins = {s.text: [] for s in parsed}
    devs: dict = {}
    done = 0
    worst = None
    t0 = time.time()
    for i in range(n):
        changes, dev = _draws(c, rng, tol)
        try:
            ci = c.with_values(changes)
            ev = Evaluator(ci)
            res = [ev.evaluate(s) for s in parsed]
        except Exception:
            res = None
        done += 1
        ok_all = res is not None and all(r.passed for r in res)
        passes += ok_all
        for k, v in dev.items():
            devs.setdefault(k, []).append(v)
        for j, s in enumerate(parsed):
            r = res[j] if res else None
            if r is None or not r.passed:
                fail_counts[s.text] += 1
            m = r.margin if (r and r.margin is not None and math.isfinite(r.margin)) else float("nan")
            margins[s.text].append(m)
        if res:
            score = min((r.margin for r in res if r.margin is not None), default=None)
            if score is not None and (worst is None or score < worst[0]):
                worst = (score, {k: round(v, 6) for k, v in changes.items() if "." in k or k[0] in "RCL"})
        if time.time() - t0 > max_seconds:
            break

    # sensitivity: |correlation| of each parameter's deviation with each spec margin
    sens = {}
    for s in parsed:
        y = np.asarray(margins[s.text], dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 8 or np.std(y[mask]) == 0:
            continue
        for k, xs in devs.items():
            x = np.asarray(xs, dtype=float)[: len(y)]
            if len(x) != len(y) or np.std(x[mask]) == 0:
                continue
            r = abs(float(np.corrcoef(x[mask], y[mask])[0, 1]))
            if math.isfinite(r):
                sens[k] = max(sens.get(k, 0.0), r * r)  # explained variance share
    total = sum(sens.values()) or 1.0
    ranked = sorted(sens.items(), key=lambda kv: -kv[1])[:6]
    sensitivity = [{"param": k, "part": k.split(".")[0], "share_pct": round(100 * v / total, 1)} for k, v in ranked]

    by_spec = []
    for s in parsed:
        ys = np.asarray(margins[s.text], dtype=float)
        ys = ys[np.isfinite(ys)]
        by_spec.append({
            "spec": s.text,
            "fail_pct": round(100 * fail_counts[s.text] / done, 1),
            "margin_p5": float(f"{np.percentile(ys, 5):.4g}") if len(ys) else None,
            "margin_median": float(f"{np.median(ys):.4g}") if len(ys) else None,
        })
    # Wilson score interval: how far the true yield could be from this estimate
    z = 1.96
    p_hat = passes / done
    den = 1 + z * z / done
    centre = (p_hat + z * z / (2 * done)) / den
    half = z * math.sqrt(p_hat * (1 - p_hat) / done + z * z / (4 * done * done)) / den
    ci = [round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)]
    return {
        "yield_pct": round(100 * p_hat, 1),
        "ci95_pct": ci,
        "builds": done,
        "requested": n,
        "seed": seed,
        "tolerances": {k: v for k, v in tol.items()},
        "by_spec": by_spec,
        "sensitivity": sensitivity,
        "note": (("large-signal specs need a time simulation per build, so this run used "
                  f"{done} builds. " if heavy else "")
                 + f"95% interval {ci[0]:g}–{ci[1]:g}%: don't rank two designs whose intervals overlap."),
    }
