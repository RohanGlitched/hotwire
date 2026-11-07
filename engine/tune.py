"""Pick standard part values that make the specs pass with the most margin.

Coordinate search over preferred (E-series) values: for each part in ``vary``,
try its neighbouring standard values, keep the best, repeat until no move
improves the score. The score first minimises total normalised violation, then
maximises the smallest normalised margin, so a passing design keeps getting more
robust instead of stopping at the first value that scrapes through.
"""
from __future__ import annotations

import math

from .netlist import Circuit, NetlistError, format_value
from .parts import neighbours, snap
from .specs import Evaluator, parse_spec

__all__ = ["tune"]


def _scale(s, measured):
    if s.op == "in":
        return max(abs(s.hi - s.lo) / 2, 1e-12)
    ref = abs(s.lo) if isinstance(s.lo, (int, float)) else 1.0
    if s.unit == "dB":
        return 3.0
    return max(ref * 0.1, 1e-12)


def _score(c: Circuit, parsed):
    ev = Evaluator(c)
    viol, worst = 0.0, float("inf")
    results = []
    for s in parsed:
        r = ev.evaluate(s)
        results.append(r)
        if r.margin is None or not math.isfinite(r.margin):
            viol += 10.0 if not r.passed else 0.0
            continue
        m = r.margin / _scale(s, r.measured)
        if m < 0:
            viol += -m
        worst = min(worst, m)
    return (viol, -worst if math.isfinite(worst) else 0.0), results


def tune(c: Circuit, vary: list, specs=None, series: str | None = None, rounds: int = 6,
         steps: int = 6) -> dict:
    texts = list(c.specs) + [s for s in (specs or []) if s not in c.specs]
    parsed = [parse_spec(t) for t in texts]
    if not parsed:
        raise NetlistError(0, "nothing to tune for: add '*@ spec' lines or pass specs")
    slow = [s.text for s in parsed if s.needs == "tran"]
    fast = [s for s in parsed if s.needs != "tran"]
    skipped = slow if fast else []
    parsed = fast or parsed  # search on fast specs; large-signal ones are re-checked at the end
    if not vary:
        raise NetlistError(0, "name the parts to tune, e.g. vary=['R3', 'C2'] (circuit_explain shows which parts set each spec)")
    names = []
    for v in vary:
        e = c.element(v)
        if e.kind not in ("R", "C", "L"):
            raise NetlistError(e.line, f"{e.name} isn't a resistor, capacitor or inductor; only those can be tuned")
        if e.pot:
            raise NetlistError(e.line, f"{e.name} is half of pot {e.pot}; tune the pot position instead")
        names.append(e.name)

    def ser(e):
        if series:
            return series
        return "E24" if e.kind == "R" else "E12"

    # start from snapped values
    cur = {n: snap(c.element(n).value, ser(c.element(n))) for n in names}
    before_score, before = _score(c, parsed)
    best_c = c.with_values(cur)
    best_score, best_res = _score(best_c, parsed)
    evals = 2
    for _ in range(rounds):
        improved = False
        for n in names:
            e = c.element(n)
            for cand in neighbours(cur[n], ser(e), steps):
                if cand == cur[n]:
                    continue
                trial = dict(cur)
                trial[n] = cand
                tc = c.with_values(trial)
                try:
                    sc, res = _score(tc, parsed)
                except Exception:
                    continue
                evals += 1
                if sc < best_score:
                    best_score, best_res, best_c, cur = sc, res, tc, trial
                    improved = True
        if not improved:
            break
    changes = [{"part": n, "from": format_value(c.element(n).value), "to": format_value(cur[n])}
               for n in names if abs(cur[n] - c.element(n).value) / c.element(n).value > 1e-6]
    return {
        "circuit": best_c,
        "values": {n: cur[n] for n in names},
        "changes": changes,
        "before": [r.as_dict() for r in before],
        "after": [r.as_dict() for r in best_res],
        "all_pass": all(r.passed for r in best_res),
        "evaluations": evals,
        "skipped_during_search": skipped,
    }
