"""Measurable specs: parse ``*@ spec`` lines and evaluate them with the simulator.

Grammar (one spec per string)::

    metric(args) <op> value[unit]        ops: < <= > >= == ==~ (within 5%)
    metric(args) in lo..hi
    region(Q1) == active

Every result carries the measured value, the target, the margin (positive when
passing, in the metric's unit) and a plain-English sentence, so the agent can
explain a failure without reading code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .netlist import Circuit, NetlistError, format_value, parse_value

__all__ = ["Spec", "SpecResult", "parse_spec", "Evaluator", "evaluate", "METRICS"]

# metric -> (unit, needs, description)
METRICS = {
    "gain_db": ("dB", "ac", "small-signal gain from input to output at a frequency"),
    "phase": ("°", "ac", "phase shift at a frequency"),
    "fc": ("Hz", "ac", "-3 dB corner of a low-pass or high-pass response"),
    "f3db_low": ("Hz", "ac", "lower -3 dB point relative to the passband peak"),
    "f3db_high": ("Hz", "ac", "upper -3 dB point relative to the passband peak"),
    "peak_db": ("dB", "ac", "highest gain anywhere in the band"),
    "peak_f": ("Hz", "ac", "frequency of the highest gain"),
    "v": ("V", "op", "DC voltage at a node (to ground) at idle"),
    "i": ("A", "op", "DC current through an element at idle"),
    "p": ("W", "op", "DC power in an element at idle"),
    "supply_current": ("A", "op", "total current drawn from the supply sources at idle"),
    "battery_hours": ("h", "op", "hours on a battery of the given capacity (mAh)"),
    "thd": ("%", "tran", "total harmonic distortion for a sine input (f, peak volts)"),
    "vout_peak": ("V", "tran", "output peak for a sine input (f, peak volts)"),
    "gain_tran": ("dB", "tran", "large-signal gain for a sine input (f, peak volts)"),
    "headroom": ("V", "tran", "largest input peak before THD passes 1% (f)"),
    "thd_gv": ("%", "tran", "THD through a guitar pickup with its volume knob at a setting (f, pickup peak volts, volume 0..1)"),
    "gain_gv": ("dB", "tran", "large-signal gain through a guitar with its volume at a setting (f, pickup peak volts, volume 0..1)"),
    "zin": ("Ω", "ac", "input impedance seen by the source at a frequency (what loads a guitar pickup)"),
    "cleanup_vol": ("", "tran", "guitar volume setting (0..1) below which THD drops under a threshold (f, pickup peak volts, THD %)"),
    "region": ("", "op", "operating region of a transistor at idle"),
}
_SPEC_RE = re.compile(
    r"^\s*(?P<metric>[a-z_0-9]+)\s*(?:\((?P<args>[^)]*)\))?\s*"
    r"(?:(?P<op><=|>=|==~|==|<|>|=)\s*(?P<val>[^\s].*?)|in\s+(?P<lo>\S+?)\s*\.\.\s*(?P<hi>[^\s]+(?:\s*[A-Za-z%°]+)?))\s*$",
    re.I,
)


@dataclass
class Spec:
    text: str
    metric: str
    args: list
    op: str            # '<' '<=' '>' '>=' '==' '==~' 'in'
    lo: float | str | None
    hi: float | None = None

    @property
    def unit(self) -> str:
        return METRICS[self.metric][0]

    @property
    def needs(self) -> str:
        return METRICS[self.metric][1]


@dataclass
class SpecResult:
    spec: str
    metric: str
    measured: float | str | None
    target: str
    passed: bool
    margin: float | None
    unit: str
    explanation: str
    error: str | None = None
    extra: dict = field(default_factory=dict)
    label: str = ""

    def as_dict(self) -> dict:
        d = {
            "spec": self.spec, "label": self.label or self.spec, "metric": self.metric,
            "measured": _round(self.measured), "measured_text": _fmt(self.measured, self.unit),
            "target": self.target, "target_text": self.target, "pass": self.passed,
            "margin": _round(self.margin), "unit": self.unit, "explanation": self.explanation,
        }
        if self.error:
            d["error"] = self.error
        if self.extra:
            d.update(self.extra)
        return d


def _round(x):
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return float(f"{x:.4g}")
    return x


def _num(text: str, unit: str) -> float:
    t = text.strip()
    if t.endswith("%"):
        return float(parse_value(t[:-1]))
    # strip a trailing unit that matches the metric's own unit (dB, Hz, V, A, W, h)
    for u in ("dB", "db", "Hz", "hz", "°", "deg"):
        if t.endswith(u) and len(t) > len(u):
            t = t[: -len(u)]
            break
    t = t.strip()
    if t.endswith("%"):
        return float(parse_value(t[:-1]))
    return parse_value(t)


def parse_spec(text: str) -> Spec:
    m = _SPEC_RE.match(text or "")
    if not m:
        raise NetlistError(0, f"can't read spec '{text}'. Examples: 'gain_db(1k) > 18', 'fc in 2k..3k', 'region(Q1) == active'")
    metric = m.group("metric").lower()
    metric = {"bias": "region", "gain": "gain_db", "f3db": "fc", "i_supply": "supply_current",
              "current": "supply_current"}.get(metric, metric)
    if metric not in METRICS:
        raise NetlistError(0, f"unknown spec metric '{metric}' (known: {', '.join(METRICS)})")
    raw_args = [a.strip() for a in (m.group("args") or "").split(",") if a.strip()]
    unit = METRICS[metric][0]
    if m.group("op"):
        op = m.group("op")
        op = "==" if op == "=" else op
        val = m.group("val").strip()
        if metric == "region":
            return Spec(text, metric, raw_args, "==", val.lower())
        return Spec(text, metric, raw_args, op, _num(val, unit))
    return Spec(text, metric, raw_args, "in", _num(m.group("lo"), unit), _num(m.group("hi"), unit))


# --------------------------------------------------------------------------- evaluation

def _fmt(v, unit: str) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if unit == "":
        return f"{v:.3g}"
    if unit in ("%", "°"):
        return f"{v:.3g}{unit}"
    if unit in ("dB", "h"):
        return f"{v:.3g} {unit}"
    return format_value(v, unit)


class Evaluator:
    """Evaluates specs on one circuit, caching each analysis it runs."""

    AC_START, AC_STOP, AC_PPD = 1.0, 200e3, 60

    def __init__(self, circuit: Circuit, sr: int = 96000):
        self.c = circuit
        self.sr = sr
        self._op = None
        self._ac = None
        self._tran: dict = {}
        self._guitar: dict = {}

    # -- cached analyses ---------------------------------------------------
    @property
    def op(self):
        if self._op is None:
            from .analyses import op
            self._op = op(self.c)
        return self._op

    @property
    def acr(self):
        if self._ac is None:
            from .analyses import ac
            if not self.c.input or not self.c.output:
                raise NetlistError(0, "AC specs need '*@ in <source>' and '*@ out <node>' in the netlist")
            self._ac = ac(self.c, fstart=self.AC_START, fstop=self.AC_STOP, points_per_decade=self.AC_PPD)
        return self._ac

    def _gain_curve(self):
        r = self.acr
        f = np.asarray(r.f, dtype=float)
        mag = np.abs(np.asarray(r.h))
        db = 20 * np.log10(np.maximum(mag, 1e-30))
        return f, db, np.angle(np.asarray(r.h), deg=True)

    def gain_at(self, freq: float) -> float:
        f, db, _ = self._gain_curve()
        return float(np.interp(math.log10(freq), np.log10(f), db))

    def phase_at(self, freq: float) -> float:
        f, _, ph = self._gain_curve()
        return float(np.interp(math.log10(freq), np.log10(f), np.unwrap(np.radians(ph)) * 180 / math.pi))

    def passband_corner(self, lowpass: bool = True):
        f, db, _ = self._gain_curve()
        ref = float(db[0] if lowpass else db[-1])
        thr = ref - 10 * math.log10(2)
        idx = range(len(f)) if lowpass else range(len(f) - 1, -1, -1)
        prev = None
        for j in idx:
            if prev is not None and (db[prev] - thr) * (db[j] - thr) <= 0 and db[j] < thr:
                fa, fb = sorted((float(f[prev]), float(f[j])))
                ff, dd = self._local(fa, fb, 120)
                for i in range(len(ff) - 1):
                    if (dd[i] - thr) * (dd[i + 1] - thr) <= 0:
                        return _xcross(ff[i], ff[i + 1], dd[i], dd[i + 1], thr)
                return _xcross(f[prev], f[j], db[prev], db[j], thr) if lowpass else _xcross(f[j], f[prev], db[j], db[prev], thr)
            prev = j
        return None

    def _local(self, fa: float, fb: float, n: int = 80):
        """Fine AC sweep between two frequencies (for sharp peaks and exact corners)."""
        from .analyses import ac
        ppd = max(40, int(n / max(math.log10(fb / fa), 1e-6)))
        r = ac(self.c, fstart=fa, fstop=fb, points_per_decade=ppd)
        f = np.asarray(r.f, dtype=float)
        db = 20 * np.log10(np.maximum(np.abs(np.asarray(r.h)), 1e-30))
        return f, db

    def corners(self):
        """(f_low, f_high, peak_db, peak_f); None where the response never drops 3 dB.
        The coarse sweep finds them; a fine local sweep pins each one down."""
        if getattr(self, "_corners", None) is not None:
            return self._corners
        f, db, _ = self._gain_curve()
        k = int(np.argmax(db))
        peak, pf = float(db[k]), float(f[k])
        try:
            if 0 < k < len(f) - 1:
                fa, fb = float(f[k - 1]), float(f[k + 1])
                for _ in range(6):  # zoom in until the peak is resolved (Q of thousands)
                    ff, dd = self._local(fa, fb, 60)
                    j = int(np.argmax(dd))
                    if dd[j] > peak:
                        peak, pf = float(dd[j]), float(ff[j])
                    fa, fb = float(ff[max(j - 1, 0)]), float(ff[min(j + 1, len(ff) - 1)])
                    if fb / fa < 1 + 1e-7:
                        break
        except Exception:
            pass
        thr = peak - 10 * math.log10(2)  # half-power point (3.0103 dB), as in 1/(2πRC)

        def gain_at_exact(freq):
            ff, dd = self._local(freq, freq * 1.0000001, 2)
            return float(dd[0])

        def bisect(fa, fb):
            """fa below the threshold, fb above it (or vice versa): find the crossing in log f."""
            la, lb = math.log10(fa), math.log10(fb)
            ga = gain_at_exact(fa) - thr
            for _ in range(48):
                lm = (la + lb) / 2
                gm = gain_at_exact(10 ** lm) - thr
                if (gm < 0) == (ga < 0):
                    la, ga = lm, gm
                else:
                    lb = lm
                if abs(lb - la) < 1e-9:
                    break
            return float(10 ** ((la + lb) / 2))

        lo = hi = None
        try:
            left = [j for j in range(k, -1, -1) if db[j] < thr and f[j] < pf]
            if left:
                lo = bisect(float(f[left[0]]), pf)
            right = [j for j in range(k, len(f)) if db[j] < thr and f[j] > pf]
            if right:
                hi = bisect(pf, float(f[right[0]]))
        except Exception:
            lo = hi = None
            for j in range(k, 0, -1):
                if db[j - 1] < thr <= db[j]:
                    lo = _xcross(f[j - 1], f[j], db[j - 1], db[j], thr)
                    break
            for j in range(k, len(f) - 1):
                if db[j + 1] < thr <= db[j]:
                    hi = _xcross(f[j], f[j + 1], db[j], db[j + 1], thr)
                    break
        self._corners = (lo, hi, peak, pf)
        return self._corners

    def sine_run(self, freq: float, vin: float):
        """Steady-state output for a sine input: (t, out) over whole measured cycles."""
        key = (round(freq, 6), round(vin, 9))
        if key in self._tran:
            return self._tran[key]
        from .analyses import process_audio
        if not self.c.input or not self.c.output:
            raise NetlistError(0, "large-signal specs need '*@ in <source>' and '*@ out <node>'")
        sr = 48000 if freq <= 2500 else self.sr
        settle = max(0.04, 8 / freq)
        n_meas_cycles = max(4, int(round(0.02 * freq)))
        period_samples = sr / freq
        n_meas = int(round(n_meas_cycles * period_samples))
        n_settle = int(round(settle * sr))
        t = np.arange(n_settle + n_meas) / sr
        x = vin * np.sin(2 * math.pi * freq * t)
        # fade in over 5 ms so the coupling caps don't see a step
        fade = np.minimum(1.0, t / 0.005)
        if getattr(self, "_steady", None) is None:
            # steady-state measurement: capacitor initial conditions (a start-up transient)
            # don't belong in a THD or gain reading, so start from the operating point
            import copy
            self._steady = copy.deepcopy(self.c)
            for e in self._steady.elements:
                e.params.pop("ic", None)
        y = process_audio(self._steady, x * fade, sr)
        out = np.asarray(y[n_settle:], dtype=float)
        self._tran[key] = (t[n_settle:], out, n_meas_cycles)
        return self._tran[key]

    def thd_pct(self, freq: float, vin: float) -> float:
        _, y, cycles = self.sine_run(freq, vin)
        y = y - np.mean(y)
        spec = np.abs(np.fft.rfft(y))
        fund = spec[cycles] if cycles < len(spec) else 0.0
        if fund <= 1e-9 * max(1, len(y)):
            raise NetlistError(0, f"no {freq:g} Hz signal reaches the output, so distortion can't be measured "
                                  "(check the signal path from the input to the output)")
        harm = [spec[k * cycles] for k in range(2, 16) if k * cycles < len(spec)]
        return float(100 * math.sqrt(sum(h * h for h in harm)) / fund)

    def vout_peak(self, freq: float, vin: float) -> float:
        _, y, _ = self.sine_run(freq, vin)
        y = y - np.mean(y)
        return float(max(np.max(y), -np.min(y)))

    def headroom(self, freq: float, limit_pct: float = 1.0) -> float:
        lo, hi = 0.005, 5.0
        if self.thd_pct(freq, lo) > limit_pct:
            return lo
        if self.thd_pct(freq, hi) <= limit_pct:
            return hi
        for _ in range(7):
            mid = math.sqrt(lo * hi)
            if self.thd_pct(freq, mid) > limit_pct:
                hi = mid
            else:
                lo = mid
        return lo

    def input_impedance(self, freq: float) -> float:
        """|Zin| at ``freq``: put a 1 kΩ test resistor in series with the input source and
        compare the voltages on either side of it."""
        import copy
        from .analyses import ac
        from .netlist import Element
        if not self.c.input:
            raise NetlistError(0, "zin needs '*@ in <source>'")
        t = copy.deepcopy(self.c)
        src = t.element(t.input)
        k = 0 if src.nodes[0] != "0" else 1
        hot = src.nodes[k]
        src.nodes = list(src.nodes)
        src.nodes[k] = "zin_test"
        t.elements.insert(t.elements.index(src) + 1, Element("RZIN_TEST", "R", ["zin_test", hot], 1e3, line=src.line))
        r = ac(t, fstart=freq, fstop=freq * 1.0001, points_per_decade=40)
        vs = complex(r.v.get("zin_test", [1.0])[0])
        vh = complex(r.v.get(hot, [0.0])[0]) if hot != "0" else 0j
        if abs(vs - vh) < 1e-15:
            return float("inf")
        return float(abs(1e3 * vh / (vs - vh)))

    def _guitar_eval(self, vol: float) -> "Evaluator":
        key = round(vol, 3)
        sub = self._guitar.get(key)
        if sub is None:
            from .guitar import with_guitar
            sub = self._guitar[key] = Evaluator(with_guitar(self.c, vol), self.sr)
        return sub

    def cleanup_volume(self, freq: float, vin: float, limit_pct: float = 5.0) -> float:
        """Highest guitar-volume setting at which THD stays under ``limit_pct``.
        1.0 means it never distorts that much; 0.0 means it distorts even near silence."""
        if self._guitar_eval(1.0).thd_pct(freq, vin) <= limit_pct:
            return 1.0
        lo, hi = 0.02, 1.0
        if self._guitar_eval(lo).thd_pct(freq, vin) > limit_pct:
            return 0.0
        for _ in range(7):
            mid = (lo + hi) / 2
            if self._guitar_eval(mid).thd_pct(freq, vin) > limit_pct:
                hi = mid
            else:
                lo = mid
        return round(lo, 3)

    def supply_current(self) -> float:
        """Current drawn from the most heavily loaded supply rail.

        Each source counts only when it delivers power (P = -V·I > 0, SPICE signs), so a
        negative rail written ``VN vn 0 -12`` and a positive one can't cancel. For a single
        9 V supply this is the battery current; for ±V rails it's the per-rail current."""
        names = self.c.supplies or [e.name for e in self.c.elements
                                    if e.kind == "V" and e.name != self.c.input and (e.value or 0) != 0]
        pos = neg = 0.0
        for n in names:
            i = float(self.op.i.get(n, 0.0))
            try:
                e = self.c.element(n)
                v = float(e.value or 0.0)
                if e.nodes[0] == "0" and e.nodes[1] != "0":
                    v = -v  # written "V 0 rail value": the rail sits at -value
            except Exception:
                v = 0.0
            if -float(self.c.element(n).value or 0.0) * i > 0 or v == 0:
                if v >= 0:
                    pos += abs(i)
                else:
                    neg += abs(i)
        return max(pos, neg)

    # -- one spec ----------------------------------------------------------
    def measure(self, s: Spec):
        a = s.args
        m = s.metric
        if m in ("gain_db", "phase"):
            f = parse_value(a[0]) if a else 1000.0
            return self.gain_at(f) if m == "gain_db" else self.phase_at(f)
        if m in ("fc", "f3db_low", "f3db_high", "peak_db", "peak_f"):
            lo, hi, peak, pf = self.corners()
            if m == "peak_db":
                return peak
            if m == "peak_f":
                return pf
            if m == "f3db_low":
                return lo
            if m == "f3db_high":
                return hi
            # fc: the -3 dB point from the passband (low-pass: the gain at low frequency;
            # high-pass: the gain at high frequency), so a resonant peak doesn't move it
            f, db, _ = self._gain_curve()
            return self.passband_corner(lowpass=db[0] >= db[-1])
        if m == "v":
            if not a:
                raise NetlistError(0, "v() needs a node, e.g. v(c1)")
            node = a[0].lower()
            node = "0" if node in ("gnd", "0") else node
            for nm in [node] + ([a[1].lower()] if len(a) > 1 else []):
                nm = "0" if nm in ("gnd", "0") else nm
                if nm != "0" and nm not in self.op.v:
                    raise NetlistError(0, f"v(...): no node named {nm} (nodes: {', '.join(list(self.op.v)[:12])})")
            if len(a) > 1:  # v(a, b) differential
                other = "0" if a[1].lower() in ("gnd", "0") else a[1].lower()
                return float(self.op.v.get(node, 0.0)) - float(self.op.v.get(other, 0.0))
            return float(self.op.v.get(node, 0.0))
        if m in ("i", "p"):
            if not a:
                raise NetlistError(0, f"{m}() needs an element, e.g. {m}(R1)")
            name = a[0].upper()
            self.c.element(name)
            if m == "i":
                val = self.op.i.get(name)
                if val is None:
                    dev = self.op.devices.get(name, {})
                    val = dev.get("ic", dev.get("id", 0.0))
                return float(val)
            return float(self.op.power.get(name, 0.0))
        if m == "supply_current":
            return self.supply_current()
        if m == "battery_hours":
            cap = parse_value(a[0]) if a else 500.0
            ma = self.supply_current() * 1000
            if ma <= 1e-6:
                raise NetlistError(0, "battery_hours: the circuit draws no current from a supply, so battery life can't be measured (is the supply named with '*@ supply'?)")
            return cap / ma
        if m in ("thd", "vout_peak", "gain_tran"):
            f = parse_value(a[0]) if a else 1000.0
            vin = parse_value(a[1]) if len(a) > 1 else 0.2
            if m == "thd":
                return self.thd_pct(f, vin)
            vp = self.vout_peak(f, vin)
            return vp if m == "vout_peak" else 20 * math.log10(max(vp, 1e-12) / vin)
        if m == "headroom":
            f = parse_value(a[0]) if a else 1000.0
            return self.headroom(f)
        if m == "zin":
            f = parse_value(a[0]) if a else 1000.0
            return self.input_impedance(f)
        if m == "cleanup_vol":
            f = parse_value(a[0]) if a else 300.0
            vin = parse_value(a[1]) if len(a) > 1 else 0.1
            limit = parse_value(a[2].rstrip("%")) if len(a) > 2 else 5.0
            return self.cleanup_volume(f, vin, limit)
        if m in ("thd_gv", "gain_gv"):
            f = parse_value(a[0]) if a else 500.0
            vin = parse_value(a[1]) if len(a) > 1 else 0.2
            vol = float(a[2]) if len(a) > 2 else 1.0
            key = round(vol, 3)
            sub = self._guitar.get(key)
            if sub is None:
                from .guitar import with_guitar
                sub = self._guitar[key] = Evaluator(with_guitar(self.c, vol), self.sr)
            if m == "thd_gv":
                return sub.thd_pct(f, vin)
            return 20 * math.log10(max(sub.vout_peak(f, vin), 1e-12) / vin)
        if m == "region":
            if not a:
                raise NetlistError(0, "region() needs a transistor, e.g. region(Q1)")
            dev = self.op.devices.get(a[0].upper())
            if dev is None:
                raise NetlistError(0, f"region({a[0]}): no transistor named {a[0]}")
            return str(dev.get("region", "unknown")).lower()
        raise NetlistError(0, f"metric {m} not implemented")

    def evaluate(self, s: Spec) -> SpecResult:
        unit = s.unit
        target = _target_text(s)
        try:
            val = self.measure(s)
        except NetlistError as e:
            return SpecResult(s.text, s.metric, None, target, False, None, unit, e.message, error=e.message)
        except Exception as e:  # simulator failure: report, never crash the tool
            msg = f"couldn't measure: {e}"
            return SpecResult(s.text, s.metric, None, target, False, None, unit, msg, error=msg)
        if s.metric == "region":
            ok = val == s.lo
            # a JFET or MOSFET amplifies in "saturation" (SPICE naming); accept "active" for it,
            # and "linear" for a MOSFET's "triode"
            fet = s.args[0].upper().rsplit(".", 1)[-1][:1] if s.args else ""
            if not ok and fet in ("J", "M") and {val, s.lo} <= {"active", "saturation"}:
                ok = True
            if not ok and fet == "M" and {val, s.lo} <= {"linear", "triode"}:
                ok = True
            return SpecResult(s.text, s.metric, val, target, ok, None, unit,
                              f"{s.args[0].upper()} sits in the {val} region at idle"
                              + ("" if ok else f", not {s.lo}"))
        if val is None or (isinstance(val, float) and math.isnan(val)):
            what = METRICS[s.metric][2]
            return SpecResult(s.text, s.metric, None, target, False, None, unit,
                              f"there is no {what} in 1 Hz–200 kHz (the response never drops 3 dB there)")
        if isinstance(val, float) and not math.isfinite(val):
            return SpecResult(s.text, s.metric, None, target, False, None, unit,
                              "the measurement came out infinite, so this spec can't be judged")
        ok, margin = _compare(val, s)
        expl = _explain(s, val, ok, margin)
        return SpecResult(s.text, s.metric, float(val), target, ok, margin, unit, expl)


def _xcross(f1, f2, d1, d2, thr):
    if d1 == d2:
        return float(f1)
    t = (thr - d1) / (d2 - d1)
    return float(10 ** (math.log10(f1) + t * (math.log10(f2) - math.log10(f1))))


def _compare(val: float, s: Spec):
    if s.op == "in":
        margin = min(val - s.lo, s.hi - val)
        return margin >= 0, margin
    if s.op in ("<", "<="):
        margin = s.lo - val
        return (margin > 0 or (s.op == "<=" and margin == 0)), margin
    if s.op in (">", ">="):
        margin = val - s.lo
        return (margin > 0 or (s.op == ">=" and margin == 0)), margin
    if s.op == "==~":
        tol = 0.05 * abs(s.lo) if s.lo else 1e-12
        margin = tol - abs(val - s.lo)
        return margin >= 0, margin
    tol = max(1e-3 * abs(s.lo), 1e-3)
    margin = tol - abs(val - s.lo)
    return margin >= 0, margin


def _target_text(s: Spec) -> str:
    u = s.unit
    if s.op == "in":
        return f"{_fmt(s.lo, u)} to {_fmt(s.hi, u)}"
    if s.metric == "region":
        return f"{s.lo}"
    sym = {"<": "below", "<=": "at most", ">": "above", ">=": "at least", "==": "exactly", "==~": "within 5% of"}[s.op]
    return f"{sym} {_fmt(s.lo, u)}"


def _explain(s: Spec, val: float, ok: bool, margin: float) -> str:
    u = s.unit
    lab = spec_label(s)
    what = lab[0].lower() + lab[1:] if lab and not lab[:2].isupper() and not lab[0].isdigit() else lab
    measured = _fmt(val, u)
    if ok:
        return f"{what} is {measured}, {_fmt(abs(margin), u)} inside the target"
    if s.op == "in":
        side = "below" if val < s.lo else "above"
        return f"{what} is {measured}, {_fmt(abs(margin), u)} {side} the {_target_text(s)} window"
    return f"{what} is {measured}; target is {_target_text(s)} (off by {_fmt(abs(margin), u)})"


def _hz(text: str) -> str:
    try:
        return format_value(parse_value(text), "Hz")
    except Exception:
        return f"{text} Hz"


def _volts(text: str) -> str:
    try:
        return format_value(parse_value(text), "V")
    except Exception:
        return f"{text} V"


def spec_label(s: Spec) -> str:
    """Plain-English name of a spec, for people rather than parsers."""
    a = s.args
    m = s.metric
    f = _hz(a[0]) if a else "1 kHz"
    if m == "gain_db":
        return f"Gain at {f}"
    if m == "phase":
        return f"Phase at {f}"
    if m == "fc":
        return "Corner frequency (−3 dB)"
    if m == "f3db_low":
        return "Low cut-off (−3 dB)"
    if m == "f3db_high":
        return "High cut-off (−3 dB)"
    if m == "peak_db":
        return "Peak gain"
    if m == "peak_f":
        return "Frequency of the peak"
    if m == "v":
        return f"Voltage at {a[0]}" + (f" relative to {a[1]}" if len(a) > 1 else "") if a else "Voltage"
    if m == "i":
        return f"Current in {a[0].upper()}" if a else "Current"
    if m == "p":
        return f"Power in {a[0].upper()}" if a else "Power"
    if m == "supply_current":
        return "Supply current"
    if m == "battery_hours":
        return f"Battery life on {a[0] if a else '500'} mAh"
    if m == "thd":
        return f"Distortion (THD), {_volts(a[1]) if len(a) > 1 else '0.2 V'} at {f}"
    if m == "vout_peak":
        return f"Output peak, {_volts(a[1]) if len(a) > 1 else '0.2 V'} at {f}"
    if m == "gain_tran":
        return f"Large-signal gain, {_volts(a[1]) if len(a) > 1 else '0.2 V'} at {f}"
    if m == "headroom":
        return f"Headroom before 1% THD at {f}"
    if m == "zin":
        return f"Input impedance at {f}"
    if m == "cleanup_vol":
        lim = a[2] if len(a) > 2 else "5"
        return f"Guitar volume where it cleans up (THD under {lim.rstrip('%')}%)"
    if m in ("thd_gv", "gain_gv"):
        vol = a[2] if len(a) > 2 else "1"
        try:
            pct = f"{float(vol) * 100:.0f}%"
        except ValueError:
            pct = vol
        what = "Distortion" if m == "thd_gv" else "Gain"
        return f"{what} with the guitar volume at {pct}"
    if m == "region":
        return f"{a[0].upper()} biased in its {s.lo} region" if a else "Bias region"
    return s.text


def evaluate(circuit: Circuit, specs=None, evaluator: Evaluator | None = None) -> list:
    """Evaluate the circuit's own specs plus any extra spec strings."""
    texts = list(circuit.specs) + [s for s in (specs or []) if s not in circuit.specs]
    ev = evaluator or Evaluator(circuit)
    out = []
    for t in texts:
        try:
            s = parse_spec(t)
        except NetlistError as e:
            out.append(SpecResult(t, "?", None, "—", False, None, "", e.message, error=e.message, label=t))
            continue
        r = ev.evaluate(s)
        r.label = spec_label(s)
        out.append(r)
    return out
