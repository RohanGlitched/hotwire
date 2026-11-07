"""Pure-SVG plots (numpy only): Bode, oscilloscope, spectrum and audio waveform.

Every plot themes like the schematics: strokes and text use ``currentColor``
with the root colour set to ``var(--hotwire-ink, #1d1d1f)``, spec marks use
``--hotwire-pass`` / ``--hotwire-fail``, the background is transparent and the
``viewBox`` lets the plot scale to any width (readable from about 480 px).

    bode_svg(f, h, specs=None, title=None, phase=True)
    scope_svg(t, {"in": x, "out": y}, title=None)
    spectrum_svg(samples, sr, ref=None, title=None)
    waveform_svg(samples, sr, width=600, height=80)
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

import numpy as np

__all__ = ["bode_svg", "scope_svg", "spectrum_svg", "waveform_svg"]

INK = "var(--hotwire-ink, #1d1d1f)"
ACCENT = "var(--hotwire-accent, #c2410c)"
PASS = "var(--hotwire-pass, #15803d)"
FAIL = "var(--hotwire-fail, #b91c1c)"
MONO = "ui-monospace, 'JetBrains Mono', monospace"
SANS = "ui-sans-serif, system-ui, sans-serif"
TRACES = (("currentColor", ""), (ACCENT, ""), ("var(--hotwire-trace-3, #2563eb)", ""),
          ("var(--hotwire-trace-4, #7c3aed)", "5 3"))
DARK = ("<style>@media (prefers-color-scheme: dark){.hw-ink{color:var(--hotwire-ink,#e8e8ed)}}"
        "</style>")


# --------------------------------------------------------------------------- helpers

def _n(v):
    if not math.isfinite(v):
        return "0"
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _fmt_f(f):
    if f >= 1e6:
        return f"{f / 1e6:g}M"
    if f >= 1e3:
        return f"{f / 1e3:g}k"
    return f"{f:g}"


def _fmt_hz(f):
    if f >= 1e3:
        return f"{f / 1e3:.3g} kHz"
    return f"{f:.3g} Hz"


def _nice_step(span, target=5):
    span = abs(span) or 1.0
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _ticks(lo, hi, target=5):
    step = _nice_step(hi - lo, target)
    start = math.ceil(lo / step - 1e-9) * step
    out = []
    v = start
    while v <= hi + 1e-9 * step:
        out.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return out, step


def _fmt_tick(v, step):
    if step >= 1:
        return f"{v:.0f}".replace("-", "−")
    dec = max(0, -int(math.floor(math.log10(step))))
    if step * 10 ** dec % 1 > 1e-6:
        dec += 1
    return f"{v:.{dec}f}".replace("-", "−")


def _open(w, h, title):
    t = escape(title or "plot")
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{t}" class="hotwire-plot" style="color: {INK}">',
            f"<title>{t}</title>", DARK,
            f'<g class="hw-ink" font-family="{MONO}" font-size="12" '
            f'style="font-variant-numeric: tabular-nums">']


def _close(parts):
    parts.append("</g></svg>")
    return "".join(parts)


def _title(parts, title, x, y=20):
    if title:
        parts.append(f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="13" font-weight="600" '
                     f'fill="currentColor">{escape(str(title))}</text>')


def _path(xs, ys):
    """Polyline path, broken at non-finite points."""
    d = []
    pen = False
    for x, y in zip(xs, ys):
        if not (math.isfinite(x) and math.isfinite(y)):
            pen = False
            continue
        d.append(("L" if pen else "M") + f"{x:.1f} {y:.1f}")
        pen = True
    return "".join(d)


def _decimate(x, y, n=900):
    if len(x) <= n:
        return x, y
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[idx], y[idx]


class _LogX:
    def __init__(self, f0, f1, x0, x1):
        self.l0, self.l1 = math.log10(f0), math.log10(f1)
        self.x0, self.x1 = x0, x1

    def __call__(self, f):
        f = np.asarray(f, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.x0 + (np.log10(f) - self.l0) / (self.l1 - self.l0) * (self.x1 - self.x0)

    def grid(self, parts, y0, y1, labels=True, ylab=None):
        d_major, d_minor = [], []
        for dec in range(math.floor(self.l0), math.ceil(self.l1) + 1):
            for m in range(1, 10):
                f = m * 10.0 ** dec
                lf = math.log10(f)
                if lf < self.l0 - 1e-9 or lf > self.l1 + 1e-9:
                    continue
                x = float(self(f))
                (d_major if m == 1 else d_minor).append(f"M{x:.1f} {y0}V{y1}")
                if m == 1 and labels:
                    parts.append(f'<text x="{x:.1f}" y="{(ylab or y1) + 15}" text-anchor="middle" '
                                 f'fill="currentColor" fill-opacity="0.7">{_fmt_f(f)}</text>')
        if d_minor:
            parts.append(f'<path d="{"".join(d_minor)}" stroke="currentColor" stroke-opacity="0.07" stroke-width="1"/>')
        if d_major:
            parts.append(f'<path d="{"".join(d_major)}" stroke="currentColor" stroke-opacity="0.18" stroke-width="1"/>')


def _ygrid(parts, ticks, step, ymap, x0, x1, unit=""):
    d = []
    for v in ticks:
        y = ymap(v)
        d.append(f"M{x0} {y:.1f}H{x1}")
        parts.append(f'<text x="{x0 - 6}" y="{y + 3.8:.1f}" text-anchor="end" fill="currentColor" '
                     f'fill-opacity="0.7">{_fmt_tick(v, step)}{unit}</text>')
    if d:
        parts.append(f'<path d="{"".join(d)}" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>')


def _frame(parts, x0, y0, x1, y1):
    parts.append(f'<path d="M{x0} {y0}V{y1}H{x1}" fill="none" stroke="currentColor" stroke-opacity="0.55" '
                 f'stroke-width="1" shape-rendering="crispEdges"/>')


# --------------------------------------------------------------------------- Bode

def bode_svg(f, h, specs=None, title=None, phase=True, width=560):
    """Magnitude (dB) on a log-frequency axis, optional phase subplot, spec marks.

    ``specs``: dicts such as ``{"kind": "gain_db", "f": 1000, "lo": 18, "hi": None, "pass": True,
    "measured": 19.2}`` or ``{"kind": "fc" | "f3db_low" | "f3db_high", "lo": 900, "hi": 1100,
    "measured": 1012, "pass": True}`` (frequencies in Hz).
    """
    f = np.asarray(f, float).ravel()
    h = np.asarray(h).ravel()
    ok = np.isfinite(f) & (f > 0)
    f, h = f[ok], h[ok]
    if len(f) < 2:
        f = np.array([10.0, 100e3])
        h = np.array([1.0, 1.0])
    order = np.argsort(f)
    f, h = f[order], h[order]
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    ph = np.degrees(np.unwrap(np.angle(h)))
    W = width
    L, R, T = 52, 18, 34 if title else 14
    HM = 210
    HP = 110 if phase else 0
    gap = 34 if phase else 0
    H = T + HM + gap + HP + 40
    x0, x1 = L, W - R
    f0 = 10 ** math.floor(math.log10(f[0]) + 1e-9)
    f1 = float(f[-1])
    X = _LogX(f0, f1, x0, x1)
    parts = _open(W, H, title or "Bode plot")
    import hashlib
    cid = "hwc" + hashlib.md5(f"{title}{len(f)}{float(np.sum(np.abs(h))):.6g}".encode()).hexdigest()[:8]
    _title(parts, title, x0)
    # magnitude range
    fin = mag[np.isfinite(mag)]
    hi_v = float(np.max(fin)) if len(fin) else 0.0
    lo_v = float(np.min(fin)) if len(fin) else -40.0
    for s in specs or []:
        for k in ("lo", "hi", "measured"):
            v = s.get(k)
            if s.get("kind") == "gain_db" and v is not None and math.isfinite(float(v)):
                hi_v, lo_v = max(hi_v, float(v)), min(lo_v, float(v))
    lo_v = max(lo_v, hi_v - 100)
    pad = max(3.0, (hi_v - lo_v) * 0.08)
    ticks, step = _ticks(lo_v - pad, hi_v + pad, 5)
    ylo, yhi = min(ticks[0], lo_v - pad), max(ticks[-1], hi_v + pad)
    ylo, yhi = ticks[0] if ticks[0] <= lo_v else ylo, ticks[-1] if ticks[-1] >= hi_v else yhi
    my0, my1 = T, T + HM

    def ym(v):
        return my1 - (v - ylo) / ((yhi - ylo) or 1) * HM

    X.grid(parts, my0, my1, labels=not phase)
    _ygrid(parts, [t for t in ticks if ylo - 1e-9 <= t <= yhi + 1e-9], step, ym, x0, x1)
    parts.append(f'<text x="{x1}" y="{my0 - 8}" text-anchor="end" fill="currentColor" fill-opacity="0.7">gain, dB</text>')
    parts.append(f'<clipPath id="{cid}m"><rect x="{x0}" y="{my0}" width="{x1 - x0}" height="{my1 - my0}"/></clipPath>')
    # spec marks
    for s in specs or []:
        _bode_spec(parts, s, X, ym, my0, my1, ylo, yhi, x0, x1)
    xs = X(f)
    ys = np.clip(np.array([ym(v) for v in mag]), my0 - 2, my1 + 2)
    xd, yd = _decimate(xs, ys)
    parts.append(f'<path d="{_path(xd, yd)}" fill="none" stroke="currentColor" stroke-width="1.8" '
                 f'stroke-linejoin="round" clip-path="url(#{cid}m)"/>')
    _frame(parts, x0, my0, x1, my1)
    if phase:
        py0 = my1 + gap
        py1 = py0 + HP
        # wrap to (-180°, 180°] and break the trace where it wraps
        phw = (ph + 180.0) % 360.0 - 180.0
        plo, phi, pstep = -180.0, 180.0, 90.0
        pt = [-180.0, -90.0, 0.0, 90.0, 180.0]

        def yp(v):
            return py1 - (v - plo) / (phi - plo) * HP

        X.grid(parts, py0, py1, labels=True)
        _ygrid(parts, pt, pstep, yp, x0, x1, "°")
        parts.append(f'<clipPath id="{cid}p"><rect x="{x0}" y="{py0}" width="{x1 - x0}" height="{py1 - py0}"/></clipPath>')
        cuts = np.where(np.abs(np.diff(phw)) > 180)[0] + 1
        for seg in np.split(np.arange(len(phw)), cuts):
            if len(seg) < 2:
                continue
            xs2, ys2 = _decimate(xs[seg], np.array([yp(v) for v in phw[seg]]))
            parts.append(f'<path d="{_path(xs2, ys2)}" fill="none" stroke="currentColor" stroke-opacity="0.75" '
                         f'stroke-width="1.5" clip-path="url(#{cid}p)"/>')
        _frame(parts, x0, py0, x1, py1)
        parts.append(f'<text x="{x1}" y="{py0 - 8}" text-anchor="end" fill="currentColor" fill-opacity="0.7">phase</text>')
        hz_y = py1 + 30
    else:
        hz_y = my1 + 30
    parts.append(f'<text x="{x0 - 14}" y="{hz_y}" text-anchor="end" fill="currentColor" fill-opacity="0.7">Hz</text>')
    return _close(parts)


def _bode_spec(parts, s, X, ym, y0, y1, ylo, yhi, x0, x1):
    col = PASS if s.get("pass", True) else FAIL
    kind = str(s.get("kind", ""))
    g = [f'<g style="color: {col}" fill="currentColor" stroke="currentColor">']

    def fx(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(v) and v > 0):
            return None
        x = float(X(v))
        return x if x0 - 1 <= x <= x1 + 1 else None

    if kind == "gain_db":
        x = fx(s.get("f"))
        if x is None:
            return
        lo, hi = s.get("lo"), s.get("hi")
        top = ym(min(float(hi), yhi)) if hi is not None else y0
        bot = ym(max(float(lo), ylo)) if lo is not None else y1
        g.append(f'<rect x="{x - 5:.1f}" y="{min(top, bot):.1f}" width="10" height="{abs(bot - top):.1f}" '
                 f'fill-opacity="0.14" stroke="none"/>')
        caps = []
        if hi is not None:
            caps.append(f"M{x - 7:.1f} {top:.1f}H{x + 7:.1f}")
        if lo is not None:
            caps.append(f"M{x - 7:.1f} {bot:.1f}H{x + 7:.1f}")
        if caps:
            g.append(f'<path d="{"".join(caps)}" stroke-width="2" fill="none"/>')
        m = s.get("measured")
        if m is not None and math.isfinite(float(m)):
            y = min(max(ym(float(m)), y0), y1)
            g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" stroke="none"/>')
            g.append(f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" stroke="none" font-size="10">'
                     f'{float(m):.1f} dB</text>')
    elif kind in ("fc", "f3db_low", "f3db_high", "f3db"):
        lo, hi = fx(s.get("lo")), fx(s.get("hi"))
        if lo is not None or hi is not None:
            a = lo if lo is not None else x0
            b = hi if hi is not None else x1
            g.append(f'<rect x="{min(a, b):.1f}" y="{y0}" width="{abs(b - a):.1f}" height="{y1 - y0}" '
                     f'fill-opacity="0.08" stroke="none"/>')
        m = fx(s.get("measured"))
        if m is not None:
            g.append(f'<path d="M{m:.1f} {y0}V{y1}" stroke-width="1.5" stroke-dasharray="4 3" fill="none"/>')
            name = {"fc": "fc", "f3db_low": "fₗ", "f3db_high": "fₕ"}.get(kind, "f")
            # low corners label to the left of their line, high corners to the right,
            # so two close corners never print on top of each other
            if kind == "f3db_low":
                g.append(f'<text x="{m - 5:.1f}" y="{y0 + 13}" text-anchor="end" stroke="none" font-size="10">'
                         f'{name} {_fmt_hz(float(s.get("measured")))}</text>')
            else:
                g.append(f'<text x="{m + 5:.1f}" y="{y0 + (27 if kind == "f3db_high" else 13)}" stroke="none" font-size="10">'
                         f'{name} {_fmt_hz(float(s.get("measured")))}</text>')
    else:
        return
    g.append("</g>")
    parts.extend(g)


# --------------------------------------------------------------------------- scope

def scope_svg(t, traces, title=None, width=560, height=290):
    """Oscilloscope-style plot: time in ms, up to four named traces with a legend."""
    t = np.asarray(t, float).ravel()
    items = []
    for name, y in list((traces or {}).items())[:4]:
        y = np.asarray(y, float).ravel()
        n = min(len(t), len(y))
        items.append((str(name), y[:n]))
    W, H = width, height
    L, R = 56, 18
    T = 40 if title else 30
    B = 40
    x0, x1, y0, y1 = L, W - R, T, H - B
    tms = t * 1e3
    if len(tms) < 2:
        tms = np.array([0.0, 1.0])
    ta, tb = float(tms[0]), float(tms[-1])
    if tb <= ta:
        tb = ta + 1.0
    allv = np.concatenate([y[np.isfinite(y)] for _, y in items]) if items else np.array([-1.0, 1.0])
    if allv.size == 0:
        allv = np.array([-1.0, 1.0])
    vlo, vhi = float(np.min(allv)), float(np.max(allv))
    if vhi - vlo < 1e-9:
        vlo, vhi = vlo - 1, vhi + 1
    pad = (vhi - vlo) * 0.08
    yt, ystep = _ticks(vlo - pad, vhi + pad, 5)
    ylo, yhi = min(yt[0], vlo), max(yt[-1], vhi)
    xt, xstep = _ticks(ta, tb, 6)

    def X(v):
        return x0 + (v - ta) / (tb - ta) * (x1 - x0)

    def Y(v):
        return y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)

    parts = _open(W, H, title or "Oscilloscope")
    _title(parts, title, x0)
    d = []
    for v in xt:
        if ta - 1e-9 <= v <= tb + 1e-9:
            x = X(v)
            d.append(f"M{x:.1f} {y0}V{y1}")
            parts.append(f'<text x="{x:.1f}" y="{y1 + 15}" text-anchor="middle" fill="currentColor" '
                         f'fill-opacity="0.7">{_fmt_tick(v, xstep)}</text>')
    if d:
        parts.append(f'<path d="{"".join(d)}" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>')
    _ygrid(parts, [v for v in yt if ylo - 1e-9 <= v <= yhi + 1e-9], ystep, Y, x0, x1)
    if ylo < 0 < yhi:
        parts.append(f'<path d="M{x0} {Y(0):.1f}H{x1}" stroke="currentColor" stroke-opacity="0.35" stroke-width="1"/>')
    parts.append(f'<text x="{x0 - 14}" y="{y1 + 30}" text-anchor="end" fill="currentColor" fill-opacity="0.7">ms</text>')
    parts.append(f'<text x="{x0 - 50}" y="{y0 - 10}" fill="currentColor" fill-opacity="0.7">V</text>')
    lx = x0 + (180 if title else 6)
    ly = (T - 22) if title else 16
    if title:
        lx = x1 - sum(_text_len(nm) * 7.2 + 34 for nm, _ in items)
    for i, (name, y) in enumerate(items):
        col, dash = TRACES[i]
        n = len(y)
        xs = np.array([X(v) for v in tms[:n]]) if n <= 2000 else X(tms[:n])
        ys = Y(y)
        xs, ys = _envelope_or_line(np.asarray(xs, float), np.asarray(ys, float), x1 - x0)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<g style="color: {col}"><path d="{_path(xs, ys)}" fill="none" stroke="currentColor" '
                     f'stroke-width="1.6" stroke-linejoin="round"{da}/></g>')
        parts.append(f'<g style="color: {col}"><path d="M{lx:.1f} {ly - 4}h18" stroke="currentColor" '
                     f'stroke-width="2.4"{da}/><text x="{lx + 24:.1f}" y="{ly}" fill="currentColor">'
                     f'{escape(name)}</text></g>')
        lx += _text_len(name) * 7.2 + 34
    _frame(parts, x0, y0, x1, y1)
    return _close(parts)


def _text_len(s):
    return len(str(s))


def _envelope_or_line(xs, ys, width_px):
    """Keep at most ~2 points per pixel column (min/max) so long traces stay light."""
    n = len(xs)
    if n <= 2 * width_px + 2:
        return xs, ys
    cols = int(width_px)
    idx = np.minimum(((xs - xs[0]) / max(xs[-1] - xs[0], 1e-9) * (cols - 1)).astype(int), cols - 1)
    ox, oy = [], []
    order = np.argsort(idx, kind="stable")
    idx_s, ys_s, xs_s = idx[order], ys[order], xs[order]
    bounds = np.searchsorted(idx_s, np.arange(cols + 1))
    for c in range(cols):
        a, b = bounds[c], bounds[c + 1]
        if a == b:
            continue
        seg = ys_s[a:b]
        i_min, i_max = int(np.argmin(seg)), int(np.argmax(seg))
        for i in sorted((i_min, i_max)):
            ox.append(xs_s[a + i])
            oy.append(seg[i])
    return np.array(ox), np.array(oy)


# --------------------------------------------------------------------------- spectrum

def _spectrum_db(x, sr, nfft=None):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 64:
        x = np.concatenate([x, np.zeros(64 - len(x))])
    x = x - np.mean(x)
    n = min(len(x), 1 << 16)
    # average Hann-windowed frames (Welch) for a steady estimate
    seg = min(n, 8192)
    hop = seg // 2
    win = np.hanning(seg)
    acc = np.zeros(seg // 2 + 1)
    count = 0
    for start in range(0, max(1, len(x) - seg + 1), hop):
        fr = x[start:start + seg]
        if len(fr) < seg:
            fr = np.concatenate([fr, np.zeros(seg - len(fr))])
        acc += np.abs(np.fft.rfft(fr * win)) ** 2
        count += 1
        if count >= 64:
            break
    p = acc / max(count, 1)
    freqs = np.fft.rfftfreq(seg, 1.0 / sr)
    return freqs, p


def _smooth_log(freqs, p, f_lo, f_hi, per_oct=12, points=240):
    centers = np.geomspace(f_lo, f_hi, points)
    half = 2 ** (1 / (2 * per_oct))
    out = np.empty(points)
    cum = np.concatenate([[0.0], np.cumsum(p)])
    for i, fc in enumerate(centers):
        a = np.searchsorted(freqs, fc / half)
        b = np.searchsorted(freqs, fc * half, side="right")
        if b <= a:
            j = min(max(np.searchsorted(freqs, fc), 1), len(freqs) - 1)
            out[i] = p[j]
        else:
            out[i] = (cum[b] - cum[a]) / (b - a)
    return centers, out


def spectrum_svg(samples, sr, ref=None, title=None, width=560, height=270):
    """Log-frequency magnitude spectrum (Hann/Welch, 1/12-octave smoothing); ``ref`` = dry trace for A/B."""
    sr = float(sr)
    f_lo, f_hi = 20.0, min(20000.0, sr / 2 * 0.98)
    fr, p = _spectrum_db(samples, sr)
    c, s = _smooth_log(fr, p, f_lo, f_hi)
    curves = [("circuit", c, s)]
    if ref is not None:
        fr2, p2 = _spectrum_db(ref, sr)
        c2, s2 = _smooth_log(fr2, p2, f_lo, f_hi)
        curves.insert(0, ("dry", c2, s2))
    peak = max(float(np.max(cv[2])) for cv in curves) or 1e-30
    dbs = [(nm, cf, 10 * np.log10(np.maximum(sv / peak, 1e-12))) for nm, cf, sv in curves]
    W, H = width, height
    L, R = 52, 18
    T = 40 if title else 30
    B = 40
    x0, x1, y0, y1 = L, W - R, T, H - B
    X = _LogX(f_lo, 20000 if f_hi >= 20000 * 0.97 else f_hi, x0, x1)
    ylo, yhi = -80.0, 0.0
    yt, ystep = _ticks(ylo, yhi, 4)

    def Y(v):
        return y1 - (np.clip(v, ylo, yhi) - ylo) / (yhi - ylo) * (y1 - y0)

    parts = _open(W, H, title or "Spectrum")
    _title(parts, title, x0)
    d = []
    for f in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
        if f > f_hi * 1.001:
            continue
        x = float(X(f))
        d.append(f"M{x:.1f} {y0}V{y1}")
        parts.append(f'<text x="{x:.1f}" y="{y1 + 15}" text-anchor="middle" fill="currentColor" '
                     f'fill-opacity="0.7">{_fmt_f(f)}</text>')
    parts.append(f'<path d="{"".join(d)}" stroke="currentColor" stroke-opacity="0.12" stroke-width="1"/>')
    _ygrid(parts, yt, ystep, Y, x0, x1)
    parts.append(f'<text x="{x0 - 46}" y="{y0 - 10}" fill="currentColor" fill-opacity="0.7">dB</text>')
    parts.append(f'<text x="{x0 - 14}" y="{y1 + 30}" text-anchor="end" fill="currentColor" fill-opacity="0.7">Hz</text>')
    lx = x1 - 6
    legend = []
    for nm, cf, db in dbs:
        xs = X(cf)
        ys = Y(db)
        if nm == "dry":
            parts.append(f'<path d="{_path(xs, ys)}" fill="none" stroke="currentColor" stroke-opacity="0.45" '
                         f'stroke-width="1.4" stroke-dasharray="4 3"/>')
        else:
            area = _path(xs, ys) + f"L{xs[-1]:.1f} {y1}L{xs[0]:.1f} {y1}Z"
            parts.append(f'<g style="color: {ACCENT}"><path d="{area}" fill="currentColor" fill-opacity="0.10" '
                         f'stroke="none"/><path d="{_path(xs, ys)}" fill="none" stroke="currentColor" '
                         f'stroke-width="1.8" stroke-linejoin="round"/></g>')
        legend.append(nm)
    for nm in reversed(legend):
        w = len(nm) * 7.2
        if nm == "dry":
            parts.append(f'<path d="M{lx - w - 26:.1f} {T - 14}h18" stroke="currentColor" stroke-opacity="0.45" '
                         f'stroke-width="2" stroke-dasharray="4 3"/>')
            parts.append(f'<text x="{lx - w:.1f}" y="{T - 10}" fill="currentColor" fill-opacity="0.7">{nm}</text>')
        else:
            parts.append(f'<g style="color: {ACCENT}"><path d="M{lx - w - 26:.1f} {T - 14}h18" stroke="currentColor" '
                         f'stroke-width="2.4"/><text x="{lx - w:.1f}" y="{T - 10}" fill="currentColor">{nm}</text></g>')
        lx -= w + 40
    _frame(parts, x0, y0, x1, y1)
    return _close(parts)


# --------------------------------------------------------------------------- waveform

def waveform_svg(samples, sr, width=600, height=80):
    """Compact min/max envelope of an audio clip (for audio cards)."""
    x = np.asarray(samples, float).ravel()
    x = np.where(np.isfinite(x), x, 0.0)
    W, H = int(width), int(height)
    cols = max(W // 2, 1)
    mid = H / 2
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    scale = (H / 2 - 3) / pk if pk > 0 else 0.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'preserveAspectRatio="none" role="img" aria-label="waveform, {len(x) / max(float(sr), 1):.2f} s" '
             f'class="hotwire-wave" style="color: {INK}">',
             "<title>waveform</title>", DARK, '<g class="hw-ink">']
    if x.size:
        edges = np.linspace(0, x.size, cols + 1).astype(int)
        tops, bots = [], []
        for i in range(cols):
            a, b = edges[i], max(edges[i + 1], edges[i] + 1)
            seg = x[a:min(b, x.size)]
            if seg.size == 0:
                seg = np.zeros(1)
            tops.append(mid - float(np.max(seg)) * scale)
            bots.append(mid - float(np.min(seg)) * scale)
        xs = (np.arange(cols) + 0.5) * (W / cols)
        top = [min(t, mid - 0.5) for t in tops]
        bot = [max(b, mid + 0.5) for b in bots]
        d = "M" + "L".join(f"{xv:.1f} {yv:.1f}" for xv, yv in zip(xs, top))
        d += "L" + "L".join(f"{xv:.1f} {yv:.1f}" for xv, yv in zip(xs[::-1], bot[::-1])) + "Z"
        parts.append(f'<path d="{d}" fill="currentColor" fill-opacity="0.85"/>')
    parts.append(f'<path d="M0 {mid}H{W}" stroke="currentColor" stroke-opacity="0.25" stroke-width="1"/>')
    parts.append("</g></svg>")
    return "".join(parts)
