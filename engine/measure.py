"""Measurements on analysis results (pure numpy).

AC helpers accept an :class:`~engine.analyses.AcResult` or an ``(f, h)``
tuple.  Frequencies are interpolated on a log axis, gains in dB.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = ["db", "gain_db", "phase_deg", "corners", "f3db_low", "f3db_high", "fc", "peak_db",
           "q_factor", "rms", "peak", "peak_to_peak", "thd", "harmonics", "supply_current",
           "headroom", "rms_error"]


def _fh(ac) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(ac, tuple):
        f, h = ac
    else:
        f, h = ac.f, ac.h
    return np.asarray(f, dtype=float), np.asarray(h, dtype=complex)


def db(x) -> np.ndarray:
    """20 log10 |x| with a -600 dB floor."""
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def gain_db(ac, f: float) -> float:
    """Gain in dB at frequency ``f`` (log-frequency interpolation)."""
    fr, h = _fh(ac)
    if not fr[0] <= f <= fr[-1]:
        raise ValueError(f"{f:g} Hz is outside the sweep ({fr[0]:g}..{fr[-1]:g} Hz)")
    return float(np.interp(math.log10(f), np.log10(fr), db(h)))


def phase_deg(ac, f: float) -> float:
    """Phase in degrees at ``f`` (unwrapped along the sweep)."""
    fr, h = _fh(ac)
    ph = np.degrees(np.unwrap(np.angle(h)))
    return float(np.interp(math.log10(f), np.log10(fr), ph))


def _cross(f1: float, f2: float, d1: float, d2: float, thr: float) -> float:
    """Log-frequency interpolation of the threshold crossing between two points."""
    if d2 == d1:
        return math.sqrt(f1 * f2)
    t = (thr - d1) / (d2 - d1)
    return 10 ** (math.log10(f1) + t * (math.log10(f2) - math.log10(f1)))


HALF_POWER_DB = 10.0 * math.log10(2.0)      # 3.0103 dB: the textbook corner 1/(2 pi R C)


def corners(ac, drop_db: float = HALF_POWER_DB) -> tuple[float | None, float | None, float, float]:
    """(f_low, f_high, peak_db, peak_f): the half-power (-3.01 dB) points around the gain peak.

    ``None`` where the response never falls ``drop_db`` below the peak on that side.
    """
    fr, h = _fh(ac)
    g = db(h)
    k = int(np.argmax(g))
    thr = g[k] - drop_db
    lo = hi = None
    for j in range(k, 0, -1):
        if g[j - 1] < thr <= g[j]:
            lo = _cross(fr[j - 1], fr[j], g[j - 1], g[j], thr)
            break
    for j in range(k, len(fr) - 1):
        if g[j + 1] < thr <= g[j]:
            hi = _cross(fr[j], fr[j + 1], g[j], g[j + 1], thr)
            break
    return lo, hi, float(g[k]), float(fr[k])


def f3db_low(ac) -> float | None:
    return corners(ac)[0]


def f3db_high(ac) -> float | None:
    return corners(ac)[1]


def fc(ac) -> float | None:
    """Corner of a low- or high-pass: the -3 dB point on the falling side."""
    fr, h = _fh(ac)
    lo, hi, _, _ = corners(ac)
    g = db(h)
    return hi if g[0] >= g[-1] else lo


def peak_db(ac) -> float:
    return corners(ac)[2]


def q_factor(ac) -> float | None:
    """Centre frequency over -3 dB bandwidth for a resonant peak."""
    lo, hi, _, pf = corners(ac)
    if lo is None or hi is None or hi <= lo:
        return None
    return math.sqrt(lo * hi) / (hi - lo)


# --------------------------------------------------------------------------- waveforms

def rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def peak(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x))) if x.size else 0.0


def peak_to_peak(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.ptp(x)) if x.size else 0.0


def rms_error(x, ref) -> float:
    """RMS of (x - ref) relative to the AC RMS of ``ref`` (fraction)."""
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    sig = rms(ref - ref.mean())
    return rms(x - ref) / sig if sig > 0 else math.inf


def harmonics(x, cycles: int, n: int = 10) -> np.ndarray:
    """Amplitudes of harmonics 1..n of a record holding exactly ``cycles`` periods."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) * 2.0 / len(x)
    out = np.zeros(n)
    for k in range(1, n + 1):
        b = k * cycles
        if b < len(spec):
            out[k - 1] = spec[b]
    return out


def thd(x, cycles: int | None = None, fs: float | None = None, f0: float | None = None,
        n_harmonics: int = 10) -> float:
    """Total harmonic distortion in percent.

    Give either ``cycles`` (the record holds exactly that many periods) or
    ``fs`` and ``f0``: then the last whole number of periods is taken and
    resampled to an integer number of points per period, so FFT bins land on
    the harmonics without leakage.
    """
    x = np.asarray(x, dtype=float)
    if cycles is None:
        if not fs or not f0:
            raise ValueError("thd needs cycles, or fs and f0")
        per = fs / f0
        cycles = int(len(x) // per)
        if cycles < 1:
            raise ValueError("the record is shorter than one period")
        n_keep = cycles * per
        t_old = np.arange(len(x)) / fs
        t0 = t_old[-1] - n_keep / fs + 1.0 / fs
        pts = 256
        t_new = t0 + np.arange(cycles * pts) / (pts * f0)
        x = np.interp(t_new, t_old, x)
    h = harmonics(x, cycles, n_harmonics)
    if h[0] <= 0:
        return math.nan
    return float(100.0 * math.sqrt(float(np.sum(h[1:] ** 2))) / h[0])


# --------------------------------------------------------------------------- DC

def supply_current(op, c=None, supplies: list | None = None) -> float:
    """Total current (A) the supply sources deliver at the operating point.

    Uses ``supplies`` or the circuit's ``*@ supply`` names; each source's
    current magnitude is summed (a +9 V and a -9 V rail both count).
    """
    names = supplies
    if names is None and c is not None:
        names = list(c.supplies) or [e.name for e in c.elements
                                     if e.kind == "V" and e.name != c.input and (e.value or 0) != 0]
    if not names:
        return 0.0
    return float(sum(abs(op.i.get(n.upper(), 0.0)) for n in names))


# --------------------------------------------------------------------------- headroom

def headroom(thd_at: Callable[[float], float], lo: float = 1e-3, hi: float = 10.0,
             limit_pct: float = 1.0, iters: int = 14) -> float:
    """Largest input level whose THD stays at or under ``limit_pct`` (log bisection).

    ``thd_at(level)`` returns THD in percent.  Returns ``lo`` when even the
    smallest level distorts and ``hi`` when the largest stays clean.
    """
    if thd_at(lo) > limit_pct:
        return lo
    if thd_at(hi) <= limit_pct:
        return hi
    a, b = lo, hi
    for _ in range(iters):
        mid = math.sqrt(a * b)
        if thd_at(mid) > limit_pct:
            b = mid
        else:
            a = mid
        if b / a < 1.005:
            break
    return a
