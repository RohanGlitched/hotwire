"""Match a recording: find knob settings (or the library circuit) that sound closest.

Give it a dry take (a direct-injected guitar) and the same take through a real
pedal or amp (re-amped), and it searches pot positions so the simulated circuit
reproduces the recorded tone, or ranks every audio circuit in the library by how
close it gets. The distance combines two things the ear notices first:

* the long-term spectrum in third-octave bands (tone), level-normalised, and
* the harmonic richness / clipping (how hard it distorts), from the spectral
  flatness of the residual above the input's own spectrum and the crest factor.

It's a tone-matching aid, not a circuit identifier: a low distance means "these
settings get you close", and the result says how close.
"""
from __future__ import annotations

import math

import numpy as np

from .netlist import Circuit

__all__ = ["tone_features", "distance", "fit_pots", "load_pair"]

_BANDS = np.array([31.5 * 2 ** (k / 3) for k in range(0, 28)])  # 31.5 Hz .. ~16 kHz


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Band-limited resampling via the FFT (no aliasing, unlike linear interpolation)."""
    if sr == target:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    n_new = int(round(len(x) * target / sr))
    X = np.fft.rfft(x)
    keep = n_new // 2 + 1
    Y = np.zeros(keep, dtype=complex)
    m = min(keep, len(X))
    Y[:m] = X[:m]
    return np.fft.irfft(Y, n_new) * (n_new / len(x))


def tone_features(y: np.ndarray, sr: int) -> dict:
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    rms = float(np.sqrt(np.mean(y * y))) or 1e-12
    y = y / rms
    n = 4096
    hop = n // 2
    win = np.hanning(n)
    frames = [y[i:i + n] * win for i in range(0, max(1, len(y) - n), hop)]
    if not frames:
        frames = [np.pad(y, (0, n - len(y))) * win]
    spec = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    bands = []
    for fc in _BANDS[_BANDS < sr / 3]:  # stay well below Nyquist, where simulation aliasing lives
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        m = (freqs >= lo) & (freqs < hi)
        bands.append(10 * math.log10(float(np.sum(spec[m])) + 1e-12) if m.any() else -120.0)
    bands = np.array(bands)
    bands -= np.max(bands)
    crest = float(np.max(np.abs(y)))           # y is RMS-normalised: peak/RMS
    hf = spec[(freqs > 2000) & (freqs < 8000)]
    flat = float(np.exp(np.mean(np.log(hf + 1e-20))) / (np.mean(hf) + 1e-20)) if len(hf) else 0.0
    return {"bands": bands, "crest": crest, "flatness": flat}


def distance(a: dict, b: dict) -> float:
    """dB-ish distance: band RMS difference plus distortion-character terms."""
    keep = (a["bands"] > -60) | (b["bands"] > -60)
    tone = float(np.sqrt(np.mean((a["bands"][keep] - b["bands"][keep]) ** 2))) if keep.any() else 0.0
    crest = 6.0 * abs(math.log(max(a["crest"], 1e-6) / max(b["crest"], 1e-6)))
    flat = 10.0 * abs(a["flatness"] - b["flatness"])
    return tone + crest + flat


def load_pair(dry_path: str, wet_path: str, sr: int = 24000, seconds: float = 1.0):
    from .audio import read_wav, user_wav
    d, sr_d = read_wav(user_wav(dry_path))
    w, sr_w = read_wav(user_wav(wet_path))
    d = _resample(np.asarray(d, float), sr_d, sr)
    w = _resample(np.asarray(w, float), sr_w, sr)
    n = min(len(d), len(w))
    if n < sr * 0.3:
        raise ValueError("recordings must be at least 0.3 s long")
    # take the loudest window of the dry take (where the playing is)
    seg = int(min(seconds, n / sr) * sr)
    if n > seg:
        energy = np.convolve(d[:n] ** 2, np.ones(sr // 10), mode="valid")
        start = int(np.argmax(energy[: n - seg])) if n - seg > 0 else 0
        start = max(0, min(start, n - seg))
    else:
        start = 0
    return d[start:start + seg], w[start:start + seg], sr


def _render(c: Circuit, x: np.ndarray, sr: int, level: float) -> np.ndarray:
    from .analyses import process_audio
    pk = float(np.max(np.abs(x))) or 1.0
    return np.asarray(process_audio(c, x / pk * level, sr), dtype=float)


def fit_pots(c: Circuit, dry: np.ndarray, wet: np.ndarray, sr: int, level: float = 0.2,
             max_evals: int = 36, time_budget: float = 60.0) -> dict:
    """Coordinate search over pot positions (coarse grid, then refine) to match ``wet``."""
    import time
    target = tone_features(wet, sr)
    # level/volume knobs only change loudness, which the comparison normalises away
    skip = ("level", "volume", "output", "master", "gain out")
    pots = [p for p in c.pots if not any(w in (c.pots[p].label or "").lower() for w in skip)]
    t0 = time.time()
    cache = {}

    def score(pos: dict) -> float:
        key = tuple(round(pos[p], 3) for p in pots)
        if key in cache:
            return cache[key]
        y = _render(c.with_pots(pos) if pos else c, dry, sr, level)
        s = distance(tone_features(y, sr), target)
        cache[key] = s
        return s

    best = {p: c.pots[p].pos for p in pots}
    best_s = score(best)
    if not pots:
        return {"positions": {}, "distance": round(best_s, 2), "evaluations": 1}
    for step in (0.25, 0.1, 0.04):
        improved = True
        while improved and len(cache) < max_evals and time.time() - t0 < time_budget:
            improved = False
            for p in pots:
                for d in (-step, step):
                    trial = dict(best)
                    trial[p] = min(1.0, max(0.0, best[p] + d))
                    if trial[p] == best[p]:
                        continue
                    s = score(trial)
                    if s < best_s - 1e-6:
                        best, best_s, improved = trial, s, True
                if len(cache) >= max_evals or time.time() - t0 > time_budget:
                    break
    return {"positions": {p: round(v, 2) for p, v in best.items()},
            "labels": {p: c.pots[p].label for p in pots},
            "ignored": [p for p in c.pots if p not in pots],
            "distance": round(best_s, 2), "evaluations": len(cache),
            "seconds": round(time.time() - t0, 1)}
