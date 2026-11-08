"""Audio I/O and the generated demo sources.

* WAV read/write with the standard-library ``wave`` module (PCM; mono out,
  stereo is mixed down on read) and RMS loudness matching.
* Deterministic demo sources: ``source(name, sr, seconds)`` returns samples
  peak-normalised to 1.0 (the caller scales to volts); names are
  ``guitar_riff`` (Karplus-Strong strings with pick clicks playing a short
  original blues riff with power chords and palm mutes), ``guitar_chords``
  (strummed open chords), ``bass_line``, ``drum_loop`` and ``sine_sweep``, or
  a path to a .wav file.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

__all__ = ["source", "read_wav", "write_wav", "resample", "rms", "match_loudness", "normalize_peak",
           "fade", "guitar_riff", "guitar_chords", "bass_line", "sine_sweep", "drum_loop", "generate",
           "SOURCES"]

SR = 48000


# --------------------------------------------------------------------------- wav

AUDIO_MAX = 12_000_000  # bytes; same cap as the dashboard's upload limit


def user_wav(path: str | Path) -> Path:
    """A user-supplied recording: must be a regular ``.wav`` file of at most 12 MB."""
    p = Path(str(path)).expanduser()
    if p.suffix.lower() != ".wav":
        raise ValueError(f"{p.name} is not a .wav file")
    if not p.is_file():
        raise ValueError(f"can't find the WAV file {path}")
    size = p.stat().st_size
    if size > AUDIO_MAX:
        raise ValueError(f"{p.name} is {size / 1e6:.1f} MB; recordings must be at most {AUDIO_MAX / 1e6:g} MB")
    return p


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float samples in [-1, 1] and its sample rate.

    Handles 8/16/24/32-bit integer PCM and 32/64-bit IEEE float (including
    WAVE_FORMAT_EXTENSIBLE).  The stdlib ``wave`` module rejects float files,
    so the RIFF chunks are walked directly.  Multi-channel audio is averaged.
    """
    p = Path(path)
    if not p.is_file() or p.stat().st_size > AUDIO_MAX:
        raise ValueError(f"{p.name} is not a WAV file of at most {AUDIO_MAX / 1e6:g} MB")
    data = p.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    pos = 12
    fmt = None
    raw = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag = int.from_bytes(body[0:2], "little")
            nch = int.from_bytes(body[2:4], "little")
            sr = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            if tag == 0xFFFE and len(body) >= 26:          # extensible: real tag in the sub-format GUID
                tag = int.from_bytes(body[24:26], "little")
            fmt = (tag, nch, sr, bits)
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, nch, sr, bits = fmt
    width = bits // 8
    raw = raw[: len(raw) - len(raw) % max(1, width * nch)]
    if tag == 3 and width == 4:
        x = np.frombuffer(raw, dtype="<f4").astype(float)
    elif tag == 3 and width == 8:
        x = np.frombuffer(raw, dtype="<f8").astype(float)
    elif tag == 1 and width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(float) - 128.0) / 128.0
    elif tag == 1 and width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    elif tag == 1 and width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(float) / float(1 << 23)
    elif tag == 1 and width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(float) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported WAV encoding (format {tag}, {bits}-bit)")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return np.nan_to_num(x), sr


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling: FFT-domain low-pass at the lower Nyquist, then interpolation."""
    x = np.asarray(x, dtype=float)
    if sr_in == sr_out or len(x) == 0:
        return x.copy()
    if sr_out < sr_in:          # anti-alias before decimating
        x = _shape(x, sr_in, lambda f: 1.0 / np.sqrt(1.0 + (f / (0.45 * sr_out)) ** 16))
    n_out = int(round(len(x) * sr_out / sr_in))
    t_out = np.arange(n_out) * (sr_in / sr_out)
    return np.interp(t_out, np.arange(len(x)), x)


def write_wav(path: str | Path, samples: np.ndarray, sr: int = SR, peak: float | None = None) -> Path:
    """Write mono 16-bit PCM.  ``peak`` rescales so max |x| equals it; otherwise clip to [-1, 1]."""
    x = np.asarray(samples, dtype=float).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if peak is not None:
        x = normalize_peak(x, peak)
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def normalize_peak(x: np.ndarray, peak: float = 0.89) -> np.ndarray:
    """Scale so the largest |sample| equals ``peak`` (silence stays silent)."""
    x = np.asarray(x, dtype=float)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > 0 else x.copy()


def match_loudness(x: np.ndarray, ref: np.ndarray, max_peak: float = 0.98) -> np.ndarray:
    """Scale ``x`` to the RMS of ``ref`` (A/B at equal loudness), backing off to keep peaks under ``max_peak``."""
    x = np.asarray(x, dtype=float)
    rx, rr = rms(x), rms(ref)
    if rx <= 0 or rr <= 0:
        return x.copy()
    y = x * (rr / rx)
    m = float(np.max(np.abs(y)))
    return y * (max_peak / m) if m > max_peak else y


def fade(x: np.ndarray, sr: int = SR, fade_in: float = 0.005, fade_out: float = 0.05) -> np.ndarray:
    """Raised-cosine fades so a clip starts and ends without clicks."""
    y = np.array(x, dtype=float)
    n_in = min(len(y), int(fade_in * sr))
    n_out = min(len(y), int(fade_out * sr))
    if n_in > 0:
        y[:n_in] *= 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n_in))
    if n_out > 0:
        y[-n_out:] *= 0.5 + 0.5 * np.cos(np.linspace(0, math.pi, n_out))
    return y


# --------------------------------------------------------------------------- filters (FFT domain, zero phase)

def _shape(x: np.ndarray, sr: int, gain_fn) -> np.ndarray:
    """Apply a magnitude response gain_fn(f) with one FFT (zero-phase, deterministic)."""
    n = len(x)
    nfft = 1 << max(1, (n - 1).bit_length())
    X = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    return np.fft.irfft(X * gain_fn(f), nfft)[:n]


def _resonant_lp(f: np.ndarray, fc: float, q: float) -> np.ndarray:
    """|H| of a 2nd-order low-pass (pickup inductance with cable capacitance)."""
    r = f / fc
    return 1.0 / np.sqrt((1 - r * r) ** 2 + (r / q) ** 2)


def _highpass(f: np.ndarray, fc: float, order: int = 1) -> np.ndarray:
    r = f / fc
    return (r / np.sqrt(1 + r * r)) ** order


# --------------------------------------------------------------------------- Karplus-Strong

def _pluck(freq: float, dur: float, sr: int, rng: np.random.Generator, *, t60: float = 3.0,
           bright: float = 0.6, pick: float = 0.18, vel: float = 1.0) -> np.ndarray:
    """One plucked string: Karplus-Strong with fractional-delay tuning, vectorised per period.

    The loop filter is a 3-tap FIR (two-point average for string damping,
    folded together with linear interpolation for exact tuning), so each
    period depends only on earlier periods and is one numpy expression.
    The excitation mixes the string's displaced shape at the pick point with
    filtered noise; ``bright`` sets the pick hardness, ``pick`` the position
    (fraction of the string length from the bridge).
    """
    n = max(1, int(dur * sr))
    period = sr / freq
    big_n = max(2, int(math.floor(period - 0.5)))
    d = period - 0.5 - big_n
    w0, w1, w2 = 0.5 * (1 - d), 0.5, 0.5 * d
    g = 10 ** (-3.0 / (t60 * freq))               # loop gain per period for the requested T60
    # excitation: triangle (string pulled at the pick point) plus noise for the pick scrape
    pos = max(1, int(round(pick * big_n)))
    tri = np.concatenate((np.linspace(0, 1, pos, endpoint=False), np.linspace(1, 0, big_n - pos)))
    noise = rng.uniform(-1, 1, big_n)
    k = max(1, int(round((1.0 - bright) * 10)) + 1)
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    exc = 0.55 * (tri - tri.mean()) + 0.45 * bright * noise
    exc = exc - np.concatenate((np.zeros(pos), exc[:-pos])) * 0.5    # pick-position comb
    exc -= exc.mean()
    exc *= vel / (np.max(np.abs(exc)) + 1e-12)
    y = np.zeros(n + big_n + 2)
    y[:big_n] = exc[: min(big_n, len(y))]
    for start in range(big_n, n, big_n):
        stop = min(start + big_n, n)
        idx = np.arange(start, stop)
        y[idx] = g * (w0 * y[idx - big_n] + w1 * y[idx - big_n - 1]
                      + w2 * np.where(idx - big_n - 2 >= 0, y[idx - big_n - 2], 0.0))
    return y[:n]


def _pick_click(sr: int, rng: np.random.Generator, vel: float) -> np.ndarray:
    """The few-millisecond scrape of a plectrum leaving the string (bright, decaying)."""
    m = int(0.004 * sr)
    tt = np.arange(m) / sr
    click = np.diff(rng.uniform(-1, 1, m + 1)) * np.exp(-tt / 0.0008)
    return 0.12 * vel * click


def _note(freq: float, dur: float, sr: int, rng: np.random.Generator, *, mute: bool = False,
          vel: float = 1.0, ring: float = 0.0, detune_cents: float = 2.0) -> np.ndarray:
    """A guitar note: open ring or palm mute (short, dark, thumpy), pick click and release."""
    f = freq * 2 ** (rng.normal(0.0, detune_cents) / 1200.0)
    if mute:
        body = _pluck(f, dur + 0.03, sr, rng, t60=0.22, bright=0.2, pick=0.3, vel=vel)
        body *= np.exp(-np.arange(len(body)) / (0.08 * sr))
    else:
        # lower strings sustain longer; the bridge-side damping shortens the top
        t60 = 4.5 if f < 130 else (3.4 if f < 260 else 2.4)
        body = _pluck(f, dur + ring, sr, rng, t60=t60, bright=0.75, pick=0.14, vel=vel)
    click = _pick_click(sr, rng, vel * (0.5 if mute else 1.0))
    body[: len(click)] += click[: len(body)]
    rel = min(len(body), int(0.015 * sr))
    if rel:
        body[-rel:] *= np.linspace(1.0, 0.0, rel)
    return body


_RAW = False  # True while rendering the bare string voltage for engine.guitar's pickup model


def _di(y: np.ndarray, sr: int) -> np.ndarray:
    """Passive pickup into a cable: resonant peak near 3.8 kHz, gentle low cut.
    Skipped (only the low cut kept) when the caller models the pickup itself."""
    if _RAW:
        return _shape(y, sr, lambda f: _highpass(f, 60.0, 1))
    return _shape(y, sr, lambda f: _resonant_lp(f, 3800.0, 1.8) * _highpass(f, 60.0, 1))


def _place(out: np.ndarray, note: np.ndarray, at_s: float, sr: int) -> None:
    i0 = max(0, int(at_s * sr))
    if i0 >= len(out):
        return
    seg = note[: len(out) - i0]
    out[i0:i0 + len(seg)] += seg


E2, F2S, G2, A2, B2 = 82.41, 92.50, 98.00, 110.00, 123.47
C3, D3, E3, G3, A3, B3 = 130.81, 146.83, 164.81, 196.00, 220.00, 246.94
C4, D4, E4, F4S, G4 = 261.63, 293.66, 329.63, 369.99, 392.00
SW = 2.0 / 3.0          # swung off-beat position within a beat


def _riff_events() -> list:
    """Two-bar original riff in E: (beat, freq, mute, velocity, length in beats)."""
    ev = []

    def power(beat, root, fifth, octave, dur, vel=1.0):
        for j, f in enumerate((root, fifth, octave)):        # quick down-strum
            ev.append((beat + j * 0.03, f, False, vel * (1.0 - 0.07 * j), dur))

    # bar 1: E5 stab, chugs, chromatic walk up to A5
    power(0.0, E2, B2, E3, 1.0)
    ev += [(1.0, E2, True, 0.8, 0.3), (1 + SW, E2, True, 0.65, 0.25),
           (2.0, E2, True, 0.85, 0.3), (2 + SW, G2, False, 0.9, 0.3)]
    power(3.0, A2, E3, A3, 0.6, 0.95)
    ev += [(3 + SW, E2, True, 0.7, 0.25)]
    # bar 2: pentatonic lick with the flat-five blue note, back down to the root
    ev += [(4.0, E3, False, 0.9, 0.3), (4 + SW, G3, False, 0.85, 0.3), (5.0, A3, False, 1.0, 0.6),
           (5 + SW, G3, False, 0.8, 0.3), (6.0, E3, False, 0.85, 0.3), (6 + SW, D3, False, 0.8, 0.22),
           (6.9, 155.56, False, 0.75, 0.12), (7.0, E3, False, 0.9, 0.3), (7 + SW, G2, False, 0.8, 0.3)]
    return ev


def _guitar_riff_raw(seconds: float, sr: int, tempo: float = 112.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n + sr)
    beat = 60.0 / tempo
    bar2 = 8.0
    events = _riff_events()
    t0 = 0.0
    while t0 * beat < seconds:
        for b, f, mute, vel, dur in events:
            start = (t0 + b) * beat
            if start >= seconds:
                continue
            jitter = rng.normal(0.0, 0.004)                       # human timing, still seeded
            v = vel * (1.0 + rng.normal(0.0, 0.05))
            note = _note(f, dur * beat, sr, rng, mute=mute, vel=v, ring=0.0 if mute else 0.2)
            _place(out, note, start + jitter, sr)
        t0 += bar2
    return _di(out[:n], sr)


def _guitar_chords_raw(seconds: float, sr: int, tempo: float = 100.0, seed: int = 5) -> np.ndarray:
    """Open-chord strumming G - Em - C - D: down-strum on 1, light up-strum on 3."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n + sr)
    beat = 60.0 / tempo
    chords = [(G2, B2, D3, G3, B3, G4), (E2, B2, E3, G3, B3, E4),
              (C3, E3, G3, C4, E4), (D3, A3, D4, F4S)]
    bar = 0
    while bar * 4 * beat < seconds:
        notes = chords[bar % len(chords)]
        t_bar = bar * 4 * beat
        ring = 4 * beat
        for j, f in enumerate(notes):                              # down-strum, low to high
            v = 0.9 * (1.0 + rng.normal(0, 0.05)) * (1.0 - 0.04 * j)
            _place(out, _note(f, ring, sr, rng, vel=v, detune_cents=3.0), t_bar + j * 0.014, sr)
        for j, f in enumerate(reversed(notes[-3:])):               # up-strum, top strings
            v = 0.45 * (1.0 + rng.normal(0, 0.05))
            _place(out, _note(f, 2 * beat, sr, rng, vel=v, detune_cents=3.0),
                   t_bar + 2 * beat + SW * beat + j * 0.012, sr)
        bar += 1
    return _di(out[:n], sr)


def guitar_riff(seconds: float = 6.0, sr: int = SR, level: float = 0.2, tempo: float = 112.0,
                seed: int = 7) -> np.ndarray:
    """Original two-bar blues riff in E (volts, ``level`` V peak, like a passive pickup).

    Power-chord stabs, palm-muted chugs and a pentatonic lick with the blue
    note, on Karplus-Strong strings with pick clicks, slight detune and a
    pickup-style resonance.  Loops to fill ``seconds``; deterministic.
    """
    return fade(normalize_peak(_guitar_riff_raw(seconds, sr, tempo, seed), level), sr, 0.002, 0.08)


def guitar_chords(seconds: float = 6.0, sr: int = SR, level: float = 0.2) -> np.ndarray:
    """Strummed open chords (G - Em - C - D) that ring and decay naturally (``level`` V peak)."""
    return fade(normalize_peak(_guitar_chords_raw(seconds, sr), level), sr, 0.002, 0.08)


def bass_line(seconds: float = 6.0, sr: int = SR, level: float = 0.3, tempo: float = 108.0,
              seed: int = 11) -> np.ndarray:
    """A walking blues bass line in E (root-fifth-sixth-flat-seventh), plucked, ``level`` V peak."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n + sr)
    beat = 60.0 / tempo
    e1, a1, b1 = 41.20, 55.00, 61.74
    pattern = [e1, e1 * 1.25, b1, e1 * 5 / 3, e1 * 2, e1 * 5 / 3, b1, e1 * 1.25,
               a1, a1 * 1.25, a1 * 1.5, a1 * 5 / 3, b1, b1 * 1.25, b1 * 1.5, e1 * 1.5]
    k = 0
    t = 0.0
    while t < seconds:
        f = pattern[k % len(pattern)]
        off = (k % 2) * (SW - 0.5) * beat        # swing the off-beats
        dur = 0.5 * beat * (0.9 if k % 2 == 0 else 0.75)
        note = _pluck(f, dur + 0.05, sr, rng, t60=1.2, bright=0.35, pick=0.22, vel=0.9 + 0.1 * (k % 2 == 0))
        note *= np.exp(-np.arange(len(note)) / (0.6 * sr))
        rel = min(len(note), int(0.02 * sr))
        note[-rel:] *= np.linspace(1, 0, rel)
        i0 = int((t + off) * sr)
        seg = note[: max(0, len(out) - i0)]
        out[i0:i0 + len(seg)] += seg
        t += 0.5 * beat
        k += 1
    y = out[:n]
    y = _shape(y, sr, lambda f: _resonant_lp(f, 2200.0, 1.2) * _highpass(f, 30.0, 1))
    return fade(normalize_peak(y, level), sr, 0.002, 0.08)


def sine_sweep(seconds: float = 4.0, sr: int = SR, level: float = 0.2, f0: float = 40.0,
               f1: float = 12000.0) -> np.ndarray:
    """Exponential sine sweep from f0 to f1 (``level`` V peak), with short fades."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    k = math.log(f1 / f0)
    phase = 2 * math.pi * f0 * seconds / k * (np.exp(t / seconds * k) - 1.0)
    return fade(level * np.sin(phase), sr, 0.01, 0.05)


def drum_loop(seconds: float = 6.0, sr: int = SR, level: float = 0.3, tempo: float = 108.0,
              seed: int = 3) -> np.ndarray:
    """A synthesized kick/snare/hi-hat groove (swung eighths), ``level`` V peak."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n + sr)
    beat = 60.0 / tempo

    def kick(vel: float) -> np.ndarray:
        m = int(0.45 * sr)
        tt = np.arange(m) / sr
        f = 48.0 + 110.0 * np.exp(-tt / 0.035)
        ph = 2 * math.pi * np.cumsum(f) / sr
        body = np.sin(ph) * np.exp(-tt / 0.22)
        click = rng.uniform(-1, 1, m) * np.exp(-tt / 0.002) * 0.3
        return vel * (body + click)

    def snare(vel: float) -> np.ndarray:
        m = int(0.3 * sr)
        tt = np.arange(m) / sr
        noise = rng.uniform(-1, 1, m)
        noise = np.diff(noise, prepend=0.0) * 0.6          # brighter
        tone = np.sin(2 * math.pi * 190 * tt) * np.exp(-tt / 0.06)
        return vel * (0.8 * noise * np.exp(-tt / 0.09) + 0.6 * tone)

    def hat(vel: float, open_: bool = False) -> np.ndarray:
        m = int((0.25 if open_ else 0.06) * sr)
        tt = np.arange(m) / sr
        noise = rng.uniform(-1, 1, m)
        noise = np.diff(np.diff(noise, prepend=0.0), prepend=0.0) * 0.35
        return vel * noise * np.exp(-tt / (0.08 if open_ else 0.015))

    def put(sample: np.ndarray, at: float) -> None:
        i0 = int(at * sr)
        if i0 >= len(out):
            return
        seg = sample[: len(out) - i0]
        out[i0:i0 + len(seg)] += seg

    bar = 0
    while bar * 4 * beat < seconds:
        b0 = bar * 4 * beat
        put(kick(1.0), b0)
        put(kick(0.8), b0 + (2 + SW) * beat)
        put(kick(0.95), b0 + 2.0 * beat)
        put(snare(0.9), b0 + 1.0 * beat)
        put(snare(1.0), b0 + 3.0 * beat)
        if bar % 2 == 1:
            put(snare(0.35), b0 + (3 + SW) * beat)          # ghost note
        for q in range(4):
            put(hat(0.7), b0 + q * beat)
            put(hat(0.45, open_=(q == 3 and bar % 2 == 1)), b0 + (q + SW) * beat)
        bar += 1
    y = out[:n]
    return fade(normalize_peak(y, level), sr, 0.001, 0.05)


SOURCES = {
    "guitar_riff": guitar_riff,
    "guitar_chords": guitar_chords,
    "bass_line": bass_line,
    "sine_sweep": sine_sweep,
    "drum_loop": drum_loop,
}

DEFAULT_LEVEL = {"guitar_riff": 0.2, "guitar_chords": 0.2, "bass_line": 0.3, "sine_sweep": 0.2,
                 "drum_loop": 0.3}


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim, or loop with a short crossfade, to exactly ``n`` samples."""
    if len(x) == 0:
        return np.zeros(n)
    if len(x) >= n:
        return x[:n].copy()
    reps = int(math.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def source(name: str, sr: int = SR, seconds: float = 6.0, raw: bool = False) -> np.ndarray:
    """A named demo source, or a ``.wav`` file, as samples peak-normalised to 1.0.

    Names: guitar_riff, guitar_chords, bass_line, drum_loop, sine_sweep.  A
    path ending in .wav is loaded (mono mix-down), resampled to ``sr`` and
    trimmed or looped to ``seconds``.  The caller scales to volts.
    Deterministic: the same arguments always give the same samples.
    """
    n = max(1, int(round(float(seconds) * int(sr))))
    key = str(name or "").strip()
    if key.lower().endswith(".wav"):
        x, sr_in = read_wav(key)
        x = resample(x - float(np.mean(x)), sr_in, int(sr))
        y = _fit_length(x, n)
    else:
        fn = SOURCES.get(key.lower())
        if fn is None:
            raise ValueError(f"unknown source '{name}' (choose from {', '.join(SOURCES)}, or a .wav path)")
        global _RAW
        _RAW = bool(raw)
        try:
            y = fn(seconds=n / int(sr), sr=int(sr), level=1.0)
        finally:
            _RAW = False
        y = _fit_length(y, n)
    return normalize_peak(y, 1.0)


def generate(name: str, seconds: float = 6.0, sr: int = SR, level: float | None = None) -> np.ndarray:
    """A named demo source in volts: ``level`` V peak (the source's default when None)."""
    key = (name or "").lower()
    lv = DEFAULT_LEVEL.get(key, 0.2) if level is None else float(level)
    return source(name, sr, seconds) * lv
