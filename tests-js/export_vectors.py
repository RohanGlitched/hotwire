"""Export test vectors for the JS live engine: compile_json + input + Python process_audio output.

    python tests-js/export_vectors.py            # writes tests-js/vectors/*.json

Each circuit in tests-js/circuits (and every library/*.cir that has *@ in and *@ out)
is rendered at 48 kHz with a plucked-string riff; the node test runs the same input
through the JS engine and compares.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.analyses import op, process_audio  # noqa: E402
from engine.export import compile_json  # noqa: E402
from engine.netlist import parse  # noqa: E402

SR = 48000
SECONDS = 0.2


def riff(sr: int, seconds: float, seed: int = 3) -> np.ndarray:
    """Karplus-Strong plucks (E2 A2 D3 G3 ...), deterministic."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    out = np.zeros(n)
    notes = [82.41, 110.0, 146.83, 196.0, 110.0, 82.41]
    step = n // len(notes)
    for k, f in enumerate(notes):
        period = int(sr / f)
        buf = rng.uniform(-1, 1, period)
        start = k * step
        for i in range(start, n):
            j = (i - start) % period
            nxt = (j + 1) % period
            v = buf[j]
            buf[j] = 0.996 * 0.5 * (buf[j] + buf[nxt])
            out[i] += v * math.exp(-(i - start) / (0.6 * sr))
    out /= np.max(np.abs(out)) or 1.0
    # short fade-in so the first sample is exactly zero (process_audio's reference)
    ramp = np.minimum(1.0, np.arange(n) / 64.0)
    return out * ramp


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def export(path: Path, out_dir: Path, level: float, pots: dict | None = None, tag: str = ""):
    c = parse(path.read_text(encoding="utf-8"))
    if not c.input or not c.output:
        return None
    if pots:
        c = c.with_pots(pots)
    x = np.array([float(f"{v:.7g}") for v in riff(SR, SECONDS) * level])
    t0 = time.time()
    y = process_audio(c, x, SR)
    secs = time.time() - t0
    cj = compile_json(c, op(c), SR)
    name = ("lib-" if path.parent.name == "library" else "") + path.stem + (f"-{tag}" if tag else "")
    rec = {"name": name, "title": c.title, "sr": SR, "level": level, "compiled": clean(cj),
           "input": [float(v) for v in x], "output": [float(f"{v:.8g}") for v in y], "python_seconds": secs}
    (out_dir / f"{name}.json").write_text(json.dumps(rec, allow_nan=False), encoding="utf-8")
    return name, secs


def main():
    out_dir = HERE / "vectors"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    jobs = []
    for p in sorted((HERE / "circuits").glob("*.cir")):
        jobs.append((p, 0.3, None, ""))
    jobs.append((HERE / "circuits" / "fuzz.cir", 0.2, {"RV1": 0.1, "RV2": 0.9}, "knobs"))
    jobs.append((HERE / "circuits" / "opamp_od.cir", 0.5, {"RV1": 0.95}, "hot"))
    for p in sorted((ROOT / "library").glob("*.cir")):
        jobs.append((p, 0.2, None, ""))
    for p, level, pots, tag in jobs:
        try:
            r = export(p, out_dir, level, pots, tag)
            if r:
                print(f"{r[0]:<34} python {r[1]:.2f} s")
        except Exception as e:  # a library circuit the engine can't run yet
            print(f"{p.stem:<34} skipped: {e}")


if __name__ == "__main__":
    main()
