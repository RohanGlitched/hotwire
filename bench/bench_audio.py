"""Speed benchmark: seconds of CPU to push 1 s of 48 kHz audio through a circuit.

    python bench/bench_audio.py            # all circuits, 3 repeats each, best time kept
    python bench/bench_audio.py --json     # machine-readable

The input is the deterministic guitar riff at 0.2 V peak (pickup level), so
the fuzz is driven hard into clipping, the worst case for Newton.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from engine import audio, parse, process_audio  # noqa: E402

SR = 48000
CIRCUITS = {
    "two-transistor fuzz (bench/fuzz2.cir, 2x BC549C)": (ROOT / "bench" / "fuzz2.cir").read_text(),
}


def _xcheck(name: str) -> str:
    import ngspice_util as nu
    return nu.circuit_text(name)


EXTRA = ["fuzz_two_transistor", "diode_clipper_sym", "opamp_soft_clipper", "jfet_booster",
         "ce_amp", "mosfet_cs_amp", "mosfet_source_follower", "baxandall_tone_stack"]
LIBRARY = ["silicon_fuzz", "germanium_fuzz", "octave_fuzz", "softclip_overdrive", "treble_booster",
           "ce_amplifier", "jfet_boost", "mosfet_boost"]


def bench(text: str, x: np.ndarray, repeats: int = 3) -> dict:
    c = parse(text)
    best = None
    stats: dict = {}
    for _ in range(repeats):
        st: dict = {}
        t0 = time.perf_counter()
        y = process_audio(c, x, SR, stats=st)
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best, stats = dt, st
    return {"seconds": best, "realtime_factor": best / (len(x) / SR),
            "newton_per_sample": stats["newton_iterations"] / max(1, stats["steps"]),
            "halved_steps": stats["halved_steps"], "failed_steps": stats["failed_steps"],
            "out_peak_v": float(np.max(np.abs(y)))}


def main() -> None:
    x = audio.source("guitar_riff", SR, 1.0) * 0.2
    rows = []
    for label, text in CIRCUITS.items():
        rows.append((label, bench(text, x)))
    for name in EXTRA:
        rows.append((f"xcheck: {name}", bench(_xcheck(name), x)))
    for name in LIBRARY:
        path = ROOT / "library" / f"{name}.cir"
        if path.exists():
            rows.append((f"library: {name}", bench(path.read_text(encoding="utf-8"), x)))
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "cpu": platform.processor() or platform.machine(), "os": platform.platform(),
            "cpus": os.cpu_count()}
    if "--json" in sys.argv:
        print(json.dumps({"machine": info, "results": rows}, indent=1))
        return
    print(f"Python {info['python']}, numpy {info['numpy']}, {info['cpu']}")
    print(f"{'circuit':58s} {'s / 1 s audio':>13s} {'Newton/sample':>14s}")
    for label, r in rows:
        print(f"{label:58s} {r['seconds']:13.2f} {r['newton_per_sample']:14.2f}")


if __name__ == "__main__":
    main()
