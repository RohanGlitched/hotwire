"""Render example plots into docs/gallery/plots/ (needs the simulator in engine/)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from engine import plot  # noqa: E402
from engine.analyses import ac, process_audio  # noqa: E402
from engine.netlist import parse  # noqa: E402


def main():
    c = parse((ROOT / "library/sallen_key_lowpass.cir").read_text())
    r = ac(c, fstart=10, fstop=100e3)
    specs = [{"kind": "fc", "lo": 1000, "hi": 1150, "measured": 1080, "pass": True},
             {"kind": "gain_db", "f": 100, "lo": -0.3, "hi": 0.3, "measured": 0.0, "pass": True},
             {"kind": "gain_db", "f": 10000, "lo": None, "hi": -36, "measured": -39.5, "pass": True}]
    (HERE / "bode.svg").write_text(plot.bode_svg(r.f, r.h, specs=specs, title=c.title), encoding="utf-8")
    fail = [{"kind": "gain_db", "f": 1000, "lo": 18, "hi": None, "measured": 16.4, "pass": False}]
    (HERE / "bode-fail.svg").write_text(plot.bode_svg(r.f, r.h * 6.6, specs=fail, title="Gain spec failing",
                                                      phase=False), encoding="utf-8")
    sr = 96000
    t = np.arange(int(sr * 0.004)) / sr
    x = 0.3 * np.sin(2 * np.pi * 1000 * t)
    c2 = parse((ROOT / "library/diode_hard_clipper.cir").read_text())
    y = process_audio(c2, np.concatenate([np.zeros(2000), x]), sr)[2000:]
    (HERE / "scope.svg").write_text(plot.scope_svg(t, {"in": x, "out": y}, title="Hard clipper, 1 kHz"),
                                    encoding="utf-8")
    sr = 48000
    tt = np.arange(sr) / sr
    dry = 0.3 * np.sin(2 * np.pi * 220 * tt) * np.exp(-tt * 2) + 0.1 * np.sin(2 * np.pi * 440 * tt)
    wet = np.tanh(4 * dry)
    (HERE / "spectrum.svg").write_text(plot.spectrum_svg(wet, sr, ref=dry, title="Spectrum, dry vs circuit"),
                                       encoding="utf-8")
    (HERE / "waveform.svg").write_text(plot.waveform_svg(wet, sr), encoding="utf-8")


if __name__ == "__main__":
    main()
