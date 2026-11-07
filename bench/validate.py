"""Print the ngspice cross-check table used in docs/VALIDATION.md.

    python bench/validate.py           # uses cached ngspice results in tests/_ngspice
    python bench/validate.py --force   # re-run ngspice-42 in WSL for every circuit
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import ngspice_util as nu  # noqa: E402

AUDIO = ["diode_clipper_sym", "diode_clipper_asym", "fuzz_two_transistor", "opamp_soft_clipper",
         "opamp_inverting_clip", "mosfet_cs_amp", "mosfet_source_follower",
         "jfet_booster", "ce_amp"]


def fmt(x, spec: str, none: str = "n/a") -> str:
    return none if x is None else format(x, spec)


def main() -> None:
    force = "--force" in sys.argv
    print("| circuit | DC max error | DC worst node | AC max error | -3 dB error | tran RMS error | Newton/step |")
    print("|---|---:|---|---:|---:|---:|---:|")
    worst = {"dc": 0.0, "ac": 0.0, "f3": 0.0, "tr": 0.0}
    for name in nu.CIRCUITS:
        r = nu.compare(name, force=force)
        st = r.get("tran_stats", {})
        nps = st.get("newton_iterations", 0) / max(1, st.get("steps", 1))
        print(f"| `{name}` | {r['dc_max_err_mv'] * 1e3:.3g} µV | {r['dc_worst_node']} | "
              f"{fmt(r.get('ac_max_err_db'), '.2e')} dB | {fmt(r.get('f3db_err_pct'), '.2e')} % | "
              f"{fmt(r.get('tran_rms_err_pct'), '.3f')} % | {nps:.2f} |")
        worst["dc"] = max(worst["dc"], r["dc_ratio"])
        worst["ac"] = max(worst["ac"], r.get("ac_max_err_db") or 0.0)
        worst["f3"] = max(worst["f3"], r.get("f3db_err_pct") or 0.0)
        worst["tr"] = max(worst["tr"], r.get("tran_rms_err_pct") or 0.0)
    print()
    print(f"worst: DC {worst['dc'] * 100:.4f}% of allowance, AC {worst['ac']:.2e} dB, "
          f"-3 dB {worst['f3']:.2e} %, tran {worst['tr']:.3f} %")
    print()
    print("| circuit (process_audio at 48 kHz) | RMS error vs ngspice |")
    print("|---|---:|")
    for name in AUDIO:
        r = nu.compare_audio(name)
        print(f"| `{name}` | {r['audio_rms_err_pct']:.3f} % |")


if __name__ == "__main__":
    main()
