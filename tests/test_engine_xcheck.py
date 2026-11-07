"""Cross-check the engine against ngspice-42 on the circuits in ngspice_util.CIRCUITS.

Targets (docs/SPEC.md): DC node voltages within 1% or 1 mV, AC gain within
0.1 dB and -3 dB frequency within 1%, transient RMS error within 2% of the
signal RMS.  ngspice runs in WSL; results are cached in tests/_ngspice so the
suite also runs where WSL is missing, as long as the cache is present.
"""
from __future__ import annotations

import numpy as np
import pytest

import ngspice_util as nu

_AVAILABLE = None


def _can_run(name: str) -> bool:
    global _AVAILABLE
    if (nu.ROOT / name / "op.txt").exists():
        return True
    if _AVAILABLE is None:
        _AVAILABLE = nu.available()
    return _AVAILABLE


@pytest.mark.parametrize("name", sorted(nu.CIRCUITS))
def test_against_ngspice(name):
    if not _can_run(name):
        pytest.skip("ngspice (WSL) not available and no cached reference")
    r = nu.compare(name)
    assert r["dc_converged"]
    assert r["dc_ratio"] <= 1.0, f"DC error {r['dc_max_err_mv']:.3f} mV at {r['dc_worst_node']}"
    if "ac_max_err_db" in r:
        assert r["ac_max_err_db"] <= 0.1
        if r["f3db_err_pct"] is not None:
            assert r["f3db_err_pct"] <= 1.0
    if "tran_rms_err_pct" in r:
        assert r["tran_converged"]
        assert r["tran_rms_err_pct"] <= 2.0


@pytest.mark.parametrize("name", ["diode_clipper_sym", "fuzz_two_transistor", "opamp_soft_clipper",
                                  "mosfet_cs_amp", "mosfet_source_follower"])
def test_audio_rate_against_ngspice(name):
    """process_audio at 48 kHz (one trapezoidal step per sample) vs ngspice's adaptive run."""
    if not _can_run(name):
        pytest.skip("ngspice (WSL) not available and no cached reference")
    r = nu.compare_audio(name)
    assert r["audio_rms_err_pct"] <= 2.0


def test_deck_contains_macromodel():
    from engine import parse
    c = parse(nu.circuit_text("sallen_key_lp"))
    text = nu.deck(c, out_dir=nu.ROOT / "x", ac=(10, 1e5, 10))
    assert ".subckt OA_0 inp inn vcc vee out" in text
    assert "tanh(" in text and "rshunt=1e12" in text
    assert np.isfinite(len(text))
