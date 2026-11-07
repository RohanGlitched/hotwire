"""Tone matching recovers hidden knob settings (slow: set HOTWIRE_SLOW=1 to run)."""
import os

import numpy as np
import pytest

from engine import parse
from engine import match as m
from engine.audio import generate

pytestmark = pytest.mark.skipif(not os.environ.get("HOTWIRE_SLOW"), reason="slow: set HOTWIRE_SLOW=1")


def test_fit_recovers_drive_and_tone():
    c = parse(open("library/softclip_overdrive.cir", encoding="utf-8").read())
    sr = 24000
    dry = generate("guitar_riff", seconds=1.0, sr=sr)
    dry = dry / np.max(np.abs(dry))
    wet = m._render(c.with_pots({"RV1": 0.85, "RV2": 0.2}), dry, sr, 0.2)
    r = m.fit_pots(c, dry, wet, sr)
    assert abs(r["positions"]["RV1"] - 0.85) <= 0.15
    assert abs(r["positions"]["RV2"] - 0.2) <= 0.15
    assert "RV3" in r["ignored"]
    assert r["distance"] < 2


def test_resample_is_band_limited():
    sr = 48000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 15000 * t)          # above the new Nyquist (12 kHz)
    y = m._resample(x, sr, 24000)
    assert np.max(np.abs(y)) < 0.05             # removed, not aliased to 9 kHz
