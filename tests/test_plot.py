"""SVG plots: well-formed, themeable, sane extents."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from engine.plot import bode_svg, scope_svg, spectrum_svg, waveform_svg

NS = "{http://www.w3.org/2000/svg}"


def _check(text, min_w=300):
    root = ET.fromstring(text)
    assert root.tag == NS + "svg"
    x, y, w, h = (float(v) for v in root.get("viewBox").split())
    assert w >= min_w and h > 30
    assert root.find(NS + "title") is not None
    assert "var(--hotwire-ink, #1d1d1f)" in text
    assert "nan" not in text.lower() and "inf" not in text.lower().replace("info", "")
    # every coordinate in path data stays inside a generous margin around the viewBox
    for d in re.findall(r' d="([^"]+)"', text):
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
        assert all(-5 <= v <= max(w, h) + 5 for v in nums), d[:80]
    return root


def _rc(f, fc=1000.0):
    return 1 / (1 + 1j * f / fc)


def test_bode_with_specs():
    f = np.geomspace(10, 100e3, 161)
    specs = [{"kind": "fc", "lo": 900, "hi": 1100, "measured": 1000, "pass": True},
             {"kind": "gain_db", "f": 100, "lo": -1, "hi": 1, "measured": -0.04, "pass": True},
             {"kind": "gain_db", "f": 10000, "lo": None, "hi": -25, "measured": -20, "pass": False}]
    text = bode_svg(f, _rc(f), specs=specs, title="RC low-pass")
    _check(text)
    assert "var(--hotwire-pass, #15803d)" in text and "var(--hotwire-fail, #b91c1c)" in text
    for lab in (">10<", ">100<", ">1k<", ">10k<", ">100k<"):
        assert lab in text
    assert "phase" in text


def test_bode_without_phase_and_degenerate_input():
    f = np.geomspace(20, 20000, 50)
    _check(bode_svg(f, np.ones_like(f), phase=False))
    _check(bode_svg([], []))
    _check(bode_svg(f, np.zeros_like(f)))


def test_scope_four_traces_and_legend():
    t = np.arange(4800) / 96000
    tr = {f"v{i}": np.sin(2 * math.pi * 1000 * t + i) * (i + 1) for i in range(5)}
    text = scope_svg(t, tr, title="scope")
    _check(text)
    assert text.count("<path d=\"M") >= 4
    assert ">v0<" in text and ">v3<" in text and ">v4<" not in text   # at most four traces
    assert ">ms<" in text


def test_scope_flat_and_short():
    _check(scope_svg([0.0], {"x": [1.0]}))
    t = np.linspace(0, 0.01, 100)
    _check(scope_svg(t, {"dc": np.full(100, 4.5)}))


def test_spectrum_with_reference():
    sr = 48000
    t = np.arange(sr // 2) / sr
    dry = 0.2 * np.sin(2 * math.pi * 220 * t)
    wet = np.tanh(5 * dry)
    text = spectrum_svg(wet, sr, ref=dry, title="A/B")
    _check(text)
    assert ">dry<" in text and ">circuit<" in text and ">1k<" in text


@pytest.mark.parametrize("n", [0, 10, 48000])
def test_waveform(n):
    x = np.sin(np.arange(n) / 30.0)
    text = waveform_svg(x, 48000, width=480, height=64)
    root = ET.fromstring(text)
    assert root.get("viewBox") == "0 0 480 64"
