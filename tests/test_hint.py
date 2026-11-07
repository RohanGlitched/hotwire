"""The context hint must fire on electronics questions and stay out of ordinary chat."""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
hw = importlib.import_module(ROOT.name)


def test_fires_on_electronics():
    for t in ["design me a fuzz pedal", "why does my transistor get hot?", "check this netlist",
              "a 1 kHz low-pass filter for a 9 V circuit", "my op-amp clips", "boost with more treble for my guitar"]:
        assert hw._is_electronics(t), t


def test_quiet_on_ordinary_chat():
    for t in ["low-pass filter my pandas time series", "boost wifi signal", "filter this CSV by date", "gender bias in this dataset", "boost my productivity",
              "distortion in media coverage", "solder-free fix for my chair", "my bike pedal squeaks",
              "summarise this paper on signal processing in economics"]:
        assert not hw._is_electronics(t), t
