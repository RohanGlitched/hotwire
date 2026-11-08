"""Regression tests for defects found by the adversarial review rounds."""
import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
import importlib  # noqa: E402

hw = importlib.import_module(ROOT.name)
from engine import parse  # noqa: E402
from engine.specs import Evaluator, evaluate  # noqa: E402

LATCH = """*@ in VIN
*@ out out
*@ spec gain_db(1k) > 15
VIN in 0 DC 0 AC 1
VP vp 0 DC 12
VN vn 0 DC -12
R1 in a 10k
RF a out 100k
X1 a 0 vp vn out TL072
"""
SLOW_WIEN = """*@ in VIN
*@ out out
*@ spec gain_db(1k) > -40
VIN in 0 DC 0 AC 1
VP vp 0 DC 15
VN vn 0 DC -15
RS in p 1meg
R1 p a 16k
C1 a out 10n
R2 p 0 16k
C2 p 0 10n
RF out m 20.5k
RG m 0 10k
X1 p m vp vn out TL072
"""


@pytest.fixture()
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTWIRE_DATA", str(tmp_path))
    hw.tools._BENCHES.clear()
    yield hw.tools
    hw.tools._BENCHES.clear()


def call(tools, name, **args):
    return json.loads(getattr(tools, name)(args))


def test_ac_simulate_works(tools):
    r = call(tools, "circuit_simulate", path="silicon_fuzz", analysis="ac")
    assert r["ok"] and len(r["points"]) > 5


def test_latch_fails_check_tune_and_yield(tools):
    assert call(tools, "circuit_check", netlist=LATCH)["verdict"] == "fail"
    t = call(tools, "circuit_tune", netlist=LATCH, vary=["RF"])
    assert t["ok"] and t["all_pass"] is False
    y = call(tools, "circuit_yield", netlist=LATCH)
    assert y["ok"] is False and "nominal" in y["error"]


def test_slow_oscillator_is_caught(tools):
    r = call(tools, "circuit_check", netlist=SLOW_WIEN)
    assert r["stability"]["state"] == "oscillates" and r["verdict"] == "fail"


def test_no_false_alarm_from_ic_or_lfo(tools):
    ce = (ROOT / "library" / "ce_amplifier.cir").read_text(encoding="utf-8")
    ic = ce.replace("C1 in b 1u", "C1 in b 1u ic=0").replace("C2 c out 1u", "C2 c out 1u ic=0")
    assert call(tools, "circuit_check", netlist=ic)["verdict"] == "pass"
    lfo = "*@ in VIN\n*@ out out\n*@ spec gain_db(1k) in -10..0\nVIN in 0 DC 0 AC 1\nVLFO lfo 0 DC 0 SIN(0 1 5)\nR1 in out 10k\nR2 lfo out 10k\nR3 out 0 10k\n"
    assert call(tools, "circuit_check", netlist=lfo)["verdict"] == "pass"


def test_guitar_volume_percent(tools):
    r = call(tools, "circuit_audition", path="rc_lowpass", guitar_volume="70%", seconds=0.5)
    assert r["guitar"]["volume"] == pytest.approx(0.7)


def test_specs_as_json_string(tools):
    r = call(tools, "circuit_check", path="rc_lowpass", specs='["fc < 2k", "gain_db(100) > -1"]')
    assert all(s["pass"] for s in r["specs"])


def test_fc_from_passband_on_peaked_lowpass():
    c = parse("*@ in VIN\n*@ out out\nVIN in 0 DC 0 AC 1\nR1 in a 10k\nR2 a b 10k\nC1 a out 40n\nC2 b 0 2.5n\nE1 out 0 b 0 1\n")
    fc = evaluate(c, ["fc > 1"])[-1].measured
    f0 = 1 / (2 * math.pi * 10e3 * 1e-8)
    u = (1.75 + math.sqrt(1.75 ** 2 + 4)) / 2      # Q = 2: |H| = 1/sqrt(2) at x^2 = u
    assert fc == pytest.approx(f0 * math.sqrt(u), rel=1e-3)


def test_dead_output_cleanup_fails(tools):
    dead = "*@ in VIN\n*@ out out\n*@ spec cleanup_vol(300, 0.1, 5) > 0.5\nVIN in 0 DC 0 AC 1\nRIN in 0 10k\nVCC vcc 0 DC 9\nR1 vcc out 10k\nR2 out 0 10k\n"
    r = call(tools, "circuit_check", netlist=dead)
    assert r["verdict"] == "fail" and "no 300 Hz signal" in r["summary"]


def test_supply_current_adds_same_polarity_rails():
    c = parse("*@ supply VA VB\nVA a 0 DC 9\nVB b 0 DC 9\nRA a 0 9k\nRB b 0 9k\n")
    assert Evaluator(c).supply_current() == pytest.approx(2e-3, rel=1e-6)


def test_reversed_input_source_zin():
    c = parse("*@ in VIN\n*@ out o\nVIN 0 in AC 1\nR1 in o 10k\nR2 o 0 10k\n")
    assert evaluate(c, ["zin(1k) > 1"])[-1].measured == pytest.approx(20e3, rel=1e-3)


def test_scientific_notation_in_params():
    from engine.netlist import eval_expr
    assert eval_expr("1e3") == 1000.0 and eval_expr("4.7e3") == 4700.0 and eval_expr("1e-9*2") == 2e-9
    c = parse(".param RR=4.7e3\nVIN in 0 AC 1\nR1 in o {RR}\nC1 o 0 10n\n")
    assert c.element("R1").value == 4700.0


def test_common_parts_resolve():
    c = parse("VCC v 0 9\nR1 v b 470k\nQ1 c b 0 BC547B\nR2 v c 10k\nX1 c 0 v 0 o LM741\nX2 c 0 v 0 o2 TL074\nQ2 v c 0 2N5089\n")
    assert {"BC547B", "LM741", "TL074", "2N5089"} <= set(c.models)


def test_tune_never_saves_a_regression(tools):
    od = (ROOT / "library" / "softclip_overdrive.cir").read_text(encoding="utf-8")
    r = call(tools, "circuit_tune", netlist=od, vary=["RPD"], specs=["fc < 10"])
    assert r["ok"]
    assert r.get("kept_original") or r["all_pass"] or not r.get("changes")


def test_high_voltage_warning(tools):
    r = call(tools, "circuit_check", netlist="V1 a 0 DC 0 SIN(0 325 50)\nR1 a 0 10k\n")
    assert any(i["code"] == "high_voltage" for i in r["erc"])


# --- catalog review (teknium1, Nov 8): output paths, WAV inputs, per-profile Bench ---------

def test_pot_name_cannot_steer_paths(tools, tmp_path):
    from engine.netlist import NetlistError
    bad = (ROOT / "library" / "diode_hard_clipper.cir").read_text(encoding="utf-8").replace(
        "*@ pot RV1 ", "*@ pot ../../../evil/RV1 ")
    with pytest.raises(NetlistError, match="pot name"):
        parse(bad)
    r = call(tools, "circuit_audition", netlist=bad, seconds=0.5)
    assert r["ok"] is False and "pot name" in r["error"]
    assert not (tmp_path.parent / "evil").exists()


def test_store_file_refuses_escapes(tmp_path):
    from engine.store import Store
    st = Store(tmp_path)
    d = st.root / "x" / "v1"
    d.mkdir(parents=True)
    assert st.file(d, "dry-guitar-riff.wav").parent == d.resolve()
    for name in ("../a.wav", "a/../../b.wav", "..", "x/y.wav", ""):
        with pytest.raises(ValueError):
            st.file(d, name)
    with pytest.raises(ValueError):
        st.file(tmp_path.parent, "a.wav")


def test_audition_files_stay_in_store(tools, tmp_path):
    r = call(tools, "circuit_audition", path="diode_hard_clipper", seconds=0.5, pots={"RV1": 0.3})
    assert r["ok"]
    for f in r["media"] + [r["wet"], r["dry"]]:
        assert pathlib.Path(f).resolve().is_relative_to(tmp_path.resolve())
    assert "rv130" in pathlib.Path(r["wet"]).name


def test_wav_inputs_are_checked(tools, tmp_path):
    import numpy as np
    from engine import audio
    good = tmp_path / "dry.wav"
    audio.write_wav(good, np.sin(np.linspace(0, 2000, 24000)) * 0.3, 24000)
    notwav = tmp_path / "dry.txt"
    notwav.write_bytes(good.read_bytes())
    (tmp_path / "folder.wav").mkdir()
    big = tmp_path / "big.wav"
    with open(big, "wb") as f:
        f.truncate(audio.AUDIO_MAX + 1)
    for p, msg in ((notwav, "not a .wav"), (big, "at most"), (tmp_path / "missing.wav", "can't find"),
                   (tmp_path / "folder.wav", "can't find")):
        with pytest.raises(ValueError, match=msg):
            audio.user_wav(p)
        r = call(tools, "circuit_match", path="diode_hard_clipper", dry=str(p), wet=str(good))
        assert r["ok"] is False
    r = call(tools, "circuit_audition", path="rc_lowpass", source=str(big), seconds=0.5)
    assert r["ok"] is False and "at most" in r["error"]


def test_bench_follows_the_active_profile(tools, tmp_path, monkeypatch):
    a, b = tmp_path / "profile-a", tmp_path / "profile-b"
    monkeypatch.setenv("HOTWIRE_DATA", str(a))
    ra = json.loads(tools.circuit_check({"path": "rc_lowpass", "name": "alpha"}))
    monkeypatch.setenv("HOTWIRE_DATA", str(b))
    rb = json.loads(tools.circuit_check({"path": "rc_lowpass", "name": "beta"}))
    assert (a / "designs" / ra["design"]).is_dir() and not (a / "designs" / rb["design"]).exists()
    assert (b / "designs" / rb["design"]).is_dir()
    assert tools.bench().store.root == b / "designs"
