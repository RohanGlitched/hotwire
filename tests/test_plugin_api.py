"""REST routes the Desktop Bench pane calls (dashboard/plugin_api.py)."""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BOOST = """*@ title Test booster
*@ in VIN
*@ out out
*@ supply VCC
*@ pot RV1 0 vol_w vol_a 100k taper=log pos=0.5 label="Volume"
*@ spec gain_db(1k) > 0
*@ spec v(c) in 3..7
VCC vcc 0 DC 9
VIN in 0 DC 0 AC 1
C1 in b 100n
R1 vcc b 1meg
R2 b 0 220k
Q1 c b e 2N3904
R3 vcc c 10k
R4 e 0 1k
C2 c vol_a 1u
RL vol_w out 10
R5 out 0 1meg
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTWIRE_DATA", str(tmp_path))
    spec = importlib.util.spec_from_file_location("hotwire_plugin_api_test", ROOT / "dashboard" / "plugin_api.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hotwire_plugin_api_test"] = mod
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/hotwire")
    c = TestClient(app)
    c.mod = mod
    yield c
    sys.modules.pop("hotwire_plugin_api_test", None)


def _seed(client, text=BOOST, note=""):
    return client.mod.bench().check(netlist=text, note=note)


def test_empty_list(client):
    r = client.get("/api/plugins/hotwire/designs")
    assert r.status_code == 200
    assert r.json() == {"designs": []}


def test_design_detail_versions_and_pots(client):
    r1 = _seed(client)
    did = r1["design"]
    client.mod.bench().check(design=did, pots={"RV1": 0.9}, note="more volume")
    r = client.get(f"/api/plugins/hotwire/designs/{did}")
    assert r.status_code == 200
    d = r.json()
    assert d["id"] == did and d["latest"] == 2
    assert [v["v"] for v in d["versions"]] == [1, 2]
    assert all(v["checked"] for v in d["versions"])
    assert d["versions"][1]["note"] == "more volume"
    cur = d["current"]
    assert cur["version"] == 2
    assert "pos=0.9" in cur["netlist"]
    assert cur["check"]["verdict"] in ("pass", "fail", "pass_with_warnings")
    pot = cur["pots"][0]
    assert pot["name"] == "RV1" and pot["label"] == "Volume" and pot["taper"] == "log"
    assert pot["pos"] == pytest.approx(0.9)

    lst = client.get("/api/plugins/hotwire/designs").json()["designs"]
    assert lst[0]["id"] == did and lst[0]["verdict"]

    v1 = client.get(f"/api/plugins/hotwire/designs/{did}/v/1").json()
    assert v1["version"] == 1 and v1["pots"][0]["pos"] == pytest.approx(0.5)


def test_check_with_pots_does_not_fork(client):
    did = _seed(client)["design"]
    r = client.post("/api/plugins/hotwire/check", json={"design": did, "pots": {"RV1": 0.9}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pots"] == {"RV1": 0.9}
    assert body["specs"] and "verdict" in body
    assert "design" not in body  # save=False
    meta = client.get(f"/api/plugins/hotwire/designs/{did}").json()
    assert meta["latest"] == 1
    # louder with the volume up
    lo = client.post("/api/plugins/hotwire/check", json={"design": did, "pots": {"RV1": 0.1}}).json()
    g_hi = next(s for s in body["specs"] if s["metric"] == "gain_db")["measured"]
    g_lo = next(s for s in lo["specs"] if s["metric"] == "gain_db")["measured"]
    assert g_hi > g_lo


def test_check_inline_netlist(client):
    r = client.post("/api/plugins/hotwire/check", json={"netlist": BOOST})
    assert r.status_code == 200
    assert r.json()["title"] == "Test booster"


@pytest.mark.parametrize("body,code", [
    ({"design": "../etc"}, 400),
    ({"design": "nope"}, 404),
    ({"netlist": ""}, 400),
    ({"netlist": "R1 a"}, 422),
])
def test_check_validation(client, body, code):
    r = client.post("/api/plugins/hotwire/check", json=body)
    assert r.status_code == code
    assert "error" in r.json() and "Traceback" not in r.text


def test_check_bad_pots(client):
    did = _seed(client)["design"]
    for pots in ({"RV9": 0.5}, {"RV1": "x"}, {"bad name": 1}, "RV1"):
        r = client.post("/api/plugins/hotwire/check", json={"design": did, "pots": pots})
        assert r.status_code == 400, pots
        assert "error" in r.json()


def test_bad_json_body(client):
    r = client.post("/api/plugins/hotwire/check", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_compile(client):
    did = _seed(client)["design"]
    r = client.get(f"/api/plugins/hotwire/compile/{did}", params={"pots": "RV1:0.25"})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["version"] == 1 and c["input"] == "VIN"
    assert isinstance(c["output"], int) and c["output"] >= 1
    assert len(c["op"]) == len(c["nodes"])
    pot = c["pots"][0]
    assert pot["pos"] == pytest.approx(0.25) and pot["halves"] == ["RV1_A", "RV1_B"]
    halves = {e["name"]: e["value"] for e in c["elements"] if e["name"].startswith("RV1_")}
    frac = (11 ** 0.25 - 1) / 10
    assert halves["RV1_A"] == pytest.approx(100e3 * frac)
    json.dumps(c, allow_nan=False)
    assert client.get("/api/plugins/hotwire/compile/nope").status_code == 404
    assert client.get(f"/api/plugins/hotwire/compile/{did}", params={"pots": "RV1"}).status_code == 400


def test_file_guard(client, tmp_path):
    did = _seed(client)["design"]
    inside = tmp_path / "designs" / did / "v1" / "netlist.cir"
    r = client.get("/api/plugins/hotwire/file", params={"path": str(inside)})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    outside = tmp_path.parent / "secret.svg"
    outside.write_text("<svg/>")
    r = client.get("/api/plugins/hotwire/file", params={"path": str(outside)})
    assert r.status_code == 403
    r = client.get("/api/plugins/hotwire/file", params={"path": str(tmp_path / "designs" / ".." / ".." / "secret.svg")})
    assert r.status_code == 403
    (tmp_path / "x.py").write_text("print(1)")
    r = client.get("/api/plugins/hotwire/file", params={"path": str(tmp_path / "x.py")})
    assert r.status_code == 403
    assert client.get("/api/plugins/hotwire/file", params={"path": ""}).status_code == 400
    svg = tmp_path / "designs" / did / "v1" / "t.svg"
    svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    r = client.get("/api/plugins/hotwire/file", params={"path": str(svg)})
    assert r.headers["content-type"] == "image/svg+xml"
    r = client.get("/api/plugins/hotwire/svg", params={"path": str(svg)})
    assert r.json()["svg"].startswith("<svg")


def test_audition_and_audio(client):
    did = _seed(client)["design"]
    r = client.post("/api/plugins/hotwire/audition",
                    json={"design": did, "pots": {"RV1": 0.7}, "source": "sine_sweep", "seconds": 0.5})
    if r.status_code == 503:
        pytest.skip("engine.audio not written yet")
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["design"] == did and a["version"] == 1  # no fork for a knob move
    w = client.get("/api/plugins/hotwire/audio", params={"path": a["wet"]}).json()
    raw = base64.b64decode(w["b64"])
    assert raw[:4] == b"RIFF"
    d = client.get(f"/api/plugins/hotwire/designs/{did}").json()
    assert d["latest_audition"]["wet"] == a["wet"]
    bad = client.post("/api/plugins/hotwire/audition", json={"design": did, "source": "/etc/passwd"})
    assert bad.status_code == 400
    bad = client.post("/api/plugins/hotwire/audition", json={"design": did, "seconds": 60})
    assert bad.status_code == 400


def test_library_and_bom(client):
    r = client.get("/api/plugins/hotwire/library")
    assert r.status_code == 200 and "circuits" in r.json()
    did = _seed(client)["design"]
    b = client.get(f"/api/plugins/hotwire/bom/{did}")
    if b.status_code == 200:
        assert b.json()["rows"]
    assert client.get(f"/api/plugins/hotwire/bom/{did}", params={"series": "E7"}).status_code == 400


def test_spec_rows_get_plain_labels_and_units(client):
    did = _seed(client)["design"]
    chk = client.get(f"/api/plugins/hotwire/designs/{did}").json()["current"]["check"]
    gain = next(s for s in chk["specs"] if s["metric"] == "gain_db")
    assert gain["label"] == "Gain at 1 kHz"
    assert gain["measured_text"].endswith(" dB") and gain["target_text"].startswith("above")
    v = next(s for s in chk["specs"] if s["metric"] == "v")
    assert v["target_text"] == "3 V to 7 V" and v["measured_text"].endswith(" V")
    assert chk["outdated"] is False


def test_old_check_is_filled_in_and_flagged(client):
    fill = client.mod._fill_specs
    rows = fill([
        {"spec": "gain_db(1k) > 0 dB", "metric": "gain_db", "measured": 12.3456, "target": "above 0dB", "pass": True, "unit": "dB"},
        {"spec": "supply_current < 2mA", "metric": "supply_current", "measured": 0.00102, "target": "below 2mA", "pass": True, "unit": "A"},
        {"spec": "thd_gv(300, 0.1, 0.25) < 5 %", "metric": "?", "measured": None, "target": "—", "pass": False,
         "explanation": "unknown spec metric 'thd_gv' (known: gain_db, ...)", "error": "unknown spec metric 'thd_gv'"},
    ])
    assert rows[0]["label"] == "Gain at 1 kHz" and rows[0]["measured_text"] == "12.3 dB"
    assert rows[1]["measured_text"] == "1.02 mA" and rows[1]["target_text"] == "below 2 mA"
    assert rows[2]["outdated"] is True and rows[2]["label"].startswith("Distortion with the guitar volume")
    assert "outdated" not in rows[0]


def test_stress_view_headline_and_plain_polarity(client):
    lib = (ROOT / "library" / "silicon_fuzz.cir").read_text(encoding="utf-8")
    did = client.mod.bench().check(netlist=lib)["design"]
    sv = client.get(f"/api/plugins/hotwire/designs/{did}").json()["current"]["check"]["stress_view"]
    assert sv["tone"] == "pass" and sv["parts"] == 12
    assert sv["headline"].startswith("All 12 parts within ratings · worst: ")
    texts = [r["text"] for r in sv["rows"]]
    assert any(t.startswith("C1 (electrolytic): + side toward Q1's base") for t in texts), texts
    assert not any("'" + "b1" + "'" in t for t in texts)
    levels = [r["level"] for r in sv["rows"]]
    assert levels == sorted(levels, key=lambda lv: {"error": 0, "warning": 1}.get(lv, 2))


def test_recheck_saves_in_place(client, tmp_path):
    did = _seed(client)["design"]
    p = tmp_path / "designs" / did / "v1" / "check.json"
    old = json.loads(p.read_text(encoding="utf-8"))
    old["specs"][0]["error"] = "unknown spec metric 'gain_db'"
    old["specs"][0]["pass"] = False
    p.write_text(json.dumps(old), encoding="utf-8")
    d = client.get(f"/api/plugins/hotwire/designs/{did}").json()
    assert d["current"]["check"]["outdated"] is True
    r = client.post("/api/plugins/hotwire/check", json={"design": did, "version": 1, "save": True})
    assert r.status_code == 200, r.text
    assert r.json()["saved"] is True
    d = client.get(f"/api/plugins/hotwire/designs/{did}").json()
    assert d["latest"] == 1  # no new version
    assert d["current"]["check"]["outdated"] is False
    assert client.post("/api/plugins/hotwire/check", json={"design": did, "save": True, "pots": {"RV1": 0.2}}).status_code == 400


def test_audition_guitar_volume_and_spectrum(client):
    lib = (ROOT / "library" / "silicon_fuzz.cir").read_text(encoding="utf-8")
    did = client.mod.bench().check(netlist=lib)["design"]
    r = client.post("/api/plugins/hotwire/audition",
                    json={"design": did, "source": "guitar_riff", "seconds": 0.5, "guitar_volume": 0.25})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["guitar"]["volume"] == pytest.approx(0.25)
    assert a["spectrum"] and a["spectrum"].endswith(".svg")
    takes = client.get(f"/api/plugins/hotwire/designs/{did}").json()["current"]["auditions"]
    assert takes[0]["guitar"]["volume"] == pytest.approx(0.25) and takes[0]["spectrum"]
    for bad in (2, -0.1, "loud", True):
        r = client.post("/api/plugins/hotwire/audition", json={"design": did, "seconds": 0.5, "guitar_volume": bad})
        assert r.status_code == 400, bad
    assert client.post("/api/plugins/hotwire/audition", json={"design": did, "seconds": 0.5, "pots": {"RV9": 0.1}}).status_code == 400


def test_library_schematic_and_titles(client):
    r = client.get("/api/plugins/hotwire/library/rc_lowpass/schematic")
    assert r.status_code == 200, r.text
    assert r.json()["svg"].startswith("<svg")
    again = client.get("/api/plugins/hotwire/library/rc_lowpass/schematic").json()
    assert again["svg"] == r.json()["svg"]
    assert client.get("/api/plugins/hotwire/library/nope/schematic").status_code == 404
    assert client.get("/api/plugins/hotwire/library/..%2Fx/schematic").status_code in (400, 404)
    did = _seed(client)["design"]
    row = client.get("/api/plugins/hotwire/designs").json()["designs"][0]
    assert row["id"] == did and row["title"] == "Test booster"
