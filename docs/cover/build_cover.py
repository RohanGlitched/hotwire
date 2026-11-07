"""Builds docs/cover/cover.html from real artifacts; screenshot it to docs/cover.png.

    python docs/cover/build_cover.py
    npx playwright screenshot --viewport-size=1600,800 --wait-for-timeout=1500 file:///<abs>/docs/cover/cover.html docs/cover.png
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import parse, process_audio  # noqa: E402
from engine.audio import source  # noqa: E402
from engine.guitar import with_guitar  # noqa: E402
from engine.schematic import svg as schematic_svg  # noqa: E402

c = parse((ROOT / "library" / "silicon_fuzz.cir").read_text(encoding="utf-8"))
sr = 48000
x = source("guitar_riff", sr, 2.0) * 0.15
y = process_audio(with_guitar(c, 1.0), x, sr)
win = int(0.016 * sr)
e = np.convolve(x ** 2, np.ones(win), "valid")
i = int(np.argmax(e[int(0.3 * sr):])) + int(0.3 * sr)
dx, dy = x[i:i + win], y[i:i + win]


def path(sig, w, h, scale):
    return "M" + " L".join(f"{k / (len(sig) - 1) * w:.1f},{h / 2 - v * scale:.1f}" for k, v in enumerate(sig))


W, H = 760, 250
dp = path(dx / np.max(np.abs(dx)), W, H, H * 0.18)
wy = dy - np.mean(dy)
wp = path(wy / np.max(np.abs(wy)), W, H, H * 0.38)
sch = schematic_svg(c, title=False)
sch = re.sub(r"<\?xml[^>]*>", "", sch)
sch = re.sub(r'(<svg[^>]*?)\s(width|height)="[^"]*"', r"\1", sch, count=1)
sch = re.sub(r'(<svg[^>]*?)\s(width|height)="[^"]*"', r"\1", sch, count=1)

html = f'''<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--mask:#0e3b2c;--silk:#f4f1e8;--copper:#d39a62}}
*{{box-sizing:border-box;margin:0}}
body{{width:1600px;height:800px;background:var(--mask);font-family:Archivo,system-ui;color:var(--silk);overflow:hidden;position:relative}}
.vias{{position:absolute;inset:0;background-image:radial-gradient(circle at 1px 1px, rgba(244,241,232,.07) 1.2px, transparent 1.6px);background-size:28px 28px}}
.left{{position:absolute;left:84px;top:96px;width:560px}}
.mark{{font-weight:800;font-size:118px;letter-spacing:-3px;line-height:1}}
.tag{{margin-top:26px;font-size:30px;line-height:1.35;font-weight:500;color:rgba(244,241,232,.92)}}
.steps{{margin-top:44px;font-family:'JetBrains Mono';font-size:19px;color:rgba(244,241,232,.75);line-height:1.9}}
.steps b{{color:var(--copper);font-weight:600}}
.foot{{position:absolute;left:84px;bottom:64px;width:560px;line-height:1.6;font-family:'JetBrains Mono';font-size:18px;color:rgba(244,241,232,.6)}}
.paper{{position:absolute;right:70px;top:70px;width:820px;height:400px;background:var(--silk);border-radius:6px;box-shadow:0 18px 50px rgba(0,0,0,.35);padding:22px 26px;color:#1d1d1f;display:flex}}
.paper svg{{width:100%;height:100%;flex:1}}
.scope{{position:absolute;right:70px;bottom:62px;width:820px;height:250px;background:#07241a;border:1px solid rgba(211,154,98,.35);border-radius:6px;overflow:hidden}}
.scope .lbl{{position:absolute;left:16px;top:12px;font-family:'JetBrains Mono';font-size:15px;color:rgba(244,241,232,.65)}}
.scope .lbl i{{font-style:normal;color:var(--copper)}}
</style></head><body><div class="vias"></div>
<div class="left"><div class="mark">Hotwire</div>
<div class="tag">Hermes designs analog circuits, proves them in simulation, and lets you hear them before you solder anything.</div>
<div class="steps"><b>01</b> describe it &nbsp; <b>02</b> prove it<br><b>03</b> hear it &nbsp;&nbsp;&nbsp;&nbsp; <b>04</b> build it</div></div>
<div class="foot">A Hermes Agent plugin with its own circuit simulator, checked against ngspice</div>
<div class="paper">{sch}</div>
<div class="scope"><div class="lbl">16 ms of guitar: the pickup signal, and <i>the same notes through the fuzz</i></div>
<svg viewBox="0 0 {W} {H}" width="820" height="250" style="position:absolute;left:30px;top:0" preserveAspectRatio="none">
<path d="{dp}" fill="none" stroke="rgba(244,241,232,.55)" stroke-width="2"/>
<path d="{wp}" fill="none" stroke="#d39a62" stroke-width="3.2"/></svg></div>
</body></html>'''
(ROOT / "docs" / "cover" / "cover.html").write_text(html, encoding="utf-8")
print("wrote docs/cover/cover.html")
