"""Render the schematic gallery: every library circuit plus docs/gallery/extra/*.cir.

    python docs/gallery/build.py            # writes docs/gallery/*.svg and index.html
    python docs/gallery/build.py --png      # also screenshots index.html with headless Chrome

Prints the layout quality (overlaps, crossings, unrouted pins) for each circuit.
"""
from __future__ import annotations

import html
import re
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from engine.netlist import parse  # noqa: E402
from engine.schematic import layout, svg  # noqa: E402


def sources():
    files = sorted((ROOT / "library").glob("*.cir")) + sorted((HERE / "extra").glob("*.cir"))
    return files


def build(verbose=True):
    cards = []
    report = []
    for f in sources():
        c = parse(f.read_text(encoding="utf-8"))
        t0 = time.time()
        L = layout(c)
        dt = time.time() - t0
        text = svg(c)
        (HERE / f"{f.stem}.svg").write_text(text, encoding="utf-8")
        ov = L.overlaps()
        report.append((f.stem, len(ov), L.crossings, len(L.failed), dt, L.strategy, ov))
        cards.append((f.stem, c.title, text))
    if verbose:
        for stem, nov, cr, fl, dt, strat, ov in report:
            print(f"{stem:28s} overlaps={nov} crossings={cr} unrouted={fl} {dt:5.2f}s [{strat}]")
            for o in ov[:6]:
                print("     ", o)
    page = ["<!doctype html><html lang=en><head><meta charset=utf-8>",
            "<meta name=viewport content='width=device-width,initial-scale=1'>",
            "<title>Hotwire schematic gallery</title><style>",
            ":root{--bg:#f7f6f3;--card:#fff;--line:#e4e2dc;--muted:#6b6a66;--hotwire-ink:#1d1d1f;--hotwire-accent:#c2410c}",
            "body.dark{--bg:#141416;--card:#1c1c1f;--line:#2c2c31;--muted:#9a9aa2;--hotwire-ink:#e8e8ed;--hotwire-accent:#fb923c}",
            "body{margin:0;background:var(--bg);color:var(--hotwire-ink);font:14px/1.4 ui-sans-serif,system-ui,sans-serif}",
            "header{padding:28px 32px 8px;display:flex;align-items:baseline;gap:16px}",
            "h1{font-size:20px;margin:0}header p{margin:0;color:var(--muted)}",
            "button{margin-left:auto;font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:inherit}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(600px,1fr));gap:20px;padding:20px 32px 40px}",
            ".card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;overflow:hidden}",
            ".card svg{max-width:100%;height:auto;display:block;margin:0 auto}",
            ".card.wide{grid-column:1/-1}",
            ".card .meta{font:12px ui-monospace,monospace;color:var(--muted);margin-top:8px}",
            "</style></head><body>",
            "<header><h1>Hotwire schematic gallery</h1><p>auto-laid-out from SPICE netlists</p>",
            "<button onclick=\"document.body.classList.toggle('dark')\">light / dark</button></header>",
            "<main class=grid>"]
    for stem, title, text in cards:
        body = text.split("?>", 1)[-1]
        w = int(re.search(r'width="(\d+)"', body).group(1))
        cls = "card wide" if w > 640 else "card"
        page.append(f"<figure class='{cls}' id='{html.escape(stem)}'>{body}"
                    f"<figcaption class=meta>{html.escape(stem)}.cir</figcaption></figure>")
    plots = sorted((HERE / "plots").glob("*.svg"))
    if plots:
        page.append("</main><h2 style='font-size:16px;margin:8px 32px 0'>Plots</h2><main class=grid>")
        for f in plots:
            page.append(f"<figure class=card>{f.read_text(encoding='utf-8')}"
                        f"<figcaption class=meta>plots/{html.escape(f.name)}</figcaption></figure>")
    page.append("</main><script>if(location.hash==='#dark')document.body.classList.add('dark')</script></body></html>")
    (HERE / "index.html").write_text("\n".join(page), encoding="utf-8")
    return report


def chrome():
    for c in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium")):
        if c and os.path.exists(c):
            return c
    return None


def screenshot(out, width=1400, height=6000, dark=False):
    exe = chrome()
    if not exe:
        print("no Chrome found; skipping screenshot")
        return
    url = (HERE / "index.html").as_uri() + ("#dark" if dark else "")
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={width},{height}",
                    f"--screenshot={out}", url], check=False, capture_output=True, timeout=120)
    print("screenshot:", out)


if __name__ == "__main__":
    build()
    if "--png" in sys.argv:
        screenshot(str(HERE / "contact-sheet.png"), height=int(sys.argv[sys.argv.index("--png") + 1])
                   if len(sys.argv) > sys.argv.index("--png") + 1 else 9000)
        if "--dark" in sys.argv:
            screenshot(str(HERE / "contact-sheet-dark.png"), height=3000, dark=True)
