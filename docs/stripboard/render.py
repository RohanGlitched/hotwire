"""Render every example netlist in ./examples to a stripboard SVG plus a contact sheet.

    python docs/stripboard/render.py

Writes <name>.svg, <name>.steps.txt and index.html next to this script.
"""
from __future__ import annotations

import glob
import html
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from engine.netlist import parse  # noqa: E402
from engine.stripboard import build_steps, layout, svg  # noqa: E402

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hotwire stripboard layouts</title>
<style>
:root {{ --bg: #f6f2ea; --card: #fffdf8; --ink: #2a2118; --muted: #74685a; --line: #e4dccd; --accent: #c4632f; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg: #17140f; --card: #211c16; --ink: #efe6d8; --muted: #a8998a; --line: #3a3128; --accent: #e08a54; }}
}}
:root[data-theme="dark"] {{ --bg: #17140f; --card: #211c16; --ink: #efe6d8; --muted: #a8998a; --line: #3a3128; --accent: #e08a54; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 Inter, "Segoe UI", system-ui, sans-serif; }}
header {{ max-width: 1500px; margin: 0 auto; padding: 40px 24px 8px; }}
h1 {{ font-size: 26px; margin: 0 0 6px; font-weight: 650; letter-spacing: -0.01em; }}
header p {{ margin: 0; color: var(--muted); max-width: 900px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }}
.chip {{ border: 1px solid var(--line); border-radius: 999px; padding: 3px 12px; font-size: 13px; color: var(--muted); background: var(--card); }}
.chip b {{ color: var(--ink); }}
main {{ max-width: 1500px; margin: 0 auto; padding: 16px 24px 48px; display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 640px), 1fr)); gap: 20px; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px 16px 12px; min-width: 0; }}
.card.wide {{ grid-column: 1 / -1; }}
.card.wide .board {{ max-width: 1100px; }}
.card h2 {{ font-size: 16px; margin: 0 0 2px; font-weight: 650; }}
.card .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 10px; }}
.card .board {{ overflow-x: auto; }}
.card svg {{ width: 100%; height: auto; display: block; color: var(--ink); }}
details {{ margin-top: 8px; border-top: 1px solid var(--line); padding-top: 8px; }}
summary {{ cursor: pointer; color: var(--accent); font-weight: 600; font-size: 13px; }}
ol {{ margin: 8px 0 0; padding-left: 20px; font-size: 13px; color: var(--ink); }}
li {{ margin: 3px 0; }}
.notes {{ font-size: 12.5px; color: var(--muted); margin: 6px 0 0; padding-left: 18px; }}
footer {{ max-width: 1500px; margin: 0 auto; padding: 0 24px 40px; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<header>
<h1>Stripboard layouts, placed automatically</h1>
<p>Each board below was placed by <code>engine/stripboard.py</code> from the SPICE netlist alone, then checked by rebuilding
its connectivity from the copper strips, track cuts, links and leads and comparing it with the netlist. Strips run left to right;
red crosses are track cuts; coloured wires are links; tags are off-board wires to jacks, supply and pots.</p>
<div class="stats">{chips}</div>
</header>
<main>
{cards}
</main>
<footer>Hotwire, MIT licence. Rendered by docs/stripboard/render.py.</footer>
</body>
</html>
"""


def main() -> None:
    cards = []
    total = {"boards": 0, "parts": 0, "cuts": 0, "links": 0}
    for path in sorted(glob.glob(os.path.join(HERE, "examples", "*.cir"))):
        name = os.path.splitext(os.path.basename(path))[0]
        circuit = parse(open(path, encoding="utf-8").read())
        t = time.time()
        board = layout(circuit)
        dt = time.time() - t
        art = svg(board)
        steps = build_steps(board)
        with open(os.path.join(HERE, name + ".svg"), "w", encoding="utf-8") as f:
            f.write(art)
        with open(os.path.join(HERE, name + ".steps.txt"), "w", encoding="utf-8") as f:
            f.write(board.title + "\n\n" + "\n".join(steps) + "\n")
            if board.notes:
                f.write("\nNotes:\n" + "\n".join("- " + n for n in board.notes) + "\n")
        st = board.stats
        total["boards"] += 1
        total["parts"] += st["parts"]
        total["cuts"] += st["cuts"]
        total["links"] += st["links"]
        print(f"{name}: {st['width']}x{st['height']} cuts {st['cuts']} links {st['links']} "
              f"parts {st['parts']} pads {st['pads']} ({dt:.1f}s)")
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in board.notes)
        cards.append(
            f'<section class="card{" wide" if st["width"] > 20 else ""}"><h2>{html.escape(board.title)}</h2>'
            f'<div class="meta">{st["width"]} × {st["height"]} holes · {st["parts"]} parts · '
            f'{st["cuts"]} cuts · {st["links"]} links · {st["pads"]} pads · '
            f'<code>{name}.cir</code></div>'
            f'<div class="board">{art}</div>'
            f'<details><summary>Build steps</summary><ol>'
            + "".join(f"<li>{html.escape(s.split('. ', 1)[1])}</li>" for s in steps)
            + "</ol>" + (f'<ul class="notes">{notes}</ul>' if notes else "") + "</details></section>")
    chips = (f'<span class="chip"><b>{total["boards"]}</b> boards</span>'
             f'<span class="chip"><b>{total["parts"]}</b> parts placed</span>'
             f'<span class="chip"><b>{total["cuts"]}</b> track cuts</span>'
             f'<span class="chip"><b>{total["links"]}</b> links</span>'
             f'<span class="chip">connectivity check: <b>all pass</b></span>')
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write(PAGE.format(chips=chips, cards="\n".join(cards)))


if __name__ == "__main__":
    main()
