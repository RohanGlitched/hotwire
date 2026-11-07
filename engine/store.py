"""Design history on disk: every check saves a new version, so v1 → vN is browsable.

Layout under the plugin data directory::

    designs/<design_id>/meta.json          name, created, latest version
    designs/<design_id>/v<N>/netlist.cir   the netlist at that version
    designs/<design_id>/v<N>/check.json    verdict, specs, stress, op summary
    designs/<design_id>/v<N>/*.svg|*.wav   schematic, plots, audio
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

__all__ = ["data_dir", "Store"]


def data_dir() -> Path:
    """Hermes' per-plugin data dir when running inside Hermes, else a local folder."""
    env = os.environ.get("HOTWIRE_DATA")
    if env:
        p = Path(env)
    else:
        try:
            from plugins.plugin_storage import plugin_data_dir  # Hermes public API
            p = Path(plugin_data_dir("hotwire"))
        except Exception:
            home = os.environ.get("HERMES_HOME")
            p = Path(home) / "plugin-data" / "hotwire" if home else Path.home() / ".hotwire"
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40] or "circuit"


class Store:
    def __init__(self, root: Path | None = None):
        self.root = (root or data_dir()) / "designs"
        self.root.mkdir(parents=True, exist_ok=True)

    # -- ids ---------------------------------------------------------------
    def _meta_path(self, did: str) -> Path:
        return self.root / did / "meta.json"

    def exists(self, did: str) -> bool:
        return bool(did) and re.fullmatch(r"[a-z0-9-]{1,48}", did) is not None and self._meta_path(did).exists()

    def meta(self, did: str) -> dict:
        return json.loads(self._meta_path(did).read_text(encoding="utf-8"))

    def new_id(self, name: str) -> str:
        base = slugify(name)
        did, k = base, 2
        while (self.root / did).exists():
            did = f"{base}-{k}"
            k += 1
        return did

    # -- versions ------------------------------------------------------------
    def latest_netlist(self, did: str) -> str:
        m = self.meta(did)
        return (self.root / did / f"v{m['latest']}" / "netlist.cir").read_text(encoding="utf-8")

    def find_by_netlist(self, netlist: str):
        """Design whose latest version has exactly this netlist (so re-checks don't fork)."""
        norm = netlist.strip()
        for d in sorted(self.root.iterdir(), key=lambda p: -p.stat().st_mtime)[:30]:
            mp = d / "meta.json"
            if not mp.exists():
                continue
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                if (d / f"v{m['latest']}" / "netlist.cir").read_text(encoding="utf-8").strip() == norm:
                    return d.name, m["latest"]
            except Exception:
                continue
        return None

    def save_version(self, did: str | None, name: str, netlist: str, note: str = "") -> tuple:
        """Create design (if needed) and a new version folder; returns (id, version, path)."""
        if not did or not self.exists(did):
            did = self.new_id(name)
            meta = {"id": did, "name": name, "created": time.time(), "latest": 0, "versions": []}
        else:
            meta = self.meta(did)
        v = meta["latest"] + 1
        vdir = self.root / did / f"v{v}"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "netlist.cir").write_text(netlist, encoding="utf-8")
        meta["latest"] = v
        meta["updated"] = time.time()
        meta["versions"].append({"v": v, "note": note, "time": time.time()})
        self._meta_path(did).parent.mkdir(parents=True, exist_ok=True)
        self._meta_path(did).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return did, v, vdir

    def version_dir(self, did: str, v: int | None = None) -> Path:
        m = self.meta(did)
        return self.root / did / f"v{v or m['latest']}"

    def update_version(self, did: str, v: int, **fields):
        meta = self.meta(did)
        for row in meta["versions"]:
            if row["v"] == v:
                row.update(fields)
        self._meta_path(did).write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def list(self, limit: int = 50) -> list:
        out = []
        for d in self.root.iterdir():
            mp = d / "meta.json"
            if mp.exists():
                try:
                    out.append(json.loads(mp.read_text(encoding="utf-8")))
                except Exception:
                    pass
        out.sort(key=lambda m: -m.get("updated", m.get("created", 0)))
        return out[:limit]

    def contains(self, path: Path) -> bool:
        try:
            return Path(path).resolve().is_relative_to(self.root.parent.resolve())
        except Exception:
            return False
