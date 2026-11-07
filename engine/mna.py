"""Modified nodal analysis: assembly, DC Newton, AC solve and the transient core.

Unknowns are ``[circuit nodes, internal nodes, branch currents]``.  The
circuit is split into a linear part (``G`` conductances, ``Cm`` capacitances
and inductances, ``Bsrc`` independent-source incidence) and nonlinear device
groups that act through voltage ports ``v = Nv @ x`` and inject currents
``-Ni.T @ i(v)``.  The linear system reads

    G x + Cm dx/dt + Ni.T i(Nv x) = Bsrc u(t)

Inductors and voltage-type elements carry branch currents, so ``Cm`` holds
``-L`` on an inductor's branch diagonal and the same matrices serve DC, AC
and transient analysis.

Transient analysis uses the trapezoidal rule.  Because the step is fixed,
the linear part is folded into a constant matrix once per step size and the
Newton iteration runs only over the nonlinear port voltages (the "nodal
DK" reduction used in circuit-modelling audio work).  The linear part is
linearised at the DC operating point, so a small-signal step converges in
one or two iterations and a 2-transistor fuzz runs much faster than real
time would need on a full-matrix solve.
"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field

import numpy as np

from . import devices as dv
from .netlist import Circuit, Element, NetlistError

__all__ = ["SimulationError", "Network", "Source", "Transient", "GSHUNT"]

GSHUNT = 1e-12          # node-to-ground conductance so floating (cap-only) nodes stay solvable
SWITCH_RON = 0.1
SWITCH_ROFF = 1e9
IC_G = 1e3              # conductance (S) holding a capacitor at its ic= during the initial point


class SimulationError(RuntimeError):
    """An analysis could not produce a trustworthy result; the message says why."""


# --------------------------------------------------------------------------- sources

@dataclass
class Source:
    """An independent source column of ``Bsrc``.  ``kind``: 'V', 'I' or 'IQ' (op-amp supply draw)."""

    name: str
    kind: str
    dc: float
    ac: complex = 0.0
    sin: tuple | None = None
    pulse: tuple | None = None
    wave: np.ndarray | None = None      # sampled override (input drive), added to dc
    wave_dt: float = 0.0

    def at(self, t: np.ndarray, tstep: float = 0.0, tstop: float = 0.0) -> np.ndarray:
        """Source value at times ``t`` (array)."""
        t = np.asarray(t, dtype=float)
        if self.wave is not None:
            idx = t / self.wave_dt
            w = np.interp(idx, np.arange(len(self.wave)), self.wave,
                          left=self.wave[0] if len(self.wave) else 0.0,
                          right=self.wave[-1] if len(self.wave) else 0.0)
            return self.dc + w
        if self.sin is not None:
            vo, va, f, td, theta = (list(self.sin) + [0.0] * 5)[:5]
            phase = self.sin[5] if len(self.sin) > 5 else 0.0
            tt = t - td
            val = vo + va * np.exp(-np.maximum(tt, 0.0) * theta) * np.sin(
                2 * math.pi * f * np.maximum(tt, 0.0) + math.radians(phase))
            return np.where(tt >= 0, val, vo + va * math.sin(math.radians(phase)))
        if self.pulse is not None:
            v1, v2, td, tr, tf, pw, per = (list(self.pulse) + [0.0] * 7)[:7]
            tr = tr if tr > 0 else (tstep or 1e-9)
            tf = tf if tf > 0 else (tstep or 1e-9)
            pw = pw if pw > 0 else (tstop or math.inf)
            per = per if per > 0 else math.inf
            tt = t - td
            if math.isfinite(per):
                tt = np.where(tt >= 0, np.mod(tt, per), tt)
            out = np.select(
                [tt < 0, tt < tr, tt < tr + pw, tt < tr + pw + tf],
                [np.full_like(tt, v1), v1 + (v2 - v1) * tt / tr, np.full_like(tt, v2),
                 v2 + (v1 - v2) * (tt - tr - pw) / tf],
                default=v1)
            return out
        return np.full(t.shape, self.dc)

    def at0(self) -> float:
        """Value at t = 0 (the initial operating point of a transient)."""
        return float(self.at(np.zeros(1))[0])


# --------------------------------------------------------------------------- network

@dataclass
class _Rec:
    """What the reporting code needs to know about one element."""
    el: Element
    kind: str
    n: tuple = ()            # global indices of the external nodes (-1 = ground)
    br: int = -1             # branch row
    g: float = 0.0           # conductance (R, S) or gm (G)
    group: object = None
    slot: int = 0            # index within its device group
    extra: dict = field(default_factory=dict)


class Network:
    """A circuit compiled to dense MNA matrices plus nonlinear device groups."""

    def __init__(self, c: Circuit, input_name: str | None = None):
        self.c = c
        self.node_names: list[str] = [n for n in c.nodes() if n != "0"]
        idx = {n: i for i, n in enumerate(self.node_names)}
        idx["0"] = -1
        self.index = idx
        nn = len(self.node_names)

        # ---- pass 1: resolve models and count internal nodes ----------------
        plans = []
        n_int = 0
        n_br = 0
        for e in c.elements:
            mp = None
            if e.kind in "DQJXM":
                mp, mtype = self._model(e)
            else:
                mtype = None
            need = 0
            if e.kind == "D":
                need = int(mp["RS"] > 0)
            elif e.kind == "Q":
                need = int(mp["RB"] > 0) + int(mp["RC"] > 0) + int(mp["RE"] > 0)
            elif e.kind == "J":
                need = int(mp["RD"] > 0) + int(mp["RS"] > 0)
            elif e.kind == "M":
                try:
                    geo = dv.mos_geometry(mp)
                except ValueError as exc:
                    raise NetlistError(e.line, f"{e.name}: {exc}") from None
                need = int(geo["rd"] > 0) + int(geo["rs"] > 0)
            elif e.kind == "X":
                need = 1
            if e.kind in "VLE":
                n_br += 1
            n_int += need
            plans.append((e, mp, mtype))
        self.n_nodes = nn + n_int            # node rows (gshunt applies to these)
        n = self.n_nodes + n_br
        self.n = n
        self.internal_names: list[str] = []

        G = np.zeros((n, n))
        reactive: list[tuple[int, int, float]] = []   # (plus, minus, value) ; minus = -2 marks a branch
        sources: list[Source] = []
        bcols: list[list[tuple[int, float]]] = []
        self.recs: dict[str, _Rec] = {}
        next_int = nn
        next_br = self.n_nodes
        dio, bjt, jft, opa, mos = [], [], [], [], []
        ic_caps: list[tuple[int, int, float]] = []

        def node(name: str) -> int:
            return idx[name]

        def new_internal(label: str) -> int:
            nonlocal next_int
            self.internal_names.append(label)
            next_int += 1
            return next_int - 1

        def stamp_g(a: int, b: int, g: float) -> None:
            if a >= 0:
                G[a, a] += g
            if b >= 0:
                G[b, b] += g
            if a >= 0 and b >= 0:
                G[a, b] -= g
                G[b, a] -= g

        def stamp_branch(a: int, b: int, br: int) -> None:
            if a >= 0:
                G[a, br] += 1.0
                G[br, a] += 1.0
            if b >= 0:
                G[b, br] -= 1.0
                G[br, b] -= 1.0

        def add_source(name: str, kind: str, e: Element | None, plus: int, minus: int, br: int = -1):
            p = e.params if e is not None else {}
            acm = float(p.get("ac", 0.0) or 0.0)
            acp = float(p.get("ac_phase", 0.0) or 0.0)
            s = Source(name, kind, float(e.value or 0.0) if e is not None else 0.0,
                       acm * complex(math.cos(math.radians(acp)), math.sin(math.radians(acp))),
                       p.get("sin"), p.get("pulse"))
            col = []
            if kind == "V":
                col.append((br, 1.0))
            else:      # current flows from plus through the source to minus
                if plus >= 0:
                    col.append((plus, -1.0))
                if minus >= 0:
                    col.append((minus, 1.0))
            sources.append(s)
            bcols.append(col)
            return s

        # ---- pass 2: stamp --------------------------------------------------
        for e, mp, mtype in plans:
            k = e.kind
            ns = tuple(node(x) for x in e.nodes)
            rec = _Rec(e, k, ns)
            if k == "R":
                if not e.value:
                    raise NetlistError(e.line, f"{e.name} has zero resistance")
                rec.g = 1.0 / float(e.value)
                stamp_g(ns[0], ns[1], rec.g)
            elif k == "S":
                on = bool(e.params.get("on", False))
                r = float(e.params.get("ron" if on else "roff", SWITCH_RON if on else SWITCH_ROFF))
                rec.g = 1.0 / r
                stamp_g(ns[0], ns[1], rec.g)
            elif k == "C":
                reactive.append((ns[0], ns[1], float(e.value)))
                if e.params.get("ic") is not None:
                    ic_caps.append((ns[0], ns[1], float(e.params["ic"])))
            elif k == "L":
                br = next_br
                next_br += 1
                rec.br = br
                stamp_branch(ns[0], ns[1], br)
                reactive.append((br, -2, -float(e.value)))
            elif k == "V":
                br = next_br
                next_br += 1
                rec.br = br
                stamp_branch(ns[0], ns[1], br)
                rec.extra["src"] = add_source(e.name, "V", e, ns[0], ns[1], br)
            elif k == "I":
                rec.extra["src"] = add_source(e.name, "I", e, ns[0], ns[1])
            elif k == "E":
                br = next_br
                next_br += 1
                rec.br = br
                stamp_branch(ns[0], ns[1], br)
                gain = float(e.value)
                if ns[2] >= 0:
                    G[br, ns[2]] -= gain
                if ns[3] >= 0:
                    G[br, ns[3]] += gain
            elif k == "G":
                gm = float(e.value)
                rec.g = gm
                for row, sr in ((ns[0], 1.0), (ns[1], -1.0)):
                    if row < 0:
                        continue
                    if ns[2] >= 0:
                        G[row, ns[2]] += sr * gm
                    if ns[3] >= 0:
                        G[row, ns[3]] -= sr * gm
            elif k == "D":
                a, kk = ns
                ai = a
                if mp["RS"] > 0:
                    ai = new_internal(f"{e.name}#a")
                    stamp_g(a, ai, 1.0 / mp["RS"])
                rec.extra["ports"] = (ai, kk)
                dio.append((e, mp, rec))
            elif k == "Q":
                c_, b_, e_ = ns
                ci, bi, ei = c_, b_, e_
                if mp["RC"] > 0:
                    ci = new_internal(f"{e.name}#c")
                    stamp_g(c_, ci, 1.0 / mp["RC"])
                if mp["RB"] > 0:
                    bi = new_internal(f"{e.name}#b")
                    stamp_g(b_, bi, 1.0 / mp["RB"])
                if mp["RE"] > 0:
                    ei = new_internal(f"{e.name}#e")
                    stamp_g(e_, ei, 1.0 / mp["RE"])
                rec.extra["ports"] = (ci, bi, ei)
                bjt.append((e, mp, rec, 1.0 if mtype == "NPN" else -1.0))
            elif k == "J":
                d_, g_, s_ = ns
                di, si = d_, s_
                if mp["RD"] > 0:
                    di = new_internal(f"{e.name}#d")
                    stamp_g(d_, di, 1.0 / mp["RD"])
                if mp["RS"] > 0:
                    si = new_internal(f"{e.name}#s")
                    stamp_g(s_, si, 1.0 / mp["RS"])
                rec.extra["ports"] = (di, g_, si)
                jft.append((e, mp, rec, 1.0 if mtype == "NJF" else -1.0))
            elif k == "M":
                d_, g_, s_, b_ = ns
                geo = dv.mos_geometry(mp)
                di, si = d_, s_
                if geo["rd"] > 0:
                    di = new_internal(f"{e.name}#d")
                    stamp_g(d_, di, 1.0 / geo["rd"])
                if geo["rs"] > 0:
                    si = new_internal(f"{e.name}#s")
                    stamp_g(s_, si, 1.0 / geo["rs"])
                # constant gate overlap capacitances are linear: stamp them with the C's
                for a_, b2, cval in ((g_, si, geo["cgs"]), (g_, di, geo["cgd"]), (g_, b_, geo["cgb"])):
                    if cval > 0 and a_ != b2:
                        reactive.append((a_, b2, cval))
                tied = b_ == si
                rec.extra.update(ports=(di, g_, si, b_), tied=tied)
                mos.append((e, mp, rec, 1.0 if mtype == "NMOS" else -1.0, tied))
            elif k == "X":
                inp, inn, vcc, vee, out = ns
                p = new_internal(f"{e.name}#p")
                if mp["RIN"] > 0 and math.isfinite(mp["RIN"]):
                    stamp_g(inp, inn, 1.0 / mp["RIN"])
                a0 = max(mp["A0"], 1.0)
                gm = a0 / dv.OPAMP_RP
                if inp >= 0:
                    G[p, inp] -= gm
                if inn >= 0:
                    G[p, inn] += gm
                stamp_g(p, -1, 1.0 / dv.OPAMP_RP)
                fp = max(mp["GBW"], 1.0) / a0
                reactive.append((p, -1, 1.0 / (2 * math.pi * fp * dv.OPAMP_RP)))
                gout = 1.0 / max(mp["ROUT"], 1e-3)
                stamp_g(out, -1, gout)
                rec.extra.update(ports=(p, vcc, vee, out), gout=gout, iq=mp["IQ"],
                                 vsh=mp["VSAT_HI"], vsl=mp["VSAT_LO"])
                if mp["IQ"] != 0:
                    s = Source(f"{e.name}#IQ", "IQ", float(mp["IQ"]))
                    col = []
                    if vcc >= 0:
                        col.append((vcc, -1.0))
                    if vee >= 0:
                        col.append((vee, 1.0))
                    sources.append(s)
                    bcols.append(col)
                opa.append((e, mp, rec))
            self.recs[e.name] = rec

        self.G = G
        # capacitor ic= values: during the initial transient operating point each
        # such cap is held at its ic by a stiff Norton source (like SPICE .ic),
        # then released at t = 0+ (this is what lets an oscillator start).
        self.force_ic = False
        self.ic_G = np.zeros((n, n))
        self.ic_b = np.zeros(n)
        for a, b, val in ic_caps:
            for p, q, sgn in ((a, b, 1.0), (b, a, -1.0)):
                if p >= 0:
                    self.ic_G[p, p] += IC_G
                    self.ic_b[p] += sgn * IC_G * val
                    if q >= 0:
                        self.ic_G[p, q] -= IC_G
        self.has_ic = bool(ic_caps)
        self.sources = sources
        self.src_index = {s.name: j for j, s in enumerate(sources)}
        B = np.zeros((n, len(sources)))
        for j, col in enumerate(bcols):
            for row, val in col:
                B[row, j] += val
        self.Bsrc = B

        # reactive elements: Cm = S.T diag(cval) S
        nr = len(reactive)
        S = np.zeros((nr, n))
        cval = np.zeros(nr)
        for r, (a, b, val) in enumerate(reactive):
            if b == -2:          # inductor branch
                S[r, a] = 1.0
            else:
                if a >= 0:
                    S[r, a] = 1.0
                if b >= 0:
                    S[r, b] = -1.0
            cval[r] = val
        self.S = S
        self.cval = cval
        self.Cm = S.T @ (cval[:, None] * S)

        # ---- device groups and ports ----------------------------------------
        self.groups: list[dv.DeviceGroup] = []
        vports: list[tuple[int, int]] = []
        iports: list[tuple[int, int]] = []
        pattern: list[tuple[int, int]] = []

        def add_group(grp, vp, ip, pat, recs):
            grp.vs = len(vports)
            grp.ve = grp.vs + len(vp)
            grp.is_ = len(iports)
            grp.ie = grp.is_ + len(ip)
            grp.vports, grp.iports, grp.pattern = vp, ip, pat
            for (a, b) in pat:
                pattern.append((grp.is_ + a, grp.vs + b))
            vports.extend(vp)
            iports.extend(ip)
            for j, r in enumerate(recs):
                r.group = grp
                r.slot = j
            self.groups.append(grp)

        def two_port_pattern(k: int) -> list:
            return ([(j, j) for j in range(k)] + [(j, k + j) for j in range(k)]
                    + [(k + j, j) for j in range(k)] + [(k + j, k + j) for j in range(k)])

        if dio:
            grp = dv.DiodeGroup([e.name for e, _, _ in dio], [mp for _, mp, _ in dio])
            ports = [r.extra["ports"] for _, _, r in dio]
            add_group(grp, ports, ports, [(j, j) for j in range(len(dio))], [r for _, _, r in dio])
        if bjt:
            grp = dv.BJTGroup([b[0].name for b in bjt], [b[1] for b in bjt], [b[3] for b in bjt])
            pr = [b[2].extra["ports"] for b in bjt]
            ports = [(bi, ei) for (ci, bi, ei) in pr] + [(bi, ci) for (ci, bi, ei) in pr]
            add_group(grp, ports, ports, two_port_pattern(len(bjt)), [b[2] for b in bjt])
        if jft:
            grp = dv.JFETGroup([b[0].name for b in jft], [b[1] for b in jft], [b[3] for b in jft])
            pr = [b[2].extra["ports"] for b in jft]
            ports = [(g_, si) for (di, g_, si) in pr] + [(g_, di) for (di, g_, si) in pr]
            add_group(grp, ports, ports, two_port_pattern(len(jft)), [b[2] for b in jft])
        for tied in (True, False):
            sel = [m for m in mos if m[4] == tied]
            if not sel:
                continue
            grp = dv.MOSFETGroup([m[0].name for m in sel], [m[1] for m in sel], [m[3] for m in sel], tied)
            pr = [m[2].extra["ports"] for m in sel]
            vp = [(g_, si) for (di, g_, si, b_) in pr] + [(di, si) for (di, g_, si, b_) in pr]
            ip = [(di, si) for (di, g_, si, b_) in pr]
            if not tied:
                vp += [(b_, si) for (di, g_, si, b_) in pr]
                ip += [(b_, si) for (di, g_, si, b_) in pr] + [(b_, di) for (di, g_, si, b_) in pr]
            add_group(grp, vp, ip, grp.local_pattern(), [m[2] for m in sel])
        if opa:
            pr = [o[2].extra["ports"] for o in opa]
            kk = len(opa)
            fixed = {-1: None}            # node -> (source column, sign); ground is fixed at 0 V
            for name, r in self.recs.items():
                if r.kind != "V" or name == (input_name or c.input or "").upper():
                    continue
                if "sin" in r.el.params or "pulse" in r.el.params:
                    continue
                col = self.src_index[name]
                if r.n[1] == -1 and r.n[0] >= 0:
                    fixed.setdefault(r.n[0], (col, 1.0))
                elif r.n[0] == -1 and r.n[1] >= 0:
                    fixed.setdefault(r.n[1], (col, -1.0))
            all_fixed = all(vcc in fixed and vee in fixed for (_, vcc, vee, _) in pr)
            grp = dv.OpAmpGroup([o[0].name for o in opa], [o[1] for o in opa], fixed_rails=all_fixed)
            # output Norton currents, then the anti-windup clamp currents out of each pole node
            ip = [(-1, out) for (_, _, _, out) in pr] + [(p, -1) for (p, _, _, _) in pr]
            if all_fixed:
                self._rail_map = [(fixed[vcc], fixed[vee]) for (_, vcc, vee, _) in pr]
                vp = [(p, -1) for (p, _, _, _) in pr]
            else:
                vp = ([(p, -1) for (p, _, _, _) in pr] + [(vcc, -1) for (_, vcc, _, _) in pr]
                      + [(vee, -1) for (_, _, vee, _) in pr])
            add_group(grp, vp, ip, grp.local_pattern(), [o[2] for o in opa])
            self._opamp_group = grp

        self.mv = len(vports)
        self.mi = len(iports)
        Nv = np.zeros((self.mv, n))
        for r, (a, b) in enumerate(vports):
            if a >= 0:
                Nv[r, a] += 1.0
            if b >= 0:
                Nv[r, b] -= 1.0
        Ni = np.zeros((self.mi, n))
        for r, (a, b) in enumerate(iports):
            if a >= 0:
                Ni[r, a] += 1.0
            if b >= 0:
                Ni[r, b] -= 1.0
        self.Nv, self.Ni = Nv, Ni
        if not hasattr(self, "_rail_map"):
            self._rail_map = None
        pat = np.array(pattern, dtype=int).reshape(-1, 2)
        self.jflat = pat[:, 0] * self.mv + pat[:, 1] if len(pat) else np.zeros(0, dtype=int)
        self.has_charge = any(g.has_charge for g in self.groups)
        self.gdiag = np.zeros(n)
        self.gdiag[: self.n_nodes] = GSHUNT

        self.input_name = (input_name or c.input or None)
        if self.input_name:
            self.input_name = self.input_name.upper()
            if self.input_name not in self.src_index:
                raise SimulationError(f"input source '{self.input_name}' is not a V or I source in this circuit")

    # ------------------------------------------------------------------ models
    def _model(self, e: Element) -> tuple[dict, str]:
        m = self.c.models.get(e.model) if e.model else None
        if m is None and e.model:
            part = dv.part_library().get(e.model.upper())
            if part is not None:
                mtype, params = part["type"], part["params"]
            else:
                raise NetlistError(e.line, f"{e.name}: unknown model '{e.model}'. Add a .model line or use a built-in part")
        elif m is None:
            raise NetlistError(e.line, f"{e.name} needs a model name")
        else:
            mtype, params = m.type, m.params
        mtype = mtype.upper()
        want = {"D": ("D",), "Q": ("NPN", "PNP"), "J": ("NJF", "PJF"), "X": ("OPAMP",),
                "M": ("NMOS", "PMOS")}[e.kind]
        if mtype not in want:
            raise NetlistError(e.line, f"{e.name}: model '{e.model}' is a {mtype} model, which doesn't fit a {e.kind} element")
        mp = dv.model_params(mtype, params)
        if e.kind == "M":       # instance W= L= M= override the model
            for key, dst in (("W", "W"), ("L", "L"), ("M", "MULT")):
                if key in e.params:
                    mp[dst] = float(e.params[key])
        return mp, mtype

    def set_rails(self, u: np.ndarray) -> None:
        """Give fixed-rail op-amps their supply voltages for source vector ``u``."""
        if self._rail_map is None:
            return
        def val(spec):
            return 0.0 if spec is None else spec[1] * float(u[spec[0]])
        self._opamp_group.set_rails(np.array([val(a) for a, _ in self._rail_map]),
                                    np.array([val(b) for _, b in self._rail_map]))

    # ------------------------------------------------------------------ devices
    def eval(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All port currents and Jacobian values at port voltages ``v``."""
        if not self.groups:
            return np.zeros(0), np.zeros(0)
        if len(self.groups) == 1:
            return self.groups[0].eval(v)
        ii, jj = [], []
        for g in self.groups:
            i, jv = g.eval(v[g.vs:g.ve])
            ii.append(i)
            jj.append(jv)
        return np.concatenate(ii), np.concatenate(jj)

    def charges(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All port charges and their Jacobian values (zeros for charge-free groups)."""
        qs, cs = [], []
        for g in self.groups:
            vi = v[g.vs:g.ve]
            if g.has_charge:
                q, cv = g.charges(vi)
            else:
                q, cv = np.zeros(g.ie - g.is_), np.zeros(len(g.pattern))
            qs.append(q)
            cs.append(cv)
        if not qs:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(qs), np.concatenate(cs)

    def jac(self, jv: np.ndarray) -> np.ndarray:
        J = np.zeros((self.mi, self.mv))
        if len(jv):
            J.flat[self.jflat] = jv
        return J

    def limit(self, vnew: np.ndarray, vold: np.ndarray) -> np.ndarray:
        if len(self.groups) == 1:
            return self.groups[0].limit(vnew, vold)
        out = vnew.copy()
        for g in self.groups:
            out[g.vs:g.ve] = g.limit(vnew[g.vs:g.ve], vold[g.vs:g.ve])
        return out

    def initial_ports(self) -> np.ndarray:
        v = np.zeros(self.mv)
        for g in self.groups:
            v[g.vs:g.ve] = g.initial()
        return v

    # ------------------------------------------------------------------ sources
    def u_dc(self) -> np.ndarray:
        """Source vector for a DC operating point (DC values)."""
        return np.array([s.dc for s in self.sources], dtype=float)

    def u_at(self, t: np.ndarray, tstep: float = 0.0, tstop: float = 0.0) -> np.ndarray:
        """Source matrix (n_src x len(t)) for a transient."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        U = np.empty((len(self.sources), len(t)))
        for j, s in enumerate(self.sources):
            U[j] = s.at(t, tstep, tstop)
        return U

    # ------------------------------------------------------------------ DC Newton
    def newton(self, x0: np.ndarray, u: np.ndarray, gextra: float = 0.0, maxiter: int = 150,
               init_junctions: bool = False, reltol: float = 1e-6, vntol: float = 1e-9,
               abstol: float = 1e-14) -> tuple[np.ndarray, bool, int]:
        """Newton-Raphson on the full MNA system with SPICE junction limiting."""
        n = self.n
        b = self.Bsrc @ u
        self.set_rails(u)
        Gb = self.G + np.diag(self.gdiag + np.where(np.arange(n) < self.n_nodes, gextra, 0.0))
        if self.force_ic:
            Gb = Gb + self.ic_G
            b = b + self.ic_b
        tol_abs = np.where(np.arange(n) < self.n_nodes, vntol, abstol)
        x = x0.copy()
        mv = self.mv
        v_old = self.initial_ports() if init_junctions else self.Nv @ x
        NiT = self.Ni.T
        for it in range(1, maxiter + 1):
            limited = False
            if mv:
                if it == 1 and init_junctions:
                    v = v_old
                else:
                    v_raw = self.Nv @ x
                    v = self.limit(v_raw, v_old)
                    limited = bool(np.any(np.abs(v - v_raw) > 1e-12 * (1 + np.abs(v_raw))))
                i, jv = self.eval(v)
                J = self.jac(jv)
                M = Gb + NiT @ J @ self.Nv
                rhs = b - NiT @ (i - J @ v)
                v_old = v
            else:
                M, rhs = Gb, b
            try:
                xn = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                return x, False, it
            if not np.all(np.isfinite(xn)):
                return x, False, it
            dx = np.abs(xn - x)
            ok = bool(np.all(dx <= reltol * np.maximum(np.abs(xn), np.abs(x)) + tol_abs))
            x = xn
            if not mv:
                return x, True, it
            if ok and not limited and it > 1:
                return x, True, it
        return x, False, maxiter

    def solve_dc(self, u: np.ndarray, x0: np.ndarray | None = None, maxiter: int = 150
                 ) -> tuple[np.ndarray, bool, str, int]:
        """DC operating point: Newton, then gmin stepping, then source stepping."""
        n = self.n
        total = 0
        start = np.zeros(n) if x0 is None else x0
        x, ok, it = self.newton(start, u, maxiter=maxiter, init_junctions=x0 is None)
        total += it
        if ok:
            return x, True, "newton", total
        # gmin stepping: a large shunt makes the matrix diagonal-dominant, then relax it
        x = np.zeros(n)
        g = 1e-2
        good = None
        factor = 10.0
        first = True
        while g >= 1e-12:
            xn, ok, it = self.newton(x, u, gextra=g, maxiter=maxiter, init_junctions=first)
            total += it
            first = False
            if ok:
                x, good = xn, g
                g /= factor
            else:
                if good is None:
                    break
                factor = math.sqrt(factor)
                if factor < 1.05:
                    break
                g = good / factor
        if good is not None:
            xn, ok, it = self.newton(x, u, maxiter=maxiter)
            total += it
            if ok:
                return xn, True, "gmin", total
        # source stepping: ramp every source from zero
        x = np.zeros(n)
        alpha, step = 0.0, 0.1
        ok_any = True
        while alpha < 1.0:
            a_try = min(1.0, alpha + step)
            xn, ok, it = self.newton(x, u * a_try, maxiter=maxiter, init_junctions=alpha == 0.0)
            total += it
            if ok:
                x, alpha = xn, a_try
                step = min(step * 1.5, 0.5)
            else:
                step /= 4.0
                if step < 1e-5:
                    ok_any = False
                    break
        if ok_any and alpha >= 1.0:
            return x, True, "source", total
        return x, False, "failed", total

    # ------------------------------------------------------------------ AC
    def ac_solve(self, x_op: np.ndarray, freqs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Small-signal node solutions (len(freqs) x n) for the complex RHS ``rhs``."""
        Gs = self.G + np.diag(self.gdiag)
        Cs = self.Cm.copy()
        if self.mv:
            v0 = self.Nv @ x_op
            _, jv = self.eval(v0)
            Gs = Gs + self.Ni.T @ self.jac(jv) @ self.Nv
            if self.has_charge:
                _, cv = self.charges(v0)
                Cs = Cs + self.Ni.T @ self.jac(cv) @ self.Nv
        w = 2 * math.pi * np.asarray(freqs, dtype=float)
        out = np.empty((len(w), self.n), dtype=complex)
        chunk = max(1, int(4e6 // max(1, self.n * self.n)))
        for s in range(0, len(w), chunk):
            ws = w[s:s + chunk]
            Y = Gs[None, :, :] + 1j * ws[:, None, None] * Cs[None, :, :]
            out[s:s + chunk] = np.linalg.solve(Y, np.broadcast_to(rhs, (len(ws), self.n))[..., None])[..., 0]
        return out

    # ------------------------------------------------------------------ reporting
    def report(self, x: np.ndarray, u: np.ndarray) -> tuple[dict, dict, dict, dict]:
        """(node voltages, element currents, power, device info) at solution ``x``."""
        def vn(i: int) -> float:
            return float(x[i]) if i >= 0 else 0.0

        volts = {name: float(x[i]) for i, name in enumerate(self.node_names)}
        cur: dict[str, float] = {}
        pw: dict[str, float] = {}
        devs: dict[str, dict] = {}
        v = self.Nv @ x if self.mv else np.zeros(0)
        infos = {}
        for g in self.groups:
            infos[id(g)] = g.info(v[g.vs:g.ve])
        for name, r in self.recs.items():
            ns = r.n
            k = r.kind
            if k in ("R", "S"):
                vd = vn(ns[0]) - vn(ns[1])
                cur[name] = vd * r.g
                pw[name] = vd * vd * r.g
            elif k == "C":
                cur[name] = 0.0
                pw[name] = 0.0
            elif k in ("L", "V", "E"):
                i = float(x[r.br])
                cur[name] = i
                pw[name] = (vn(ns[0]) - vn(ns[1])) * i
            elif k == "I":
                val = float(u[self.src_index[name]])
                cur[name] = val
                pw[name] = (vn(ns[0]) - vn(ns[1])) * val
            elif k == "G":
                i = r.g * (vn(ns[2]) - vn(ns[3]))
                cur[name] = i
                pw[name] = (vn(ns[0]) - vn(ns[1])) * i
            elif k == "D":
                d = dict(infos[id(r.group)][r.slot])
                d["vd"] = vn(ns[0]) - vn(ns[1])       # terminal voltage, includes RS drop
                cur[name] = d["id"]
                pw[name] = d["vd"] * d["id"]
                devs[name] = d
            elif k == "Q":
                d = dict(infos[id(r.group)][r.slot])
                c_, b_, e_ = ns
                d["vbe"] = vn(b_) - vn(e_)
                d["vce"] = vn(c_) - vn(e_)
                d["vbc"] = vn(b_) - vn(c_)
                cur[name] = d["ic"]
                pw[name] = d["ic"] * d["vce"] + d["ib"] * d["vbe"]
                d["power"] = pw[name]
                devs[name] = d
            elif k == "J":
                d = dict(infos[id(r.group)][r.slot])
                d_, g_, s_ = ns
                d["vgs"] = vn(g_) - vn(s_)
                d["vds"] = vn(d_) - vn(s_)
                cur[name] = d["id"]
                pw[name] = d["id"] * d["vds"] + d["ig"] * d["vgs"]
                d["power"] = pw[name]
                devs[name] = d
            elif k == "M":
                d = dict(infos[id(r.group)][r.slot])
                d_, g_, s_, b_ = ns
                d["vgs"] = vn(g_) - vn(s_)
                d["vds"] = vn(d_) - vn(s_)
                d["vbs"] = vn(b_) - vn(s_)
                cur[name] = d["id"]
                pw[name] = d["id"] * d["vds"] + d["ib"] * d["vbs"]
                d["power"] = pw[name]
                devs[name] = d
            elif k == "X":
                d = dict(infos[id(r.group)][r.slot])
                p, vcc, vee, out = r.extra["ports"]
                vout = vn(out)
                iout = (d["vout_internal"] - vout) * r.extra["gout"]
                vsup = vn(vcc) - vn(vee)
                rail = vn(vcc) if iout >= 0 else vn(vee)
                pd = r.extra["iq"] * vsup + iout * (rail - vout)
                d.update(vout=vout, iout=iout, iq=r.extra["iq"],
                         vd=vn(ns[0]) - vn(ns[1]), supply_span=vsup)
                cur[name] = iout
                pw[name] = pd
                devs[name] = d
        return volts, cur, pw, devs


# --------------------------------------------------------------------------- transient

SMALL_M = 10            # port count up to which Newton runs on Python floats
PRED_MAX = 0.05         # largest per-step port change (V) the predictor extrapolates


_mul = operator.mul


def _solve4(A: list, b: list) -> list | None:
    """4x4 solve by 2x2 block elimination (Schur complement), fully unrolled.

    Two BJTs give exactly this size, so it is the hot case of the audio path.
    Returns None when the leading block or the Schur complement is badly
    conditioned; the caller then falls back to pivoted elimination.
    """
    (a11, a12, b11, b12), (a21, a22, b21, b22), (c11, c12, d11, d12), (c21, c22, d21, d22) = A
    det = a11 * a22 - a12 * a21
    if not abs(det) > 1e-9 * (abs(a11 * a22) + abs(a12 * a21)):
        return None
    i11, i12, i21, i22 = a22 / det, -a12 / det, -a21 / det, a11 / det
    # E = A^-1 B, y = A^-1 b1
    e11 = i11 * b11 + i12 * b21
    e12 = i11 * b12 + i12 * b22
    e21 = i21 * b11 + i22 * b21
    e22 = i21 * b12 + i22 * b22
    r1, r2, r3, r4 = b
    y1 = i11 * r1 + i12 * r2
    y2 = i21 * r1 + i22 * r2
    # Schur complement S = D - C E, right-hand side t = b2 - C y
    s11 = d11 - c11 * e11 - c12 * e21
    s12 = d12 - c11 * e12 - c12 * e22
    s21 = d21 - c21 * e11 - c22 * e21
    s22 = d22 - c21 * e12 - c22 * e22
    t1 = r3 - c11 * y1 - c12 * y2
    t2 = r4 - c21 * y1 - c22 * y2
    dets = s11 * s22 - s12 * s21
    if not abs(dets) > 1e-9 * (abs(s11 * s22) + abs(s12 * s21)):
        return None
    x3 = (t1 * s22 - s12 * t2) / dets
    x4 = (s11 * t2 - s21 * t1) / dets
    return [y1 - e11 * x3 - e12 * x4, y2 - e21 * x3 - e22 * x4, x3, x4]


def _solve_small(A: list, b: list) -> list | None:
    """Solve a small dense system given as row lists (A is overwritten).

    2x2 uses Cramer's rule; larger sizes Gaussian elimination with partial
    pivoting.  Returns None for a singular or non-finite matrix.
    """
    n = len(b)
    if n == 1:
        a = A[0][0]
        return [b[0] / a] if a != 0.0 and a == a else None
    if n == 2:
        (a, bb), (c, d) = A
        det = a * d - bb * c
        if not (det != 0.0 and abs(det) > 1e-300 * (abs(a * d) + abs(bb * c))):
            return None
        return [(b[0] * d - bb * b[1]) / det, (a * b[1] - c * b[0]) / det]
    if n == 3:
        (a, bb, cc), (d, e, f), (g, h, i) = A
        c0, c1, c2 = e * i - f * h, f * g - d * i, d * h - e * g
        det = a * c0 + bb * c1 + cc * c2
        if abs(det) > 1e-9 * (abs(a * c0) + abs(bb * c1) + abs(cc * c2)):
            r0, r1, r2 = b
            return [(r0 * c0 + r1 * (cc * h - bb * i) + r2 * (bb * f - cc * e)) / det,
                    (r0 * c1 + r1 * (a * i - cc * g) + r2 * (cc * d - a * f)) / det,
                    (r0 * c2 + r1 * (bb * g - a * h) + r2 * (a * e - bb * d)) / det]
    if n == 4:
        x = _solve4(A, b)
        if x is not None:
            return x
    elif n >= 5:            # LAPACK beats interpreted elimination from here on
        try:
            x = np.linalg.solve(np.array(A), np.array(b))
        except np.linalg.LinAlgError:
            return None
        return x.tolist() if np.all(np.isfinite(x)) else None
    for c in range(n):
        piv = c
        best = abs(A[c][c])
        for r in range(c + 1, n):
            t = abs(A[r][c])
            if t > best:
                best, piv = t, r
        if not best > 0.0:          # zero pivot or NaN
            return None
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            b[c], b[piv] = b[piv], b[c]
        rowc = A[c]
        inv = 1.0 / rowc[c]
        bc = b[c]
        tail = rowc[c + 1:]
        for r in range(c + 1, n):
            row = A[r]
            f = row[c] * inv
            if f != 0.0:
                row[c + 1:] = [x - f * y for x, y in zip(row[c + 1:], tail)]
                b[r] -= f * bc
    x = [0.0] * n
    for c in range(n - 1, -1, -1):
        row = A[c]
        acc = b[c]
        for k in range(c + 1, n):
            acc -= row[k] * x[k]
        x[c] = acc / row[c]
    return x


class _Level:
    """Matrices of the reduced (DK) system for one step size h.

    With A = G + (2/h) Cm + Ni.T Jlin Nv (Jlin: device Jacobian at the
    operating point, charges included), one trapezoidal step reads

        x = Mu u + Mh hv - Mi f~,   f~ = i(v) - Jlin v,   v = Nv x

    where hv are the reactive history currents.  Projecting on the ports gives
    the small nonlinear system  Lm v + K i(v) = p  with K = Nv Mi,
    Lm = I - K Jlin and p = Nv (Mu u + Mh hv).
    """

    def __init__(self, net: Network, h: float, J0: np.ndarray, Cq0: np.ndarray, out_rows: np.ndarray):
        a = 2.0 / h
        self.h, self.a = h, a
        Jlin = J0 + a * Cq0 if net.mv else J0
        A = net.G + np.diag(net.gdiag) + a * net.Cm
        if net.mv:
            A = A + net.Ni.T @ Jlin @ net.Nv
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError as exc:
            raise SimulationError("the transient matrix is singular: check for a loop of voltage "
                                  "sources/inductors or a node with no path to the rest of the circuit") from exc
        if not np.all(np.isfinite(Ainv)):
            raise SimulationError("the transient matrix is singular (non-finite inverse)")
        self.Jlin = Jlin
        Mu = Ainv @ net.Bsrc
        Mh = Ainv @ net.S.T
        Mi = Ainv @ net.Ni.T
        nr = net.S.shape[0]
        no = len(out_rows)
        self.nr, self.no = nr, no
        self.Pu, self.Ph, self.K = net.Nv @ Mu, net.Nv @ Mh, net.Nv @ Mi
        self.Lm = np.eye(net.mv) - self.K @ Jlin
        # reactive state rows then output rows, as one stacked system
        Wu = np.vstack((net.S @ Mu, Mu[out_rows]))
        Wh = np.vstack((net.S @ Mh, Mh[out_rows]))
        Wi = np.vstack((net.S @ Mi, Mi[out_rows]))
        self.Wu = Wu
        self.Hall = np.vstack((self.Ph, Wh))                 # everything hv feeds, in one matvec
        self.Wiv = np.hstack((-Wi, Wi @ Jlin))                # acts on concat(i_final, v_final)
        self.ac = a * net.cval
        self.mv = net.mv
        # plain-list copies for the small-system Newton path
        self.small = 0 < net.mv <= SMALL_M
        jf = net.jflat
        self.pat = [(int(f) // net.mv, int(f) % net.mv) for f in jf] if net.mv else []
        self.LK_l = np.hstack((self.Lm, self.K)).tolist() if net.mv else []
        self.Kt_l = self.K.T.tolist()
        LmT = self.Lm.T.tolist()
        self.colspec = [(LmT[vc], [(pidx, ir) for pidx, (ir, c) in enumerate(self.pat) if c == vc])
                        for vc in range(net.mv)]


class Transient:
    """Fixed-step trapezoidal integration with Newton on the nonlinear ports only.

    The state is physical (capacitor voltages and currents, inductor currents
    and voltages, junction charges and charge currents), so a step whose
    Newton iteration fails can be retried as smaller sub-steps directly.
    """

    def __init__(self, net: Network, x_op: np.ndarray, out_rows, *,
                 vtol: float = 1e-6, reltol: float = 1e-6, maxiter: int = 40, max_halvings: int = 8,
                 predict: bool = True):
        self.net = net
        self.out_rows = np.asarray(out_rows, dtype=int)
        self.vtol, self.reltol, self.maxiter = vtol, reltol, maxiter
        self.max_halvings = max_halvings
        self.predict = predict
        mv, mi = net.mv, net.mi
        v0 = net.Nv @ x_op if mv else np.zeros(0)
        self.v = v0.tolist()
        if mv:
            _, jv = net.eval(v0)
            self.J0 = net.jac(jv)
            if net.has_charge:
                q0, cv = net.charges(v0)
                self.Cq0 = net.jac(cv)
            else:
                q0 = np.zeros(mi)
                self.Cq0 = np.zeros_like(self.J0)
        else:
            self.J0 = self.Cq0 = np.zeros((0, 0))
            q0 = np.zeros(0)
        self.q = q0.tolist()
        self.iq = [0.0] * mi
        self.y = net.S @ x_op
        self.z = np.zeros(len(net.cval))
        self.levels: dict[int, _Level] = {}
        self.failed_steps = 0
        self.substeps = 0
        self.iterations = 0
        self._J = np.zeros((mi, mv))
        self._groups = list(net.groups)
        self._single = len(self._groups) == 1
        self._last_out = x_op[self.out_rows]

    def level(self, h: float, depth: int) -> _Level:
        lv = self.levels.get(depth)
        if lv is None:
            lv = _Level(self.net, h, self.J0, self.Cq0, self.out_rows)
            self.levels[depth] = lv
        return lv

    # -- devices ------------------------------------------------------------
    def _eval(self, vl: list, a: float):
        if self._single:
            return self._groups[0].fast_eval(vl, a, self.q, self.iq)
        il: list = []
        jl: list = []
        ql: list = []
        anyq = False
        for g in self._groups:
            qs = self.q[g.is_:g.ie]
            i_, j_, q_ = g.fast_eval(vl[g.vs:g.ve], a, qs, self.iq[g.is_:g.ie])
            il += i_
            jl += j_
            if q_ is None:
                ql += qs
            else:
                anyq = True
                ql += q_
        return il, jl, (ql if anyq else None)

    def _limit(self, vnew: list, vold: list) -> list:
        if self._single:
            return self._groups[0].fast_limit(vnew, vold)
        out: list = []
        for g in self._groups:
            out += g.fast_limit(vnew[g.vs:g.ve], vold[g.vs:g.ve])
        return out

    def _newton(self, lv: _Level, p: np.ndarray, vl: list):
        """Solve Lm v + K i(v) = p for the port voltages, starting from ``vl``.

        Returns (ok, v_final, i_final, q) with lists.  The final Newton update
        is applied together with its linear current correction, so accepting
        a step of size |dv| leaves an error of order |dv|^2 / (n Vt).
        """
        vtol, rtol, a = self.vtol, self.reltol, lv.a
        pat = lv.pat
        small = lv.small
        if small:
            LK, Kt, colspec = lv.LK_l, lv.Kt_l, lv.colspec
            pl = p.tolist()
        else:
            J, jflat, Kn, Lmn = self._J, self.net.jflat, lv.K, lv.Lm
        il: list = []
        ql = None
        for _ in range(self.maxiter):
            self.iterations += 1
            il, jl, ql = self._eval(vl, a)
            if small:
                vi = vl + il
                r = [pa - sum(map(_mul, row, vi)) for pa, row in zip(pl, LK)]
                # M = Lm + K J, built column by column (J is sparse: a few entries per column)
                cols = []
                for base, terms in colspec:
                    col = base
                    for pidx, ir in terms:
                        val = jl[pidx]
                        if val:
                            col = [x + val * y for x, y in zip(col, Kt[ir])]
                    cols.append(col)
                M = [list(row) for row in zip(*cols)]
                dvl = _solve_small(M, r)
                if dvl is None:
                    return False, vl, il, ql
            else:
                J.put(jflat, jl)
                vn = np.array(vl)
                try:
                    dvl = np.linalg.solve(Lmn + Kn @ J, p - Lmn @ vn - Kn @ np.array(il)).tolist()
                except np.linalg.LinAlgError:
                    return False, vl, il, ql
            done = True
            for d, x in zip(dvl, vl):
                if not abs(d) <= vtol + rtol * abs(x):     # also catches NaN
                    done = False
                    break
            if done:
                ifin = list(il)
                for (ir, vc), val in zip(pat, jl):
                    ifin[ir] += val * dvl[vc]
                return True, [x + d for x, d in zip(vl, dvl)], ifin, ql
            if not all(map(math.isfinite, dvl)):
                return False, vl, il, ql
            vl = self._limit([x + d for x, d in zip(vl, dvl)], vl)
        return False, vl, il, ql

    def _commit(self, lv: _Level, wu: np.ndarray, hv: np.ndarray, hw: np.ndarray,
                v: np.ndarray, i: np.ndarray, q) -> None:
        """Accept a step: update reactive/charge state and remember the outputs."""
        w = wu + hw
        if lv.mv:
            w = w + lv.Wiv @ np.array(i + v)
        nr = lv.nr
        y_new = w[:nr]
        self._last_out = w[nr:]
        self.z = lv.ac * y_new - hv
        self.y = y_new
        if q is not None:
            a = lv.a
            self.iq = [a * (qn - qo) - io for qn, qo, io in zip(q, self.q, self.iq)]
            self.q = q
        self.v = v

    def _substep(self, h: float, depth: int, t0: float, src) -> bool:
        """Advance from t0 by h, halving h recursively when Newton fails."""
        lv = self.level(h, depth)
        u = src(np.array([t0 + h]))[:, 0]
        hv = lv.ac * self.y + self.z
        hall = lv.Hall @ hv
        mv = lv.mv
        p = lv.Pu @ u + hall[:mv]
        if mv:
            ok, v, i, q = self._newton(lv, p, self.v)
        else:
            ok, v, i, q = True, self.v, [], None
        if not ok and depth < self.max_halvings:
            self.substeps += 1
            first = self._substep(h / 2, depth + 1, t0, src)
            second = self._substep(h / 2, depth + 1, t0 + h / 2, src)
            return first and second
        if not ok:
            self.failed_steps += 1
        self._commit(lv, lv.Wu @ u, hv, hall[mv:], v, i, q)
        return ok

    def run(self, h: float, nt: int, src, x_op: np.ndarray) -> np.ndarray:
        """Integrate ``nt - 1`` steps of size h; returns outputs (nt x len(out_rows)).

        ``src(t)`` returns the source matrix (n_src x len(t)).  Row 0 of the
        result is the operating point ``x_op`` (the state at t = 0).
        """
        lv = self.level(h, 0)
        t = np.arange(nt) * h
        U = src(t)
        PU = (lv.Pu @ U).T.copy()
        WU = (lv.Wu @ U).T.copy()
        Y = np.empty((nt, len(self.out_rows)))
        Y[0] = x_op[self.out_rows]
        Hall, Wiv, acv, a, nr, mv = lv.Hall, lv.Wiv, lv.ac, lv.a, lv.nr, lv.mv
        newton = self._newton
        array = np.array
        predict = self.predict
        self._vprev = self.v
        for k in range(1, nt):
            hv = acv * self.y + self.z
            hall = Hall @ hv
            if not mv:
                w = WU[k] + hall
                y_new = w[:nr]
                Y[k] = w[nr:]
                self.z = acv * y_new - hv
                self.y = y_new
                continue
            vp = self.v
            if predict:
                # linear extrapolation, but only for ports that moved gently last step:
                # extrapolating a junction that just snapped on or off overshoots badly
                vp = [a1 + (a1 - a0 if -PRED_MAX < a1 - a0 < PRED_MAX else 0.0)
                      for a1, a0 in zip(vp, self._vprev)]
            ok, v, i, q = newton(lv, PU[k] + hall[:mv], vp)
            if not ok:
                if self.max_halvings > 0:
                    self.substeps += 1
                    self._substep(h / 2, 1, t[k - 1], src)
                    self._substep(h / 2, 1, t[k - 1] + h / 2, src)
                    Y[k] = self._last_out
                    continue
                self.failed_steps += 1
            w = WU[k] + hall[mv:] + Wiv @ array(i + v)
            y_new = w[:nr]
            Y[k] = w[nr:]
            self.z = acv * y_new - hv
            self.y = y_new
            if q is not None:
                self.iq = [a * (qn - qo) - io for qn, qo, io in zip(q, self.q, self.iq)]
                self.q = q
            self._vprev = self.v
            self.v = v
        return Y
