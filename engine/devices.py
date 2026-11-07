"""Nonlinear device models for the Hotwire simulator.

Every nonlinear device is described as a set of *ports*: a voltage port is the
difference between two node voltages, and a current port is a current that
flows through the device from one node to another.  A model maps its voltage
ports to its port currents and returns the Jacobian entries.  The MNA core
(``mna.py``) only ever sees ports, which is what lets the same model code
serve the DC Newton solver, the AC linearisation and the fast audio path.

Models follow ngspice-42 level-1 equations and defaults at the nominal
temperature (27 C), so cross-checks against ngspice agree:

* Diode: Shockley with N, IS, RS (internal node), CJO/VJ/M/FC depletion
  charge, TT diffusion charge, optional BV/IBV reverse breakdown.
* BJT: Gummel-Poon DC subset (BF, BR, IS, NF, NR, VAF, VAR, IKF, IKR, ISE,
  NE, ISC, NC), RB/RC/RE as internal nodes, CJE/CJC depletion and TF/TR
  diffusion charge.  With the default IKF = ISE = ISC = 0 it reduces to the
  Ebers-Moll transport model with Early effect.  NPN and PNP.
* JFET: Shichman-Hodges (VTO, BETA, LAMBDA, B), gate diodes (IS, N),
  RD/RS, CGS/CGD.  N- and P-channel.
* MOSFET: SPICE level 1 (Shichman-Hodges, ngspice ``mos1``): VTO, KP,
  LAMBDA, GAMMA/PHI body effect, W/L/LD/M, RD/RS, bulk-drain and
  bulk-source junction diodes (IS) with CBD/CBS depletion charge (PB, MJ,
  FC), and constant gate overlap capacitances (CGSO/CGDO per metre of W,
  CGBO per metre of L, or CGS/CGD in farads).  N- and P-channel.  The
  overlap capacitances are linear, so ``mna.py`` stamps them directly.
* Op-amp: macromodel with RIN, A0, one dominant pole at GBW/A0, ROUT,
  tanh output limiter between (VEE + VSAT_LO, VCC - VSAT_HI) and a
  quiescent current IQ drawn from VCC to VEE.  The linear parts (RIN, gain
  stage, pole, ROUT, IQ) are stamped by ``mna.py``; this module only holds
  the limiter, which is the nonlinear part.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "VT", "GMIN", "DEFAULTS", "part_library", "model_params",
    "pnjlim", "fetlim", "limvds", "DiodeGroup", "BJTGroup", "JFETGroup", "MOSFETGroup", "OpAmpGroup",
    "mos_geometry", "mos_ngspice_params", "mos_core",
]

# ngspice-42 constants (src/include/ngspice/const.h) at the nominal 27 C.
BOLTZ = 1.38064852e-23
CHARGE = 1.6021766208e-19
TNOM = 300.15
VT = BOLTZ * TNOM / CHARGE          # thermal voltage, 25.864 mV
GMIN = 1e-12                        # conductance across every pn junction, as SPICE
EXP_MAX = 80.0                      # beyond this the exponential continues linearly
OPAMP_RP = 1e3                      # internal pole-node resistance of the macromodel
OPAMP_HALF_MIN = 0.01               # smallest output half-swing (V) when supplies collapse
OPAMP_CLAMP_K = 4.0                 # pole node held within mid +/- K*half (tanh(4) = 0.9993)
OPAMP_CLAMP_EPS = 0.01              # softplus corner width of that clamp (V)


def opamp_clamp_g(a0: float) -> float:
    """Anti-windup clamp conductance (S): stiff next to the gain stage's A0/RP."""
    return 1e3 + 10.0 * max(float(a0), 1.0) / OPAMP_RP


def _softplus(x: float) -> tuple[float, float]:
    """(EPS*ln(1 + exp(x/EPS)), its slope), overflow-safe."""
    z = x / OPAMP_CLAMP_EPS
    if z > 0.0:
        e = math.exp(-z)
        return x + OPAMP_CLAMP_EPS * math.log1p(e), 1.0 / (1.0 + e)
    e = math.exp(z)
    return OPAMP_CLAMP_EPS * math.log1p(e), e / (1.0 + e)

INF = math.inf

# ngspice level-1 defaults.  Parameters ngspice treats as "0 = infinite" are
# normalised to inf in model_params().
DEFAULTS: dict[str, dict[str, float]] = {
    "D": dict(IS=1e-14, N=1.0, RS=0.0, CJO=0.0, VJ=1.0, M=0.5, FC=0.5, TT=0.0,
              BV=INF, IBV=1e-3, NBV=0.0),
    "NPN": dict(IS=1e-16, BF=100.0, NF=1.0, VAF=INF, IKF=INF, ISE=0.0, NE=1.5,
                BR=1.0, NR=1.0, VAR=INF, IKR=INF, ISC=0.0, NC=2.0,
                RB=0.0, RE=0.0, RC=0.0, CJE=0.0, VJE=0.75, MJE=0.33, TF=0.0,
                CJC=0.0, VJC=0.75, MJC=0.33, TR=0.0, FC=0.5),
    "NJF": dict(VTO=-2.0, BETA=1e-4, LAMBDA=0.0, B=1.0, IS=1e-14, N=1.0, RD=0.0,
                RS=0.0, CGS=0.0, CGD=0.0, PB=1.0, M=0.5, FC=0.5),
    "OPAMP": dict(A0=2e5, GBW=1e6, RIN=1e12, ROUT=75.0, VSAT_HI=1.5, VSAT_LO=1.5,
                  IQ=1e-3),
    # ngspice mos1 defaults; W and L default to ngspice's DEFW = DEFL = 100 um, so
    # KP is the whole device's beta unless W/L say otherwise.  CGS/CGD (farads) and
    # MULT (the instance multiplier m=) are Hotwire conveniences.
    "NMOS": dict(VTO=0.0, KP=2e-5, GAMMA=0.0, PHI=0.6, LAMBDA=0.0, RD=0.0, RS=0.0,
                 CBD=0.0, CBS=0.0, IS=1e-14, PB=0.8, MJ=0.5, FC=0.5,
                 CGSO=0.0, CGDO=0.0, CGBO=0.0, CGS=0.0, CGD=0.0, LD=0.0,
                 W=1e-4, L=1e-4, MULT=1.0),
}
DEFAULTS["PNP"] = dict(DEFAULTS["NPN"])
DEFAULTS["PJF"] = dict(DEFAULTS["NJF"])
DEFAULTS["PMOS"] = dict(DEFAULTS["NMOS"])

# Alternative spellings accepted on .model lines (SPICE dialects).
ALIASES = {
    "D": {"CJ0": "CJO", "CJ": "CJO", "PB": "VJ", "MJ": "M"},
    "NPN": {"VA": "VAF", "VB": "VAR", "IK": "IKF", "C2": "ISE", "C4": "ISC",
            "PE": "VJE", "ME": "MJE", "PC": "VJC", "MC": "MJC"},
    "NJF": {"VT0": "VTO"},
    "OPAMP": {"AOL": "A0", "GAIN": "A0", "GB": "GBW", "RI": "RIN", "RO": "ROUT"},
    "NMOS": {"VT0": "VTO", "U0": "UO"},
}
ALIASES["PNP"] = ALIASES["NPN"]
ALIASES["PJF"] = ALIASES["NJF"]
ALIASES["PMOS"] = ALIASES["NMOS"]

# ngspice reads 0 for these as "infinite".
_ZERO_IS_INF = {"VAF", "VAR", "IKF", "IKR", "BV"}


def part_library() -> dict:
    """Built-in parts (``engine.parts_data.PARTS``), or {} when that table is absent."""
    try:
        from .parts_data import PARTS
    except ImportError:
        return {}
    return PARTS


def __getattr__(name: str) -> Any:   # PEP 562: PART_LIBRARY is loaded lazily
    if name == "PART_LIBRARY":
        return part_library()
    raise AttributeError(name)


def model_params(mtype: str, params: dict) -> dict:
    """Merge a model's parameters over the ngspice defaults for its type."""
    mtype = mtype.upper()
    base = dict(DEFAULTS[mtype])
    alias = ALIASES.get(mtype, {})
    for key, val in (params or {}).items():
        k = alias.get(key.upper(), key.upper())
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if k in _ZERO_IS_INF and v == 0:
            v = INF
        base[k] = v
    return base


# --------------------------------------------------------------------------- helpers

def _exp(x: np.ndarray) -> np.ndarray:
    """exp(x) continued linearly above EXP_MAX; the slope equals the value up there."""
    if x.size and x.max() > EXP_MAX:
        e = np.exp(np.minimum(x, EXP_MAX))
        return np.where(x > EXP_MAX, e * (1.0 + x - EXP_MAX), e)
    return np.exp(x)


def _dexp(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Derivative matching _exp."""
    if x.size and x.max() > EXP_MAX:
        return np.where(x > EXP_MAX, math.exp(EXP_MAX), e)
    return e


def depletion(v: np.ndarray, cj: np.ndarray, vj: np.ndarray, m: np.ndarray,
              fc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SPICE depletion charge and capacitance with the FC linearisation."""
    fcv = fc * vj
    below = v < fcv
    arg = 1.0 - np.where(below, v, 0.0) / vj
    sarg = np.exp(-m * np.log(arg))
    q_lo = vj * cj * (1.0 - arg * sarg) / (1.0 - m)
    c_lo = cj * sarg
    f1 = vj * (1.0 - (1.0 - fc) ** (1.0 - m)) / (1.0 - m)
    f2 = (1.0 - fc) ** (1.0 + m)
    f3 = 1.0 - fc * (1.0 + m)
    q_hi = cj * f1 + cj / f2 * (f3 * (v - fcv) + m / (2.0 * vj) * (v * v - fcv * fcv))
    c_hi = cj / f2 * (f3 + m * v / vj)
    return np.where(below, q_lo, q_hi), np.where(below, c_lo, c_hi)


def pnjlim(vnew: np.ndarray, vold: np.ndarray, vt: np.ndarray, vcrit: np.ndarray) -> np.ndarray:
    """SPICE junction-voltage limiting (DEVpnjlim from ngspice-42), vectorised."""
    big = (vnew > vcrit) & (np.abs(vnew - vold) > 2.0 * vt)
    out = vnew.copy()
    if big.any():
        vo, vn, t, vc = vold[big], vnew[big], vt[big], vcrit[big]
        arg = 1.0 + (vn - vo) / t
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(arg > 0, vo + t * np.log(np.maximum(arg, 1e-300)), vc)
            fresh = t * np.log(np.maximum(vn / t, 1e-300))
        out[big] = np.where(vo > 0, up, fresh)
    neg = (~big) & (vnew < 0)
    if neg.any():
        vo = vold[neg]
        floor = np.where(vo > 0, -vo - 1.0, 2.0 * vo - 1.0)
        out[neg] = np.maximum(vnew[neg], floor)
    return out


def fetlim(vnew: float, vold: float, vto: float) -> float:
    """SPICE FET gate-voltage limiting (DEVfetlim), scalar."""
    vtsthi = abs(2.0 * (vold - vto)) + 2.0
    vtstlo = abs(vold - vto) + 1.0
    vtox = vto + 3.5
    delv = vnew - vold
    if vold >= vto:
        if vold >= vtox:
            if delv <= 0:
                if vnew >= vtox:
                    if -delv > vtstlo:
                        vnew = vold - vtstlo
                else:
                    vnew = max(vnew, vto + 2.0)
            elif delv >= vtsthi:
                vnew = vold + vtsthi
        else:
            if delv <= 0:
                vnew = max(vnew, vto - 0.5)
            else:
                vnew = min(vnew, vto + 4.0)
    else:
        if delv <= 0:
            if -delv > vtsthi:
                vnew = vold - vtsthi
        else:
            vtemp = vto + 0.5
            if vnew <= vtemp:
                if delv > vtstlo:
                    vnew = vold + vtstlo
            else:
                vnew = vtemp
    return vnew


def limvds(vnew: float, vold: float) -> float:
    """SPICE drain-source step limiting for MOSFETs (DEVlimvds), scalar."""
    if vold >= 3.5:
        if vnew > vold:
            vnew = min(vnew, 3.0 * vold + 2.0)
        elif vnew < 3.5:
            vnew = max(vnew, 2.0)
    elif vnew > vold:
        vnew = min(vnew, 4.0)
    else:
        vnew = max(vnew, -0.5)
    return vnew


def _vcrit(vte: np.ndarray, isat: np.ndarray) -> np.ndarray:
    return vte * np.log(vte / (math.sqrt(2.0) * np.maximum(isat, 1e-300)))


# --------------------------------------------------------------------------- scalar fast path
# The transient inner loop evaluates a handful of devices per Newton iteration.
# numpy's per-call overhead (~0.6 us) dwarfs the arithmetic on 2-element arrays,
# so each group also has a plain-float version used by the transient engine.

_EMAX = math.exp(EXP_MAX)


def _sexp(x: float) -> float:
    """Scalar exp with the same linear continuation as _exp (slope = value)."""
    if x > EXP_MAX:
        return _EMAX * (1.0 + x - EXP_MAX)
    return math.exp(x)


def _sdexp(x: float, e: float) -> float:
    return _EMAX if x > EXP_MAX else e


def _dep_consts(vj: float, m: float, fc: float) -> tuple:
    f1 = vj * (1.0 - (1.0 - fc) ** (1.0 - m)) / (1.0 - m)
    f2 = (1.0 - fc) ** (1.0 + m)
    f3 = 1.0 - fc * (1.0 + m)
    return fc * vj, f1, f2, f3


def _sdep(v: float, cj: float, vj: float, m: float, k: tuple) -> tuple[float, float]:
    """Scalar depletion charge and capacitance (k from _dep_consts)."""
    if cj == 0.0:
        return 0.0, 0.0
    fcv, f1, f2, f3 = k
    if v < fcv:
        arg = 1.0 - v / vj
        sarg = math.exp(-m * math.log(arg))
        return vj * cj * (1.0 - arg * sarg) / (1.0 - m), cj * sarg
    return (cj * f1 + cj / f2 * (f3 * (v - fcv) + m / (2.0 * vj) * (v * v - fcv * fcv)),
            cj / f2 * (f3 + m * v / vj))


def _spnjlim(vnew: float, vold: float, vt: float, vcrit: float) -> float:
    """Scalar DEVpnjlim."""
    if vnew > vcrit and abs(vnew - vold) > vt + vt:
        if vold > 0:
            arg = 1.0 + (vnew - vold) / vt
            return vold + vt * math.log(arg) if arg > 0 else vcrit
        return vt * math.log(vnew / vt)
    if vnew < 0:
        floor = -vold - 1.0 if vold > 0 else 2.0 * vold - 1.0
        return vnew if vnew >= floor else floor
    return vnew


def _arr(insts: list, key: str) -> np.ndarray:
    return np.array([float(p[key]) for p in insts], dtype=float)


# --------------------------------------------------------------------------- base

class DeviceGroup:
    """All instances of one device type, evaluated together with numpy.

    Port layout is block-wise so each quantity is a contiguous slice: a BJT
    group with k devices has voltage ports [vbe_0..vbe_k-1, vbc_0..vbc_k-1].
    ``vports``/``iports`` list (plus_node, minus_node) global unknown indices
    (-1 = ground); ``pattern`` lists the (i_port, v_port) Jacobian entries in
    the order ``eval`` returns them.
    """

    kind = ""
    has_charge = False

    def __init__(self, names: list[str]):
        self.names = list(names)
        self.k = len(names)
        self.vports: list[tuple[int, int]] = []
        self.iports: list[tuple[int, int]] = []
        self.pattern: list[tuple[int, int]] = []

    @property
    def nv(self) -> int:
        return len(self.vports)

    @property
    def ni(self) -> int:
        return len(self.iports)

    def eval(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Port currents and Jacobian values at port voltages ``v`` (local)."""
        raise NotImplementedError

    def charges(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Port charges and their Jacobian values (same pattern as eval)."""
        raise NotImplementedError

    def limit(self, vnew: np.ndarray, vold: np.ndarray) -> np.ndarray:
        """Newton step limiting; returns the limited voltages."""
        return vnew

    def initial(self) -> np.ndarray:
        """Port voltages for the first DC Newton iteration (SPICE MODEINITJCT)."""
        return np.zeros(self.nv)

    def info(self, v: np.ndarray) -> list[dict]:
        """Operating-point report per instance."""
        return [{} for _ in range(self.k)]

    def fast_eval(self, v: list, a: float, qp: list | None, iqp: list | None):
        """Plain-float transient evaluation: (i, jv, q) as lists.

        With ``a = 2/h`` and the previous port charges ``qp`` and charge
        currents ``iqp``, the trapezoidal companion a*(q - qp) - iqp is added
        to ``i`` and a*dq/dv to ``jv``; ``a = 0`` means DC (no charges).
        """
        va = np.array(v)
        i, jv = self.eval(va)
        if a and self.has_charge:
            q, cv = self.charges(va)
            i = i + a * (q - np.asarray(qp)) - np.asarray(iqp)
            jv = jv + a * cv
            return i.tolist(), jv.tolist(), q.tolist()
        return i.tolist(), jv.tolist(), None

    def fast_limit(self, vnew: list, vold: list) -> list:
        return self.limit(np.array(vnew), np.array(vold)).tolist()


# --------------------------------------------------------------------------- diode

class DiodeGroup(DeviceGroup):
    """Junction diodes.  One port per device: (anode_internal, cathode)."""

    kind = "D"

    def __init__(self, names: list[str], params: list[dict]):
        super().__init__(names)
        self.IS = _arr(params, "IS")
        self.N = _arr(params, "N")
        self.vte = self.N * VT
        self.vcrit = _vcrit(self.vte, self.IS)
        self.TT = _arr(params, "TT")
        self.CJO = _arr(params, "CJO")
        self.VJ = _arr(params, "VJ")
        self.M = np.minimum(_arr(params, "M"), 0.9)
        self.FC = np.minimum(_arr(params, "FC"), 0.95)
        self.has_charge = bool(np.any(self.CJO > 0) or np.any(self.TT > 0))
        bv = _arr(params, "BV")
        self.has_bv = bool(np.any(np.isfinite(bv)))
        if self.has_bv:
            nbv = _arr(params, "NBV")
            self.vtbv = np.where(nbv > 0, nbv, self.N) * VT
            ibv = _arr(params, "IBV")
            # breakdown knee shifted so the current at -BV equals IBV (ngspice diotemp)
            self.tbv = np.where(np.isfinite(bv), bv - self.vtbv * np.log1p(ibv / self.IS), np.inf)

    def eval(self, v):
        x = v / self.vte
        e = _exp(x)
        cd = self.IS * (e - 1.0) + GMIN * v
        gd = self.IS * _dexp(x, e) / self.vte + GMIN
        if self.has_bv:
            brk = v < -self.tbv
            if brk.any():
                xb = np.where(brk, -(self.tbv + v) / self.vtbv, 0.0)
                eb = _exp(xb)
                cd = np.where(brk, -self.IS * eb + GMIN * v, cd)
                gd = np.where(brk, self.IS * _dexp(xb, eb) / self.vtbv + GMIN, gd)
        return cd, gd

    def charges(self, v):
        x = v / self.vte
        e = _exp(x)
        cd = self.IS * (e - 1.0)
        gd = self.IS * _dexp(x, e) / self.vte
        q, c = depletion(v, self.CJO, self.VJ, self.M, self.FC)
        return self.TT * cd + q, self.TT * gd + c

    def limit(self, vnew, vold):
        return pnjlim(vnew, vold, self.vte, self.vcrit)

    def initial(self):
        return self.vcrit.copy()

    def info(self, v):
        cd, gd = self.eval(v)
        return [{"id": float(cd[j]), "vd": float(v[j]), "gd": float(gd[j])} for j in range(self.k)]

    def _fast_setup(self):
        self._fp = [(float(self.IS[j]), float(self.vte[j]), float(self.vcrit[j]), float(self.TT[j]),
                     float(self.CJO[j]), float(self.VJ[j]), float(self.M[j]),
                     _dep_consts(float(self.VJ[j]), float(self.M[j]), float(self.FC[j])),
                     float(self.tbv[j]) if self.has_bv else math.inf,
                     float(self.vtbv[j]) if self.has_bv else 1.0)
                    for j in range(self.k)]

    def fast_eval(self, v, a, qp, iqp):
        if not hasattr(self, "_fp"):
            self._fast_setup()
        ii, jj, qq = [], [], []
        charge = bool(a) and self.has_charge
        for j, (isat, vte, _vc, tt, cj, vj, m, kd, tbv, vtbv) in enumerate(self._fp):
            vd = v[j]
            x = vd / vte
            e = _sexp(x)
            cd = isat * (e - 1.0)
            gd = isat * _sdexp(x, e) / vte
            if vd < -tbv:
                xb = -(tbv + vd) / vtbv
                eb = _sexp(xb)
                cdt, gdt = -isat * eb, isat * _sdexp(xb, eb) / vtbv
            else:
                cdt, gdt = cd, gd
            cdt += GMIN * vd
            gdt += GMIN
            if charge:
                qd, cdp = _sdep(vd, cj, vj, m, kd)
                q = tt * cd + qd
                cdt += a * (q - qp[j]) - iqp[j]
                gdt += a * (tt * gd + cdp)
                qq.append(q)
            ii.append(cdt)
            jj.append(gdt)
        return ii, jj, (qq if charge else None)

    def fast_limit(self, vnew, vold):
        if not hasattr(self, "_fp"):
            self._fast_setup()
        return [_spnjlim(vnew[j], vold[j], p[1], p[2]) for j, p in enumerate(self._fp)]


# --------------------------------------------------------------------------- BJT

class BJTGroup(DeviceGroup):
    """Bipolar transistors.  Ports: (b', e') and (b', c').

    Port currents: i_be = Ic + Ib (flows b'->e' through the device) and
    i_bc = -Ic (flows b'->c'), which reproduces the terminal currents
    Ib into the base, Ic into the collector and -(Ib+Ic) into the emitter.
    """

    kind = "Q"

    def __init__(self, names: list[str], params: list[dict], pol: list[float]):
        super().__init__(names)
        self.pol = np.array(pol, dtype=float)
        g = lambda key: _arr(params, key)  # noqa: E731
        self.IS, self.BF, self.BR = g("IS"), g("BF"), g("BR")
        self.vtf, self.vtr = g("NF") * VT, g("NR") * VT
        self.ivaf = 1.0 / g("VAF")
        self.ivar = 1.0 / g("VAR")
        self.iikf = 1.0 / g("IKF")
        self.iikr = 1.0 / g("IKR")
        self.has_ik = bool(np.any(self.iikf > 0) or np.any(self.iikr > 0))
        self.ISE, self.ISC = g("ISE"), g("ISC")
        self.vte, self.vtc = g("NE") * VT, g("NC") * VT
        self.has_leak = bool(np.any(self.ISE > 0) or np.any(self.ISC > 0))
        self.vcrit_be = _vcrit(self.vtf, self.IS)
        self.vcrit_bc = _vcrit(self.vtr, self.IS)
        self.CJE, self.VJE, self.MJE = g("CJE"), g("VJE"), np.minimum(g("MJE"), 0.9)
        self.CJC, self.VJC, self.MJC = g("CJC"), g("VJC"), np.minimum(g("MJC"), 0.9)
        self.TF, self.TR, self.FC = g("TF"), g("TR"), np.minimum(g("FC"), 0.95)
        self.has_charge = bool(np.any(self.CJE > 0) or np.any(self.CJC > 0)
                               or np.any(self.TF > 0) or np.any(self.TR > 0))

    def _core(self, v):
        k = self.k
        vbe = self.pol * v[:k]
        vbc = self.pol * v[k:]
        xe = vbe / self.vtf
        ee = _exp(xe)
        cbe = self.IS * (ee - 1.0)
        gbe = self.IS * _dexp(xe, ee) / self.vtf
        xc = vbc / self.vtr
        ec = _exp(xc)
        cbc = self.IS * (ec - 1.0)
        gbc = self.IS * _dexp(xc, ec) / self.vtr
        if self.has_leak:
            xle, xlc = vbe / self.vte, vbc / self.vtc
            ele, elc = _exp(xle), _exp(xlc)
            cben = self.ISE * (ele - 1.0) + GMIN * vbe
            gben = self.ISE * _dexp(xle, ele) / self.vte + GMIN
            cbcn = self.ISC * (elc - 1.0) + GMIN * vbc
            gbcn = self.ISC * _dexp(xlc, elc) / self.vtc + GMIN
        else:
            cben, gben = GMIN * vbe, GMIN
            cbcn, gbcn = GMIN * vbc, GMIN
        q1 = 1.0 / (1.0 - self.ivaf * vbc - self.ivar * vbe)
        if self.has_ik:
            q2 = self.iikf * cbe + self.iikr * cbc
            sq = np.sqrt(np.maximum(1.0 + 4.0 * q2, 1e-30))
            qb = q1 * (1.0 + sq) / 2.0
            dqbdve = q1 * (qb * self.ivar + self.iikf * gbe / sq)
            dqbdvc = q1 * (qb * self.ivaf + self.iikr * gbc / sq)
        else:
            qb = q1
            dqbdve = q1 * qb * self.ivar
            dqbdvc = q1 * qb * self.ivaf
        return vbe, vbc, cbe, gbe, cbc, gbc, cben, gben, cbcn, gbcn, qb, dqbdve, dqbdvc

    def eval(self, v):
        (vbe, vbc, cbe, gbe, cbc, gbc, cben, gben, cbcn, gbcn,
         qb, dqbdve, dqbdvc) = self._core(v)
        it = (cbe - cbc) / qb
        ic = it - cbc / self.BR - cbcn
        ib = cbe / self.BF + cben + cbc / self.BR + cbcn
        dic_e = (gbe - it * dqbdve) / qb
        dic_c = (-gbc - it * dqbdvc) / qb - gbc / self.BR - gbcn
        dib_e = gbe / self.BF + gben
        dib_c = gbc / self.BR + gbcn
        i = np.concatenate((self.pol * (ic + ib), -self.pol * ic))
        jv = np.concatenate((dic_e + dib_e, dic_c + dib_c, -dic_e, -dic_c))
        return i, jv

    def charges(self, v):
        (vbe, vbc, cbe, gbe, cbc, gbc, _a, _b, _c, _d,
         qb, dqbdve, dqbdvc) = self._core(v)
        # diffusion charge TF*Ic_f/qb when forward biased (ngspice bjtload, XTF = 0)
        fwd = vbe > 0
        cbe_m = np.where(fwd, cbe / qb, cbe)
        gbe_m = np.where(fwd, (gbe - cbe_m * dqbdve) / qb, gbe)
        geqcb = np.where(fwd, -self.TF * cbe_m * dqbdvc / qb, 0.0)
        qde, cde = depletion(vbe, self.CJE, self.VJE, self.MJE, self.FC)
        qdc, cdc = depletion(vbc, self.CJC, self.VJC, self.MJC, self.FC)
        qbe = self.TF * cbe_m + qde
        qbc = self.TR * cbc + qdc
        q = np.concatenate((self.pol * qbe, self.pol * qbc))
        z = np.zeros(self.k)
        cv = np.concatenate((self.TF * gbe_m + cde, geqcb, z, self.TR * gbc + cdc))
        return q, cv

    def limit(self, vnew, vold):
        p = np.concatenate((self.pol, self.pol))
        vt = np.concatenate((self.vtf, self.vtr))
        vc = np.concatenate((self.vcrit_be, self.vcrit_bc))
        return p * pnjlim(p * vnew, p * vold, vt, vc)

    def initial(self):
        return np.concatenate((self.pol * self.vcrit_be, np.zeros(self.k)))

    def info(self, v):
        k = self.k
        i, jv = self.eval(v)
        out = []
        for j in range(k):
            pol = self.pol[j]
            vbe, vbc = float(v[j]), float(v[k + j])
            ic = -float(i[k + j])
            ib = float(i[j]) - ic
            # device-polarity quantities decide the region
            vbe_d, vbc_d, ic_d = pol * vbe, pol * vbc, pol * ic
            vce_d = vbe_d - vbc_d
            if ic_d < max(1e-9, 100.0 * self.IS[j]) and vbe_d < self.vcrit_be[j]:
                region = "reverse" if vbc_d > vbe_d and vbc_d > 0.3 else "cutoff"
            elif vce_d < 0 and vbc_d > vbe_d:
                region = "reverse"
            elif vce_d < 0.2 * self.vtf[j] / VT:
                region = "saturation"
            else:
                region = "active"
            gm = -float(jv[2 * k + j])  # dIc/dVbe in device polarity
            out.append({
                "ic": ic, "ib": ib, "ie": -(ic + ib),
                "vbe": vbe, "vbc": vbc, "vce": vbe - vbc,
                "region": region, "gm": gm,
                "beta": (ic / ib) if ib != 0 else math.inf,
                "type": "npn" if pol > 0 else "pnp",
            })
        return out

    def _fast_setup(self):
        f = float
        self._fp = []
        for j in range(self.k):
            self._fp.append((
                f(self.pol[j]), f(self.IS[j]), f(self.BF[j]), f(self.BR[j]), f(self.vtf[j]), f(self.vtr[j]),
                f(self.ivaf[j]), f(self.ivar[j]), f(self.iikf[j]), f(self.iikr[j]),
                f(self.ISE[j]), f(self.vte[j]), f(self.ISC[j]), f(self.vtc[j]),
                f(self.CJE[j]), f(self.VJE[j]), f(self.MJE[j]),
                _dep_consts(f(self.VJE[j]), f(self.MJE[j]), f(self.FC[j])),
                f(self.CJC[j]), f(self.VJC[j]), f(self.MJC[j]),
                _dep_consts(f(self.VJC[j]), f(self.MJC[j]), f(self.FC[j])),
                f(self.TF[j]), f(self.TR[j]), f(self.vcrit_be[j]), f(self.vcrit_bc[j])))

    def fast_eval(self, v, a, qp, iqp):
        if not hasattr(self, "_fp"):
            self._fast_setup()
        k = self.k
        i1, i2 = [0.0] * k, [0.0] * k
        j1, j2, j3, j4 = [0.0] * k, [0.0] * k, [0.0] * k, [0.0] * k
        charge = bool(a) and self.has_charge
        q1l, q2l = ([0.0] * k, [0.0] * k) if charge else (None, None)
        for j, (pol, isat, bf, br, vtf, vtr, ivaf, ivar, iikf, iikr, ise, vte, isc, vtc,
                cje, vje, mje, kde, cjc, vjc, mjc, kdc, tf, tr, _vc1, _vc2) in enumerate(self._fp):
            vbe = pol * v[j]
            vbc = pol * v[k + j]
            xe = vbe / vtf
            ee = _sexp(xe)
            cbe = isat * (ee - 1.0)
            gbe = isat * _sdexp(xe, ee) / vtf
            xc = vbc / vtr
            ec = _sexp(xc)
            cbc = isat * (ec - 1.0)
            gbc = isat * _sdexp(xc, ec) / vtr
            cben = GMIN * vbe
            gben = GMIN
            cbcn = GMIN * vbc
            gbcn = GMIN
            if ise:
                xl = vbe / vte
                el = _sexp(xl)
                cben += ise * (el - 1.0)
                gben += ise * _sdexp(xl, el) / vte
            if isc:
                xl = vbc / vtc
                el = _sexp(xl)
                cbcn += isc * (el - 1.0)
                gbcn += isc * _sdexp(xl, el) / vtc
            q1 = 1.0 / (1.0 - ivaf * vbc - ivar * vbe)
            if iikf or iikr:
                q2 = iikf * cbe + iikr * cbc
                sq = math.sqrt(max(1.0 + 4.0 * q2, 1e-30))
                qb = q1 * (1.0 + sq) / 2.0
                dqbdve = q1 * (qb * ivar + iikf * gbe / sq)
                dqbdvc = q1 * (qb * ivaf + iikr * gbc / sq)
            else:
                qb = q1
                dqbdve = q1 * qb * ivar
                dqbdvc = q1 * qb * ivaf
            it = (cbe - cbc) / qb
            ic = it - cbc / br - cbcn
            ib = cbe / bf + cben + cbc / br + cbcn
            dic_e = (gbe - it * dqbdve) / qb
            dic_c = (-gbc - it * dqbdvc) / qb - gbc / br - gbcn
            dib_e = gbe / bf + gben
            dib_c = gbc / br + gbcn
            ibe = pol * (ic + ib)
            ibc = -pol * ic
            jee, jec, jce, jcc = dic_e + dib_e, dic_c + dib_c, -dic_e, -dic_c
            if charge:
                if vbe > 0:
                    cbe_m = cbe / qb
                    gbe_m = (gbe - cbe_m * dqbdve) / qb
                    geqcb = -tf * cbe_m * dqbdvc / qb
                else:
                    cbe_m, gbe_m, geqcb = cbe, gbe, 0.0
                qde, cde = _sdep(vbe, cje, vje, mje, kde)
                qdc, cdc = _sdep(vbc, cjc, vjc, mjc, kdc)
                qbe = pol * (tf * cbe_m + qde)
                qbc = pol * (tr * cbc + qdc)
                q1l[j] = qbe
                q2l[j] = qbc
                ibe += a * (qbe - qp[j]) - iqp[j]
                ibc += a * (qbc - qp[k + j]) - iqp[k + j]
                jee += a * (tf * gbe_m + cde)
                jec += a * geqcb
                jcc += a * (tr * gbc + cdc)
            i1[j] = ibe
            i2[j] = ibc
            j1[j] = jee
            j2[j] = jec
            j3[j] = jce
            j4[j] = jcc
        return i1 + i2, j1 + j2 + j3 + j4, (q1l + q2l if charge else None)

    def fast_limit(self, vnew, vold):
        if not hasattr(self, "_fp"):
            self._fast_setup()
        k = self.k
        out = list(vnew)
        for j, p in enumerate(self._fp):
            pol = p[0]
            out[j] = pol * _spnjlim(pol * vnew[j], pol * vold[j], p[4], p[24])
            out[k + j] = pol * _spnjlim(pol * vnew[k + j], pol * vold[k + j], p[5], p[25])
        return out


# --------------------------------------------------------------------------- JFET

class JFETGroup(DeviceGroup):
    """Junction FETs.  Ports: (g, s') and (g, d').

    i_gs = Igs + Ids and i_gd = Igd - Ids, where Ids is the channel current
    from drain to source; this reproduces the three terminal currents.
    """

    kind = "J"

    def __init__(self, names: list[str], params: list[dict], pol: list[float]):
        super().__init__(names)
        self.pol = np.array(pol, dtype=float)
        g = lambda key: _arr(params, key)  # noqa: E731
        self.VTO, self.BETA, self.LAMBDA = g("VTO"), g("BETA"), g("LAMBDA")
        self.B = g("B")
        pb = g("PB")
        self.bfac = (1.0 - self.B) / (pb - self.VTO)
        self.IS = g("IS")
        self.vtn = g("N") * VT
        self.vcrit = _vcrit(self.vtn, self.IS)
        self.CGS, self.CGD, self.PB = g("CGS"), g("CGD"), pb
        self.M, self.FC = np.minimum(g("M"), 0.9), np.minimum(g("FC"), 0.95)
        self.has_charge = bool(np.any(self.CGS > 0) or np.any(self.CGD > 0))

    def _channel(self, vgst, vds):
        """ngspice JFET level-1 drain current for vds >= 0: (cd, gm, gds)."""
        beta, lam, b = self.BETA, self.LAMBDA, self.B
        on = vgst > 0
        betap = beta * (1.0 + lam * vds)
        lin = on & (vgst >= vds)
        # linear region
        apart = 2.0 * b + 3.0 * self.bfac * (vgst - vds)
        cpart_l = vds * (vds * (self.bfac * vds - b) + vgst * apart)
        cd_l = betap * cpart_l
        gm_l = betap * vds * (apart + 3.0 * self.bfac * vgst)
        gds_l = betap * (vgst - vds) * apart + beta * lam * cpart_l
        # saturation region
        bf = vgst * self.bfac
        cpart_s = vgst * vgst * (b + bf)
        cd_s = betap * cpart_s
        gm_s = betap * vgst * (2.0 * b + 3.0 * bf)
        gds_s = lam * beta * cpart_s
        cd = np.where(on, np.where(lin, cd_l, cd_s), 0.0)
        gm = np.where(on, np.where(lin, gm_l, gm_s), 0.0)
        gds = np.where(on, np.where(lin, gds_l, gds_s), 0.0)
        return cd, gm, gds

    def _ids(self, vgs, vgd):
        """Channel current drain->source and its partials (device polarity)."""
        vds = vgs - vgd
        normal = vds >= 0
        vctl = np.where(normal, vgs, vgd) - self.VTO
        cd, gm, gds = self._channel(vctl, np.abs(vds))
        ids = np.where(normal, cd, -cd)
        dids_gs = np.where(normal, gm + gds, gds)
        dids_gd = np.where(normal, -gds, -(gm + gds))
        return ids, dids_gs, dids_gd

    def eval(self, v):
        k = self.k
        vgs = self.pol * v[:k]
        vgd = self.pol * v[k:]
        xs, xd = vgs / self.vtn, vgd / self.vtn
        es, ed = _exp(xs), _exp(xd)
        igs = self.IS * (es - 1.0) + GMIN * vgs
        ggs = self.IS * _dexp(xs, es) / self.vtn + GMIN
        igd = self.IS * (ed - 1.0) + GMIN * vgd
        ggd = self.IS * _dexp(xd, ed) / self.vtn + GMIN
        ids, dids_gs, dids_gd = self._ids(vgs, vgd)
        i = np.concatenate((self.pol * (igs + ids), self.pol * (igd - ids)))
        jv = np.concatenate((ggs + dids_gs, dids_gd, -dids_gs, ggd - dids_gd))
        return i, jv

    def charges(self, v):
        k = self.k
        vgs = self.pol * v[:k]
        vgd = self.pol * v[k:]
        qs, cs = depletion(vgs, self.CGS, self.PB, self.M, self.FC)
        qd, cdd = depletion(vgd, self.CGD, self.PB, self.M, self.FC)
        z = np.zeros(k)
        return np.concatenate((self.pol * qs, self.pol * qd)), np.concatenate((cs, z, z, cdd))

    def limit(self, vnew, vold):
        k = self.k
        p = np.concatenate((self.pol, self.pol))
        vt = np.concatenate((self.vtn, self.vtn))
        vc = np.concatenate((self.vcrit, self.vcrit))
        out = pnjlim(p * vnew, p * vold, vt, vc)
        vo = p * vold
        for j in range(2 * k):
            out[j] = fetlim(float(out[j]), float(vo[j]), float(self.VTO[j % k]))
        return p * out

    def initial(self):
        return np.concatenate((-self.pol, -self.pol))  # ngspice starts at vgs = vgd = -1

    def info(self, v):
        k = self.k
        i, jv = self.eval(v)
        out = []
        for j in range(k):
            pol = self.pol[j]
            vgs, vgd = float(v[j]), float(v[k + j])
            vds = vgs - vgd
            vgst = pol * vgs - self.VTO[j]
            if vgst <= 0:
                region = "cutoff"
            elif abs(vds) < vgst:
                region = "linear"
            else:
                region = "saturation"
            ids, dgs, dgd = self._ids(self.pol * v[:k], self.pol * v[k:])
            ig = float(i[j] + i[k + j])
            idrain = pol * float(ids[j])
            out.append({"id": idrain, "ig": ig, "is": -(idrain + ig), "vgs": vgs, "vds": vds,
                        "region": region, "gm": float(dgs[j] + dgd[j]),
                        "type": "njf" if pol > 0 else "pjf"})
        return out

    def _fast_setup(self):
        f = float
        self._fp = [(f(self.pol[j]), f(self.VTO[j]), f(self.BETA[j]), f(self.LAMBDA[j]), f(self.B[j]),
                     f(self.bfac[j]), f(self.IS[j]), f(self.vtn[j]), f(self.vcrit[j]),
                     f(self.CGS[j]), f(self.CGD[j]), f(self.PB[j]), f(self.M[j]),
                     _dep_consts(f(self.PB[j]), f(self.M[j]), f(self.FC[j])))
                    for j in range(self.k)]

    @staticmethod
    def _schan(vgst: float, vds: float, beta: float, lam: float, b: float, bfac: float):
        """Scalar ngspice level-1 channel current for vds >= 0: (cd, gm, gds)."""
        if vgst <= 0:
            return 0.0, 0.0, 0.0
        betap = beta * (1.0 + lam * vds)
        if vgst >= vds:
            apart = 2.0 * b + 3.0 * bfac * (vgst - vds)
            cpart = vds * (vds * (bfac * vds - b) + vgst * apart)
            return (betap * cpart, betap * vds * (apart + 3.0 * bfac * vgst),
                    betap * (vgst - vds) * apart + beta * lam * cpart)
        bf = vgst * bfac
        cpart = vgst * vgst * (b + bf)
        return betap * cpart, betap * vgst * (2.0 * b + 3.0 * bf), lam * beta * cpart

    def fast_eval(self, v, a, qp, iqp):
        if not hasattr(self, "_fp"):
            self._fast_setup()
        k = self.k
        i1, i2 = [0.0] * k, [0.0] * k
        j1, j2, j3, j4 = [0.0] * k, [0.0] * k, [0.0] * k, [0.0] * k
        charge = bool(a) and self.has_charge
        q1l, q2l = ([0.0] * k, [0.0] * k) if charge else (None, None)
        sch = self._schan
        for j, (pol, vto, beta, lam, b, bfac, isat, vtn, _vc, cgs, cgd, pb, m, kd) in enumerate(self._fp):
            vgs = pol * v[j]
            vgd = pol * v[k + j]
            xs, xd = vgs / vtn, vgd / vtn
            es, ed = _sexp(xs), _sexp(xd)
            igs = isat * (es - 1.0) + GMIN * vgs
            ggs = isat * _sdexp(xs, es) / vtn + GMIN
            igd = isat * (ed - 1.0) + GMIN * vgd
            ggd = isat * _sdexp(xd, ed) / vtn + GMIN
            vds = vgs - vgd
            if vds >= 0:
                cd, gm, gds = sch(vgs - vto, vds, beta, lam, b, bfac)
                ids, d_gs, d_gd = cd, gm + gds, -gds
            else:
                cd, gm, gds = sch(vgd - vto, -vds, beta, lam, b, bfac)
                ids, d_gs, d_gd = -cd, gds, -(gm + gds)
            ia = pol * (igs + ids)
            ib = pol * (igd - ids)
            jaa, jab, jba, jbb = ggs + d_gs, d_gd, -d_gs, ggd - d_gd
            if charge:
                qs, cs = _sdep(vgs, cgs, pb, m, kd)
                qd, cdd = _sdep(vgd, cgd, pb, m, kd)
                qs, qd = pol * qs, pol * qd
                q1l[j], q2l[j] = qs, qd
                ia += a * (qs - qp[j]) - iqp[j]
                ib += a * (qd - qp[k + j]) - iqp[k + j]
                jaa += a * cs
                jbb += a * cdd
            i1[j], i2[j] = ia, ib
            j1[j], j2[j], j3[j], j4[j] = jaa, jab, jba, jbb
        return i1 + i2, j1 + j2 + j3 + j4, (q1l + q2l if charge else None)

    def fast_limit(self, vnew, vold):
        if not hasattr(self, "_fp"):
            self._fast_setup()
        k = self.k
        out = list(vnew)
        for j, p in enumerate(self._fp):
            pol, vto, vtn, vcrit = p[0], p[1], p[7], p[8]
            for idx in (j, k + j):
                vn = _spnjlim(pol * vnew[idx], pol * vold[idx], vtn, vcrit)
                out[idx] = pol * fetlim(vn, pol * vold[idx], vto)
        return out


# --------------------------------------------------------------------------- MOSFET

# ngspice mos1 model parameters a deck may carry (everything else Hotwire knows is
# folded in: CGS/CGD farads become CGSO/CGDO per metre, W/L/MULT go on the instance).
_MOS_NGSPICE_KEYS = ("VTO", "KP", "GAMMA", "PHI", "LAMBDA", "RD", "RS", "CBD", "CBS", "IS",
                     "PB", "MJ", "FC", "CGSO", "CGDO", "CGBO", "LD")


def mos_geometry(mp: dict) -> dict:
    """Per-instance constants of a level-1 MOSFET from its merged model + instance parameters.

    Same scaling as ngspice mos1temp at the nominal temperature: beta = KP * M * W / Leff
    with Leff = L - 2 LD, junction saturation current and capacitances times M, RD/RS
    divided by M, overlap capacitances CGSO*W*M, CGDO*W*M, CGBO*Leff*M.
    """
    m = float(mp.get("MULT", 1.0))
    w, l_ = float(mp["W"]), float(mp["L"])
    leff = l_ - 2.0 * float(mp["LD"])
    if not (m > 0 and w > 0 and leff > 0):
        raise ValueError("MOSFET W, L - 2*LD and M must all be positive")
    return {
        "beta": float(mp["KP"]) * m * w / leff,
        "isat": float(mp["IS"]) * m,
        "cbd": float(mp["CBD"]) * m,
        "cbs": float(mp["CBS"]) * m,
        "rd": float(mp["RD"]) / m,
        "rs": float(mp["RS"]) / m,
        "cgs": (float(mp["CGSO"]) * w + float(mp["CGS"])) * m,
        "cgd": (float(mp["CGDO"]) * w + float(mp["CGD"])) * m,
        "cgb": float(mp["CGBO"]) * leff * m,
    }


def mos_ngspice_params(mp: dict) -> tuple[dict, dict]:
    """(model params, instance params) that make ngspice-42 level 1 simulate exactly what
    Hotwire does with the merged parameters ``mp`` (see :func:`mos_geometry`)."""
    w = float(mp["W"])
    model = {k: float(mp[k]) for k in _MOS_NGSPICE_KEYS if k in mp and math.isfinite(float(mp[k]))}
    model["CGSO"] = float(mp["CGSO"]) + float(mp["CGS"]) / w
    model["CGDO"] = float(mp["CGDO"]) + float(mp["CGD"]) / w
    inst = {"W": w, "L": float(mp["L"])}
    if float(mp.get("MULT", 1.0)) != 1.0:
        inst["M"] = float(mp["MULT"])
    return model, inst


def mos_core(vgs: float, vds: float, vbs: float, beta: float, lam: float, vbi: float,
             gamma: float, phi: float, sqphi: float) -> tuple:
    """ngspice mos1load channel current in device polarity.

    Returns (ids, dIds/dVgs, dIds/dVds, dIds/dVbs, vgst, |vds|) with ids flowing
    drain -> source; for vds < 0 drain and source swap roles (mode -1), as in SPICE.
    ``vbi`` is the device-polarity flat-band term VTO*type - GAMMA*sqrt(PHI).
    """
    if vds >= 0.0:
        vb, vg, vd = vbs, vgs, vds
    else:
        vb, vg, vd = vbs - vds, vgs - vds, -vds
    if gamma:
        if vb <= 0.0:
            sarg = math.sqrt(phi - vb)
        else:
            sarg = sqphi - vb / (sqphi + sqphi)
            if sarg < 0.0:
                sarg = 0.0
        von = vbi + gamma * sarg
        arg = gamma / (sarg + sarg) if sarg > 0.0 else 0.0
    else:
        von = vbi
        arg = 0.0
    vgst = vg - von
    if vgst <= 0.0:
        cd = gm = gds = 0.0
    else:
        betap = beta * (1.0 + lam * vd)
        if vgst <= vd:
            cd = 0.5 * betap * vgst * vgst
            gm = betap * vgst
            gds = 0.5 * lam * beta * vgst * vgst
        else:
            cd = betap * vd * (vgst - 0.5 * vd)
            gm = betap * vd
            gds = betap * (vgst - vd) + lam * beta * vd * (vgst - 0.5 * vd)
    gmbs = gm * arg
    if vds >= 0.0:
        return cd, gm, gds, gmbs, vgst, vd
    return -cd, -gm, gm + gds + gmbs, -gmbs, vgst, vd


_VT3 = -3.0 * VT


def _bulk(v: float, isat: float) -> tuple[float, float]:
    """ngspice mos1 bulk junction diode (N = 1): (current, conductance), device polarity."""
    if v <= _VT3:
        return GMIN * v - isat, GMIN
    x = v / VT
    e = _sexp(x)
    return isat * (e - 1.0) + GMIN * v, isat * _sdexp(x, e) / VT + GMIN


class MOSFETGroup(DeviceGroup):
    """Level-1 MOSFETs.

    ``tied`` groups hold devices whose bulk is the source node (the usual discrete
    part): voltage ports (g, s') and (d', s'), one current port (d', s') carrying
    the channel current minus the bulk-drain diode current.  Other devices use
    voltage ports (g, s'), (d', s'), (b, s') and current ports (d', s') channel,
    (b, s') bulk-source diode and (b, d') bulk-drain diode.  The gate draws no DC
    current; its overlap capacitances are linear and stamped by mna.py.
    """

    kind = "M"

    def __init__(self, names: list[str], params: list[dict], pol: list[float], tied: bool):
        super().__init__(names)
        self.tied = bool(tied)
        self.pol = np.array(pol, dtype=float)
        self._fp = []
        for mp, tp in zip(params, pol):
            g = mos_geometry(mp)
            gamma, phi = float(mp["GAMMA"]), float(mp["PHI"])
            if gamma and not phi > 0:
                raise ValueError("MOSFET PHI must be positive when GAMMA is set")
            sqphi = math.sqrt(phi) if phi > 0 else 0.0
            pb, mj = float(mp["PB"]), min(float(mp["MJ"]), 0.9)
            isat = g["isat"]
            self._fp.append((
                float(tp), g["beta"], float(mp["LAMBDA"]), tp * float(mp["VTO"]) - gamma * sqphi,
                gamma, phi, sqphi, isat, VT * math.log(VT / (math.sqrt(2.0) * max(isat, 1e-300))),
                g["cbd"], g["cbs"], pb, mj, _dep_consts(pb, mj, min(float(mp["FC"]), 0.95)),
                float(mp["VTO"])))
        self.has_charge = any(p[9] > 0 for p in self._fp) or (
            not self.tied and any(p[10] > 0 for p in self._fp))
        self.VTO = np.array([p[14] for p in self._fp])

    def local_ports(self) -> tuple[int, int]:
        """(voltage ports, current ports) per group."""
        k = self.k
        return (2 * k, k) if self.tied else (3 * k, 3 * k)

    def local_pattern(self) -> list:
        k = self.k
        if self.tied:
            return [(j, j) for j in range(k)] + [(j, k + j) for j in range(k)]
        return ([(j, j) for j in range(k)] + [(j, k + j) for j in range(k)]
                + [(j, 2 * k + j) for j in range(k)] + [(k + j, 2 * k + j) for j in range(k)]
                + [(2 * k + j, 2 * k + j) for j in range(k)] + [(2 * k + j, k + j) for j in range(k)])

    # -- plain-float evaluation (transient hot path; eval/charges wrap it) -----
    def fast_eval(self, v, a, qp, iqp):
        k = self.k
        charge = bool(a) and self.has_charge
        core = mos_core
        if self.tied:
            ii = [0.0] * k
            j1, j2 = [0.0] * k, [0.0] * k
            qq = [0.0] * k if charge else None
            for j, (tp, beta, lam, vbi, gamma, phi, sqphi, isat, _vc, cbd, _cbs, pb, mj, kd, _v) in enumerate(self._fp):
                vgs = tp * v[j]
                vds = tp * v[k + j]
                ids, ggs, gds, _gbs, _x, _y = core(vgs, vds, 0.0, beta, lam, vbi, gamma, phi, sqphi)
                # bulk-drain diode, vbd = -vds (bulk on the source)
                vbd = -vds
                if vbd <= _VT3:
                    cbd_i, gbd = GMIN * vbd - isat, GMIN
                else:
                    x = vbd / VT
                    e = _sexp(x)
                    cbd_i, gbd = isat * (e - 1.0) + GMIN * vbd, isat * _sdexp(x, e) / VT + GMIN
                i = tp * (ids - cbd_i)
                jd = gds + gbd
                if charge:
                    if cbd:
                        qd, cdd = _sdep(vbd, cbd, pb, mj, kd)
                    else:
                        qd = cdd = 0.0
                    q = -tp * qd
                    qq[j] = q
                    i += a * (q - qp[j]) - iqp[j]
                    jd += a * cdd
                ii[j] = i
                j1[j] = ggs
                j2[j] = jd
            return ii, j1 + j2, qq
        i1, i2, i3 = [0.0] * k, [0.0] * k, [0.0] * k
        p1, p2, p3, p4, p5, p6 = [0.0] * k, [0.0] * k, [0.0] * k, [0.0] * k, [0.0] * k, [0.0] * k
        q2, q3 = ([0.0] * k, [0.0] * k) if charge else (None, None)
        for j, (tp, beta, lam, vbi, gamma, phi, sqphi, isat, _vc, cbd, cbs, pb, mj, kd, _v) in enumerate(self._fp):
            vgs = tp * v[j]
            vds = tp * v[k + j]
            vbs = tp * v[2 * k + j]
            ids, ggs, gds, gbs_ch, _x, _y = core(vgs, vds, vbs, beta, lam, vbi, gamma, phi, sqphi)
            vbd = vbs - vds
            # bulk junction diodes (inlined _bulk: this is the transient hot path)
            if vbs <= _VT3:
                jbs = GMIN
                ib_s = tp * (GMIN * vbs - isat)
            else:
                x = vbs / VT
                e = _sexp(x)
                ib_s = tp * (isat * (e - 1.0) + GMIN * vbs)
                jbs = isat * _sdexp(x, e) / VT + GMIN
            if vbd <= _VT3:
                jbd = GMIN
                ib_d = tp * (GMIN * vbd - isat)
            else:
                x = vbd / VT
                e = _sexp(x)
                ib_d = tp * (isat * (e - 1.0) + GMIN * vbd)
                jbd = isat * _sdexp(x, e) / VT + GMIN
            if charge:
                qs, cs = _sdep(vbs, cbs, pb, mj, kd) if cbs else (0.0, 0.0)
                qd, cdd = _sdep(vbd, cbd, pb, mj, kd) if cbd else (0.0, 0.0)
                qs, qd = tp * qs, tp * qd
                q2[j], q3[j] = qs, qd
                ib_s += a * (qs - qp[k + j]) - iqp[k + j]
                ib_d += a * (qd - qp[2 * k + j]) - iqp[2 * k + j]
                jbs += a * cs
                jbd += a * cdd
            i1[j], i2[j], i3[j] = tp * ids, ib_s, ib_d
            p1[j], p2[j], p3[j] = ggs, gds, gbs_ch
            p4[j], p5[j], p6[j] = jbs, jbd, -jbd
        if charge:
            # the channel port carries no charge: keep its history at zero
            return i1 + i2 + i3, p1 + p2 + p3 + p4 + p5 + p6, [0.0] * k + q2 + q3
        return i1 + i2 + i3, p1 + p2 + p3 + p4 + p5 + p6, None

    def eval(self, v):
        i, jv, _ = self.fast_eval(np.asarray(v, dtype=float).tolist(), 0.0, None, None)
        return np.array(i), np.array(jv)

    def charges(self, v):
        k = self.k
        vl = np.asarray(v, dtype=float).tolist()
        if self.tied:
            q, c = [0.0] * k, [0.0] * (2 * k)
            for j, p in enumerate(self._fp):
                tp, cbd, pb, mj, kd = p[0], p[9], p[11], p[12], p[13]
                vbd = -tp * vl[k + j]
                qd, cdd = _sdep(vbd, cbd, pb, mj, kd)
                q[j] = -tp * qd
                c[k + j] = cdd
            return np.array(q), np.array(c)
        q, c = [0.0] * (3 * k), [0.0] * (6 * k)
        for j, p in enumerate(self._fp):
            tp, cbd, cbs, pb, mj, kd = p[0], p[9], p[10], p[11], p[12], p[13]
            vds, vbs = tp * vl[k + j], tp * vl[2 * k + j]
            qs, cs = _sdep(vbs, cbs, pb, mj, kd)
            qd, cdd = _sdep(vbs - vds, cbd, pb, mj, kd)
            q[k + j], q[2 * k + j] = tp * qs, tp * qd
            c[3 * k + j], c[4 * k + j], c[5 * k + j] = cs, cdd, -cdd
        return np.array(q), np.array(c)

    # -- limiting (ngspice mos1load: fetlim on vgs or vgd, limvds, pnjlim on the bulk)
    def fast_limit(self, vnew, vold):
        k = self.k
        out = list(vnew)
        tied = self.tied
        for j, (tp, _b, _l, vbi, gamma, phi, sqphi, _is, vcrit, *_rest) in enumerate(self._fp):
            vgs_n, vds_n = tp * vnew[j], tp * vnew[k + j]
            vgs_o, vds_o = tp * vold[j], tp * vold[k + j]
            if tied:
                vbs_n = vbs_o = 0.0
            else:
                vbs_n, vbs_o = tp * vnew[2 * k + j], tp * vold[2 * k + j]
            vb = vbs_o if vds_o >= 0.0 else vbs_o - vds_o
            if gamma:
                if vb <= 0.0:
                    sarg = math.sqrt(phi - vb)
                else:
                    sarg = max(sqphi - vb / (sqphi + sqphi), 0.0)
                von = vbi + gamma * sarg
            else:
                von = vbi
            vgd_n = vgs_n - vds_n
            if vds_o >= 0.0:
                vgs_n = fetlim(vgs_n, vgs_o, von)
                vds_n = limvds(vgs_n - vgd_n, vds_o)
            else:
                vgd_n = fetlim(vgd_n, vgs_o - vds_o, von)
                vds_n = -limvds(-(vgs_n - vgd_n), -vds_o)
                vgs_n = vgd_n + vds_n
            out[j], out[k + j] = tp * vgs_n, tp * vds_n
            if not tied:
                if vds_n >= 0.0:
                    vbs_n = _spnjlim(vbs_n, vbs_o, VT, vcrit)
                else:
                    vbs_n = _spnjlim(vbs_n - vds_n, vbs_o - vds_o, VT, vcrit) + vds_n
                out[2 * k + j] = tp * vbs_n
        return out

    def limit(self, vnew, vold):
        return np.array(self.fast_limit(np.asarray(vnew, dtype=float).tolist(),
                                        np.asarray(vold, dtype=float).tolist()))

    def initial(self):
        # SPICE MODEINITJCT: vgs = type*VTO (device polarity), vds = 0, vbs = -1
        k = self.k
        out = [p[14] for p in self._fp] + [0.0] * k
        if not self.tied:
            out += [-p[0] for p in self._fp]
        return np.array(out)

    def info(self, v):
        k = self.k
        vl = np.asarray(v, dtype=float).tolist()
        i, jv, _ = self.fast_eval(vl, 0.0, None, None)
        out = []
        for j, (tp, beta, lam, vbi, gamma, phi, sqphi, isat, *_r) in enumerate(self._fp):
            vgs, vds = vl[j], vl[k + j]
            vbs = 0.0 if self.tied else vl[2 * k + j]
            ids, ggs, gds, gbs, vgst, vdabs = mos_core(tp * vgs, tp * vds, tp * vbs, beta, lam, vbi,
                                                       gamma, phi, sqphi)
            if vgst <= 0.0:
                region = "cutoff"
            elif vdabs < vgst:
                region = "triode"
            else:
                region = "saturation"
            if self.tied:
                idrain, ib = i[j], 0.0
            else:
                idrain = i[j] - i[2 * k + j]          # channel out of d' minus bulk-drain diode into d'
                ib = i[k + j] + i[2 * k + j]
            out.append({"id": idrain, "ib": ib, "is": -(idrain + ib), "vgs": vgs, "vds": vds, "vbs": vbs,
                        "region": region, "gm": ggs, "gds": gds, "gmbs": gbs,
                        # threshold (von) in circuit polarity, referred to the conducting source
                        "vth": tp * ((tp * vgs if tp * vds >= 0 else tp * (vgs - vds)) - vgst),
                        "type": "nmos" if tp > 0 else "pmos"})
        return out


# --------------------------------------------------------------------------- op-amp limiter

class OpAmpGroup(DeviceGroup):
    """The op-amp's nonlinear parts: the output limiter and the anti-windup clamp.

    Output limiter: a Norton current f(vp, vcc, vee) / ROUT into ``out`` with
    vo = mid + half * tanh((vp - mid) / half), hi = vcc - VSAT_HI,
    lo = vee + VSAT_LO, mid = (hi + lo) / 2, half = (hi - lo) / 2.

    Anti-windup clamp: the gain stage integrates into the pole node ``p``, which
    without a limit winds up to A0 times the overdrive (kilovolts) in saturation
    and then takes milliseconds to come back.  A smooth one-sided conductance
    GC = 1000 S + 10 A0/RP from ``p`` to ground holds it inside
    mid +/- OPAMP_CLAMP_K * half, where the tanh is 99.93 % saturated, so the full
    output swing is kept and recovery takes microseconds:

        i_clamp = GC * (softplus(vp - (mid + K half)) - softplus((mid - K half) - vp)),
        softplus(x) = EPS * ln(1 + exp(x / EPS)),  EPS = 10 mV.

    Voltage ports: [p, vcc, vee] (to ground); current ports: ground -> out (k),
    then p -> ground (k).

    With ``fixed_rails`` every amplifier's supply pins sit on ideal DC sources
    (or ground), so the rails are constants set through :meth:`set_rails` and
    the group needs only the [p] ports: a smaller Newton system per sample.
    """

    kind = "X"

    def __init__(self, names: list[str], params: list[dict], fixed_rails: bool = False):
        super().__init__(names)
        self.VSH = _arr(params, "VSAT_HI")
        self.VSL = _arr(params, "VSAT_LO")
        self.GOUT = 1.0 / np.maximum(_arr(params, "ROUT"), 1e-3)
        self.GC = np.array([opamp_clamp_g(p["A0"]) for p in params], dtype=float)
        self._vsh = self.VSH.tolist()
        self._vsl = self.VSL.tolist()
        self._gout = self.GOUT.tolist()
        self._gc = self.GC.tolist()
        self.fixed = fixed_rails
        self.vcc = np.zeros(self.k)
        self.vee = np.zeros(self.k)

    def local_pattern(self) -> list:
        """(i_port, v_port) entries: output rows then clamp rows."""
        k = self.k
        if self.fixed:
            return [(j, j) for j in range(k)] + [(k + j, j) for j in range(k)]
        rows = []
        for base in (0, k):
            rows += ([(base + j, j) for j in range(k)] + [(base + j, k + j) for j in range(k)]
                     + [(base + j, 2 * k + j) for j in range(k)])
        return rows

    def set_rails(self, vcc: np.ndarray, vee: np.ndarray) -> None:
        """Supply voltages for fixed-rail mode."""
        self.vcc = np.asarray(vcc, dtype=float)
        self.vee = np.asarray(vee, dtype=float)

    def _rails(self, v, j):
        k = self.k
        if self.fixed:
            return float(self.vcc[j]), float(self.vee[j])
        return v[k + j], v[2 * k + j]

    def fast_eval(self, v, a, qp, iqp):
        k = self.k
        io, ic = [0.0] * k, [0.0] * k
        o1, o2, o3 = [0.0] * k, [0.0] * k, [0.0] * k
        c1, c2, c3 = [0.0] * k, [0.0] * k, [0.0] * k
        fixed = self.fixed
        K = OPAMP_CLAMP_K
        for j in range(k):
            vcc, vee = self._rails(v, j)
            hi = vcc - self._vsh[j]
            lo = vee + self._vsl[j]
            mid = 0.5 * (hi + lo)
            half_raw = 0.5 * (hi - lo)
            free = half_raw > OPAMP_HALF_MIN
            half = half_raw if free else OPAMP_HALF_MIN
            vp = v[j]
            u = (vp - mid) / half
            t = math.tanh(u)
            s = 1.0 - t * t
            g = self._gout[j]
            io[j] = (mid + half * t) * g
            o1[j] = s * g
            # clamp
            gc = self._gc[j]
            sph, sgh = _softplus(vp - mid - K * half)
            spl, sgl = _softplus(mid - K * half - vp)
            ic[j] = gc * (sph - spl)
            c1[j] = gc * (sgh + sgl)
            if not fixed:
                dhalf = (t - s * u) if free else 0.0
                o2[j] = (0.5 * (1.0 - s) + 0.5 * dhalf) * g
                o3[j] = (0.5 * (1.0 - s) - 0.5 * dhalf) * g
                kd = 0.5 * K if free else 0.0
                # d(mid + K half)/dvcc = 0.5 + kd, /dvee = 0.5 - kd; d(mid - K half): 0.5 - kd, 0.5 + kd
                c2[j] = -gc * (sgh * (0.5 + kd) + sgl * (0.5 - kd))
                c3[j] = -gc * (sgh * (0.5 - kd) + sgl * (0.5 + kd))
        if fixed:
            return io + ic, o1 + c1, None
        return io + ic, o1 + o2 + o3 + c1 + c2 + c3, None

    def eval(self, v):
        i, jv, _ = self.fast_eval(np.asarray(v, dtype=float).tolist(), 0.0, None, None)
        return np.array(i), np.array(jv)

    def limit(self, vnew, vold):
        return np.array(self.fast_limit(np.asarray(vnew, dtype=float).tolist(),
                                        np.asarray(vold, dtype=float).tolist()))

    def info(self, v):
        vl = np.asarray(v, dtype=float).tolist()
        out = []
        for j in range(self.k):
            vcc, vee = self._rails(vl, j)
            hi, lo = vcc - self._vsh[j], vee + self._vsl[j]
            mid = 0.5 * (hi + lo)
            half = max(0.5 * (hi - lo), OPAMP_HALF_MIN)
            u = (vl[j] - mid) / half
            t = math.tanh(u)
            out.append({"vout_internal": mid + half * t, "saturated": bool(abs(t) > 0.95), "drive": u})
        return out

    def fast_limit(self, vnew, vold):
        # Coming out of saturation the limiter slope is ~0 and Newton overshoots to
        # the far side; pull such a step back into the linear range.
        k = self.k
        out = list(vnew)
        for j in range(k):
            vcc, vee = self._rails(vnew, j)
            hi = vcc - self._vsh[j]
            lo = vee + self._vsl[j]
            mid = 0.5 * (hi + lo)
            half = max(0.5 * (hi - lo), OPAMP_HALF_MIN)
            un = (vnew[j] - mid) / half
            if self.fixed:
                uo = (vold[j] - mid) / half
            else:
                vcc_o, vee_o = self._rails(vold, j)
                hi_o, lo_o = vcc_o - self._vsh[j], vee_o + self._vsl[j]
                uo = (vold[j] - 0.5 * (hi_o + lo_o)) / max(0.5 * (hi_o - lo_o), OPAMP_HALF_MIN)
            if abs(un) > 1.0 and abs(uo) > 1.0 and (un > 0) != (uo > 0):
                out[j] = mid + math.copysign(0.5 * half, un)
        return out
