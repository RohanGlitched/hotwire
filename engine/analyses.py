"""Public analyses: operating point, AC sweep, transient and audio processing.

All functions take a parsed :class:`~engine.netlist.Circuit`.  They raise
:class:`~engine.netlist.NetlistError` for problems in the netlist and
:class:`~engine.mna.SimulationError` when the numerics cannot be trusted;
both carry a plain-English message.

Sign conventions follow SPICE: the current of a voltage source, inductor or
VCVS is the current flowing from its + node through the element to its -
node, so a battery that delivers power shows a negative current.  Device
currents (``ic``, ``ib``, ``id``) flow *into* the named terminal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mna import Network, SimulationError, Source, Transient
from .netlist import Circuit, NetlistError

__all__ = ["OpResult", "AcResult", "TranResult", "op", "ac", "tran", "process_audio",
           "sine_response", "thd_at", "headroom_at"]


# --------------------------------------------------------------------------- results

@dataclass
class OpResult:
    """DC operating point.

    ``v``: node -> volts.  ``i``: element -> amps (V/L/E branch current, SPICE
    sign; R/S n1->n2; D anode->cathode; Q collector; J drain; X output current
    into the output node).  ``power``: element -> watts dissipated (sources
    delivering power are negative).  ``devices``: per-device details: diodes
    ``id, vd``; BJTs ``ic, ib, ie, vbe, vce, vbc, region, gm, beta``; JFETs
    ``id, ig, vgs, vds, region, gm``; MOSFETs ``id, ib, vgs, vds, vbs, region``
    (cutoff / triode / saturation) ``, gm, gds, gmbs, vth, power``; op-amps
    ``vout, iout, saturated, vd``.
    """

    v: dict
    i: dict
    power: dict
    devices: dict
    converged: bool
    method: str
    iterations: int = 0
    x: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {"converged": self.converged, "method": self.method, "v": self.v,
                "devices": self.devices}


@dataclass
class AcResult:
    """Small-signal sweep.  ``h`` is V(out) / input-source AC value at each ``f``."""

    f: np.ndarray
    h: np.ndarray
    input: str | None = None
    output: str | None = None
    v: dict = field(default_factory=dict, repr=False)   # node -> complex response

    @property
    def gain_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(np.abs(self.h), 1e-30))

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(np.unwrap(np.angle(self.h)))


@dataclass
class TranResult:
    """Transient waveforms: ``t`` seconds, ``v[node]`` volts, ``i[source]`` amps."""

    t: np.ndarray
    v: dict
    i: dict = field(default_factory=dict)
    converged: bool = True
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers

def _network(c: Circuit, input_name: str | None = None) -> Network:
    if not isinstance(c, Circuit):
        raise NetlistError(0, "expected a parsed Circuit (use engine.parse(text) first)")
    return Network(c, input_name)


def _dc(net: Network, u: np.ndarray) -> tuple[np.ndarray, str, int]:
    x, ok, method, iters = net.solve_dc(u)
    if not ok:
        raise SimulationError(
            "the DC operating point did not converge (tried Newton, gmin stepping and source "
            "stepping). Check for nodes with no DC path to ground, transistors with no base "
            "bias, or supplies wired backwards")
    return x, method, iters


def _initial_point(net: Network, u0: np.ndarray) -> tuple[np.ndarray, str]:
    """Transient starting point: DC with sources at t = 0 and capacitors held at their ic=."""
    net.force_ic = net.has_ic
    try:
        x, method, _ = _dc(net, u0)
    finally:
        net.force_ic = False
    return x, method


def op(c: Circuit) -> OpResult:
    """DC operating point (capacitors open, inductors shorted, sources at DC value).

    Non-convergence is reported through ``converged=False`` and ``method="failed"``
    with the best estimate found, rather than raised.
    """
    net = _network(c)
    u = net.u_dc()
    x, ok, method, iters = net.solve_dc(u)
    volts, cur, pw, devs = net.report(x, u)
    return OpResult(volts, cur, pw, devs, ok, method, iters, x)


def _resolve_io(c: Circuit, input: str | None, output: str | None, need_input: bool = True):
    inp = (input or c.input)
    out = (output or c.output)
    if need_input and not inp:
        raise NetlistError(0, "no input source: add '*@ in VIN' (naming the signal source) to the netlist")
    if not out:
        raise NetlistError(0, "no output node: add '*@ out <node>' to the netlist")
    out = out.lower()
    out = "0" if out in ("gnd", "0") else out
    if out == "0":
        raise NetlistError(0, "the output node can't be ground")
    if out not in c.nodes():
        raise NetlistError(0, f"output node '{out}' isn't in the circuit")
    return (inp.upper() if inp else None), out


def ac(c: Circuit, fstart: float = 10.0, fstop: float = 100e3, points_per_decade: int = 40,
       input: str | None = None, output: str | None = None) -> AcResult:
    """Small-signal AC sweep, linearised at the DC operating point.

    The input source (``*@ in`` or ``input``) gets AC magnitude 1 and every other
    source AC 0, so ``h`` is the transfer function out/in.  Without an input
    source, the sources' own ``AC`` values from the netlist drive the circuit.
    """
    if not (fstart > 0 and fstop > fstart):
        raise SimulationError(f"AC sweep needs 0 < fstart < fstop (got {fstart}, {fstop})")
    inp, out = _resolve_io(c, input, output, need_input=False)
    net = _network(c, inp)
    x, _, _ = _dc(net, net.u_dc())
    ndec = math.log10(fstop / fstart)
    npts = max(2, int(round(ndec * points_per_decade)) + 1)
    f = np.logspace(math.log10(fstart), math.log10(fstop), npts)
    if inp:
        rhs = net.Bsrc[:, net.src_index[inp]].astype(complex)
    else:
        acv = np.array([s.ac for s in net.sources], dtype=complex)
        if not np.any(acv):
            raise NetlistError(0, "no input for the AC sweep: add '*@ in VIN' or 'AC 1' on a source")
        rhs = net.Bsrc.astype(complex) @ acv
    try:
        X = net.ac_solve(x, f, rhs)
    except np.linalg.LinAlgError as exc:
        raise SimulationError("the AC matrix is singular: look for a floating sub-circuit or a loop "
                              "of voltage sources and inductors") from exc
    h = X[:, net.index[out]]
    vdict = {name: X[:, i] for i, name in enumerate(net.node_names)}
    return AcResult(f, h, inp, out, vdict)


def _drive(net: Network, inp: str | None, wave: np.ndarray | None, dt: float) -> None:
    if wave is None or inp is None:
        return
    s: Source = net.sources[net.src_index[inp]]
    s.wave = np.asarray(wave, dtype=float)
    s.wave_dt = dt


def tran(c: Circuit, tstop: float, tstep: float, input_wave: np.ndarray | None = None,
         input: str | None = None, nodes: list | None = None, *, max_halvings: int = 8,
         vtol: float = 1e-7) -> TranResult:
    """Transient analysis with the trapezoidal rule at a fixed step.

    The run starts from the DC operating point with every source at its t = 0
    value and every capacitor with ``ic=`` held at that voltage (then released).  ``input_wave`` (volts, sampled every ``tstep``) replaces the input
    source's waveform and is added to its DC value; otherwise SIN/PULSE
    sources follow their netlist definition.  A step whose Newton iteration
    fails is retried as two half steps, recursively up to ``max_halvings``.
    """
    if not (tstep > 0 and tstop > 0):
        raise SimulationError(f"transient needs positive tstop and tstep (got {tstop}, {tstep})")
    nt = int(math.floor(tstop / tstep + 1e-9)) + 1
    if nt > 5_000_000:
        raise SimulationError(f"{nt} time steps is too many; use a larger tstep or a shorter tstop")
    inp = (input or c.input)
    inp = inp.upper() if inp else None
    net = _network(c, inp)
    _drive(net, inp, input_wave, tstep)

    def src(t: np.ndarray) -> np.ndarray:
        return net.u_at(t, tstep, tstop)

    u0 = src(np.zeros(1))[:, 0]
    x_op, method = _initial_point(net, u0)
    if nodes is None:
        names = list(net.node_names)
    else:
        names = []
        for n in nodes:
            key = "0" if str(n).lower() in ("0", "gnd") else str(n).lower()
            if key != "0" and key not in net.index:
                raise NetlistError(0, f"tran: no node named '{n}'")
            names.append(key)
    rows = [net.index[n] if n != "0" else -1 for n in names]
    brs = [(name, r.br) for name, r in net.recs.items() if r.kind == "V"] if nodes is None else []
    out_rows = [r for r in rows if r >= 0] + [b for _, b in brs]
    eng = Transient(net, x_op, out_rows, max_halvings=max_halvings, vtol=vtol)
    Y = eng.run(tstep, nt, src, x_op)
    t = np.arange(nt) * tstep
    v = {}
    col = 0
    for n, r in zip(names, rows):
        if r < 0:
            v[n] = np.zeros(nt)
        else:
            v[n] = Y[:, col]
            col += 1
    i = {name: Y[:, col + j] for j, (name, _) in enumerate(brs)}
    stats = {"steps": nt - 1, "newton_iterations": eng.iterations, "halved_steps": eng.substeps,
             "failed_steps": eng.failed_steps, "op_method": method}
    return TranResult(t, v, i, eng.failed_steps == 0, stats)


def process_audio(c: Circuit, samples: np.ndarray, sr: int, input: str | None = None,
                  output: str | None = None, input_scale: float = 1.0, *,
                  stats: dict | None = None) -> np.ndarray:
    """Run audio through the circuit.

    ``samples`` (volts) drive the input source on top of its DC value, one
    trapezoidal step per sample at rate ``sr``.  Returns the output node
    voltage minus its DC operating value (what a coupling cap would pass),
    the same length as ``samples``.  Pass a dict as ``stats`` to receive the
    Newton/step counters.
    """
    x_in = np.asarray(samples, dtype=float).reshape(-1)
    if not np.all(np.isfinite(x_in)):
        raise SimulationError("the audio contains NaN or infinite samples")
    if sr <= 0:
        raise SimulationError(f"sample rate must be positive (got {sr})")
    inp, out = _resolve_io(c, input, output)
    net = _network(c, inp)
    n = len(x_in)
    if n == 0:
        return np.zeros(0)
    h = 1.0 / sr
    wave = x_in * float(input_scale)
    _drive(net, inp, wave, h)

    def src(t: np.ndarray) -> np.ndarray:
        return net.u_at(t, h, n * h)

    u0 = src(np.zeros(1))[:, 0]
    x_op, method = _initial_point(net, u0)
    orow = net.index[out]
    # the plain DC operating point (input at its DC value, no ic=) is the reference
    base = x_op
    if wave[0] != 0.0 or net.has_ic:
        base, _, _ = _dc(net, net.u_dc())
    # Newton accepts a step once |dv| < 50 uV: the update is applied with its
    # linear correction, so the remaining error is ~dv^2/(2 n Vt), about 50 nV.
    eng = Transient(net, x_op, [orow], vtol=5e-5, reltol=1e-6)
    Y = eng.run(h, n, src, x_op)       # row k is the response at t = k*h, driven by samples[k]
    y = Y[:, 0] - base[orow]
    if stats is not None:
        stats.update(steps=n, newton_iterations=eng.iterations, halved_steps=eng.substeps,
                     failed_steps=eng.failed_steps, op_method=method)
    return y


# --------------------------------------------------------------------------- large-signal helpers

def sine_response(c: Circuit, freq: float, vin_peak: float, *, cycles: int = 10,
                  settle_cycles: int = 20, steps_per_cycle: int = 256,
                  input: str | None = None, output: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state output for a sine input: (t, vout) over exactly ``cycles`` periods.

    The step is chosen so each period is an integer number of samples, which
    makes the FFT bins line up with the harmonics.
    """
    sr = freq * steps_per_cycle
    n_settle = settle_cycles * steps_per_cycle
    n_meas = cycles * steps_per_cycle
    k = np.arange(n_settle + n_meas)
    x = vin_peak * np.sin(2 * math.pi * k / steps_per_cycle)
    y = process_audio(c, x, sr, input=input, output=output)
    t = k[n_settle:] / sr
    return t, y[n_settle:]


def thd_at(c: Circuit, freq: float, vin_peak: float, **kw) -> float:
    """THD (%) at the output for a sine of ``vin_peak`` volts at ``freq`` Hz."""
    from .measure import thd
    cycles = kw.pop("cycles", 10)
    _, y = sine_response(c, freq, vin_peak, cycles=cycles, **kw)
    return thd(y, cycles=cycles)


def headroom_at(c: Circuit, freq: float = 1000.0, limit_pct: float = 1.0,
                lo: float = 1e-3, hi: float = 10.0) -> float:
    """Largest input peak (V) whose output THD stays at or under ``limit_pct``."""
    from .measure import headroom
    return headroom(lambda a: thd_at(c, freq, a), lo=lo, hi=hi, limit_pct=limit_pct)
