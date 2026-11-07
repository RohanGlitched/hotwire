# Hotwire simulator: measured accuracy and speed

Hotwire's simulator is pure Python and numpy. This page compares it with **ngspice-42**,
the reference open-source SPICE, and records how fast it runs. Every number below comes
from the scripts listed at the end, run on 7 Nov 2026.

## Targets (from docs/SPEC.md)

| quantity | target | worst measured |
|---|---|---|
| DC node voltages | within 1% or 1 mV | 0.35 µV (0.0008% of the allowance) |
| AC gain | within 0.1 dB | 5.6e-7 dB |
| -3 dB frequency | within 1% | 7.1e-6 % |
| Transient RMS error (clippers, fuzz, switches) | within 2% of signal RMS | 1.75% (P-MOSFET switch), 1.29% (LED driver), 0.12% (fuzz) |
| 1 s of 48 kHz audio through a two-transistor fuzz | under 10 s in CPython | 3.6 s |

All 19 cross-check circuits meet the DC, AC and transient targets, including the four MOSFET circuits (common-source amplifier
with body effect, source follower, CMOS inverter, P-channel high-side switch). At audio rate (48 kHz, one step per sample)
one circuit misses the 2% target: the op-amp inverting clipper driven into its rails reads 2.41% (0.16% at 96 kHz); see below.

## How the cross-check works

`tests/ngspice_util.py` turns each parsed Hotwire circuit into an ngspice deck, so both
simulators see exactly the same parameters:

- Every model is written as an explicit `.model` line. The BJT models use the full
  Gummel-Poon DC terms Hotwire implements (IKF, ISE/NE, VAF, RB/RC), plus CJE/CJC and TF/TR.
- MOSFETs get one level-1 `.model` per instance with `W`, `L` (and `M`) on the element line;
  Hotwire's CGS/CGD in farads are written as CGSO/CGDO per metre of W, so ngspice's own
  overlap-capacitance scaling gives the same farads.
- Op-amps become a subcircuit holding the same macromodel (`engine.export.opamp_subckt_lines`,
  shared with `circuit_export`): RIN, a VCCS `A0/RP` into `RP || CP` (one pole at GBW/A0), the
  anti-windup clamp as a softplus B-source current out of the pole node, a B-source `tanh`
  limiter between `VEE+VSAT_LO` and `VCC-VSAT_HI`, ROUT and IQ.
- `.options rshunt=1e12` reproduces Hotwire's 1e-12 S node-to-ground shunt.
- ngspice runs `op`, `ac dec` and `tran` (step = the Hotwire step, with `tmax` set equal to it),
  then `linearize` and `wrdata`. Results are cached in `tests/_ngspice/<circuit>/` together with
  the deck, keyed by a hash of the deck.

The metrics:

- **DC**: worst node error. It is also expressed as a fraction of max(1% of |V|, 1 mV).
- **AC**: worst |dB difference| at every point within 60 dB of the peak, so the deep stop band
  (−100 dB and below, where both simulators are at numerical noise) doesn't count. The -3 dB
  corners are the half-power points around the peak, compared as a ratio.
- **Transient**: RMS of (Hotwire − ngspice) divided by the AC RMS of the ngspice waveform, on
  the output node over the whole run. Hotwire runs at the fixed step shown in the table.
  ngspice uses its own adaptive steps, capped at that step size.

## Cross-check table (Hotwire vs ngspice-42)

| circuit | DC max error | DC worst node | AC max error | -3 dB error | tran RMS error | Newton/step |
|---|---:|---|---:|---:|---:|---:|
| `diode_clipper_sym` (1N4148 pair, 2 V sine) | 1.7e-19 µV | out | 2.1e-11 dB | 2.3e-10 % | 0.026 % | 2.28 |
| `diode_clipper_asym` (2 Si one way, 1 Ge the other) | 2.4e-13 µV | m | 4.8e-11 dB | 3.1e-11 % | 0.011 % | 2.10 |
| `ce_amp` (2N3904, full GP terms) | 0.024 µV | c | 9.6e-10 dB | 5.7e-09 % | 0.010 % | 2.29 |
| `fuzz_two_transistor` (fuzz-face topology, BC549) | 0.023 µV | o1 | 4.9e-08 dB | 4.7e-07 % | 0.117 % | 2.28 |
| `jfet_booster` (J201 common source) | 0.0004 µV | g | 4.3e-09 dB | 8.9e-09 % | 0.011 % | 2.00 |
| `sallen_key_lp` (op-amp macromodel) | 0 µV | out | 3.3e-11 dB | 1.1e-12 % | 0.009 % | 2.00 |
| `opamp_soft_clipper` (diodes in the feedback loop) | 2.1e-15 µV | out | 2.7e-09 dB | 2.9e-08 % | 0.038 % | 2.23 |
| `led_driver` (pulse-switched NPN + LED) | 3.0e-07 µV | c | n/a | n/a | 1.289 % | 2.92 |
| `germanium_pnp_stage` (AC128-style PNP, -9 V) | 0.012 µV | e | 1.5e-07 dB | 1.8e-06 % | 0.015 % | 2.00 |
| `rlc_bandpass` (parallel LC tank, Q = 3.2) | 0 µV | m | 5.4e-12 dB | 6.5e-12 % | 0.010 % | linear |
| `baxandall_tone_stack` (passive, two log pots) | 0 µV | n4 | 7.6e-12 dB | 5.7e-11 % | 0.001 % | linear |
| `opamp_inverting_clip` (gain -10 into the rails) | 0 µV | out | 3.2e-11 dB | 3.9e-10 % | 0.011 % | 3.38 |
| `push_pull_follower` (class B, crossover) | 1.7e-13 µV | out | n/a | n/a | 0.005 % | 2.12 |
| `half_wave_rectifier` (100 µF reservoir) | 9.1e-25 µV | out | n/a | n/a | 0.001 % | 2.00 |
| `npn_pnp_cascade` (NPN follower into PNP CE) | 0.35 µV | c2 | 5.6e-07 dB | 7.1e-06 % | 0.035 % | 3.11 |
| `mosfet_cs_amp` (level-1 NMOS, bulk to ground: body effect, RD/RS, CBD/CBS) | 1.5e-06 µV | d | 1.2e-11 dB | 5.0e-12 % | 0.010 % | 2.00 |
| `mosfet_source_follower` (BS170, bulk on source) | 1.8e-07 µV | s | 4.5e-13 dB | n/a | 0.010 % | 2.00 |
| `cmos_inverter` (NMOS + PMOS, 1 nF load, 0.2 µs step) | 4.4e-07 µV | out | n/a | n/a | 0.011 % | 1.75 |
| `pmos_high_side_switch` (IRF9510 + 2N3904 level shifter) | 0.12 µV | g | n/a | n/a | 1.749 % | 2.68 |

"n/a" marks circuits with no AC sweep: switching and class-B stages, whose small-signal
response at idle isn't meaningful. "linear" marks circuits with no nonlinear devices, which
need no Newton iterations.

Hotwire implements the same level-1 equations and defaults as ngspice. That is why the DC
and AC agreement is at round-off level.

Transient differences come from the time step, not the device models. The two largest,
`pmos_high_side_switch` (1.75%) and `led_driver` (1.29%), are hard-switching edges, which
ngspice resolves with breakpoints and smaller adaptive steps. Shrinking Hotwire's fixed
step shows this:

| circuit | step | RMS error |
|---|---:|---:|
| `led_driver` | 2 µs | 1.29 % |
| `led_driver` | 0.5 µs | 0.22 % |
| `led_driver` | 0.125 µs | 0.05 % |
| `pmos_high_side_switch` | 0.5 µs | 1.75 % |
| `pmos_high_side_switch` | 0.25 µs | 1.39 % |
| `pmos_high_side_switch` | 0.1 µs | 0.10 % |

`opamp_inverting_clip` used to be the other large one (0.94% at 2 µs). The op-amp's pole node now
has an anti-windup clamp, so the output leaves the rail within microseconds instead of
unwinding a kilovolt-sized integrator state, and the error fell to 0.011% at the same step.

### Audio rate: `process_audio` at 48 kHz

`process_audio` takes one trapezoidal step per audio sample (20.8 µs). The table compares
it with the ngspice transient of the same sine input, after subtracting the DC operating
point.

| circuit | RMS error vs ngspice |
|---|---:|
| `diode_clipper_sym` | 0.51 % |
| `diode_clipper_asym` | 0.20 % |
| `fuzz_two_transistor` | 1.44 % |
| `opamp_soft_clipper` | 0.59 % |
| `opamp_inverting_clip` | 2.41 % (0.16 % at 96 kHz) |
| `mosfet_cs_amp` | 0.03 % |
| `mosfet_source_follower` | 0.02 % |
| `jfet_booster` | 0.02 % |
| `ce_amp` | 0.03 % |

All but one stay inside the 2% transient target at the coarse audio step. The exception,
`opamp_inverting_clip` (gain -10 driven 10x past the rails, 1 kHz), has rail-to-rail edges that
are two samples long at 48 kHz; at 96 kHz it is 0.16%. The fuzz is at 1.44% for the same
reason. Run at a higher sample rate when the exact edge shape matters (the THD specs use 96 kHz).

## Analytic checks (`tests/test_engine_analytic.py`)

| test | reference | tolerance |
|---|---|---|
| resistive divider, source current, resistor power | Ohm's law | 1e-6 |
| RC low-pass corner, gain and phase at fc | 1/(2πRC), −3.01 dB, −45° | 0.1%, 0.01 dB, 0.2° |
| RC step response at t = τ | 1 − e⁻¹ | 2 mV |
| common-emitter bias point | hand iteration of Ebers-Moll with the base Thevenin network | 0.2% |
| BJT saturation / cut-off, PNP mirrors NPN | region logic, sign symmetry | exact |
| inverting and non-inverting op-amp gain | A_cl = G/(1 + 1/(A0·β)), bandwidth GBW·β, clip at the rails | 1e-4, 3%, 50 mV |
| parallel LC resonance: peak, Q | 1/(2π√LC), R√(C/L) | 0.2%, 1% |
| LC ringing frequency in a transient | damped ω₀√(1 − 1/4Q²) | 0.2% |
| diode forward drop | n·Vt·ln(I/IS + 1) | 1 µV |
| JFET saturation current | BETA·VTO² | 1e-6 |
| MOSFET saturation and triode current, gm (`tests/test_engine_mosfet.py`) | KP/2·(VGS−VTO)²·(1+λVDS), KP·VDS·(VGS−VTO−VDS/2)·(1+λVDS) | 1e-12 relative |
| MOSFET body effect: threshold and gmbs | VTO + γ(√(φ+VSB) − √φ), gm·γ/(2√(φ+VSB)) | 1e-12 V, 1e-9 |
| MOSFET W/L and M scaling, PMOS mirrors NMOS, reverse mode | β = KP·M·W/Leff, sign symmetry, drain/source swap | 1e-8 |
| gate overlap capacitance pole (1 MΩ into CGS = 100 pF) | 1/(2πRC) | 0.2% |
| op-amp anti-windup (`tests/test_opamp_windup.py`): Schmitt trigger (±1.23 V thresholds, 2.5 V input) | toggles between both rails | ±13.48 V reached |
| op-amp anti-windup: gain-101 rail clipper, output zero-crossing lag | under 50 µs at 1 V (was 213 µs), under 20 µs at 0.2 V (was 36 µs) | 12.0 µs, 5.4 µs |
| `process_audio` on a divider | exactly 0.5 × input, zero for silence | 1e-9 |
| Wien-bridge oscillator started by `C ic=0.1` | sustained oscillation | > 10 V p-p |
| THD of a known two-tone signal | 10.00% | 1e-6 |

## Speed (`bench/bench_audio.py`)

The benchmark pushes 1 s of 48 kHz audio through each circuit. The input is the built-in
guitar riff at 0.2 V peak, which drives the fuzzes hard into clipping: the worst case for
Newton. The best of 3 runs is kept.

Machine: Intel Core i5-11400H laptop, Windows 11, CPython 3.12.10, numpy 2.4.6. Other
heavy processes were running at the same time (about 85% CPU load), so an idle machine
will be faster.

| circuit | s per 1 s of audio | Newton iterations / sample |
|---|---:|---:|
| two-transistor fuzz (`bench/fuzz2.cir`, 2× BC549C) | **3.59** | 3.25 |
| `fuzz_two_transistor` (cross-check model with junction caps) | 3.74 | 3.17 |
| `diode_clipper_sym` | 0.92 | 1.73 |
| `opamp_soft_clipper` | 2.23 | 2.80 |
| `jfet_booster` | 1.37 | 2.05 |
| `ce_amp` | 1.79 | 2.69 |
| `baxandall_tone_stack` (linear) | 0.19 | 0 |
| library `silicon_fuzz` | 2.96 | 2.71 |
| library `germanium_fuzz` | 3.80 | 3.20 |
| library `octave_fuzz` (2 BJTs + diode rectifier, 6 ports) | 6.51 | 2.34 |
| library `softclip_overdrive` | 3.31 | 2.72 |
| library `treble_booster` | 1.93 | 2.47 |
| library `ce_amplifier` (1 BJT) | 1.34 | 2.04 |
| library `jfet_boost` (1 JFET) | 1.35 | 1.95 |
| library `mosfet_boost` (1 BS170, bulk on source: 2 ports) | 1.06 | 1.91 |
| `ce_amp` cross-check (1 BJT, full Gummel-Poon) | 1.69 | 2.69 |
| `mosfet_cs_amp` cross-check (1 MOSFET, separate bulk: 3 ports) | 1.89 | 2.51 |
| `mosfet_source_follower` cross-check | 0.97 | 1.74 |

The MOSFET rows (measured 7 Nov 2026, same machine) show the per-device cost: a MOSFET with its
bulk on its source, the normal discrete part, is two Newton ports and runs faster than a BJT stage;
one with a separate bulk (body effect) needs a third port and runs about 12% slower than the full
Gummel-Poon BJT stage.

What makes it fast:

1. **Fixed-step reduction.** The step is fixed, so the linear part of the circuit (R, C, L,
   sources, the op-amp's linear stages) is folded into one matrix. The engine inverts it
   once per step size. The Newton iteration then runs only over the nonlinear *ports*:
   2 per BJT or JFET, 1 per diode, 1 per op-amp limiter. A two-transistor fuzz is a 4×4
   nonlinear system instead of a 20×20 MNA matrix.
2. **Linearised around the operating point.** The reduced matrix already contains each
   device's Jacobian at the DC operating point. Small signals converge almost at once, and
   the matrix stays well conditioned even where a node connects only to transistor pins.
3. **Plain-float inner loop.** Device equations, the residual, the Jacobian and the solve
   (closed-form 2×2 and 3×3, unrolled 4×4 block elimination) work on Python floats, because
   numpy's ~0.6 µs per call costs more than the arithmetic on 4-element arrays. numpy is
   used per step for the state update, and LAPACK for systems with 5 or more ports.
4. **Warm start, step-size predictor and SPICE limiting.** Newton starts from a
   linear extrapolation of the last two samples. A port that just snapped on or off is not
   extrapolated. `pnjlim`/`fetlim` keep junction steps safe. A step is accepted when
   |Δv| < 50 µV. Because the final update is applied with its linear current correction,
   the remaining error is about Δv²/(2nVt) ≈ 50 nV. Against a 1e-8 V tolerance, the fuzz
   output differs by −87 dBFS.
5. **Recovery.** A sample whose Newton iteration doesn't converge is retried as two half
   steps, recursively down to h/256. The state is stored physically (capacitor voltage and
   current, junction charge and charge current), so sub-steps need no conversion.

## Reproduce

```
python -m pytest tests/test_engine_analytic.py tests/test_engine_xcheck.py -q
python bench/validate.py            # this page's tables (cached ngspice results)
python bench/validate.py --force    # re-run ngspice-42 in WSL for every circuit
python bench/bench_audio.py         # speed table
```

ngspice-42 runs inside WSL:
`wsl -e bash -lc "LD_LIBRARY_PATH=~/ngs/root/usr/lib/x86_64-linux-gnu ~/ngs/root/usr/bin/ngspice -b deck.cir"`.
The cached results in `tests/_ngspice/` let the cross-check tests run without WSL.

## Known limitations

- **Temperature**: every device runs at 27 °C. There are no XTB/EG/XTI temperature terms.
- **BJT**: Gummel-Poon DC terms are complete except for the IRB/RBM current-dependent base
  resistance and excess phase (PTF). XTF is ignored. All of CJC sits on the internal base
  (XCJC = 1). There is no substrate capacitance.
- **Diode**: reverse breakdown uses ngspice's exponential knee with the IBV fit done in
  closed form, without ngspice's iteration. It agrees at −BV and is approximate a few
  hundred mV past the knee. There are no sidewall or recombination terms.
- **MOSFET**: SPICE level 1 only (other LEVELs and VDMOS are refused with a message). No Meyer
  gate capacitances (TOX only sets KP), no subthreshold conduction, no NSUB-derived parameters,
  no sidewall/area junctions (AD/AS/PD/PS are ignored). The built-in IRF510/IRF9510 are simplified
  level-1 fits, good near their fitted current, not vendor subcircuits.
- **Op-amp macromodel**: there is one pole, and slew rate isn't modelled. In saturation the pole
  node is clamped at mid ± 4·half of the output window, so recovery is fast but the output reaches
  99.93% of the rail-to-rail window rather than 100%. The output stage
  draws its load current from ground rather than from the rails, so supply current = IQ
  plus resistor and transistor currents. Common-mode input range and input bias currents
  aren't modelled.
- **Time step**: `tran` uses a fixed step, with step halving only when Newton fails. It has
  no local-truncation-error control, so hard switching edges need a step small enough for
  them. The table above shows the trade-off.
- **Capacitor `ic=`**: the cap is held at its `ic` voltage during the initial operating
  point and released at t = 0+, like SPICE `.ic`. The `UIC` mode, where every node is
  zero, isn't offered.
- **Audio aliasing**: `process_audio` runs at the audio rate with no oversampling. Hard
  clippers alias at 48 kHz, as a non-oversampled real-time model would.
