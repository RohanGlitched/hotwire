# Hotwire reference library

Sixteen reference circuits in `library/*.cir`. The agent starts new designs from these, and the demo plays them.
Every file uses the Hotwire netlist format (SPICE plus `*@` annotations, see `docs/SPEC.md`), names parts from the built-in library (`engine/parts_data.py`), and carries measurable `*@ spec` lines.

Every circuit converges and passes all of its specs in **ngspice-42**. The measured numbers below come from `library/index.json`, which `tests/test_library_ngspice.py --write-index` generates. Hotwire's own engine is checked against that file. All specs also pass in the Hotwire engine (`engine/specs.py`).

All pedal circuits run from a single 9 V supply (`VCC vcc 0 DC 9`) with the input source `VIN in 0 DC 0 AC 1`. Guitar level is about 0.1–0.3 V peak. Measurements use the pot positions written in each file.

## Circuits

| id | category | what it does / what to listen for | knobs (taper, default) | measured in ngspice-42 |
|---|---|---|---|---|
| `rc_lowpass` | filter | First-order 10k / 15n low-pass; a gentle treble roll-off | none | fc 1058 Hz; 100 Hz −0.04 dB, 1 kHz −2.76 dB, 5 kHz −13.7 dB, 10 kHz −19.5 dB; phase −45.0° at 1061 Hz |
| `sallen_key_lowpass` | filter | Unity-gain Sallen-Key (TL072, 10k/10k, 22n/10n, Q ≈ 0.74), −40 dB/decade; a darker, smoother cut | none | fc 1118 Hz; 100 Hz −0.08 dB, 5 kHz −26.7 dB, 10 kHz −38.8 dB; THD 0.0001 % at 1 V; 1.85 mA |
| `ce_amplifier` | amp | 2N3904 common emitter, 4.7k collector over 220 Ω unbypassed emitter: about 19× (25 dB), clean then softly asymmetric | none | Vc 5.09 V, Ic 0.83 mA; 24.6 / 24.9 / 24.9 dB at 100 / 1k / 5k Hz; f−3dB low 27 Hz; THD 0.73 % at 50 mV; headroom (1 % THD) 67 mV in; 0.99 mA |
| `jfet_buffer` | utility | J201 source follower, 1 M input; it should sound like nothing (keeps the treble through long cables) | none | Vs 4.74 V, Id 0.22 mA, Vgs −0.24 V; −0.51 dB flat, f−3dB low 2.5 Hz; THD 0.35 % at 2 V peak in; 0.22 mA |
| `jfet_boost` | amp | MPF102 common source with a 100 Ω unbypassed source: about +15 dB of clean-ish boost with warm even harmonics | Level 100k log (1.0) | Vd 4.90 V, Id 0.87 mA; 14.7 dB flat (6 Hz to over 200 kHz); THD 1.19 % at 0.1 V (mostly 2nd); 0.87 mA |
| `diode_hard_clipper` | drive | TL071 non-inverting stage (1 + 1M / (4.7k + Distortion), up to 46 dB) into 10k and two 1N4148 to ground: buzzy, square, raspy | Distortion 1M revlog rheostat (0.7), Level 100k log (0.7) | gain stage 20 dB at default, 45 dB at max (46.6 dB peak at 3.2 kHz); overall 11.5 / 12.0 / 11.7 dB; THD 28 % and 0.22 V peak out for 0.2 V in; 1.85 mA |
| `softclip_overdrive` | drive | RC4558 non-inverting stage with 1N914s in the feedback loop, 4.7k + 47n mid hump (lows cut below about 720 Hz), 1k / 220n treble cut and an active tone stage | Drive 500k log (0.5), Tone 20k lin (0.5), Level 100k log (0.7) | clipping stage 19.5 / 28.7 / 37.5 dB at 1 kHz (Drive 0 / 0.5 / 1; 39.5 dB peak at max); overall 6.5 / 17.3 / 7.2 dB at 100 / 1k / 5k Hz; THD 9.1 % at 0.2 V in, 0.195 V peak out; 2.95 mA |
| `silicon_fuzz` | fuzz | Two direct-coupled 2N3904s with 100k DC feedback from Q2's emitter to Q1's base: thick, sputtery square fuzz | Fuzz 1k lin (1.0), Volume 500k log (0.5) | Q1 Vce 1.45 V (Ic 0.22 mA), Q2 Vc 4.93 V (Ic 0.79 mA); small-signal 61 dB at 1 kHz; THD 110 % at 0.1 V / 500 Hz, 1.46 V peak out; 1.02 mA |
| `germanium_fuzz` | fuzz | The same feedback fuzz with leaky AC128 germanium PNPs, negative-ground (emitters to +9 V): rounder, warmer, touch-sensitive | Fuzz 1k lin (1.0), Volume 500k log (0.5) | Q1 Vce −0.63 V, Vbe −0.17 V; Q2 collector 3.80 V above ground (Ic 0.44 mA); small-signal 63 dB at 1 kHz; THD 95 % at 0.1 V / 500 Hz, 1.49 V peak out; 0.69 mA |
| `passive_tone_stack` | tone | Three-knob bass/middle/treble passive amp tone stack (470p, 22n, 22n, 33k slope, 250k/1M/25k pots) into a 1 M load: the mid scoop at noon | Treble 250k log (0.5), Bass 1M log (0.5), Middle 25k lin (0.5) | −2.9 / −9.2 / −7.3 dB at 100 / 1k / 5k Hz (peak −2.8 dB at 78 Hz); Middle 0→1 moves 1 kHz from −15.9 to −6.5 dB; Bass 0→1 moves 100 Hz from −13.4 to −2.2 dB; Treble 0→1 moves 10 kHz from −10.5 to −0.1 dB |
| `baxandall_tone` | tone | NE5532 inverting Baxandall bass/treble network: flat at noon, shelves either way, mids left alone | Bass 100k lin (0.5), Treble 100k lin (0.5) | flat within 0.02 dB at noon; Bass ±13.8 dB at 100 Hz (±1.0 dB at 1 kHz); Treble ±15.6 dB at 10 kHz (±1.6 dB at 1 kHz); THD 0.00004 % at 0.5 V; 4.45 mA |
| `treble_booster` | drive | One AC128 germanium PNP common emitter with a 4.7n input cap: cuts lows, adds over 30 dB of upper mids and treble to push a loud amp | Boost 10k log collector pot (1.0) | Vc 2.17 V above ground, Ic 0.22 mA; 7.5 / 27.0 / 34.6 dB at 100 / 1k / 5k Hz, f−3dB low 2.5 kHz; 0.23 mA |
| `inductor_wah` | filter | BC549C stage loaded by a 500 mH tank, emitter-follower buffer; the Pedal pot bootstraps the 220n tank cap to sweep the resonance | Pedal 1k lin (0.5, mid-sweep) | at mid: peak 8.3 dB at 662 Hz, −3 dB 549–806 Hz; heel (0): peak 16.5 dB at 479 Hz; toe (1): peak 14.8 dB at 3.0 kHz; 100 Hz −19 dB; 2.04 mA |
| `octave_fuzz` | fuzz | 2N3904 gain stage into a 2N3904 phase splitter; 1N34A germanium diodes full-wave rectify the two opposite phases so the octave-up dominates | Level 100k log (0.9) | splitter Vc 6.84 V / Ve 2.18 V; at 0.2 V / 500 Hz the 2nd harmonic is 14.8× the fundamental (THD 1480 %), 0.27 V peak out; 1.56 mA |
| `power_filter_led` | utility | 1N5817 series polarity protection, 47 Ω + 470 µF ripple filter, red LED indicator, 1.8k standing in for a ~5 mA pedal (VIN is ripple on the 9 V) | none | rail 8.52 V; ripple −23.9 dB at 100 Hz, −43.9 dB at 1 kHz; LED 1.47 mA; Schottky drop 0.19 V; 6.21 mA total |
| `wien_oscillator` | oscillator | TL072 Wien bridge (16k / 10n), 1N4148 amplitude limiting; a transient sanity test | none | 993 Hz, 1.54 V peak, THD 1.26 % (steady state, free-running, 0.2 s run); 1.85 mA |

Supply currents include the op-amps' quiescent current (IQ per amplifier from the part data) and the 10k/10k reference dividers.

### Pot conventions used in the library

* `*@ pot RVx a w b value`: `a` is the counter-clockwise end. Volume and level pots put `a` on ground, so `pos` turns the signal up.
* A pot used as a variable resistor ties the wiper to one end, for example `*@ pot RV1 y y x 1meg` (the active part is w→b) or `*@ pot RVB b1 m m 1meg` (the active part is a→w). That end's half becomes a resistor with both ends on the same node, which is harmless.
* Taper law (SPEC): the fraction of resistance a→w is `pos` (lin), `(11**pos-1)/10` (log), `1-(11**(1-pos)-1)/10` (revlog). Each half is at least 1 Ω.

## Built-in parts (`engine/parts_data.py`)

| part | type | key parameters | source |
|---|---|---|---|
| 1N4148, 1N914 | D | IS 2.52n, N 1.752, RS 0.568, CJO 4p | Fairchild/Philips model (TT, BV dropped) |
| 1N34A | D (Ge) | IS 0.2u, N 1.3, RS 5 (≈0.29 V at 1 mA) | fitted to datasheet curves |
| 1N5817 | D (Schottky) | IS 31.7u, N 1.373, RS 0.051 | onsemi model (LTspice library) |
| 1N4001 | D | IS 14.11n, N 1.984, RS 33.9m | Motorola/onsemi model |
| LED_RED | D | IS 93.2p, N 3.73 (≈1.85 V at 20 mA) | widely circulated red LED model |
| LED_GREEN | D | IS 1.7p, N 3.5 (≈2.1 V at 20 mA) | fitted to GaP green datasheet |
| BAT41 | D (Schottky) | IS 0.83n, N 1.05, RS 2 (≈0.38 V at 1 mA) | fitted to ST datasheet |
| 2N3904 | NPN | IS 6.734f, BF 134*, VAF 74 | Fairchild model |
| 2N2222 | NPN | IS 14.34f, BF 148*, VAF 74 | Fairchild model |
| BC549C | NPN | IS 23.9f, BF 520 | Philips/NXP model, datasheet hFE |
| 2N5088 | NPN | IS 5.911f, BF 628*, VAF 62 | Fairchild model |
| MPSA18 | NPN | IS 33.6f, BF 900, VAF 100 | Fairchild model, datasheet hFE |
| 2N3906 | PNP | IS 1.41f, BF 180.7, VAF 18.7 | Fairchild model |
| BC559C | PNP | IS 10.2f, BF 500 | Philips/NXP family, datasheet hFE |
| AC128 | PNP (Ge) | IS 0.3u, BF 90, RB 50 (Vbe ≈ 0.21 V at 1 mA, leaky) | fitted to datasheet and vintage samples |
| J201 | NJF | VTO −0.7, BETA 1.0m (IDSS 0.49 mA) | fitted to datasheet |
| 2N5457 | NJF | VTO −1.372, BETA 1.125m (IDSS 2.1 mA) | Fairchild model (LTspice library) |
| MPF102 | NJF | VTO −3.0, BETA 0.6m (IDSS 5.4 mA) | fitted to datasheet |
| TL072 / TL071 | OPAMP | A0 2e5, GBW 3 MHz, RIN 1e12, ROUT 75, swing 1.5/1.5 V, IQ 1.4 mA | TI datasheet typicals |
| NE5532 | OPAMP | A0 1e5, GBW 10 MHz, RIN 300k, ROUT 30, swing 2/2 V, IQ 4 mA | TI datasheet typicals |
| LM358 | OPAMP | A0 1e5, GBW 0.7 MHz, swing 1.5 / 0.02 V, IQ 0.35 mA | TI datasheet typicals |
| RC4558 | OPAMP | A0 3e5, GBW 3 MHz, RIN 5M, swing 1/1 V, IQ 1.25 mA | TI datasheet typicals |
| OPAMP_IDEAL | OPAMP | A0 1e6, GBW 1 GHz, rail to rail, no IQ | idealisation |

\* Hotwire simulates the Ebers-Moll subset of Gummel-Poon. For these parts the published model's ISE/IKF terms change the current gain a lot, so BF is set to the full model's effective hFE at Ic = 1 mA (2N3904: 416 → 134, 2N2222: 256 → 148, 2N5088: 1122 → 628). Each part's `source` field records the original numbers.

## How the ngspice ground truth is produced

`tests/test_library_ngspice.py` has its own small netlist reader, independent of `engine/netlist.py`. For each circuit it writes decks to `tests/ngspice_library/`:

* `*@ pot` lines become `RVx_A` / `RVx_B` resistors.
* Parts become `.model` lines from `PARTS`.
* Each op-amp becomes this macromodel subcircuit:
  * input: `RIN` across the two inputs;
  * gain stage: a VCCS with gm = A0/1 MΩ into 1 MΩ ∥ C, giving one pole at GBW/A0;
  * clamp: a 1000 S softplus clamp (10 mV corner), referenced to ground, holds the gain node between `VEE+VSAT_LO` and `VCC−VSAT_HI`, so the stage cannot wind up;
  * output: a unity VCVS behind `ROUT`;
  * supply: a constant `IQ` current source from VCC to VEE.

Measurements follow the same rules as `engine/specs.py`:

* **AC:** 1 Hz–200 kHz at 100 points per decade, log-f interpolation. The −3 dB points are measured from the sweep maximum.
* **Transient:** a 5 ms fade-in, settling for max(60 ms, 8 cycles), then a window of max(4, 0.02·f) cycles. THD covers H2..H15, and the peak is taken after removing the mean.
* **Headroom:** geometric bisection from 5 mV to 5 V.
* **Simulator settings:** ngspice runs at 27 °C with the trapezoidal method and a maximum step of T/200.

The oscillator runs free for 0.2 s with `uic`; the capacitor `ic=` values give it the kick to start.

```
python tests/test_library_ngspice.py              # report, no files written
python tests/test_library_ngspice.py --write-index # rewrite library/index.json and the decks
pytest tests/test_library_ngspice.py               # assert every spec passes in ngspice
```

The test finds ngspice through `$HOTWIRE_NGSPICE`, a native `ngspice` on PATH, or (on Windows) WSL at `~/ngs/root/usr/bin/ngspice`. If none of these is found, the test is skipped.

## Monte Carlo yield (Hotwire, seed 1)

Each circuit was built 200 times with 5% resistors and 10% capacitors (or the `*@ tol` written in the file), ±40% transistor gain, ±50% saturation current and ±0.5 V JFET pinch-off. Circuits with time-domain specs (THD, guitar volume) need a transient run per build, so they get 60 builds or as many as fit in 60 s. Intervals are 95% Wilson intervals.

| circuit | specs | yield | 95% interval | builds |
|---|---:|---:|---|---:|
| `baxandall_tone` | 6/6 | 98.3% | 91.1–99.7% | 60 |
| `ce_amplifier` | 8/8 | 98% | 89.3–99.6% | 49 |
| `diode_hard_clipper` | 6/6 | 98.3% | 91.1–99.7% | 60 |
| `germanium_fuzz` | 6/6 | 100% | 94–100% | 60 |
| `inductor_wah` | 8/8 | 100% | 98.1–100% | 200 |
| `jfet_boost` | 7/7 | 96.7% | 88.6–99.1% | 60 |
| `jfet_buffer` | 7/7 | 100% | 94–100% | 60 |
| `octave_fuzz` | 6/6 | 100% | 94–100% | 60 |
| `passive_tone_stack` | 3/3 | 100% | 98.1–100% | 200 |
| `power_filter_led` | 6/6 | 100% | 98.1–100% | 200 |
| `rc_lowpass` | 4/4 | 100% | 98.1–100% | 200 |
| `sallen_key_lowpass` | 6/6 | 100% | 94–100% | 60 |
| `silicon_fuzz` | 8/8 | 98.3% | 91.1–99.7% | 60 |
| `softclip_overdrive` | 7/7 | 100% | 94–100% | 60 |
| `treble_booster` | 5/5 | 96.5% | 93–98.3% | 200 |
| `wien_oscillator` | 2/2 | 99.5% | 97.2–99.9% | 200 |
| `mosfet_boost` | 7/7 | 100% | 94–100% | 60 |
