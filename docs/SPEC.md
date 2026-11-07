# Hotwire: design specification

The contracts Hotwire's modules are written against: the netlist format, the engine API, the specs, the tools and the Desktop half. Where this page and the code disagree, the code and its tests win; this page is kept as the design reference.

**Hermes designs real analog circuits from plain English, proves them in simulation, and lets you hear and play through them before you solder anything.**

## Who uses it
- Guitar-pedal and synth DIY builders ("a fuzz that cleans up when I roll back the guitar volume").
- Electronics students and hobbyists ("why does my common-emitter amp clip?", "photo of a 1970s magazine schematic: what does it sound like?").
- Engineers sanity-checking a filter, LED driver or bias network.

## Rules every module follows (Hermes catalog requirements)
- Pure Python 3.11–3.14 + numpy only (`numpy>=1.26,<3`). No scipy, no matplotlib, no compiled code, no network calls.
- Plugin handlers: `def handler(args: dict, **kwargs) -> str` returning a JSON string, never raising.
- Files go under `ctx`-provided data dirs (`plugins.plugin_storage.plugin_data_dir("hotwire")`, i.e. `<HERMES_HOME>/plugin-data/hotwire/`). Never hardcode `~/.hermes`.
- Desktop half: single ESM `desktop/plugin.js`; only imports `@hermes/plugin-sdk`, `react`, `react/jsx-runtime`; no JSX syntax (use `jsx()`/`jsxs()`); no eval/new Function; theme variables only (no hardcoded colours); timers via `ctx.setInterval` etc.
- Author: Rohan Borade. License MIT.

## Repository layout (repo root = plugin root)
```
plugin.yaml
__init__.py                 register(ctx): tools, /hotwire command, skill, hooks
tools.py, schemas.py        tool handlers + JSON schemas
engine/                     the simulator (pure numpy)
  __init__.py               public API re-exports
  netlist.py                SPICE-subset parser + Hotwire *@ annotations
  devices.py                device models + part library params
  mna.py                    MNA assembly, Newton, DC/AC/transient cores
  analyses.py               op(), ac(), tran(), process_audio()
  measure.py                gain_db, f3db, thd, rms, peak, supply current, headroom...
  specs.py                  spec DSL parse + evaluate
  erc.py                    electrical rule check
  stress.py                 part stress vs ratings (power, voltage, current)
  montecarlo.py             tolerance Monte Carlo yield + sensitivity
  parts.py                  E-series snapping, BOM, part catalog (ratings, packages)
  schematic.py              netlist -> SVG schematic (auto layout)
  plot.py                   SVG Bode / scope / spectrum plots (no matplotlib)
  audio.py                  wav read/write (stdlib wave), demo DI generator (Karplus-Strong guitar)
  export.py                 KiCad netlist (.net), clean SPICE .cir, compile-to-JSON for the JS engine
library/                    reference circuits (*.cir) with specs, generic names, no trademarks
skills/analog-design/SKILL.md
dashboard/manifest.json, dashboard/plugin_api.py   REST for the Desktop half
desktop/plugin.js           Bench pane, tool media renderer, live AudioWorklet engine
tests/                      pytest: analytic + ngspice cross-checks
docs/                       SPEC.md (this), VALIDATION.md (measured accuracy)
README.md, LICENSE
```

## Netlist format (canonical): SPICE subset + `*@` annotations
LLMs already write SPICE fluently, so the agent writes SPICE. Case-insensitive, `+` continuation, `*` comments, engineering suffixes (f p n u m k meg g t; `meg` before `m`; trailing units ignored: `10kOhm`, `4.7uF`), node `0` or `gnd` is ground.

Elements:
- `Rname n1 n2 value`
- `Cname n1 n2 value [ic=v]`
- `Lname n1 n2 value`
- `Vname n+ n- [DC] v [AC mag] [SIN(vo va f [td] [theta])] [PULSE(v1 v2 td tr tf pw per)]`
- `Iname n+ n- [DC] v [AC mag]`
- `Dname a k model`
- `Qname c b e model` (NPN/PNP by model type)
- `Jname d g s model` (NJF/PJF)
- `Mname d g s b model [W=] [L=] [M=]` (NMOS/PMOS, SPICE level 1); the 3-node form `Mname d g s model` ties the bulk to the source. AD/AS/PD/PS/NRD/NRS are accepted and ignored.
- `Xname in+ in- vcc vee out OPAMPMODEL` (op-amp macromodel from the library; 5 pins in that order), or `Xname n1 n2 ... SUBNAME [param=value ...]` when SUBNAME is a `.subckt` in the netlist (the subcircuit wins over a part of the same name)
- `Ename n+ n- nc+ nc- gain` (VCVS), `Gname n+ n- nc+ nc- gm` (VCCS)
- `Sname n1 n2 on|off` (ideal switch, Hotwire extension, ron 0.1, roff 1e9)
- `.model name D(IS= N= RS= CJO= BV=)`, `.model name NPN(IS= BF= BR= VAF= ...)`, `.model name PNP(...)`, `.model name NJF(VTO= BETA= LAMBDA=)`, `PJF`, `.model name NMOS(VTO= KP= LAMBDA= GAMMA= PHI= RD= RS= CBD= CBS= IS= PB= MJ= CGSO= CGDO= CGBO= CGS= CGD= LD= W= L=)`, `PMOS`. `LEVEL` must be 1 (or absent); `TOX` (with `UO`) sets KP when KP isn't given, as in ngspice; VDMOS models are refused with a conversion hint.
- `.subckt NAME pin1 pin2 ... [params: a=1k b={2*a}]` ... `.ends [NAME]`: flattened at parse time. Internal nodes become `x1.node`, internal elements `X1.R1` (kind = first letter after the last dot), models defined inside become `X1.MODEL`; nesting up to 4 levels; recursion, pin-count mismatches, unknown parameters and missing `.ends` are errors. `*@` lines inside a subcircuit are ignored.
- `.param name=value` (global, any order) and `{expression}` anywhere in a value: numbers with SPICE suffixes, parameter names, `+ - * / ** ^ %`, parentheses, `pi`, `e`, and sqrt/exp/ln/log/log10/sin/cos/tan/asin/acos/atan/sinh/cosh/tanh/abs/min/max/pow/floor/ceil/int/round. Evaluated by a small AST interpreter (no eval).
- `.include`/`.inc`/`.lib` are NOT supported: Hotwire never reads files named in a netlist; paste the `.model` or `.subckt` text inline. Built-in part models are available by part name without a `.model` line (e.g. `D1 a k 1N4148`, `Q1 c b e 2N3904`, `M1 d g s 2N7000`, `X1 ... TL072`). `.control ... .endc` blocks are skipped.
- Analyses lines `.op .ac .tran` are accepted and ignored by the parser (tools choose analyses); `.end` optional.

Hotwire annotations (comment lines, ignored by other SPICE tools):
- `*@ title Two-transistor fuzz`
- `*@ in VIN` (input source; its AC/transient waveform is replaced during audition/AC) and `*@ out out` (output node)
- `*@ supply VCC` (supply source(s) for current/battery checks; repeatable)
- `*@ pot RV1 a w b 100k taper=log|lin|revlog pos=0.5 label="Gain"`: a potentiometer between a and b with wiper w. Expands to two resistors `RV1_A a w` and `RV1_B w b` (min 1 ohm each). `pos` 0..1; log taper: fraction of resistance a→w = pos^... use the 10%-at-midpoint audio taper curve `(11**pos-1)/10` style (document it).
- `*@ spec <expr>` (repeatable), see Specs.
- `*@ tol R=5% C=10% hfe=40%` default tolerances for Monte Carlo (defaults R 5%, C 10%, L 10%, BJT BF ±40%, JFET VTO ±0.5V absolute, MOSFET threshold `VTH` ±0.3 V absolute and `KP` ±20%, IS ±50%).
- `*@ rating C3 25V` / `*@ rating R5 0.5W` per-part rating override (defaults: resistor 0.25 W, electrolytic/film cap 25 V, ceramic 50 V; semis from part catalog).

## Engine API (Python, `engine/__init__.py`)
```python
c = parse(text: str) -> Circuit                 # raises NetlistError(line, message) with a precise message
c.with_pots({"RV1": 0.8}) -> Circuit            # new circuit with pot positions
op(c) -> OpResult                               # .v[node] volts, .i[element] amps (branch currents for V/L/E, device currents), .devices[name] dict (BJT: ic, ib, vbe, vce, region; diode: id, vd; jfet: id, vgs, vds, region; mosfet: id, vgs, vds, vbs, region, gm, gds, gmbs, vth, power; opamp: vout, saturated), .power[name] watts, .converged, .method ("newton", "gmin", "source")
ac(c, fstart=10, fstop=100e3, points_per_decade=40, input=None, output=None) -> AcResult   # .f (np.array), .h (complex transfer out/in), helpers .gain_db, .phase_deg
tran(c, tstop, tstep, input_wave=None, input=None, nodes=None) -> TranResult  # .t, .v[node] arrays; trapezoidal with Newton; must handle stiff diode clipping
process_audio(c, samples: np.ndarray, sr: int, input=None, output=None, input_scale=1.0) -> np.ndarray  # drive *@ in source with samples (volts), return *@ out node voltage, AC-coupled measurement (subtract DC op of output)
```
Accuracy target vs ngspice-42 (cross-check suite in tests/, numbers published in docs/VALIDATION.md): DC node voltages within 1% (or 1 mV), AC gain within 0.1 dB and -3 dB frequency within 1%, transient RMS error within 2% of signal RMS on clippers and fuzz.
Speed target: process 1 s of 48 kHz audio through a 2-transistor fuzz in under 10 s in CPython (reuse factorisations, preallocate, solve with numpy.linalg on small dense matrices; Newton on the nonlinear devices only, damped with junction voltage limiting like SPICE `pnjlim`).
Convergence: gmin stepping then source stepping fallback; never hang (iteration caps); report non-convergence clearly.

Device models (match ngspice level-1 defaults so cross-checks agree):
- Diode: Shockley with N, IS, RS (series resistance as internal node when RS>0), junction capacitance CJO (VJ 1, M 0.5) for transient, BV ignored unless provided (reverse breakdown model optional).
- BJT: Gummel-Poon subset = Ebers-Moll transport model with BF, BR, IS, VAF (Early), NF/NR, RB/RC/RE optional, CJE/CJC optional for transient. PNP mirrored.
- JFET: Shichman-Hodges (VTO, BETA, LAMBDA, IS for gate diodes).
- MOSFET: SPICE level 1 exactly as ngspice `mos1` at 27 °C: square law with LAMBDA, body effect (GAMMA, PHI), drain/source swap for VDS < 0, RD/RS internal nodes, bulk-drain and bulk-source diodes (IS) with CBD/CBS depletion charge, constant gate overlap capacitances (CGSO·W, CGDO·W, CGBO·Leff, or CGS/CGD in farads). Newton limiting like `mos1load` (fetlim on VGS or VGD, limvds, pnjlim on the bulk). Regions: cutoff, triode, saturation (the amplifying region; specs accept `active`). No Meyer gate capacitances, subthreshold current or temperature terms. A MOSFET with its bulk on its source costs 2 Newton ports, otherwise 3.
- Op-amp anti-windup: the macromodel's pole node is held within mid ± 4·half of the output window by a stiff softplus clamp (the tanh is 99.93% saturated there), so the stage recovers from saturation in microseconds; the exported/ngspice decks carry the same clamp as a B-source.
- Op-amp macromodel (library parts): differential input resistance, open-loop gain A0, single dominant pole from GBW, output resistance, output swing limited softly to (VEE+headroom, VCC-headroom) with a smooth tanh-like limiter, slew-rate ignored or approximated; supply current constant from VCC to VEE (Iq per amp).

Built-in part library (`devices.py` PART_LIBRARY, with ratings for stress and a one-line description for the BOM):
diodes 1N4148, 1N914, 1N34A (germanium), 1N5817 (Schottky), 1N4001, LED_RED, LED_GREEN, BAT41;
NPN 2N3904, 2N2222, BC549C, 2N5088, MPSA18; PNP 2N3906, BC559C, AC128 (germanium PNP);
JFET J201, 2N5457, MPF102;
MOSFET 2N7000, BS170 (N, TO-92), BS250 (P, TO-92), IRF510, IRF9510 (TO-220, simplified level-1 fits);
op-amps TL072, TL071, NE5532, LM358, RC4558 (generic dual), plus `OPAMP_IDEAL`.
Use widely published SPICE parameters (manufacturer models). Note sources in a comment.

## Specs DSL (`*@ spec ...` or passed to the check tool)
Each spec: `metric(args) op value [unit]` or `metric(args) in lo..hi`. Ops `< <= > >= ==~` (==~ means within 5%).
Metrics:
- `gain_db(f)` AC gain in/out at f (Hz). `phase(f)` degrees.
- `f3db_low`, `f3db_high` (-3 dB points relative to passband max), `fc` (for low/high pass, the -3 dB point), `q`/`peak_db` for resonant filters.
- `v(node)` DC operating voltage; `i(element)` DC current; `p(element)` power.
- `supply_current` total DC current from `*@ supply` sources (A); `battery_hours(capacity_mAh)`.
- `thd(f, vin_peak)` % THD at the output for a sine of amplitude vin_peak volts (tran, steady state, FFT of last N cycles).
- `vout_peak(f, vin_peak)`, `gain_tran(f, vin_peak)` dB large-signal gain.
- `headroom(f)` largest input peak (V) before THD exceeds 1% (bisection).
- `bias(Qn) == active` / `region(Qn)` region check (MOSFETs and JFETs: `saturation`, alias `active`; MOSFET `triode`, alias `linear`).
- `noise` is out of scope.
Every evaluation returns `{spec, measured, target, pass, margin, unit, explanation}` with a plain-English explanation ("gain at 1 kHz is 17.2 dB, 0.8 dB short of the 18 dB floor").

Automatic checks (always run by `circuit_check`):
- ERC: floating node (only one connection), node with no DC path to ground, voltage-source loop, shorted source, missing *@ in/out when an audio/AC check is requested, unknown model, op-amp without supply pins connected.
- Stress: each part's DC power / voltage / current at the idle operating point vs its rating, with 50% derating for resistors (signal peaks are not part of the stress check); electrolytic polarity reversed (negative DC across a cap whose name or value suggests electrolytic, value >= 1 uF); LED/diode current over rating; transistor VCE over VCEO, IC over rating.
- Bias sanity: transistors meant as amplifiers sitting in saturation/cutoff at idle are flagged (warning, not error).

## Monte Carlo yield (`montecarlo.py`)
`yield_(c, specs, n=200, seed=1, tol=None) -> {yield_pct, n, failures_by_spec, sensitivity:[{part, param, share_pct}], worst_case}`. Vary R, C, L, BF, IS, VTO per tolerance (uniform for R/C, normal clipped for semis). Sensitivity by correlation of each varied parameter with the spec margin (rank and normalise). Deterministic with seed. Must finish n=200 AC-only spec sets in < 20 s for 20-part circuits (reuse structure; AC-only specs don't need tran).

## BOM and parts (`parts.py`)
- `snap(value, series="E24")` nearest preferred value; `bom(c) -> list of {ref, value, snapped, part, package, rating, qty, note}` grouped by identical value; flags non-standard values and suggests the snapped one; adds pots with taper and value, sockets for ICs, a 9 V DC jack + polarity diode note when a 9 V supply is present.
- No prices (we don't publish numbers we can't verify).

## Schematic (`schematic.py`)
`svg(c, theme="auto") -> str`: readable left-to-right signal-flow schematic. Ground symbols instead of a ground net line where cleaner; supply rails at top as labelled arrows/flags (VCC, VEE); input at left, output at right; standard symbols (resistor zigzag or IEC box — pick zigzag), capacitor (polarised for >=1 uF), inductor, diode, LED, NPN/PNP, JFET, op-amp triangle, pot with wiper arrow, battery/source, switch. Labels: ref + value. Uses `currentColor` and CSS variables so it themes in Desktop; no hardcoded colours except via `var(--hotwire-ink, currentColor)`. Must lay out the full library without overlapping parts or labels.

## Plots (`plot.py`)
SVG only: `bode_svg(ac_result, specs=None)` (gain dB vs log f, spec targets drawn as shaded bands/markers), `scope_svg(tran, nodes)`, `spectrum_svg(samples, sr)`, `compare_svg(...)`. Same theming rule as schematics. Also `png` NOT required.

## Tools (agent half; names are the public contract)
| tool | args | does |
|---|---|---|
| `circuit_check` | `netlist` (str) or `path`, optional `specs` [str], `pots` {name: pos} | parse + ERC + OP + all specs + stress; returns verdict (`pass`/`fail`/`warn`), per-spec results, stress table, op summary, and writes `<design>/check.json`, `schematic.svg`, `bode.svg`; prints MEDIA lines for the SVGs |
| `circuit_simulate` | `netlist`/`path`, `analysis` op/ac/tran, params | raw-ish results (compact, decimated), SVG plot MEDIA |
| `circuit_audition` | `netlist`/`path`, `source`: `guitar_riff`/`bass_line`/`sine_sweep`/`drum_loop`/path to wav, `pots`, `level` (input volts peak, default 0.2 V for guitar), `seconds` (<=8) | renders dry and wet wav (48 kHz 16-bit), loudness-matched A/B, spectrum SVG; MEDIA lines for the wavs |
| `circuit_yield` | `netlist`/`path`, `n`, `seed` | Monte Carlo yield + sensitivity |
| `circuit_bom` | `netlist`/`path`, `series` | BOM table (markdown + json) |
| `circuit_export` | `netlist`/`path`, `format` kicad/spice/json | writes file, returns path |
| `circuit_library` | `query` optional | lists/returns reference circuits with descriptions and specs |
Every tool saves artifacts under `<data>/designs/<slug>/v<N>/` so the design history (v1 fails, v2 passes) is browsable; returns `design_id` and `version`.
Each tool call also calls `broadcast_plugin_event("hotwire", "design.updated", {...})` (best-effort, wrapped in try) so the Desktop Bench pane follows the agent live.

## Slash command
`/hotwire` (no args: list library + open hint; `/hotwire check <path>`; `/hotwire play <design_id>`).

## Skill `skills/analog-design/SKILL.md`
Teaches the agent the loop: clarify the target (sound/behaviour, supply, parts on hand) → start from the closest library topology → write the SPICE netlist with `*@` annotations and measurable specs → `circuit_check` → fix and re-check until every spec passes and stress is clean → `circuit_yield` (aim ≥90%) → `circuit_audition` for audio circuits → `circuit_bom` → present: what it does, why each part is there, the verdict table, the yield, the BOM, and the media. Also: transcribing a schematic photo into a netlist (name every part, check the parts count matches, then run ERC), and honesty rules (never claim a spec passes without a tool result).

## Desktop half (`desktop/plugin.js`) + backend (`dashboard/plugin_api.py`)
- Bench pane (right, 420px) and a full page route `/hotwire` with sidebar nav: current design schematic, verdict chips (specs pass/fail), Bode plot, scope, BOM, design history timeline (v1 → vN with what changed and which specs flipped), pot knobs (one per `*@ pot`) that re-run checks via REST and update plots live.
- **Play**: A/B player (dry vs circuit) for the rendered wavs; **Live**: the circuit compiled by `export.compile_json` runs in an AudioWorklet (MNA per sample in JS, same models) so knobs change the sound while it plays; source = bundled riff loop or the microphone/instrument input if the browser grants it. JS engine output must match Python `process_audio` within 1% RMS on the library (test with node).
- Tool media renderer for `circuit_audition` (A/B card with waveform) and `circuit_check` (schematic + verdict card).
- REST (`plugin_api.py`, FastAPI APIRouter): `GET /designs`, `GET /designs/{id}` (latest version + history), `POST /check` {netlist, pots}, `POST /audition`, `GET /compile/{id}`, `GET /file?path=` (only within the data dir).

## Shared data model (engine/netlist.py; every module codes against this exactly)
```python
from dataclasses import dataclass, field

@dataclass
class Model:
    name: str              # upper-case, e.g. "1N4148", "MYQ"
    type: str              # 'D' | 'NPN' | 'PNP' | 'NJF' | 'PJF' | 'NMOS' | 'PMOS' | 'OPAMP'
    params: dict           # upper-case keys: IS, N, RS, CJO, BF, VAF, VTO, BETA, A0, GBW, ...

@dataclass
class Element:
    name: str              # as written, upper-cased: "R1", "Q2", "XU1", "RV1_A"
    kind: str              # 'R','C','L','V','I','D','Q','J','M','X','E','G','S'
    nodes: list            # lower-cased node names; ground is always '0' ('gnd' normalised to '0')
                           # D: [a, k]; Q: [c, b, e]; J: [d, g, s]; M: [d, g, s, b]; X: [in+, in-, vcc, vee, out]; E/G: [n+, n-, nc+, nc-]
    value: float | None    # R ohms, C farads, L henries, V/I DC value, E gain, G gm; None for D/Q/J/X/S
    model: str | None = None   # upper-case model/part name for D/Q/J/X
    params: dict = field(default_factory=dict)  # 'ac': mag, 'sin': (vo,va,f,td,theta), 'pulse': (...), 'ic': v, 'on': bool (S), 'W'/'L'/'M' (MOSFET instance), 'sub': 'X1' (flattened from a .subckt)
    line: int = 0          # 1-based source line
    pot: str | None = None # 'RV1' when this resistor is a half of a *@ pot; params['pot_half'] = 'A' | 'B'

@dataclass
class Pot:
    name: str; a: str; w: str; b: str; value: float; taper: str; pos: float; label: str

@dataclass
class Circuit:
    title: str
    elements: list         # list[Element], in source order (pots expanded in place of the *@ pot line)
    models: dict           # name -> Model: explicit .model lines PLUS built-in parts actually referenced
    pots: dict             # name -> Pot
    input: str | None      # name of the input source element (e.g. "VIN")
    output: str | None     # output node name
    supplies: list         # supply source element names
    specs: list            # raw spec strings from *@ spec lines
    tol: dict              # e.g. {"R": 0.05, "C": 0.10, "BF": 0.40}
    ratings: dict          # element name -> (float, unit) e.g. {"C3": (25.0, "V"), "R5": (0.5, "W")}
    source: str            # original text
    def nodes(self) -> list: ...           # all non-ground nodes, sorted by first appearance
    def element(self, name) -> Element: ...
    def with_pots(self, positions: dict) -> "Circuit": ...
    def with_values(self, changes: dict) -> "Circuit": ...   # {"R1": 4700, "Q1.BF": 300}
```
Built-in part parameters live in `engine/parts_data.py` as `PARTS = {"1N4148": {"type": "D", "params": {...}, "rating": {...}, "desc": "...", "package": "DO-35", "source": "..."}, ...}`; ratings keys: `pd_w`, `vmax_v`, `imax_a`, `vceo_v`, `ic_a`.

## compile_json (engine/export.py) -> the JS live engine input
```json
{"version": 1, "title": "...", "nodes": ["in", "n1", "out"],   // index k>0 means nodes[k-1]; 0 is ground
 "input": "VIN", "output": 3, "sr_hint": 48000,
 "elements": [
   {"name": "R1", "kind": "R", "n": [1, 2], "value": 1000},
   {"name": "C1", "kind": "C", "n": [2, 0], "value": 1e-7},
   {"name": "L1", "kind": "L", "n": [2, 3], "value": 0.5},
   {"name": "VCC", "kind": "V", "n": [4, 0], "dc": 9.0},
   {"name": "VIN", "kind": "V", "n": [1, 0], "dc": 0.0, "input": true},
   {"name": "D1", "kind": "D", "n": [3, 0], "p": {"IS": 2.52e-9, "N": 1.752, "RS": 0.568, "CJO": 4e-12, "VJ": 1, "M": 0.5}},
   {"name": "Q1", "kind": "Q", "type": "npn", "n": [5, 6, 0], "p": {"IS": 6.7e-15, "BF": 416, "BR": 0.74, "VAF": 74, "NF": 1, "NR": 1}},
   {"name": "J1", "kind": "J", "type": "njf", "n": [7, 8, 0], "p": {"VTO": -0.8, "BETA": 1e-3, "LAMBDA": 0.01}},
   {"name": "M1", "kind": "M", "type": "nmos", "n": [7, 8, 0, 0], "p": {"VTO": 2.0, "KP": 0.25, "LAMBDA": 0.01, "W": 1e-4}},   // instance W/L/M (as MULT) merged into p
   {"name": "XU1", "kind": "X", "n": [9, 10, 4, 0, 11], "p": {"A0": 2e5, "GBW": 3e6, "RIN": 1e12, "ROUT": 75, "VSAT_HI": 1.5, "VSAT_LO": 1.5, "IQ": 1.4e-3}},
   {"name": "E1", "kind": "E", "n": [1, 0, 2, 0], "value": 10},
   {"name": "S1", "kind": "S", "n": [1, 2], "on": true}
 ],
 "pots": [{"name": "RV1", "a": 2, "w": 3, "b": 0, "value": 100000, "taper": "log", "pos": 0.5, "label": "Gain", "halves": ["RV1_A", "RV1_B"]}],
 "op": [0.0, 4.5, 4.1]}   // DC operating point node voltages (same order as nodes) to start the live engine
```
Pot taper law (both engines): fraction of total resistance between a and w = `pos` (lin), `(11**pos - 1)/10` (log, 10% at mid-travel), `1 - (11**(1-pos) - 1)/10` (revlog). Each half min 1 ohm.
