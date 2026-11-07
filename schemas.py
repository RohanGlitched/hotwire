"""Tool schemas: what the model reads to decide when and how to call Hotwire."""

_NETLIST = {
    "type": "string",
    "description": (
        "The circuit as a SPICE netlist (R C L V I D Q J M X E G S elements, .model lines, "
        ".subckt/.ends with X instances, .param and {expressions}; no .include: paste models inline) with Hotwire "
        "annotations in comments: '*@ in VIN' (input source), '*@ out out' (output node), "
        "'*@ supply VCC', '*@ pot RV1 a wiper b 100k taper=log pos=0.5 label=\"Gain\"', "
        "'*@ spec gain_db(1k) > 18'. Built-in parts need no .model line: 1N4148 1N914 1N34A 1N5817 "
        "1N4001 LED_RED LED_GREEN BAT41 2N3904 2N2222 BC549C 2N5088 MPSA18 2N3906 BC559C AC128 "
        "J201 2N5457 MPF102; MOSFETs 'M1 d g s 2N7000' (2N7000 BS170 BS250 IRF510 IRF9510, level 1, "
        "or '.model N1 NMOS(VTO= KP= LAMBDA=)'); op-amps 'X1 in+ in- vcc vee out TL072' (TL072 TL071 NE5532 LM358 "
        "RC4558 OPAMP_IDEAL). Ground is node 0. Values like 4.7k 100n 2.2u 1meg (M means milli)."
    ),
}
_DESIGN = {
    "type": "string",
    "description": "A design id returned by an earlier Hotwire call (reuses its latest netlist). Use this OR netlist OR path.",
}
_PATH = {"type": "string", "description": "Path to a .cir file, or a library id such as 'softclip_overdrive'."}
_POTS = {
    "type": "object",
    "description": "Pot wiper positions 0..1 by pot name, e.g. {\"RV1\": 0.8}. Unlisted pots keep their *@ pot pos.",
    "additionalProperties": {"type": "number"},
}
_NAME = {"type": "string", "description": "Short name for the design (used for its id and folder), e.g. 'warm-fuzz'."}


def _circuit_props(extra: dict | None = None) -> dict:
    props = {"netlist": _NETLIST, "design": _DESIGN, "path": _PATH, "name": _NAME, "pots": _POTS}
    props.update(extra or {})
    return props


CIRCUIT_CHECK = {
    "name": "circuit_check",
    "description": (
        "Prove an analog circuit works before anyone builds it. Parses the netlist, runs an electrical "
        "rule check (floating nodes, no DC path, shorted sources), solves the DC operating point, "
        "measures every '*@ spec' (and any extra specs you pass), and checks every part against its "
        "ratings (resistor power, capacitor voltage, transistor and LED current). Returns a pass/fail "
        "verdict per spec with the measured value and margin, part stress, transistor bias regions, "
        "and a schematic and Bode plot. Saves a new version of the design so its history is kept. "
        "Call it after every change; never tell the user a spec passes without this tool's result."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "specs": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Extra specs, same syntax as '*@ spec': 'gain_db(1k) in 18..24', 'fc < 2k', "
                    "'f3db_low < 40', 'thd(1k, 0.2) > 5%', 'supply_current < 5m', 'v(c1) in 4..5', "
                    "'region(Q1) == active', 'headroom(1k) > 1', 'battery_hours(500) > 100', 'thd_gv(300, 0.1, 0.25) < 5%' (THD with a real guitar's pickup, volume pot and cable modelled in front of the circuit, volume knob at 0.25: don't add the guitar to the netlist yourself), 'cleanup_vol(300, 0.1, 5) in 0.2..0.6' (guitar-volume setting where THD falls under 5%), 'zin(1k) > 250k' (input impedance: low values dull a pickup's treble)."
                ),
            },
            "note": {"type": "string", "description": "One line on what changed in this version (shown in the design history)."},
        }),
    },
}

CIRCUIT_SIMULATE = {
    "name": "circuit_simulate",
    "description": (
        "Run one raw analysis on a circuit: 'op' (DC node voltages, currents, device regions), "
        "'ac' (frequency response from the *@ in source to the *@ out node), or 'tran' (time "
        "waveforms with a sine or pulse input). Returns compact numbers plus an SVG plot. Use "
        "circuit_check for verdicts; use this to investigate."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "analysis": {"type": "string", "enum": ["op", "ac", "tran"]},
            "fstart": {"type": "number", "description": "AC start frequency in Hz (default 10)."},
            "fstop": {"type": "number", "description": "AC stop frequency in Hz (default 100000)."},
            "freq": {"type": "number", "description": "Tran: sine frequency in Hz (default 1000)."},
            "amplitude": {"type": "number", "description": "Tran: sine peak volts at the input (default 0.2)."},
            "cycles": {"type": "number", "description": "Tran: how many cycles to show (default 4)."},
            "nodes": {"type": "array", "items": {"type": "string"}, "description": "Nodes to plot (default input and output)."},
        }),
        "required": ["analysis"],
    },
}

CIRCUIT_AUDITION = {
    "name": "circuit_audition",
    "description": (
        "Hear the circuit. Plays an instrument recording through the simulated circuit, sample by "
        "sample with the real nonlinear device models, and writes two loudness-matched WAV files "
        "(dry and through the circuit) plus a spectrum plot, so the user can A/B it. Sources: "
        "'guitar_riff', 'guitar_chords', 'bass_line', 'drum_loop', 'sine_sweep', or a path to the "
        "user's own WAV. Use for pedals, amps, filters and anything audio. Include the returned "
        "MEDIA lines in your reply so the user can play them."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "source": {"type": "string", "description": "guitar_riff (default), guitar_chords, bass_line, drum_loop, sine_sweep, or a .wav path."},
            "level": {"type": "number", "description": "Input peak in volts (guitar pickup ~0.2, line level ~1.0). Default 0.2."},
            "seconds": {"type": "number", "description": "Length to render, up to 8 s (default 4)."},
            "guitar_volume": {"type": "number", "description": "Guitar sources play through a modelled pickup, volume pot and cable (pedals load the guitar). Volume 0..1, default 1. Lower it to hear clean-up."},
            "pickup": {"type": "string", "enum": ["single_coil", "humbucker", "p90"], "description": "Pickup model for guitar sources (default single_coil)."},
            "sweep": {
                "type": "object",
                "description": "Optional knob sweep while it plays: {\"pot\": \"RV1\", \"from\": 0.1, \"to\": 1.0}. Use pot 'guitar_volume' to roll the guitar's own volume knob (hear a fuzz clean up).",
            },
        }),
    },
}

CIRCUIT_TUNE = {
    "name": "circuit_tune",
    "description": (
        "Let the solver pick part values. Searches standard E-series values for the parts you list "
        "until the specs pass with the most margin (coordinate search over E12/E24 values, AC and DC "
        "specs). Use it when a design is close but a gain, corner frequency or bias point is off: "
        "e.g. vary ['R3', 'C2'] to hit 'fc in 2.8k..3.2k'. Returns the chosen values, before/after "
        "spec margins, and the updated netlist as a new design version."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "vary": {"type": "array", "items": {"type": "string"}, "description": "Parts whose values may change, e.g. ['R3', 'C2']."},
            "specs": {"type": "array", "items": {"type": "string"}, "description": "Specs to satisfy (defaults to the netlist's *@ spec lines)."},
            "series": {"type": "string", "enum": ["E6", "E12", "E24", "E48", "E96"], "description": "Value series (default E24 for resistors, E12 for capacitors)."},
        }),
        "required": ["vary"],
    },
}

CIRCUIT_EXPLAIN = {
    "name": "circuit_explain",
    "description": (
        "What does each part actually do? Nudges every resistor, capacitor and inductor by 10% "
        "(and every pot by a tenth of a turn) and measures how each gain, corner-frequency, voltage "
        "and current spec moves. Returns the most influential parts per spec and the parts that "
        "barely matter. Use it to explain a design ('R3 sets the gain: +10% gives +0.8 dB'), to "
        "teach, or to decide which part to change or tune."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "specs": {"type": "array", "items": {"type": "string"}, "description": "Extra specs to explain (small-signal and DC metrics)."},
        }),
    },
}

CIRCUIT_MATCH = {
    "name": "circuit_match",
    "description": (
        "Match a recorded tone. Give a dry guitar recording (DI) and the same take through a real "
        "pedal or amp (re-amped) as WAV files. With a circuit, it searches the circuit's knob positions "
        "so the simulation sounds like the recording (third-octave tone plus clipping character) and "
        "returns the settings and a closeness score (under 3 close, over 8 a different sound). With no "
        "circuit, it ranks the library's audio circuits by how close each gets. Takes up to a minute."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "dry": {"type": "string", "description": "Path to the dry (direct) WAV."},
            "wet": {"type": "string", "description": "Path to the WAV of the same take through the real pedal/amp."},
            "level": {"type": "number", "description": "Input peak volts of the dry take (default 0.2, a guitar pickup)."},
        }),
        "required": ["dry", "wet"],
    },
}

CIRCUIT_YIELD = {
    "name": "circuit_yield",
    "description": (
        "Will it work when someone builds it with real parts? Monte Carlo over part tolerances "
        "(5% resistors, 10% capacitors, transistor gain spread, diode and JFET spread; override with "
        "'*@ tol'). Returns the percentage of builds that meet every spec, which specs fail most, and "
        "which parts drive the variation, so you can tighten the right tolerance or redesign."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "n": {"type": "integer", "description": "Number of virtual builds (default 200, max 1000)."},
            "seed": {"type": "integer", "description": "Random seed for repeatable results (default 1)."},
        }),
    },
}

CIRCUIT_BOM = {
    "name": "circuit_bom",
    "description": (
        "Bill of materials a builder can order from: parts grouped by value, snapped to standard "
        "E-series values (flags non-standard values), semiconductor part names, packages, ratings, pot "
        "tapers, IC sockets, and for a 9 V pedal the jacks, footswitch, LED, enclosure and battery snap. "
        "Markdown table plus JSON."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "series": {"type": "string", "enum": ["E12", "E24", "E96"], "description": "Resistor series (default E24)."},
        }),
    },
}

CIRCUIT_BUILD = {
    "name": "circuit_build",
    "description": (
        "Turn a proven design into something you can solder: an automatically placed stripboard "
        "(veroboard) layout with every part, wire link and track cut, drawn as an SVG with a "
        "numbered build order and an off-board wiring list (jacks, pots, battery, LED). Run "
        "circuit_check first; only build designs that pass."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "max_width": {"type": "integer", "description": "Board columns limit (default 30)."},
        }),
    },
}

CIRCUIT_EXPORT = {
    "name": "circuit_export",
    "description": (
        "Export a design: 'kicad' (KiCad netlist .net to start a PCB), 'spice' (clean netlist with "
        ".model lines for ngspice/LTspice), 'svg' (schematic), or 'json' (compiled circuit). Returns "
        "the file path."
    ),
    "parameters": {
        "type": "object",
        "properties": _circuit_props({
            "format": {"type": "string", "enum": ["kicad", "spice", "svg", "json"]},
        }),
        "required": ["format"],
    },
}

CIRCUIT_LIBRARY = {
    "name": "circuit_library",
    "description": (
        "Reference circuits proven against ngspice, to start a design from: filters, transistor and "
        "JFET stages, op-amp overdrive, hard clipper, silicon and germanium fuzz, tone stacks, treble "
        "booster, wah, octave, oscillator, LED/power. With no query, lists them; with an id, returns "
        "the netlist, its knobs and its measured numbers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A library id or search words, e.g. 'fuzz', 'tone stack', 'low-pass'."},
        },
    },
}

ALL = [CIRCUIT_CHECK, CIRCUIT_SIMULATE, CIRCUIT_AUDITION, CIRCUIT_TUNE, CIRCUIT_EXPLAIN, CIRCUIT_MATCH, CIRCUIT_YIELD,
       CIRCUIT_BOM, CIRCUIT_BUILD, CIRCUIT_EXPORT, CIRCUIT_LIBRARY]
