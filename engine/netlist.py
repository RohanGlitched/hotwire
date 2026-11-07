"""SPICE-subset netlist parser with Hotwire ``*@`` annotations.

The agent writes ordinary SPICE (which language models already know well);
Hotwire adds a few comment annotations that other SPICE tools ignore:

    *@ title Two-transistor fuzz
    *@ in VIN                     input source (replaced by audio / AC drive)
    *@ out out                    output node
    *@ supply VCC                 supply source(s), repeatable
    *@ pot RV1 a w b 100k taper=log pos=0.5 label="Gain"
    *@ spec gain_db(1k) > 18      measurable requirement, repeatable
    *@ tol R=1% C=5% BF=40%       Monte Carlo tolerances
    *@ rating C3 16V              part rating override (V, W or A)

Beyond the element lines, ``.subckt NAME pins [params: k=v]`` ... ``.ends`` blocks
are flattened at parse time (``X1.R1`` on nodes ``x1.n``), ``.param`` and
``{expression}`` values are evaluated by a whitelisted AST interpreter, and
``.include``/``.lib`` are refused: no netlist ever makes Hotwire read a file.

Every public function raises :class:`NetlistError` with the line number and a
plain-English message instead of a traceback, so the agent can fix its own
netlist from the error text alone.
"""
from __future__ import annotations

import copy
import math
import re
import shlex
from dataclasses import dataclass, field

__all__ = [
    "NetlistError", "Model", "Element", "Pot", "Circuit", "parse", "parse_value",
    "format_value", "pot_fraction", "eval_expr",
]


class NetlistError(ValueError):
    """A netlist could not be understood. ``line`` is 1-based (0 = whole file)."""

    def __init__(self, line: int, message: str):
        self.line = line
        self.message = message
        super().__init__(f"line {line}: {message}" if line else message)


# --------------------------------------------------------------------------- values

_SUFFIX = [("meg", 1e6), ("mil", 25.4e-6), ("f", 1e-15), ("p", 1e-12), ("n", 1e-9),
           ("u", 1e-6), ("µ", 1e-6), ("m", 1e-3), ("k", 1e3), ("g", 1e9), ("t", 1e12)]
_NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(.*)$", re.I)
_RKM = re.compile(r"^(\d+)([rkmμunpf])(\d+)$", re.I)  # 4k7, 2r2, 4n7, 2p2 (European notation)


def parse_value(text: str, line: int = 0) -> float:
    """Parse ``10k``, ``4.7uF``, ``2meg``, ``1e-9``, ``4k7``, ``100mV`` into a float."""
    s = str(text).strip().replace("Ω", "").replace("ohm", "").replace("Ohm", "")
    m = _RKM.match(s)
    if m:
        mult = {"r": 1.0, "k": 1e3, "m": 1e6, "u": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}[m.group(2).lower()]
        return float(f"{m.group(1)}.{m.group(3)}") * mult
    m = _NUM.match(s)
    if not m:
        raise NetlistError(line, f"'{text}' is not a number (examples: 10k, 4.7u, 100n, 2meg)")
    num = float(m.group(1))
    rest = m.group(2).strip().lower()
    for suf, mult in _SUFFIX:
        if rest.startswith(suf):
            return num * mult
    # remaining text is a unit (V, A, F, H, Hz, s, W) or empty
    if rest and not re.match(r"^[a-z]*$", rest):
        raise NetlistError(line, f"'{text}' has an unexpected suffix '{rest}'")
    return num


def format_value(v: float, unit: str = "") -> str:
    """Engineering notation. Without a unit it's netlist style (4.7k, 100n, 2.2u, 1M);
    with a unit it's display style with a space and a real micro sign (4.7 kΩ, 2.2 µF)."""
    sep = " " if unit else ""
    if v == 0 or not math.isfinite(v):
        return f"{v:g}{sep}{unit}"
    a = abs(v)
    for mult, suf in [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"),
                      (1e-6, "u"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f")]:
        if a >= mult * 0.9999:
            x = v / mult
            s = f"{x:.3g}"
            if unit and suf == "u":
                suf = "µ"
            return f"{s}{sep}{suf}{unit}"
    return f"{v:.3g}{sep}{unit}"


def pot_fraction(pos: float, taper: str) -> float:
    """Fraction of a pot's resistance between terminal ``a`` and the wiper."""
    pos = min(1.0, max(0.0, float(pos)))
    t = (taper or "lin").lower()
    if t in ("log", "a", "audio"):
        return (11 ** pos - 1) / 10
    if t in ("revlog", "rev", "c"):
        return 1 - (11 ** (1 - pos) - 1) / 10
    return pos


# --------------------------------------------------------------------------- model types

@dataclass
class Model:
    name: str
    type: str
    params: dict


@dataclass
class Element:
    name: str
    kind: str
    nodes: list
    value: float | None
    model: str | None = None
    params: dict = field(default_factory=dict)
    line: int = 0
    pot: str | None = None


@dataclass
class Pot:
    name: str
    a: str
    w: str
    b: str
    value: float
    taper: str
    pos: float
    label: str


@dataclass
class Circuit:
    title: str
    elements: list
    models: dict
    pots: dict
    input: str | None
    output: str | None
    supplies: list
    specs: list
    tol: dict
    ratings: dict
    source: str

    # -- queries ---------------------------------------------------------
    def nodes(self) -> list:
        seen: list = []
        for e in self.elements:
            for n in e.nodes:
                if n != "0" and n not in seen:
                    seen.append(n)
        return seen

    def element(self, name: str) -> Element:
        key = name.upper()
        for e in self.elements:
            if e.name == key:
                return e
        raise NetlistError(0, f"no element named {name}")

    def has(self, name: str) -> bool:
        key = name.upper()
        return any(e.name == key for e in self.elements)

    # -- derived circuits --------------------------------------------------
    def with_pots(self, positions: dict) -> "Circuit":
        """A copy with pot wipers moved. ``positions``: {"RV1": 0.8} (0..1)."""
        c = copy.deepcopy(self)
        for name, pos in (positions or {}).items():
            key = name.upper()
            if key not in c.pots:
                raise NetlistError(0, f"no pot named {name} (pots: {', '.join(c.pots) or 'none'})")
            p = c.pots[key]
            p.pos = min(1.0, max(0.0, float(pos)))
            ra, rb = _pot_halves(p)
            for e in c.elements:
                if e.pot == key:
                    e.value = ra if e.params.get("pot_half") == "A" else rb
        return c

    def with_values(self, changes: dict) -> "Circuit":
        """A copy with values changed: {"R1": 4700, "Q1.BF": 300, "D1.IS": 1e-9}."""
        c = copy.deepcopy(self)
        for key, val in (changes or {}).items():
            if "." in key:
                ename, pname = key.split(".", 1)
                e = c.element(ename)
                mname = e.model
                if not mname or mname not in c.models:
                    raise NetlistError(e.line, f"{e.name} has no model parameters to change")
                # give this element a private model so siblings are untouched
                private = f"{mname}__{e.name}"
                if private not in c.models:
                    c.models[private] = Model(private, c.models[mname].type, dict(c.models[mname].params))
                c.models[private].params[pname.upper()] = float(val)
                e.model = private
            else:
                c.element(key).value = float(val)
        return c


def _pot_halves(p: Pot) -> tuple:
    frac = pot_fraction(p.pos, p.taper)
    return max(1.0, p.value * frac), max(1.0, p.value * (1 - frac))


# --------------------------------------------------------------------------- parsing

def parts_data_lookup(name: str):
    """Built-in part entry from ``engine.parts_data.PARTS`` (None if unknown)."""
    try:
        from .parts_data import PARTS
    except ImportError:  # parts table not available: only .model lines work
        return None
    key = (name or "").upper()
    if key in PARTS:
        return PARTS[key]
    # tolerate common aliases: "LED" -> LED_RED, "OPAMP" -> OPAMP_IDEAL
    alias = {"LED": "LED_RED", "OPAMP": "OPAMP_IDEAL", "IDEAL": "OPAMP_IDEAL"}.get(key)
    return PARTS.get(alias) if alias else None


_KINDS = set("RCLVIDQJXEGSM")
_MODEL_TYPES = {"D": "D", "NPN": "NPN", "PNP": "PNP", "NJF": "NJF", "PJF": "PJF",
                "OPAMP": "OPAMP", "NMOS": "NMOS", "PMOS": "PMOS"}
_PARAM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s,()]+)")
MAX_SUBCKT_DEPTH = 4
# MOSFET instance parameters that only matter for area/perimeter-scaled junctions
# (CJ, CJSW, JS, RSH), which Hotwire's level-1 model doesn't use: accepted, ignored.
_MOS_IGNORED_INST = {"AD", "AS", "PD", "PS", "NRD", "NRS", "IC", "OFF", "TEMP", "DTEMP"}


# --------------------------------------------------------------------------- .param expressions

_EXPR_FUNCS = {
    "sqrt": math.sqrt, "exp": math.exp, "ln": math.log, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "abs": abs, "min": min, "max": max, "pow": pow, "floor": math.floor, "ceil": math.ceil,
    "int": math.trunc, "round": round,
}
_EXPR_CONSTS = {"pi": math.pi, "e": math.e}
# a number followed by a unit suffix (4.7k, 100n, 2meg); the suffix may not start an exponent,
# so 1e3, 4.7e3 and 1e-9 stay plain numbers instead of being read as "1" with an "e" unit
_NUM_SUFFIX_RE = re.compile(r"(?<![A-Za-z_\d.])((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?![eE][+-]?\d)([A-Za-zµ]+)")


def _expr_prep(expr: str, line: int) -> str:
    """SPICE expression text -> Python expression text (number suffixes, ^ for power)."""
    def num(m):
        return repr(parse_value(m.group(1) + m.group(2), line))
    s = _NUM_SUFFIX_RE.sub(num, expr.strip())
    return s.replace("^", "**")


def eval_expr(expr: str, scope: dict | None = None, line: int = 0) -> float:
    """Evaluate a SPICE parameter expression such as ``2*R1`` or ``1/(2*pi*f*C)``.

    Only numbers, parameter names, + - * / ** % ^, parentheses and a short list of
    math functions are accepted: the text is parsed with :mod:`ast` and walked by a
    small interpreter, so nothing in a netlist can run code.
    """
    import ast

    scope = scope or {}
    text = str(expr).strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    if not text:
        raise NetlistError(line, "empty expression")
    if len(text) > 400:
        raise NetlistError(line, "expression is too long (400 characters at most)")
    try:
        tree = ast.parse(_expr_prep(text, line), mode="eval")
    except SyntaxError:
        raise NetlistError(line, f"can't read the expression '{text}'") from None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name):
            key = n.id.lower()
            if key in scope:
                return float(scope[key])
            if key in _EXPR_CONSTS:
                return _EXPR_CONSTS[key]
            raise NetlistError(line, f"'{n.id}' isn't defined: add '.param {n.id}=…' before using it")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            x = ev(n.operand)
            return x if isinstance(n.op, ast.UAdd) else -x
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = n.op
            try:
                if isinstance(op, ast.Add):
                    return a + b
                if isinstance(op, ast.Sub):
                    return a - b
                if isinstance(op, ast.Mult):
                    return a * b
                if isinstance(op, ast.Div):
                    return a / b
                if isinstance(op, ast.Mod):
                    return math.fmod(a, b)
                if isinstance(op, ast.Pow):
                    if abs(b) > 1e3:
                        raise OverflowError
                    return float(a ** b)
            except ZeroDivisionError:
                raise NetlistError(line, f"division by zero in '{text}'") from None
            except (OverflowError, ValueError, TypeError):
                raise NetlistError(line, f"'{text}' has no finite value") from None
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords
                and n.func.id.lower() in _EXPR_FUNCS and 1 <= len(n.args) <= 4):
            args = [ev(a) for a in n.args]
            try:
                return float(_EXPR_FUNCS[n.func.id.lower()](*args))
            except (ValueError, OverflowError, TypeError):
                raise NetlistError(line, f"{n.func.id}() can't take those arguments in '{text}'") from None
        raise NetlistError(line, f"'{text}' uses something an expression can't: only numbers, .param names, "
                                 f"+ - * / ** and {', '.join(sorted(_EXPR_FUNCS))} are allowed")

    val = ev(tree)
    if not math.isfinite(val):
        raise NetlistError(line, f"'{text}' has no finite value")
    return val


_BRACE_RE = re.compile(r"\{([^{}]*)\}")


def _subst(text: str, scope: dict, line: int) -> str:
    """Replace every {expression} in ``text`` with its value."""
    if "{" not in text:
        return text
    out = _BRACE_RE.sub(lambda m: f"{eval_expr(m.group(1), scope, line):.15g}", text)
    if "{" in out or "}" in out:
        raise NetlistError(line, "unbalanced { } in an expression")
    return out


def _split_assign(text: str) -> list:
    """Tokens of a line, keeping 'k = {a b}' together as 'k={a b}'."""
    text = re.sub(r"\s*=\s*", "=", text)
    toks, buf, depth = [], "", 0
    for ch in text:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth = max(0, depth - 1)
        if ch.isspace() and depth == 0:
            if buf:
                toks.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        toks.append(buf)
    return toks


def _assignments(toks: list, scope: dict, line: int, what: str) -> dict:
    """{name: value} from 'name=expr' tokens evaluated in ``scope`` (names lower-cased)."""
    out = {}
    for t in toks:
        if t.lower() in ("params:", "param:"):
            continue
        if "=" not in t:
            raise NetlistError(line, f"{what}: expected name=value, got '{t}'")
        k, v = t.split("=", 1)
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            raise NetlistError(line, f"{what}: '{k}' isn't a valid parameter name")
        out[k.lower()] = eval_expr(v, {**scope, **out}, line)
    return out


def _eval_params(lines: list, scope: dict) -> dict:
    """Evaluate '.param a=1 b={2*a}' lines into a copy of ``scope``; order-independent."""
    pending = []
    for ln, line in lines:
        body = re.sub(r"^\.param\s*", "", line, flags=re.I)
        for t in _split_assign(body):
            if "=" not in t:
                raise NetlistError(ln, f".param needs name=value, got '{t}'")
            k, v = t.split("=", 1)
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
                raise NetlistError(ln, f".param: '{k}' isn't a valid parameter name")
            pending.append((ln, k.lower(), v))
    out = dict(scope)
    while pending:
        rest, err = [], None
        for ln, k, v in pending:
            try:
                out[k] = eval_expr(v, out, ln)
            except NetlistError as e:
                if "isn't defined" not in e.message:
                    raise
                rest.append((ln, k, v))
                err = e
        if len(rest) == len(pending):
            raise err
        pending = rest
    return out


# --------------------------------------------------------------------------- subcircuits

@dataclass
class _Subckt:
    name: str
    pins: list
    defaults: list           # raw 'name=expr' tokens
    body: list               # [(line, text)]
    line: int


def _split_blocks(lines: list) -> tuple:
    """Split logical lines into (main lines, {name: _Subckt}, top-level .param lines).

    .control/.endc blocks (ngspice scripts) are dropped; everything after .end is ignored.
    """
    main, subs, params = [], {}, []
    cur = None
    in_control = False
    for ln, line in lines:
        low = _strip_inline_comment(line).lower() if not line.startswith("*") else ""
        word = low.split()[0] if low.split() else ""
        if in_control:
            if word == ".endc":
                in_control = False
            continue
        if word == ".control":
            in_control = True
            continue
        if cur is not None:
            if word == ".ends":
                subs[cur.name] = cur
                cur = None
            elif word == ".subckt":
                raise NetlistError(ln, f"a .subckt can't be defined inside another (.subckt {cur.name} "
                                       "has no .ends yet); define each one at the top level")
            elif word == ".end":
                raise NetlistError(cur.line, f".subckt {cur.name} has no .ends")
            else:
                cur.body.append((ln, line))
            continue
        if word == ".subckt":
            toks = _split_assign(_strip_inline_comment(line))
            if len(toks) < 2:
                raise NetlistError(ln, ".subckt needs a name and its pins, e.g. '.subckt STAGE in out vcc'")
            name = toks[1].upper()
            if name in subs:
                raise NetlistError(ln, f".subckt {name} is defined twice")
            pins, defaults, seen_params = [], [], False
            for t in toks[2:]:
                if t.lower() in ("params:", "param:"):
                    seen_params = True
                elif "=" in t:
                    defaults.append(t)
                elif seen_params:
                    raise NetlistError(ln, f".subckt {name}: '{t}' after params: needs a value (name=value)")
                else:
                    pins.append(_norm_node(t))
            if "0" in pins:
                raise NetlistError(ln, f".subckt {name}: ground (0) can't be a pin; it is global already")
            if len(set(pins)) != len(pins):
                raise NetlistError(ln, f".subckt {name} lists a pin twice")
            cur = _Subckt(name, pins, defaults, [], ln)
            continue
        if word == ".ends":
            raise NetlistError(ln, ".ends without a .subckt")
        if word == ".param":
            params.append((ln, _strip_inline_comment(line)))
            continue
        if word == ".end":
            break
        main.append((ln, line))
    if cur is not None:
        raise NetlistError(cur.line, f".subckt {cur.name} has no .ends")
    return main, subs, params


def _element_layout(kind: str, toks: list) -> tuple:
    """(node token indices, model token index or None) for an element line's tokens."""
    pos = [i for i, t in enumerate(toks) if i > 0 and "=" not in t]
    if kind in "RCLVIS":
        return [1, 2], None
    if kind == "D":
        return [1, 2], 3 if len(toks) > 3 else None
    if kind in "QJ":
        return [1, 2, 3], 4 if len(toks) > 4 else None
    if kind in "EG":
        return [1, 2, 3, 4], None
    if kind in "MX":
        return pos[:-1], (pos[-1] if pos else None)
    return [], None


def _expand(inst: str, nodes: list, sub: "_Subckt", given: dict, gscope: dict, subs: dict,
            depth: int, stack: tuple, line: int) -> list:
    """Flatten one X instance: [(line, text, instance path)] element and .model lines."""
    if depth > MAX_SUBCKT_DEPTH:
        raise NetlistError(line, f"{inst}: subcircuits nest deeper than {MAX_SUBCKT_DEPTH} levels")
    if sub.name in stack:
        raise NetlistError(line, f"{inst}: .subckt {sub.name} uses itself ({' -> '.join(stack + (sub.name,))})")
    if len(nodes) != len(sub.pins):
        raise NetlistError(line, f"{inst} connects {len(nodes)} node{'s' if len(nodes) != 1 else ''} but "
                                 f".subckt {sub.name} has {len(sub.pins)} pins ({' '.join(sub.pins)})")
    declared = [d.split("=", 1)[0] for d in sub.defaults]
    for k in given:
        if k not in {d.lower() for d in declared}:
            raise NetlistError(line, f"{inst}: .subckt {sub.name} has no parameter '{k}' "
                                     f"(it has: {', '.join(declared) or 'none'})")
    scope = dict(gscope)
    for d in sub.defaults:          # defaults in order; instance values win and feed later defaults
        k, v = d.split("=", 1)
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            raise NetlistError(sub.line, f".subckt {sub.name}: '{k}' isn't a valid parameter name")
        k = k.lower()
        scope[k] = given[k] if k in given else eval_expr(v, scope, sub.line)
    body_params = [(ln, _strip_inline_comment(t)) for ln, t in sub.body
                   if t.lower().startswith(".param")]
    scope = _eval_params(body_params, scope)
    pinmap = dict(zip(sub.pins, nodes))
    prefix = inst.upper() + "."
    nprefix = inst.lower() + "."
    local_models = set()
    for ln, t in sub.body:
        s = _strip_inline_comment(t)
        if s.lower().startswith(".model"):
            parts = s.split()
            if len(parts) > 1:
                local_models.add(parts[1].upper())

    def node(tok):
        n = _norm_node(tok)
        if n == "0":
            return "0"
        return pinmap.get(n, nprefix + n)

    out = []
    for ln, t in sub.body:
        if t.startswith("*") or not t.strip():
            continue
        s = _strip_inline_comment(t)
        if not s:
            continue
        low = s.lower()
        if low.startswith("."):
            word = low.split()[0]
            if word == ".model":
                s = _subst(s, scope, ln)
                parts = s.split(None, 2)
                if len(parts) < 3:
                    raise NetlistError(ln, ".model needs a name and a type, e.g. '.model D1N D(IS=2n N=1.8)'")
                out.append((ln, f".model {prefix}{parts[1].upper()} {parts[2]}", inst.upper()))
            elif word in (".include", ".inc", ".lib"):
                raise NetlistError(ln, _include_message(word))
            # .param handled above; .ends/.end can't appear; other dot lines ignored
            continue
        s = _subst(s, scope, ln)
        toks = _split_assign(s)
        local = toks[0].upper()
        kind = local[0]
        name = prefix + local
        if kind == "X":
            pos = [i for i, x in enumerate(toks) if i > 0 and "=" not in x]
            target = toks[pos[-1]].upper() if pos else ""
            if target in subs:
                kids = [node(toks[i]) for i in pos[:-1]]
                given_kid = _assignments([x for x in toks[1:] if "=" in x], scope, ln, name)
                out += _expand(name, kids, subs[target], given_kid, gscope, subs, depth + 1,
                               stack + (sub.name,), ln)
                continue
        node_idx, model_idx = _element_layout(kind, toks)
        new = list(toks)
        new[0] = name
        for i in node_idx:
            if i < len(new):
                new[i] = node(new[i])
        if model_idx is not None and model_idx < len(new) and new[model_idx].upper() in local_models:
            new[model_idx] = prefix + new[model_idx].upper()
        out.append((ln, " ".join(new), inst.upper()))
    return out


def _include_message(word: str) -> str:
    return (f"{word} isn't supported: Hotwire never reads files named in a netlist. Paste the model or "
            "subcircuit text inline instead (.model lines, or .subckt NAME pins … .ends), or use a "
            "built-in part name (1N4148, 2N3904, 2N7000, TL072, …)")


def _norm_node(n: str) -> str:
    n = n.strip().lower()
    return "0" if n in ("0", "gnd", "ground", "gnd!") else n


def _logical_lines(text: str):
    """Yield (line_no, text) joining '+' continuation lines; keeps '*@' lines."""
    buf, start = None, 0
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("+") and buf is not None:
            buf += " " + stripped[1:]
            continue
        if buf is not None:
            yield start, buf
        buf, start = stripped, i
    if buf is not None:
        yield start, buf


def _strip_inline_comment(s: str) -> str:
    for mark in (";", "$ "):
        idx = s.find(mark)
        if idx >= 0:
            s = s[:idx]
    return s.strip()


def _func_args(token_text: str, name: str, line: int) -> tuple:
    m = re.search(name + r"\s*\(([^)]*)\)", token_text, re.I)
    if not m:
        return ()
    parts = [p for p in re.split(r"[\s,]+", m.group(1).strip()) if p]
    return tuple(parse_value(p, line) for p in parts)


def _parse_source(name: str, kind: str, toks: list, rest_text: str, line: int) -> Element:
    if len(toks) < 3:
        raise NetlistError(line, f"{name} needs two nodes and a value, e.g. '{name} n1 0 DC 9'")
    nodes = [_norm_node(toks[1]), _norm_node(toks[2])]
    params: dict = {}
    text = " ".join(toks[3:])
    sin = _func_args(text, "sin", line)
    if sin:
        if len(sin) < 3:
            raise NetlistError(line, f"{name}: SIN needs (offset amplitude freq)")
        params["sin"] = tuple(sin) + (0.0,) * (5 - len(sin))
    pulse = _func_args(text, "pulse", line)
    if pulse:
        if len(pulse) < 2:
            raise NetlistError(line, f"{name}: PULSE needs at least (v1 v2)")
        params["pulse"] = tuple(pulse) + (0.0,) * (7 - len(pulse))
    # strip function calls, then read DC / AC keywords
    flat = re.sub(r"(sin|pulse|pwl|exp)\s*\([^)]*\)", " ", text, flags=re.I).split()
    dc = 0.0
    have_dc = False
    i = 0
    while i < len(flat):
        t = flat[i].lower()
        if t == "dc" and i + 1 < len(flat):
            dc, have_dc = parse_value(flat[i + 1], line), True
            i += 2
        elif t == "ac" and i + 1 < len(flat):
            params["ac"] = parse_value(flat[i + 1], line)
            if i + 2 < len(flat):
                try:
                    params["ac_phase"] = parse_value(flat[i + 2], line)
                    i += 1
                except NetlistError:
                    pass
            i += 2
        elif not have_dc:
            dc, have_dc = parse_value(flat[i], line), True
            i += 1
        else:
            raise NetlistError(line, f"{name}: didn't understand '{flat[i]}'")
    if "sin" in params and not have_dc:
        dc = params["sin"][0]
    return Element(name, kind, nodes, dc, None, params, line)


def _parse_mosfet(name: str, toks: list, ln: int, extra: dict) -> Element:
    """``Mname d g s [b] model [W=] [L=] [M=]``; the 3-node form ties bulk to source."""
    pos = [t for t in toks[1:] if "=" not in t]
    if pos and pos[-1].lower() == "off":
        pos = pos[:-1]
    if len(pos) not in (4, 5):
        raise NetlistError(ln, f"{name} needs drain, gate, source, an optional bulk and a model, e.g. "
                               f"'{name} d g s s 2N7000' or '{name} d g s 2N7000' (got {len(pos)} fields)")
    nodes = [_norm_node(t) for t in pos[:-1]]
    if len(nodes) == 3:
        nodes.append(nodes[2])
    params = dict(extra)
    for t in toks[1:]:
        if "=" not in t:
            continue
        k, v = t.split("=", 1)
        key = k.upper()
        if key in ("W", "L", "M"):
            val = parse_value(v, ln)
            if not val > 0:
                raise NetlistError(ln, f"{name}: {key}={v} must be positive")
            params[key] = val
        elif key in _MOS_IGNORED_INST:
            continue
        else:
            raise NetlistError(ln, f"{name}: unknown MOSFET instance parameter '{k}' (W, L and M are used; "
                                   "AD, AS, PD, PS, NRD, NRS are accepted and ignored)")
    return Element(name, "M", nodes, None, pos[-1].upper(), params, ln)


def _mos_model_params(mname: str, params: dict, ln: int) -> dict:
    """Check and normalise a level-1 NMOS/PMOS .model (LEVEL, TOX -> KP, physical ranges)."""
    p = dict(params)
    level = p.pop("LEVEL", 1.0)
    if level != 1:
        raise NetlistError(ln, f"model {mname}: LEVEL={level:g} isn't simulated; Hotwire's MOSFET is SPICE "
                               "level 1 (Shichman-Hodges). Give level-1 VTO, KP, LAMBDA (and GAMMA/PHI) "
                               "or use a built-in part")
    if "VT0" in p and "VTO" not in p:
        p["VTO"] = p.pop("VT0")
    tox = p.get("TOX", 0.0)
    if tox and tox > 0:
        if "KP" not in p:          # ngspice: KP = UO * eps_ox / TOX (UO in cm^2/Vs)
            p["KP"] = p.get("UO", p.get("U0", 600.0)) * 1e-4 * 3.9 * 8.854214871e-12 / tox
        if "NSUB" in p and not all(k in p for k in ("VTO", "GAMMA", "PHI")):
            raise NetlistError(ln, f"model {mname}: NSUB-derived VTO/GAMMA/PHI aren't computed; give VTO, "
                                   "GAMMA and PHI directly")
    for key in ("KP", "W", "L", "PHI", "PB"):
        if key in p and not p[key] > 0:
            raise NetlistError(ln, f"model {mname}: {key}={p[key]:g} isn't physical (it must be positive)")
    for key in ("GAMMA", "LAMBDA", "RD", "RS", "CBD", "CBS", "CGSO", "CGDO", "CGBO", "CGS", "CGD", "LD"):
        if key in p and p[key] < 0:
            raise NetlistError(ln, f"model {mname}: {key}={p[key]:g} can't be negative")
    if "MJ" in p and not 0 < p["MJ"] < 1:
        raise NetlistError(ln, f"model {mname}: MJ={p['MJ']:g} must be between 0 and 1")
    return p


def parse(text: str) -> Circuit:
    """Parse a SPICE-subset netlist with Hotwire annotations into a :class:`Circuit`.

    ``.subckt``/``.ends`` definitions are flattened at parse time: ``X1 a b STAGE r=2k``
    becomes elements ``X1.R1``... with internal nodes ``x1.n1``...  ``.param`` values and
    ``{expressions}`` are evaluated with a safe arithmetic evaluator.
    """
    if not isinstance(text, str) or not text.strip():
        raise NetlistError(0, "the netlist is empty")
    elements: list = []
    names: set = set()
    instances: set = set()
    models: dict = {}
    pots: dict = {}
    title = ""
    inp = out = None
    supplies: list = []
    specs: list = []
    tol = {"R": 0.05, "C": 0.10, "L": 0.10, "BF": 0.40, "IS": 0.50, "VTO": 0.5}
    ratings: dict = {}
    first_content_line = True

    main, subckts, param_lines = _split_blocks(list(_logical_lines(text)))
    gscope = _eval_params(param_lines, {})
    work = [(ln, line, None) for ln, line in main]
    wi = 0
    while wi < len(work):
        ln, line, sub = work[wi]
        wi += 1
        if not line:
            continue
        if line.startswith("*@") and sub is None:
            body = line[2:].strip()
            if not body:
                continue
            key, _, arg = body.partition(" ")
            key, arg = key.lower(), arg.strip()
            if key == "title":
                title = arg
            elif key == "in":
                inp = arg.split()[0].upper() if arg else None
            elif key == "out":
                out = _norm_node(arg.split()[0]) if arg else None
            elif key == "supply":
                supplies += [s.upper() for s in arg.split()]
            elif key == "spec":
                if arg:
                    specs.append(arg)
            elif key == "tol":
                for k, v in _PARAM_RE.findall(arg):
                    v = v.strip()
                    pct = v.endswith("%")
                    num = parse_value(v.rstrip("%"), ln)
                    frac = num / 100 if pct else num
                    if k.upper() not in ("VTO", "VTH") and not 0 <= frac < 1:
                        raise NetlistError(ln, f"tolerance {k}={v} means ±{frac * 100:g}%; write it with a % sign, e.g. {k}=5%")
                    tol[k.upper()] = frac
            elif key == "rating":
                toks = arg.split()
                if len(toks) != 2:
                    raise NetlistError(ln, "rating needs a part and a value, e.g. '*@ rating C3 16V'")
                unit = toks[1][-1].upper()
                if unit not in "VWA":
                    raise NetlistError(ln, "rating value must end in V, W or A, e.g. 16V, 0.5W")
                ratings[toks[0].upper()] = (parse_value(toks[1][:-1], ln), unit)
            elif key == "pot":
                try:
                    toks = shlex.split(arg)
                except ValueError:
                    toks = arg.split()
                if len(toks) < 5:
                    raise NetlistError(ln, "pot needs: *@ pot RV1 a wiper b 100k [taper=log] [pos=0.5] [label=\"Gain\"]")
                pname = toks[0].upper()
                if pname in pots or pname in names:
                    raise NetlistError(ln, f"{pname} is defined twice")
                kv = dict(t.split("=", 1) for t in toks[5:] if "=" in t)
                taper = kv.get("taper", "lin").lower()
                if taper not in ("lin", "log", "revlog", "a", "b", "c", "audio", "rev"):
                    raise NetlistError(ln, f"pot taper '{taper}' must be lin, log or revlog")
                taper = {"a": "log", "audio": "log", "b": "lin", "c": "revlog", "rev": "revlog"}.get(taper, taper)
                try:
                    pos = float(kv.get("pos", 0.5))
                except ValueError:
                    raise NetlistError(ln, f"pot position must be a number from 0 to 1, got {kv.get('pos')}")
                if not 0 <= pos <= 1:
                    raise NetlistError(ln, f"pot position must be 0..1, got {pos}")
                p = Pot(pname, _norm_node(toks[1]), _norm_node(toks[2]), _norm_node(toks[3]),
                        parse_value(_subst(toks[4], gscope, ln), ln), taper, pos, kv.get("label", pname))
                if p.value <= 0:
                    raise NetlistError(ln, f"pot {pname} needs a positive value")
                pots[pname] = p
                ra, rb = _pot_halves(p)
                for half, (a_, b_, r_) in (("A", (p.a, p.w, ra)), ("B", (p.w, p.b, rb))):
                    elements.append(Element(pname + "_" + half, "R", [a_, b_], r_, None, {"pot_half": half}, ln, pname))
                    names.add(pname + "_" + half)
            # unknown *@ keys are ignored so newer netlists still load
            continue
        if line.startswith("*"):
            if first_content_line and not title and sub is None:
                title = line.lstrip("* ").strip()
            first_content_line = False
            continue
        line = _strip_inline_comment(line)
        if not line:
            continue
        if sub is None:
            line = _subst(line, gscope, ln)
        lower = line.lower()
        if lower.startswith("."):
            directive = lower.split()[0]
            if directive == ".model":
                toks = line.split(None, 2)
                if len(toks) < 3:
                    raise NetlistError(ln, ".model needs a name and a type, e.g. '.model D1N D(IS=2n N=1.8)'")
                mname = toks[1].upper()
                m = re.match(r"([A-Za-z]+)\s*\(?(.*?)\)?\s*$", toks[2], re.S)
                mtype = (m.group(1) if m else toks[2]).upper()
                if mtype == "VDMOS":
                    raise NetlistError(ln, f"model {mname}: VDMOS power-MOSFET models aren't simulated. Use a "
                                           "level-1 '.model NAME NMOS(VTO= KP= LAMBDA= RD= CGS= CGD=)' (or PMOS): "
                                           "for the same square law set KP to twice the VDMOS Kp, or use a built-in "
                                           "part (2N7000, BS170, BS250, IRF510, IRF9510)")
                if mtype not in _MODEL_TYPES:
                    raise NetlistError(ln, f"model type '{mtype}' isn't supported (D, NPN, PNP, NJF, PJF, NMOS, PMOS, OPAMP)")
                params = {}
                for k, v in _PARAM_RE.findall(m.group(2) if m else ""):
                    if k.upper() in ("MFG", "TYPE", "ICRATING", "VCEO", "ICRATE", "VCEO_", "IAVE", "VPK") and not re.match(r"^[+-]?[\d.]", v):
                        continue
                    try:
                        params[k.upper()] = parse_value(v, ln)
                    except NetlistError:
                        if not re.match(r"^[+-]?[\d.]", v):
                            continue  # descriptive text such as mfg=Philips
                        raise
                for key in ("IS", "N", "BF", "BR", "NF", "NR", "BETA", "A0", "GBW"):
                    if key in params and params[key] <= 0:
                        raise NetlistError(ln, f"model {mname}: {key}={params[key]:g} isn't physical (it must be positive)")
                if mtype in ("NMOS", "PMOS"):
                    params = _mos_model_params(mname, params, ln)
                models[mname] = Model(mname, _MODEL_TYPES[mtype], params)
            elif directive in (".include", ".inc", ".lib"):
                raise NetlistError(ln, _include_message(directive))
            elif directive == ".end":
                break
            # .op .ac .tran .options .global ...: ignored, tools pick analyses
            first_content_line = False
            continue
        if first_content_line and line[0].upper() not in _KINDS and sub is None:
            title = title or line
            first_content_line = False
            continue
        first_content_line = False

        toks = line.split()
        name = toks[0].upper()
        kind = name.rsplit(".", 1)[-1][:1]
        if kind not in _KINDS:
            raise NetlistError(ln, f"'{toks[0]}' isn't a supported element (R C L V I D Q J M X E G S)")
        if kind == "X":
            atoks = _split_assign(line)
            pos = [t for t in atoks[1:] if "=" not in t and t.lower() not in ("params:", "param:")]
            target = pos[-1].upper() if pos else ""
            if target in subckts:
                if name in instances or name in names:
                    raise NetlistError(ln, f"{name} is defined twice")
                instances.add(name)
                given = _assignments([t for t in atoks[1:] if "=" in t], gscope, ln, name)
                kids = [_norm_node(t) for t in pos[:-1]]
                work[wi:wi] = _expand(name, kids, subckts[target], given, gscope, subckts, 1, (), ln)
                continue
        if name in names:
            raise NetlistError(ln, f"{name} is defined twice")
        names.add(name)
        extra = {"sub": sub} if sub else {}

        if kind in "RCL":
            if len(toks) < 4:
                raise NetlistError(ln, f"{name} needs two nodes and a value, e.g. '{name} in out 10k'")
            val = parse_value(toks[3], ln)
            if val <= 0:
                raise NetlistError(ln, f"{name} must have a positive value")
            params = dict(extra)
            for t in toks[4:]:
                if t.lower().startswith("ic="):
                    params["ic"] = parse_value(t[3:], ln)
            elements.append(Element(name, kind, [_norm_node(toks[1]), _norm_node(toks[2])], val, None, params, ln))
        elif kind in "VI":
            el = _parse_source(name, kind, toks, line, ln)
            el.params.update(extra)
            elements.append(el)
        elif kind == "D":
            if len(toks) < 4:
                raise NetlistError(ln, f"{name} needs anode, cathode and a model, e.g. '{name} a k 1N4148'")
            elements.append(Element(name, "D", [_norm_node(toks[1]), _norm_node(toks[2])], None, toks[3].upper(), dict(extra), ln))
        elif kind in "QJ":
            if len(toks) < 5:
                ex = "c b e 2N3904" if kind == "Q" else "d g s J201"
                raise NetlistError(ln, f"{name} needs three nodes and a model, e.g. '{name} {ex}'")
            elements.append(Element(name, kind, [_norm_node(t) for t in toks[1:4]], None, toks[4].upper(), dict(extra), ln))
        elif kind == "M":
            elements.append(_parse_mosfet(name, _split_assign(line), ln, extra))
        elif kind == "X":
            if len(toks) != 7:
                last = toks[-1].upper() if len(toks) > 1 else ""
                raise NetlistError(ln, f"{name}: '{last}' isn't a .subckt in this netlist, and op-amps take 5 pins "
                                       f"then a part: '{name} in+ in- vcc vee out TL072' (got {len(toks) - 1} fields)")
            elements.append(Element(name, "X", [_norm_node(t) for t in toks[1:6]], None, toks[6].upper(), dict(extra), ln))
        elif kind in "EG":
            if len(toks) < 6:
                raise NetlistError(ln, f"{name} needs n+ n- nc+ nc- gain")
            elements.append(Element(name, kind, [_norm_node(t) for t in toks[1:5]], parse_value(toks[5], ln), None, dict(extra), ln))
        elif kind == "S":
            if len(toks) < 4 or toks[3].lower() not in ("on", "off"):
                raise NetlistError(ln, f"{name} needs two nodes and on/off, e.g. '{name} a b on'")
            elements.append(Element(name, "S", [_norm_node(toks[1]), _norm_node(toks[2])], None, None, {"on": toks[3].lower() == "on", **extra}, ln))

    if not elements:
        raise NetlistError(0, "the netlist has no components")

    # resolve models: explicit .model first, then built-in parts
    needed = {"D": "D", "Q": ("NPN", "PNP"), "J": ("NJF", "PJF"), "X": "OPAMP", "M": ("NMOS", "PMOS")}
    for e in elements:
        if e.kind not in needed:
            continue
        if e.model not in models:
            part = parts_data_lookup(e.model)
            if part is None:
                raise NetlistError(e.line, f"{e.name}: unknown part '{e.model}'. Add a .model line or use a built-in part")
            models[e.model] = Model(e.model, part["type"], dict(part["params"]))
        mt = models[e.model].type
        ok = needed[e.kind]
        if (isinstance(ok, tuple) and mt not in ok) or (isinstance(ok, str) and mt != ok):
            raise NetlistError(e.line, f"{e.name}: '{e.model}' is a {mt} model, which doesn't fit a {e.kind} element")

    names = {e.name for e in elements}
    if inp and inp not in names:
        raise NetlistError(0, f"*@ in names '{inp}', but there is no source with that name")
    if inp and inp[0] not in "VI":
        raise NetlistError(0, f"*@ in must name a voltage or current source, not {inp}")
    for s in supplies:
        if s not in names:
            raise NetlistError(0, f"*@ supply names '{s}', but there is no source with that name")
    all_nodes = {n for e in elements for n in e.nodes}
    if out and out not in all_nodes:
        raise NetlistError(0, f"*@ out names node '{out}', which nothing connects to")
    for r in ratings:
        if r not in names:
            raise NetlistError(0, f"*@ rating names '{r}', which isn't a part")
    if "0" not in all_nodes:
        raise NetlistError(0, "nothing connects to ground (node 0 or gnd)")

    return Circuit(title or "Untitled circuit", elements, models, pots, inp, out, supplies,
                   specs, tol, ratings, text)
