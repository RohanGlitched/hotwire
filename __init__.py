"""Hotwire: an analog circuit design bench for Hermes.

Hermes writes a SPICE netlist from plain English, proves it against measurable
specs with Hotwire's own simulator, lets you hear it, and hands you a parts list
and a stripboard layout you can solder.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from . import schemas, tools

logger = logging.getLogger(__name__)

_SKILLS = Path(__file__).parent / "skills"

# Strong terms mean electronics on their own; weak ones (filter, boost, bias…) also appear
# in ordinary chat, so they only count when two of them, or one strong term, show up.
_STRONG = re.compile(
    r"\b(circuits?|schematics?|netlists?|spice|resistors?|capacitors?|inductors?|transistors?|mosfets?|"
    r"jfets?|bjts?|op-?amps?|diodes?|leds?|breadboard|stripboard|veroboard|perfboard|pcb|kicad|"
    r"ltspice|ngspice|soldering|tl07[12]|ne5532|2n3904|2n3906|1n4148|1n914|ohms?|"
    r"guitar pedal|fuzz pedal|overdrive pedal|effects? pedal|stompbox|tone stack|wah)\b", re.I)
_WEAK = re.compile(
    r"\b(pedal|fuzz|overdrive|preamp|amplifier|oscillator|solder|9 ?v|volts?|milliamps?|ma|khz|"
    r"guitar|treble|low-?pass|high-?pass|band-?pass)\b", re.I)


def _is_electronics(text: str) -> bool:
    if _STRONG.search(text):
        return True
    return len({m.group(0).lower() for m in _WEAK.finditer(text)}) >= 2


_HINT = (
    "Hotwire circuit tools are available (circuit_check, circuit_audition, circuit_tune, "
    "circuit_explain, circuit_yield, circuit_bom, circuit_build, circuit_library). For circuit design or "
    "analysis, load the skill 'hotwire:analog-design' first and prove every claim with a tool result."
)


def _hint_hook(ctx):
    def pre_llm_call(*args, **kwargs):
        try:
            if not ctx.get_config("design_hint", default=True):
                return None
            text = " ".join(
                str(kwargs.get(k) or "") for k in ("user_message", "user_task", "message", "prompt")
            )
            if not text:
                for m in reversed(kwargs.get("messages") or []):
                    if isinstance(m, dict) and m.get("role") == "user":
                        content = m.get("content")
                        text = content if isinstance(content, str) else str(content)
                        break
            if text and _is_electronics(text):
                return {"context": _HINT}
        except Exception:  # a hint must never break a turn
            logger.debug("hotwire hint hook skipped", exc_info=True)
        return None

    return pre_llm_call


def register(ctx):
    try:
        tools.configure(sample_rate=ctx.get_config("sample_rate", default=48000))
    except Exception:
        logger.debug("hotwire: using the default sample rate", exc_info=True)
    handlers = {
        "circuit_check": tools.circuit_check,
        "circuit_simulate": tools.circuit_simulate,
        "circuit_audition": tools.circuit_audition,
        "circuit_tune": tools.circuit_tune,
        "circuit_explain": tools.circuit_explain,
        "circuit_match": tools.circuit_match,
        "circuit_yield": tools.circuit_yield,
        "circuit_bom": tools.circuit_bom,
        "circuit_build": tools.circuit_build,
        "circuit_export": tools.circuit_export,
        "circuit_library": tools.circuit_library,
    }
    for schema in schemas.ALL:
        ctx.register_tool(name=schema["name"], toolset="hotwire", schema=schema,
                          handler=handlers[schema["name"]])

    if _SKILLS.is_dir():
        for child in sorted(_SKILLS.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                ctx.register_skill(child.name, skill_md)

    ctx.register_hook("pre_llm_call", _hint_hook(ctx))
    ctx.register_command(
        "hotwire",
        handler=tools.slash_command,
        description="Analog circuit bench: /hotwire, /hotwire library, /hotwire designs, /hotwire check <file>",
        args_hint="[library|designs|check <file.cir>|<library id>]",
    )
