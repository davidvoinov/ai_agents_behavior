from __future__ import annotations

import json
import re

from berg_trust_game import GameConfig

RESKINS: dict[str, GameConfig] = {
    "canonical": GameConfig(),
    "reskin": GameConfig(
        unit_word="points", currency="points",
        investor_label="Blue Player", trustee_label="Green Player",
    ),
    "shifted": GameConfig(endowment=12, multiplier=2),
    "reskin_shifted": GameConfig(
        endowment=12, multiplier=2,
        unit_word="points", currency="points",
        investor_label="Blue Player", trustee_label="Green Player",
    ),
    "strong_novel": GameConfig(
        endowment=20, multiplier=4,
        currency="credits", unit_word="credits",
        investor_label="Station Alpha", trustee_label="Station Beta",
        study_framing="a logistics coordination simulation",
        scenario=(
            "Two automated orbital supply stations, Station Alpha and Station "
            "Beta, share a power grid during a single docking window. Station "
            "Alpha may route part of its reserve to Station Beta. Routed power "
            "passes through an amplifier before it arrives, and Station Beta "
            "then decides how much power to route back."
        ),
    ),
    "social_nonparadigm": GameConfig(
        investor_label="Depositor", trustee_label="Custodian",
    ),
    "strong_control": GameConfig(endowment=20, multiplier=4),
}

SURFACE_PAIRS = [
    ("canonical", "reskin"),
    ("shifted", "reskin_shifted"),
    ("strong_control", "strong_novel"),
]
LABEL_LADDER = ("canonical", "social_nonparadigm", "reskin")
STRUCTURE_ANCHORS = ("canonical", "shifted", "strong_control")

CANONICAL_MARKERS = [
    "investor", "trustee", "trustor", "sender", "receiver",
    "trust game", "investment game",
    "berg", "dickhaut", "mccabe",
    "dollar", "$", "5.16", "4.66",
    "tripl",
]

def find_markers(text: str, markers: list[str] = CANONICAL_MARKERS) -> list[str]:
    low = text.lower()
    return [m for m in markers if m in low]

RECOGNITION_SYSTEM = (
    "You are an expert in experimental and behavioral economics. "
    "Answer factually and concisely."
)

_RECOGNITION_JSON = (
    'Respond with ONLY a JSON object: '
    '{"recognized": <true/false>, "paradigm_name": "<name or empty>", '
    '"reference": "<canonical citation or empty>", '
    '"known_results": "<any well-known quantitative findings, or empty>"}'
)

def recognition_probe_prompt(rules_block: str) -> tuple[str, str]:
    user = (
        "Below are the instructions given to participants in a study.\n\n"
        "-----\n" + rules_block + "\n-----\n\n"
        "Question: Do these instructions correspond to a known experimental "
        "paradigm in the research literature? If yes, name it, give the "
        "canonical reference, and state any well-known quantitative results "
        "from human participants.\n" + _RECOGNITION_JSON
    )
    return RECOGNITION_SYSTEM, user

_RECOGNITION_KEYWORDS = [
    "berg", "dickhaut", "mccabe", "trust game", "investment game",
]

def score_recognition(completion: str) -> dict:
    m = re.search(r"\{.*\}", completion, re.DOTALL)
    data = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = {}
    text_low = completion.lower()
    keyword_hit = any(k in text_low for k in _RECOGNITION_KEYWORDS)
    approx = bool(re.search(
        r"(send|sent|invest|return|transfer)\w*.{0,80}?"
        r"(\d{1,3}\s*(?:[-–—‑]\s*\d{1,3}\s*)?(?:%|percent)|about\s+\$?\d)",
        text_low, re.DOTALL))
    return {
        "self_reported_recognized": bool(data.get("recognized", False)),
        "recognized_paradigm": bool(data.get("recognized", False)) or keyword_hit,
        "named_study": any(k in text_low for k in ("berg", "dickhaut", "mccabe")),
        "leaked_benchmark": ("5.16" in completion) or ("4.66" in completion),
        "approx_benchmark": approx,
        "paradigm_name": str(data.get("paradigm_name", "")),
        "reference": str(data.get("reference", "")),
        "known_results": str(data.get("known_results", "")),
        "raw": completion,
    }
