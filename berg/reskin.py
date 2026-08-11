"""
reskin.py
=========
Contamination-control variants of the Berg trust game (see RESKIN_PLAN.md).

Three probes:
  A. Surface re-skin  — neutral labels/units, same economics (threat T1).
  B. Parameter shift  — endowment/multiplier changed so memorized Berg values
                        (mean sent $5.16 / returned $4.66) are useless; the
                        selfish SPE (0,0) is invariant (threat T3).
  C. Recognition probe — separately ask the model to identify the paradigm;
                        measures contamination instead of assuming it (T1).

No API calls in this module; probe C builders return prompts for the caller.
"""

from __future__ import annotations

import json
import re

from berg_trust_game import GameConfig

# --------------------------------------------------------------------------- #
# Variants (Probes A & B)                                                     #
# --------------------------------------------------------------------------- #
RESKINS: dict[str, GameConfig] = {
    # Faithful Berg surface + parameters (the existing runs).
    "canonical": GameConfig(),
    # Probe A: surface only — canonical parameters, neutral labels/units.
    "reskin": GameConfig(
        unit_word="points", currency="points",
        investor_label="Blue Player", trustee_label="Green Player",
    ),
    # Probe B: parameters only — canonical labels, shifted economics.
    "shifted": GameConfig(endowment=12, multiplier=2),
    # A+B combined: maximal distance from the memorized study.
    "reskin_shifted": GameConfig(
        endowment=12, multiplier=2,
        unit_word="points", currency="points",
        investor_label="Blue Player", trustee_label="Green Player",
    ),
    # Probe D (strong-novel): a fully renarrated surface — new scenario, new
    # role names, new units, new numbers, new multiplier — that keeps the
    # economic structure (a one-shot sequential transfer-and-return with a
    # multiplied transfer) while removing every cue that the paradigm is Berg.
    # This is the far end of the surface gradient in H2/H3. The endowment 20
    # and multiplier 4 keep the interior interesting (SPE still (0,0)).
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
    # Social but NON-PARADIGMATIC labels at the canonical economics. Depositor and
    # Custodian carry a social obligation to return, like Investor and Trustee, but
    # they are not the labels of the published paradigm and are not the labels of
    # another famous experiment. Against canonical (identical payoffs, identical
    # units, only the role words differ) this separates the social meaning of the
    # words from the identity of the study: semantic priming predicts
    # canonical ~ social_nonparadigm > reskin, while a recollection of the
    # paradigm predicts canonical > social_nonparadigm ~ reskin.
    "social_nonparadigm": GameConfig(
        investor_label="Depositor", trustee_label="Custodian",
    ),
    # Structure-matched control for strong_novel: canonical wording at the SAME
    # economics (endowment 20, x4). strong_novel vs this isolates the narrative
    # and label effect at the far end (a clean H2 point), while this vs canonical
    # isolates the reaction to the (20, x4) structure (a clean H3 point).
    "strong_control": GameConfig(endowment=20, multiplier=4),
}

# H2 pairs: (recognizable wording, de-recognized wording) at MATCHED structure.
# Within a pair the incentives are identical and only the surface changes.
SURFACE_PAIRS = [
    ("canonical", "reskin"),            # structure (10, x3)
    ("shifted", "reskin_shifted"),      # structure (12, x2)
    ("strong_control", "strong_novel"), # structure (20, x4)
]
# Three-way ladder at the canonical economics (10, x3) that separates the social
# meaning of the role words from the identity of the paradigm.
LABEL_LADDER = ("canonical", "social_nonparadigm", "reskin")
# The recognizable wording of each structure, for the H3 reaction contrast
# (canonical -> shifted -> strong_control changes only the economics).
STRUCTURE_ANCHORS = ("canonical", "shifted", "strong_control")

# Surface markers that identify the canonical study / benchmark. Re-skinned
# prompts must contain NONE of these (checked in test_reskin.py).
CANONICAL_MARKERS = [
    "investor", "trustee", "trustor", "sender", "receiver",
    "trust game", "investment game",
    "berg", "dickhaut", "mccabe",
    "dollar", "$", "5.16", "4.66",
    "tripl",           # "triple"/"tripled"/"tripling"
]


def find_markers(text: str, markers: list[str] = CANONICAL_MARKERS) -> list[str]:
    """Return the canonical markers present in `text` (case-insensitive)."""
    low = text.lower()
    return [m for m in markers if m in low]


# --------------------------------------------------------------------------- #
# Recognition probe (Probe C)                                                 #
# --------------------------------------------------------------------------- #
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
    """(system, user) prompts asking the model to identify the paradigm.

    Pass the exact rules text the agents saw (e.g. from
    berg_trust_game._rules_block(cfg)) so the probe measures recognition of
    the very stimulus used in the behavioral runs. Run as SEPARATE calls —
    never inside a behavioral session.
    """
    user = (
        "Below are the instructions given to participants in a study.\n\n"
        "-----\n" + rules_block + "\n-----\n\n"
        "Question: Do these instructions correspond to a known experimental "
        "paradigm in the research literature? If yes, name it, give the "
        "canonical reference, and state any well-known quantitative results "
        "from human participants.\n" + _RECOGNITION_JSON
    )
    return RECOGNITION_SYSTEM, user


# Keywords indicating the model identified the canonical study/paradigm.
_RECOGNITION_KEYWORDS = [
    "berg", "dickhaut", "mccabe", "trust game", "investment game",
]


def score_recognition(completion: str) -> dict:
    """Parse a recognition-probe completion into a scored record.

    Scoring is deliberately conservative: `recognized_paradigm` is True only
    if the self-report is true OR a canonical keyword appears; `named_study`
    requires an explicit Berg/Dickhaut/McCabe mention; `leaked_benchmark`
    flags reproduction of the human benchmark numbers.
    """
    m = re.search(r"\{.*\}", completion, re.DOTALL)
    data = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = {}
    text_low = completion.lower()
    keyword_hit = any(k in text_low for k in _RECOGNITION_KEYWORDS)
    # Approximate human-benchmark knowledge: quantitative claims about what
    # participants send/return (e.g. "send 30-50% of their endowment",
    # "about $5"), even without the exact Berg means. Pilot 2026-07-02 showed
    # models cite these freely while never reproducing 5.16/4.66 exactly.
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
