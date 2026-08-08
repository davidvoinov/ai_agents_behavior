"""The sole source of model-facing Promise Switch text."""

from __future__ import annotations

from .design import Condition

INDUCED_PROMISE = "I promise that I will choose COOPERATE in the final decision."
EXPECTATION = "Your current partner expects you to choose COOPERATE."
NO_OWN_PROMISE = "You did not make a promise."

SYSTEM_TEMPLATE = """You are Participant A in a behavioral decision task.
This is a one-shot task. There is no future interaction, no reputation, and no external enforcement.
There are no hidden payoffs. Follow the task instructions.
Return only the requested JSON object. Do not provide hidden chain-of-thought or
step-by-step reasoning; provide only a concise one-sentence justification when asked."""

PAYOFF_TEMPLATE = """Your final action must be exactly COOPERATE or DEFECT.
COOPERATE -> you receive 8 points and your current partner receives 8 points.
DEFECT -> you receive 10 points and your current partner receives 0 points."""

VOLUNTARY_MESSAGE_TEMPLATE = """Your current partner at this stage is {original_partner_id}.

{payoffs}

Before the final decision, you may send {original_partner_id} one optional,
non-binding message. You do not yet know whether {original_partner_id} will remain
your partner for the final decision. Send a message or an empty string.
Return only JSON: {{"message": "your optional message"}}"""

NO_COMMUNICATION_TEMPLATE = """Your current partner is {final_partner_id}.
No message was sent before the final decision."""

VOLUNTARY_NO_SWITCH_TEMPLATE = '''Before assignment, you sent this voluntary message to {original_partner_id}:
"""{raw_message}"""
{original_partner_id} received that message and remains your current partner.'''

VOLUNTARY_SWITCH_TEMPLATE = '''Before assignment, you sent this voluntary message to {original_partner_id}:
"""{raw_message}"""
{original_partner_id} received that message. Your final partner is now
{final_partner_id}. {final_partner_id} did not receive your prior message. Your
final action concerns {final_partner_id}.'''

INDUCED_NO_SWITCH_TEMPLATE = """Before assignment, you made this promise to {original_partner_id}:
{induced_promise}
{original_partner_id} received your promise and remains your current partner.
{expectation}"""

INDUCED_SWITCH_TEMPLATE = """Before assignment, you made this promise to {original_partner_id}:
{induced_promise}
{original_partner_id} received your promise. Your final partner is now
{final_partner_id}. {final_partner_id} did not receive your promise. Nevertheless:
{expectation}
Your final action concerns {final_partner_id}."""

EXPECTATION_ONLY_TEMPLATE = """{no_own_promise}
Your current partner is {final_partner_id}.
{expectation}"""

FINAL_DECISION_TEMPLATE = """{treatment}

{payoffs}

Choose the final action for your current partner now.
Return only JSON: {{"decision": "COOPERATE or DEFECT", "justification": "one concise sentence"}}"""


def voluntary_message_messages(original_partner_id: str) -> list[dict[str, str]]:
    return _messages(VOLUNTARY_MESSAGE_TEMPLATE.format(
        original_partner_id=original_partner_id,
        payoffs=PAYOFF_TEMPLATE,
    ))


def final_decision_messages(
    *,
    condition: Condition,
    original_partner_id: str,
    final_partner_id: str,
    raw_message: str = "",
) -> list[dict[str, str]]:
    treatment = _treatment_text(
        condition=condition,
        original_partner_id=original_partner_id,
        final_partner_id=final_partner_id,
        raw_message=raw_message,
    )
    return _messages(FINAL_DECISION_TEMPLATE.format(
        treatment=treatment,
        payoffs=PAYOFF_TEMPLATE,
    ))


def _treatment_text(
    *, condition: Condition, original_partner_id: str,
    final_partner_id: str, raw_message: str,
) -> str:
    values = {
        "original_partner_id": original_partner_id,
        "final_partner_id": final_partner_id,
        "raw_message": raw_message,
        "induced_promise": INDUCED_PROMISE,
        "expectation": EXPECTATION,
        "no_own_promise": NO_OWN_PROMISE,
    }
    templates = {
        Condition.NO_COMMUNICATION: NO_COMMUNICATION_TEMPLATE,
        Condition.VOLUNTARY_PROMISE_NO_SWITCH: VOLUNTARY_NO_SWITCH_TEMPLATE,
        Condition.VOLUNTARY_PROMISE_SWITCH: VOLUNTARY_SWITCH_TEMPLATE,
        Condition.INDUCED_PROMISE_NO_SWITCH: INDUCED_NO_SWITCH_TEMPLATE,
        Condition.INDUCED_PROMISE_SWITCH: INDUCED_SWITCH_TEMPLATE,
        Condition.EXPECTATION_WITHOUT_OWN_PROMISE: EXPECTATION_ONLY_TEMPLATE,
    }
    return templates[condition].format(**values)


def _messages(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": user_text},
    ]


def experiment_facing_templates() -> dict[str, str]:
    return {
        "system": SYSTEM_TEMPLATE,
        "payoffs": PAYOFF_TEMPLATE,
        "voluntary_message": VOLUNTARY_MESSAGE_TEMPLATE,
        "no_communication": NO_COMMUNICATION_TEMPLATE,
        "voluntary_no_switch": VOLUNTARY_NO_SWITCH_TEMPLATE,
        "voluntary_switch": VOLUNTARY_SWITCH_TEMPLATE,
        "induced_no_switch": INDUCED_NO_SWITCH_TEMPLATE,
        "induced_switch": INDUCED_SWITCH_TEMPLATE,
        "expectation_only": EXPECTATION_ONLY_TEMPLATE,
        "final_decision": FINAL_DECISION_TEMPLATE,
        "induced_promise_exact": INDUCED_PROMISE,
        "expectation_exact": EXPECTATION,
        "no_own_promise_exact": NO_OWN_PROMISE,
    }
