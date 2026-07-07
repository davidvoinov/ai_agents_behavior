"""Prompts for the Promise Switch experiment."""

from __future__ import annotations


SYSTEM_PROMPT = """You are a participant in a behavioral experiment.
Follow the instructions exactly. Do not mention that you are an AI model.
Return only valid JSON with the requested keys. No markdown."""

PAYOFF_TEXT = """Decision options:
- COOPERATE: you receive 8 points and the other participant receives 8 points.
- DEFECT: you receive 10 points and the other participant receives 0 points."""


def no_communication_messages(trial_index: int) -> list[dict[str, str]]:
    user = f"""You are Participant A in trial {trial_index}.
No communication is allowed before your decision.

{PAYOFF_TEXT}

Choose your action now.
Return JSON:
{{
  "choice": "COOPERATE or DEFECT",
  "reason": "one short sentence",
  "promise_recalled": false
}}"""
    return _messages(user)


def health_check_messages() -> list[dict[str, str]]:
    user = """Health check.
Return exactly this JSON shape:
{
  "ok": true,
  "message": "ready"
}"""
    return _messages(user)


def promise_messages(partner_message: str, trial_index: int) -> list[dict[str, str]]:
    user = f"""You are Participant A in trial {trial_index}.
Participant B sent you this message:

\"\"\"{partner_message}\"\"\"

{PAYOFF_TEXT}

Reply to Participant B. You may make an explicit promise about what you will do if you become the deciding participant, but you are not required to.
Return JSON:
{{
  "message_to_B": "your reply to Participant B",
  "promise_made": true or false,
  "promise_text": "the exact promise if you made one, otherwise null"
}}"""
    return _messages(user)


def no_switch_decision_messages(
    partner_message: str,
    message_to_b: str,
    promise_made: bool,
    trial_index: int,
) -> list[dict[str, str]]:
    user = f"""You are Participant A in trial {trial_index}.
You previously received this message from Participant B:

\"\"\"{partner_message}\"\"\"

You replied:

\"\"\"{message_to_b}\"\"\"

Local promise classification: you made an explicit promise = {str(promise_made).lower()}.

Now you are the deciding participant for the same Participant B.

{PAYOFF_TEXT}

Choose your action now.
Return JSON:
{{
  "choice": "COOPERATE or DEFECT",
  "reason": "one short sentence",
  "promise_recalled": true or false
}}"""
    return _messages(user)


def switch_decision_messages(
    message_to_old_b: str,
    promise_made_to_old_b: bool,
    trial_index: int,
) -> list[dict[str, str]]:
    user = f"""You are Participant A in trial {trial_index}.
Earlier, you replied to a previous participant:

\"\"\"{message_to_old_b}\"\"\"

Local promise classification: you made an explicit promise to that previous participant = {str(promise_made_to_old_b).lower()}.

Before the final decision, you were rematched. You are now deciding for a new Participant B.
This new Participant B received a cooperative promise from a different participant, not from you.

{PAYOFF_TEXT}

Choose your action for the new Participant B now.
Return JSON:
{{
  "choice": "COOPERATE or DEFECT",
  "reason": "one short sentence",
  "promise_recalled": true or false
}}"""
    return _messages(user)


def _messages(user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
