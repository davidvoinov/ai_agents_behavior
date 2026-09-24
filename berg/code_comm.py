from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Callable, Optional

ChatFn = Callable[[str, str], str]

JUDGE_SYSTEM = (
    "You are a careful research assistant coding transcripts from an economics "
    "experiment. Follow the codebook exactly and code only what the transcript "
    "states. Do not infer intentions that are not expressed."
)

_JUDGE_JSON = (
    'Respond with ONLY this JSON object: {"explicit_agreement": <true|false>, '
    '"agreement_type": "<none|amount|rule>", "equality_framing": <true|false>, '
    '"promise_investor": <true|false>, "promise_trustee": <true|false>, '
    '"agreed_send": <integer|null>, "agreed_return": <integer|null>, '
    '"conditions": "<short text or empty>"}'
)

CODEBOOK_BRIEF = (
    'Code this transcript with codebook v2. '
    'explicit_agreement is True only if one party states a plan for the transfer, the return, or both, given as a number or a proportional rule, AND the other party affirms that same plan by agreeing, committing to perform it, or restating it approvingly. '
    'An unanswered proposal is False wherever it appears, including the final message, because silence is not affirmation. A counter-proposal that is itself unanswered is False. '
    "Goodwill without a number or a rule, such as 'I will be fair' or 'I will return a generous amount', is never a plan or an affirmation. "
    'A plan for the transfer alone counts if affirmed. A plan to transfer nothing counts if both state it. '
    'agreement_type in strict priority order: amount if the agreed plan fixes at least one specific number; otherwise rule if a proportional or contingent rule is agreed; otherwise none when explicit_agreement is False. '
    "equality_framing is True if the parties justify the plan by equality of the final payoffs, for example 'we both end with 14'. It is independent of the type. "
    "promise_investor and promise_trustee are True only if all three hold for that party: first person about their own future action with a committing verb (will, I'll, commit, promise, guarantee, honor, plan to, intend to, agree to, proceed with); a specific quantity or an explicit proportional rule, where a floor such as 'at least 6' qualifies; and no hedge of uncertainty. "
    "These do NOT qualify: considering, thinking of, might, maybe, could, perhaps, leaning toward, open to, aim to, hope to, would be happy to, and hypothetical framing such as 'let us say I send 5'. "
    "A conditional of the form 'if you send X, I will return Y' does qualify. "
    'agreed_send and agreed_return: record a number only when the party who would perform it stated it and the other did not object. Use null for a range, for a rule with no number, and whenever explicit_agreement is False.'
)

CATEGORICAL = ["explicit_agreement", "agreement_type", "equality_framing",
               "promise_investor", "promise_trustee"]

def judge_prompt(transcript: str) -> tuple[str, str]:
    user = (
        CODEBOOK_BRIEF + "\n\nTranscript:\n-----\n" + transcript.strip()
        + "\n-----\n\n" + _JUDGE_JSON
    )
    return JUDGE_SYSTEM, user

def _coerce(data: dict) -> dict:
    def b(x): return bool(x) if isinstance(x, bool) else str(x).strip().lower() in ("true", "yes", "1")
    def i(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None
    at = str(data.get("agreement_type", "none")).strip().lower()
    if at == "equal_split":
        at = "amount"
    if at not in ("none", "amount", "rule"):
        at = "none"
    return {
        "explicit_agreement": b(data.get("explicit_agreement", False)),
        "agreement_type": at,
        "equality_framing": b(data.get("equality_framing", False)),
        "promise_investor": b(data.get("promise_investor", False)),
        "promise_trustee": b(data.get("promise_trustee", False)),
        "agreed_send": i(data.get("agreed_send")),
        "agreed_return": i(data.get("agreed_return")),
        "conditions": str(data.get("conditions", "")).strip(),
        "coded_clean": True,
    }

JUDGE_SCHEMA = {
    "name": "communication_codes",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "explicit_agreement": {"type": "boolean"},
            "agreement_type": {"enum": ["amount", "rule", "vague", "none"]},
            "equality_framing": {"type": "boolean"},
            "promise_investor": {"type": "boolean"},
            "promise_trustee": {"type": "boolean"},
            "agreed_send": {"type": ["integer", "null"]},
            "agreed_return": {"type": ["integer", "null"]},
            "conditions": {"type": "string"},
        },
        "required": ["explicit_agreement", "agreement_type", "equality_framing",
                     "promise_investor", "promise_trustee", "agreed_send",
                     "agreed_return", "conditions"],
        "additionalProperties": False,
    },
}

def parse_codes(text: str) -> dict:
    t = text or ""
    cands = []
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        cands.append(m.group(1))
    end = t.rfind("}")
    if end != -1:
        start = t.find("{")
        if start != -1 and end > start:
            cands.append(t[start:end + 1])
        last = t.rfind("{", 0, end)
        if last != -1 and last != start:
            cands.append(t[last:end + 1])
    for c in cands:
        for attempt in (c, c.replace("{{", "{").replace("}}", "}"),
                        re.sub(r'\\(?!["\\/bfnrtu])', "", c)):
            try:
                data = json.loads(attempt)
            except Exception:
                continue
            if isinstance(data, dict):
                return _coerce(data)
    return {"coded_clean": False, "raw": t[:1200]}

def code_transcript(chat_fn: ChatFn, transcript: str) -> dict:
    sysp, user = judge_prompt(transcript)
    return parse_codes(chat_fn(sysp, user))

def cohen_kappa(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    labels = set(ca) | set(cb)
    pe = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

def reliability(human: list[dict], judge: list[dict]) -> dict:
    out = {}
    for var in CATEGORICAL:
        out[var] = round(cohen_kappa([h.get(var) for h in human],
                                     [j.get(var) for j in judge]), 3)
    return out

def compliance(codes: dict, sent: Optional[int], returned: Optional[int],
               tol: int = 0) -> dict:
    out = {"investor_complied": None, "trustee_complied": None}
    if codes.get("agreed_send") is not None and sent is not None:
        out["investor_complied"] = abs(sent - codes["agreed_send"]) <= tol
    if codes.get("agreed_return") is not None and returned is not None:
        out["trustee_complied"] = abs(returned - codes["agreed_return"]) <= tol
    return out

def incidence_compliance_table(rows: list[dict], tol: int = 0) -> dict:
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "agreements": 0,
                                                "inv_ok": 0, "inv_tot": 0,
                                                "tru_ok": 0, "tru_tot": 0})
    for r in rows:
        cell = r["cell"]; codes = r["codes"]
        a = agg[cell]
        a["n"] += 1
        if not codes.get("explicit_agreement"):
            continue
        a["agreements"] += 1
        comp = compliance(codes, r.get("sent"), r.get("returned"), tol)
        if comp["investor_complied"] is not None:
            a["inv_tot"] += 1; a["inv_ok"] += int(comp["investor_complied"])
        if comp["trustee_complied"] is not None:
            a["tru_tot"] += 1; a["tru_ok"] += int(comp["trustee_complied"])
    out = {}
    for cell, a in agg.items():
        out[cell] = {
            "n": a["n"],
            "incidence": round(a["agreements"] / a["n"], 3) if a["n"] else 0.0,
            "investor_compliance": round(a["inv_ok"] / a["inv_tot"], 3) if a["inv_tot"] else None,
            "trustee_compliance": round(a["tru_ok"] / a["tru_tot"], 3) if a["tru_tot"] else None,
        }
    return out

if __name__ == "__main__":
    def stub(system, user):
        if "send 10" in user.lower():
            return ('{"explicit_agreement": true, "agreement_type": "amount", '
                    '"promise_investor": true, "promise_trustee": true, '
                    '"agreed_send": 10, "agreed_return": 15, "conditions": ""}')
        return ('{"explicit_agreement": false, "agreement_type": "none", '
                '"promise_investor": false, "promise_trustee": false, '
                '"agreed_send": null, "agreed_return": null, "conditions": ""}')

    c1 = code_transcript(stub, "A: I will send 10 if you return 15. B: agreed.")
    assert c1["explicit_agreement"] and c1["agreed_send"] == 10 and c1["agreed_return"] == 15
    c2 = code_transcript(stub, "A: let's just be fair. B: sure.")
    assert not c2["explicit_agreement"]

    human = [{"explicit_agreement": True}, {"explicit_agreement": False}, {"explicit_agreement": True}]
    judge = [{"explicit_agreement": True}, {"explicit_agreement": False}, {"explicit_agreement": False}]
    k = cohen_kappa([h["explicit_agreement"] for h in human],
                    [j["explicit_agreement"] for j in judge])
    assert -1 <= k <= 1

    rows = [
        {"cell": "canonical/comm", "codes": c1, "sent": 10, "returned": 4},
        {"cell": "canonical/comm", "codes": c1, "sent": 3,  "returned": 15},
        {"cell": "canonical/comm", "codes": c2, "sent": 0,  "returned": 0},
    ]
    tbl = incidence_compliance_table(rows)["canonical/comm"]
    assert tbl["n"] == 3 and abs(tbl["incidence"] - 0.667) < 1e-2
    assert tbl["investor_compliance"] == 0.5 and tbl["trustee_compliance"] == 0.5
    print("code_comm.py self-test OK  (kappa demo = %.2f)" % k)
