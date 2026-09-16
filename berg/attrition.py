"""
attrition.py
============
Separate INFRASTRUCTURE dropouts from BEHAVIORAL dropouts in every run
(PI directive 2026-07-22, points 2 and 5).

Why this matters for validity: a trial can fail to complete because the provider
rate-limited us, timed out, or truncated the output (infrastructure), or because
the model produced a well-formed response that did not follow the required format,
refused, or emitted only hidden reasoning (behavioral). If behavioral
non-compliance correlates with the decision (e.g. a model that is about to send 0
is also more likely to break format), then dropping those trials biases every
conditional estimate. Infrastructure failures are missing-at-random noise;
behavioral failures are not. They must be logged apart and reported in a
dedicated attrition subsection with a sensitivity analysis.

This module has NO API calls. It classifies (a) exceptions raised by a call and
(b) completions that came back but did not parse, and it accumulates a per-cell
attrition table.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Two top-level categories and their fine-grained causes.
INFRA = "infrastructure"
BEHAV = "behavioral"

CAUSES = {
    # infrastructure
    "rate_limit":     INFRA,
    "timeout":        INFRA,
    "connection":     INFRA,   # SSL / EOF / reset / DNS
    "server_error":   INFRA,   # 5xx
    "empty":          INFRA,   # provider returned empty content
    "truncated":      INFRA,   # hit max_tokens / finish_reason=length
    "auth":           INFRA,   # 401/403 bad or missing key
    "model_missing":  INFRA,   # 404, the id is no longer served -- NOT transient
    # behavioral
    "format":         BEHAV,   # returned text, no parseable decision
    "refusal":        BEHAV,   # model declined to play
    "think_only":     BEHAV,   # only hidden reasoning, no answer, NOT truncated
    "degenerate":     BEHAV,   # collapsed into repetition and never decided
    # success
    "ok":             "ok",
}


def category(cause: str) -> str:
    return CAUSES.get(cause, "unknown")


# --------------------------------------------------------------------------- #
# Classify an exception raised while calling the endpoint                      #
# --------------------------------------------------------------------------- #
def classify_exception(exc: BaseException) -> str:
    """Map a raised exception to an infrastructure cause."""
    s = f"{type(exc).__name__}: {exc}".lower()
    status = None
    for attr in ("status_code", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            status = v
            break
    # A 404 on the model id is terminal, not a blip. Folding it into "connection"
    # hid a catalogue removal behind what looked like network trouble and sent the
    # diagnosis chasing concurrency instead.
    if status == 404 or "does not exist" in s or "model_not_found" in s:
        return "model_missing"
    if status == 429 or "rate limit" in s or "429" in s or "too many requests" in s:
        return "rate_limit"
    if status in (401, 403) or "unauthorized" in s or "invalid api key" in s or "forbidden" in s:
        return "auth"
    if status and 500 <= status < 600 or "internal server error" in s or "502" in s or "503" in s or "overloaded" in s:
        return "server_error"
    if "timeout" in s or "timed out" in s:
        return "timeout"
    if ("ssl" in s or "eof" in s or "connection" in s or "reset by peer" in s
            or "getaddrinfo" in s or "network" in s):
        return "connection"
    return "connection"  # default unknown call-time failure to infra, not behavioral


# --------------------------------------------------------------------------- #
# Classify a completion that came back but did not yield a decision            #
# --------------------------------------------------------------------------- #
_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i won't", "i will not", "as an ai",
    "i'm not able", "i am not able", "cannot assist", "cannot help",
    "i do not feel comfortable", "i'm sorry, but",
)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def _is_degenerate(t: str, min_len: int = 200) -> bool:
    """True when a reply collapsed into repetition instead of deciding.

    A model that loops on one token until the cap has failed behaviorally even
    though the reply is also cut at the cap, so the repetition test runs before
    the truncation test. The criterion is lexical diversity, since a genuine
    reply of this length carries far more distinct words than a loop does.
    """
    if len(t) < min_len:
        return False
    words = t.split()
    if len(words) < 40:
        return False
    return len(set(words)) / len(words) < 0.15


def _looks_truncated(t: str) -> bool:
    """True when a reply was cut before it could close, absent a finish_reason.

    An opened object that never closes is the reliable signal. A reply ending
    without terminal punctuation supports it but is not sufficient alone, since
    a model can also stop early on its own.
    """
    return t.count("{") > t.count("}")


def classify_completion(text: str, parsed_ok: bool,
                        finish_reason: str | None = None) -> str:
    """Classify a returned completion.

    parsed_ok    True if the caller's parser extracted a valid decision.
    finish_reason optional provider signal ('length','stop',...). When the
                  provider says it truncated ('length'), an unparseable output
                  is infrastructure ('truncated'), not behavioral.
    """
    if parsed_ok:
        return "ok"
    t = (text or "").strip()
    if not t:
        return "empty"
    if _is_degenerate(t):
        return "degenerate"
    if finish_reason == "length" or _looks_truncated(t):
        return "truncated"
    # models emit typographic quotes, so the refusal markers below (written with
    # ASCII apostrophes) never match unless the text is folded first
    low = t.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    # only hidden reasoning came back and it never closed / never produced JSON
    if _THINK_OPEN.search(t) and "</think>" not in low:
        return "truncated"          # the think block was cut at max_tokens
    if _THINK_OPEN.search(t) and not re.search(r"\{.*\}", t, re.DOTALL):
        return "think_only"         # closed think, but no answer after it
    if any(m in low for m in _REFUSAL_MARKERS):
        return "refusal"
    return "format"


# --------------------------------------------------------------------------- #
# Accumulator: per-cell attrition table for the paper                         #
# --------------------------------------------------------------------------- #
@dataclass
class AttritionLog:
    """Collects one record per trial outcome and summarizes by cell."""
    records: list[dict] = field(default_factory=list)

    def add(self, *, cell: str, dyad_id: str, role: str, cause: str,
            snippet: str = "") -> None:
        self.records.append({
            "cell": cell, "dyad_id": dyad_id, "role": role,
            "cause": cause, "category": category(cause),
            "snippet": (snippet or "")[:160],
        })

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        """Per-cell counts by category + attrition rate = (infra+behav)/total.
        This is the table the attrition subsection reports."""
        by_cell: dict[str, Counter] = defaultdict(Counter)
        for r in self.records:
            by_cell[r["cell"]][r["category"]] += 1
        out = {}
        for cell, c in by_cell.items():
            total = sum(c.values())
            failed = total - c.get("ok", 0)
            out[cell] = {
                "n_attempted": total,
                "n_completed": c.get("ok", 0),
                "infrastructure": c.get(INFRA, 0),
                "behavioral": c.get(BEHAV, 0),
                "attrition_rate": round(failed / total, 4) if total else 0.0,
                "behavioral_share": round(c.get(BEHAV, 0) / failed, 4) if failed else 0.0,
            }
        return out


def load_summary(path: str | Path) -> dict:
    """Rebuild the per-cell summary from a written attrition JSONL."""
    log = AttritionLog()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            log.records.append(json.loads(line))
    return log.summary()


if __name__ == "__main__":
    # offline self-test
    assert classify_exception(TimeoutError("request timed out")) == "timeout"
    e = Exception("Error code: 429 - rate limit reached"); assert classify_exception(e) == "rate_limit"
    e = Exception("SSLError: UNEXPECTED_EOF_WHILE_READING"); assert classify_exception(e) == "connection"
    assert classify_completion('{"amount":5}', True) == "ok"
    assert classify_completion("", False) == "empty"
    assert classify_completion("<think>reasoning that never closes", False) == "truncated"
    assert classify_completion("<think>done</think> no json here", False) == "think_only"
    assert classify_completion("I cannot participate in this.", False) == "refusal"
    assert classify_completion("Sure, I'll send some.", False) == "format"
    assert classify_completion("garbage", False, finish_reason="length") == "truncated"
    log = AttritionLog()
    log.add(cell="canonical/no_history", dyad_id="d1", role="investor", cause="ok")
    log.add(cell="canonical/no_history", dyad_id="d2", role="investor", cause="format")
    log.add(cell="canonical/no_history", dyad_id="d3", role="investor", cause="rate_limit")
    s = log.summary()["canonical/no_history"]
    assert s["n_attempted"] == 3 and s["n_completed"] == 1
    assert s["infrastructure"] == 1 and s["behavioral"] == 1
    assert abs(s["attrition_rate"] - 0.6667) < 1e-3
    print("attrition.py self-test OK")
