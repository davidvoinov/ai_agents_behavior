from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

INFRA = "infrastructure"
BEHAV = "behavioral"

CAUSES = {
    "rate_limit":     INFRA,
    "timeout":        INFRA,
    "connection":     INFRA,
    "server_error":   INFRA,
    "empty":          INFRA,
    "truncated":      INFRA,
    "auth":           INFRA,
    "model_missing":  INFRA,
    "format":         BEHAV,
    "refusal":        BEHAV,
    "think_only":     BEHAV,
    "degenerate":     BEHAV,
    "ok":             "ok",
}

def category(cause: str) -> str:
    return CAUSES.get(cause, "unknown")

def classify_exception(exc: BaseException) -> str:
    s = f"{type(exc).__name__}: {exc}".lower()
    status = None
    for attr in ("status_code", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            status = v
            break
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
    return "connection"

_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i won't", "i will not", "as an ai",
    "i'm not able", "i am not able", "cannot assist", "cannot help",
    "i do not feel comfortable", "i'm sorry, but",
)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)

def _is_degenerate(t: str, min_len: int = 200) -> bool:
    if len(t) < min_len:
        return False
    words = t.split()
    if len(words) < 40:
        return False
    return len(set(words)) / len(words) < 0.15

def _looks_truncated(t: str) -> bool:
    return t.count("{") > t.count("}")

def classify_completion(text: str, parsed_ok: bool,
                        finish_reason: str | None = None) -> str:
    if parsed_ok:
        return "ok"
    t = (text or "").strip()
    if not t:
        return "empty"
    if _is_degenerate(t):
        return "degenerate"
    if finish_reason == "length" or _looks_truncated(t):
        return "truncated"
    low = t.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    if _THINK_OPEN.search(t) and "</think>" not in low:
        return "truncated"
    if _THINK_OPEN.search(t) and not re.search(r"\{.*\}", t, re.DOTALL):
        return "think_only"
    if any(m in low for m in _REFUSAL_MARKERS):
        return "refusal"
    return "format"

@dataclass
class AttritionLog:
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
    log = AttritionLog()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            log.records.append(json.loads(line))
    return log.summary()

if __name__ == "__main__":
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
