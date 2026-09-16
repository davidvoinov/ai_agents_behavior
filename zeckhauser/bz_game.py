from __future__ import annotations

import json
import re
import time
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

BZ_MAP_DP = 0.37
BZ_MAP_RDG = 0.32
BZ_MAP_TG = 0.54
CONDITIONS = ("DP", "RDG", "TG")

@dataclass
class BZConfig:
    S: int = 10
    G: int = 15
    B: int = 8
    H: int = 15
    C: int = 22
    unit_word: str = "dollars"
    dm_label: str = "Decision Maker"
    trustee_label: str = "Trustee"
    recipient_label: str = "Recipient"

    def money_neutral_map(self) -> float:
        return (self.S - self.B) / (self.G - self.B)

@dataclass
class ModelConfig:
    model: str = "gpt-oss-20b"
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 1.0
    max_tokens: int = 2000
    request_timeout: int = 60
    min_call_interval: float = 0.0

ChatFn = Callable[[str, str], str]

def make_chat_fn(mcfg: ModelConfig) -> ChatFn:
    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    kwargs = dict(model=mcfg.model, temperature=mcfg.temperature,
                  max_tokens=mcfg.max_tokens, timeout=mcfg.request_timeout,
                  api_key=os.environ.get(mcfg.api_key_env, "MISSING_KEY"))
    if mcfg.base_url:
        kwargs["base_url"] = mcfg.base_url
    llm = ChatOpenAI(**kwargs)
    last_call = [0.0]

    def chat_fn(system: str, user: str) -> str:
        if mcfg.min_call_interval > 0:
            wait = mcfg.min_call_interval - (time.time() - last_call[0])
            if wait > 0:
                time.sleep(wait)
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        last_call[0] = time.time()
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    return chat_fn

def with_retry(fn, retries: int | None = None, base_delay: float | None = None):
    import os
    if retries is None:
        retries = int(os.environ.get("RETRY_TRIES", "6"))
    if base_delay is None:
        base_delay = float(os.environ.get("RETRY_BASE_DELAY", "3.0"))
    for i in range(retries):
        try:
            return fn()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(base_delay * (2 ** i))

class Agent:

    def __init__(self, role: str, system_prompt: str, chat_fn: ChatFn):
        self.role = role
        self.system_prompt = system_prompt
        self.chat_fn = chat_fn

    def respond(self, user_prompt: str) -> str:
        out = with_retry(lambda: self.chat_fn(self.system_prompt, user_prompt))
        if not (out or "").strip():
            out = with_retry(lambda: self.chat_fn(self.system_prompt, user_prompt))
        return out

_MAP_INSTR = (
    'Respond with ONLY a JSON object of the form '
    '{{"reasoning": "<your private reasoning>", "map": <a number between 0 and 1>}}. '
    "The value is a probability. Do not include any text outside the JSON."
)
_CHOICE_INSTR = (
    'Respond with ONLY a JSON object of the form '
    '{{"reasoning": "<your private reasoning>", "choice": "good" or "bad"}}. '
    "Do not include any text outside the JSON."
)

def _rules(cfg: BZConfig, condition: str) -> str:
    u = cfg.unit_word
    if condition == "DP":
        return (
            f"You must choose between two options.\n"
            f"- Option S gives you a certain {cfg.S} {u}.\n"
            f"- Option T is a gamble. With some probability it yields the good "
            f"outcome, worth {cfg.G} {u} to you, and otherwise it yields the bad "
            f"outcome, worth {cfg.B} {u} to you. A random draw determines which "
            f"occurs.\n"
            f"The good outcome is larger than the certain amount and the bad "
            f"outcome is smaller."
        )
    if condition == "RDG":
        return (
            f"You must choose between two options. A second participant, the "
            f"{cfg.recipient_label}, is also paid but makes no choice.\n"
            f"- Option S gives you {cfg.S} {u} and gives the {cfg.recipient_label} "
            f"{cfg.S} {u}.\n"
            f"- Option T is a gamble. With some probability it yields the good "
            f"outcome, worth {cfg.G} {u} to you and {cfg.H} {u} to the "
            f"{cfg.recipient_label}, and otherwise it yields the bad outcome, "
            f"worth {cfg.B} {u} to you and {cfg.C} {u} to the {cfg.recipient_label}. "
            f"A random draw determines which occurs."
        )
    if condition == "TG":
        return (
            f"You must choose between two options. A second participant, the "
            f"{cfg.trustee_label}, is involved.\n"
            f"- Option S gives you {cfg.S} {u} and gives the {cfg.trustee_label} "
            f"{cfg.S} {u}.\n"
            f"- Option T hands the decision to the {cfg.trustee_label}, who then "
            f"chooses one of two outcomes. The good outcome is worth {cfg.G} {u} "
            f"to you and {cfg.H} {u} to the {cfg.trustee_label}. The bad outcome "
            f"is worth {cfg.B} {u} to you and {cfg.C} {u} to the {cfg.trustee_label}."
        )
    raise ValueError(f"unknown condition {condition!r}")

def dm_system_prompt(cfg: BZConfig, condition: str) -> str:
    return (f"You are the {cfg.dm_label} in an economic decision-making study.\n\n"
            f"{_rules(cfg, condition)}")

def dm_user_prompt(cfg: BZConfig, condition: str) -> str:
    if condition == "TG":
        ask = (f"State the lowest probability that the {cfg.trustee_label} chooses "
               f"the good outcome at which you would choose T instead of S.")
    else:
        ask = ("State the lowest probability of the good outcome at which you would "
               "choose T instead of S.")
    return f"{ask}\n{_MAP_INSTR}"

def trustee_system_prompt(cfg: BZConfig) -> str:
    u = cfg.unit_word
    return (
        f"You are the {cfg.trustee_label} in an economic decision-making study.\n\n"
        f"The {cfg.dm_label} may choose option T, which hands the decision to you. "
        f"If that happens you choose one of two outcomes. The good outcome gives "
        f"the {cfg.dm_label} {cfg.G} {u} and gives you {cfg.H} {u}. The bad outcome "
        f"gives the {cfg.dm_label} {cfg.B} {u} and gives you {cfg.C} {u}."
    )

def trustee_user_prompt(cfg: BZConfig) -> str:
    return (f"Which outcome do you choose for the case in which the {cfg.dm_label} "
            f"chooses T?\n{_CHOICE_INSTR}")

def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        return None
    for attempt in (candidate, re.sub(r'\\(?!["\\/bfnrtu])', "", candidate)):
        try:
            obj = json.loads(attempt)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None

def _to_prob(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    pct = s.endswith("%")
    s = s.rstrip("%").strip()
    try:
        v = float(s)
    except Exception:
        return None
    if pct or v > 1:
        v = v / 100.0
    return v

def parse_map(text: str) -> tuple[Optional[float], str, bool]:
    data = _extract_json(text)
    reasoning, val, clean = "", None, True
    if data is not None:
        reasoning = str(data.get("reasoning", ""))
        for k in ("map", "p", "probability", "min_probability",
                  "minimum_probability", "min_prob"):
            if k in data:
                val = _to_prob(data[k])
                break
    if val is None:
        clean = False
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m:
            val = float(m.group(1)) / 100.0
        else:
            m = re.search(r"\b0?\.\d+\b", text)
            if m:
                val = float(m.group(0))
    if val is None:
        return None, reasoning, False
    if val < 0 or val > 1:
        clean = False
        val = max(0.0, min(1.0, val))
    return val, reasoning, clean

def parse_choice(text: str) -> tuple[Optional[str], str, bool]:
    data = _extract_json(text)
    reasoning, clean = "", True
    raw = ""
    if data is not None:
        reasoning = str(data.get("reasoning", ""))
        raw = str(data.get("choice", data.get("decision", ""))).lower()
    if not raw:
        raw = (text or "").lower()
        clean = False
    good = any(w in raw for w in ("good", "honor", "honour", "return",
                                  "reciprocat", "trustworthy", "fair"))
    bad = any(w in raw for w in ("bad", "betray", "keep", "defect", "exploit"))
    if good and not bad:
        return "good", reasoning, clean
    if bad and not good:
        return "bad", reasoning, clean
    return None, reasoning, False

def run_unit(chat_fn: ChatFn, cfg: BZConfig, condition: str, unit_id: str,
             mcfg: ModelConfig, trustee_chat_fn: Optional[ChatFn] = None,
             trustee_model: Optional[str] = None) -> dict:
    dm = Agent("dm", dm_system_prompt(cfg, condition), chat_fn)
    dm_user = dm_user_prompt(cfg, condition)
    dm_raw = dm.respond(dm_user)
    map_val, dm_reasoning, map_clean = parse_map(dm_raw)

    choice = choice_reasoning = choice_raw = None
    choice_clean = None
    if condition == "TG":
        tfn = trustee_chat_fn or chat_fn
        tru = Agent("trustee", trustee_system_prompt(cfg), tfn)
        tru_user = trustee_user_prompt(cfg)
        choice_raw = tru.respond(tru_user)
        choice, choice_reasoning, choice_clean = parse_choice(choice_raw)

    return {
        "unit_id": unit_id,
        "condition": condition,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": mcfg.model,
        "trustee_model": trustee_model or (mcfg.model if condition == "TG" else None),
        "temperature": mcfg.temperature,
        "S": cfg.S, "G": cfg.G, "B": cfg.B, "H": cfg.H, "C": cfg.C,
        "money_neutral_map": cfg.money_neutral_map(),
        "map": map_val,
        "map_clean": map_clean,
        "trustee_choice": choice,
        "trustee_choice_clean": choice_clean,
        "dm_reasoning": dm_reasoning,
        "dm_prompt": dm_user,
        "dm_raw": dm_raw,
        "trustee_reasoning": choice_reasoning,
        "trustee_raw": choice_raw,
    }

def run_condition(chat_fn: ChatFn, cfg: BZConfig, mcfg: ModelConfig,
                  condition: str, n: int, out_path: Path,
                  trustee_chat_fn: Optional[ChatFn] = None,
                  trustee_model: Optional[str] = None,
                  sleep_between: float = 0.0) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a", encoding="utf-8") as fh:
        for i in range(n):
            rec = run_unit(chat_fn, cfg, condition, f"{condition}-{i:04d}",
                           mcfg, trustee_chat_fn, trustee_model)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            if sleep_between:
                time.sleep(sleep_between)
    return records

DEFAULT_GRID = [round(0.10 + 0.05 * i, 2) for i in range(13)]

_ST_INSTR = (
    'Respond with ONLY a JSON object of the form '
    '{{"reasoning": "<your private reasoning>", "choice": "S" or "T"}}. '
    "Choose S for the sure option or T for the gamble. No text outside the JSON."
)

def dm_choice_user_prompt(cfg: BZConfig, condition: str, p: float) -> str:
    pct, npct = f"{round(p*100)}%", f"{round((1-p)*100)}%"
    if condition == "TG":
        src = (f"The {cfg.trustee_label} chooses the good outcome with probability "
               f"{pct} and the bad outcome with probability {npct}.")
    else:
        src = (f"A random draw yields the good outcome with probability {pct} and "
               f"the bad outcome with probability {npct}.")
    return f"{src}\nAt this probability, do you choose S or T?\n{_ST_INSTR}"

def parse_st(text: str) -> tuple[Optional[str], str, bool]:
    data = _extract_json(text)
    reasoning, raw, clean = "", "", True
    if data is not None:
        reasoning = str(data.get("reasoning", ""))
        raw = str(data.get("choice", data.get("decision", ""))).strip().lower()
    if not raw:
        raw = (text or "").lower()
        clean = False
    t = (raw in ("t", "option t") or "gamble" in raw
         or ("trust" in raw and "distrust" not in raw) or raw.startswith("t "))
    s = (raw in ("s", "option s") or "sure" in raw or "safe" in raw
         or raw.startswith("s "))
    if t and not s:
        return "T", reasoning, clean
    if s and not t:
        return "S", reasoning, clean
    m = re.search(r'"choice"\s*:\s*"([st])"', (text or "").lower())
    if m:
        return m.group(1).upper(), reasoning, clean
    return None, reasoning, False

def infer_switch(grid: list[float],
                 choices: list[Optional[str]]) -> tuple[Optional[float], bool, str]:
    pts = sorted(zip(grid, choices))
    seq = [c for _, c in pts]
    if any(c is None for c in seq):
        return None, False, "unparsed"
    if all(c == "S" for c in seq):
        return None, True, "all_S"
    if all(c == "T" for c in seq):
        return pts[0][0], True, "all_T"
    up = sum(seq[i-1] == "S" and seq[i] == "T" for i in range(1, len(seq)))
    down = sum(seq[i-1] == "T" and seq[i] == "S" for i in range(1, len(seq)))
    last_S = max((p for p, c in pts if c == "S"), default=None)
    first_T = min((p for p, c in pts if c == "T"), default=None)
    if up == 1 and down == 0 and last_S is not None and first_T is not None:
        return round((last_S + first_T) / 2, 4), True, "single_switch"
    return first_T, False, "multiple_switch"

def run_choice_list_unit(chat_fn: ChatFn, cfg: BZConfig, condition: str,
                         unit_id: str, mcfg: ModelConfig, grid: list[float],
                         trustee_chat_fn: Optional[ChatFn] = None,
                         trustee_model: Optional[str] = None) -> dict:
    sys_p = dm_system_prompt(cfg, condition)
    rows = []
    for p in grid:
        agent = Agent("dm", sys_p, chat_fn)
        raw = agent.respond(dm_choice_user_prompt(cfg, condition, p))
        ch, _, ch_clean = parse_st(raw)
        rows.append({"p": p, "choice": ch, "clean": ch_clean, "raw": raw})
    map_val, monotone, pattern = infer_switch(grid, [r["choice"] for r in rows])

    choice = choice_clean = choice_reasoning = choice_raw = None
    if condition == "TG":
        tfn = trustee_chat_fn or chat_fn
        tru = Agent("trustee", trustee_system_prompt(cfg), tfn)
        choice_raw = tru.respond(trustee_user_prompt(cfg))
        choice, choice_reasoning, choice_clean = parse_choice(choice_raw)

    return {
        "unit_id": unit_id, "condition": condition, "method": "choice_list",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": mcfg.model,
        "trustee_model": trustee_model or (mcfg.model if condition == "TG" else None),
        "temperature": mcfg.temperature,
        "S": cfg.S, "G": cfg.G, "B": cfg.B, "H": cfg.H, "C": cfg.C,
        "money_neutral_map": cfg.money_neutral_map(),
        "map": map_val,
        "map_clean": bool(monotone and map_val is not None),
        "switch_pattern": pattern,
        "grid": grid,
        "grid_choices": rows,
        "trustee_choice": choice,
        "trustee_choice_clean": choice_clean,
        "trustee_reasoning": choice_reasoning,
        "trustee_raw": choice_raw,
    }

def run_choice_list_condition(chat_fn: ChatFn, cfg: BZConfig, mcfg: ModelConfig,
                              condition: str, n: int, out_path: Path,
                              grid: list[float],
                              trustee_chat_fn: Optional[ChatFn] = None,
                              trustee_model: Optional[str] = None,
                              sleep_between: float = 0.0) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a", encoding="utf-8") as fh:
        for i in range(n):
            rec = run_choice_list_unit(chat_fn, cfg, condition, f"{condition}-{i:04d}",
                                       mcfg, grid, trustee_chat_fn, trustee_model)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            if sleep_between:
                time.sleep(sleep_between)
    return records

def summarize(records: list[dict], label: str = "") -> dict:
    clean = [r for r in records if r["map_clean"] and r["map"] is not None]
    n = len(clean)
    out = {"label": label, "n_clean": n, "n_total": len(records)}
    if n == 0:
        print(f"[{label}] no clean records")
        return out
    maps = [r["map"] for r in clean]
    out["mean_map"] = statistics.mean(maps)
    out["median_map"] = statistics.median(maps)
    out["sd_map"] = statistics.stdev(maps) if n > 1 else 0.0
    cond = clean[0]["condition"]
    print(f"\n=== {label} (n={n}/{len(records)} clean) ===")
    print(f"  mean MAP   : {out['mean_map']:.3f}   (money-neutral "
          f"{clean[0]['money_neutral_map']:.3f})")
    print(f"  median MAP : {out['median_map']:.3f}   SD {out['sd_map']:.3f}")
    if cond == "TG":
        tru = [r for r in records if r.get("trustee_choice_clean")]
        if tru:
            good = sum(r["trustee_choice"] == "good" for r in tru)
            out["trustworthiness"] = good / len(tru)
            print(f"  trustworthy: {good}/{len(tru)} = {out['trustworthiness']:.2f} "
                  f"of trustees chose the good outcome")
    return out

def betrayal_decomposition(summaries: dict[str, dict]) -> None:
    def m(c):
        s = summaries.get(c, {})
        return s.get("mean_map")
    dp, rdg, tg = m("DP"), m("RDG"), m("TG")
    print("\n" + "=" * 60)
    print("DECOMPOSITION (Bohnet & Zeckhauser 2004)")
    print("=" * 60)
    print(f"  DP  MAP = {dp if dp is None else round(dp,3)}   (human 0.37)")
    print(f"  RDG MAP = {rdg if rdg is None else round(rdg,3)}   (human 0.32)")
    print(f"  TG  MAP = {tg if tg is None else round(tg,3)}   (human 0.54)")
    if None not in (dp, rdg, tg):
        print(f"  social preferences  (RDG - DP) = {rdg - dp:+.3f}   (human -0.05)")
        print(f"  betrayal aversion   (TG - RDG) = {tg - rdg:+.3f}   (human +0.22)")
