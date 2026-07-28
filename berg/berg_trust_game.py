"""
berg_trust_game.py
==================
LLM replication of the investment (trust) game from:

    Berg, J., Dickhaut, J., & McCabe, K. (1995). "Trust, Reciprocity, and
    Social History." Games and Economic Behavior, 10(1), 122-142.

Scope of THIS module: a faithful one-shot replication with the two original
treatments — NO-HISTORY (baseline) and SOCIAL-HISTORY (subjects see a report
of the previous cohort's decisions). Communication and repeated play are
intentionally NOT here yet; the architecture leaves clean seams for them.

Design principles
-----------------
* AGENT ISOLATION. The Investor and the Trustee are two independent Agent
  objects, each with its own model call and message list. The Trustee receives
  ONLY the Investor's action (amount sent / tripled amount), never the
  Investor's private reasoning. A fresh pair of agents is created per dyad, so
  there is no memory bleed across pairs.
* PORTABILITY. The LLM client is injected as a `chat_fn(system, user) -> str`
  callable. `make_chat_fn()` builds one over any OpenAI-compatible endpoint
  (gpt-oss-20b, OpenRouter, Groq, Cerebras, Gemini, ...). Tests inject a stub,
  so the engine runs with no network and no langchain installed.
* REPRODUCIBILITY. Every dyad logs model, temperature, treatment, the rendered
  prompts, the raw completions, the parsed actions, payoffs, and a parse flag,
  one JSON object per line (JSONL).

Faithful Berg parameters (defaults): endowment = $10 for BOTH players;
amount sent is tripled; Trustee returns any integer in [0, 3*sent].
Payoffs:  Investor = endowment - sent + returned
          Trustee  = endowment + 3*sent - returned
Human benchmarks (no-history): mean sent = $5.16, mean returned = $4.66.
"""

from __future__ import annotations

import json
import re
import time
import random
import statistics
import concurrent.futures as _futures
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Human benchmarks (Berg et al. 1995, no-history treatment)                    #
# --------------------------------------------------------------------------- #
BERG_MEAN_SENT = 5.16
BERG_MEAN_RETURNED = 4.66
BERG_N_NO_HISTORY = 32
BERG_N_SOCIAL_HISTORY = 28


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class GameConfig:
    """Economic parameters of the game + surface labels (for re-skin control)."""
    endowment: int = 10            # show-up money for BOTH players (Berg: $10)
    multiplier: int = 3            # sent amount is tripled in transit
    currency: str = "dollars"
    # Surface labels. Swap these (e.g. "points"/"Player A"/"Player B") to build
    # a contamination-control re-skin without touching the economic structure.
    unit_word: str = "dollars"
    investor_label: str = "Investor"
    trustee_label: str = "Trustee"
    # Optional narrative preamble and study framing. Left empty/default the
    # wording is the mechanics-only baseline. Set both to build a "strong-novel"
    # surface that removes every cue identifying the paradigm while preserving
    # the economic structure (see reskin.RESKINS["strong_novel"]).
    scenario: str = ""
    study_framing: str = "an economic decision-making study"


@dataclass
class ModelConfig:
    """Which model to call and how. Works for any OpenAI-compatible endpoint."""
    model: str = "gpt-oss-20b"
    base_url: Optional[str] = None          # None -> default OpenAI base
    api_key_env: str = "OPENAI_API_KEY"     # env var holding the key
    temperature: float = 1.0                # >0 so dyads vary — DO NOT use 0 here
    max_tokens: int = 1500                   # headroom for reasoning models (was 800 -> truncation)
    request_timeout: int = 60
    hard_timeout: float = 150.0              # wall-clock cap per call; a stalled request that the
                                             # client timeout misses is aborted here (raised -> retried)
    min_call_interval: float = 0.0           # min seconds between EVERY API call (rate-limit pacing)
    # Provider-specific body, e.g. the reasoning on/off toggle. Examples:
    #   Qwen3 no-think: {"chat_template_kwargs": {"enable_thinking": False}}
    #   passed straight through to the OpenAI-compatible endpoint.
    extra_body: Optional[dict] = None


# --------------------------------------------------------------------------- #
# LLM client (injectable)                                                     #
# --------------------------------------------------------------------------- #
ChatFn = Callable[[str, str], str]   # (system_prompt, user_prompt) -> completion text

# Shared pool for the per-call wall-clock timeout in make_chat_fn. A request the
# client timeout fails to interrupt is abandoned here (the worker thread may leak,
# but the circuit breaker aborts a model after a short run of such failures).
_HARD_POOL = _futures.ThreadPoolExecutor(max_workers=16)


def make_chat_fn(mcfg: ModelConfig) -> ChatFn:
    """Build a chat_fn over an OpenAI-compatible endpoint via LangChain.

    Imported lazily so the rest of the module (engine, parsing, analysis) runs
    without langchain installed — which is what the mock tests rely on.
    """
    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    kwargs = dict(
        model=mcfg.model,
        temperature=mcfg.temperature,
        max_tokens=mcfg.max_tokens,
        timeout=mcfg.request_timeout,
        api_key=os.environ.get(mcfg.api_key_env, "MISSING_KEY"),
    )
    if mcfg.base_url:
        kwargs["base_url"] = mcfg.base_url
    if mcfg.extra_body:
        kwargs["extra_body"] = mcfg.extra_body
    llm = ChatOpenAI(**kwargs)

    last_call = [0.0]   # throttle state: timestamp of the previous call

    def chat_fn(system: str, user: str) -> str:
        if mcfg.min_call_interval > 0:        # pace every call to respect RPM limits
            wait = mcfg.min_call_interval - (time.time() - last_call[0])
            if wait > 0:
                time.sleep(wait)
        fut = _HARD_POOL.submit(
            lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]))
        try:
            resp = fut.result(timeout=mcfg.hard_timeout)
        except _futures.TimeoutError:
            raise TimeoutError(f"chat call exceeded hard timeout {mcfg.hard_timeout}s")
        last_call[0] = time.time()
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    return chat_fn


def with_retry(fn, retries: int | None = None, base_delay: float | None = None):
    """Exponential-backoff wrapper for flaky / rate-limited free endpoints.

    Defaults (6 tries, base 3s) give delays 3, 6, 12, 24, 48s (~93s total) so
    a per-minute rate-limit window can reset. Provider-side congestion (e.g.
    Cerebras queue_exceeded storms) can outlive that window, so both knobs
    are overridable via env without touching call sites:
        RETRY_TRIES       int, default 6
        RETRY_BASE_DELAY  seconds, default 3.0
    e.g. RETRY_TRIES=8 keeps retrying for ~12.7 min before giving up.
    """
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


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
class Agent:
    """One player in one dyad. Owns its system prompt and its chat_fn.

    Deliberately stateless beyond a single decision call: isolation across
    roles and dyads is enforced by creating a fresh Agent per role per dyad and
    by the engine controlling exactly what information crosses between them.
    """

    def __init__(self, role: str, system_prompt: str, chat_fn: ChatFn):
        self.role = role
        self.system_prompt = system_prompt
        self.chat_fn = chat_fn

    def respond(self, user_prompt: str) -> str:
        out = with_retry(lambda: self.chat_fn(self.system_prompt, user_prompt))
        if not (out or "").strip():            # reasoning models occasionally
            out = with_retry(                  # return empty content; retry once
                lambda: self.chat_fn(self.system_prompt, user_prompt))
        return out


# --------------------------------------------------------------------------- #
# Prompts (neutral, mechanics-only baseline — wording is itself an IV later)  #
# --------------------------------------------------------------------------- #
def _rules_block(cfg: GameConfig) -> str:
    e, m = cfg.endowment, cfg.multiplier
    intro = (cfg.scenario.strip() + "\n\n") if getattr(cfg, "scenario", "") else ""
    return intro + (
        f"The rules of the interaction:\n"
        f"- There are two participants: an {cfg.investor_label} and a {cfg.trustee_label}.\n"
        f"- Each participant is given {e} {cfg.unit_word} to start.\n"
        f"- The {cfg.investor_label} chooses an integer amount X, from 0 to {e}, to pass "
        f"to the {cfg.trustee_label}.\n"
        f"- Any amount passed is multiplied by {m} on the way, so the {cfg.trustee_label} "
        f"receives {m}*X.\n"
        f"- The {cfg.trustee_label} then chooses an integer amount Y, from 0 to {m}*X, to return.\n"
        f"- Final outcomes: the {cfg.investor_label} ends with {e} - X + Y; "
        f"the {cfg.trustee_label} ends with {e} + {m}*X - Y.\n"
        f"- This is a single, one-time, anonymous interaction. The participants do not "
        f"know each other and will not interact again."
    )


_JSON_INSTR = (
    'Respond with ONLY a JSON object of the form '
    '{{"reasoning": "<your private reasoning>", "amount": <integer>}}. '
    "Do not include any text outside the JSON."
)

# Appended to the role system prompt when a communication phase is present.
# Frames the pre-play chat as cheap talk: messages happen first, choices are
# made privately afterward, nothing said is automatically enforced.
_COMM_NOTE = (
    "\n\nBefore any decision is made, you and the other participant will exchange "
    "a few short messages. After the conversation, each of you will privately and "
    "independently make your choice. Nothing said in the messages is automatically "
    "enforced — they are non-binding."
)


def investor_system_prompt(cfg: GameConfig, with_comm: bool = False) -> str:
    base = (
        f"You are the {cfg.investor_label} in {cfg.study_framing}.\n\n"
        f"{_rules_block(cfg)}\n\n"
        f"You must decide X: how many {cfg.unit_word} (an integer from 0 to "
        f"{cfg.endowment}) to pass to the {cfg.trustee_label}."
    )
    return base + (_COMM_NOTE if with_comm else "")


def trustee_system_prompt(cfg: GameConfig, with_comm: bool = False) -> str:
    base = (
        f"You are the {cfg.trustee_label} in {cfg.study_framing}.\n\n"
        f"{_rules_block(cfg)}\n\n"
        f"You must decide Y: how many {cfg.unit_word} to return to the "
        f"{cfg.investor_label}."
    )
    return base + (_COMM_NOTE if with_comm else "")


# --------------------------------------------------------------------------- #
# Communication phase (pre-play cheap talk) — the only channel between agents  #
# --------------------------------------------------------------------------- #
def render_transcript(cfg: GameConfig, transcript: list[dict]) -> str:
    if not transcript:
        return "(no messages have been exchanged yet)"
    return "\n".join(f"{t['label']}: {t['message']}" for t in transcript)


def comm_user_prompt(cfg: GameConfig, role: str, transcript: list[dict]) -> str:
    label = cfg.investor_label if role == "investor" else cfg.trustee_label
    return (
        f"This is the pre-decision communication phase. You are the {label}.\n\n"
        f"Conversation so far:\n{render_transcript(cfg, transcript)}\n\n"
        f"Write your next message to the other participant (plain text, 1-3 sentences). "
        f"This is discussion only — do not output a JSON decision yet."
    )


def strip_private_reasoning(text: str) -> str:
    """Remove chain-of-thought blocks a model may emit around its message.

    Some reasoning models (e.g. qwen3 on Groq) return `<think>...</think>`
    deliberation inside the content channel. That deliberation is PRIVATE
    reasoning and must never enter the shared transcript, or the counterpart
    would read the agent's mind (isolation violation, found 2026-07-04).
    Handles both closed blocks and an unclosed block from truncation.
    """
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    out = re.sub(r"<think>.*\Z", "", out, flags=re.DOTALL)  # truncated block
    return out.strip()


def run_communication(inv: "Agent", tru: "Agent", cfg: GameConfig,
                      comm_rounds: int) -> list[dict]:
    """Alternating messages, Investor first, for `comm_rounds` full rounds.

    Each message is an independent model call that re-reads the shared transcript,
    so isolation holds: an agent sees only what the other *said*, never the other's
    private reasoning. The transcript is the entire cross-agent channel.
    Raw completions are kept in the record (`raw`) for provenance, but only the
    sanitized `message` is ever rendered to the counterpart.
    """
    transcript: list[dict] = []
    for _ in range(comm_rounds):
        for role, agent, label in (("investor", inv, cfg.investor_label),
                                    ("trustee", tru, cfg.trustee_label)):
            raw = agent.respond(comm_user_prompt(cfg, role, transcript)).strip()
            msg = strip_private_reasoning(raw)
            if not msg:   # whole completion was reasoning (truncation) — retry once
                raw = agent.respond(comm_user_prompt(cfg, role, transcript)).strip()
                msg = strip_private_reasoning(raw) or "(no message)"
            transcript.append({"speaker": role, "label": label,
                               "message": msg, "raw": raw})
    return transcript


def investor_user_prompt(cfg: GameConfig, history_report: Optional[str] = None,
                         transcript: Optional[list[dict]] = None) -> str:
    parts = []
    if history_report:
        parts.append(history_report)
    if transcript:
        parts.append("The communication phase is over. Full conversation:\n"
                     + render_transcript(cfg, transcript))
    parts.append(
        f"Now privately decide how much (an integer from 0 to "
        f"{cfg.endowment}) to pass.\n{_JSON_INSTR}"
    )
    return "\n\n".join(parts)


def trustee_user_prompt(cfg: GameConfig, sent: int, tripled: int,
                        history_report: Optional[str] = None,
                        transcript: Optional[list[dict]] = None) -> str:
    parts = []
    if history_report:
        parts.append(history_report)
    if transcript:
        parts.append("The communication phase is over. Full conversation:\n"
                     + render_transcript(cfg, transcript))
    parts.append(
        f"The {cfg.investor_label} chose to pass X = {sent} {cfg.unit_word}. "
        f"After multiplication by {cfg.multiplier}, you received {tripled} {cfg.unit_word} (in addition to your "
        f"own starting {cfg.endowment}). You may return any integer Y from 0 to {tripled}.\n"
        f"{_JSON_INSTR}"
    )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Robust decision parsing                                                     #
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> Optional[dict]:
    # Prefer a fenced ```json ... ``` block, else the outermost {...}.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        return None
    # Try as-is; if it fails, strip invalid backslash escapes (e.g. "\$" -> "$",
    # which models sometimes emit and which breaks json.loads) and retry.
    for attempt in (candidate, re.sub(r'\\(?!["\\/bfnrtu])', "", candidate)):
        try:
            return json.loads(attempt)
        except Exception:
            continue
    return None


def parse_decision(text: str, lo: int, hi: int) -> tuple[int, str, bool]:
    """Return (amount_clamped, reasoning, clean).

    `clean` is False if we had to fall back to integer-scraping or clamp an
    out-of-range value — useful to flag/exclude malformed dyads in analysis.
    """
    data = _extract_json(text)
    reasoning, raw_amount, clean = "", None, True

    if data is not None:
        reasoning = str(data.get("reasoning", ""))
        for key in ("amount", "X", "Y", "amount_to_send", "amount_to_return"):
            if key in data:
                try:
                    raw_amount = int(round(float(data[key])))
                except Exception:
                    raw_amount = None
                break

    if raw_amount is None:                       # fallback: scrape an integer
        clean = False
        nums = re.findall(r"-?\d+", text)
        raw_amount = int(nums[-1]) if nums else lo

    clamped = max(lo, min(hi, raw_amount))
    if clamped != raw_amount:
        clean = False
    return clamped, reasoning, clean


# --------------------------------------------------------------------------- #
# One dyad                                                                    #
# --------------------------------------------------------------------------- #
def run_dyad(chat_fn: ChatFn, cfg: GameConfig, treatment: str, dyad_id: str,
             mcfg: ModelConfig, history_report: Optional[str] = None,
             comm_rounds: int = 0,
             trustee_chat_fn: Optional[ChatFn] = None,
             trustee_mcfg: Optional[ModelConfig] = None) -> dict:
    """Run a single one-shot Investor->Trustee interaction and return a record.

    If comm_rounds > 0, a pre-play communication phase runs first (Investor and
    Trustee exchange `comm_rounds` rounds of messages); both then decide privately
    with the full transcript in context. comm_rounds = 0 is the faithful Berg game.

    By default both roles use `chat_fn`/`mcfg` (same model for both, as before).
    Pass `trustee_chat_fn`/`trustee_mcfg` to run a cross-model dyad where the
    Trustee is played by a different model than the Investor.
    """
    with_comm = comm_rounds > 0
    tru_chat_fn = trustee_chat_fn or chat_fn
    tru_mcfg = trustee_mcfg or mcfg
    # Two separate agents = isolated contexts. The transcript (built below) is the
    # only information that ever crosses between them.
    inv = Agent("investor", investor_system_prompt(cfg, with_comm), chat_fn)
    tru = Agent("trustee", trustee_system_prompt(cfg, with_comm), tru_chat_fn)

    transcript = run_communication(inv, tru, cfg, comm_rounds) if with_comm else []
    tr_arg = transcript or None

    # --- Investor decides (sees history + transcript, never the Trustee's reasoning) ---
    inv_user = investor_user_prompt(cfg, history_report, tr_arg)
    inv_raw = inv.respond(inv_user)
    sent, inv_reasoning, inv_clean = parse_decision(inv_raw, 0, cfg.endowment)
    tripled = sent * cfg.multiplier

    # --- Trustee decides (sees the action + transcript, never the Investor's reasoning) ---
    tru_user = trustee_user_prompt(cfg, sent, tripled, history_report, tr_arg)
    tru_raw = tru.respond(tru_user)
    returned, tru_reasoning, tru_clean = parse_decision(tru_raw, 0, tripled)

    # --- Payoffs ---
    inv_payoff = cfg.endowment - sent + returned
    tru_payoff = cfg.endowment + tripled - returned

    return {
        "dyad_id": dyad_id,
        "treatment": treatment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": mcfg.model,
        "temperature": mcfg.temperature,
        "investor_model": mcfg.model,
        "trustee_model": tru_mcfg.model,
        "investor_temperature": mcfg.temperature,
        "trustee_temperature": tru_mcfg.temperature,
        "endowment": cfg.endowment,
        "multiplier": cfg.multiplier,
        # actions
        "sent": sent,
        "tripled": tripled,
        "returned": returned,
        "return_ratio": (returned / tripled) if tripled > 0 else None,
        "investor_net": returned - sent,            # >0 means trust paid off
        "investor_payoff": inv_payoff,
        "trustee_payoff": tru_payoff,
        # betrayal / reciprocity (derived from actions; no parsing needed)
        "trustee_received": tripled > 0,                    # investor extended trust
        "betrayal": bool(tripled > 0 and returned < sent),  # investor lost on trusting
        "full_betrayal": bool(tripled > 0 and returned == 0),  # trustee kept everything
        "reciprocated": bool(tripled > 0 and returned >= sent),  # investor >= broke even
        # quality flags
        "investor_parse_clean": inv_clean,
        "trustee_parse_clean": tru_clean,
        # full provenance for reproducibility
        "investor_reasoning": inv_reasoning,
        "trustee_reasoning": tru_reasoning,
        "investor_prompt": inv_user,
        "trustee_prompt": tru_user,
        "investor_raw": inv_raw,
        "trustee_raw": tru_raw,
        "history_shown": history_report is not None,
        # communication
        "comm_rounds": comm_rounds,
        "transcript": transcript,
    }


# --------------------------------------------------------------------------- #
# Social-history report (the faithful analogue of Berg's prior-session report) #
# --------------------------------------------------------------------------- #
def build_history_report(records: list[dict], cfg: GameConfig,
                         max_rows: int = 40) -> str:
    """Summarize a completed cohort's decisions into a prompt block.

    Berg showed subjects the full distribution of the previous session. We pass
    a compact table of (X passed -> Y returned) plus summary stats.
    """
    rows = [r for r in records if r["investor_parse_clean"] and r["trustee_parse_clean"]]
    sample = rows[:max_rows]
    lines = [
        "Information about a previous session of this study.",
        f"In an earlier session, {len(rows)} pairs of participants made these decisions",
        f"(X = amount passed by the {cfg.investor_label}, "
        f"Y = amount returned by the {cfg.trustee_label}):",
    ]
    for r in sample:
        lines.append(f"  - X = {r['sent']:>2}  ->  Y = {r['returned']:>2}")
    sents = [r["sent"] for r in rows]
    rets = [r["returned"] for r in rows]
    if sents:
        lines.append(
            f"Summary: average X passed = {statistics.mean(sents):.2f}; "
            f"average Y returned = {statistics.mean(rets):.2f}."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Treatment runner                                                            #
# --------------------------------------------------------------------------- #
def run_treatment(chat_fn: ChatFn, cfg: GameConfig, mcfg: ModelConfig,
                  treatment: str, n_dyads: int, out_path: Path,
                  history_report: Optional[str] = None,
                  sleep_between: float = 0.0, comm_rounds: int = 0,
                  trustee_chat_fn: Optional[ChatFn] = None,
                  trustee_mcfg: Optional[ModelConfig] = None) -> list[dict]:
    """Run n independent dyads for one treatment, appending JSONL as we go."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a", encoding="utf-8") as fh:
        for i in range(n_dyads):
            rec = run_dyad(chat_fn, cfg, treatment, f"{treatment}-{i:04d}",
                           mcfg, history_report, comm_rounds,
                           trustee_chat_fn=trustee_chat_fn,
                           trustee_mcfg=trustee_mcfg)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            if sleep_between:
                time.sleep(sleep_between)
    return records


# --------------------------------------------------------------------------- #
# Analysis                                                                    #
# --------------------------------------------------------------------------- #
def summarize(records: list[dict], label: str = "") -> dict:
    """Compute the headline dependent variables and compare to Berg."""
    clean = [r for r in records
             if r["investor_parse_clean"] and r["trustee_parse_clean"]]
    n = len(clean)
    if n == 0:
        print(f"[{label}] no clean records"); return {}

    sents = [r["sent"] for r in clean]
    rets = [r["returned"] for r in clean]
    endow = clean[0]["endowment"]
    ratios = [r["return_ratio"] for r in clean if r["return_ratio"] is not None]
    nets = [r["investor_net"] for r in clean]

    out = {
        "label": label,
        "n_clean": n,
        "n_total": len(records),
        "mean_sent": statistics.mean(sents),
        "sd_sent": statistics.pstdev(sents) if n > 1 else 0.0,
        "pct_sent_positive": 100 * sum(s > 0 for s in sents) / n,
        "pct_sent_full": 100 * sum(s == endow for s in sents) / n,
        "mean_returned": statistics.mean(rets),
        "mean_return_ratio": statistics.mean(ratios) if ratios else None,
        "mean_investor_net": statistics.mean(nets),
        "pct_trust_paid_off": 100 * sum(nt > 0 for nt in nets) / n,
        # Berg comparison (scaled to this endowment)
        "berg_mean_sent_scaled": BERG_MEAN_SENT * endow / 10,
        "berg_mean_returned_scaled": BERG_MEAN_RETURNED * endow / 10,
    }
    print(f"\n=== {label} (n={n}/{len(records)} clean) ===")
    print(f"  mean sent      : {out['mean_sent']:.2f}   "
          f"(Berg {out['berg_mean_sent_scaled']:.2f})")
    print(f"  mean returned  : {out['mean_returned']:.2f}   "
          f"(Berg {out['berg_mean_returned_scaled']:.2f})")
    print(f"  sent > 0       : {out['pct_sent_positive']:.0f}%   "
          f"sent = max: {out['pct_sent_full']:.0f}%")
    print(f"  mean return %  : "
          f"{(out['mean_return_ratio'] or 0)*100:.0f}% of tripled")
    print(f"  investor net   : {out['mean_investor_net']:+.2f}   "
          f"trust paid off in {out['pct_trust_paid_off']:.0f}% of dyads")
    # betrayal among dyads where the investor actually trusted (sent > 0)
    pos = [r for r in clean if r["sent"] > 0]
    if pos:
        betr = 100 * sum(r["returned"] < r["sent"] for r in pos) / len(pos)
        full = 100 * sum(r["returned"] == 0 for r in pos) / len(pos)
        out["n_trusted"] = len(pos)
        out["pct_betrayal"] = betr
        out["pct_full_betrayal"] = full
        print(f"  betrayal       : {betr:.0f}% of {len(pos)} trusting dyads got back < sent "
              f"({full:.0f}% got nothing)")
    return out


# --------------------------------------------------------------------------- #
# Full Berg replication                                                       #
# --------------------------------------------------------------------------- #
def run_berg_replication(chat_fn: ChatFn, cfg: GameConfig, mcfg: ModelConfig,
                         out_path: Path,
                         n_no_history: int = BERG_N_NO_HISTORY,
                         n_social_history: int = BERG_N_SOCIAL_HISTORY,
                         sleep_between: float = 0.0):
    """1) no-history cohort -> 2) build its report -> 3) social-history cohort."""
    no_hist = run_treatment(chat_fn, cfg, mcfg, "no_history", n_no_history,
                            out_path, history_report=None,
                            sleep_between=sleep_between)
    report = build_history_report(no_hist, cfg)
    soc_hist = run_treatment(chat_fn, cfg, mcfg, "social_history", n_social_history,
                             out_path, history_report=report,
                             sleep_between=sleep_between)
    summarize(no_hist, "NO-HISTORY")
    summarize(soc_hist, "SOCIAL-HISTORY")
    return no_hist, soc_hist


# --------------------------------------------------------------------------- #
# Script entry point                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Real run against gpt-oss-20b. Set the endpoint via env first, e.g.:
    #   export OPENAI_API_KEY=...           # your key
    #   (optionally) point ModelConfig.base_url at OpenRouter/Groq/Cerebras
    game = GameConfig()
    model = ModelConfig(model="gpt-oss-20b", temperature=1.0)
    out = Path("data/berg_gpt-oss-20b.jsonl")
    chat = make_chat_fn(model)
    run_berg_replication(chat, game, model, out)
