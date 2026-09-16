#!/usr/bin/env python3
"""
anthropic_chat.py — a chat_fn over the native Claude Messages API.

Why this exists rather than routing Claude through the OpenAI-compatible
endpoint already registered in providers.py. The compatibility layer exposes no
`output_config`, returns an empty `completion_tokens_details`, and offers no
Batch API. All three matter here, since the arms are defined by the thinking
switch, the cost estimate needs the token split, and the run is meant to go
through Message Batches at half price. Verified against the compatibility table
on 2026-09-11.

Thinking on Claude Sonnet 5 is adaptive and on by default. It is switched off
with thinking {"type": "disabled"}. Manual extended thinking, meaning
{"type": "enabled", "budget_tokens": N}, is removed on this model and returns a
400, which is a different parameter and not the off switch.

`thinking.type` and `effort` are different controls and not alternatives. The
type is adaptive or disabled and says whether Claude thinks at all, while effort
is low, medium, high, xhigh or max and says how much work goes into the whole
response. The documentation states outright that adaptive must not be passed as
an effort value.

`display` defaults to "omitted" on this model, so a thinking block comes back
with an EMPTY thinking field and the text is invisible unless "summarized" is
asked for. Counting characters of that field therefore measures the display
setting and not whether the model thought. The block count is the honest signal,
and the token total is the honest cost, since an omitted block is billed in
full.

The engine injects `chat_fn(system, user) -> str`, so this file only has to
produce that callable. Token usage is recorded as a side effect in USAGE, which
the smoke test reads to measure what thinking actually costs.

    from anthropic_chat import make_anthropic_chat_fn, USAGE
    chat = make_anthropic_chat_fn("claude-sonnet-5", effort="medium",
                                  thinking={"type": "disabled"})
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

# Every call appends one record, so a caller can price a run after the fact.
USAGE: list[dict] = []
_LOCK = threading.Lock()

# Claude Sonnet 5 rejects a non-default temperature with a 400, so the parameter
# is sent only when it equals the model default. The study runs at 1.0, which is
# that default, and the field is therefore simply omitted.
_DEFAULT_TEMPERATURE = 1.0


def _client(api_key_env: str = "ANTHROPIC_API_KEY"):
    try:
        import anthropic
    except ImportError as exc:                                  # noqa: BLE001
        raise SystemExit(
            "the anthropic package is required for the native Claude client\n"
            "    pip install anthropic --break-system-packages"
        ) from exc
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise SystemExit(f"set {api_key_env} before calling Claude")
    return anthropic.Anthropic(api_key=key)


def make_anthropic_chat_fn(
    model: str = "claude-sonnet-5",
    effort: Optional[str] = None,
    thinking: Optional[dict] = None,
    display: Optional[str] = "summarized",
    max_tokens: int = 4000,
    temperature: float = _DEFAULT_TEMPERATURE,
    api_key_env: str = "ANTHROPIC_API_KEY",
    min_call_interval: float = 0.0,
    max_retries: int = 4,
) -> Callable[[str, str], str]:
    """Build the injectable chat_fn.

    `effort` is low, medium, high, xhigh or max and shapes every output token.
    `thinking` is {"type": "adaptive"} or {"type": "disabled"} and is the on/off
    switch. Holding effort fixed across the two arms leaves thinking as the only
    thing that differs between them.
    """
    client = _client(api_key_env)
    last = [0.0]

    def chat_fn(system: str, user: str) -> str:
        if min_call_interval > 0:
            wait = min_call_interval - (time.time() - last[0])
            if wait > 0:
                time.sleep(wait)
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if effort:
            kwargs["output_config"] = {"effort": effort}
        if thinking:
            th = dict(thinking)
            # display is rejected alongside type disabled, since there is
            # nothing to display, so it is attached only to a thinking arm.
            if display and th.get("type") != "disabled":
                th.setdefault("display", display)
            kwargs["thinking"] = th
        if temperature != _DEFAULT_TEMPERATURE:
            kwargs["temperature"] = temperature

        delay = 2.0
        for attempt in range(max_retries):
            try:
                r = client.messages.create(**kwargs)
                break
            except Exception as exc:                            # noqa: BLE001
                name = type(exc).__name__
                retryable = any(s in name for s in
                                ("RateLimit", "Overloaded", "APIConnection",
                                 "InternalServer", "Timeout"))
                if not retryable or attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        last[0] = time.time()

        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        # Count BLOCKS, because the text inside one is empty whenever display is
        # omitted. A block present with zero characters still means Claude thought.
        blocks = [b for b in r.content
                  if getattr(b, "type", "") in ("thinking", "redacted_thinking")]
        think = sum(len(getattr(b, "thinking", "") or "") for b in blocks)
        u = r.usage
        with _LOCK:
            USAGE.append({
                "model": model,
                "effort": effort or "high (default)",
                "thinking": (thinking or {}).get("type", "adaptive (default)"),
                "input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None),
                "thinking_blocks": len(blocks),
                "thinking_chars_visible": think,
                "stop_reason": getattr(r, "stop_reason", None),
            })
        return text

    return chat_fn


def usage_total(price_in: float, price_out: float, batch: bool = False) -> dict:
    """Sum the recorded usage and price it. Prices are USD per 1M tokens."""
    ti = sum(u["input_tokens"] or 0 for u in USAGE)
    to = sum(u["output_tokens"] or 0 for u in USAGE)
    f = 0.5 if batch else 1.0
    return {"calls": len(USAGE), "input_tokens": ti, "output_tokens": to,
            "usd": (ti / 1e6 * price_in + to / 1e6 * price_out) * f}
