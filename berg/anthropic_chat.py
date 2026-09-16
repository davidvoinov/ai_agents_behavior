from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

USAGE: list[dict] = []
_LOCK = threading.Lock()

_DEFAULT_TEMPERATURE = 1.0

def _client(api_key_env: str = "ANTHROPIC_API_KEY"):
    try:
        import anthropic
    except ImportError as exc:
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
            except Exception as exc:
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
    ti = sum(u["input_tokens"] or 0 for u in USAGE)
    to = sum(u["output_tokens"] or 0 for u in USAGE)
    f = 0.5 if batch else 1.0
    return {"calls": len(USAGE), "input_tokens": ti, "output_tokens": to,
            "usd": (ti / 1e6 * price_in + to / 1e6 * price_out) * f}
