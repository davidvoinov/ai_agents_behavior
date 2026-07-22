"""
providers.py
============
Central registry of API providers and model configurations for the confirmatory
runs. One place to define every endpoint, model id, price, and the reasoning
on/off toggle, so runners never hard-code a base_url again.

All endpoints are OpenAI-compatible and are reached through
berg_trust_game.ModelConfig / make_chat_fn. Closed frontier models (GPT, Claude)
are reached either natively or through OpenRouter, which is also OpenAI-compatible.

Design factors this registry encodes (PI directive 2026-07-22):
  * model family
  * reasoning ON vs OFF, as a WITHIN-FAMILY toggle
      - DeepSeek : v4-pro (reasoning) vs v4-flash (plain)
      - Qwen3    : same weights, enable_thinking True vs False (extra_body)
  * provider, so the provider-confound control can route ONE model through TWO
    providers by overriding `provider=` at build time.

Prices are July 2026, USD per 1M tokens (input, output); see smeta_berg_experiment.xlsx.
Keep in sync with that file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from berg_trust_game import ModelConfig, make_chat_fn


# --------------------------------------------------------------------------- #
# Providers: name -> (base_url, api_key_env, signup)                          #
# --------------------------------------------------------------------------- #
PROVIDERS: dict[str, tuple[str, str, str]] = {
    # Open-weights, NO rate throttling — the pilot workhorse (see NEBIUS_GUIDE.md).
    "nebius":     ("https://api.studio.nebius.com/v1/", "NEBIUS_API_KEY",     "https://studio.nebius.com"),
    # DeepSeek direct — 5M free tokens on signup; cheapest for DeepSeek models.
    "deepseek":   ("https://api.deepseek.com",          "DEEPSEEK_API_KEY",   "https://platform.deepseek.com"),
    "moonshot":   ("https://api.moonshot.ai/v1",        "MOONSHOT_API_KEY",   "https://platform.moonshot.ai"),
    "minimax":    ("https://api.minimax.io/v1",         "MINIMAX_API_KEY",    "https://www.minimax.io"),
    "zhipu":      ("https://api.z.ai/api/paas/v4",       "ZHIPU_API_KEY",      "https://z.ai"),
    "openrouter": ("https://openrouter.ai/api/v1",      "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "openai":     (None,                                 "OPENAI_API_KEY",     "https://platform.openai.com/api-keys"),
    # legacy free tiers kept only for reference — they cause the attrition we log.
    "groq":       ("https://api.groq.com/openai/v1",    "GROQ_API_KEY",       "https://console.groq.com/keys"),
    "cerebras":   ("https://api.cerebras.ai/v1",        "CEREBRAS_API_KEY",   "https://cloud.cerebras.ai"),
}

# extra_body that turns Qwen3 thinking OFF on an OpenAI-compatible endpoint.
QWEN_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class ModelSpec:
    """A named model configuration used in the confirmatory pack."""
    key: str
    provider: str
    model: str                 # exact model id on that provider (confirm in console)
    reasoning: bool            # True = reasoning mode, False = plain/conversational
    price_in: float            # USD / 1M input tokens
    price_out: float           # USD / 1M output tokens
    batch: bool = False        # closed frontier with a 50%-off Batch API
    extra_body: Optional[dict] = None
    toggle: str = ""           # within-family reasoning toggle group; "" = fixed mode
    note: str = ""


# --------------------------------------------------------------------------- #
# The confirmatory pack (matches smeta_berg_experiment.xlsx)                   #
# --------------------------------------------------------------------------- #
# Routing decision (David, 2026-07-22): the pilot runs on DeepSeek (its own free
# 5M tokens); every other open model runs through Nebius Token Factory; only the
# closed frontier (GPT, Claude) uses its own API.
PACK: dict[str, ModelSpec] = {
    "deepseek-v4-pro":  ModelSpec("deepseek-v4-pro",  "deepseek", "deepseek-v4-pro",   True,  0.435, 0.87,
                                  toggle="deepseek", note="reasoning arm of the DeepSeek toggle"),
    "deepseek-v4-flash":ModelSpec("deepseek-v4-flash","deepseek", "deepseek-v4-flash", False, 0.14,  0.28,
                                  toggle="deepseek", note="plain arm of the DeepSeek toggle"),
    "qwen3-think":      ModelSpec("qwen3-think",      "nebius", "Qwen/Qwen3-235B-A22B", True,  0.20,  0.60,
                                  toggle="qwen3", note="reasoning arm; enable_thinking default True"),
    "qwen3-nothink":    ModelSpec("qwen3-nothink",    "nebius", "Qwen/Qwen3-235B-A22B", False, 0.20,  0.60,
                                  extra_body=QWEN_NO_THINK, toggle="qwen3",
                                  note="plain arm; same weights, thinking OFF"),
    # fixed-mode models: no clean documented on/off flag, so each is one point on
    # the BETWEEN-model reasoning-vs-plain axis rather than a within-family toggle.
    "minimax-m2":       ModelSpec("minimax-m2",       "nebius", "MiniMaxAI/MiniMax-M2", True,  0.30,  1.20,
                                  note="confirm exact Nebius model id in console"),
    "glm-4.6":          ModelSpec("glm-4.6",          "nebius", "zai-org/GLM-4.6",      True,  0.43,  1.74,
                                  note="GLM has a thinking flag; add a nothink twin here if a toggle pair is wanted"),
    "kimi-k2.6":        ModelSpec("kimi-k2.6",        "nebius", "moonshotai/Kimi-K2.6", True,  0.60,  2.50,
                                  note="confirm exact Nebius model id in console"),
    "gpt-5.6-luna":     ModelSpec("gpt-5.6-luna",     "openai", "gpt-5.6-luna",         True,  1.00,  6.00, batch=True),
    "claude-sonnet-5":  ModelSpec("claude-sonnet-5",  "openrouter","anthropic/claude-sonnet-5", True, 2.00, 10.00, batch=True),
    # Free-tier pilot fallbacks (Cerebras). Free limits 2026: 5 req/min, 30K
    # tokens/min, 1M tokens/day, 8192-token context — pace with MIN_CALL_INTERVAL=12
    # and keep to no_history (communication transcripts risk the 8192 cap).
    "gpt-oss-120b":     ModelSpec("gpt-oss-120b",     "cerebras", "gpt-oss-120b",       True,  0.35,  0.75,
                                  note="Cerebras free tier; our headline model"),
    "glm-4.7":          ModelSpec("glm-4.7",          "cerebras", "glm-4.7",            True,  0.0,   0.0,
                                  note="Cerebras free tier; second free model"),
}

# Free-tier pilot: no card, no DeepSeek credit needed.
PILOT_FREE_KEYS = ["gpt-oss-120b"]

# CLEAN within-family reasoning toggles (the causal test of H4). Only pairs where
# the SAME model runs in two modes count here. GLM, Kimi, and Claude expose a
# thinking flag too and can be added as pairs when needed; MiniMax and GPT do
# not toggle cleanly and stay fixed-mode.
TOGGLE_PAIRS = [("deepseek-v4-pro", "deepseek-v4-flash"),
                ("qwen3-think", "qwen3-nothink")]

# Pilot pack: DeepSeek on its own free credit; nothing closed.
PILOT_KEYS = ["deepseek-v4-pro", "deepseek-v4-flash"]
# Full open pack (DeepSeek + everything routed via Nebius).
OPEN_KEYS = ["deepseek-v4-pro", "deepseek-v4-flash", "qwen3-think",
             "qwen3-nothink", "minimax-m2", "glm-4.6", "kimi-k2.6"]


def reasoning_class(key: str) -> str:
    """Label a model for the between-model reasoning-vs-plain contrast."""
    return "reasoning" if PACK[key].reasoning else "plain"


def model_config(key: str, *, provider: Optional[str] = None,
                 temperature: float = 1.0, max_tokens: int = 1500,
                 min_call_interval: float = 0.0) -> ModelConfig:
    """Build a ModelConfig for a pack key.

    `provider=` overrides the default provider for the SAME model — this is how
    the provider-confound control (PI point 4) routes one model through a second
    endpoint (e.g. model_config("qwen3-think", provider="openrouter")).
    """
    spec = PACK[key]
    prov = provider or spec.provider
    if prov not in PROVIDERS:
        raise ValueError(f"unknown provider {prov!r}; options: {', '.join(PROVIDERS)}")
    base_url, key_env, _ = PROVIDERS[prov]
    return ModelConfig(
        model=spec.model,
        base_url=base_url,
        api_key_env=key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        min_call_interval=min_call_interval,
        extra_body=spec.extra_body,
    )


def chat_for(key: str, **kw):
    """Convenience: build the injectable chat_fn for a pack key."""
    return make_chat_fn(model_config(key, **kw))


def require_keys(keys: list[str]) -> list[str]:
    """Return the pack keys whose API key env var is missing (so a runner can
    fail fast with a clear message instead of mid-run)."""
    missing = []
    for k in keys:
        env = PROVIDERS[PACK[k].provider][1]
        if env and not os.environ.get(env, "").strip():
            missing.append(f"{k} -> set {env}")
    return missing


if __name__ == "__main__":
    # quick offline sanity check — no network
    print("Providers:", ", ".join(PROVIDERS))
    print("\nConfirmatory pack:")
    for k, s in PACK.items():
        mode = "reasoning" if s.reasoning else "plain"
        tog = f" toggle:{s.toggle}" if s.toggle else ""
        print(f"  {k:18} {s.provider:10} {s.model:28} {mode:9} "
              f"${s.price_in}/{s.price_out}" + ("  [batch]" if s.batch else "") + tog)
    print("\nClean within-family reasoning toggles:", TOGGLE_PAIRS)
    print("Pilot keys (DeepSeek only):", PILOT_KEYS)
    mc = model_config("qwen3-nothink")
    assert mc.extra_body == QWEN_NO_THINK, "no-think toggle not wired"
    mc2 = model_config("qwen3-think", provider="openrouter")
    assert "openrouter" in (mc2.base_url or ""), "provider override failed"
    print("\nOK: no-think toggle wired, provider override works.")
