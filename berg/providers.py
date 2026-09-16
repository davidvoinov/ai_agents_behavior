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
    # Legacy studio host; routes to the region of the project the key belongs to.
    "nebius":     ("https://api.studio.nebius.com/v1/", "NEBIUS_API_KEY",     "https://studio.nebius.com"),
    # Region-pinned hosts. Model availability differs by region, and a model that
    # is listed but not served in the caller's region HANGS instead of returning
    # 404, which is how GLM-5.2 was lost on a us-central1 project. A key is issued
    # per project, so an EU project needs its own key.
    "nebius-us":  ("https://api.tokenfactory.us-central1.nebius.com/v1/", "NEBIUS_API_KEY",    "https://tokenfactory.nebius.com"),
    "nebius-eu":  ("https://api.tokenfactory.eu-central1.nebius.com/v1/", "NEBIUS_API_KEY", "https://tokenfactory.nebius.com"),
    # GLM-5.2 is served from uk-south1. Routing is automatic on the legacy studio
    # host, but a model absent from the routed region hangs rather than failing,
    # so it is pinned here explicitly.
    "nebius-uk":  ("https://api.tokenfactory.uk-south1.nebius.com/v1/", "NEBIUS_API_KEY", "https://tokenfactory.nebius.com"),
    # DeepSeek direct — 5M free tokens on signup; cheapest for DeepSeek models.
    "deepseek":   ("https://api.deepseek.com",          "DEEPSEEK_API_KEY",   "https://platform.deepseek.com"),
    "moonshot":   ("https://api.moonshot.ai/v1",        "MOONSHOT_API_KEY",   "https://platform.moonshot.ai"),
    "minimax":    ("https://api.minimax.io/v1",         "MINIMAX_API_KEY",    "https://www.minimax.io"),
    "zhipu":      ("https://api.z.ai/api/paas/v4",       "ZHIPU_API_KEY",      "https://z.ai"),
    "openrouter": ("https://openrouter.ai/api/v1",      "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "openai":     (None,                                 "OPENAI_API_KEY",     "https://platform.openai.com/api-keys"),
    # Anthropic's OpenAI-compatible endpoint — lets ChatOpenAI call Claude directly.
    "anthropic":  ("https://api.anthropic.com/v1/",       "ANTHROPIC_API_KEY",  "https://console.anthropic.com"),
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
    api: str = "openai"        # "openai" for any OpenAI-compatible endpoint,
                               # "anthropic" for the native Claude Messages API


# --------------------------------------------------------------------------- #
# The confirmatory pack (matches smeta_berg_experiment.xlsx)                   #
# --------------------------------------------------------------------------- #
# Routing decision (David, 2026-07-22): the pilot runs on DeepSeek (its own free
# 5M tokens); every other open model runs through Nebius Token Factory; only the
# closed frontier (GPT, Claude) uses its own API.
# Toggle payloads. CONFIRM each against the Nebius model page before the
# confirmatory run, since the flag name is provider and build specific.
QWEN_NO_THINK   = {"chat_template_kwargs": {"enable_thinking": False}}
QWEN_THINK      = {"chat_template_kwargs": {"enable_thinking": True}}
# MiniMax M3 takes `thinking_mode` inside chat_template_kwargs, with three
# values: enabled (think before every response), disabled (answer directly) and
# adaptive, which is the DEFAULT WHEN UNSET and lets the model choose per turn.
# The same value tunes the minimax_m3 reasoning parser, so reasoning and content
# are split correctly in each mode. Leaving the field out yields a mixture of
# thinking and non-thinking turns, which is not a usable arm of a toggle.
MINIMAX_NO_THINK= {"chat_template_kwargs": {"thinking_mode": "disabled"}}
MINIMAX_THINK   = {"chat_template_kwargs": {"thinking_mode": "enabled"}}
GLM_NO_THINK    = {"chat_template_kwargs": {"enable_thinking": False}}
GLM_THINK       = {"chat_template_kwargs": {"enable_thinking": True}}
KIMI_LOW_EFFORT = {"reasoning_effort": "low"}                             # cannot be fully disabled
# DeepSeek V4 answers with NO reasoning by default and is switched on with
# reasoning_effort. Levels low and max return an essentially empty content field
# (one character) because the whole answer goes into the reasoning block, so only
# high leaves a usable completion. Verified on Nebius at max_tokens 4000.
DEEPSEEK_THINK  = {"reasoning_effort": "high"}

# ---- closed frontier, added 2026-09-11 ----
# Both families expose a real off switch, so both give the on/off contrast H4
# needs. The effort level at which each is run differs, because the level that
# produces thinking differs by family, and H4 is estimated within a family.
#
# GPT-5.6 Luna takes reasoning_effort with none, low, medium (default), high,
# xhigh and max. Verified on the model page 2026-09-11.
GPT56_NO_THINK  = {"reasoning_effort": "none"}
GPT56_THINK     = {"reasoning_effort": "medium"}
# Claude Sonnet 5 runs adaptive thinking by default and is switched off with
# thinking {"type": "disabled"}. What returns 400 is MANUAL extended thinking,
# meaning {"type": "enabled", "budget_tokens": N}, which is a different thing.
#
# The effort level is xhigh and not medium, for a measured reason. A sweep on
# 2026-09-11 over the exact investor prompt found adaptive thinking returning NO
# thinking block at low, medium or high, and one block per call at xhigh, where
# the arm also draws 278 output tokens above the disabled one. Below xhigh the
# two arms are the same manipulation twice, so xhigh is the only setting at
# which this family tests H4 at all. Effort is pinned in BOTH arms, since it
# shapes every output token and not only the thinking, which leaves the thinking
# switch as the one thing that differs.
SONNET5_NO_THINK = {"thinking": {"type": "disabled"}, "effort": "xhigh"}
SONNET5_THINK    = {"thinking": {"type": "adaptive"}, "effort": "xhigh"}

PACK: dict[str, ModelSpec] = {
    # ---- within-family reasoning toggles (the causal test of H4) ----
    "qwen3-think":       ModelSpec("qwen3-think",       "nebius", "Qwen/Qwen3-32B", True,  0.10, 0.30,
                                   extra_body=QWEN_THINK, toggle="qwen3"),
    "qwen3-nothink":     ModelSpec("qwen3-nothink",     "nebius", "Qwen/Qwen3-32B", False, 0.10, 0.30,
                                   extra_body=QWEN_NO_THINK, toggle="qwen3"),
    # DeepSeek at the server default, which emits no reasoning. Pro against Flash
    # is therefore a contrast of model TIERS with reasoning off in both, and not a
    # contrast of reasoning modes. Kept as a capability comparison.
    "deepseek-v4-pro":   ModelSpec("deepseek-v4-pro",   "nebius", "deepseek-ai/DeepSeek-V4-Pro",   False, 0.435, 0.87,
                                   toggle="deepseek",
                                   note="server default, reasoning off; the no-think arm of the DeepSeek toggle"),
    "deepseek-v4-flash": ModelSpec("deepseek-v4-flash", "nebius", "deepseek-ai/DeepSeek-V4-Flash", False, 0.14,  0.28,
                                   note="server default, reasoning off; smaller tier"),
    "deepseek-v4-pro-think": ModelSpec("deepseek-v4-pro-think", "nebius", "deepseek-ai/DeepSeek-V4-Pro", True, 0.435, 0.87,
                                   extra_body=DEEPSEEK_THINK, toggle="deepseek",
                                   note="reasoning on via reasoning_effort=high; PILOT FIRST, the content field "
                                        "can come back empty and that failure would fall on this arm alone"),
    "minimax-m3-think":  ModelSpec("minimax-m3-think",  "nebius", "MiniMaxAI/MiniMax-M3", True,  0.30, 1.20,
                                   extra_body=MINIMAX_THINK, toggle="minimax"),
    "minimax-m3-nothink":ModelSpec("minimax-m3-nothink","nebius", "MiniMaxAI/MiniMax-M3", False, 0.30, 1.20,
                                   extra_body=MINIMAX_NO_THINK, toggle="minimax"),
    "glm-5.2-think":     ModelSpec("glm-5.2-think",     "nebius-uk", "zai-org/GLM-5.2", True,  0.43, 1.74,
                                   extra_body=GLM_THINK, toggle="glm",
                                   note="served from uk-south1; pinned to that region because an absent model hangs"),
    "glm-5.2-nothink":   ModelSpec("glm-5.2-nothink",   "nebius-uk", "zai-org/GLM-5.2", False, 0.43, 1.74,
                                   extra_body=GLM_NO_THINK, toggle="glm",
                                   note="served from uk-south1; pinned to that region because an absent model hangs"),
    # ---- fixed-mode models (between-model axis only) ----
    "llama-3.3-70b":     ModelSpec("llama-3.3-70b",     "nebius", "meta-llama/Llama-3.3-70B-Instruct", False, 0.13, 0.40,
                                   note="conversational, high baseline trust, carries the variance"),
    "gpt-oss-120b":      ModelSpec("gpt-oss-120b",      "nebius", "openai/gpt-oss-120b", True, 0.15, 0.60),

    # ---- closed frontier, two more within-family reasoning toggles ----
    # GPT-5.6 Luna at $0.20/$1.20 list, halved by the Batch API.
    "gpt-5.6-think":     ModelSpec("gpt-5.6-think",     "openai", "gpt-5.6-luna", True,  0.20, 1.20,
                                   batch=True, extra_body=GPT56_THINK, toggle="gpt56",
                                   note="reasoning_effort=medium, the model default"),
    "gpt-5.6-nothink":   ModelSpec("gpt-5.6-nothink",   "openai", "gpt-5.6-luna", False, 0.20, 1.20,
                                   batch=True, extra_body=GPT56_NO_THINK, toggle="gpt56",
                                   note="reasoning_effort=none, a true off setting on this model"),
    # Claude Sonnet 5 at $2/$10 list, halved by Message Batches. Reached through
    # the native Messages API, because the OpenAI compatibility layer ignores
    # the thinking field, returns an empty completion_tokens_details and offers
    # no Batch API. temperature must stay at the model default, since a
    # non-default value returns 400 here.
    "sonnet-5-think":    ModelSpec("sonnet-5-think",    "anthropic", "claude-sonnet-5", True,  2.00, 10.00,
                                   batch=True, extra_body=SONNET5_THINK, toggle="sonnet5",
                                   api="anthropic",
                                   note="adaptive thinking on, effort xhigh, the only level at which it thinks"),
    "sonnet-5-nothink":  ModelSpec("sonnet-5-nothink",  "anthropic", "claude-sonnet-5", False, 2.00, 10.00,
                                   batch=True, extra_body=SONNET5_NO_THINK, toggle="sonnet5",
                                   api="anthropic",
                                   note="thinking disabled, effort held at xhigh to match the thinking arm"),

    # ---- coding judge, deliberately outside the study set ----
    # The transcript coder must not be a model whose own transcripts it reads,
    # and every other entry here is under study, so the judge is registered
    # separately and appears in no estimate. Prices are placeholders until the
    # first invoice, since only the relative cost of the coding run depends on
    # them and the run is small.
    "judge-nemotron":    ModelSpec("judge-nemotron",    "nebius",
                                   "nvidia/Nemotron-3_5-Lightning", False, 0.10, 0.40,
                                   note="coding judge only, never a player"),
    # The coding judge in use. gpt-oss-120b also appears in the study, so it reads
    # its own 700 transcripts among the 7,925, which is a dependence the paper
    # states. It is registered under its own key so the choice is visible in the
    # `coder` field of every coded row and cannot be mistaken for an oversight.
    # Candidates outside the study were screened first and failed on output
    # format, which is why the dependence is accepted rather than avoided.
    "judge-gptoss":      ModelSpec("judge-gptoss",      "nebius",
                                   "openai/gpt-oss-120b", True, 0.15, 0.60,
                                   note="coding judge, overlaps the study by one configuration"),

    # The same weights at the same settings under a second coder label, so the
    # four frontier arms get the reread that gives the machine reliability
    # figure. The eleven earlier arms have one because an earlier judge key
    # covered 775 of them, and without this the new arms would carry a human
    # comparison and no machine one. The resume in run_judge.py keys on the
    # coder, so a distinct key is what makes a second pass happen at all.
    # Prefer a genuinely independent judge if screen_judges.py finds one that
    # honours the schema, since a reread measures how reproducible the
    # instrument is and not how much two instruments agree.
    "judge-gptoss-b":    ModelSpec("judge-gptoss-b",    "nebius",
                                   "openai/gpt-oss-120b", True, 0.15, 0.60,
                                   note="second reading by the same judge, for the reliability figure"),

    # Hermes-4-405B shares no family with any model in the study, so it carries
    # no lineage in common with the transcripts it reads, and the Hermes line is
    # tuned for structured output, which is where the previous candidate failed.
    # Its 70B sibling was withdrawn from serverless in August, so the id is worth
    # confirming against the live catalogue before a run.
    "judge-hermes":      ModelSpec("judge-hermes",      "nebius",
                                   "NousResearch/Hermes-4-405B", False, 0.50, 1.50,
                                   note="coding judge only, never a player"),

    # GLM-5.3-Flash is not a model under study, since the study carries GLM-5.2
    # in its two modes and this is a different version at a different tier. The
    # family and the developer are shared, which is a weaker dependence than a
    # model reading its own transcripts and is reported as a limitation.
    # Registered on both regions, because the 5.2 line was served only from
    # uk-south1 and the 5.3 line may follow it.
    "judge-glm53":       ModelSpec("judge-glm53",       "nebius",
                                   "zai-org/GLM-5.3-Flash", False, 0.20, 0.80,
                                   note="coding judge only, never a player"),
    "judge-glm53-uk":    ModelSpec("judge-glm53-uk",    "nebius-uk",
                                   "zai-org/GLM-5.3-Flash", False, 0.20, 0.80,
                                   note="coding judge only, never a player"),
    "kimi-k3":           ModelSpec("kimi-k3",           "nebius", "moonshotai/Kimi-K3", True, 0.60, 2.50,
                                   extra_body=KIMI_LOW_EFFORT,
                                   note="thinking cannot be fully disabled, only reduced; excluded from the toggle test"),
    # ---- provider-bias control: same weights, second provider ----
    "llama-3.3-70b@groq":ModelSpec("llama-3.3-70b@groq","groq", "llama-3.3-70b-versatile", False, 0.59, 0.79,
                                   note="provider control against llama-3.3-70b on Nebius"),
    "gpt-oss-120b@groq": ModelSpec("gpt-oss-120b@groq", "groq", "openai/gpt-oss-120b", True, 0.15, 0.75,
                                   note="provider control against gpt-oss-120b on Nebius"),
}

# Clean within-family toggles: SAME weights or same family, two reasoning modes.
TOGGLE_PAIRS = [("qwen3-think", "qwen3-nothink"),
                ("minimax-m3-think", "minimax-m3-nothink"),
                ("glm-5.2-think", "glm-5.2-nothink"),
                ("deepseek-v4-pro-think", "deepseek-v4-pro"),   # pilot before committing
                ("gpt-5.6-think", "gpt-5.6-nothink"),
                ("sonnet-5-think", "sonnet-5-nothink")]

# Same family, different capability tier, reasoning off in both. This is a
# capability comparison and must not be reported as a reasoning contrast.
TIER_CONTRASTS = [("deepseek-v4-pro", "deepseek-v4-flash")]


# Same weights on two providers, run on one matched cell to separate the model
# from its serving stack.
PROVIDER_CONTROLS = [("llama-3.3-70b", "llama-3.3-70b@groq"),
                     ("gpt-oss-120b", "gpt-oss-120b@groq")]

CONFIRMATORY_KEYS = [k for k in PACK if not k.endswith("@groq")]
PILOT_KEYS = ["qwen3-nothink", "qwen3-think"]
OPEN_KEYS = CONFIRMATORY_KEYS


def reasoning_class(key: str) -> str:
    """Label a model for the between-model reasoning-vs-plain contrast."""
    return "reasoning" if PACK[key].reasoning else "plain"


def model_config(key: str, *, provider: Optional[str] = None,
                 temperature: float = 1.0, max_tokens: int = 1500,
                 min_call_interval: float = 0.0,
                 hard_timeout: float = 300.0) -> ModelConfig:
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
        hard_timeout=hard_timeout,
    )


def chat_for(key: str, **kw):
    """Build the injectable chat_fn for a pack key, on whichever API it needs.

    Every runner goes through here, so a model that is not OpenAI-shaped is
    dispatched once rather than in each script. Claude Sonnet 5 is the case that
    forced this: its arms are defined by `thinking` and `output_config.effort`,
    and the OpenAI compatibility layer accepts neither, so routing it through
    make_chat_fn leaves the two arms running the same configuration twice.
    """
    spec = PACK[key]
    if spec.api == "anthropic":
        from anthropic_chat import make_anthropic_chat_fn
        eb = spec.extra_body or {}
        return make_anthropic_chat_fn(
            model=spec.model,
            effort=eb.get("effort"),
            thinking=eb.get("thinking"),
            max_tokens=kw.get("max_tokens", 4000),
            temperature=kw.get("temperature", 1.0),
            min_call_interval=kw.get("min_call_interval", 0.0),
            api_key_env=PROVIDERS[spec.provider][1],
        )
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
