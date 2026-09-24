from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from berg_trust_game import ModelConfig, make_chat_fn

PROVIDERS: dict[str, tuple[str, str, str]] = {
    "nebius":     ("https://api.studio.nebius.com/v1/", "NEBIUS_API_KEY",     "https://studio.nebius.com"),
    "nebius-us":  ("https://api.tokenfactory.us-central1.nebius.com/v1/", "NEBIUS_API_KEY",    "https://tokenfactory.nebius.com"),
    "nebius-eu":  ("https://api.tokenfactory.eu-central1.nebius.com/v1/", "NEBIUS_API_KEY", "https://tokenfactory.nebius.com"),
    "nebius-uk":  ("https://api.tokenfactory.uk-south1.nebius.com/v1/", "NEBIUS_API_KEY", "https://tokenfactory.nebius.com"),
    "deepseek":   ("https://api.deepseek.com",          "DEEPSEEK_API_KEY",   "https://platform.deepseek.com"),
    "moonshot":   ("https://api.moonshot.ai/v1",        "MOONSHOT_API_KEY",   "https://platform.moonshot.ai"),
    "minimax":    ("https://api.minimax.io/v1",         "MINIMAX_API_KEY",    "https://www.minimax.io"),
    "zhipu":      ("https://api.z.ai/api/paas/v4",       "ZHIPU_API_KEY",      "https://z.ai"),
    "openrouter": ("https://openrouter.ai/api/v1",      "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "openai":     (None,                                 "OPENAI_API_KEY",     "https://platform.openai.com/api-keys"),
    "anthropic":  ("https://api.anthropic.com/v1/",       "ANTHROPIC_API_KEY",  "https://console.anthropic.com"),
    "groq":       ("https://api.groq.com/openai/v1",    "GROQ_API_KEY",       "https://console.groq.com/keys"),
    "cerebras":   ("https://api.cerebras.ai/v1",        "CEREBRAS_API_KEY",   "https://cloud.cerebras.ai"),
}

QWEN_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

@dataclass
class ModelSpec:
    key: str
    provider: str
    model: str
    reasoning: bool
    price_in: float
    price_out: float
    batch: bool = False
    extra_body: Optional[dict] = None
    toggle: str = ""
    note: str = ""
    api: str = "openai"

QWEN_NO_THINK   = {"chat_template_kwargs": {"enable_thinking": False}}
QWEN_THINK      = {"chat_template_kwargs": {"enable_thinking": True}}
MINIMAX_NO_THINK= {"chat_template_kwargs": {"thinking_mode": "disabled"}}
MINIMAX_THINK   = {"chat_template_kwargs": {"thinking_mode": "enabled"}}
GLM_NO_THINK    = {"chat_template_kwargs": {"enable_thinking": False}}
GLM_THINK       = {"chat_template_kwargs": {"enable_thinking": True}}
KIMI_LOW_EFFORT = {"reasoning_effort": "low"}
DEEPSEEK_THINK  = {"reasoning_effort": "high"}

GPT56_NO_THINK  = {"reasoning_effort": "none"}
GPT56_THINK     = {"reasoning_effort": "medium"}
SONNET5_NO_THINK = {"thinking": {"type": "disabled"}, "effort": "xhigh"}
SONNET5_THINK    = {"thinking": {"type": "adaptive"}, "effort": "xhigh"}

PACK: dict[str, ModelSpec] = {
    "qwen3-think":       ModelSpec("qwen3-think",       "nebius", "Qwen/Qwen3-32B", True,  0.10, 0.30,
                                   extra_body=QWEN_THINK, toggle="qwen3"),
    "qwen3-nothink":     ModelSpec("qwen3-nothink",     "nebius", "Qwen/Qwen3-32B", False, 0.10, 0.30,
                                   extra_body=QWEN_NO_THINK, toggle="qwen3"),
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
    "llama-3.3-70b":     ModelSpec("llama-3.3-70b",     "nebius", "meta-llama/Llama-3.3-70B-Instruct", False, 0.13, 0.40,
                                   note="conversational, high baseline trust, carries the variance"),
    "gpt-oss-120b":      ModelSpec("gpt-oss-120b",      "nebius", "openai/gpt-oss-120b", True, 0.15, 0.60),

    "gpt-5.6-think":     ModelSpec("gpt-5.6-think",     "openai", "gpt-5.6-luna", True,  0.20, 1.20,
                                   batch=True, extra_body=GPT56_THINK, toggle="gpt56",
                                   note="reasoning_effort=medium, the model default"),
    "gpt-5.6-nothink":   ModelSpec("gpt-5.6-nothink",   "openai", "gpt-5.6-luna", False, 0.20, 1.20,
                                   batch=True, extra_body=GPT56_NO_THINK, toggle="gpt56",
                                   note="reasoning_effort=none, a true off setting on this model"),
    "sonnet-5-think":    ModelSpec("sonnet-5-think",    "anthropic", "claude-sonnet-5", True,  2.00, 10.00,
                                   batch=True, extra_body=SONNET5_THINK, toggle="sonnet5",
                                   api="anthropic",
                                   note="adaptive thinking on, effort xhigh, the only level at which it thinks"),
    "sonnet-5-nothink":  ModelSpec("sonnet-5-nothink",  "anthropic", "claude-sonnet-5", False, 2.00, 10.00,
                                   batch=True, extra_body=SONNET5_NO_THINK, toggle="sonnet5",
                                   api="anthropic",
                                   note="thinking disabled, effort held at xhigh to match the thinking arm"),

    "judge-nemotron":    ModelSpec("judge-nemotron",    "nebius",
                                   "nvidia/Nemotron-3_5-Lightning", False, 0.10, 0.40,
                                   note="coding judge only, never a player"),
    "judge-gptoss":      ModelSpec("judge-gptoss",      "nebius",
                                   "openai/gpt-oss-120b", True, 0.15, 0.60,
                                   note="coding judge, overlaps the study by one configuration"),

    "judge-gptoss-b":    ModelSpec("judge-gptoss-b",    "nebius",
                                   "openai/gpt-oss-120b", True, 0.15, 0.60,
                                   note="second reading by the same judge, for the reliability figure"),

    "judge-hermes":      ModelSpec("judge-hermes",      "nebius",
                                   "NousResearch/Hermes-4-405B", False, 0.50, 1.50,
                                   note="coding judge only, never a player"),

    "judge-glm53":       ModelSpec("judge-glm53",       "nebius",
                                   "zai-org/GLM-5.3-Flash", False, 0.20, 0.80,
                                   note="coding judge only, never a player"),
    "judge-glm53-uk":    ModelSpec("judge-glm53-uk",    "nebius-uk",
                                   "zai-org/GLM-5.3-Flash", False, 0.20, 0.80,
                                   note="coding judge only, never a player"),
    "kimi-k3":           ModelSpec("kimi-k3",           "nebius", "moonshotai/Kimi-K3", True, 0.60, 2.50,
                                   extra_body=KIMI_LOW_EFFORT,
                                   note="thinking cannot be fully disabled, only reduced; excluded from the toggle test"),
    "llama-3.3-70b@groq":ModelSpec("llama-3.3-70b@groq","groq", "llama-3.3-70b-versatile", False, 0.59, 0.79,
                                   note="provider control against llama-3.3-70b on Nebius"),
    "gpt-oss-120b@groq": ModelSpec("gpt-oss-120b@groq", "groq", "openai/gpt-oss-120b", True, 0.15, 0.75,
                                   note="provider control against gpt-oss-120b on Nebius"),
}

TOGGLE_PAIRS = [("qwen3-think", "qwen3-nothink"),
                ("minimax-m3-think", "minimax-m3-nothink"),
                ("glm-5.2-think", "glm-5.2-nothink"),
                ("deepseek-v4-pro-think", "deepseek-v4-pro"),
                ("gpt-5.6-think", "gpt-5.6-nothink"),
                ("sonnet-5-think", "sonnet-5-nothink")]

TIER_CONTRASTS = [("deepseek-v4-pro", "deepseek-v4-flash")]

PROVIDER_CONTROLS = [("llama-3.3-70b", "llama-3.3-70b@groq"),
                     ("gpt-oss-120b", "gpt-oss-120b@groq")]

CONFIRMATORY_KEYS = [k for k in PACK if not k.endswith("@groq")]
PILOT_KEYS = ["qwen3-nothink", "qwen3-think"]
OPEN_KEYS = CONFIRMATORY_KEYS

def reasoning_class(key: str) -> str:
    return "reasoning" if PACK[key].reasoning else "plain"

def model_config(key: str, *, provider: Optional[str] = None,
                 temperature: float = 1.0, max_tokens: int = 1500,
                 min_call_interval: float = 0.0,
                 hard_timeout: float = 300.0) -> ModelConfig:
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
    missing = []
    for k in keys:
        env = PROVIDERS[PACK[k].provider][1]
        if env and not os.environ.get(env, "").strip():
            missing.append(f"{k} -> set {env}")
    return missing

if __name__ == "__main__":
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
