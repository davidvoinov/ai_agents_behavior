"""Independent provider/model configuration for Promise Switch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .schemas import ModelConfig, ReasoningMode


@dataclass(frozen=True)
class ProviderDefinition:
    name: str
    base_url: str
    api_key_env: str
    rate_limit_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelDefinition:
    provider: str
    model_alias: str
    model_id: str
    reasoning_capability: Literal["toggle", "fixed_on", "fixed_off", "unavailable"]
    default_reasoning_mode: ReasoningMode
    request_kwargs_by_mode: dict[str, dict[str, Any]] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderDefinition] = {
    "nebius": ProviderDefinition(
        name="nebius",
        base_url="https://api.studio.nebius.com/v1/",
        api_key_env="NEBIUS_API_KEY",
        rate_limit_metadata={
            "requests_per_minute": None,
            "tokens_per_minute": None,
            "source": "confirm in provider console before pilot",
        },
    ),
}

MODEL_DEFINITIONS: dict[tuple[str, str], ModelDefinition] = {
    ("nebius", "qwen3_32b"): ModelDefinition(
        provider="nebius",
        model_alias="qwen3_32b",
        model_id="Qwen/Qwen3-32B",
        reasoning_capability="toggle",
        default_reasoning_mode="on",
        request_kwargs_by_mode={
            "on": {"chat_template_kwargs": {"enable_thinking": True}},
            "off": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ),
    ("nebius", "llama_3_3_70b"): ModelDefinition(
        provider="nebius",
        model_alias="llama_3_3_70b",
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        reasoning_capability="fixed_off",
        default_reasoning_mode="off",
    ),
    ("nebius", "gpt_oss_120b_nebius"): ModelDefinition(
        provider="nebius",
        model_alias="gpt_oss_120b_nebius",
        model_id="openai/gpt-oss-120b",
        reasoning_capability="fixed_on",
        default_reasoning_mode="on",
        request_kwargs_by_mode={"on": {"reasoning_effort": "low"}},
    ),
    ("nebius", "deepseek_v4_pro_nebius"): ModelDefinition(
        provider="nebius",
        model_alias="deepseek_v4_pro_nebius",
        model_id="deepseek-ai/DeepSeek-V4-Pro",
        reasoning_capability="unavailable",
        default_reasoning_mode="unavailable",
    ),
    ("nebius", "kimi_k3_nebius"): ModelDefinition(
        provider="nebius",
        model_alias="kimi_k3_nebius",
        model_id="moonshotai/Kimi-K3",
        reasoning_capability="unavailable",
        default_reasoning_mode="unavailable",
    ),
    ("nebius", "minimax_m3_nebius"): ModelDefinition(
        provider="nebius",
        model_alias="minimax_m3_nebius",
        model_id="MiniMaxAI/MiniMax-M3",
        reasoning_capability="unavailable",
        default_reasoning_mode="unavailable",
    ),
    ("nebius", "glm_5_2_nebius"): ModelDefinition(
        provider="nebius",
        model_alias="glm_5_2_nebius",
        model_id="zai-org/GLM-5.2",
        reasoning_capability="unavailable",
        default_reasoning_mode="unavailable",
    ),
}


def create_model_config(
    *, provider: str, model_alias: str, reasoning_mode: ReasoningMode | None = None
) -> ModelConfig:
    provider_definition = PROVIDERS.get(provider)
    definition = MODEL_DEFINITIONS.get((provider, model_alias))
    if provider_definition is None:
        raise ValueError(f"Unknown Promise Switch provider: {provider}")
    if definition is None:
        raise ValueError(f"Unknown model alias {model_alias!r} for provider {provider!r}")
    selected_mode = reasoning_mode or definition.default_reasoning_mode
    _validate_reasoning_mode(definition, selected_mode)
    return ModelConfig(
        provider=provider_definition.name,
        model_alias=definition.model_alias,
        model_id=definition.model_id,
        base_url=provider_definition.base_url,
        api_key_env=provider_definition.api_key_env,
        reasoning_capability=definition.reasoning_capability,
        reasoning_mode=selected_mode,
        provider_request_kwargs=dict(definition.request_kwargs_by_mode.get(selected_mode, {})),
        rate_limit_metadata=dict(provider_definition.rate_limit_metadata),
    )


def _validate_reasoning_mode(definition: ModelDefinition, mode: ReasoningMode) -> None:
    allowed = {
        "toggle": {"on", "off"},
        "fixed_on": {"on"},
        "fixed_off": {"off"},
        "unavailable": {"unavailable"},
    }[definition.reasoning_capability]
    if mode not in allowed:
        raise ValueError(
            f"Reasoning mode {mode!r} is incompatible with {definition.reasoning_capability}"
        )
