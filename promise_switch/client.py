"""Provider-agnostic request foundation with no built-in network transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .schemas import ModelConfig, NormalizedResponse, sanitize_for_serialization


@dataclass(frozen=True)
class RequestSpec:
    base_url: str
    model_id: str
    api_key_env: str
    body: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        """Return safe request metadata; no secret value can be represented."""
        return sanitize_for_serialization({
            "base_url": self.base_url,
            "model_id": self.model_id,
            "api_key_env": self.api_key_env,
            "body": self.body,
            "metadata": self.metadata,
        })


class CompletionTransport(Protocol):
    def complete(self, request: RequestSpec) -> NormalizedResponse: ...


class ProviderAgnosticClient:
    """Build requests now; accept a real or fake transport in a later layer."""

    def __init__(self, model_config: ModelConfig, transport: CompletionTransport | None = None):
        self.model_config = model_config
        self.transport = transport

    def build_request(
        self,
        *,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> RequestSpec:
        body: dict[str, Any] = {
            "model": self.model_config.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            self.model_config.output_token_field: max_tokens,
        }
        body.update(self.model_config.provider_request_kwargs)
        return RequestSpec(
            base_url=self.model_config.base_url,
            model_id=self.model_config.model_id,
            api_key_env=self.model_config.api_key_env,
            body=body,
            metadata={
                "provider": self.model_config.provider,
                "model_alias": self.model_config.model_alias,
                "reasoning_mode": self.model_config.reasoning_mode,
            },
        )

    def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> NormalizedResponse:
        if self.transport is None:
            raise RuntimeError("No transport configured; foundation performs no API calls")
        request = self.build_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self.transport.complete(request)
