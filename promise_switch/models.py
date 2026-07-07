"""Model catalog filtering and metadata helpers."""

from __future__ import annotations

from typing import Any


def filter_zero_price_text_models(
    models: list[dict[str, Any]],
    exclude_model_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = exclude_model_ids or set()
    filtered = []
    for model in models:
        if is_zero_price_text_model(model, excluded):
            filtered.append(model)
    filtered.sort(key=lambda item: item.get("id", ""))
    return filtered


def is_zero_price_text_model(model: dict[str, Any], excluded: set[str]) -> bool:
    model_id = str(model.get("id", ""))
    if not model_id or model_id in excluded:
        return False
    pricing = model.get("pricing") or {}
    if str(pricing.get("prompt")) != "0" or str(pricing.get("completion")) != "0":
        return False
    architecture = model.get("architecture") or {}
    inputs = set(architecture.get("input_modalities") or [])
    outputs = set(architecture.get("output_modalities") or [])
    modality = str(architecture.get("modality") or "").lower()
    name = str(model.get("name") or "").lower()
    if "text" not in inputs or "text" not in outputs:
        return False
    if "audio" in outputs or "audio" in modality:
        return False
    if "router" in model_id.lower() or model_id == "openrouter/free":
        return False
    if "content safety" in name or "content-safety" in model_id.lower():
        return False
    return True


def models_from_config_ids(model_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": model_id,
            "name": model_id,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "modality": "text->text",
            },
            "supported_parameters": [
                "include_reasoning",
                "max_tokens",
                "reasoning",
                "temperature",
                "response_format",
            ],
        }
        for model_id in model_ids
    ]


def supported_parameters(model: dict[str, Any]) -> set[str]:
    values = model.get("supported_parameters") or []
    return {str(value) for value in values}
