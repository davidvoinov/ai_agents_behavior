"""Minimal OpenRouter client."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ApiResult:
    ok: bool
    status: int | None
    data: dict[str, Any] | None
    text: str
    error: str | None
    latency_s: float
    headers: dict[str, str]


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: int = 90,
        app_title: str = "Promise Switch Experiment",
        http_referer: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.app_title = app_title
        self.http_referer = http_referer

    def get_models(self) -> ApiResult:
        return self._request("GET", "/models", auth=False)

    def get_key_info(self) -> ApiResult:
        return self._request("GET", "/key", auth=True)

    def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        supported_parameters: set[str] | None = None,
        use_response_format: bool = True,
    ) -> ApiResult:
        supported = supported_parameters or set()
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "include_reasoning" in supported:
            body["include_reasoning"] = False
        if "reasoning" in supported:
            body["reasoning"] = {"enabled": False}
        if use_response_format and "response_format" in supported:
            body["response_format"] = {"type": "json_object"}
        return self._request("POST", "/chat/completions", body=body, auth=True)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        auth: bool = True,
    ) -> ApiResult:
        if auth and not self.api_key:
            return ApiResult(
                ok=False,
                status=None,
                data=None,
                text="",
                error="Missing OpenRouter API key",
                latency_s=0.0,
                headers={},
            )

        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-Title": self.app_title,
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.monotonic()
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=body,
                timeout=self.timeout_s,
            )
            text = response.text
            latency_s = time.monotonic() - started
            data = _loads_json(text)
            ok = 200 <= response.status_code < 300
            return ApiResult(
                ok=ok,
                status=response.status_code,
                data=data,
                text=text,
                error=None if ok else _error_message(data, text, f"HTTP {response.status_code}"),
                latency_s=latency_s,
                headers=dict(response.headers.items()),
            )
        except requests.RequestException as exc:
            latency_s = time.monotonic() - started
            return ApiResult(
                ok=False,
                status=None,
                data=None,
                text="",
                error=str(exc),
                latency_s=latency_s,
                headers={},
            )


def _loads_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _error_message(data: dict[str, Any] | None, text: str, fallback: str) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        if isinstance(error, str) and error:
            return error
    return text[:500] if text else fallback
