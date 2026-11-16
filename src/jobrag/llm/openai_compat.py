"""Generation through any OpenAI-compatible /chat/completions endpoint.

Not used by default — it exists so phase 9 can compare the local 3B model with
an API model on the same questions, which is the only honest way to say whether
the local model is the limiting factor. OpenAI, Mistral, Groq, Together and a
self-hosted vLLM all speak this protocol, so one client covers the comparison.

The key is read from the environment and never logged.
"""

import os
import time
from typing import Any

import httpx

from .base import Completion, LLMError


class OpenAICompatLLM:
    def __init__(
        self,
        name: str,
        *,
        base_url: str | None = None,
        api_key_env: str = "LLM_API_KEY",
        timeout: float = 120.0,
    ) -> None:
        self.name = name
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        if not self.base_url:
            raise LLMError("no base_url given and LLM_BASE_URL is not set")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> Completion:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMError(f"{self.api_key_env} is not set")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            # The message may contain the URL but never the key.
            raise LLMError(f"{self.base_url} failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        choices = body.get("choices") or []
        if not choices:
            raise LLMError("the API returned no choice")
        usage = body.get("usage") or {}
        return Completion(
            text=(choices[0].get("message") or {}).get("content", "").strip(),
            model=body.get("model", self.name),
            latency_ms=round(latency_ms, 1),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
