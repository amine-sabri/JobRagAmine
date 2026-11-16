"""Local generation through Ollama.

Uses /api/chat with `stream: false`: the endpoint answers in one piece, which
is what `/ask` needs, and the response carries token counts and timings that
make the latency breakdown real rather than estimated.
"""

import time
from typing import Any

import httpx

from .base import Completion, LLMError


class OllamaLLM:
    """Ollama client. `name` is the model tag, e.g. 'qwen2.5:3b-instruct'."""

    def __init__(
        self,
        name: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 600.0,
        keep_alive: str = "5m",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Keeping the model resident avoids a reload per question; on a machine
        # this tight it is also the knob that frees ~2.5 GB when set to "0s".
        self.keep_alive = keep_alive

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 250,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        started = time.perf_counter()
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unreachable at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"ollama returned invalid JSON: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        text = (body.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"ollama returned an empty answer: {body!r}"[:300])
        return Completion(
            text=text.strip(),
            model=body.get("model", self.name),
            latency_ms=round(latency_ms, 1),
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            raw={key: body[key] for key in ("total_duration", "done_reason") if key in body},
        )
