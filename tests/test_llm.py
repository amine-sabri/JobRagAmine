"""The generation backends, with the HTTP layer faked."""

import httpx
import pytest

from jobrag.llm import LLM, Completion, LLMError, OllamaLLM, OpenAICompatLLM

OLLAMA_BODY = {
    "model": "qwen2.5:3b-instruct",
    "message": {"role": "assistant", "content": "  Ingénieur DevOps chez ACME [1].  "},
    "prompt_eval_count": 877,
    "eval_count": 74,
    "total_duration": 116_000_000_000,
    "done_reason": "stop",
}


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_both_backends_satisfy_the_protocol():
    assert isinstance(OllamaLLM("m"), LLM)
    assert isinstance(OpenAICompatLLM("m", base_url="https://api.example.com/v1"), LLM)


def test_ollama_sends_the_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse(OLLAMA_BODY)

    monkeypatch.setattr("jobrag.llm.ollama.httpx.post", fake_post)
    llm = OllamaLLM("qwen2.5:3b-instruct", keep_alive="10m")
    completion = llm.complete("la question", system="les règles", max_tokens=200)

    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "les règles"},
        {"role": "user", "content": "la question"},
    ]
    # Reproducibility: a RAG answer must not change between identical runs, or
    # the faithfulness measurement is measuring sampling noise.
    assert captured["json"]["options"]["temperature"] == 0.0
    assert captured["json"]["options"]["num_predict"] == 200
    assert captured["json"]["stream"] is False
    assert captured["json"]["keep_alive"] == "10m"

    assert isinstance(completion, Completion)
    assert completion.text == "Ingénieur DevOps chez ACME [1]."  # stripped
    assert (completion.prompt_tokens, completion.completion_tokens) == (877, 74)
    assert completion.latency_ms >= 0
    assert completion.raw["done_reason"] == "stop"


def test_ollama_without_a_system_prompt_sends_one_message(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "jobrag.llm.ollama.httpx.post",
        lambda url, *, json, timeout: captured.update(json=json) or FakeResponse(OLLAMA_BODY),
    )
    OllamaLLM("m").complete("juste la question")
    assert [message["role"] for message in captured["json"]["messages"]] == ["user"]


def test_ollama_turns_a_network_error_into_llmerror(monkeypatch):
    def fake_post(url, *, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("jobrag.llm.ollama.httpx.post", fake_post)
    with pytest.raises(LLMError, match="unreachable"):
        OllamaLLM("m").complete("question")


def test_ollama_rejects_an_empty_answer(monkeypatch):
    monkeypatch.setattr(
        "jobrag.llm.ollama.httpx.post",
        lambda url, *, json, timeout: FakeResponse({"message": {"content": ""}}),
    )
    with pytest.raises(LLMError, match="empty"):
        OllamaLLM("m").complete("question")


def test_ollama_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "jobrag.llm.ollama.httpx.post",
        lambda url, *, json, timeout: FakeResponse(ValueError("not json")),
    )
    with pytest.raises(LLMError, match="invalid JSON"):
        OllamaLLM("m").complete("question")


def test_openai_compat_requires_a_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(LLMError, match="LLM_BASE_URL"):
        OpenAICompatLLM("gpt-x")


def test_openai_compat_requires_a_key_but_never_logs_it(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    llm = OpenAICompatLLM("gpt-x", base_url="https://api.example.com/v1")
    with pytest.raises(LLMError) as error:
        llm.complete("question")
    assert "LLM_API_KEY" in str(error.value)
    assert "secret" not in str(error.value)


def test_openai_compat_sends_a_bearer_token(monkeypatch):
    captured = {}
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")

    def fake_post(url, *, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers)
        return FakeResponse(
            {
                "model": "gpt-x",
                "choices": [{"message": {"content": "une réponse [1]"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )

    monkeypatch.setattr("jobrag.llm.openai_compat.httpx.post", fake_post)
    completion = OpenAICompatLLM("gpt-x", base_url="https://api.example.com/v1").complete("q")
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-secret-value"
    assert completion.text == "une réponse [1]"
    assert completion.prompt_tokens == 100


def test_openai_compat_error_message_carries_no_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")

    def fake_post(url, *, json, headers, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("jobrag.llm.openai_compat.httpx.post", fake_post)
    with pytest.raises(LLMError) as error:
        OpenAICompatLLM("gpt-x", base_url="https://api.example.com/v1").complete("q")
    assert "sk-secret-value" not in str(error.value)
