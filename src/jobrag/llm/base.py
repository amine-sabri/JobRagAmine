"""The generation interface.

One small Protocol, for one reason: the brief needs a local model now and an
API model later, compared on the same questions. Anything that can turn a
prompt into text behind `complete()` can be measured without touching the
prompt, the retrieval or the evaluation.

Temperature defaults to 0. A RAG answer has to be reproducible: the same
question over the same sources must give the same answer, or the faithfulness
measurement of phase 7 is measuring sampling noise.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Completion:
    """What every backend returns, so the caller never branches on the model."""

    text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """The backend could not answer: unreachable, timed out, refused."""


@runtime_checkable
class LLM(Protocol):
    """A text completion backend."""

    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> Completion: ...
