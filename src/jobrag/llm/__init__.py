"""Generation backends, behind one Protocol."""

from .base import LLM, Completion, LLMError
from .ollama import OllamaLLM
from .openai_compat import OpenAICompatLLM

__all__ = ["LLM", "Completion", "LLMError", "OllamaLLM", "OpenAICompatLLM"]
