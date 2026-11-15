"""Embedding backend.

The prefix rule is the part worth being careful about: e5 models are trained
asymmetrically and expect `query: ` on questions and `passage: ` on documents.
Omitting them raises no error and costs real recall — one of the most common
silent RAG bugs — so the rule is a tested pure function, independent of the
model actually loading.
"""

from collections.abc import Sequence
from typing import Any, Literal, Protocol

Kind = Literal["query", "passage"]

# Per model family, the prefix to prepend for queries and for passages.
PREFIX_RULES: list[tuple[str, dict[str, str]]] = [
    ("e5", {"query": "query: ", "passage": "passage: "}),
    ("bge-m3", {"query": "", "passage": ""}),
    ("bge-reranker", {"query": "", "passage": ""}),
    # Legacy English bge models want an instruction on the query side only.
    (
        "bge-large-en",
        {"query": "Represent this sentence for searching relevant passages: ", "passage": ""},
    ),
]


def prefix_for(model_name: str, kind: Kind) -> str:
    """The prefix this model expects for `kind`. Empty string when none."""
    lowered = model_name.lower()
    for marker, prefixes in PREFIX_RULES:
        if marker in lowered:
            return prefixes[kind]
    return ""


def apply_prefix(texts: Sequence[str], model_name: str, kind: Kind) -> list[str]:
    """Prepend the model's prefix to every text, once."""
    prefix = prefix_for(model_name, kind)
    if not prefix:
        return list(texts)
    return [text if text.startswith(prefix) else f"{prefix}{text}" for text in texts]


class Embedder(Protocol):
    """What the retrieval code needs: two encode methods and a dimension."""

    dimension: int

    def encode_passages(self, texts: Sequence[str], **kwargs: Any) -> Any: ...

    def encode_queries(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


class SentenceTransformerEmbedder:
    """sentence-transformers backend, CPU by default (no GPU on this machine).

    The model is loaded lazily so importing this module stays cheap: the API and
    the tests must not pay a 1.1 GB model load to answer /health.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.threads = threads
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.threads:
                torch.set_num_threads(self.threads)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self.model  # noqa: B018 - triggers the lazy load
        assert self._dimension is not None
        return self._dimension

    def _encode(self, texts: Sequence[str], kind: Kind, **kwargs: Any) -> Any:
        prefixed = apply_prefix(texts, self.model_name, kind)
        return self.model.encode(
            prefixed,
            batch_size=kwargs.pop("batch_size", self.batch_size),
            # e5 similarity is cosine, so the vectors are L2-normalised here and
            # the database can use the cheaper inner product ordering.
            normalize_embeddings=True,
            convert_to_numpy=True,
            **kwargs,
        )

    def encode_passages(self, texts: Sequence[str], **kwargs: Any) -> Any:
        return self._encode(texts, "passage", **kwargs)

    def encode_queries(self, texts: Sequence[str], **kwargs: Any) -> Any:
        return self._encode(texts, "query", **kwargs)


def get_embedder(model_name: str | None = None, **kwargs: Any) -> SentenceTransformerEmbedder:
    """The configured embedder (EMBEDDING_MODEL in .env)."""
    from .config import get_settings

    return SentenceTransformerEmbedder(model_name or get_settings().embedding_model, **kwargs)
