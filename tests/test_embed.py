"""The prefix rule, tested without loading a 1.1 GB model."""

import pytest

from jobrag.embed import SentenceTransformerEmbedder, apply_prefix, prefix_for


@pytest.mark.parametrize(
    ("model", "kind", "expected"),
    [
        ("intfloat/multilingual-e5-base", "query", "query: "),
        ("intfloat/multilingual-e5-base", "passage", "passage: "),
        ("intfloat/multilingual-e5-small", "query", "query: "),
        ("BAAI/bge-m3", "query", ""),
        ("BAAI/bge-m3", "passage", ""),
        ("BAAI/bge-reranker-base", "query", ""),
        ("some/unknown-model", "query", ""),
    ],
)
def test_prefix_rules(model, kind, expected):
    assert prefix_for(model, kind) == expected


def test_queries_and_passages_get_different_prefixes():
    # The asymmetry is the whole point: mixing them up costs recall silently.
    model = "intfloat/multilingual-e5-base"
    assert apply_prefix(["à Lille ?"], model, "query") == ["query: à Lille ?"]
    assert apply_prefix(["Dev · ACME"], model, "passage") == ["passage: Dev · ACME"]


def test_prefixing_is_idempotent():
    model = "intfloat/multilingual-e5-base"
    once = apply_prefix(["Kubernetes"], model, "query")
    assert apply_prefix(once, model, "query") == once


def test_a_model_without_prefixes_is_left_alone():
    assert apply_prefix(["texte"], "BAAI/bge-m3", "passage") == ["texte"]


def test_the_model_is_not_loaded_on_construction(monkeypatch):
    # Importing or constructing must stay free: /health must not pay a model load.
    def explode(*args, **kwargs):
        raise AssertionError("the model was loaded eagerly")

    monkeypatch.setattr(
        "jobrag.embed.SentenceTransformerEmbedder.model",
        property(explode),
    )
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-base")
    assert embedder.model_name == "intfloat/multilingual-e5-base"
    assert embedder.batch_size == 16


def test_encode_applies_the_prefix_and_normalises(monkeypatch):
    captured = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured["texts"] = texts
            captured["kwargs"] = kwargs
            return [[0.0] * 768 for _ in texts]

        def get_sentence_embedding_dimension(self):
            return 768

    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-base")
    embedder._model = FakeModel()
    embedder._dimension = 768

    embedder.encode_queries(["combien de CDI à Lille ?"])
    assert captured["texts"] == ["query: combien de CDI à Lille ?"]
    assert captured["kwargs"]["normalize_embeddings"] is True

    embedder.encode_passages(["Dev Python · ACME · Lille"], batch_size=4)
    assert captured["texts"] == ["passage: Dev Python · ACME · Lille"]
    assert captured["kwargs"]["batch_size"] == 4
