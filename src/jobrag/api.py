"""FastAPI application.

/health (phase 0), /search (retrieval only, for debugging and for the phase-5
measurements), /ask (an answer with citations, or a refusal).
"""

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .db import REQUIRED_EXTENSIONS, check_database
from .embed import get_embedder
from .generate import answer_question
from .llm import LLMError, OllamaLLM
from .retrieve.fusion import fuse_hits
from .retrieve.keyword import search_chunks as keyword_search_chunks
from .retrieve.vector import search_chunks

app = FastAPI(title="JobRAG", version=__version__)


@app.get("/health")
def health() -> JSONResponse:
    """Liveness + readiness: is the database reachable and correctly set up?

    Returns 503 when it is not, so `make up`, CI and a container healthcheck can
    all rely on the status code instead of parsing the body.
    """
    database = check_database()
    ready = database["reachable"] and all(
        database["extensions"].get(name, False) for name in REQUIRED_EXTENSIONS
    )
    payload = {
        "status": "ok" if ready else "degraded",
        "version": __version__,
        "database": database,
        "config": get_settings().public_dict(),
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@lru_cache(maxsize=1)
def _embedder():
    """The embedding model, loaded once, on the first /search — not at startup.

    A 1.1 GB model load must not delay /health or block the container's
    readiness check.
    """
    return get_embedder(threads=4)


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="the question, in French or English"),
    k: int = Query(5, ge=1, le=50),
    mode: Literal["vector", "keyword", "hybrid"] = Query("vector"),
    ef_search: int = Query(80, ge=10, le=500, description="HNSW accuracy/speed knob"),
    rrf_k: int = Query(60, ge=1, le=1000, description="RRF damping constant"),
) -> JSONResponse:
    """Retrieval only, for debugging and for the phase-5 measurements.

    Returns the chunks themselves with their scores and timings: this endpoint
    exists to make retrieval inspectable, not to answer questions. `mode`
    selects the retriever being measured — vectors alone (the baseline),
    keywords alone, or the two fused with Reciprocal Rank Fusion.
    """
    try:
        if mode == "keyword":
            return JSONResponse(keyword_search_chunks(q, k=k))
        if mode == "vector":
            return JSONResponse(search_chunks(q, _embedder(), k=k, ef_search=ef_search))

        # Hybrid: each retriever goes deeper than k, because fusion only helps
        # if it can promote something the other list ranked low.
        depth = max(k * 4, 20)
        vector_run = search_chunks(q, _embedder(), k=depth, ef_search=ef_search)
        keyword_run = keyword_search_chunks(q, k=depth)
        fused = fuse_hits([vector_run["results"], keyword_run["results"]], k=rrf_k, limit=k)
        return JSONResponse(
            {
                "query": q,
                "k": k,
                "mode": mode,
                "rrf_k": rrf_k,
                "depth": depth,
                "results": fused,
                "timing_ms": {
                    "vector": vector_run["timing_ms"]["total"],
                    "keyword": keyword_run["timing_ms"]["total"],
                    "total": round(
                        vector_run["timing_ms"]["total"] + keyword_run["timing_ms"]["total"], 1
                    ),
                },
            }
        )
    except ImportError as exc:  # sentence-transformers / torch not installed
        return JSONResponse(
            {"error": "embedding backend unavailable", "detail": str(exc)},
            status_code=503,
        )


@lru_cache(maxsize=1)
def _llm() -> OllamaLLM:
    """The generation backend, built once from .env."""
    settings = get_settings()
    return OllamaLLM(settings.ollama_model, base_url=settings.ollama_base_url)


@app.get("/ask")
def ask(
    q: str = Query(..., min_length=3, description="the question, in French or English"),
    k: int = Query(3, ge=1, le=10, description="offers put in the prompt"),
    mode: Literal["vector", "keyword", "hybrid"] = Query("hybrid"),
    filters: bool = Query(True, description="extract city/contract/date filters from the question"),
    relax_filters: bool = Query(
        False,
        description="search without the filters when they match nothing "
        "(measured to cause fabricated answers)",
    ),
) -> JSONResponse:
    """Answer `q` from the indexed offers, with citations, or refuse.

    Slow by nature here: generation runs on CPU at a few tokens per second, so
    a question costs roughly two minutes. The timings are in the response.
    """
    try:
        embedder = _embedder() if mode in {"vector", "hybrid"} else None
        payload = answer_question(
            q,
            llm=_llm(),
            embedder=embedder,
            k=k,
            mode=mode,
            use_filters=filters,
            relax_filters=relax_filters,
        )
    except ImportError as exc:  # sentence-transformers / torch not installed
        return JSONResponse(
            {"error": "embedding backend unavailable", "detail": str(exc)}, status_code=503
        )
    except LLMError as exc:
        return JSONResponse({"error": "generation backend unavailable", "detail": str(exc)}, 503)
    if payload.get("error"):
        return JSONResponse(payload, status_code=503)
    return JSONResponse(payload)
