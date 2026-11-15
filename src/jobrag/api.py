"""FastAPI application.

/health (phase 0), /search (phase 2b: retrieval only, for debugging).
/ask (answer + citations) arrives in phase 6.
"""

from functools import lru_cache

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .db import REQUIRED_EXTENSIONS, check_database
from .embed import get_embedder
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
    ef_search: int = Query(80, ge=10, le=500, description="HNSW accuracy/speed knob"),
) -> JSONResponse:
    """Vectors-only retrieval, for debugging and for the phase-5 measurements.

    Returns the chunks themselves with their scores and timings: this endpoint
    exists to make retrieval inspectable, not to answer questions.
    """
    try:
        payload = search_chunks(q, _embedder(), k=k, ef_search=ef_search)
    except ImportError as exc:  # sentence-transformers / torch not installed
        return JSONResponse(
            {"error": "embedding backend unavailable", "detail": str(exc)},
            status_code=503,
        )
    return JSONResponse(payload)
