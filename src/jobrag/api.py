"""FastAPI application.

Phase 0 exposes /health only. /search (retrieval, for debugging) arrives in
phase 3 and /ask (answer + citations) in phase 6.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .db import REQUIRED_EXTENSIONS, check_database

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
