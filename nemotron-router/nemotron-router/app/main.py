import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nemotron-router-main")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model discovery on startup (blocking call -> run in a worker thread)
    logger.info("Initializing Nemotron Router model discovery...")
    await run_in_threadpool(router.discover_models)
    logger.info(
        f"Active Model Configuration: "
        f"Small={router.tier_models['small']} | "
        f"Mid={router.tier_models['mid']} | "
        f"Large={router.tier_models['large']}"
    )
    yield


app = FastAPI(
    title="Nemotron Router",
    description="Cost-Optimized Intelligent LLM Router for NVIDIA Nemotron Models via Nebius Token Factory",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000, description="The user prompt to route and answer")


@app.get("/")
async def get_dashboard():
    """Serve the single-page dashboard HTML."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(index_file)


@app.post("/api/ask")
async def ask_endpoint(payload: AskRequest) -> Dict[str, Any]:
    """
    Route prompt to cheapest capable Nemotron model, execute request,
    calculate cost savings vs large baseline, and clean any think tags.
    """
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        # router.ask is blocking (network I/O); keep it off the event loop.
        return await run_in_threadpool(router.ask, prompt)
    except Exception as e:
        logger.error(f"Routing request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def stats_endpoint() -> Dict[str, Any]:
    """Return aggregated routing statistics, cost savings, tier distribution, and last 20 requests."""
    return router.get_stats()


@app.get("/api/health")
async def health_endpoint() -> Dict[str, Any]:
    """Return setup status, API connectivity, model discovery state, and pricing flags."""
    return {
        "status": "ok" if (router.models_discovered_ok or settings.model_small) else "warning",
        "has_api_key": settings.has_api_key,
        "base_url": settings.base_url,
        "models_discovered_count": len(router.discovered_models),
        "selected_models": router.tier_models,
        "is_placeholder_pricing": settings.prices_are_placeholder,
        "discovery_error": router.discovery_error,
    }


@app.get("/api/models")
async def models_endpoint() -> Dict[str, Any]:
    """List discovered NVIDIA Nemotron models and active tier assignments."""
    return {
        "selected": router.tier_models,
        "discovered_models": router.discovered_models,
        "overrides": {
            "small": settings.model_small,
            "mid": settings.model_mid,
            "large": settings.model_large,
        },
    }
