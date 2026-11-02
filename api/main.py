"""
CleanCut API

Proje kök dizininden çalıştırın:
    uvicorn api.main:app --reload --no-access-log

Dokümantasyon: http://127.0.0.1:8000/docs
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from core.models import CutMode
from api.config import (
    DEFAULT_COMPUTE_TYPE, DEFAULT_CUT_MODE, DEFAULT_DEVICE, DEFAULT_LANGUAGE, DEFAULT_MODEL,
    ensure_dirs, genius_token, load_system_prompt,
)
from api.routers import files, pipeline, steps
from api.schemas import OptionsOut

logger = logging.getLogger("cleancut")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    yield


app = FastAPI(
    title="CleanCut API",
    description="Şarkılardaki uygunsuz içeriği otomatik tespit edip kesen servis.",
    version="0.2.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Her isteğin başlangıcını ve süresini loglar; uzun işler konsoldan izlenebilir."""
    started = time.perf_counter()
    logger.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("← %s %s %s (%.1fs)", request.method, request.url.path,
                response.status_code, time.perf_counter() - started)
    return response


app.include_router(files.songs)
app.include_router(files.outputs)
app.include_router(steps.router)
app.include_router(pipeline.router)


@app.get("/health", tags=["Sistem"])
def health():
    """Sunucunun ayakta olduğunu ve gerekli ayarların yapıldığını kontrol eder."""
    return {
        "status": "ok",
        "genius_token": genius_token() is not None,
        "system_prompt": load_system_prompt() is not None,
    }


@app.get("/options", response_model=OptionsOut, tags=["Sistem"])
def options():
    """Arayüzün seçenek listeleri ve varsayılan değerler (ör. kesim modu seçimi)."""
    return {
        "cut_modes": [
            {"value": m, "label": m.label, "description": m.description, "beats": m.beats}
            for m in CutMode
        ],
        "defaults": {
            "cut_mode": DEFAULT_CUT_MODE,
            "language_code": DEFAULT_LANGUAGE,
            "device": DEFAULT_DEVICE,
            "compute_type": DEFAULT_COMPUTE_TYPE,
            "model_name": DEFAULT_MODEL,
        },
    }
