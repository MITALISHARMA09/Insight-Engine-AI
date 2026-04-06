"""
InsightEngine AI — Frontend Integration Patch
Apply this to main.py to serve the frontend from FastAPI.

Changes:
  1. Mount /frontend as static files at /app
  2. Serve index.html at root /
  3. CORS updated to allow frontend origin
"""

# ─── In main.py, add these imports ───────────────────────────────────────────
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# import os

# ─── After app = FastAPI(...) add ────────────────────────────────────────────
# FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
# if os.path.exists(FRONTEND_DIR):
#     app.mount("/app", StaticFiles(directory=FRONTEND_DIR), name="frontend")
#
# @app.get("/", include_in_schema=False)
# async def serve_frontend():
#     index = os.path.join(FRONTEND_DIR, "index.html")
#     if os.path.exists(index):
#         return FileResponse(index)
#     return {"app": "InsightEngine AI", "docs": "/docs"}

# ─── Full updated main.py (complete replacement) ──────────────────────────────
FULL_MAIN = '''
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.db.models import init_db
from app.api.routes import upload, query, datasets, health

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    os.makedirs(settings.DATASETS_BASE_PATH, exist_ok=True)
    await init_db()
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key":
        logger.warning("GROQ_API_KEY not configured")
    if not settings.OPENROUTER_API_KEY or settings.OPENROUTER_API_KEY == "your_openrouter_api_key":
        logger.warning("OPENROUTER_API_KEY not configured")
    logger.info("InsightEngine AI ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    messages = [f"{'.'.join(str(l) for l in e.get('loc',[]))}: {e.get('msg')}" for e in errors]
    return JSONResponse(status_code=422, content={"error": "Invalid request", "detail": "; ".join(messages)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={
        "error": "Something went wrong. Please try again.",
        "detail": str(exc) if settings.DEBUG else None,
    })


app.include_router(health.router)
app.include_router(upload.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")

# ─── Serve frontend static files ─────────────────────────────────────────────
if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index = os.path.join(FRONTEND_DIR, "index.html")
        return FileResponse(index) if os.path.exists(index) else JSONResponse(
            {"app": settings.APP_NAME, "docs": "/docs"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
'''