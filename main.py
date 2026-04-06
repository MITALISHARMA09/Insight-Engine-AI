from __future__ import annotations
"""
InsightEngine AI - Main Application
"""
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.db.models import init_db
from app.api.routes import upload, query, datasets, health

# ─── Resolve paths absolutely (works on Windows when cwd == project dir) ──────
BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE   = FRONTEND_DIR / "index.html"

# ─── Read index.html at import time — no startup event needed ─────────────────
# This is the most reliable approach: runs once when Python loads the module.
if INDEX_FILE.exists():
    _INDEX_HTML: str | None = INDEX_FILE.read_text(encoding="utf-8")
    print(f"[InsightEngine] Frontend loaded: {INDEX_FILE}")
else:
    _INDEX_HTML = None
    print(f"[InsightEngine] WARNING: {INDEX_FILE} not found — UI will not load")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.DATASETS_BASE_PATH, exist_ok=True)
    await init_db()
    logger.info("Database ready")
    if _INDEX_HTML:
        logger.info("UI ready → http://localhost:8000/")
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key":
        logger.warning("GROQ_API_KEY not configured")
    if not settings.OPENROUTER_API_KEY or settings.OPENROUTER_API_KEY == "your_openrouter_api_key":
        logger.warning("OPENROUTER_API_KEY not configured")
    yield


# ─── App ──────────────────────────────────────────────────────────────────────
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


# ─── Error handlers ───────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    msgs = [f"{'.'.join(str(l) for l in e.get('loc',[]))}: {e.get('msg')}" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Invalid request", "detail": "; ".join(msgs)})

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"{request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Something went wrong.", "detail": str(exc) if settings.DEBUG else None})


# ─── API routes ───────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(upload.router,   prefix="/api/v1")
app.include_router(query.router,    prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")


# ─── Frontend route ───────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui():
    if _INDEX_HTML:
        return HTMLResponse(content=_INDEX_HTML, status_code=200)
    # Helpful fallback if index.html is missing
    return HTMLResponse(content=f"""<!DOCTYPE html><html><head>
<title>InsightEngine AI</title>
<style>body{{font-family:sans-serif;background:#09090F;color:#E8E4DC;display:flex;
align-items:center;justify-content:center;min-height:100vh;margin:0}}
.b{{border:1px solid #2A2A3A;border-radius:8px;padding:40px;max-width:480px;text-align:center}}
h2{{color:#00C896}}code{{background:#1C1C28;padding:2px 8px;border-radius:4px;color:#00C896}}
a{{color:#00C896}}</style></head><body><div class='b'>
<h2>InsightEngine AI is running</h2>
<p>Backend OK — but <code>frontend/index.html</code> was not found.</p>
<p>Expected location:<br><code>{INDEX_FILE}</code></p>
<p><a href='/docs'>Open API docs →</a></p>
</div></body></html>""", status_code=200)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)