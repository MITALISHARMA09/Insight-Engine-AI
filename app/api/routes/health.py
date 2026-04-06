"""
InsightEngine AI - API Route: /health
System status and configuration check.
"""
from fastapi import APIRouter
from app.core.config import settings
from app.api.schemas import HealthResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check():
    """Returns system status and configuration health."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        app_name=settings.APP_NAME,
        groq_configured=bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_groq_api_key"),
        openrouter_configured=bool(settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY != "your_openrouter_api_key"),
    )