"""
InsightEngine AI - API Schemas
Pydantic models for request validation and response serialization.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional
from datetime import datetime
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class DatasetStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class ResponseTone(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


# ─── Upload ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned after successful file upload + processing."""
    dataset_id: str
    original_filename: str
    status: DatasetStatus
    row_count: int
    column_count: int
    columns: list[str]
    domain: str
    quality_score: float
    quality_summary: str
    cleaning_summary: Optional[str] = None
    has_embeddings: bool
    message: str = "Your dataset is ready for analysis!"

    class Config:
        use_enum_values = True


class UploadErrorResponse(BaseModel):
    """Returned when upload fails validation."""
    error: str
    detail: Optional[str] = None


# ─── Dataset Info ─────────────────────────────────────────────────────────────

class DatasetInfoResponse(BaseModel):
    """Full metadata for an existing dataset."""
    dataset_id: str
    original_filename: str
    status: DatasetStatus
    row_count: Optional[int]
    column_count: Optional[int]
    columns: Optional[list[str]]
    dtypes: Optional[dict]
    domain: Optional[str]
    domain_confidence: Optional[float]
    quality_score: Optional[float]
    is_cleaned: bool
    has_embeddings: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    likely_questions: Optional[list[str]] = None

    class Config:
        use_enum_values = True


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfoResponse]
    total: int


# ─── Query / Chat ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """User sends a natural language query about their dataset."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about the dataset",
        examples=["What are the top 5 products by sales?"],
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation thread ID for multi-turn context",
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        return v.strip()


class ChartConfig(BaseModel):
    """Chart configuration for frontend rendering."""
    type: str          # bar, line, pie, scatter
    title: str
    labels: Optional[list] = None
    datasets: Optional[list] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None


class QueryResponse(BaseModel):
    """Full response to a user's data question."""
    success: bool
    story: str                              # Human-friendly explanation (main output)
    tone: ResponseTone = ResponseTone.INFO
    result: Optional[Any] = None           # Raw result (table, number, list)
    result_summary: Optional[str] = None   # One-line key metric
    output_type: Optional[str] = None      # dataframe | number | text | list | dict
    chart_config: Optional[dict] = None    # Chart.js-compatible config
    execution_time_ms: Optional[int] = None
    history_id: Optional[str] = None       # Saved to DB
    debug: Optional[dict] = None           # Only in DEBUG mode

    class Config:
        use_enum_values = True


# ─── History ─────────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    id: str
    user_query: str
    story_response: Optional[str]
    execution_success: bool
    created_at: Optional[datetime]


class HistoryResponse(BaseModel):
    dataset_id: str
    items: list[HistoryItem]
    total: int


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    app_name: str
    groq_configured: bool
    openrouter_configured: bool


# ─── Error ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[int] = None