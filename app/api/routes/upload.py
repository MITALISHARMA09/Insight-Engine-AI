from __future__ import annotations
"""
InsightEngine AI - API Route: /upload
Handles dataset file upload, validation, and pipeline trigger.
"""
import uuid
import logging
import io
import os
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from app.core.config import settings
from app.db.models import get_db, DatasetMeta
from app.engine.orchestrator import orchestrator
from app.api.schemas import UploadResponse

logger = logging.getLogger(__name__)



def _sanitize_for_json(obj):
    """
    Recursively convert any non-JSON-serializable pandas/numpy types to strings.
    Handles: Timestamp, NaT, numpy int/float/bool, NaN, Timedelta.
    """
    import math
    import pandas as pd
    import numpy as np

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if not pd.isnull(obj) else None
    if isinstance(obj, pd.NaT.__class__):
        return None
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    return obj

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post(
    "/",
    response_model=UploadResponse,
    summary="Upload a dataset",
    description="Upload a CSV or XLSX file. The system will automatically clean, analyze, and index it.",
)
async def upload_dataset(
    file: Annotated[UploadFile, File(description="CSV or XLSX file (max 50MB)")],
    db: AsyncSession = Depends(get_db),
):
    """
    Full upload flow:
    1. Validate file type and size
    2. Parse into DataFrame
    3. Validate row/column limits
    4. Trigger full processing pipeline (async)
    5. Return metadata + processing results
    """
    # ── Validate file extension ───────────────────────────────────────────────
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload a CSV or XLSX file.",
        )

    # ── Read raw bytes ────────────────────────────────────────────────────────
    content = await file.read()
    file_size = len(content)

    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large ({file_size // 1024 // 1024}MB). Maximum allowed is {settings.MAX_FILE_SIZE_MB}MB.",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # ── Parse into DataFrame ──────────────────────────────────────────────────
    try:
        df = _parse_file(content, ext, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── Validate dimensions ───────────────────────────────────────────────────
    if len(df) == 0:
        raise HTTPException(status_code=422, detail="The file has no data rows.")

    if len(df) > settings.MAX_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset has {len(df):,} rows. Maximum allowed is {settings.MAX_ROWS:,} rows.",
        )

    if len(df.columns) > settings.MAX_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset has {len(df.columns)} columns. Maximum allowed is {settings.MAX_COLUMNS}.",
        )

    if len(df.columns) < 2:
        raise HTTPException(
            status_code=422,
            detail="Dataset must have at least 2 columns to be useful for analysis.",
        )

    # ── Generate dataset ID ───────────────────────────────────────────────────
    dataset_id = str(uuid.uuid4())
    logger.info(f"New upload: dataset_id={dataset_id}, file={filename}, rows={len(df)}, cols={len(df.columns)}")

    # ── Save initial DB record ────────────────────────────────────────────────
    db_record = DatasetMeta(
        id=dataset_id,
        original_filename=filename,
        file_size_bytes=file_size,
        row_count=len(df),
        column_count=len(df.columns),
        columns_json=list(df.columns),
        dtypes_json={col: str(df[col].dtype) for col in df.columns},  # already strings
        sample_rows_json=_sanitize_for_json(df.head(5).to_dict(orient="records")),
        status="processing",
    )
    db.add(db_record)
    await db.flush()

    # ── Run full pipeline ─────────────────────────────────────────────────────
    try:
        pipeline_result = await orchestrator.process_upload(
            df_raw=df,
            dataset_id=dataset_id,
            original_filename=filename,
        )
    except Exception as e:
        logger.error(f"Pipeline failed for {dataset_id}: {e}")
        db_record.status = "error"
        db_record.error_message = str(e)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your file. Please try again.",
        )

    # ── Update DB with results ────────────────────────────────────────────────
    db_record.status = "ready"
    db_record.domain = pipeline_result.get("domain", "general")
    db_record.domain_confidence = (
        pipeline_result.get("domain_details", {}).get("confidence", 0.5)
    )
    db_record.is_cleaned = pipeline_result.get("is_cleaned", False)
    db_record.has_embeddings = pipeline_result.get("has_embeddings", False)
    db_record.profiling_report_json = pipeline_result.get("profiling_report", {})
    db_record.cleaning_report_json = pipeline_result.get("cleaning_report", {})

    # ── Build response ────────────────────────────────────────────────────────
    domain_details = pipeline_result.get("domain_details", {})
    likely_questions = domain_details.get("likely_questions", [])

    message_parts = ["✅ Your dataset is ready for analysis!"]
    if likely_questions:
        message_parts.append(f"Try asking: \"{likely_questions[0]}\"")

    return UploadResponse(
        dataset_id=dataset_id,
        original_filename=filename,
        status="ready",
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        domain=pipeline_result.get("domain", "general"),
        quality_score=pipeline_result.get("quality_score", 100.0),
        quality_summary=pipeline_result.get("quality_summary", ""),
        cleaning_summary=pipeline_result.get("cleaning_summary"),
        has_embeddings=pipeline_result.get("has_embeddings", False),
        message=" ".join(message_parts),
    )


def _parse_file(content: bytes, ext: str, filename: str) -> pd.DataFrame:
    """Parse uploaded file content into a DataFrame."""
    try:
        if ext == ".csv":
            # Try multiple encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not read CSV file. Please ensure it is UTF-8 or Latin-1 encoded.")

        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl" if ext == ".xlsx" else None)
            return df

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not parse '{filename}': {str(e)}")