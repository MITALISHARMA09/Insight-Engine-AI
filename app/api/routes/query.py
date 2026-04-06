"""
InsightEngine AI - API Route: /datasets/{id}/query
Handles natural language queries against uploaded datasets.
"""
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import get_db, DatasetMeta, UserHistory
from app.engine.orchestrator import orchestrator
from app.api.schemas import QueryRequest, QueryResponse, HistoryResponse, HistoryItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Query"])


@router.post(
    "/{dataset_id}/query",
    response_model=QueryResponse,
    summary="Ask a question about your dataset",
    description="Send a natural language question. AI will analyze the data and explain the answer.",
)
async def query_dataset(
    dataset_id: str = Path(..., description="Dataset ID from upload response"),
    request: QueryRequest = ...,
    db: AsyncSession = Depends(get_db),
):
    """
    Full query pipeline:
    RAG → CoderA‖CoderB → Judge → Sandbox → Fix? → Storyteller
    """
    # ── Verify dataset exists and is ready ───────────────────────────────────
    result = await db.execute(
        select(DatasetMeta).where(DatasetMeta.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Please upload your file first.",
        )

    if dataset.status != "ready":
        if dataset.status == "processing":
            raise HTTPException(
                status_code=202,
                detail="Your dataset is still being processed. Please wait a moment and try again.",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Dataset is not ready for queries (status: {dataset.status}).",
        )

    # ── Run query pipeline ────────────────────────────────────────────────────
    try:
        pipeline_response = await orchestrator.process_query(
            user_query=request.question,
            dataset_id=dataset_id,
        )
    except Exception as e:
        logger.error(f"Query pipeline error for {dataset_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while analyzing your question. Please try again.",
        )

    # ── Save to history ───────────────────────────────────────────────────────
    history_id = str(uuid.uuid4())
    try:
        history_record = UserHistory(
            id=history_id,
            dataset_id=dataset_id,
            user_query=request.question,
            generated_code=pipeline_response.get("debug", {}).get("code_used"),
            code_output_json=pipeline_response.get("result"),
            story_response=pipeline_response.get("story"),
            rag_chunks_used=pipeline_response.get("debug", {}).get("rag_chunks_count"),
            execution_success=pipeline_response.get("success", False),
            execution_time_ms=pipeline_response.get("execution_time_ms"),
        )
        db.add(history_record)
        await db.flush()
    except Exception as e:
        logger.warning(f"Failed to save query history: {e}")

    # ── Return response ───────────────────────────────────────────────────────
    return QueryResponse(
        success=pipeline_response.get("success", False),
        story=pipeline_response.get("story", ""),
        tone=pipeline_response.get("tone", "info"),
        result=pipeline_response.get("result"),
        result_summary=pipeline_response.get("result_summary"),
        output_type=pipeline_response.get("output_type"),
        chart_config=pipeline_response.get("chart_config"),
        execution_time_ms=pipeline_response.get("execution_time_ms"),
        history_id=history_id,
        debug=pipeline_response.get("debug"),
    )


@router.get(
    "/{dataset_id}/history",
    response_model=HistoryResponse,
    summary="Get query history for a dataset",
)
async def get_history(
    dataset_id: str = Path(...),
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Returns the last N queries and responses for a dataset."""
    result = await db.execute(
        select(UserHistory)
        .where(UserHistory.dataset_id == dataset_id)
        .order_by(UserHistory.created_at.desc())
        .limit(limit)
    )
    records = result.scalars().all()

    items = [
        HistoryItem(
            id=r.id,
            user_query=r.user_query,
            story_response=r.story_response,
            execution_success=r.execution_success,
            created_at=r.created_at,
        )
        for r in records
    ]

    return HistoryResponse(
        dataset_id=dataset_id,
        items=items,
        total=len(items),
    )