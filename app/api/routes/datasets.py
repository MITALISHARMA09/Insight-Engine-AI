from __future__ import annotations
"""
InsightEngine AI - API Route: /datasets
Dataset listing, info, preview, download, and deletion.
"""
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
import pandas as pd
import io

from app.db.models import get_db, DatasetMeta, UserHistory
from app.core.config import settings
from app.api.schemas import DatasetInfoResponse, DatasetListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("/", response_model=DatasetListResponse, summary="List all datasets")
async def list_datasets(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DatasetMeta).order_by(DatasetMeta.created_at.desc()).limit(limit)
    )
    datasets = result.scalars().all()
    items = [_to_response(d) for d in datasets]
    return DatasetListResponse(datasets=items, total=len(items))


@router.get("/{dataset_id}", response_model=DatasetInfoResponse, summary="Get dataset details")
async def get_dataset(dataset_id: str = Path(...), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DatasetMeta).where(DatasetMeta.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return _to_response(dataset)


@router.get("/{dataset_id}/preview", summary="Preview dataset rows")
async def preview_dataset(
    dataset_id: str = Path(...),
    rows: int = 10,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(DatasetMeta).where(DatasetMeta.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    csv_path = _resolve_csv(dataset_id)
    if not csv_path:
        raise HTTPException(status_code=404, detail="Dataset file not found on disk.")

    df = pd.read_csv(csv_path, nrows=min(rows, 100))
    return {
        "dataset_id": dataset_id,
        "rows_shown": len(df),
        "total_rows": dataset.row_count,
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
    }


@router.get(
    "/{dataset_id}/download",
    summary="Download cleaned dataset",
    description="Download the cleaned CSV. Pass ?format=xlsx to get an Excel file instead.",
    response_class=StreamingResponse,
)
async def download_dataset(
    dataset_id: str = Path(...),
    format: str = Query(default="csv", pattern="^(csv|xlsx)$"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(DatasetMeta).where(DatasetMeta.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    csv_path = _resolve_csv(dataset_id)
    if not csv_path:
        raise HTTPException(status_code=404, detail="Dataset file not found on disk.")

    df = pd.read_csv(csv_path)

    # Build a clean filename from the original, e.g. "sales_cleaned.csv"
    base = os.path.splitext(dataset.original_filename or "dataset")[0]
    is_cleaned = "cleaned" in csv_path  # True if cleaned.csv was found

    if format == "xlsx":
        filename = f"{base}_cleaned.xlsx" if is_cleaned else f"{base}.xlsx"
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    else:
        filename = f"{base}_cleaned.csv" if is_cleaned else f"{base}.csv"
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )




@router.get("/{dataset_id}/dashboard", summary="Get auto-generated dashboard")
async def get_dashboard(
    dataset_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns pre-computed dashboard tiles for a dataset.
    Generated automatically after upload — no extra cost.
    """
    result = await db.execute(select(DatasetMeta).where(DatasetMeta.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Try pre-computed JSON first (fast path)
    import json as _json
    dash_path = os.path.join(settings.dataset_path(dataset_id), "dashboard.json")
    if os.path.exists(dash_path):
        with open(dash_path) as f:
            return _json.load(f)

    # Regenerate on-demand if file is missing
    csv_path = _resolve_csv(dataset_id)
    if not csv_path:
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    from app.analysis.auto_dashboard import auto_dashboard
    from app.analysis.insight_narrator import insight_narrator
    df = pd.read_csv(csv_path)
    dashboard = auto_dashboard.generate(
        df=df,
        domain=dataset.domain or "general",
        dataset_id=dataset_id,
    )
    dashboard = await insight_narrator.narrate(dashboard, domain=dataset.domain or "general")

    # Cache it for next time
    with open(dash_path, "w") as f:
        _json.dump(dashboard, f, default=str)

    return dashboard


@router.get(
    "/{dataset_id}/cleaning-report",
    summary="Full cleaning transparency report",
    description="Returns before/after stats, per-action change log, and post-clean validation.",
)
async def get_cleaning_report(
    dataset_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(DatasetMeta).where(DatasetMeta.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Load stored reports from metadata file
    meta_path = os.path.join(settings.dataset_path(dataset_id), "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        import json as _json
        with open(meta_path) as f:
            metadata = _json.load(f)

    profiling   = metadata.get("profiling_report", {}) or {}
    cleaning    = metadata.get("cleaning_report",  {}) or {}
    validation  = metadata.get("validation_report", {}) or {}

    # Build human-readable action log from actions_applied
    action_log = _build_action_log(
        actions_applied=cleaning.get("actions_applied", []),
        actions_skipped=cleaning.get("actions_skipped", []),
        profiling_report=profiling,
    )

    return {
        "dataset_id": dataset_id,
        "filename": dataset.original_filename,
        # ── Before / after summary ────────────────────────────────────
        "summary": {
            "rows_before":      cleaning.get("rows_before", dataset.row_count),
            "rows_after":       cleaning.get("rows_after",  dataset.row_count),
            "rows_removed":     cleaning.get("rows_removed", 0),
            "cols_before":      cleaning.get("columns_before", dataset.column_count),
            "cols_after":       cleaning.get("columns_after", dataset.column_count),
            "cols_removed":     cleaning.get("columns_removed", 0),
            "quality_before":   profiling.get("quality_score", 0),
            "quality_after":    validation.get("quality_after", profiling.get("quality_score", 0)),
            "quality_improvement": validation.get("quality_improvement", 0),
            "actions_applied":  len(cleaning.get("actions_applied", [])),
            "actions_skipped":  len(cleaning.get("actions_skipped", [])),
        },
        # ── Per-action change log ─────────────────────────────────────
        "action_log": action_log,
        # ── Post-clean validation ─────────────────────────────────────
        "validation": {
            "overall_passed":   validation.get("overall_passed", True),
            "resolved_count":   validation.get("resolved_count", 0),
            "partial_count":    validation.get("partial_count", 0),
            "unresolved_count": validation.get("unresolved_count", 0),
            "checks":           validation.get("checks", []),
            "per_column":       validation.get("per_column", []),
            "summary":          metadata.get("validation_summary", ""),
        },
    }


def _build_action_log(
    actions_applied: list,
    actions_skipped: list,
    profiling_report: dict,
) -> list:
    """
    Translate raw action dicts into user-friendly log entries.
    Uses pre-computed reason_data when available (from executor).
    Falls back to ReasonEngine for any action that lacks it.
    """
    from app.cleaning.reason_engine import reason_engine, SEVERITY_CONFIG

    log = []

    ACTION_LABELS = {
        "fill_missing":      "Filled missing values",
        "remove_duplicates": "Removed duplicate rows",
        "drop_column":       "Removed column",
        "fix_dtype":         "Fixed data type",
        "cap_outliers":      "Capped extreme values",
        "drop_missing_rows": "Removed rows with missing data",
    }

    for action in actions_applied:
        atype  = action.get("action", "")
        col    = action.get("column")

        base_label = ACTION_LABELS.get(atype, atype.replace("_", " ").capitalize())

        # Use pre-computed reason_data if present, else generate now
        reason_data = action.get("reason_data")
        if not reason_data:
            reason_data = reason_engine.explain(action, profiling_report, {})

        severity   = reason_data.get("severity", "medium")
        sev_config = SEVERITY_CONFIG.get(severity, SEVERITY_CONFIG["medium"])

        # Short detail line (for the log header)
        missing_info = profiling_report.get("missing_values", {}).get(col, {})
        count = reason_data.get("evidence_numbers", {}).get("missing_count",
                               missing_info.get("count", 0))

        detail_parts = []
        if col:
            detail_parts.append(f"Column: {col}")
        if count:
            detail_parts.append(f"{count:,} values affected")
        fill_val = reason_data.get("evidence_numbers", {}).get("fill_value")
        if fill_val is not None:
            fv = f"{fill_val:,.2f}" if isinstance(fill_val, float) else str(fill_val)
            detail_parts.append(f"Filled with: {fv}")

        detail = " · ".join(detail_parts) if detail_parts else ""

        log.append({
            "status":     "applied",
            "action":     atype,
            "label":      base_label,
            "detail":     detail,
            "column":     col,
            "severity":   severity,
            "severity_label": sev_config["label"],
            "severity_color": sev_config["color"],
            # ── The three-part explanation ───────────────────────────────
            "why":        reason_data.get("why", ""),
            "risk":       reason_data.get("risk", ""),
            "method_why": reason_data.get("method_why", ""),
            "evidence":   reason_data.get("evidence_numbers", {}),
        })

    for action in actions_skipped:
        atype  = action.get("action", "")
        col    = action.get("column")
        reason = action.get("reason", "No action was needed.")
        base_label = ACTION_LABELS.get(atype, atype.replace("_", " ").capitalize())
        log.append({
            "status":     "skipped",
            "action":     atype,
            "label":      base_label,
            "detail":     f"Column: {col}" if col else "",
            "column":     col,
            "severity":   "low",
            "severity_label": "No change",
            "severity_color": "teal",
            "why":        reason,
            "risk":       "",
            "method_why": "",
            "evidence":   {},
        })

    return log


@router.delete("/{dataset_id}", summary="Delete a dataset")
async def delete_dataset(dataset_id: str = Path(...), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DatasetMeta).where(DatasetMeta.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    await db.execute(delete(UserHistory).where(UserHistory.dataset_id == dataset_id))
    await db.execute(delete(DatasetMeta).where(DatasetMeta.id == dataset_id))

    dataset_dir = settings.dataset_path(dataset_id)
    if os.path.exists(dataset_dir):
        import shutil
        shutil.rmtree(dataset_dir)
        logger.info(f"Deleted dataset directory: {dataset_dir}")

    return {"message": "Dataset and all associated data have been deleted.", "dataset_id": dataset_id}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_csv(dataset_id: str) -> str | None:
    """Return path to cleaned CSV, falling back to raw if cleaned doesn't exist."""
    cleaned = settings.cleaned_csv_path(dataset_id)
    if os.path.exists(cleaned):
        return cleaned
    raw = settings.raw_csv_path(dataset_id)
    if os.path.exists(raw):
        return raw
    return None


def _to_response(d: DatasetMeta) -> DatasetInfoResponse:
    return DatasetInfoResponse(
        dataset_id=d.id,
        original_filename=d.original_filename,
        status=d.status,
        row_count=d.row_count,
        column_count=d.column_count,
        columns=d.columns_json,
        dtypes=d.dtypes_json,
        domain=d.domain,
        domain_confidence=d.domain_confidence,
        quality_score=None,
        is_cleaned=d.is_cleaned,
        has_embeddings=d.has_embeddings,
        created_at=d.created_at,
        updated_at=d.updated_at,
    )