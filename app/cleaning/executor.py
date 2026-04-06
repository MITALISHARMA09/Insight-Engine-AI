from __future__ import annotations
"""
InsightEngine AI - Cleaning Module: Executor
Applies the AI-generated cleaning plan to the DataFrame using pure pandas.
The LLM decided WHAT to do — Python decides HOW.
"""
import logging
from typing import Optional
import pandas as pd
import numpy as np
from app.cleaning.reason_engine import reason_engine

logger = logging.getLogger(__name__)


class CleaningExecutor:
    """
    Deterministically applies a validated JSON cleaning plan to a DataFrame.
    Every action is implemented in safe, predictable pandas code.
    """

    def execute(
        self,
        df: pd.DataFrame,
        plan: list[dict],
        profiling_report: dict | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Apply all cleaning actions from the plan.

        Args:
            df: Input DataFrame (will NOT be modified in-place — copy is made)
            plan: Validated list of cleaning action dicts
            profiling_report: Optional profiler output used to generate reasons

        Returns:
            Tuple of (cleaned_df, execution_report)
        """
        profiling_report = profiling_report or {}
        df_clean = df.copy()
        report = {
            "actions_applied": [],
            "actions_skipped": [],
            "rows_before": len(df),
            "rows_after": 0,
            "columns_before": len(df.columns),
            "columns_after": 0,
        }

        for action in plan:
            try:
                # Capture runtime context (actual fill values, row counts)
                ctx = self._build_execution_context(df_clean, action)
                result = self._apply_action(df_clean, action)
                if result is not None:
                    ctx["rows_removed"] = len(df_clean) - len(result)
                    df_clean = result

                # Generate deterministic reason from profiling + context
                reason = reason_engine.explain(action, profiling_report, ctx)
                enriched = {**action, "reason_data": reason}
                report["actions_applied"].append(enriched)
                logger.info(f"Applied: {action}")
            except Exception as e:
                logger.warning(f"Skipped action {action} due to error: {e}")
                skipped = {**action, "reason": str(e)}
                report["actions_skipped"].append(skipped)

        report["rows_after"] = len(df_clean)
        report["columns_after"] = len(df_clean.columns)
        report["rows_removed"] = report["rows_before"] - report["rows_after"]
        report["columns_removed"] = report["columns_before"] - report["columns_after"]

        logger.info(
            f"Cleaning complete: {len(df)} → {len(df_clean)} rows | "
            f"{len(df.columns)} → {len(df_clean.columns)} cols | "
            f"{len(report['actions_applied'])} actions applied"
        )
        return df_clean, report

    def _build_execution_context(self, df: pd.DataFrame, action: dict) -> dict:
        """Capture actual runtime values before the action runs."""
        ctx = {}
        atype = action.get("action", "")
        col   = action.get("column")

        if atype == "fill_missing" and col and col in df.columns:
            method = action.get("method", "median")
            series = df[col].dropna()
            try:
                if method == "median":
                    ctx["fill_value"] = round(float(series.median()), 4)
                elif method == "mean":
                    ctx["fill_value"] = round(float(series.mean()), 4)
                elif method == "mode":
                    mode_vals = series.mode()
                    ctx["fill_value"] = mode_vals.iloc[0] if len(mode_vals) else None
                elif method == "zero":
                    ctx["fill_value"] = 0
                ctx["actual_count"] = int(df[col].isnull().sum())
            except Exception:
                pass

        elif atype == "remove_duplicates":
            ctx["rows_removed"] = int(df.duplicated().sum())

        elif atype == "cap_outliers" and col and col in df.columns:
            series = df[col].dropna()
            lower = action.get("lower")
            upper = action.get("upper")
            mask = pd.Series([False] * len(series))
            if lower is not None:
                mask |= series < lower
            if upper is not None:
                mask |= series > upper
            ctx["actual_count"] = int(mask.sum())

        return ctx

    def _apply_action(self, df: pd.DataFrame, action: dict) -> Optional[pd.DataFrame]:
        """Dispatch a single action to its handler."""
        action_type = action.get("action")

        handlers = {
            "fill_missing": self._fill_missing,
            "remove_duplicates": self._remove_duplicates,
            "drop_column": self._drop_column,
            "fix_dtype": self._fix_dtype,
            "cap_outliers": self._cap_outliers,
            "drop_missing_rows": self._drop_missing_rows,
        }

        handler = handlers.get(action_type)
        if not handler:
            logger.warning(f"Unknown action: {action_type}")
            return None

        return handler(df, action)

    # ─── Action Handlers ─────────────────────────────────────────────────────

    def _fill_missing(self, df: pd.DataFrame, action: dict) -> pd.DataFrame:
        col = action["column"]
        method = action.get("method", "median")

        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        series = df[col]

        if method == "median":
            fill_val = series.median()
        elif method == "mean":
            fill_val = series.mean()
        elif method == "mode":
            mode_vals = series.mode()
            fill_val = mode_vals[0] if len(mode_vals) > 0 else None
        elif method == "zero":
            fill_val = 0
        elif method == "ffill":
            df[col] = series.ffill()
            return df
        elif method == "bfill":
            df[col] = series.bfill()
            return df
        elif method == "drop":
            return df.dropna(subset=[col]).reset_index(drop=True)
        else:
            fill_val = series.median() if pd.api.types.is_numeric_dtype(series) else series.mode()[0]

        df[col] = series.fillna(fill_val)
        return df

    def _remove_duplicates(self, df: pd.DataFrame, action: dict) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)

    def _drop_column(self, df: pd.DataFrame, action: dict) -> pd.DataFrame:
        col = action["column"]
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        return df.drop(columns=[col])

    def _fix_dtype(self, df: pd.DataFrame, action: dict) -> pd.DataFrame:
        col = action["column"]
        target_type = action.get("target_type", "numeric")

        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        if target_type == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif target_type == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        elif target_type == "string":
            df[col] = df[col].astype(str)

        return df

    def _cap_outliers(self, df: pd.DataFrame, action: dict) -> pd.DataFrame:
        """Winsorize: cap values at specified bounds."""
        col = action["column"]
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric, cannot cap outliers")

        lower = action.get("lower")
        upper = action.get("upper")

        if lower is not None:
            df[col] = df[col].clip(lower=lower)
        if upper is not None:
            df[col] = df[col].clip(upper=upper)

        return df

    def _drop_missing_rows(self, df: pd.DataFrame, action: dict) -> pd.DataFrame:
        """Drop rows where a specific column has missing values."""
        col = action.get("column")
        threshold = action.get("threshold")  # Drop rows where % missing exceeds this

        if col:
            return df.dropna(subset=[col]).reset_index(drop=True)
        elif threshold:
            # Drop rows with more than (threshold * 100)% missing columns
            min_non_null = int((1 - threshold) * len(df.columns))
            return df.dropna(thresh=min_non_null).reset_index(drop=True)
        else:
            return df.dropna().reset_index(drop=True)

    def cleaning_report_summary(self, report: dict) -> str:
        """User-friendly summary of what was cleaned."""
        applied = len(report["actions_applied"])
        rows_removed = report.get("rows_removed", 0)
        cols_removed = report.get("columns_removed", 0)

        lines = [f"We cleaned your dataset with {applied} improvements:"]
        if rows_removed > 0:
            lines.append(f"• Removed {rows_removed} duplicate or incomplete rows")
        if cols_removed > 0:
            lines.append(f"• Removed {cols_removed} columns that had no useful data")

        for action in report["actions_applied"]:
            if action["action"] == "fill_missing":
                lines.append(f"• Filled in missing values in '{action['column']}'")
            elif action["action"] == "fix_dtype":
                lines.append(f"• Fixed data type for '{action['column']}'")
            elif action["action"] == "cap_outliers":
                lines.append(f"• Corrected extreme values in '{action['column']}'")

        skipped = len(report["actions_skipped"])
        if skipped > 0:
            lines.append(f"• {skipped} minor fixes were skipped (already clean)")

        return "\n".join(lines)


# Module-level singleton
cleaning_executor = CleaningExecutor()