from __future__ import annotations
"""
InsightEngine AI - Cleaning Validator
Re-runs the profiler on the cleaned dataset and diffs it against
the pre-clean report to verify every issue was actually resolved.
Catches silent failures like: fill_missing ran but values still null.
"""
import logging
from typing import Optional
import pandas as pd

from app.cleaning.profiler import DataProfiler

logger = logging.getLogger(__name__)

data_profiler = DataProfiler()


class CleaningValidator:
    """
    Runs profiler twice (before + after) and produces a structured
    validation report with per-issue resolution status.
    """

    def validate(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        cleaning_report: dict,
    ) -> dict:
        """
        Compare profiler reports before and after cleaning.

        Returns a validation_report dict with:
          - quality_before / quality_after
          - resolved / partial / unresolved issue lists
          - per_column results
          - overall_passed bool
        """
        try:
            before = data_profiler.profile(df_raw)
            after  = data_profiler.profile(df_clean)
        except Exception as e:
            logger.error(f"Validation profiling failed: {e}")
            return {"error": str(e), "overall_passed": False}

        report = {
            "quality_before": before.get("quality_score", 0),
            "quality_after":  after.get("quality_score", 0),
            "quality_improvement": round(
                after.get("quality_score", 0) - before.get("quality_score", 0), 1
            ),
            "rows_before":    before.get("total_rows", 0),
            "rows_after":     after.get("total_rows", 0),
            "rows_removed":   before.get("total_rows", 0) - after.get("total_rows", 0),
            "cols_before":    before.get("total_columns", 0),
            "cols_after":     after.get("total_columns", 0),
            "cols_removed":   before.get("total_columns", 0) - after.get("total_columns", 0),
            "checks": [],
            "per_column": [],
            "overall_passed": False,
        }

        # ── Check: missing values ─────────────────────────────────────────────
        missing_before = before.get("missing_values", {})
        missing_after  = after.get("missing_values", {})

        for col, info in missing_before.items():
            count_before = info["count"]
            count_after  = missing_after.get(col, {}).get("count", 0)

            if count_after == 0:
                status = "resolved"
                message = f"All {count_before} missing values in '{col}' were filled."
            elif count_after < count_before:
                status = "partial"
                message = (f"'{col}' still has {count_after} missing values "
                           f"(reduced from {count_before}).")
            else:
                status = "unresolved"
                message = f"'{col}' still has {count_after} missing values — cleaning had no effect."

            report["checks"].append({
                "issue": "missing_values",
                "column": col,
                "count_before": count_before,
                "count_after": count_after,
                "status": status,
                "message": message,
            })

        # ── Check: duplicates ─────────────────────────────────────────────────
        dup_before = before.get("duplicates", {})
        dup_after  = after.get("duplicates", {})

        if dup_before.get("has_duplicates"):
            count_before = dup_before.get("count", 0)
            count_after  = dup_after.get("count", 0)
            if count_after == 0:
                status  = "resolved"
                message = f"All {count_before} duplicate rows were removed."
            elif count_after < count_before:
                status  = "partial"
                message = f"{count_after} duplicates remain (reduced from {count_before})."
            else:
                status  = "unresolved"
                message = f"Duplicate rows were not removed ({count_after} still present)."

            report["checks"].append({
                "issue": "duplicates",
                "column": None,
                "count_before": count_before,
                "count_after": count_after,
                "status": status,
                "message": message,
            })

        # ── Check: type issues ────────────────────────────────────────────────
        type_before = before.get("type_issues", {})
        type_after  = after.get("type_issues", {})

        for col, info in type_before.items():
            if col not in type_after:
                status  = "resolved"
                message = (f"'{col}' was successfully converted from "
                           f"{info['current_type']} to {info['likely_type']}.")
            else:
                status  = "unresolved"
                message = f"'{col}' still has a type mismatch ({info['current_type']} instead of {info['likely_type']})."

            report["checks"].append({
                "issue": "type_mismatch",
                "column": col,
                "count_before": 1,
                "count_after": 0 if status == "resolved" else 1,
                "status": status,
                "message": message,
            })

        # ── Check: outliers ───────────────────────────────────────────────────
        outliers_before = before.get("outliers", {})
        outliers_after  = after.get("outliers", {})

        # Build a set of columns where capping WAS in the cleaning report
        capped_cols = {
            a.get("column") for a in cleaning_report.get("actions_applied", [])
            if a.get("action") == "cap_outliers"
        }

        for col, info in outliers_before.items():
            count_before = info["count"]
            count_after  = outliers_after.get(col, {}).get("count", 0)
            was_planned  = col in capped_cols

            if count_after == 0:
                status  = "resolved"
                message = f"Extreme values in '{col}' were capped ({count_before} adjusted)."
            elif count_after < count_before:
                status  = "partial"
                message = (f"'{col}' still has {count_after} unusual values "
                           f"(reduced from {count_before}).")
            elif not was_planned:
                # Outliers exist but AI cleaner deliberately chose not to cap — inform, don't alarm
                status  = "info"
                message = (f"'{col}' has {count_after} values outside the typical range. "
                           f"Outlier capping was not included in the cleaning plan — "
                           f"review these values manually before analysis.")
            else:
                status  = "unresolved"
                message = (f"'{col}' still has {count_after} extreme values even after "
                           f"capping was applied — the bounds may need adjustment.")

            report["checks"].append({
                "issue": "outliers",
                "column": col,
                "count_before": count_before,
                "count_after": count_after,
                "status": status,
                "message": message,
            })

        # ── Per-column summary ────────────────────────────────────────────────
        all_cols = set(list(before.get("dtypes", {}).keys()) +
                       list(after.get("dtypes", {}).keys()))

        for col in all_cols:
            dtype_before = before.get("dtypes", {}).get(col, "—")
            dtype_after  = after.get("dtypes", {}).get(col, "dropped")
            miss_before  = before.get("missing_values", {}).get(col, {}).get("count", 0)
            miss_after   = after.get("missing_values",  {}).get(col, {}).get("count", 0)

            if dtype_after == "dropped":
                col_status = "dropped"
            elif miss_before > 0 and miss_after == 0:
                col_status = "improved"
            elif miss_after > 0:
                col_status = "partial" if miss_after < miss_before else "unchanged"
            elif dtype_before != dtype_after:
                col_status = "improved"
            else:
                col_status = "clean"

            report["per_column"].append({
                "column": col,
                "status": col_status,
                "dtype_before": dtype_before,
                "dtype_after": dtype_after,
                "missing_before": miss_before,
                "missing_after": miss_after,
            })

        # ── Overall pass/fail ─────────────────────────────────────────────────
        unresolved = [c for c in report["checks"] if c["status"] == "unresolved"]
        info_items = [c for c in report["checks"] if c["status"] == "info"]
        report["overall_passed"] = len(unresolved) == 0
        report["unresolved_count"] = len(unresolved)
        report["info_count"]       = len(info_items)
        report["resolved_count"]   = len([c for c in report["checks"] if c["status"] == "resolved"])
        report["partial_count"]    = len([c for c in report["checks"] if c["status"] == "partial"])

        logger.info(
            f"Validation: {report['resolved_count']} resolved, "
            f"{report['partial_count']} partial, "
            f"{report['unresolved_count']} unresolved | "
            f"Quality {report['quality_before']}% → {report['quality_after']}%"
        )
        return report

    def plain_english_summary(self, validation_report: dict) -> str:
        """One-paragraph user-friendly summary of the validation result."""
        if not validation_report or "error" in validation_report:
            return "Cleaning completed but validation could not run."

        qb = validation_report.get("quality_before", 0)
        qa = validation_report.get("quality_after", 0)
        improvement = validation_report.get("quality_improvement", 0)
        resolved   = validation_report.get("resolved_count", 0)
        unresolved = validation_report.get("unresolved_count", 0)
        rows_removed = validation_report.get("rows_removed", 0)

        parts = []

        if improvement > 0:
            parts.append(
                f"Cleaning improved your data quality from {qb}% to {qa}% "
                f"(+{improvement} points)."
            )
        else:
            parts.append(f"Your data quality score is {qa}%.")

        if rows_removed > 0:
            parts.append(f"{rows_removed:,} duplicate rows were removed.")

        if resolved > 0:
            parts.append(f"{resolved} issue{'s' if resolved > 1 else ''} fully resolved.")

        info_count = validation_report.get("info_count", 0)
        if unresolved > 0:
            parts.append(
                f"{unresolved} issue{'s' if unresolved > 1 else ''} could not be fixed automatically "
                "— check the cleaning report for details."
            )
        elif info_count > 0:
            parts.append(
                f"{info_count} column{'s' if info_count > 1 else ''} "
                f"{'have' if info_count > 1 else 'has'} unusual values that were not auto-corrected "
                f"— review them before analysis."
            )
        else:
            parts.append("All detected issues were resolved successfully.")

        return " ".join(parts)


cleaning_validator = CleaningValidator()