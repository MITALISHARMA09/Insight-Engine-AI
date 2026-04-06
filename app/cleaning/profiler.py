"""
InsightEngine AI - Cleaning Module: Data Profiler
Analyzes a DataFrame and produces a comprehensive quality report.
Pure Python/pandas — no LLM needed for detection.
"""
import logging
import json
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Profiles a DataFrame to detect quality issues.
    Generates a structured report used by AI Cleaner to plan fixes.
    """

    def profile(self, df: pd.DataFrame) -> dict:
        """
        Full data quality profile.

        Returns:
            dict with missing_values, duplicates, outliers, type_issues, summary
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": self._detect_missing(df),
            "duplicates": self._detect_duplicates(df),
            "outliers": self._detect_outliers(df),
            "type_issues": self._detect_type_issues(df),
            "constant_columns": self._detect_constant_columns(df),
            "high_cardinality": self._detect_high_cardinality(df),
            "quality_score": 0.0,  # Filled below
            "issues_count": 0,     # Filled below
        }

        # Compute overall quality score
        report["quality_score"] = self._compute_quality_score(report)
        report["issues_count"] = self._count_issues(report)

        logger.info(
            f"Profiled {len(df)} rows × {len(df.columns)} cols | "
            f"Quality score: {report['quality_score']:.1f}% | "
            f"Issues: {report['issues_count']}"
        )
        return report

    # ─── Individual Detectors ─────────────────────────────────────────────────

    def _detect_missing(self, df: pd.DataFrame) -> dict:
        """Detect missing values per column."""
        result = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                result[col] = {
                    "count": null_count,
                    "percentage": round(null_count / len(df) * 100, 2),
                    "dtype": str(df[col].dtype),
                }
        return result

    def _detect_duplicates(self, df: pd.DataFrame) -> dict:
        """Detect duplicate rows."""
        dup_count = int(df.duplicated().sum())
        return {
            "count": dup_count,
            "percentage": round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0,
            "has_duplicates": dup_count > 0,
        }

    def _detect_outliers(self, df: pd.DataFrame) -> dict:
        """
        Detect numeric outliers using IQR method.
        Only flags columns with significant outliers (>1% of rows).
        """
        result = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)
            outlier_count = int(outlier_mask.sum())

            if outlier_count > 0 and outlier_count / len(series) > 0.001:
                result[col] = {
                    "count": outlier_count,
                    "percentage": round(outlier_count / len(series) * 100, 2),
                    "lower_bound": round(float(lower), 4),
                    "upper_bound": round(float(upper), 4),
                    "min_value": round(float(series.min()), 4),
                    "max_value": round(float(series.max()), 4),
                }
        return result

    def _detect_type_issues(self, df: pd.DataFrame) -> dict:
        """
        Detect columns that appear to be numeric/date but stored as strings.
        """
        issues = {}
        for col in df.select_dtypes(include=["object"]).columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            # Check if it looks like numbers
            numeric_count = pd.to_numeric(series, errors="coerce").notna().sum()
            if numeric_count / len(series) > 0.9:
                issues[col] = {"likely_type": "numeric", "current_type": "object"}
                continue

            # Check if it looks like dates
            try:
                date_count = pd.to_datetime(series.head(50), errors="coerce").notna().sum()
                if date_count / min(50, len(series)) > 0.8:
                    issues[col] = {"likely_type": "datetime", "current_type": "object"}
            except Exception:
                pass

        return issues

    def _detect_constant_columns(self, df: pd.DataFrame) -> list:
        """Detect columns with only one unique value (useless for analysis)."""
        return [
            col for col in df.columns
            if df[col].nunique(dropna=False) <= 1
        ]

    def _detect_high_cardinality(self, df: pd.DataFrame) -> dict:
        """
        Flag categorical columns with very high cardinality (potential ID columns).
        High cardinality = more than 50% unique values.
        """
        result = {}
        for col in df.select_dtypes(include=["object"]).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5 and df[col].nunique() > 10:
                result[col] = {
                    "unique_count": int(df[col].nunique()),
                    "unique_ratio": round(float(unique_ratio), 3),
                }
        return result

    # ─── Quality Scoring ──────────────────────────────────────────────────────

    def _compute_quality_score(self, report: dict) -> float:
        """
        Compute a 0-100 quality score based on detected issues.
        Higher = cleaner dataset.
        """
        score = 100.0
        total_rows = report["total_rows"]
        total_cols = report["total_columns"]

        if total_rows == 0 or total_cols == 0:
            return 0.0

        # Penalize missing values
        missing = report["missing_values"]
        if missing:
            avg_missing_pct = sum(v["percentage"] for v in missing.values()) / total_cols
            score -= min(avg_missing_pct * 1.5, 30)  # Max -30

        # Penalize duplicates
        dup_pct = report["duplicates"]["percentage"]
        score -= min(dup_pct * 0.5, 15)  # Max -15

        # Penalize outliers
        outlier_count = len(report["outliers"])
        score -= min(outlier_count * 3, 20)  # Max -20

        # Penalize type issues
        type_issues = len(report["type_issues"])
        score -= min(type_issues * 2, 15)  # Max -15

        return round(max(score, 0.0), 1)

    def _count_issues(self, report: dict) -> int:
        """Total number of detected issues."""
        count = 0
        count += len(report["missing_values"])
        count += 1 if report["duplicates"]["has_duplicates"] else 0
        count += len(report["outliers"])
        count += len(report["type_issues"])
        count += len(report["constant_columns"])
        return count

    def user_friendly_summary(self, report: dict) -> str:
        """
        Generate a plain-English summary of data quality.
        Used in the UI to inform users before cleaning.
        """
        issues = []

        missing = report["missing_values"]
        if missing:
            cols = list(missing.keys())[:3]
            issues.append(f"Some entries are missing in columns like: {', '.join(cols)}")

        if report["duplicates"]["has_duplicates"]:
            count = report["duplicates"]["count"]
            issues.append(f"There are {count} repeated rows that can be removed")

        if report["outliers"]:
            cols = list(report["outliers"].keys())[:2]
            issues.append(f"Some unusual values were found in: {', '.join(cols)}")

        if report["type_issues"]:
            issues.append("Some columns have mixed data types that need fixing")

        if not issues:
            return "✅ Your dataset looks clean! No major issues found."

        score = report["quality_score"]
        intro = f"Your data has a quality score of {score}/100. Here's what we found:\n"
        return intro + "\n".join([f"• {i}" for i in issues])


# Module-level singleton
data_profiler = DataProfiler()