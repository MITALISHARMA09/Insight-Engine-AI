from __future__ import annotations
"""
InsightEngine AI - Auto Dashboard Generator
Runs after upload completes. Computes standard analyses on the
cleaned dataset and returns structured Chart.js-ready data.
Zero LLM calls — pure pandas — always works even without API keys.
"""
import logging
import math
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AutoDashboard:
    """
    Generates a dashboard payload from a cleaned DataFrame.

    Tile types:
      kpi    — single number with label and color
      bar    — vertical bar chart (Chart.js)
      line   — line / time-series chart
      pie    — pie / donut chart
      table  — top-N rows table
    """

    def generate(self, df: pd.DataFrame, domain: str = "general", dataset_id: str = "") -> dict:
        """Returns a full dashboard dict. Never raises."""
        cols     = list(df.columns)
        num_cols  = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols  = [c for c in cols
                     if not pd.api.types.is_numeric_dtype(df[c])
                     and not pd.api.types.is_datetime64_any_dtype(df[c])]
        date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]

        tiles = []

        # ── KPI row ──────────────────────────────────────────────────────────
        tiles.append(self._kpi_shape(df))
        if num_cols:
            tiles.append(self._kpi_numeric(df, num_cols[0]))
        if cat_cols:
            tiles.append(self._kpi_categories(df, cat_cols[0]))
        tiles.append(self._kpi_missing(df))

        # ── Charts ───────────────────────────────────────────────────────────
        if cat_cols and num_cols:
            tiles.append(self._bar_groupby(df, cat_cols[0], num_cols[0]))

        if num_cols:
            tiles.append(self._distribution(df, num_cols[0]))

        if cat_cols:
            tiles.append(self._pie_category(df, cat_cols[0]))

        if len(num_cols) >= 2:
            tiles.append(self._correlation_bar(df, num_cols))

        if date_cols and num_cols:
            tiles.append(self._time_series(df, date_cols[0], num_cols[0]))

        # ── Missing values visual ─────────────────────────────────────────────
        tiles.append(self._missing_bar(df))

        # ── Top-N table ───────────────────────────────────────────────────────
        if num_cols:
            tiles.append(self._top_table(df, num_cols[0]))

        tiles = [t for t in tiles if t is not None]
        logger.info(f"[{dataset_id}] Dashboard: {len(tiles)} tiles generated")

        return {
            "dataset_id": dataset_id,
            "domain": domain,
            "tile_count": len(tiles),
            "tiles": tiles,
        }

    # ─── KPI Tiles ────────────────────────────────────────────────────────────

    def _kpi_shape(self, df: pd.DataFrame) -> Optional[dict]:
        try:
            return {
                "id": "kpi_shape", "type": "kpi",
                "title": "Dataset size",
                "value": f"{len(df):,}", "unit": "rows",
                "sub": f"{len(df.columns)} columns", "color": "teal",
            }
        except Exception:
            return None

    def _kpi_numeric(self, df: pd.DataFrame, col: str) -> Optional[dict]:
        try:
            s = df[col].dropna()
            return {
                "id": f"kpi_{col}", "type": "kpi",
                "title": f"Avg {col}",
                "value": self._fmt(s.mean()), "unit": "",
                "sub": f"min {self._fmt(s.min())}  ·  max {self._fmt(s.max())}",
                "color": "blue",
            }
        except Exception:
            return None

    def _kpi_categories(self, df: pd.DataFrame, col: str) -> Optional[dict]:
        try:
            vc = df[col].value_counts()
            top = str(vc.index[0])[:22] if len(vc) else "—"
            return {
                "id": f"kpi_cat_{col}", "type": "kpi",
                "title": f"Unique {col}",
                "value": str(df[col].nunique()), "unit": "values",
                "sub": f"Top: {top}", "color": "purple",
            }
        except Exception:
            return None

    def _kpi_missing(self, df: pd.DataFrame) -> Optional[dict]:
        try:
            missing = int(df.isnull().sum().sum())
            pct = round(missing / df.size * 100, 1) if df.size else 0
            color = "green" if pct < 5 else "amber" if pct < 20 else "red"
            return {
                "id": "kpi_missing", "type": "kpi",
                "title": "Missing values",
                "value": str(missing), "unit": "cells",
                "sub": f"{pct}% of dataset", "color": color,
            }
        except Exception:
            return None

    # ─── Chart Tiles ──────────────────────────────────────────────────────────

    def _bar_groupby(self, df: pd.DataFrame, cat: str, num: str) -> Optional[dict]:
        try:
            g = df.groupby(cat)[num].mean().round(2).sort_values(ascending=False).head(10)
            if len(g) < 2:
                return None
            return {
                "id": f"bar_{cat}_{num}", "type": "bar",
                "title": f"Avg {num} by {cat}",
                "chart_data": {
                    "labels": [str(l)[:20] for l in g.index],
                    "datasets": [{"label": f"Avg {num}", "data": [float(v) for v in g.values]}],
                },
            }
        except Exception:
            return None

    def _distribution(self, df: pd.DataFrame, col: str) -> Optional[dict]:
        try:
            s = df[col].dropna()
            if len(s) < 5:
                return None
            counts, edges = np.histogram(s, bins=min(12, max(5, len(s) // 20)))
            labels = [f"{self._fmt(edges[i])}" for i in range(len(counts))]
            return {
                "id": f"dist_{col}", "type": "bar",
                "title": f"{col} distribution",
                "chart_data": {
                    "labels": labels,
                    "datasets": [{"label": "Count", "data": counts.tolist()}],
                },
                "color_scheme": "teal",
            }
        except Exception:
            return None

    def _pie_category(self, df: pd.DataFrame, col: str) -> Optional[dict]:
        try:
            vc = df[col].value_counts().head(8)
            if len(vc) < 2:
                return None
            return {
                "id": f"pie_{col}", "type": "pie",
                "title": f"{col} breakdown",
                "chart_data": {
                    "labels": [str(l)[:20] for l in vc.index],
                    "datasets": [{"data": [int(v) for v in vc.values]}],
                },
            }
        except Exception:
            return None

    def _correlation_bar(self, df: pd.DataFrame, num_cols: list[str]) -> Optional[dict]:
        try:
            top = num_cols[:6]
            if len(top) < 2:
                return None
            corr = df[top].corr().round(3)
            base = top[0]
            others = [c for c in top if c != base]
            vals = [float(corr.loc[base, c]) for c in others]
            return {
                "id": "correlation", "type": "bar",
                "title": f"Correlation with {base}",
                "chart_data": {
                    "labels": [str(c)[:20] for c in others],
                    "datasets": [{"label": "r", "data": vals}],
                },
                "options": {"yMin": -1, "yMax": 1},
                "color_scheme": "purple",
            }
        except Exception:
            return None

    def _time_series(self, df: pd.DataFrame, date_col: str, num_col: str) -> Optional[dict]:
        try:
            ts = df[[date_col, num_col]].copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna().set_index(date_col).sort_index()
            monthly = ts[num_col].resample("ME").sum().tail(24)
            if len(monthly) < 2:
                return None
            return {
                "id": f"line_{num_col}", "type": "line",
                "title": f"{num_col} over time",
                "chart_data": {
                    "labels": [d.strftime("%b %Y") for d in monthly.index],
                    "datasets": [{
                        "label": num_col,
                        "data": [round(float(v), 2) for v in monthly.values],
                        "fill": True,
                    }],
                },
            }
        except Exception:
            return None

    def _missing_bar(self, df: pd.DataFrame) -> Optional[dict]:
        try:
            miss = df.isnull().sum()
            miss = miss[miss > 0]
            if len(miss) == 0:
                return {
                    "id": "kpi_complete", "type": "kpi",
                    "title": "Data completeness",
                    "value": "100%", "unit": "",
                    "sub": "No missing values", "color": "green",
                }
            return {
                "id": "missing_bar", "type": "bar",
                "title": "Missing values per column",
                "chart_data": {
                    "labels": [str(c)[:20] for c in miss.index],
                    "datasets": [{"label": "Missing", "data": [int(v) for v in miss.values]}],
                },
                "color_scheme": "danger",
            }
        except Exception:
            return None

    def _top_table(self, df: pd.DataFrame, sort_col: str) -> Optional[dict]:
        try:
            top = df.nlargest(10, sort_col)
            cols = list(top.columns[:6])
            rows = [{c: self._safe(top.iloc[i][c]) for c in cols} for i in range(len(top))]
            return {
                "id": f"top_{sort_col}", "type": "table",
                "title": f"Top 10 by {sort_col}",
                "columns": cols, "rows": rows,
            }
        except Exception:
            return None

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _fmt(self, v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if abs(fv) >= 1_000_000:
                return f"{fv/1_000_000:.1f}M"
            if abs(fv) >= 1_000:
                return f"{fv:,.0f}"
            return f"{fv:.2f}"
        return str(v)

    def _safe(self, v) -> str | int | float:
        if v is None:
            return "—"
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            fv = float(v)
            return "—" if math.isnan(fv) else round(fv, 2)
        if isinstance(v, float) and math.isnan(v):
            return "—"
        return str(v)[:40]


auto_dashboard = AutoDashboard()