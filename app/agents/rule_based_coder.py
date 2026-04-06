from __future__ import annotations
"""
InsightEngine AI - Rule-Based Coder
Generates pandas code from natural language using pattern matching.
Works 100% offline with NO API keys needed.
Used as fallback when both CoderA and CoderB fail.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RuleBasedCoder:
    """
    Keyword-pattern → pandas code mapper.
    Covers the 30 most common data questions without any LLM.
    """

    def generate(
        self,
        user_query: str,
        columns: list[str],
        dtypes: dict,
        sample_rows: list[dict],
    ) -> dict:
        q = user_query.lower().strip()
        num_cols  = [c for c, t in dtypes.items() if "int" in t or "float" in t]
        cat_cols  = [c for c, t in dtypes.items() if "object" in t or "str" in t or "bool" in t]
        date_cols = [c for c, t in dtypes.items() if "date" in t or "time" in t]
        all_cols  = columns

        code = self._match(q, all_cols, num_cols, cat_cols, date_cols, sample_rows)

        if code:
            logger.info(f"Rule-based coder matched query: '{user_query[:60]}'")
            return {"code": code, "success": True, "error": None, "agent": "rule_based"}
        else:
            # Ultimate fallback — describe the dataset
            code = self._describe_all(all_cols, num_cols, cat_cols)
            logger.info("Rule-based coder using describe-all fallback")
            return {"code": code, "success": True, "error": None, "agent": "rule_based_fallback"}

    def _match(self, q, all_cols, num_cols, cat_cols, date_cols, sample_rows) -> Optional[str]:
        # ── Shape / size ──────────────────────────────────────────────────────
        if any(k in q for k in ["how many rows", "row count", "size of", "shape", "how big"]):
            return "result = {'rows': len(df), 'columns': len(df.columns), 'shape': str(df.shape)}"

        if any(k in q for k in ["columns", "column names", "fields", "variables"]):
            return "result = list(df.columns)"

        # ── Missing / null values ─────────────────────────────────────────────
        if any(k in q for k in ["missing", "null", "nan", "empty", "incomplete"]):
            return (
                "missing = df.isnull().sum()\n"
                "pct = (df.isnull().mean() * 100).round(2)\n"
                "result = {col: {'missing_count': int(missing[col]), 'missing_pct': float(pct[col])}\n"
                "          for col in df.columns if missing[col] > 0}\n"
                "if not result:\n"
                "    result = 'Great news — no missing values found in this dataset!'"
            )

        # ── Duplicates ────────────────────────────────────────────────────────
        if any(k in q for k in ["duplicate", "repeated", "unique rows"]):
            return (
                "dup_count = int(df.duplicated().sum())\n"
                "result = {'duplicate_rows': dup_count, 'unique_rows': len(df) - dup_count, "
                "'duplicate_pct': round(dup_count / len(df) * 100, 2)}"
            )

        # ── Summary / overview / insights ─────────────────────────────────────
        if any(k in q for k in ["summary", "overview", "describe", "insight", "tell me about",
                                  "analyse", "analyze", "explore", "statistics", "stats"]):
            return self._summary_code(all_cols, num_cols, cat_cols)

        # ── Show / give me the dataset ────────────────────────────────────────
        if any(k in q for k in ["show", "give me", "display", "view", "full dataset",
                                  "cleaned dataset", "all data", "entire dataset"]):
            return "result = df.head(50)"

        # ── Head / first rows ─────────────────────────────────────────────────
        if any(k in q for k in ["first", "top", "head", "sample"]):
            n = self._extract_number(q) or 10
            return f"result = df.head({n})"

        # ── Tail / last rows ──────────────────────────────────────────────────
        if any(k in q for k in ["last", "tail", "bottom", "recent"]):
            n = self._extract_number(q) or 10
            return f"result = df.tail({n})"

        # ── Average / mean ────────────────────────────────────────────────────
        if any(k in q for k in ["average", "mean", "avg"]):
            col = self._find_col(q, num_cols or all_cols)
            if col:
                return f"result = round(float(df['{col}'].mean()), 2)"
            if num_cols:
                return f"result = df[{num_cols}].mean().round(2).to_dict()"

        # ── Total / sum ───────────────────────────────────────────────────────
        if any(k in q for k in ["total", "sum", "overall"]):
            col = self._find_col(q, num_cols or all_cols)
            if col:
                return f"result = round(float(df['{col}'].sum()), 2)"
            if num_cols:
                return f"result = df[{num_cols}].sum().round(2).to_dict()"

        # ── Maximum / highest ─────────────────────────────────────────────────
        if any(k in q for k in ["maximum", "highest", "largest", "biggest", "most", "max"]):
            col = self._find_col(q, num_cols or all_cols)
            if col:
                idx = f"df['{col}'].idxmax()"
                return (f"max_val = df['{col}'].max()\n"
                        f"max_row = df.loc[{idx}].to_dict()\n"
                        f"result = {{'max_value': max_val, 'row': max_row}}")
            if num_cols:
                return f"result = df[{num_cols}].max().to_dict()"

        # ── Minimum / lowest ──────────────────────────────────────────────────
        if any(k in q for k in ["minimum", "lowest", "smallest", "least", "min"]):
            col = self._find_col(q, num_cols or all_cols)
            if col:
                idx = f"df['{col}'].idxmin()"
                return (f"min_val = df['{col}'].min()\n"
                        f"min_row = df.loc[{idx}].to_dict()\n"
                        f"result = {{'min_value': min_val, 'row': min_row}}")
            if num_cols:
                return f"result = df[{num_cols}].min().to_dict()"

        # ── Count / frequency / how many ──────────────────────────────────────
        if any(k in q for k in ["count", "frequency", "how many", "number of", "occurrences"]):
            col = self._find_col(q, cat_cols or all_cols)
            if col:
                return (f"counts = df['{col}'].value_counts()\n"
                        f"result = counts.reset_index().rename(columns={{'{col}': 'count', 'index': '{col}'}}).head(20)")
            return "result = df.count().to_dict()"

        # ── Distribution / spread ─────────────────────────────────────────────
        if any(k in q for k in ["distribution", "spread", "range", "variation"]):
            col = self._find_col(q, num_cols or all_cols)
            if col:
                s = f"df['{col}'].dropna()"
                return (f"s = {s}\n"
                        f"result = {{'min': round(float(s.min()),2), 'max': round(float(s.max()),2),\n"
                        f"          'mean': round(float(s.mean()),2), 'median': round(float(s.median()),2),\n"
                        f"          'std': round(float(s.std()),2)}}")

        # ── Unique values ─────────────────────────────────────────────────────
        if any(k in q for k in ["unique", "distinct", "different values", "categories"]):
            col = self._find_col(q, cat_cols or all_cols)
            if col:
                return (f"uniq = df['{col}'].dropna().unique().tolist()\n"
                        f"result = {{'column': '{col}', 'unique_count': len(uniq), 'values': uniq[:50]}}")
            return "result = {col: int(df[col].nunique()) for col in df.columns}"

        # ── Group by ──────────────────────────────────────────────────────────
        if any(k in q for k in ["group", "by", "per", "each", "breakdown", "split"]):
            grp_col = self._find_col(q, cat_cols)
            val_col = self._find_col(q, num_cols)
            if grp_col and val_col:
                return (f"result = df.groupby('{grp_col}')['{val_col}'].agg(['mean','sum','count'])"
                        f".round(2).reset_index()")
            elif grp_col:
                return f"result = df['{grp_col}'].value_counts().reset_index().head(20)"

        # ── Correlation ───────────────────────────────────────────────────────
        if any(k in q for k in ["correlation", "corr", "relationship", "related"]):
            if len(num_cols) >= 2:
                return (f"corr = df[{num_cols[:6]}].corr().round(3)\n"
                        f"result = corr")
            return "result = 'Not enough numeric columns to compute correlation.'"

        # ── Sort / ranking / top N ────────────────────────────────────────────
        if any(k in q for k in ["sort", "rank", "ranking", "top", "best", "performing", "leading"]):
            col = self._find_col(q, num_cols or all_cols)
            n = self._extract_number(q) or 10
            if col:
                return f"result = df.nlargest({n}, '{col}')"
            if num_cols:
                return f"result = df.nlargest({n}, '{num_cols[0]}')"

        # ── Outliers ──────────────────────────────────────────────────────────
        if any(k in q for k in ["outlier", "anomal", "unusual", "extreme", "weird"]):
            if num_cols:
                col = self._find_col(q, num_cols) or num_cols[0]
                return (f"q1 = df['{col}'].quantile(0.25)\n"
                        f"q3 = df['{col}'].quantile(0.75)\n"
                        f"iqr = q3 - q1\n"
                        f"mask = (df['{col}'] < q1 - 1.5*iqr) | (df['{col}'] > q3 + 1.5*iqr)\n"
                        f"result = df[mask]")

        # ── Percentage / proportion ───────────────────────────────────────────
        if any(k in q for k in ["percent", "%", "proportion", "share", "ratio"]):
            col = self._find_col(q, cat_cols or all_cols)
            if col:
                return (f"counts = df['{col}'].value_counts()\n"
                        f"pct = (counts / len(df) * 100).round(2)\n"
                        f"result = pd.DataFrame({{'count': counts, 'percentage': pct}}).reset_index()")

        # ── Date/time ─────────────────────────────────────────────────────────
        if any(k in q for k in ["trend", "over time", "monthly", "yearly", "daily", "by year", "by month"]):
            if date_cols:
                dc = date_cols[0]
                vc = num_cols[0] if num_cols else None
                if vc:
                    return (f"df['{dc}'] = pd.to_datetime(df['{dc}'], errors='coerce')\n"
                            f"result = df.groupby(df['{dc}'].dt.to_period('M'))['{vc}'].sum()"
                            f".reset_index().rename(columns={{'{dc}': 'period'}})")

        return None

    def _summary_code(self, all_cols, num_cols, cat_cols) -> str:
        lines = [
            "summary = {}",
            f"summary['shape'] = {{'rows': len(df), 'columns': len(df.columns)}}",
            f"summary['columns'] = list(df.columns)",
        ]
        if num_cols:
            lines.append(f"summary['numeric_stats'] = df[{num_cols[:5]}].describe().round(2).to_dict()")
        if cat_cols:
            lines.append(
                f"summary['categories'] = {{{', '.join([repr(c) + ': df[' + repr(c) + '].value_counts().head(5).to_dict()' for c in cat_cols[:3]])}}}"
            )
        lines.append(
            "summary['missing'] = {c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().sum() > 0}"
        )
        lines.append("result = summary")
        return "\n".join(lines)

    def _describe_all(self, all_cols, num_cols, cat_cols) -> str:
        return (
            "result = {\n"
            "  'rows': len(df),\n"
            "  'columns': list(df.columns),\n"
            + (f"  'numeric_summary': df[{num_cols[:5]}].describe().round(2).to_dict(),\n" if num_cols else "") +
            "  'missing_values': {c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().sum() > 0}\n"
            "}"
        )

    def _find_col(self, query: str, candidates: list[str]) -> Optional[str]:
        """Find the column most relevant to the query by name matching."""
        q = query.lower()
        # Exact word match first
        for col in candidates:
            if col.lower() in q:
                return col
        # Partial match
        for col in candidates:
            words = re.split(r'[_\s]', col.lower())
            if any(w in q for w in words if len(w) > 2):
                return col
        # Return first candidate as default
        return candidates[0] if candidates else None

    def _extract_number(self, query: str) -> Optional[int]:
        """Extract a number from the query, e.g. 'top 5' → 5."""
        words = {"one":1,"two":2,"three":3,"four":4,"five":5,
                 "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        for w, n in words.items():
            if w in query:
                return n
        m = re.search(r'\b(\d+)\b', query)
        return int(m.group(1)) if m else None


rule_based_coder = RuleBasedCoder()