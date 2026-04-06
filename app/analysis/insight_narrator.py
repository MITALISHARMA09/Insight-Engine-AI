from __future__ import annotations
"""
InsightEngine AI - Chart Insight Narrator
Reads each dashboard tile's data and writes a plain-English interpretation.

Two-layer approach:
  1. Rule-based narrator (always runs, zero latency, no API needed)
     — reads the actual numbers from chart_data and applies smart heuristics
  2. LLM enrichment (runs when Groq key is available)
     — rewrites the rule-based draft into more natural, contextual language

Both layers attach an `insight` string directly onto each tile dict.
"""
import logging
import math
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Prompts ──────────────────────────────────────────────────────────────────

NARRATOR_SYSTEM = """You are a concise data analyst explaining charts to a business user.
Rules:
- Maximum 2 sentences. No more.
- Plain English only — no jargon (no "std dev", "median", "outlier", "null", "correlation coefficient")
- Always state the most important finding first
- Always end with a one-sentence "so what" — what action or conclusion this suggests
- Use the domain context to make it relevant (e.g. if domain is "sales", say "sales" not "values")
- FORBIDDEN words: mean, median, mode, std, deviation, null, NaN, DataFrame, pandas, correlation, regression, p-value

Return ONLY the insight text. No intro, no markdown, no quotes."""

NARRATOR_PROMPT = """Chart title: {title}
Chart type: {chart_type}
Domain: {domain}

Data summary:
{data_summary}

Write a 2-sentence plain-English insight for this chart."""


class InsightNarrator:
    """
    Attaches an `insight` string to every tile in a dashboard payload.
    Call `await narrator.narrate(dashboard, domain)` after auto_dashboard.generate().
    """

    async def narrate(self, dashboard: dict, domain: str = "general") -> dict:
        """
        Adds `insight` field to every tile in-place. Returns the same dict.
        Never raises — worst case leaves insight as rule-based text.
        """
        tiles = dashboard.get("tiles", [])

        # Step 1: rule-based insights for all tiles (instant)
        for tile in tiles:
            try:
                tile["insight"] = self._rule_based(tile, domain)
            except Exception as e:
                logger.warning(f"Rule-based insight failed for {tile.get('id')}: {e}")
                tile["insight"] = ""

        # Step 2: LLM enrichment (fire-and-forget, skip if key missing)
        try:
            from app.core.llm_client import llm_client, LLMKeyMissingError
            from app.core.config import settings

            # Only enrich chart tiles (skip KPIs — rule-based is fine for those)
            chart_tiles = [t for t in tiles if t.get("type") not in ("kpi", "table")]

            if chart_tiles:
                tasks = [
                    self._llm_enrich(tile, domain, llm_client, settings)
                    for tile in chart_tiles
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for tile, result in zip(chart_tiles, results):
                    if isinstance(result, str) and result.strip():
                        tile["insight"] = result.strip()
                    # else keep the rule-based insight

        except ImportError:
            pass  # no LLM client available
        except Exception as e:
            logger.warning(f"LLM insight enrichment skipped: {e}")

        return dashboard

    # ─── Rule-Based Narrator ──────────────────────────────────────────────────

    def _rule_based(self, tile: dict, domain: str) -> str:
        """
        Produce a meaningful insight from the tile's numbers alone.
        No LLM — pattern matching on the chart_data values.
        """
        t = tile.get("type", "")
        title = tile.get("title", "")

        if t == "kpi":
            return self._narrate_kpi(tile, domain)
        elif t == "bar":
            return self._narrate_bar(tile, domain)
        elif t == "line":
            return self._narrate_line(tile, domain)
        elif t == "pie":
            return self._narrate_pie(tile, domain)
        elif t == "table":
            return self._narrate_table(tile, domain)
        return ""

    def _narrate_kpi(self, tile: dict, domain: str) -> str:
        tid = tile.get("id", "")
        val = tile.get("value", "")
        sub = tile.get("sub", "")
        title = tile.get("title", "")
        color = tile.get("color", "")

        if "shape" in tid or "size" in tid:
            return f"Your dataset contains {val} rows across {sub.replace('columns', 'columns').lower()}. This is a good-sized dataset for meaningful analysis."

        if "missing" in tid:
            if color == "green":
                return f"All values are present — no missing data detected. Your dataset is complete and ready for analysis."
            pct = sub.replace("% of dataset", "").strip() if "%" in sub else "?"
            return f"{val} missing values were found ({pct}% of your data). These have already been filled in during cleaning, so analysis results are reliable."

        if "complete" in tid:
            return "Every cell in your dataset has a value — nothing is missing. This means all analysis will use the full dataset."

        if "avg" in tid.lower() or "kpi_" in tid:
            return f"The typical {title.replace('Avg ', '').lower()} across the dataset is {val}. {sub}."

        if "unique" in tid or "cat_" in tid:
            return f"There are {val} different {title.replace('Unique ', '').lower()} in the dataset. The most common is {sub.replace('Top: ', '')}."

        return f"{title}: {val} {tile.get('unit', '')}. {sub}."

    def _narrate_bar(self, tile: dict, domain: str) -> str:
        title = tile.get("title", "")
        cd = tile.get("chart_data", {})
        labels = cd.get("labels", [])
        datasets = cd.get("datasets", [{}])
        values = datasets[0].get("data", []) if datasets else []
        opts = tile.get("options", {})

        if not labels or not values:
            return f"This chart shows {title.lower()}."

        # ── Correlation bar ────────────────────────────────────────────────────
        if "correlation" in title.lower() or (opts.get("yMin") == -1 and opts.get("yMax") == 1):
            if not values:
                return f"This shows how other columns relate to {title.lower()}."
            max_i = max(range(len(values)), key=lambda i: abs(values[i]))
            max_col = labels[max_i] if max_i < len(labels) else "unknown"
            max_val = values[max_i]
            direction = "positively" if max_val > 0 else "negatively"
            strength = "strongly" if abs(max_val) > 0.6 else "moderately" if abs(max_val) > 0.3 else "weakly"
            base = title.replace("Correlation with ", "")
            return (f"{max_col} is {strength} and {direction} linked to {base} "
                    f"(r = {max_val:.2f}). "
                    f"{'This is the strongest relationship in the dataset — worth investigating further.' if abs(max_val) > 0.5 else 'The relationships here are relatively weak, suggesting these columns vary independently.'}")

        # ── Missing values bar ────────────────────────────────────────────────
        if "missing" in title.lower():
            if not values:
                return "No missing values were detected."
            worst_i = max(range(len(values)), key=lambda i: values[i])
            worst_col = labels[worst_i] if worst_i < len(labels) else "unknown"
            worst_n = values[worst_i]
            return (f"{worst_col} has the most missing values ({worst_n:,} entries). "
                    f"These were filled in during cleaning using the column's typical value, so analysis is not affected.")

        # ── Distribution bar ──────────────────────────────────────────────────
        if "distribution" in title.lower():
            col_name = title.replace(" distribution", "")
            if not values:
                return f"This shows how {col_name} values are spread across the dataset."
            max_i = max(range(len(values)), key=lambda i: values[i])
            max_label = labels[max_i] if max_i < len(labels) else "unknown"
            total = sum(values)
            max_pct = round(values[max_i] / total * 100) if total else 0
            # Check if concentrated (skewed) or spread (uniform)
            top3_pct = round(sum(sorted(values, reverse=True)[:3]) / total * 100) if total else 0
            if top3_pct > 70:
                return (f"Most {col_name} values cluster around {max_label} ({max_pct}% of records fall in that range). "
                        f"The distribution is skewed — a small range accounts for the majority of data.")
            return (f"Values of {col_name} are spread fairly evenly, with the highest concentration around {max_label}. "
                    f"This balanced spread suggests no extreme outliers are distorting the analysis.")

        # ── Groupby bar ────────────────────────────────────────────────────────
        if not values:
            return f"This chart compares groups for {title.lower()}."
        max_i = max(range(len(values)), key=lambda i: values[i])
        min_i = min(range(len(values)), key=lambda i: values[i])
        top_label  = labels[max_i] if max_i < len(labels) else "top group"
        bot_label  = labels[min_i] if min_i < len(labels) else "bottom group"
        top_val    = self._fmt(values[max_i])
        bot_val    = self._fmt(values[min_i])
        gap_pct    = round((values[max_i] - values[min_i]) / values[min_i] * 100) if values[min_i] != 0 else 0
        metric     = datasets[0].get("label", title) if datasets else title

        return (f"{top_label} leads with the highest {metric.lower()} at {top_val}, "
                f"while {bot_label} is at the bottom with {bot_val} — a {gap_pct}% difference. "
                f"{'This gap is significant and may warrant further investigation.' if gap_pct > 30 else 'The gap is modest across groups.'}")

    def _narrate_line(self, tile: dict, domain: str) -> str:
        title = tile.get("title", "")
        cd = tile.get("chart_data", {})
        labels = cd.get("labels", [])
        values = (cd.get("datasets") or [{}])[0].get("data", [])

        if not values or len(values) < 2:
            return f"This shows how {title.lower()} changes over time."

        first, last = values[0], values[-1]
        change_pct = round((last - first) / first * 100, 1) if first != 0 else 0
        direction = "increased" if last > first else "decreased" if last < first else "stayed flat"
        abs_change = abs(change_pct)
        magnitude = "sharply" if abs_change > 30 else "steadily" if abs_change > 10 else "slightly"

        # Find peak and trough
        peak_i = max(range(len(values)), key=lambda i: values[i])
        peak_label = labels[peak_i] if peak_i < len(labels) else "the peak period"
        peak_val = self._fmt(values[peak_i])

        metric = title.replace(" over time", "")
        period_start = labels[0] if labels else "the start"
        period_end   = labels[-1] if labels else "the end"

        return (f"{metric} {magnitude} {direction} from {self._fmt(first)} in {period_start} "
                f"to {self._fmt(last)} in {period_end} ({'+' if change_pct >= 0 else ''}{change_pct}% overall), "
                f"peaking at {peak_val} in {peak_label}. "
                f"{'This upward trend is a positive signal.' if last > first else 'The downward trend is worth monitoring closely.' if last < first else 'The flat trend suggests stability.'}")

    def _narrate_pie(self, tile: dict, domain: str) -> str:
        title = tile.get("title", "")
        cd = tile.get("chart_data", {})
        labels = cd.get("labels", [])
        values = (cd.get("datasets") or [{}])[0].get("data", [])

        if not labels or not values:
            return f"This shows the breakdown of {title.lower()}."

        total = sum(values)
        if total == 0:
            return f"No data available for this breakdown."

        sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        top_label, top_val = sorted_pairs[0]
        top_pct = round(top_val / total * 100)

        col_name = title.replace(" breakdown", "")

        if top_pct > 60:
            return (f"{top_label} dominates {col_name} with {top_pct}% of all records. "
                    f"The dataset is heavily concentrated in one group — this may skew overall averages.")
        elif top_pct > 40:
            second_label = sorted_pairs[1][0] if len(sorted_pairs) > 1 else None
            second_pct = round(sorted_pairs[1][1] / total * 100) if len(sorted_pairs) > 1 else 0
            tail = f", followed by {second_label} at {second_pct}%" if second_label else ""
            return (f"{top_label} is the largest group at {top_pct}%{tail}. "
                    f"The distribution is moderately concentrated — a few categories drive the majority of records.")
        else:
            n = len(labels)
            top3_pct = round(sum(v for _, v in sorted_pairs[:3]) / total * 100)
            return (f"The {col_name} data is spread across {n} groups, with the top three accounting for {top3_pct}% of records. "
                    f"This balanced distribution means no single group dominates the dataset.")

    def _narrate_table(self, tile: dict, domain: str) -> str:
        rows = tile.get("rows", [])
        cols = tile.get("columns", [])
        sort_col = tile.get("title", "").replace("Top 10 by ", "")
        if not rows or not cols:
            return ""
        top_row = rows[0]
        top_val = top_row.get(sort_col, "—")
        # Try to find a name column
        name_col = next((c for c in cols if c.lower() not in (sort_col.lower(),)
                         and not any(c.lower().startswith(p) for p in ["id", "num", "count"])), None)
        if name_col and top_row.get(name_col):
            return (f"{top_row[name_col]} tops the list with {self._fmt(top_val)} {sort_col.lower()}. "
                    f"Reviewing these top performers can reveal the factors behind strong results.")
        return (f"The highest {sort_col.lower()} value is {self._fmt(top_val)}. "
                f"These top 10 records represent the strongest performers in the dataset.")

    # ─── LLM Enrichment ───────────────────────────────────────────────────────

    async def _llm_enrich(self, tile: dict, domain: str, llm_client, settings) -> str:
        """
        Sends the rule-based insight + chart data to the LLM for a
        richer, more contextual rewrite. Falls back silently on any error.
        """
        try:
            from app.core.llm_client import LLMKeyMissingError
            data_summary = self._summarize_tile_for_llm(tile)
            if not data_summary:
                return tile.get("insight", "")

            prompt = NARRATOR_PROMPT.format(
                title=tile.get("title", ""),
                chart_type=tile.get("type", ""),
                domain=domain,
                data_summary=data_summary,
            )

            response = await llm_client.chat_groq(
                prompt=prompt,
                model=settings.STORYTELLER_MODEL,
                system_prompt=NARRATOR_SYSTEM,
                temperature=0.3,
                max_tokens=120,
            )
            return response.content.strip()

        except Exception:
            return tile.get("insight", "")  # keep rule-based on any failure

    def _summarize_tile_for_llm(self, tile: dict) -> str:
        """Compact text summary of a tile's numbers for the LLM prompt."""
        cd = tile.get("chart_data", {})
        labels = cd.get("labels", [])
        datasets = cd.get("datasets", [{}])
        values = datasets[0].get("data", []) if datasets else []

        if not labels or not values:
            return ""

        # Pair labels with values, sorted by value descending
        pairs = sorted(zip(labels, values), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)

        lines = []
        for label, val in pairs[:8]:
            lines.append(f"  {label}: {self._fmt(val)}")

        if len(pairs) > 8:
            lines.append(f"  … ({len(pairs) - 8} more)")

        # Add existing rule-based insight as draft context
        existing = tile.get("insight", "")
        if existing:
            lines.append(f"\nDraft insight: {existing}")

        return "\n".join(lines)

    # ─── Helper ───────────────────────────────────────────────────────────────

    def _fmt(self, v) -> str:
        if v is None:
            return "—"
        if isinstance(v, float) and math.isnan(v):
            return "—"
        if isinstance(v, (int, float)):
            fv = float(v)
            if abs(fv) >= 1_000_000:
                return f"{fv/1_000_000:.1f}M"
            if abs(fv) >= 1_000:
                return f"{fv:,.0f}"
            return f"{fv:.2f}"
        return str(v)


# Module-level singleton
insight_narrator = InsightNarrator()