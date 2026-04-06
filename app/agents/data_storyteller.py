from __future__ import annotations
"""
InsightEngine AI - Agent: Data Storyteller (data_storyteller)
Converts raw analysis results into simple, human-friendly business insights.
Non-technical users should understand EVERY word.
"""
import logging
import json
from typing import Any, Optional

from app.core.llm_client import llm_client, LLMError
from app.core.config import settings

logger = logging.getLogger(__name__)

STORYTELLER_SYSTEM_PROMPT = """You are a friendly data analyst explaining results to a business owner with no technical background.

Your communication style:
- Use plain, everyday language (NO jargon)
- Replace technical terms: "median" → "typical value", "outlier" → "unusual value", "null" → "missing"
- Always include the "so what?" — why does this result matter?
- Use ₹ for Indian currency if the domain suggests it, otherwise $
- Keep responses to 3-5 sentences max
- Sound warm, helpful, and confident
- If there's a chart, describe what it shows first

FORBIDDEN WORDS: mean, median, mode, std, deviation, null, NaN, DataFrame, pandas, correlation, regression, p-value, percentile, outlier, boolean, dtype, axis

Always end with one actionable suggestion when possible."""

STORYTELLER_PROMPT_TEMPLATE = """Dataset domain: {domain}
User's question: "{user_query}"

Analysis result:
{result}

{chart_context}

Explain this result in 3-5 friendly sentences. Include what it means for the business.
End with a practical suggestion if relevant."""


class DataStorytellerAgent:
    """
    Agent: data_storyteller
    The final layer of the pipeline — the "translator" from data to insight.
    Ensures zero technical leakage to the end user.
    """

    async def narrate(
        self,
        result: Any,
        user_query: str,
        domain: str = "general",
        chart_config: Optional[dict] = None,
        execution_error: Optional[str] = None,
    ) -> dict:
        """
        Convert analysis result into a human-friendly explanation.

        Args:
            result: The output from sandbox execution (number, dict, list, etc.)
            user_query: What the user originally asked
            domain: Dataset domain for context
            chart_config: Optional chart data for additional context
            execution_error: If there was an error, explain it gently

        Returns:
            dict with 'story' (str) and 'tone' (info|warning|error)
        """
        if execution_error:
            return self._gentle_error_response(user_query, execution_error)

        result_text = self._format_result(result)
        chart_context = ""
        if chart_config:
            chart_context = f"A chart was also created showing: {chart_config.get('title', 'data visualization')}"

        prompt = STORYTELLER_PROMPT_TEMPLATE.format(
            domain=domain,
            user_query=user_query,
            result=result_text,
            chart_context=chart_context,
        )

        try:
            response = await llm_client.chat_groq(
                prompt=prompt,
                model=settings.STORYTELLER_MODEL,
                system_prompt=STORYTELLER_SYSTEM_PROMPT,
                temperature=0.4,
                max_tokens=512,
            )

            story = response.content.strip()
            logger.info(f"Storyteller generated {len(story)} char explanation")

            return {
                "story": story,
                "tone": "info",
                "result_summary": self._extract_key_value(result),
            }

        except LLMError as e:
            logger.error(f"Storyteller agent failed: {e}")
            return self._fallback_story(result, user_query)

    def _format_result(self, result: Any) -> str:
        """Convert any result type to a readable text description."""
        if result is None:
            return "No result was produced."

        if isinstance(result, (int, float)):
            return f"The answer is: {result:,.2f}" if isinstance(result, float) else f"The answer is: {result:,}"

        if isinstance(result, str):
            return f"Result: {result}"

        if isinstance(result, dict):
            if len(result) <= 10:
                lines = [f"  {k}: {v}" for k, v in result.items()]
                return "Results:\n" + "\n".join(lines)
            return f"A summary table with {len(result)} items was generated."

        if isinstance(result, list):
            if len(result) <= 5:
                return "Results: " + ", ".join([str(i) for i in result])
            return f"A list of {len(result)} items was produced. Top 5: {result[:5]}"

        # pandas DataFrame (represented as dict after sandbox serialization)
        if hasattr(result, 'to_dict'):
            rows = len(result)
            cols = len(result.columns)
            return f"A table with {rows} rows and {cols} columns was produced."

        return str(result)[:500]

    def _extract_key_value(self, result: Any) -> Optional[str]:
        """Extract a single key metric if the result is a simple value."""
        if isinstance(result, (int, float, str)):
            return str(result)
        return None

    def _gentle_error_response(self, user_query: str, error: str) -> dict:
        """User-friendly error message when analysis fails."""
        story = (
            "We couldn't get a clear answer for that question right now. "
            "This might be because the data doesn't have quite what's needed, "
            "or the question could be phrased a bit differently. "
            "Try asking in a different way, or ask about specific column names."
        )
        return {
            "story": story,
            "tone": "warning",
            "result_summary": None,
        }

    def _fallback_story(self, result: Any, user_query: str) -> dict:
        """
        Produces a clean, readable English summary without any LLM.
        Used when API keys are missing or the LLM call fails.
        """
        import pandas as pd

        story = self._narrate_result(result, user_query)
        return {
            "story": story,
            "tone": "info",
            "result_summary": self._extract_key_value(result),
        }

    def _narrate_result(self, result: Any, query: str) -> str:
        """Turn any result into a readable sentence — no LLM needed."""
        import pandas as pd

        if result is None:
            return "The analysis ran but produced no output. Try asking in a different way."

        # Single number
        if isinstance(result, (int, float)):
            val = f"{result:,.2f}" if isinstance(result, float) else f"{result:,}"
            return f"The answer is {val}."

        # Short string
        if isinstance(result, str):
            if len(result) < 200:
                return result
            return result[:200] + "…"

        # List
        if isinstance(result, list):
            if not result:
                return "No results were found matching your query."
            if len(result) <= 5:
                return "The results are: " + ", ".join(str(v) for v in result) + "."
            return (f"Found {len(result):,} items. "
                    f"The first few are: {', '.join(str(v) for v in result[:5])}.")

        # Dict — the most common fallback result type
        if isinstance(result, dict):
            # Check if it's a nested summary dict
            if "rows" in result and "columns" in result:
                rows = result.get("rows", 0)
                cols = result.get("columns", [])
                n_cols = len(cols) if isinstance(cols, list) else cols
                missing = result.get("missing_values", {})
                parts = [f"Your dataset has {rows:,} rows and {n_cols} columns."]
                if missing:
                    cols_with_missing = list(missing.keys())[:3]
                    parts.append(f"Missing values were found in: {', '.join(cols_with_missing)}.")
                else:
                    parts.append("No missing values detected.")
                return " ".join(parts)

            if "max_value" in result:
                return f"The highest value is {result['max_value']:,}."
            if "min_value" in result:
                return f"The lowest value is {result['min_value']:,}."
            if "duplicate_rows" in result:
                d = result["duplicate_rows"]
                return (f"Found {d:,} duplicate rows ({result.get('duplicate_pct', 0)}% of data). "
                        + ("Consider removing them for cleaner analysis." if d > 0 else "Your data has no duplicates."))
            if "missing_count" in result:
                return f"There are {result['missing_count']:,} missing values ({result.get('missing_pct', 0)}%)."
            if "unique_count" in result:
                vals = result.get("values", [])[:5]
                return (f"There are {result['unique_count']:,} unique values. "
                        f"Examples: {', '.join(str(v) for v in vals)}.")

            # Generic dict — show top 5 key-value pairs in plain text
            items = list(result.items())[:5]
            parts = []
            for k, v in items:
                if isinstance(v, float):
                    parts.append(f"{k}: {v:,.2f}")
                elif isinstance(v, int):
                    parts.append(f"{k}: {v:,}")
                elif isinstance(v, dict):
                    parts.append(f"{k}: {len(v)} entries")
                else:
                    parts.append(f"{k}: {v}")
            suffix = f" (and {len(result)-5} more)" if len(result) > 5 else ""
            return "Here's what we found — " + ", ".join(parts) + suffix + "."

        # pandas Series
        if hasattr(result, "to_dict") and hasattr(result, "index"):
            d = result.to_dict()
            if not d:
                return "The result was empty."
            items = sorted(d.items(), key=lambda x: x[1] if isinstance(x[1], (int,float)) else 0, reverse=True)[:5]
            parts = [f"{k}: {v:,.2f}" if isinstance(v, float) else f"{k}: {v}" for k,v in items]
            return "Results — " + ", ".join(parts) + ("." if len(d) <= 5 else f", and {len(d)-5} more.")

        # pandas DataFrame
        if hasattr(result, "columns") and hasattr(result, "__len__"):
            rows = len(result)
            cols = list(result.columns)
            return (f"Here's a table with {rows:,} rows and {len(cols)} columns "
                    f"({', '.join(cols[:4])}{'…' if len(cols)>4 else ''}).")

        return f"Analysis complete. {str(result)[:150]}"


# Module-level singleton
storyteller_agent = DataStorytellerAgent()