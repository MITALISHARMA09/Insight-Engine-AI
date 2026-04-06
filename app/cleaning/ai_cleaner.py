"""
InsightEngine AI - Cleaning Module: AI Cleaner
Uses LLM to generate an intelligent JSON cleaning plan based on profiling report.
The cleaning plan is validated before execution — LLM only decides, Python executes.
"""
import logging
import json
import re
from typing import Optional

from app.core.llm_client import llm_client, LLMError
from app.core.config import settings

logger = logging.getLogger(__name__)

AI_CLEANER_SYSTEM_PROMPT = """You are a data cleaning expert. 
You receive a data quality profile report and must produce a JSON cleaning plan.

Your cleaning plan MUST:
- Be a valid JSON array of action objects
- Each action must use ONLY these allowed actions:
  * fill_missing: fill missing values in a column
  * remove_duplicates: remove exact duplicate rows
  * drop_column: remove a useless column
  * fix_dtype: convert column to correct type (numeric or datetime)
  * cap_outliers: cap extreme values at bounds (winsorize)
  * drop_missing_rows: drop rows where missing > threshold

Allowed methods for fill_missing: "median", "mean", "mode", "zero", "ffill", "bfill", "drop"

Return ONLY a valid JSON array. No explanation. No markdown.

Example output:
[
  {"action": "fill_missing", "column": "age", "method": "median"},
  {"action": "fill_missing", "column": "city", "method": "mode"},
  {"action": "remove_duplicates"},
  {"action": "fix_dtype", "column": "salary", "target_type": "numeric"},
  {"action": "cap_outliers", "column": "salary", "lower": 5000, "upper": 200000}
]
"""

AI_CLEANER_PROMPT_TEMPLATE = """Dataset domain: {domain}
Dataset shape: {rows} rows × {cols} columns

Data Quality Report:
{profile_summary}

Based on this report, generate a minimal but effective cleaning plan.
Only include actions where there is a real issue to fix.
Return a JSON array of cleaning actions."""


class AICleanerAgent:
    """
    Agent: ai_cleaner
    Generates intelligent cleaning plans using LLM reasoning.
    Works alongside rule-based cleaning for hybrid approach.
    """

    async def generate_plan(
        self,
        profile_report: dict,
        domain: str = "general",
    ) -> list[dict]:
        """
        Generate a JSON cleaning plan from the profiling report.

        Returns:
            List of cleaning action dicts (validated)
        """
        profile_summary = self._summarize_profile(profile_report)

        prompt = AI_CLEANER_PROMPT_TEMPLATE.format(
            domain=domain,
            rows=profile_report.get("total_rows", 0),
            cols=profile_report.get("total_columns", 0),
            profile_summary=profile_summary,
        )

        try:
            response = await llm_client.chat_groq(
                prompt=prompt,
                model=settings.CLEANER_MODEL,
                system_prompt=AI_CLEANER_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            )

            raw_plan = self._parse_plan(response.content)
            validated_plan = self._validate_plan(raw_plan)

            logger.info(f"AI Cleaner generated {len(validated_plan)} cleaning steps")
            return validated_plan

        except (LLMError, ValueError) as e:
            logger.warning(f"AI Cleaner LLM failed: {e}. Using rule-based fallback.")
            return self._rule_based_plan(profile_report)

    def _summarize_profile(self, report: dict) -> str:
        """Convert profile report to compact text for LLM."""
        lines = []

        missing = report.get("missing_values", {})
        if missing:
            for col, info in missing.items():
                lines.append(f"MISSING: '{col}' has {info['count']} missing values ({info['percentage']}%), dtype={info['dtype']}")

        dupes = report.get("duplicates", {})
        if dupes.get("has_duplicates"):
            lines.append(f"DUPLICATES: {dupes['count']} duplicate rows found ({dupes['percentage']}%)")

        outliers = report.get("outliers", {})
        if outliers:
            for col, info in outliers.items():
                lines.append(f"OUTLIERS: '{col}' has {info['count']} extreme values, range [{info['lower_bound']}, {info['upper_bound']}]")

        type_issues = report.get("type_issues", {})
        if type_issues:
            for col, info in type_issues.items():
                lines.append(f"TYPE ISSUE: '{col}' looks like {info['likely_type']} but stored as {info['current_type']}")

        constants = report.get("constant_columns", [])
        if constants:
            lines.append(f"CONSTANT COLUMNS (no value for analysis): {constants}")

        return "\n".join(lines) if lines else "No significant issues detected."

    def _parse_plan(self, content: str) -> list:
        """Parse JSON array from LLM response."""
        content = content.strip()

        # Strip markdown fences
        if "```" in content:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            if match:
                content = match.group(1).strip()

        # Find JSON array
        array_match = re.search(r'\[.*\]', content, re.DOTALL)
        if array_match:
            content = array_match.group()

        return json.loads(content)

    def _validate_plan(self, plan: list) -> list:
        """
        Validate each action in the cleaning plan.
        Remove invalid or dangerous actions.
        """
        ALLOWED_ACTIONS = {
            "fill_missing", "remove_duplicates", "drop_column",
            "fix_dtype", "cap_outliers", "drop_missing_rows"
        }
        ALLOWED_FILL_METHODS = {"median", "mean", "mode", "zero", "ffill", "bfill", "drop"}
        ALLOWED_DTYPES = {"numeric", "datetime", "string"}

        validated = []
        for action in plan:
            if not isinstance(action, dict):
                continue

            action_type = action.get("action")
            if action_type not in ALLOWED_ACTIONS:
                logger.warning(f"Skipping invalid action: {action_type}")
                continue

            # Validate fill_missing
            if action_type == "fill_missing":
                if "column" not in action:
                    continue
                method = action.get("method", "median")
                if method not in ALLOWED_FILL_METHODS:
                    action["method"] = "median"

            # Validate fix_dtype
            if action_type == "fix_dtype":
                if "column" not in action:
                    continue
                target = action.get("target_type", "")
                if target not in ALLOWED_DTYPES:
                    continue

            # Validate cap_outliers
            if action_type == "cap_outliers":
                if "column" not in action:
                    continue
                # Ensure bounds are numbers
                if "lower" in action and not isinstance(action["lower"], (int, float)):
                    del action["lower"]
                if "upper" in action and not isinstance(action["upper"], (int, float)):
                    del action["upper"]

            validated.append(action)

        return validated

    def _rule_based_plan(self, report: dict) -> list:
        """
        Fallback: generate a safe rule-based cleaning plan
        when LLM is unavailable.
        """
        plan = []

        # Remove duplicates if found
        if report.get("duplicates", {}).get("has_duplicates"):
            plan.append({"action": "remove_duplicates"})

        # Fill missing values based on dtype
        for col, info in report.get("missing_values", {}).items():
            dtype = info.get("dtype", "object")
            if "float" in dtype or "int" in dtype:
                plan.append({"action": "fill_missing", "column": col, "method": "median"})
            elif info["percentage"] > 50:
                plan.append({"action": "drop_missing_rows", "column": col, "threshold": 0.5})
            else:
                plan.append({"action": "fill_missing", "column": col, "method": "mode"})

        # Fix type issues
        for col, info in report.get("type_issues", {}).items():
            if info["likely_type"] == "numeric":
                plan.append({"action": "fix_dtype", "column": col, "target_type": "numeric"})

        # Drop constant columns
        for col in report.get("constant_columns", []):
            plan.append({"action": "drop_column", "column": col})

        return plan


# Module-level singleton
ai_cleaner_agent = AICleanerAgent()