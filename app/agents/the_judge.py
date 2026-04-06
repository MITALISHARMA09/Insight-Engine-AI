from __future__ import annotations
"""
InsightEngine AI - Agent: The Judge (the_judge)
Compares CoderA and CoderB outputs, selects the best,
fixes any bugs, and outputs final production-ready code.
"""
import logging
import json
import re
from typing import Optional

from app.core.llm_client import llm_client, LLMError
from app.core.config import settings

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a senior Python code reviewer and fixer.
You receive two pandas code attempts (A and B) for the same task.

Your job:
1. Evaluate which code is more correct, safe, and efficient
2. Select the better one OR merge the best parts of both
3. Fix any bugs, syntax errors, or unsafe patterns
4. Ensure the final code is 100% executable

SAFETY RULES (must enforce):
- DataFrame variable is `df` — no file loading
- Final answer must be in `result`
- No os, sys, subprocess, open(), exec(), eval()
- No more than 30 lines

Return a JSON object ONLY:
{
  "selected": "A" or "B" or "merged",
  "reasoning": "brief explanation of your choice",
  "final_code": "the corrected, final Python code",
  "fixes_applied": ["list of fixes made"]
}
"""

JUDGE_PROMPT_TEMPLATE = """User Question: {user_query}

Dataset columns: {columns}
Dataset domain: {domain}

--- CODE FROM CODER A ---
{code_a}

--- CODE FROM CODER B ---
{code_b}

{error_context}

Review both, pick/fix the best, return JSON with final_code."""

FIX_SYSTEM_PROMPT = """You are a Python debugging expert.
Fix the provided pandas code so it runs correctly.

Rules:
- df is the DataFrame (already loaded, do not reload)
- Store final answer in `result`
- No unsafe imports (os, sys, subprocess, exec, eval)
- Return ONLY the fixed Python code (no markdown, no explanation)
"""

FIX_PROMPT_TEMPLATE = """The following pandas code failed with this error:

Error: {error}

Code that failed:
{code}

Dataset columns: {columns}
User question: {user_query}

Fix the code and return ONLY the corrected Python code."""


class JudgeAgent:
    """
    Agent: the_judge
    Central quality controller in the multi-agent code pipeline.
    Ensures only safe, correct code reaches the sandbox.
    """

    async def judge(
        self,
        code_a: str,
        code_b: str,
        user_query: str,
        columns: list[str],
        domain: str = "general",
        execution_error_a: Optional[str] = None,
        execution_error_b: Optional[str] = None,
    ) -> dict:
        """
        Compare CoderA and CoderB outputs and produce final code.

        Returns:
            dict with 'final_code', 'selected', 'reasoning', 'fixes_applied'
        """
        error_context = ""
        if execution_error_a:
            error_context += f"Note: Code A had execution error: {execution_error_a}\n"
        if execution_error_b:
            error_context += f"Note: Code B had execution error: {execution_error_b}\n"

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            user_query=user_query,
            columns=", ".join(columns),
            domain=domain,
            code_a=code_a or "# CoderA produced no code",
            code_b=code_b or "# CoderB produced no code",
            error_context=error_context,
        )

        try:
            response = await llm_client.chat_groq(
                prompt=prompt,
                model=settings.JUDGE_MODEL,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1536,
            )

            result = self._parse_judge_response(response.content)
            logger.info(f"Judge selected: {result.get('selected')} | fixes: {result.get('fixes_applied')}")
            return result

        except (LLMError, ValueError) as e:
            logger.error(f"Judge agent failed: {e}. Falling back to CoderA.")
            # Fallback: use CoderA if available, else CoderB
            fallback_code = code_a if code_a else code_b
            return {
                "final_code": fallback_code,
                "selected": "A" if code_a else "B",
                "reasoning": "Judge unavailable, used fallback code",
                "fixes_applied": [],
                "fallback": True,
            }

    async def fix_code(
        self,
        code: str,
        error: str,
        user_query: str,
        columns: list[str],
    ) -> dict:
        """
        Fix code that failed during sandbox execution.
        Called when sandbox returns an error.

        Returns:
            dict with 'fixed_code', 'success'
        """
        prompt = FIX_PROMPT_TEMPLATE.format(
            error=error,
            code=code,
            columns=", ".join(columns),
            user_query=user_query,
        )

        try:
            response = await llm_client.chat_groq(
                prompt=prompt,
                model=settings.JUDGE_MODEL,
                system_prompt=FIX_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            )
            fixed_code = self._clean_code(response.content)
            logger.info("Judge fixed execution error successfully")
            return {"fixed_code": fixed_code, "success": True}

        except LLMError as e:
            logger.error(f"Judge fix attempt failed: {e}")
            return {"fixed_code": code, "success": False}

    def _parse_judge_response(self, content: str) -> dict:
        """
        Parse the Judge's JSON response robustly.
        Handles: markdown fences, leading/trailing text, multiple JSON blocks,
        and truncated responses from decommissioned models.
        """
        content = content.strip()

        # Strip ``` fences (```json ... ``` or ``` ... ```)
        content = re.sub(r'```(?:json)?\s*', '', content).replace('```', '').strip()

        # Try direct parse first
        try:
            data = json.loads(content)
            return self._validate_judge_data(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Find the FIRST complete JSON object in the text
        # Walk char-by-char to find balanced braces — handles "Extra data" errors
        start = content.find('{')
        if start != -1:
            depth = 0
            for i, ch in enumerate(content[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = content[start:i+1]
                        try:
                            data = json.loads(candidate)
                            return self._validate_judge_data(data)
                        except (json.JSONDecodeError, ValueError):
                            break

        raise ValueError("Judge returned unparseable response")

    def _validate_judge_data(self, data: dict) -> dict:
        """Validate required fields and set defaults."""
        if "final_code" not in data:
            raise ValueError("Judge response missing 'final_code'")
        if "selected" not in data:
            data["selected"] = "unknown"
        data.setdefault("reasoning", "")
        data.setdefault("fixes_applied", [])
        return data

    def _clean_code(self, raw: str) -> str:
        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) >= 2 else raw
            if raw.startswith("python"):
                raw = raw[6:]
        return raw.strip()


# Module-level singleton
judge_agent = JudgeAgent()