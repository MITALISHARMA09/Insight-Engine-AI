from __future__ import annotations
"""
InsightEngine AI - Agent: CoderB (code_coder_b)
Alternative pandas code generation using OpenRouter/Mistral.
Runs in parallel with CoderA for diversity of approaches.
"""
import logging
from typing import Optional

from app.core.llm_client import llm_client, LLMError
from app.core.config import settings

logger = logging.getLogger(__name__)

CODER_B_SYSTEM_PROMPT = """You are a Python data analysis expert specializing in pandas.
Your task is to write clean, correct pandas code to answer a business question.

MANDATORY RULES:
- DataFrame is already available as variable `df` — do NOT load or read any files
- Store your final answer in variable: `result`
- `result` must be: a number, string, list, dict, or pandas DataFrame
- ONLY allowed imports: pandas as pd, numpy as np, math, json
- FORBIDDEN: os, sys, subprocess, open(), exec(), eval(), file operations
- Write readable code with inline comments
- Maximum 25 lines of code

Return ONLY Python code. No markdown. No explanation."""

CODER_B_PROMPT_TEMPLATE = """Dataset Information:
- Domain: {domain}
- Columns: {columns}
- Data types: {dtypes}
- Sample data: {sample_rows}

Context from dataset (retrieved facts):
{rag_context}

Question to answer: {user_query}

Write pandas code. Put the final answer in `result`."""


class CoderBAgent:
    """
    Agent: code_coder_b
    Generates pandas code via OpenRouter (Mistral model).
    Provides a second perspective for The Judge to compare.
    """

    async def generate(
        self,
        user_query: str,
        columns: list[str],
        dtypes: dict,
        sample_rows: list[dict],
        domain: str = "general",
        rag_context: str = "",
    ) -> dict:
        """
        Generate pandas code as alternative to CoderA.

        Returns:
            dict with 'code', 'success', 'error', 'agent', 'model'
        """
        prompt = CODER_B_PROMPT_TEMPLATE.format(
            domain=domain,
            columns=", ".join(columns),
            dtypes=str(dtypes),
            sample_rows=self._format_sample(sample_rows),
            rag_context=rag_context or "No additional context.",
            user_query=user_query,
        )

        # Use OpenRouter if key is set, otherwise fall back to Groq
        # with a different temperature to get code diversity
        use_openrouter = settings.openrouter_configured

        try:
            if use_openrouter:
                response = await llm_client.chat_openrouter(
                    prompt=prompt,
                    model=settings.CODER_B_MODEL,
                    system_prompt=CODER_B_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_tokens=1024,
                )
                model_used = settings.CODER_B_MODEL
            else:
                # Groq fallback: use a slightly higher temperature for diversity
                logger.info("CoderB: OpenRouter key not set — using Groq fallback")
                response = await llm_client.chat_groq(
                    prompt=prompt,
                    model=settings.CODER_A_MODEL,
                    system_prompt=CODER_B_SYSTEM_PROMPT,
                    temperature=0.35,   # higher than CoderA for different output
                    max_tokens=1024,
                )
                model_used = settings.CODER_A_MODEL + " (groq-fallback)"

            code = self._clean_code(response.content)
            logger.info(f"CoderB generated {len(code.splitlines())} lines of code")

            return {
                "code": code,
                "success": True,
                "error": None,
                "agent": "coder_b",
                "model": model_used,
            }

        except LLMError as e:
            logger.error(f"CoderB failed: {e}")
            return {
                "code": "",
                "success": False,
                "error": str(e),
                "agent": "coder_b",
                "model": settings.CODER_B_MODEL,
            }

    def _clean_code(self, raw: str) -> str:
        """Remove markdown fences and normalize whitespace."""
        raw = raw.strip()
        if "```" in raw:
            # Extract content between fences
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1]
                if raw.startswith("python"):
                    raw = raw[6:]
            elif len(parts) >= 2:
                raw = parts[1]
        return raw.strip()

    def _format_sample(self, sample_rows: list[dict], max_rows: int = 3) -> str:
        if not sample_rows:
            return "No sample data"
        return "\n".join([f"  {r}" for r in sample_rows[:max_rows]])


# Module-level singleton
coder_b_agent = CoderBAgent()