"""
InsightEngine AI - Agent: CoderA (code_coder_a)
Fast pandas code generation using Groq/Llama.
First code attempt in the dual-coder pipeline.
"""
import logging
import re
from typing import Optional

from app.core.llm_client import llm_client, LLMError
from app.core.config import settings

logger = logging.getLogger(__name__)

CODER_A_SYSTEM_PROMPT = """You are a Python pandas expert. 
Generate SAFE, executable pandas code to answer the user's question.

STRICT RULES:
- Only use: pandas, numpy, json, math (no other imports)
- NO file deletion, NO os commands, NO subprocess, NO exec/eval
- The DataFrame is already loaded as variable: df
- Store your FINAL result in a variable called: result
- result must be one of: a number, a string, a dict, a list, or a pandas DataFrame
- If creating a chart, store chart config as: chart_config (a dict)
- Add a brief comment above each major step
- Keep code concise (under 30 lines)

Output ONLY the Python code. No explanation, no markdown fences.
"""

CODER_A_PROMPT_TEMPLATE = """Dataset domain: {domain}
Dataset columns: {columns}
Column dtypes: {dtypes}
Sample rows:
{sample_rows}

RAG Context (relevant dataset facts):
{rag_context}

User Question: {user_query}

Generate pandas code to answer this question. Store the answer in `result`."""


class CoderAAgent:
    """
    Agent: code_coder_a
    Generates pandas analysis code using Groq (fast inference).
    Part of the parallel dual-coder system.
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
        Generate pandas code for the user query.

        Returns:
            dict with 'code' (str), 'success' (bool), 'error' (str|None)
        """
        prompt = CODER_A_PROMPT_TEMPLATE.format(
            domain=domain,
            columns=", ".join(columns),
            dtypes=str(dtypes),
            sample_rows=self._format_sample(sample_rows),
            rag_context=rag_context or "No additional context available.",
            user_query=user_query,
        )

        try:
            response = await llm_client.chat_groq(
                prompt=prompt,
                model=settings.CODER_A_MODEL,
                system_prompt=CODER_A_SYSTEM_PROMPT,
                temperature=0.15,
                max_tokens=1024,
            )

            code = self._clean_code(response.content)
            logger.info(f"CoderA generated {len(code.splitlines())} lines of code")

            return {
                "code": code,
                "success": True,
                "error": None,
                "agent": "coder_a",
                "model": settings.CODER_A_MODEL,
            }

        except LLMError as e:
            logger.error(f"CoderA failed: {e}")
            return {
                "code": "",
                "success": False,
                "error": str(e),
                "agent": "coder_a",
                "model": settings.CODER_A_MODEL,
            }

    def _clean_code(self, raw: str) -> str:
        """Strip markdown fences and whitespace from LLM output."""
        raw = raw.strip()

        # Remove ```python ... ``` or ``` ... ```
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first line (```python) and last line (```)
            raw = "\n".join(lines[1:] if lines[-1].strip() == "```" else lines[1:])
            if raw.endswith("```"):
                raw = raw[:-3].strip()

        return raw.strip()

    def _format_sample(self, sample_rows: list[dict], max_rows: int = 3) -> str:
        """Format sample rows as readable text."""
        if not sample_rows:
            return "No sample data available"
        rows = sample_rows[:max_rows]
        lines = []
        for i, row in enumerate(rows):
            lines.append(f"Row {i+1}: {row}")
        return "\n".join(lines)

    def validate_code_safety(self, code: str) -> tuple[bool, str]:
        """
        Quick static check for dangerous patterns.
        Returns (is_safe, reason).
        """
        dangerous_patterns = [
            (r'\bos\b', "os module not allowed"),
            (r'\bsubprocess\b', "subprocess not allowed"),
            (r'\bopen\s*\(', "file open not allowed"),
            (r'\bexec\s*\(', "exec not allowed"),
            (r'\beval\s*\(', "eval not allowed"),
            (r'__import__', "__import__ not allowed"),
            (r'shutil', "shutil not allowed"),
            (r'\.remove\s*\(', "file removal not allowed"),
            (r'import\s+os', "os import not allowed"),
            (r'import\s+sys', "sys import not allowed"),
        ]

        for pattern, reason in dangerous_patterns:
            if re.search(pattern, code):
                return False, reason

        return True, ""


# Module-level singleton
coder_a_agent = CoderAAgent()