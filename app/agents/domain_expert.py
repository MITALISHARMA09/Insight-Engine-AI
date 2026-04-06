"""
InsightEngine AI - Agent: Domain Expert
Detects the dataset's business domain and intent using LLM.
e.g. "sales", "HR", "finance", "healthcare", "retail"
"""
import json
import logging
from typing import Optional

from app.core.llm_client import llm_client, LLMError
from app.core.config import settings

logger = logging.getLogger(__name__)

DOMAIN_EXPERT_SYSTEM_PROMPT = """You are a data domain expert. 
You receive a list of column names and sample data from a dataset.
Your ONLY job is to identify the business domain and the likely purpose of the dataset.

Respond ONLY with a valid JSON object in this exact format (no extra text):
{
  "domain": "string (e.g. sales, HR, finance, healthcare, retail, education, logistics, marketing)",
  "sub_domain": "string (more specific, e.g. employee_payroll, product_sales, patient_records)",
  "confidence": 0.0-1.0,
  "likely_questions": ["question 1", "question 2", "question 3"],
  "key_metrics": ["metric1", "metric2", "metric3"],
  "reasoning": "brief explanation in 1-2 sentences"
}
"""

DOMAIN_EXPERT_PROMPT_TEMPLATE = """Analyze this dataset and identify its domain.

Dataset columns: {columns}

Sample data (first 3 rows):
{sample_rows}

Identify the domain and return only a JSON object."""


class DomainExpertAgent:
    """
    Agent: domain_expert
    Analyzes schema + sample rows to classify the dataset domain.
    Used during upload pipeline to contextualize the dataset.
    """

    async def detect(
        self,
        columns: list[str],
        sample_rows: list[dict],
        dtypes: Optional[dict] = None,
    ) -> dict:
        """
        Detects the dataset domain.

        Args:
            columns: List of column names
            sample_rows: First few rows as list of dicts
            dtypes: Optional column dtype mapping

        Returns:
            dict with domain, sub_domain, confidence, likely_questions, key_metrics
        """
        prompt = DOMAIN_EXPERT_PROMPT_TEMPLATE.format(
            columns=", ".join(columns),
            sample_rows=json.dumps(sample_rows[:3], indent=2, default=str),
        )

        try:
            response = await llm_client.chat_groq(
                prompt=prompt,
                model=settings.DOMAIN_EXPERT_MODEL,
                system_prompt=DOMAIN_EXPERT_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=512,
            )

            result = self._parse_response(response.content)
            logger.info(f"Domain detected: {result.get('domain')} (confidence={result.get('confidence')})")
            return result

        except LLMError as e:
            logger.warning(f"Domain expert LLM failed: {e}. Using fallback.")
            return self._fallback_domain(columns)

    def _parse_response(self, content: str) -> dict:
        """Parse and validate JSON from LLM response."""
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON block
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError("No valid JSON found in LLM response")

        # Validate required fields
        required = ["domain", "confidence"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Ensure defaults for optional fields
        data.setdefault("sub_domain", data["domain"])
        data.setdefault("likely_questions", [])
        data.setdefault("key_metrics", [])
        data.setdefault("reasoning", "")

        return data

    def _fallback_domain(self, columns: list[str]) -> dict:
        """
        Rule-based fallback when LLM is unavailable.
        Keyword matching on column names.
        """
        columns_lower = [c.lower() for c in columns]
        col_str = " ".join(columns_lower)

        if any(k in col_str for k in ["salary", "employee", "hire", "department", "position"]):
            domain = "HR"
        elif any(k in col_str for k in ["sales", "revenue", "product", "customer", "order"]):
            domain = "sales"
        elif any(k in col_str for k in ["price", "stock", "profit", "loss", "balance", "expense"]):
            domain = "finance"
        elif any(k in col_str for k in ["patient", "diagnosis", "treatment", "hospital", "age", "bmi"]):
            domain = "healthcare"
        elif any(k in col_str for k in ["student", "grade", "score", "exam", "course"]):
            domain = "education"
        else:
            domain = "general"

        return {
            "domain": domain,
            "sub_domain": domain,
            "confidence": 0.4,
            "likely_questions": [
                "What are the key trends in this dataset?",
                "Which values are highest?",
                "Are there any anomalies?",
            ],
            "key_metrics": columns[:3],
            "reasoning": "Determined by keyword matching (LLM unavailable).",
        }


# Module-level singleton
domain_expert_agent = DomainExpertAgent()