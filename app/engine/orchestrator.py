from __future__ import annotations
"""
InsightEngine AI - Engine: Orchestrator
Controls the full multi-agent pipeline from upload to response.
Single point of coordination for all agents and modules.
"""
import asyncio
import logging
import json
import os
import uuid
from typing import Optional
import pandas as pd

from app.core.config import settings
from app.agents.domain_expert import domain_expert_agent
from app.agents.code_coder_a import coder_a_agent
from app.agents.code_coder_b import coder_b_agent
from app.agents.the_judge import judge_agent
from app.agents.data_storyteller import storyteller_agent
from app.cleaning.profiler import data_profiler
from app.cleaning.ai_cleaner import ai_cleaner_agent
from app.cleaning.executor import cleaning_executor
from app.cleaning.validator import cleaning_validator
from app.engine.sandbox_runner import sandbox_runner
from app.rag.rag_module import rag_module
from app.agents.rule_based_coder import rule_based_coder
from app.analysis.auto_dashboard import auto_dashboard
from app.analysis.insight_narrator import insight_narrator

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Master controller for InsightEngine AI.

    Pipeline A — Upload & Prepare:
      Upload → Schema Extract → Domain Detection → Profile →
      AI Clean Plan → Execute Cleaning → Build RAG Index

    Pipeline B — Query & Respond:
      Query → RAG Retrieve → [CoderA ‖ CoderB] → Judge →
      Sandbox Execute → (Fix if needed) → Storyteller → Response
    """

    # ─── PIPELINE A: Upload Processing ───────────────────────────────────────

    async def process_upload(
        self,
        df_raw: pd.DataFrame,
        dataset_id: str,
        original_filename: str,
    ) -> dict:
        """
        Full upload pipeline. Called after file validation passes.

        Returns:
            Enriched metadata dict with domain, profiling, cleaning results.
        """
        logger.info(f"[{dataset_id}] Starting upload pipeline for '{original_filename}'")
        result = {
            "dataset_id": dataset_id,
            "original_filename": original_filename,
            "status": "processing",
        }

        # ── Step 1: Save raw CSV ──────────────────────────────────────────────
        raw_path = settings.raw_csv_path(dataset_id)
        os.makedirs(settings.dataset_path(dataset_id), exist_ok=True)
        df_raw.to_csv(raw_path, index=False)
        logger.info(f"[{dataset_id}] Raw file saved: {raw_path}")

        # ── Step 2: Extract schema ────────────────────────────────────────────
        schema = self._extract_schema(df_raw)
        result.update(schema)

        # ── Step 3: Domain Detection ──────────────────────────────────────────
        try:
            domain_result = await domain_expert_agent.detect(
                columns=schema["columns"],
                sample_rows=schema["sample_rows"],
                dtypes=schema["dtypes"],
            )
            result["domain"] = domain_result.get("domain", "general")
            result["domain_details"] = domain_result
            logger.info(f"[{dataset_id}] Domain detected: {result['domain']}")
        except Exception as e:
            logger.warning(f"[{dataset_id}] Domain detection failed: {e}")
            result["domain"] = "general"
            result["domain_details"] = {}

        # ── Step 4: Data Profiling ─────────────────────────────────────────────
        profile_report = {}  # always defined even if profiling throws
        try:
            profile_report = data_profiler.profile(df_raw)
            result["profiling_report"] = profile_report
            result["quality_score"] = profile_report.get("quality_score", 100)
            result["quality_summary"] = data_profiler.user_friendly_summary(profile_report)
            logger.info(f"[{dataset_id}] Profiling done. Quality: {result['quality_score']}")
        except Exception as e:
            logger.error(f"[{dataset_id}] Profiling error: {e}")
            profile_report = {}
            result["profiling_report"] = {}
            result["quality_score"] = 100

        # ── Step 5: AI Cleaning Plan ──────────────────────────────────────────
        cleaning_plan = []  # always defined
        try:
            cleaning_plan = await ai_cleaner_agent.generate_plan(
                profile_report=profile_report,
                domain=result.get("domain", "general"),
            )
            result["cleaning_plan"] = cleaning_plan
            logger.info(f"[{dataset_id}] Cleaning plan: {len(cleaning_plan)} steps")
        except Exception as e:
            logger.warning(f"[{dataset_id}] Cleaning plan generation failed: {e}")
            cleaning_plan = []
            result["cleaning_plan"] = []

        # ── Step 6: Execute Cleaning ──────────────────────────────────────────
        try:
            df_clean, clean_report = cleaning_executor.execute(df_raw, cleaning_plan, profile_report)
            result["cleaning_report"] = clean_report
            result["cleaning_summary"] = cleaning_executor.cleaning_report_summary(clean_report)
            result["is_cleaned"] = True

            # Save cleaned CSV
            clean_path = settings.cleaned_csv_path(dataset_id)
            df_clean.to_csv(clean_path, index=False)
            logger.info(f"[{dataset_id}] Cleaned data saved: {clean_path}")
        except Exception as e:
            logger.error(f"[{dataset_id}] Cleaning execution failed: {e}")
            df_clean = df_raw.copy()
            result["is_cleaned"] = False
            result["cleaning_report"] = {}

        # ── Step 6b: Validate cleaning ────────────────────────────────────────
        try:
            validation_report = cleaning_validator.validate(
                df_raw=df_raw,
                df_clean=df_clean,
                cleaning_report=result.get("cleaning_report", {}),
            )
            result["validation_report"] = validation_report
            result["validation_summary"] = cleaning_validator.plain_english_summary(validation_report)
            result["quality_after"] = validation_report.get("quality_after", 0)
            logger.info(
                f"[{dataset_id}] Validation: "
                f"{validation_report.get('resolved_count',0)} resolved, "
                f"{validation_report.get('unresolved_count',0)} unresolved"
            )
        except Exception as e:
            logger.error(f"[{dataset_id}] Validation failed: {e}")
            result["validation_report"] = {}
            result["validation_summary"] = ""

        # ── Step 7: Build RAG Index ────────────────────────────────────────────
        try:
            rag_result = await rag_module.build_index(
                df=df_clean,
                dataset_id=dataset_id,
                domain=result.get("domain", "general"),
            )
            result["has_embeddings"] = True
            result["rag_info"] = rag_result
            logger.info(f"[{dataset_id}] RAG index built: {rag_result['chunks_count']} chunks")
        except Exception as e:
            logger.error(f"[{dataset_id}] RAG indexing failed: {e}")
            result["has_embeddings"] = False
            result["rag_info"] = {}

        # ── Step 8: Auto Dashboard ────────────────────────────────────────────
        try:
            dashboard = auto_dashboard.generate(
                df=df_clean,
                domain=result.get("domain", "general"),
                dataset_id=dataset_id,
            )
            # Attach plain-English insights to every tile
            try:
                dashboard = await insight_narrator.narrate(
                    dashboard=dashboard,
                    domain=result.get("domain", "general"),
                )
                logger.info(f"[{dataset_id}] Chart insights generated")
            except Exception as _ie:
                logger.warning(f"[{dataset_id}] Insight narration failed: {_ie}")

            result["dashboard"] = dashboard
            # Save dashboard JSON next to metadata
            import json as _json
            dash_path = os.path.join(settings.dataset_path(dataset_id), "dashboard.json")
            with open(dash_path, "w") as _f:
                _json.dump(dashboard, _f, default=str)
            logger.info(f"[{dataset_id}] Dashboard saved: {len(dashboard['tiles'])} tiles")
        except Exception as e:
            logger.error(f"[{dataset_id}] Dashboard generation failed: {e}")
            result["dashboard"] = None

        # ── Step 9: Save metadata ─────────────────────────────────────────────
        self._save_metadata(dataset_id, result)

        result["status"] = "ready"
        logger.info(f"[{dataset_id}] Upload pipeline complete ✓")
        return result

    # ─── PIPELINE B: Query Processing ────────────────────────────────────────

    async def process_query(
        self,
        user_query: str,
        dataset_id: str,
        conversation_history: Optional[list] = None,
    ) -> dict:
        """
        Full query pipeline. Called when user sends a message.

        Returns:
            dict with story (user-friendly response), result, chart_config, etc.
        """
        logger.info(f"[{dataset_id}] Processing query: '{user_query[:80]}...'")

        # ── Load cleaned dataset ───────────────────────────────────────────────
        df = self._load_dataset(dataset_id)
        if df is None:
            return self._error_response("We couldn't find your dataset. Please re-upload it.")

        # ── Load metadata ──────────────────────────────────────────────────────
        metadata = self._load_metadata(dataset_id)
        domain = metadata.get("domain", "general")
        columns = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in columns}
        sample_rows = df.head(5).to_dict(orient="records")

        # ── Step 8: RAG Retrieval ──────────────────────────────────────────────
        try:
            retrieved_chunks = rag_module.retrieve(
                query=user_query,
                dataset_id=dataset_id,
                top_k=settings.TOP_K_CHUNKS,
            )
            rag_context = rag_module.format_context(retrieved_chunks)
            logger.info(f"[{dataset_id}] RAG retrieved {len(retrieved_chunks)} chunks")
        except Exception as e:
            logger.warning(f"[{dataset_id}] RAG retrieval failed: {e}")
            rag_context = ""
            retrieved_chunks = []

        # ── Steps 9+10: CoderA and CoderB in parallel ─────────────────────────
        coder_input = dict(
            user_query=user_query,
            columns=columns,
            dtypes=dtypes,
            sample_rows=sample_rows,
            domain=domain,
            rag_context=rag_context,
        )

        coder_a_result, coder_b_result = await asyncio.gather(
            coder_a_agent.generate(**coder_input),
            coder_b_agent.generate(**coder_input),
            return_exceptions=True,
        )

        # Handle exceptions from parallel calls
        if isinstance(coder_a_result, Exception):
            coder_a_result = {"code": "", "success": False, "error": str(coder_a_result)}
        if isinstance(coder_b_result, Exception):
            coder_b_result = {"code": "", "success": False, "error": str(coder_b_result)}

        code_a = coder_a_result.get("code", "")
        code_b = coder_b_result.get("code", "")

        logger.info(f"[{dataset_id}] CoderA={'ok' if code_a else 'empty'}, CoderB={'ok' if code_b else 'empty'}")

        # ── Step 10: Judge selects best code ──────────────────────────────────
        try:
            judge_result = await judge_agent.judge(
                code_a=code_a,
                code_b=code_b,
                user_query=user_query,
                columns=columns,
                domain=domain,
            )
            final_code = judge_result.get("final_code", code_a or code_b)
        except Exception as e:
            logger.error(f"[{dataset_id}] Judge failed: {e}. Using CoderA fallback.")
            judge_result = {"selected": "A", "reasoning": "fallback", "fixes_applied": []}
            final_code = code_a or code_b

        if not final_code:
            logger.info(f"[{dataset_id}] Both LLM coders empty, using rule-based fallback")
            rb = rule_based_coder.generate(
                user_query=user_query,
                columns=columns,
                dtypes=dtypes,
                sample_rows=sample_rows,
            )
            final_code = rb.get("code", "")
            judge_result = {"selected": "rule_based", "reasoning": "LLM unavailable", "fixes_applied": []}

        if not final_code:
            return self._error_response("We couldn't generate the analysis code. Please rephrase your question.")

        # ── Step 11: Sandbox execution ────────────────────────────────────────
        sandbox_result = sandbox_runner.run(code=final_code, df=df)

        # ── Step 11b: Fix if sandbox errored ─────────────────────────────────
        if not sandbox_result.success and sandbox_result.error_type == "runtime_error":
            logger.info(f"[{dataset_id}] Sandbox failed, asking Judge to fix...")
            fix_result = await judge_agent.fix_code(
                code=final_code,
                error=sandbox_result.error,
                user_query=user_query,
                columns=columns,
            )
            if fix_result["success"]:
                final_code = fix_result["fixed_code"]
                sandbox_result = sandbox_runner.run(code=final_code, df=df)
                logger.info(f"[{dataset_id}] Re-execution after fix: success={sandbox_result.success}")

        # ── Step 12: Storyteller explains result ──────────────────────────────
        sandbox_dict = sandbox_result.to_dict()
        story_result = await storyteller_agent.narrate(
            result=sandbox_dict.get("result"),
            user_query=user_query,
            domain=domain,
            chart_config=sandbox_dict.get("chart_config"),
            execution_error=sandbox_dict.get("error") if not sandbox_result.success else None,
        )

        # ── Step 13: Build final response ─────────────────────────────────────
        response = {
            "success": sandbox_result.success,
            "story": story_result["story"],
            "tone": story_result["tone"],
            "result": sandbox_dict.get("result"),
            "result_summary": story_result.get("result_summary"),
            "output_type": sandbox_dict.get("output_type"),
            "chart_config": sandbox_dict.get("chart_config"),
            "execution_time_ms": sandbox_dict.get("execution_time_ms"),
            "debug": {
                "code_used": final_code,
                "judge_selected": judge_result.get("selected"),
                "rag_chunks_count": len(retrieved_chunks),
            } if settings.DEBUG else {},
        }

        logger.info(f"[{dataset_id}] Query complete. Success={response['success']}")
        return response

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _extract_schema(self, df: pd.DataFrame) -> dict:
        """Extract schema information from DataFrame."""
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "sample_rows": df.head(5).to_dict(orient="records"),
        }

    def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load cleaned CSV (falls back to raw if cleaned not found)."""
        clean_path = settings.cleaned_csv_path(dataset_id)
        raw_path = settings.raw_csv_path(dataset_id)

        if os.path.exists(clean_path):
            return pd.read_csv(clean_path)
        elif os.path.exists(raw_path):
            logger.warning(f"Cleaned CSV not found for {dataset_id}, using raw")
            return pd.read_csv(raw_path)
        return None

    def _save_metadata(self, dataset_id: str, metadata: dict) -> None:
        """Persist metadata to JSON file, handling all non-serializable types."""
        import math
        import pandas as pd
        import numpy as np

        def _safe(obj):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return {k: _safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_safe(v) for v in obj]
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat() if not pd.isnull(obj) else None
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return None if math.isnan(float(obj)) else float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

        path = settings.metadata_path(dataset_id)
        with open(path, "w") as f:
            json.dump(_safe(metadata), f, indent=2)

    def _load_metadata(self, dataset_id: str) -> dict:
        """Load metadata JSON."""
        path = settings.metadata_path(dataset_id)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _error_response(self, message: str) -> dict:
        return {
            "success": False,
            "story": message,
            "tone": "error",
            "result": None,
            "chart_config": None,
        }


# Module-level singleton
orchestrator = Orchestrator()