from __future__ import annotations
"""
InsightEngine AI - Core Configuration

HOW TO SET API KEYS (never hardcode them):
1. Create a file called  .env  in the same folder as main.py
2. Add your keys like this (no quotes, no spaces around =):

     GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
     OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx

3. Save the file and restart the server — keys are read automatically.

Get keys:
  Groq (free, no credit card): https://console.groq.com/keys
  OpenRouter (optional):        https://openrouter.ai/keys
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
import os


class Settings(BaseSettings):
    # ─── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "InsightEngine AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"

    # ─── API Keys ───────────────────────────────────────────────────────────
    # Leave blank here — set them in your .env file instead
    GROQ_API_KEY: str = Field(default="", description="Set in .env: GROQ_API_KEY=gsk_...")
    OPENROUTER_API_KEY: str = Field(default="", description="Set in .env: OPENROUTER_API_KEY=sk-or-...")

    # ─── LLM Models ─────────────────────────────────────────────────────────
    #
    # AGENT → PROVIDER MAPPING (important — wrong provider = 400 error):
    #
    #   domain_expert  → chat_groq()       → DOMAIN_EXPERT_MODEL
    #   code_coder_a   → chat_groq()       → CODER_A_MODEL
    #   code_coder_b   → chat_openrouter() → CODER_B_MODEL  (falls back to Groq)
    #   the_judge      → chat_groq()       → JUDGE_MODEL
    #   data_storyteller → chat_groq()     → STORYTELLER_MODEL
    #   ai_cleaner     → chat_groq()       → CLEANER_MODEL
    #   insight_narrator → chat_groq()     → STORYTELLER_MODEL
    #
    # Groq model names:  use exactly as written below (no slashes, no :free)
    # OpenRouter names:  use "provider/model:free" format
    #
    # Current stable FREE Groq models (April 2026):
    #   llama-3.3-70b-versatile   — best quality, all-round (70B)
    #   llama-3.1-8b-instant      — fastest, cheapest (8B), good for simple tasks
    #   gemma2-9b-it               — Google Gemma 9B, reliable
    #   mixtral-8x7b-32768         — good at code and structured JSON
    #
    # Current stable FREE OpenRouter models (for CoderB only):
    #   deepseek/deepseek-r1:free           — excellent at code & reasoning
    #   meta-llama/llama-3.3-70b-instruct:free  — Llama 70B via OpenRouter
    #   qwen/qwen-2.5-7b-instruct:free      — fast, good at structured tasks
    #
    # ── Groq agents ──────────────────────────────────────────────────────────
    CODER_A_MODEL: str      = "llama-3.3-70b-versatile"   # best code quality
    JUDGE_MODEL: str        = "llama-3.3-70b-versatile"   # needs strong reasoning
    STORYTELLER_MODEL: str  = "llama-3.1-8b-instant"      # simple text, fast
    DOMAIN_EXPERT_MODEL: str = "llama-3.1-8b-instant"     # JSON classification, fast
    CLEANER_MODEL: str      = "llama-3.1-8b-instant"      # structured JSON plan, fast

    # ── OpenRouter agent (CoderB — alternative code perspective) ─────────────
    # If OPENROUTER_API_KEY is blank, CoderB automatically falls back to Groq
    CODER_B_MODEL: str      = "meta-llama/llama-3.3-70b-instruct:free"  # great at code

    # ─── Base URLs ───────────────────────────────────────────────────────────
    GROQ_BASE_URL: str        = "https://api.groq.com/openai/v1"
    OPENROUTER_BASE_URL: str  = "https://openrouter.ai/api/v1"

    # ─── File Limits ─────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    MIN_FILE_SIZE_MB: int = 0
    MAX_ROWS: int = 50000
    MAX_COLUMNS: int = 50
    ALLOWED_EXTENSIONS: list = [".csv", ".xlsx", ".xls"]

    # ─── Sandbox ─────────────────────────────────────────────────────────────
    SANDBOX_TIMEOUT_SECONDS: int = 10

    # ─── Storage ─────────────────────────────────────────────────────────────
    DATASETS_BASE_PATH: str = "./datasets"
    SQLITE_DB_PATH: str = "./insightengine.db"

    # ─── RAG ─────────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    TOP_K_CHUNKS: int = 5

    # Reads from .env file automatically — no code changes needed to add keys
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ─── Computed helpers ────────────────────────────────────────────────────
    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def groq_configured(self) -> bool:
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY.strip())

    @property
    def openrouter_configured(self) -> bool:
        return bool(self.OPENROUTER_API_KEY and self.OPENROUTER_API_KEY.strip())

    def dataset_path(self, dataset_id: str) -> str:
        return os.path.join(self.DATASETS_BASE_PATH, dataset_id)

    def raw_csv_path(self, dataset_id: str) -> str:
        return os.path.join(self.dataset_path(dataset_id), "raw.csv")

    def cleaned_csv_path(self, dataset_id: str) -> str:
        return os.path.join(self.dataset_path(dataset_id), "cleaned.csv")

    def faiss_index_path(self, dataset_id: str) -> str:
        return os.path.join(self.dataset_path(dataset_id), "index.faiss")

    def chunks_path(self, dataset_id: str) -> str:
        return os.path.join(self.dataset_path(dataset_id), "chunks.pkl")

    def metadata_path(self, dataset_id: str) -> str:
        return os.path.join(self.dataset_path(dataset_id), "metadata.json")


@lru_cache()
def get_settings() -> Settings:
    """Returns cached singleton settings instance."""
    return Settings()


settings = get_settings()