"""
InsightEngine AI - Database Models
SQLAlchemy models for dataset metadata and user history.
"""
from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    DateTime, Text, JSON, Boolean, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from datetime import datetime
import uuid

from app.core.config import settings

# ─── Engine & Session ─────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite+aiosqlite:///{settings.SQLITE_DB_PATH}"

async_engine = create_async_engine(
    DATABASE_URL,
    echo=settings.DEBUG,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


# ─── Models ───────────────────────────────────────────────────────────────────

class DatasetMeta(Base):
    """
    Stores metadata about uploaded datasets.
    One record per dataset_id.
    """
    __tablename__ = "dataset_meta"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    original_filename = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    columns_json = Column(JSON, nullable=True)          # list of column names
    dtypes_json = Column(JSON, nullable=True)           # {col: dtype}
    sample_rows_json = Column(JSON, nullable=True)      # first 5 rows as list of dicts
    domain = Column(String, nullable=True)              # e.g. "sales", "hr", "finance"
    domain_confidence = Column(Float, nullable=True)
    is_cleaned = Column(Boolean, default=False)
    has_embeddings = Column(Boolean, default=False)
    cleaning_report_json = Column(JSON, nullable=True)
    profiling_report_json = Column(JSON, nullable=True)
    status = Column(String, default="uploaded")         # uploaded | processing | ready | error
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to query history
    queries = relationship("UserHistory", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DatasetMeta id={self.id} file={self.original_filename} status={self.status}>"


class UserHistory(Base):
    """
    Stores every user query and AI response for a dataset.
    Enables conversation memory and audit trail.
    """
    __tablename__ = "user_history"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, ForeignKey("dataset_meta.id"), nullable=False)
    user_query = Column(Text, nullable=False)
    generated_code = Column(Text, nullable=True)        # Final code from Judge
    code_output_json = Column(JSON, nullable=True)      # Raw execution result
    story_response = Column(Text, nullable=True)        # User-friendly explanation
    rag_chunks_used = Column(JSON, nullable=True)       # Which chunks were retrieved
    execution_success = Column(Boolean, default=False)
    execution_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("DatasetMeta", back_populates="queries")

    def __repr__(self):
        return f"<UserHistory id={self.id} dataset={self.dataset_id}>"


# ─── DB Dependency (FastAPI) ──────────────────────────────────────────────────

async def get_db() -> AsyncSession:
    """FastAPI dependency that provides an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Creates all tables if they don't exist. Called on app startup."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)