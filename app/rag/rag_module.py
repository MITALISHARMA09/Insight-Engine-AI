from __future__ import annotations
"""
InsightEngine AI - RAG Module
Converts dataset to text chunks → embeddings → FAISS index.
Enables semantic retrieval for context-aware AI responses.

NOTE: Uses sentence-transformers for embeddings (local model).
If sentence-transformers is not installed, falls back to TF-IDF.
"""
import logging
import pickle
import os
import json
import math
from typing import Optional
import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGModule:
    """
    Handles the complete RAG pipeline:
    1. Chunk dataset into descriptive text segments
    2. Generate embeddings (sentence-transformers or fallback)
    3. Store in FAISS index
    4. Retrieve relevant context for user queries
    """

    def __init__(self):
        self._embedder = None
        self._faiss = None
        self._use_faiss = False
        self._try_load_dependencies()

    def _try_load_dependencies(self):
        """Try to load optional heavy dependencies."""
        try:
            from sentence_transformers import SentenceTransformer
            self._SentenceTransformer = SentenceTransformer
            self._embedder_loaded = True
        except ImportError:
            self._embedder_loaded = False
            logger.warning("sentence-transformers not available, using TF-IDF fallback")

        try:
            import faiss
            self._faiss = faiss
            self._use_faiss = True
        except ImportError:
            self._use_faiss = False
            logger.warning("faiss-cpu not available, using cosine similarity fallback")

    # ─── Build Index ─────────────────────────────────────────────────────────

    async def build_index(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        domain: str = "general",
    ) -> dict:
        """
        Convert DataFrame to searchable embedding index.

        Returns:
            dict with 'chunks_count', 'index_path', 'chunks_path'
        """
        # Step 1: Generate chunks
        chunks = self._create_chunks(df, domain)
        logger.info(f"Created {len(chunks)} text chunks for dataset {dataset_id}")

        # Step 2: Generate embeddings
        embeddings = self._generate_embeddings([c["text"] for c in chunks])

        # Step 3: Store index and chunks
        index_path = settings.faiss_index_path(dataset_id)
        chunks_path = settings.chunks_path(dataset_id)

        self._save_index(embeddings, index_path)
        self._save_chunks(chunks, chunks_path)

        logger.info(f"RAG index built and saved for dataset {dataset_id}")
        return {
            "chunks_count": len(chunks),
            "index_path": index_path,
            "chunks_path": chunks_path,
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        }

    # ─── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        dataset_id: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve top-k relevant chunks for a query.

        Returns:
            List of dicts with 'text', 'score', 'chunk_id'
        """
        top_k = top_k or settings.TOP_K_CHUNKS
        chunks_path = settings.chunks_path(dataset_id)
        index_path = settings.faiss_index_path(dataset_id)

        if not os.path.exists(chunks_path):
            logger.warning(f"No RAG index for dataset {dataset_id}")
            return []

        chunks = self._load_chunks(chunks_path)

        if self._use_faiss and os.path.exists(index_path):
            return self._faiss_retrieve(query, chunks, index_path, top_k)
        else:
            return self._cosine_retrieve(query, chunks, top_k)

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """Format retrieved chunks as a context string for LLM prompts."""
        if not retrieved_chunks:
            return "No relevant context found in the dataset."
        lines = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            lines.append(f"[Fact {i}] {chunk['text']}")
        return "\n".join(lines)

    # ─── Chunking ────────────────────────────────────────────────────────────

    def _create_chunks(self, df: pd.DataFrame, domain: str) -> list[dict]:
        """
        Convert DataFrame to text chunks.
        Strategy: statistical summaries + sample rows + column descriptions.
        """
        chunks = []
        chunk_id = 0

        # Chunk 1: Overall dataset description
        overview = self._dataset_overview_chunk(df, domain)
        chunks.append({"id": chunk_id, "text": overview, "type": "overview"})
        chunk_id += 1

        # Chunks 2-N: Per-column statistical summaries
        for col in df.columns:
            col_chunk = self._column_summary_chunk(df, col)
            if col_chunk:
                chunks.append({"id": chunk_id, "text": col_chunk, "type": "column_summary"})
                chunk_id += 1

        # Chunks: Row-based samples (every N rows)
        sample_chunks = self._sample_row_chunks(df)
        for sc in sample_chunks:
            chunks.append({"id": chunk_id, "text": sc, "type": "sample_rows"})
            chunk_id += 1

        return chunks

    def _dataset_overview_chunk(self, df: pd.DataFrame, domain: str) -> str:
        return (
            f"Dataset overview: This is a {domain} dataset with {len(df)} rows and {len(df.columns)} columns. "
            f"Columns are: {', '.join(df.columns.tolist())}."
        )

    def _column_summary_chunk(self, df: pd.DataFrame, col: str) -> Optional[str]:
        """Generate a natural language summary of a single column."""
        series = df[col].dropna()
        if len(series) == 0:
            return None

        if pd.api.types.is_numeric_dtype(series):
            return (
                f"Column '{col}' (numeric): "
                f"ranges from {series.min():.2f} to {series.max():.2f}, "
                f"average is {series.mean():.2f}, "
                f"typical value is {series.median():.2f}, "
                f"has {df[col].isnull().sum()} missing values."
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            return (
                f"Column '{col}' (date): "
                f"ranges from {series.min()} to {series.max()}."
            )
        else:
            top_vals = series.value_counts().head(5).index.tolist()
            unique_count = series.nunique()
            return (
                f"Column '{col}' (text/category): "
                f"{unique_count} unique values. "
                f"Most common: {', '.join([str(v) for v in top_vals])}."
            )

    def _sample_row_chunks(self, df: pd.DataFrame, chunk_size: int = 20) -> list[str]:
        """Create row-based chunks for row-level queries."""
        chunks = []
        total = len(df)
        num_chunks = math.ceil(total / chunk_size)

        for i in range(min(num_chunks, 50)):  # Max 50 row chunks
            start = i * chunk_size
            end = min(start + chunk_size, total)
            subset = df.iloc[start:end]
            row_text = subset.to_string(index=False, max_cols=10)
            chunks.append(f"Data rows {start+1} to {end}:\n{row_text}")

        return chunks

    # ─── Embedding & Index ───────────────────────────────────────────────────

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers or TF-IDF fallback."""
        if self._embedder_loaded:
            try:
                if self._embedder is None:
                    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
                    self._embedder = self._SentenceTransformer(settings.EMBEDDING_MODEL)
                embeddings = self._embedder.encode(texts, show_progress_bar=False)
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"Sentence-transformer failed: {e}. Using TF-IDF.")

        # TF-IDF fallback
        return self._tfidf_embeddings(texts)

    def _tfidf_embeddings(self, texts: list[str]) -> np.ndarray:
        """Simple TF-IDF based embeddings as fallback (no sklearn needed)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=512)
            matrix = vectorizer.fit_transform(texts)
            embeddings = matrix.toarray().astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms
        except ImportError:
            logger.warning("sklearn not installed — using keyword-overlap RAG (no embeddings).")
            return self._keyword_embeddings(texts)
        except Exception as e:
            logger.error(f"TF-IDF failed: {e} — using keyword-overlap RAG.")
            return self._keyword_embeddings(texts)

    def _keyword_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Zero-dependency fallback: represent each chunk as a bag-of-words
        count vector over the top 256 words in the corpus.
        Good enough for keyword-based retrieval without any ML library.
        """
        # Build vocabulary from all texts
        from collections import Counter
        all_words: list[str] = []
        tokenized = []
        for t in texts:
            words = t.lower().split()
            tokenized.append(words)
            all_words.extend(words)

        # Top 256 most common words as vocabulary
        vocab_list = [w for w, _ in Counter(all_words).most_common(256)]
        vocab = {w: i for i, w in enumerate(vocab_list)}
        dim = max(len(vocab), 1)

        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        for i, words in enumerate(tokenized):
            for w in words:
                if w in vocab:
                    embeddings[i, vocab[w]] += 1.0
            # L2 normalise
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        return embeddings

    def _save_index(self, embeddings: np.ndarray, index_path: str):
        """Save FAISS or numpy index."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if self._use_faiss:
            dim = embeddings.shape[1]
            index = self._faiss.IndexFlatIP(dim)  # Inner product (cosine after norm)
            # Normalize embeddings
            self._faiss.normalize_L2(embeddings)
            index.add(embeddings)
            self._faiss.write_index(index, index_path)
        else:
            np.save(index_path.replace(".faiss", ".npy"), embeddings)

    def _save_chunks(self, chunks: list[dict], chunks_path: str):
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

    def _load_chunks(self, chunks_path: str) -> list[dict]:
        with open(chunks_path, "rb") as f:
            return pickle.load(f)

    # ─── Retrieval ───────────────────────────────────────────────────────────

    def _faiss_retrieve(
        self, query: str, chunks: list[dict], index_path: str, top_k: int
    ) -> list[dict]:
        """Retrieve using FAISS index."""
        try:
            index = self._faiss.read_index(index_path)
            query_emb = self._generate_embeddings([query])
            self._faiss.normalize_L2(query_emb)
            scores, indices = index.search(query_emb, min(top_k, len(chunks)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunks):
                    results.append({**chunks[idx], "score": float(score)})
            return results
        except Exception as e:
            logger.error(f"FAISS retrieval failed: {e}")
            return self._cosine_retrieve(query, chunks, top_k)

    def _cosine_retrieve(
        self, query: str, chunks: list[dict], top_k: int
    ) -> list[dict]:
        """Simple keyword-based retrieval fallback."""
        query_words = set(query.lower().split())
        scored = []
        for chunk in chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(query_words & chunk_words)
            scored.append({**chunk, "score": overlap})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# Module-level singleton
rag_module = RAGModule()