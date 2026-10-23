"""LanceDB vector store implementation."""

from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from docugraph.core.config import get_config
from docugraph.core.embeddings import EmbeddingProviderBase, get_embedder
from docugraph.core.models import Chunk, SearchResult


class VectorStore:
    """Vector store backed by LanceDB."""

    TABLE_NAME = "chunks"

    def __init__(
        self,
        db_path: Path | str | None = None,
        embedder: EmbeddingProviderBase | None = None,
    ):
        """Initialize the vector store.

        Args:
            db_path: Path to the LanceDB database directory.
            embedder: Embedding provider to use. If None, uses default from config.
        """
        if db_path is None:
            config = get_config()
            db_path = config.storage.data_dir / "vectors"

        self._db_path = Path(db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(self._db_path))
        self._embedder = embedder or get_embedder()
        self._table: lancedb.table.Table | None = None
        # Rebuilt lazily by search_fts / search_hybrid after new chunks arrive.
        self._fts_stale = True
        # Why the most recent search_hybrid call fell back to vector-only, or
        # None when it ran as native hybrid. The graph leg records its reasons
        # (GraphLegStatus) instead of failing silently; the keyword side now
        # does the same, because "hybrid was unavailable" and "hybrid ran"
        # must not be indistinguishable to a caller -- the retrieval benchmark
        # refuses to score a 'lancedb-hybrid' arm that silently measured
        # vector search.
        self.last_hybrid_fallback_reason: str | None = None

    @property
    def embedding_model(self) -> str:
        """The model this store is actually using, defaults already resolved.

        Not the same as the configured value: config leaves the model unset for
        the local backends and lets each one name its own, so reporting the
        config field would show a user on fastembed that their model is None.
        """
        return self._embedder.model_name

    def _get_schema(self) -> pa.Schema:
        """Get PyArrow schema for the chunks table."""
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self._embedder.dimensions)),
                pa.field("start_char", pa.int64()),
                pa.field("end_char", pa.int64()),
                pa.field("metadata", pa.string()),  # JSON string
                pa.field("created_at", pa.string()),  # ISO format
                pa.field("embedding_model", pa.string()),  # Model that produced the vectors
            ]
        )

    def _validate_existing_table(self, table: lancedb.table.Table) -> None:
        """Guard against opening a store embedded with a different model.

        Dimensions alone are insufficient (all-MiniLM-L6-v2 and
        bge-small-en-v1.5 are both 384-dim but live in different spaces);
        model identity is checked too.
        """
        schema = table.schema
        vector_field = schema.field("vector")
        existing_dims = vector_field.type.list_size

        if existing_dims != self._embedder.dimensions:
            raise RuntimeError(
                f"Vector store was built with {existing_dims}-dim embeddings, but "
                f"'{self._embedder.model_name}' produces {self._embedder.dimensions}-dim "
                "vectors. Run 'docugraph clear' and re-index to rebuild the store."
            )

        if "embedding_model" not in schema.names:
            raise RuntimeError(
                "Vector store predates per-model tracking (no 'embedding_model' column) "
                "and may have been built with a different model. Run 'docugraph clear' "
                "and re-index to rebuild the store."
            )

        # Check the model recorded on the first row (stores are homogeneous).
        if table.count_rows() > 0:
            existing_model = table.to_arrow().column("embedding_model")[0].as_py()
            if existing_model != self._embedder.model_name:
                raise RuntimeError(
                    f"Vector store was built with '{existing_model}', but the configured "
                    f"embedder is '{self._embedder.model_name}'. Run 'docugraph clear' "
                    "and re-index to rebuild the store."
                )

    def _ensure_table(self) -> lancedb.table.Table:
        """Ensure the table exists and return it."""
        if self._table is not None:
            return self._table

        if self.TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(self.TABLE_NAME)
            self._validate_existing_table(self._table)
        else:
            # Create empty table with schema
            self._table = self._db.create_table(
                self.TABLE_NAME,
                schema=self._get_schema(),
                mode="overwrite",
            )

        return self._table

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the vector store.

        Args:
            chunks: List of chunks to add.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        table = self._ensure_table()

        # Generate embeddings for chunks without them
        texts_to_embed = []
        indices_to_embed = []
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                texts_to_embed.append(chunk.content)
                indices_to_embed.append(i)

        if texts_to_embed:
            embeddings = self._embedder.embed(texts_to_embed)
            for idx, embedding in zip(indices_to_embed, embeddings, strict=False):
                chunks[idx].embedding = embedding

        # Convert to records
        import json

        records = []
        for chunk in chunks:
            records.append(
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "vector": chunk.embedding,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": json.dumps(chunk.metadata),
                    "created_at": chunk.created_at.isoformat(),
                    "embedding_model": self._embedder.model_name,
                }
            )

        table.add(records)
        # The FTS index does not cover rows added after it was built; rebuild
        # before the next keyword search rather than serving a partial index.
        self._fts_stale = True
        return len(records)

    def _row_to_chunk(self, row: Any) -> Chunk:
        """One row of a LanceDB result frame as a Chunk."""
        import json

        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            content=row["content"],
            embedding=row["vector"].tolist() if hasattr(row["vector"], "tolist") else row["vector"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def ensure_fts_index(self) -> None:
        """Build (or rebuild) the full-text index over `content`. Idempotent.

        Nothing created this index before v0.1.1, which is why
        `query_type="hybrid"` always raised and `search_hybrid` always fell
        back to vector-only -- "keyword search" silently was vector search.
        The retrieval benchmark exposed it; this is the fix. `replace=True`
        because LanceDB's FTS index is build-time, not incremental: rows added
        after the build are invisible to it, so a stale index must be rebuilt,
        not reused.
        """
        table = self._ensure_table()
        if table.count_rows() == 0:
            return
        table.create_fts_index("content", replace=True)
        self._fts_stale = False

    def search_fts(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """Keyword (BM25 full-text) search only -- no embeddings involved.

        This is the true keyword leg: `HybridRetriever` fuses it with vector
        search, and the benchmark measures it as its own arm. Scores are the
        index's BM25 scores (higher is better), not similarities; compare them
        by rank, not magnitude.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_expr: Optional SQL-like filter expression.

        Returns:
            List of search results with BM25 scores.
        """
        table = self._ensure_table()
        if self._fts_stale:
            self.ensure_fts_index()

        search_query = table.search(query, query_type="fts").limit(top_k)
        if filter_expr:
            search_query = search_query.where(filter_expr)
        results = search_query.to_pandas()

        if results.empty:
            return []

        search_results = []
        for _, row in results.iterrows():
            # LanceDB FTS reports BM25 relevance as `_score` (newer releases)
            # or `score` (older ones within the pinned range).
            raw = row.get("_score")
            if raw is None:
                raw = row.get("score", 0.0)
            search_results.append(SearchResult(chunk=self._row_to_chunk(row), score=float(raw)))

        return search_results

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_expr: Optional SQL-like filter expression.

        Returns:
            List of search results with scores.
        """
        table = self._ensure_table()

        # Embed the query
        query_embedding = self._embedder.embed_query(query)

        # Build search query
        search_query = table.search(query_embedding).limit(top_k)

        if filter_expr:
            search_query = search_query.where(filter_expr)

        # Execute search
        results = search_query.to_pandas()

        if results.empty:
            return []

        # Convert to SearchResult objects
        import json

        search_results = []
        for _, row in results.iterrows():
            chunk = Chunk(
                id=row["id"],
                document_id=row["document_id"],
                content=row["content"],
                embedding=row["vector"].tolist()
                if hasattr(row["vector"], "tolist")
                else row["vector"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            # LanceDB returns _distance, lower is better
            # Convert to similarity score (1 / (1 + distance))
            distance = row.get("_distance", 0)
            score = 1.0 / (1.0 + distance)

            search_results.append(SearchResult(chunk=chunk, score=score))

        return search_results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining vector and full-text search (LanceDB-native fusion).

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_expr: Optional SQL-like filter expression.

        Returns:
            List of search results with combined scores.
        """
        table = self._ensure_table()
        self.last_hybrid_fallback_reason = None

        # Native hybrid needs the FTS index just as search_fts does; before
        # this was built, every hybrid query raised and the except below made
        # "hybrid" mean "vector" without anyone being told.
        if self._fts_stale:
            try:
                self.ensure_fts_index()
            except Exception as e:  # pragma: no cover - backend-version specific
                self.last_hybrid_fallback_reason = f"the FTS index could not be built: {e}"
                return self.search(query, top_k, filter_expr)

        # Embed the query
        query_embedding = self._embedder.embed_query(query)

        # LanceDB hybrid search
        try:
            search_query = (
                table.search(query_type="hybrid").vector(query_embedding).text(query).limit(top_k)
            )
            if filter_expr:
                search_query = search_query.where(filter_expr)
            results = search_query.to_pandas()
        except Exception as e:
            # Still a degrade, not a failure -- hybrid is an enhancement over
            # vector search -- but a *recorded* one. The caller can tell the
            # difference; the benchmark refuses to score this arm when the
            # reason is set.
            self.last_hybrid_fallback_reason = f"native hybrid search failed: {e}"
            return self.search(query, top_k, filter_expr)

        if results.empty:
            return []

        search_results = []
        for _, row in results.iterrows():
            # Hybrid results carry a fused `_relevance_score` (higher is
            # better). The previous code read `_distance` with a default of 0,
            # which scored every hybrid row 1.0 -- harmless under rank-based
            # fusion, wrong everywhere else.
            raw = row.get("_relevance_score")
            if raw is not None:
                score = float(raw)
            else:
                distance = row.get("_distance", 0)
                score = 1.0 / (1.0 + distance)

            search_results.append(SearchResult(chunk=self._row_to_chunk(row), score=score))

        return search_results

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk if found, None otherwise.
        """
        table = self._ensure_table()

        results = table.search().where(f"id = '{chunk_id}'").limit(1).to_pandas()

        if results.empty:
            return None

        import json

        row = results.iloc[0]
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            content=row["content"],
            embedding=row["vector"].tolist() if hasattr(row["vector"], "tolist") else row["vector"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to delete.

        Returns:
            Number of chunks deleted.
        """
        if not chunk_ids:
            return 0

        table = self._ensure_table()

        # Build delete filter
        id_list = ", ".join(f"'{cid}'" for cid in chunk_ids)
        table.delete(f"id IN ({id_list})")

        return len(chunk_ids)

    def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: The document ID.

        Returns:
            Number of chunks deleted.
        """
        table = self._ensure_table()

        # Count before delete
        count = len(table.search().where(f"document_id = '{document_id}'").to_pandas())

        table.delete(f"document_id = '{document_id}'")

        return count

    def count(self) -> int:
        """Get the total number of chunks."""
        table = self._ensure_table()
        return int(table.count_rows())

    def clear(self) -> None:
        """Clear all chunks from the store."""
        if self.TABLE_NAME in self._db.table_names():
            self._db.drop_table(self.TABLE_NAME)
        self._table = None


def get_vector_store() -> VectorStore:
    """Get the default vector store instance."""
    return VectorStore()
