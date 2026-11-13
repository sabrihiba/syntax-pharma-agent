"""In-memory cosine similarity search over pharmacology chunks.

Build once, then reuse for queries. Vectors and chunks share the same row order.
Parent sections remain available for expansion and citations.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from app.rag.chunker import Chunk, Citation, Corpus, Section
from app.rag.embeddings import Embedder


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> Citation:
        return self.chunk.citation


def _normalize_rows(values) -> np.ndarray:
    raw = np.asarray(values)

    if raw.dtype.kind not in "iuf":
        raise ValueError("vectors must contain real numbers")

    matrix = np.array(raw, dtype=np.float64, copy=True)

    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(
            "vectors must be a finite two-dimensional matrix"
        )

    if matrix.shape[0] == 0:
        return matrix.astype(np.float32)

    if matrix.shape[1] == 0:
        raise ValueError("vectors must have a positive dimension")

    scales = np.max(np.abs(matrix), axis=1, keepdims=True)

    if np.any(scales == 0):
        raise ValueError("vectors must not contain zero vectors")

    matrix /= scales
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    return matrix.astype(np.float32)


def _filter_values(
    values: Collection[str] | None,
    name: str,
) -> set[str] | None:
    if values is None:
        return None

    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{name} must be a collection of strings"
        )

    result = set()

    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{name} must contain nonempty strings"
            )

        result.add(value.strip().casefold())

    return result


class VectorStore:
    """An exact, in-memory index.

    Prefer build() to keep embeddings and chunk order aligned.

    If vectors are supplied directly, they must come from the same model
    used by embedder and follow corpus.chunks order.
    """

    def __init__(
        self,
        corpus: Corpus,
        vectors: np.ndarray,
        embedder: Embedder,
    ) -> None:
        # Snapshot the lists so caller mutations cannot reorder the index.
        self._corpus = Corpus(
            documents=list(corpus.documents),
            sections=list(corpus.sections),
            chunks=list(corpus.chunks),
        )
        self._chunks = tuple(self._corpus.chunks)

        self._validate_chunks(self._corpus)
        self._vectors = _normalize_rows(vectors)

        if self._vectors.shape[0] != len(self._chunks):
            raise ValueError(
                "one vector is required for every chunk"
            )

        self._embedder = embedder
        self._model = embedder.config.model
        self._dimension = self._vectors.shape[1]

        if (
            embedder.dimension is not None
            and embedder.dimension != self._dimension
        ):
            raise ValueError(
                "store and embedder dimensions do not match"
            )

        self._vectors.setflags(write=False)

    @staticmethod
    def _validate_chunks(corpus: Corpus) -> None:
        ids: set[str] = set()

        for chunk in corpus.chunks:
            if chunk.id in ids:
                raise ValueError(
                    f"duplicate chunk id: {chunk.id}"
                )

            ids.add(chunk.id)
            parent = corpus.section(chunk.section_id)

            if parent is None:
                raise ValueError(
                    f"missing parent section: {chunk.section_id}"
                )

            if parent.doc != chunk.doc:
                raise ValueError(
                    f"parent document mismatch: {chunk.id}"
                )

    @classmethod
    def build(
        cls,
        corpus: Corpus,
        embedder: Embedder,
    ) -> VectorStore:
        # Validate parent links before making embedding requests.
        snapshot = Corpus(
            documents=list(corpus.documents),
            sections=list(corpus.sections),
            chunks=list(corpus.chunks),
        )
        cls._validate_chunks(snapshot)

        vectors = embedder.embed_chunks(snapshot.chunks)

        return cls(snapshot, vectors, embedder)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_section(self, section_id: str) -> Section | None:
        """Return a complete parent section, including its descendants."""
        return self._corpus.section(section_id)

    def _candidates(
        self,
        top_k: int,
        min_score: float | None,
        drugs: Collection[str] | None,
        sources: Collection[str] | None,
    ) -> np.ndarray:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError(
                "top_k must be a positive integer"
            )

        if min_score is not None:
            if (
                isinstance(min_score, (bool, str, bytes))
                or not np.isscalar(min_score)
                or not np.isrealobj(min_score)
                or not np.isfinite(min_score)
                or not -1 <= min_score <= 1
            ):
                raise ValueError(
                    "min_score must be between -1 and 1"
                )

        drug_set = _filter_values(drugs, "drugs")
        source_set = _filter_values(sources, "sources")

        return np.array(
            [
                index
                for index, chunk in enumerate(self._chunks)
                if (
                    drug_set is None
                    or (chunk.doc.drug or "").casefold() in drug_set
                )
                and (
                    source_set is None
                    or chunk.doc.source.casefold() in source_set
                )
            ],
            dtype=np.intp,
        )

    def search(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
        drugs: Collection[str] | None = None,
        sources: Collection[str] | None = None,
    ) -> list[SearchResult]:
        """Embed a question and return the best matching chunks.

        Filters are applied before ranking.

        Source filters use source kinds such as:
            "dailymed", "medlineplus", "glossary"

        Scores measure similarity, not clinical certainty.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "question must be a nonempty string"
            )

        candidates = self._candidates(
            top_k,
            min_score,
            drugs,
            sources,
        )

        if candidates.size == 0:
            return []

        if self._embedder.config.model != self._model:
            raise ValueError(
                "embedding model changed; rebuild the store"
            )

        query = self._embedder.embed_query(question)

        return self._rank(
            query,
            candidates,
            top_k,
            min_score,
        )

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float | None = None,
        drugs: Collection[str] | None = None,
        sources: Collection[str] | None = None,
    ) -> list[SearchResult]:
        """Search with a vector from the same model as the stored vectors."""
        candidates = self._candidates(
            top_k,
            min_score,
            drugs,
            sources,
        )

        return self._rank(
            query_vector,
            candidates,
            top_k,
            min_score,
        )

    def _rank(
        self,
        query_vector,
        candidates: np.ndarray,
        top_k: int,
        min_score: float | None,
    ) -> list[SearchResult]:
        query = np.asarray(query_vector)

        if query.ndim != 1 or query.size == 0:
            raise ValueError(
                "query vector must be a nonempty 1D vector"
            )

        query = _normalize_rows(query[np.newaxis, :])[0]

        # An empty, uninitialized index has no dimension to compare against.
        if self._dimension == 0 and not self._chunks:
            return []

        if query.shape[0] != self._dimension:
            raise ValueError(
                "query vector dimension does not match the store"
            )

        if candidates.size == 0:
            return []

        scores = np.clip(
            self._vectors[candidates] @ query,
            -1.0,
            1.0,
        )

        # Keep original corpus order when scores are equal.
        order = np.argsort(-scores, kind="stable")

        return [
            SearchResult(
                chunk=self._chunks[int(candidates[index])],
                score=float(scores[index]),
            )
            for index in order
            if min_score is None or scores[index] >= min_score
        ][:top_k]