"""Hybrid retrieval using equal-weight Reciprocal Rank Fusion (RRF).

RRF adds 1 / (rrf_k + rank) for each retriever containing a chunk.
Ranks start at 1; raw BM25 and cosine scores are never added together.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace

from app.rag.bm25 import BM25Config, BM25Index, BM25Result
from app.rag.chunker import Chunk, Citation, Corpus, Section
from app.rag.embeddings import Embedder
from app.rag.vector_store import SearchResult, VectorStore


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer"
        )


@dataclass(frozen=True)
class HybridConfig:
    rrf_k: int = 60
    candidate_k: int = 20

    def __post_init__(self) -> None:
        _positive_integer(self.rrf_k, "rrf_k")
        _positive_integer(self.candidate_k, "candidate_k")


@dataclass(frozen=True)
class HybridResult:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None

    @property
    def citation(self) -> Citation:
        return self.chunk.citation


def reciprocal_rank_fusion(
    vector_results: Sequence[SearchResult],
    bm25_results: Sequence[BM25Result],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[HybridResult]:
    """Fuse already-ranked lists by chunk ID.

    A chunk contributes at most once per retriever. Duplicate entries retain
    their first rank. Conflicting content under the same ID is rejected.
    Exact fused-score ties are resolved by chunk ID for deterministic output.
    """
    _positive_integer(rrf_k, "rrf_k")
    _positive_integer(top_k, "top_k")

    merged: dict[str, HybridResult] = {}

    for results, name in (
        (vector_results, "vector"),
        (bm25_results, "bm25"),
    ):
        for rank, hit in enumerate(results, start=1):
            chunk_id = hit.chunk.id
            current = merged.get(chunk_id)

            if current is None:
                current = HybridResult(
                    chunk=hit.chunk,
                    score=0.0,
                )
            elif current.chunk != hit.chunk:
                raise ValueError(
                    f"conflicting chunks for id: {chunk_id}"
                )

            if getattr(current, f"{name}_rank") is not None:
                continue

            changes = {
                f"{name}_rank": rank,
                f"{name}_score": float(hit.score),
            }

            merged[chunk_id] = replace(
                current,
                score=current.score + 1.0 / (rrf_k + rank),
                **changes,
            )

    return sorted(
        merged.values(),
        key=lambda hit: (-hit.score, hit.chunk.id),
    )[:top_k]


def _filter_values(
    values: Collection[str] | None,
    name: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None

    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{name} must be a collection of strings"
        )

    normalized = set()

    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{name} must contain nonempty strings"
            )

        normalized.add(value.strip().casefold())

    return tuple(sorted(normalized))


class HybridRetriever:
    """Combine a vector store and BM25 index built from the same corpus.

    Build once and reuse across questions. API errors propagate; a failed
    vector request is not silently presented as a successful hybrid search.
    """

    def __init__(
        self,
        corpus: Corpus,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        config: HybridConfig | None = None,
    ) -> None:
        self.config = config or HybridConfig()

        self._corpus = Corpus(
            documents=list(corpus.documents),
            sections=list(corpus.sections),
            chunks=list(corpus.chunks),
        )

        self._chunks = {
            chunk.id: chunk
            for chunk in self._corpus.chunks
        }

        if len(self._chunks) != len(self._corpus.chunks):
            raise ValueError(
                "duplicate chunk ids in corpus"
            )

        self._vector_store = vector_store
        self._bm25_index = bm25_index

        if (
            len(vector_store) != len(self._chunks)
            or len(bm25_index) != len(self._chunks)
        ):
            raise ValueError(
                "retrievers must index the same corpus"
            )

        for chunk in self._chunks.values():
            parent = self._corpus.section(chunk.section_id)

            if parent is None or parent.doc != chunk.doc:
                raise ValueError(
                    f"invalid parent link: {chunk.id}"
                )

        for section in self._corpus.sections:
            if (
                vector_store.get_section(section.id) != section
                or bm25_index.get_section(section.id) != section
            ):
                raise ValueError(
                    "retriever parent sections do not match corpus"
                )

    @classmethod
    def build(
        cls,
        corpus: Corpus,
        embedder: Embedder,
        *,
        config: HybridConfig | None = None,
        bm25_config: BM25Config | None = None,
    ) -> HybridRetriever:
        config = config or HybridConfig()

        snapshot = Corpus(
            documents=list(corpus.documents),
            sections=list(corpus.sections),
            chunks=list(corpus.chunks),
        )

        bm25_index = BM25Index.build(
            snapshot,
            bm25_config,
        )
        vector_store = VectorStore.build(
            snapshot,
            embedder,
        )

        return cls(
            snapshot,
            vector_store,
            bm25_index,
            config,
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def get_section(self, section_id: str) -> Section | None:
        return self._corpus.section(section_id)

    def search(
        self,
        question: str,
        *,
        top_k: int = 5,
        drugs: Collection[str] | None = None,
        sources: Collection[str] | None = None,
    ) -> list[HybridResult]:
        """Retrieve candidates from both indexes, then fuse their rankings.

        Filters are passed to both retrievers before ranking.
        RRF scores are ranking signals, not probabilities of correctness.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "question must be a nonempty string"
            )

        _positive_integer(top_k, "top_k")

        drugs = _filter_values(drugs, "drugs")
        sources = _filter_values(sources, "sources")

        if not self._chunks or drugs == () or sources == ():
            return []

        depth = max(
            self.config.candidate_k,
            top_k,
        )

        vector_results = self._vector_store.search(
            question,
            top_k=depth,
            drugs=drugs,
            sources=sources,
        )

        bm25_results = self._bm25_index.search(
            question,
            top_k=depth,
            drugs=drugs,
            sources=sources,
        )

        # Reject stale or foreign chunks before passing evidence onward.
        for hit in [*vector_results, *bm25_results]:
            if self._chunks.get(hit.chunk.id) != hit.chunk:
                raise ValueError(
                    "retriever returned a chunk outside this corpus"
                )

        return reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            rrf_k=self.config.rrf_k,
            top_k=top_k,
        )