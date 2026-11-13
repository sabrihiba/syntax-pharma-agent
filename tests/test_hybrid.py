"""Offline RRF and hybrid integration tests.

No real API calls.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from app.rag.bm25 import BM25Index, BM25Result
from app.rag.chunker import Corpus, chunk_document
from app.rag.hybrid import (
    HybridConfig,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from app.rag.vector_store import SearchResult, VectorStore


def make_corpus():
    documents, sections, chunks = [], [], []

    for filename, body in [
        ("dailymed-warfarin.md", "Warfarin bleeding."),
        ("dailymed-apixaban.md", "Apixaban bleeding."),
        ("medlineplus-warfarin.md", "Patient monitoring."),
    ]:
        doc, parents, children = chunk_document(
            filename,
            f"# Document\n## 4 CONTRAINDICATIONS\n{body}",
        )
        documents.append(doc)
        sections.extend(parents)
        chunks.extend(children)

    return Corpus(documents, sections, chunks)


def fake_embedder():
    embedder = Mock()
    embedder.config = SimpleNamespace(model="test-model")
    embedder.dimension = None

    embedder.embed_chunks.return_value = np.array(
        [[1, 0], [0, 1], [-1, 0]],
        dtype=float,
    )
    embedder.embed_query.return_value = np.array(
        [1, 0],
        dtype=float,
    )

    return embedder


def fake_retrievers(
    corpus,
    vector_hits=(),
    bm25_hits=(),
):
    vector = MagicMock()
    bm25 = MagicMock()

    for retriever in (vector, bm25):
        retriever.__len__.return_value = len(corpus.chunks)
        retriever.get_section.side_effect = corpus.section

    vector.search.return_value = list(vector_hits)
    bm25.search.return_value = list(bm25_hits)

    return vector, bm25


@pytest.fixture
def corpus():
    return make_corpus()


def test_rrf_matches_hand_calculated_one_based_ranks(corpus):
    a, b, c = corpus.chunks

    hits = reciprocal_rank_fusion(
        [
            SearchResult(a, 0.9),
            SearchResult(b, 0.8),
        ],
        [
            BM25Result(b, 20),
            BM25Result(c, 10),
        ],
    )

    by_id = {
        hit.chunk.id: hit
        for hit in hits
    }

    assert hits[0].chunk == b
    assert by_id[a.id].score == pytest.approx(1 / 61)
    assert by_id[b.id].score == pytest.approx(
        1 / 62 + 1 / 61
    )
    assert by_id[c.id].score == pytest.approx(1 / 62)


def test_raw_score_magnitudes_do_not_affect_fusion(corpus):
    a, b, _ = corpus.chunks

    first = reciprocal_rank_fusion(
        [
            SearchResult(a, 0.9),
            SearchResult(b, 0.1),
        ],
        [
            BM25Result(b, 1000),
            BM25Result(a, 0.01),
        ],
    )

    second = reciprocal_rank_fusion(
        [
            SearchResult(a, 0.02),
            SearchResult(b, -0.8),
        ],
        [
            BM25Result(b, 2),
            BM25Result(a, 1),
        ],
    )

    assert [
        (hit.chunk.id, hit.score)
        for hit in first
    ] == [
        (hit.chunk.id, hit.score)
        for hit in second
    ]


def test_result_retains_ranks_raw_scores_and_citation(corpus):
    a = corpus.chunks[0]

    hit = reciprocal_rank_fusion(
        [SearchResult(a, 0.8)],
        [BM25Result(a, 5)],
    )[0]

    assert hit.vector_rank == 1
    assert hit.bm25_rank == 1
    assert hit.vector_score == 0.8
    assert hit.bm25_score == 5
    assert hit.citation == a.citation


def test_duplicates_do_not_add_extra_votes(corpus):
    a = corpus.chunks[0]

    hits = reciprocal_rank_fusion(
        [
            SearchResult(a, 0.9),
            SearchResult(a, 0.8),
        ],
        [
            BM25Result(a, 4),
            BM25Result(a, 3),
        ],
    )

    assert len(hits) == 1
    assert hits[0].score == pytest.approx(2 / 61)
    assert hits[0].vector_score == 0.9
    assert hits[0].bm25_score == 4


def test_conflicting_content_under_same_id_is_rejected(corpus):
    a = corpus.chunks[0]
    changed = replace(
        a,
        text="Different content",
    )

    with pytest.raises(ValueError, match="conflicting"):
        reciprocal_rank_fusion(
            [SearchResult(a, 0.9)],
            [BM25Result(changed, 5)],
        )


def test_chunks_from_same_section_remain_distinct(corpus):
    a = corpus.chunks[0]
    b = replace(
        a,
        id=a.section_id + "#1",
        part=1,
        text="Next passage",
    )

    hits = reciprocal_rank_fusion(
        [
            SearchResult(a, 0.9),
            SearchResult(b, 0.8),
        ],
        [],
    )

    assert len(hits) == 2


def test_empty_lists():
    assert reciprocal_rank_fusion([], []) == []


@pytest.mark.parametrize(
    "backend",
    ["vector", "bm25"],
)
def test_single_backend_results_still_work(corpus, backend):
    a = corpus.chunks[0]

    vectors = (
        [SearchResult(a, 0.5)]
        if backend == "vector"
        else []
    )
    lexical = (
        [BM25Result(a, 4)]
        if backend == "bm25"
        else []
    )

    hit = reciprocal_rank_fusion(
        vectors,
        lexical,
    )[0]

    assert hit.score == pytest.approx(1 / 61)
    assert (
        hit.vector_rank is not None
    ) == (backend == "vector")
    assert (
        hit.bm25_rank is not None
    ) == (backend == "bm25")


def test_custom_rrf_constant_and_top_k(corpus):
    a, b, _ = corpus.chunks

    hits = reciprocal_rank_fusion(
        [
            SearchResult(a, 0.9),
            SearchResult(b, 0.8),
        ],
        [],
        rrf_k=10,
        top_k=1,
    )

    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1 / 11)


def test_exact_ties_sort_by_chunk_id(corpus):
    a, b, _ = corpus.chunks

    hits = reciprocal_rank_fusion(
        [
            SearchResult(a, 0.9),
            SearchResult(b, 0.8),
        ],
        [
            BM25Result(b, 4),
            BM25Result(a, 3),
        ],
    )

    assert [
        hit.chunk.id
        for hit in hits
    ] == sorted([a.id, b.id])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rrf_k": 0},
        {"rrf_k": -1},
        {"rrf_k": True},
        {"rrf_k": 1.5},
        {"top_k": 0},
        {"top_k": True},
    ],
)
def test_invalid_fusion_parameters(kwargs):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [],
            [],
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rrf_k": 0},
        {"rrf_k": True},
        {"candidate_k": 0},
        {"candidate_k": -1},
        {"candidate_k": 1.5},
        {"candidate_k": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        HybridConfig(**kwargs)


def test_search_passes_same_filters_and_candidate_depth(corpus):
    a = corpus.chunks[0]

    vector, bm25 = fake_retrievers(
        corpus,
        [SearchResult(a, 0.9)],
        [BM25Result(a, 3)],
    )

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    hits = retriever.search(
        "question",
        top_k=2,
        drugs=[" WARFARIN "],
        sources=["DAILYMED"],
    )

    for backend in (vector, bm25):
        backend.search.assert_called_once_with(
            "question",
            top_k=20,
            drugs=("warfarin",),
            sources=("dailymed",),
        )

    assert len(hits) == 1
    assert hits[0].score == pytest.approx(2 / 61)


def test_candidate_depth_grows_with_requested_top_k(corpus):
    vector, bm25 = fake_retrievers(corpus)
    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    retriever.search("question", top_k=30)

    assert vector.search.call_args.kwargs["top_k"] == 30
    assert bm25.search.call_args.kwargs["top_k"] == 30


@pytest.mark.parametrize(
    "filters",
    [
        {"drugs": []},
        {"sources": []},
    ],
)
def test_empty_filters_skip_both_retrievers(corpus, filters):
    vector, bm25 = fake_retrievers(corpus)
    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    assert retriever.search(
        "question",
        **filters,
    ) == []

    vector.search.assert_not_called()
    bm25.search.assert_not_called()


def test_build_integrates_real_indexes_with_fake_embeddings(corpus):
    embedder = fake_embedder()

    retriever = HybridRetriever.build(
        corpus,
        embedder,
    )

    assert len(retriever) == 3
    embedder.embed_chunks.assert_called_once_with(
        corpus.chunks
    )

    hits = retriever.search(
        "warfarin contraindications",
        drugs=["warfarin"],
        sources=["dailymed"],
    )

    embedder.embed_query.assert_called_once()

    assert len(hits) == 1
    assert (
        hits[0].chunk.doc.file
        == "dailymed-warfarin.md"
    )
    assert hits[0].vector_rank == hits[0].bm25_rank == 1
    assert retriever.get_section(
        hits[0].chunk.section_id
    ) is not None
    assert retriever.get_section("missing") is None


def test_existing_indexes_can_be_reused_without_reembedding(corpus):
    embedder = fake_embedder()

    vector = VectorStore.build(
        corpus,
        embedder,
    )
    bm25 = BM25Index.build(corpus)

    embedder.embed_chunks.reset_mock()

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    assert retriever.search("warfarin")
    embedder.embed_chunks.assert_not_called()


def test_unknown_filters_return_no_results_without_query_api(corpus):
    embedder = fake_embedder()

    retriever = HybridRetriever.build(
        corpus,
        embedder,
    )

    assert retriever.search(
        "question",
        drugs=["unknown"],
    ) == []

    embedder.embed_query.assert_not_called()


def test_empty_corpus_skips_search():
    corpus = Corpus([], [], [])
    vector, bm25 = fake_retrievers(corpus)

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    assert retriever.search("question") == []

    vector.search.assert_not_called()
    bm25.search.assert_not_called()


def test_foreign_returned_chunk_is_rejected(corpus):
    a = replace(
        corpus.chunks[0],
        text="Stale text",
    )
    vector, bm25 = fake_retrievers(
        corpus,
        [SearchResult(a, 0.9)],
    )

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    with pytest.raises(ValueError, match="outside"):
        retriever.search("question")


def test_different_index_sizes_are_rejected(corpus):
    vector, bm25 = fake_retrievers(corpus)
    bm25.__len__.return_value = 0

    with pytest.raises(ValueError, match="same corpus"):
        HybridRetriever(
            corpus,
            vector,
            bm25,
        )


def test_different_parent_sections_are_rejected(corpus):
    vector, bm25 = fake_retrievers(corpus)
    bm25.get_section.side_effect = None
    bm25.get_section.return_value = None

    with pytest.raises(ValueError, match="parent sections"):
        HybridRetriever(
            corpus,
            vector,
            bm25,
        )


def test_duplicate_corpus_ids_are_rejected(corpus):
    corpus.chunks.append(corpus.chunks[0])
    vector, bm25 = fake_retrievers(corpus)

    with pytest.raises(ValueError, match="duplicate"):
        HybridRetriever(
            corpus,
            vector,
            bm25,
        )


def test_invalid_parent_links_are_rejected(corpus):
    bad = Corpus(
        corpus.documents,
        [],
        corpus.chunks,
    )
    vector, bm25 = fake_retrievers(bad)

    with pytest.raises(ValueError, match="parent link"):
        HybridRetriever(
            bad,
            vector,
            bm25,
        )


def test_query_errors_propagate(corpus):
    vector, bm25 = fake_retrievers(corpus)
    vector.search.side_effect = RuntimeError(
        "API unavailable"
    )

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    with pytest.raises(RuntimeError, match="API unavailable"):
        retriever.search("question")


def test_parent_expansion_retains_complete_section(corpus):
    vector, bm25 = fake_retrievers(corpus)

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    section = corpus.sections[0]

    corpus.sections.clear()
    corpus.chunks.clear()

    assert retriever.get_section(section.id) == section
    assert len(retriever) == 3


@pytest.mark.parametrize(
    "question",
    ["", " ", None, 123],
)
def test_invalid_questions_are_rejected(corpus, question):
    vector, bm25 = fake_retrievers(corpus)

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    with pytest.raises(ValueError, match="question"):
        retriever.search(question)

    vector.search.assert_not_called()


@pytest.mark.parametrize(
    "top_k",
    [0, -1, True, 1.5],
)
def test_invalid_top_k_is_rejected(corpus, top_k):
    vector, bm25 = fake_retrievers(corpus)

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    with pytest.raises(ValueError, match="top_k"):
        retriever.search(
            "question",
            top_k=top_k,
        )

    vector.search.assert_not_called()


def test_string_filter_is_rejected(corpus):
    vector, bm25 = fake_retrievers(corpus)

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    with pytest.raises(TypeError, match="collection"):
        retriever.search(
            "question",
            drugs="warfarin",
        )

    vector.search.assert_not_called()


def test_invalid_filter_values_are_rejected(corpus):
    vector, bm25 = fake_retrievers(corpus)

    retriever = HybridRetriever(
        corpus,
        vector,
        bm25,
    )

    with pytest.raises(ValueError, match="nonempty"):
        retriever.search(
            "question",
            sources=[""],
        )

    vector.search.assert_not_called()