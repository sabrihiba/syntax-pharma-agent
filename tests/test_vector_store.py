"""Offline vector-store tests.

Uses synthetic vectors and mocked embeddings.
No credentials or API requests required.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.rag.chunker import Corpus, chunk_document
from app.rag.vector_store import VectorStore


def make_corpus():
    documents, sections, chunks = [], [], []

    for name, content in [
        (
            "dailymed-warfarin.md",
            "# Warfarin\n"
            "## 4 CONTRAINDICATIONS\n"
            "Warfarin passage.",
        ),
        (
            "dailymed-apixaban.md",
            "# Apixaban\n"
            "## 4 CONTRAINDICATIONS\n"
            "Apixaban passage.",
        ),
        (
            "medlineplus-warfarin.md",
            "# Warfarin\n"
            "## Precautions\n"
            "Patient passage.",
        ),
    ]:
        doc, parents, children = chunk_document(name, content)
        documents.append(doc)
        sections.extend(parents)
        chunks.extend(children)

    return Corpus(documents, sections, chunks)


def fake_embedder(vectors=None, query=None):
    embedder = Mock()
    embedder.config = SimpleNamespace(model="test-model")
    embedder.dimension = None

    embedder.embed_chunks.return_value = (
        np.asarray(vectors, dtype=float)
        if vectors is not None
        else np.array(
            [[1, 0], [0, 1], [-1, 0]],
            dtype=float,
        )
    )

    embedder.embed_query.return_value = (
        np.asarray(query, dtype=float)
        if query is not None
        else np.array([1, 0], dtype=float)
    )

    return embedder


@pytest.fixture
def store():
    return VectorStore.build(
        make_corpus(),
        fake_embedder(),
    )


def test_build_embeds_chunks_once_in_corpus_order():
    corpus = make_corpus()
    embedder = fake_embedder()

    store = VectorStore.build(corpus, embedder)

    embedder.embed_chunks.assert_called_once_with(corpus.chunks)
    embedder.embed_query.assert_not_called()

    assert len(store) == 3
    assert store.dimension == 2


def test_search_ranks_by_cosine_similarity(store):
    hits = store.search("question")

    assert [hit.chunk.doc.file for hit in hits] == [
        "dailymed-warfarin.md",
        "dailymed-apixaban.md",
        "medlineplus-warfarin.md",
    ]

    np.testing.assert_allclose(
        [hit.score for hit in hits],
        [1, 0, -1],
    )


def test_top_k_limits_results(store):
    assert len(store.search("question", top_k=1)) == 1
    assert len(store.search("question", top_k=100)) == 3


def test_minimum_score_filters_results(store):
    assert len(store.search("question", min_score=0)) == 2
    assert len(store.search("question", min_score=0.1)) == 1
    assert len(store.search("question", min_score=-1)) == 3


def test_filters_are_applied_before_top_k(store):
    hits = store.search(
        "question",
        top_k=1,
        drugs=["apixaban"],
    )

    assert len(hits) == 1
    assert hits[0].chunk.doc.drug == "apixaban"


def test_multiple_drugs_and_source_filter(store):
    hits = store.search(
        "question",
        drugs=["WARFARIN", "Apixaban"],
        sources=[" DailyMed "],
    )

    assert len(hits) == 2
    assert {hit.chunk.doc.drug for hit in hits} == {
        "warfarin",
        "apixaban",
    }
    assert all(
        hit.chunk.doc.source == "dailymed"
        for hit in hits
    )


def test_empty_or_unknown_filters_avoid_query_embedding():
    embedder = fake_embedder()
    store = VectorStore.build(make_corpus(), embedder)

    assert store.search("question", drugs=[]) == []
    assert store.search("question", drugs=["unknown"]) == []
    assert store.search("question", sources=[]) == []

    embedder.embed_query.assert_not_called()


def test_precomputed_query_avoids_embedding_api(store):
    hits = store.search_by_vector(
        np.array([0, 2]),
        top_k=1,
    )

    assert hits[0].chunk.doc.drug == "apixaban"
    store._embedder.embed_query.assert_not_called()


def test_nonunit_vectors_use_cosine_not_magnitude():
    embedder = fake_embedder(
        vectors=[[3, 4], [100, 0], [-3, -4]],
        query=[30, 40],
    )
    store = VectorStore.build(make_corpus(), embedder)

    scores = [
        hit.score
        for hit in store.search("question")
    ]

    np.testing.assert_allclose(
        scores,
        [1, 0.6, -1],
        atol=1e-6,
    )


def test_ties_keep_original_chunk_order():
    corpus = make_corpus()
    store = VectorStore.build(
        corpus,
        fake_embedder(
            vectors=[[1, 0], [1, 0], [1, 0]]
        ),
    )

    assert [
        hit.chunk.id
        for hit in store.search("question")
    ] == [
        chunk.id
        for chunk in corpus.chunks
    ]


def test_results_preserve_citations_and_parent_links(store):
    hit = store.search("question", top_k=1)[0]

    assert hit.citation == hit.chunk.citation

    parent = store.get_section(hit.chunk.section_id)

    assert parent is not None
    assert hit.chunk.text in parent.text
    assert store.get_section("missing") is None


def test_parent_expansion_includes_descendants():
    doc, sections, chunks = chunk_document(
        "dailymed-example.md",
        "# Example\n"
        "## 5 WARNINGS\n"
        "Intro.\n"
        "### 5.1 First\n"
        "First passage.\n"
        "### 5.2 Second\n"
        "Second passage.",
    )
    corpus = Corpus([doc], sections, chunks)

    store = VectorStore.build(
        corpus,
        fake_embedder(
            vectors=[[1, 0]] * len(chunks)
        ),
    )

    parent = store.get_section(sections[0].id)

    assert "First passage." in parent.text
    assert "Second passage." in parent.text


def test_external_array_and_list_mutations_do_not_change_index():
    corpus = make_corpus()
    vectors = np.array(
        [[1, 0], [0, 1], [-1, 0]],
        dtype=float,
    )
    store = VectorStore(
        corpus,
        vectors,
        fake_embedder(),
    )

    vectors[:] = 0
    corpus.chunks.reverse()
    corpus.sections.clear()

    hit = store.search("question", top_k=1)[0]

    assert hit.chunk.doc.file == "dailymed-warfarin.md"
    assert store.get_section(hit.chunk.section_id) is not None


def test_empty_corpus_returns_no_results_without_query_embedding():
    embedder = fake_embedder(
        vectors=np.empty((0, 0))
    )
    store = VectorStore.build(
        Corpus([], [], []),
        embedder,
    )

    assert len(store) == 0
    assert store.search("question") == []
    assert store.search_by_vector(np.array([1, 0])) == []

    embedder.embed_query.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_rejected_before_query_embedding(store, top_k):
    with pytest.raises(ValueError, match="top_k"):
        store.search("question", top_k=top_k)

    store._embedder.embed_query.assert_not_called()


@pytest.mark.parametrize(
    "min_score",
    [
        -1.1,
        1.1,
        float("nan"),
        float("inf"),
        True,
        "0.5",
        1j,
    ],
)
def test_invalid_score_threshold_rejected(store, min_score):
    with pytest.raises(ValueError, match="min_score"):
        store.search("question", min_score=min_score)


@pytest.mark.parametrize(
    "question",
    ["", " ", None, 123],
)
def test_invalid_questions_rejected(store, question):
    with pytest.raises(ValueError, match="question"):
        store.search(question)


def test_string_filter_is_rejected(store):
    with pytest.raises(TypeError, match="collection"):
        store.search(
            "question",
            drugs="warfarin",
        )


def test_invalid_filter_entries_are_rejected(store):
    with pytest.raises(ValueError, match="nonempty"):
        store.search(
            "question",
            sources=[""],
        )


@pytest.mark.parametrize(
    "vectors",
    [
        [[1, 0]],
        [1, 2, 3],
        np.empty((3, 0)),
        [[0, 0], [1, 0], [0, 1]],
        [[float("nan"), 0], [1, 0], [0, 1]],
        [[float("inf"), 0], [1, 0], [0, 1]],
        [["a", "b"], ["c", "d"], ["e", "f"]],
        [[1j, 0], [1, 0], [0, 1]],
    ],
)
def test_invalid_stored_vectors_are_rejected(vectors):
    with pytest.raises(ValueError):
        VectorStore(
            make_corpus(),
            vectors,
            fake_embedder(),
        )


@pytest.mark.parametrize(
    "query",
    [
        [0, 0],
        [1, 0, 0],
        [[1, 0]],
        [],
        [float("nan"), 0],
        [float("inf"), 0],
        [1j, 0],
    ],
)
def test_invalid_query_vectors_are_rejected(store, query):
    with pytest.raises(ValueError):
        store.search_by_vector(query)


def test_duplicate_chunk_ids_rejected_before_embedding():
    corpus = make_corpus()
    corpus.chunks.append(corpus.chunks[0])
    embedder = fake_embedder()

    with pytest.raises(ValueError, match="duplicate chunk"):
        VectorStore.build(corpus, embedder)

    embedder.embed_chunks.assert_not_called()


def test_missing_parent_rejected_before_embedding():
    corpus = make_corpus()
    corpus.sections.pop(0)

    # Rebuild the lookup after removing a section.
    corpus = Corpus(
        corpus.documents,
        corpus.sections,
        corpus.chunks,
    )
    embedder = fake_embedder()

    with pytest.raises(ValueError, match="missing parent"):
        VectorStore.build(corpus, embedder)

    embedder.embed_chunks.assert_not_called()


def test_wrong_parent_document_is_rejected():
    corpus = make_corpus()
    corpus.chunks[0] = replace(
        corpus.chunks[0],
        doc=corpus.documents[1],
    )

    with pytest.raises(ValueError, match="parent document"):
        VectorStore.build(
            corpus,
            fake_embedder(),
        )


def test_known_embedder_dimension_must_match_store():
    embedder = fake_embedder()
    embedder.dimension = 3

    with pytest.raises(ValueError, match="dimensions"):
        VectorStore.build(
            make_corpus(),
            embedder,
        )


def test_changing_embedding_model_requires_rebuild(store):
    store._embedder.config = SimpleNamespace(
        model="different-model"
    )

    with pytest.raises(ValueError, match="rebuild"):
        store.search("question")

    store._embedder.embed_query.assert_not_called()


def test_query_api_errors_propagate(store):
    store._embedder.embed_query.side_effect = RuntimeError(
        "unavailable"
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        store.search("question")