"""Offline tests for parent-section evidence expansion."""

from dataclasses import replace

import pytest

from app.rag.chunker import (
    ChunkingConfig,
    Corpus,
    chunk_document,
)
from app.rag.evidence import expand_evidence
from app.rag.hybrid import HybridResult


def make_corpus():
    config = ChunkingConfig(
        max_section_chars=100,
        protected_max_chars=150,
        target_chars=80,
        overlap_max_chars=0,
    )

    documents, sections, chunks = [], [], []

    for filename, content in [
        (
            "dailymed-warfarin.md",
            "# Warfarin\n"
            "Source: https://example.com/warfarin\n\n"
            "## 4 CONTRAINDICATIONS\n\n"
            + "\n".join(
                f"- Condition {index} requires consideration."
                for index in range(12)
            ),
        ),
        (
            "dailymed-apixaban.md",
            "# Apixaban\n"
            "Source: https://example.com/apixaban\n\n"
            "## 4 CONTRAINDICATIONS\n\n"
            "Apixaban example passage.",
        ),
    ]:
        doc, parents, children = chunk_document(
            filename,
            content,
            config,
        )
        documents.append(doc)
        sections.extend(parents)
        chunks.extend(children)

    return Corpus(documents, sections, chunks)


@pytest.fixture
def corpus():
    return make_corpus()


def chunks_for(corpus, drug):
    return [
        chunk
        for chunk in corpus.chunks
        if chunk.doc.drug == drug
    ]


def hit(chunk, score):
    return HybridResult(
        chunk=chunk,
        score=score,
    )


def test_expands_partial_chunk_to_complete_section(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]
    assert chunk.is_partial

    evidence = expand_evidence(
        [hit(chunk, 0.03)],
        corpus,
    )

    assert len(evidence) == 1

    item = evidence[0]
    parent = corpus.section(chunk.section_id)

    assert item.section == parent
    assert item.section.text == parent.text
    assert len(item.section.text) > len(chunk.text)

    for index in range(12):
        assert f"Condition {index}" in item.section.text


def test_multiple_chunks_merge_into_one_evidence_item(corpus):
    first, second = chunks_for(corpus, "warfarin")[:2]

    evidence = expand_evidence(
        [
            hit(first, 0.02),
            hit(second, 0.04),
        ],
        corpus,
    )

    assert len(evidence) == 1
    assert evidence[0].score == 0.04
    assert evidence[0].matched_chunk_ids == (
        first.id,
        second.id,
    )


def test_duplicate_chunk_ids_are_recorded_once(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]

    evidence = expand_evidence(
        [
            hit(chunk, 0.02),
            hit(chunk, 0.05),
            hit(chunk, 0.01),
        ],
        corpus,
    )

    assert evidence[0].matched_chunk_ids == (chunk.id,)
    assert evidence[0].score == 0.05


def test_sections_are_sorted_by_highest_score(corpus):
    warfarin = chunks_for(corpus, "warfarin")[0]
    apixaban = chunks_for(corpus, "apixaban")[0]

    evidence = expand_evidence(
        [
            hit(warfarin, 0.01),
            hit(apixaban, 0.04),
        ],
        corpus,
    )

    assert [
        item.section.doc.drug
        for item in evidence
    ] == ["apixaban", "warfarin"]


def test_max_sections_applies_after_grouping_and_scoring(corpus):
    first, second = chunks_for(corpus, "warfarin")[:2]
    apixaban = chunks_for(corpus, "apixaban")[0]

    evidence = expand_evidence(
        [
            hit(first, 0.01),
            hit(apixaban, 0.03),
            hit(second, 0.05),
        ],
        corpus,
        max_sections=1,
    )

    assert len(evidence) == 1
    assert evidence[0].section.doc.drug == "warfarin"
    assert evidence[0].score == 0.05
    assert evidence[0].matched_chunk_ids == (
        first.id,
        second.id,
    )


def test_equal_scores_use_section_id_for_stable_order(corpus):
    warfarin = chunks_for(corpus, "warfarin")[0]
    apixaban = chunks_for(corpus, "apixaban")[0]

    forward = expand_evidence(
        [
            hit(warfarin, 0.02),
            hit(apixaban, 0.02),
        ],
        corpus,
    )
    reverse = expand_evidence(
        [
            hit(apixaban, 0.02),
            hit(warfarin, 0.02),
        ],
        corpus,
    )

    expected = sorted([
        warfarin.section_id,
        apixaban.section_id,
    ])

    assert [item.section.id for item in forward] == expected
    assert [item.section.id for item in reverse] == expected


def test_citation_matches_retrieved_chunk(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]

    item = expand_evidence(
        [hit(chunk, 0.02)],
        corpus,
    )[0]

    assert item.citation == chunk.citation
    assert item.citation.url == "https://example.com/warfarin"


def test_empty_results_return_empty_evidence(corpus):
    assert expand_evidence([], corpus) == []


@pytest.mark.parametrize(
    "max_sections",
    [0, -1, True, 1.5, None],
)
def test_invalid_max_sections_is_rejected(corpus, max_sections):
    with pytest.raises(ValueError, match="max_sections"):
        expand_evidence(
            [],
            corpus,
            max_sections=max_sections,
        )


def test_missing_parent_is_rejected(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]

    incomplete = Corpus(
        documents=corpus.documents,
        sections=[
            section
            for section in corpus.sections
            if section.id != chunk.section_id
        ],
        chunks=corpus.chunks,
    )

    with pytest.raises(ValueError, match="Missing parent"):
        expand_evidence(
            [hit(chunk, 0.02)],
            incomplete,
        )


def test_foreign_chunk_is_rejected(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]
    foreign = replace(
        chunk,
        id=chunk.id + "-foreign",
    )

    with pytest.raises(ValueError, match="outside"):
        expand_evidence(
            [hit(foreign, 0.02)],
            corpus,
        )


def test_stale_chunk_content_is_rejected(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]
    stale = replace(
        chunk,
        text="Changed text",
    )

    with pytest.raises(ValueError, match="stale"):
        expand_evidence(
            [hit(stale, 0.02)],
            corpus,
        )


def test_parent_document_mismatch_is_rejected(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]

    changed = replace(
        chunk,
        doc=corpus.documents[1],
    )

    mismatched = Corpus(
        documents=corpus.documents,
        sections=corpus.sections,
        chunks=[
            changed if item.id == chunk.id else item
            for item in corpus.chunks
        ],
    )

    with pytest.raises(ValueError, match="Parent document"):
        expand_evidence(
            [hit(changed, 0.02)],
            mismatched,
        )


@pytest.mark.parametrize(
    "score",
    [
        -0.1,
        float("nan"),
        float("inf"),
        True,
        "0.02",
        None,
    ],
)
def test_invalid_scores_are_rejected(corpus, score):
    chunk = chunks_for(corpus, "warfarin")[0]

    with pytest.raises(ValueError, match="scores"):
        expand_evidence(
            [hit(chunk, score)],
            corpus,
        )


def test_zero_score_is_accepted(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]

    evidence = expand_evidence(
        [hit(chunk, 0)],
        corpus,
    )

    assert evidence[0].score == 0.0


def test_all_results_are_validated_before_limiting(corpus):
    warfarin = chunks_for(corpus, "warfarin")[0]
    apixaban = chunks_for(corpus, "apixaban")[0]
    stale = replace(
        apixaban,
        text="Stale text",
    )

    with pytest.raises(ValueError, match="stale"):
        expand_evidence(
            [
                hit(warfarin, 0.05),
                hit(stale, 0.01),
            ],
            corpus,
            max_sections=1,
        )


def test_duplicate_corpus_chunk_ids_are_rejected(corpus):
    corpus.chunks.append(corpus.chunks[0])

    with pytest.raises(ValueError, match="Duplicate chunk"):
        expand_evidence([], corpus)


def test_duplicate_corpus_section_ids_are_rejected(corpus):
    corpus.sections.append(corpus.sections[0])

    with pytest.raises(ValueError, match="Duplicate section"):
        expand_evidence([], corpus)


def test_expansion_uses_immediate_section_not_ancestor():
    doc, sections, chunks = chunk_document(
        "dailymed-example.md",
        "# Example\n"
        "## 5 WARNINGS\n"
        "General introduction.\n"
        "### 5.1 First Warning\n"
        "First warning passage.\n"
        "### 5.2 Second Warning\n"
        "Second warning passage.",
    )
    corpus = Corpus([doc], sections, chunks)

    chunk = next(
        chunk
        for chunk in chunks
        if chunk.section_number == "5.1"
    )

    item = expand_evidence(
        [hit(chunk, 0.03)],
        corpus,
    )[0]

    assert item.section.number == "5.1"
    assert item.section.text == "First warning passage."
    assert "Second warning passage." not in item.section.text


def test_expanding_ancestor_retains_descendant_content():
    doc, sections, chunks = chunk_document(
        "dailymed-example.md",
        "# Example\n"
        "## 5 WARNINGS\n"
        "General introduction.\n"
        "### 5.1 First Warning\n"
        "First warning passage.\n"
        "### 5.2 Second Warning\n"
        "Second warning passage.",
    )
    corpus = Corpus([doc], sections, chunks)

    chunk = next(
        chunk
        for chunk in chunks
        if chunk.section_number == "5"
    )

    item = expand_evidence(
        [hit(chunk, 0.03)],
        corpus,
    )[0]

    assert item.section.number == "5"
    assert "First warning passage." in item.section.text
    assert "Second warning passage." in item.section.text


def test_expansion_does_not_modify_results_or_corpus(corpus):
    chunk = chunks_for(corpus, "warfarin")[0]
    results = [hit(chunk, 0.02)]

    original_chunks = list(corpus.chunks)
    original_sections = list(corpus.sections)
    original_results = list(results)

    expand_evidence(results, corpus)

    assert corpus.chunks == original_chunks
    assert corpus.sections == original_sections
    assert results == original_results