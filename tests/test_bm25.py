"""Offline tests for BM25 scoring, filtering, and corpus integration."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from app.rag.bm25 import BM25Config, BM25Index, tokenize
from app.rag.chunker import Corpus, chunk_document, load_corpus


def make_corpus(
    texts=("alpha alpha beta", "beta gamma", "delta"),
):
    names = (
        "dailymed-warfarin.md",
        "dailymed-apixaban.md",
        "medlineplus-warfarin.md",
    )

    documents, sections, chunks = [], [], []

    for name, text in zip(names, texts, strict=True):
        doc, parents, children = chunk_document(
            name,
            f"# Document\n## Summary\n{text}",
        )

        # Isolate scoring mathematics from contextual headers.
        children = [
            replace(chunk, embed_text=text)
            for chunk in children
        ]

        documents.append(doc)
        sections.extend(parents)
        chunks.extend(children)

    return Corpus(documents, sections, chunks)


@pytest.fixture
def index():
    return BM25Index.build(make_corpus())


def test_build_records_chunk_count_and_average_length(index):
    assert len(index) == 3
    assert index.average_length == pytest.approx(2.0)


def test_score_matches_hand_calculated_bm25(index):
    hit = index.search("alpha")[0]

    # N=3; df(alpha)=1; tf=2; length=3; average length=2.
    idf = math.log1p(
        (3 - 1 + 0.5) / (1 + 0.5)
    )
    expected = idf * (2 * 2.5) / (
        2 + 1.5 * (
            1 - 0.75 + 0.75 * (3 / 2)
        )
    )

    assert hit.score == pytest.approx(expected)
    assert hit.chunk.doc.file == "dailymed-warfarin.md"


def test_multiple_terms_add_their_contributions(index):
    alpha_score = index.search("alpha")[0].score

    beta_score = next(
        hit.score
        for hit in index.search("beta")
        if hit.chunk.doc.file == "dailymed-warfarin.md"
    )

    combined = next(
        hit.score
        for hit in index.search("alpha beta")
        if hit.chunk.doc.file == "dailymed-warfarin.md"
    )

    assert combined == pytest.approx(
        alpha_score + beta_score
    )


def test_repeated_query_terms_do_not_change_scores(index):
    assert index.search("alpha") == index.search(
        "alpha alpha alpha"
    )


def test_shorter_chunk_ranks_higher_for_equal_term_frequency():
    index = BM25Index(
        make_corpus(
            (
                "target filler filler",
                "target",
                "unrelated",
            )
        )
    )

    assert (
        index.search("target")[0].chunk.doc.file
        == "dailymed-apixaban.md"
    )


def test_higher_term_frequency_increases_score_for_equal_lengths():
    index = BM25Index(
        make_corpus(
            (
                "target target x",
                "target x x",
                "unrelated",
            )
        )
    )

    assert (
        index.search("target")[0].chunk.doc.file
        == "dailymed-warfarin.md"
    )


def test_zero_b_disables_length_normalization():
    index = BM25Index(
        make_corpus(
            (
                "target filler filler",
                "target",
                "unrelated",
            )
        ),
        BM25Config(b=0),
    )

    hits = index.search("target")

    assert hits[0].score == pytest.approx(
        hits[1].score
    )


def test_common_terms_still_have_positive_scores():
    index = BM25Index(
        make_corpus(
            ("common", "common", "common")
        )
    )

    assert len(index.search("common")) == 3
    assert all(
        hit.score > 0
        for hit in index.search("common")
    )


def test_equal_scores_keep_corpus_order():
    corpus = make_corpus(
        ("common", "common", "common")
    )
    index = BM25Index(corpus)

    assert [
        hit.chunk.id
        for hit in index.search("common")
    ] == [
        chunk.id
        for chunk in corpus.chunks
    ]


def test_top_k_limits_results(index):
    assert len(
        index.search(
            "alpha beta gamma delta",
            top_k=1,
        )
    ) == 1

    assert len(
        index.search(
            "alpha beta gamma delta",
            top_k=100,
        )
    ) == 3


def test_no_match_returns_no_results(index):
    assert index.search("notincorpus") == []
    assert index.search("!!!") == []


def test_zero_threshold_does_not_return_unmatched_chunks(index):
    hits = index.search(
        "alpha",
        min_score=0,
    )

    assert len(hits) == 1


def test_min_score_is_inclusive_and_not_limited_to_one(index):
    score = index.search("alpha")[0].score

    assert len(
        index.search("alpha", min_score=score)
    ) == 1

    assert index.search(
        "alpha",
        min_score=score + 1,
    ) == []

    assert index.search(
        "alpha",
        min_score=1000,
    ) == []


def test_filters_are_applied_before_top_k():
    index = BM25Index(
        make_corpus(
            ("common", "common", "common")
        )
    )

    hits = index.search(
        "common",
        top_k=1,
        drugs=["apixaban"],
    )

    assert len(hits) == 1
    assert hits[0].chunk.doc.drug == "apixaban"


def test_multiple_drugs_and_source_filter():
    index = BM25Index(
        make_corpus(
            ("common", "common", "common")
        )
    )

    hits = index.search(
        "common",
        drugs=[" WARFARIN ", "Apixaban"],
        sources=["DAILYMED"],
    )

    assert len(hits) == 2
    assert all(
        hit.chunk.doc.source == "dailymed"
        for hit in hits
    )


def test_filters_do_not_recalculate_corpus_statistics(index):
    unfiltered = index.search("beta")
    filtered = index.search(
        "beta",
        drugs=["apixaban"],
    )

    expected = next(
        hit.score
        for hit in unfiltered
        if hit.chunk.doc.drug == "apixaban"
    )

    assert filtered[0].score == expected


def test_empty_or_unknown_filters_return_no_results(index):
    assert index.search(
        "alpha",
        drugs=[],
    ) == []

    assert index.search(
        "alpha",
        drugs=["unknown"],
    ) == []

    assert index.search(
        "alpha",
        sources=[],
    ) == []


def test_citations_and_parent_expansion_are_preserved(index):
    hit = index.search("alpha")[0]

    assert hit.citation == hit.chunk.citation

    parent = index.get_section(
        hit.chunk.section_id
    )

    assert parent is not None
    assert hit.chunk.text in parent.text
    assert index.get_section("missing") is None


def test_contextual_text_indexes_drug_and_section_names():
    doc, sections, chunks = chunk_document(
        "dailymed-warfarin.md",
        "# Warfarin\n"
        "## 4 CONTRAINDICATIONS\n"
        "Example passage.",
    )

    index = BM25Index(
        Corpus([doc], sections, chunks)
    )

    assert index.search("warfarin")
    assert index.search("contraindications")


def test_parent_expansion_includes_subsections():
    doc, sections, chunks = chunk_document(
        "dailymed-example.md",
        "# Example\n"
        "## Warnings\n"
        "Intro.\n"
        "### First\n"
        "First passage.\n"
        "### Second\n"
        "Second passage.",
    )

    index = BM25Index(
        Corpus([doc], sections, chunks)
    )
    parent = index.get_section(sections[0].id)

    assert "First passage." in parent.text
    assert "Second passage." in parent.text


def test_external_list_mutations_do_not_change_index():
    corpus = make_corpus()
    index = BM25Index(corpus)

    corpus.chunks.clear()
    corpus.sections.clear()

    assert len(index) == 3

    hit = index.search("alpha")[0]

    assert index.get_section(
        hit.chunk.section_id
    ) is not None


def test_empty_corpus():
    index = BM25Index(
        Corpus([], [], [])
    )

    assert len(index) == 0
    assert index.average_length == 0
    assert index.search("question") == []


def test_no_token_corpus_is_safe():
    corpus = make_corpus()
    corpus.chunks = [
        replace(chunk, embed_text="!!!")
        for chunk in corpus.chunks
    ]

    index = BM25Index(corpus)

    assert index.average_length == 0
    assert index.search("alpha") == []


def test_tokenizer_handles_clinical_terms_and_decimals():
    assert tokenize(
        "CYP3A4 eGFR 25 mL/min 2.5 mg P-gp"
    ) == [
        "cyp3a4",
        "egfr",
        "25",
        "ml",
        "min",
        "2.5",
        "mg",
        "p",
        "gp",
    ]


def test_tokenizer_normalizes_unicode_and_case():
    assert tokenize(
        "WARFARIN m\u00b2 \uff12\uff15"
    ) == [
        "warfarin",
        "m2",
        "25",
    ]


def test_decimal_values_are_distinct():
    index = BM25Index(
        make_corpus(
            (
                "dose 2.5 mg",
                "dose 25 mg",
                "other",
            )
        )
    )

    hits = index.search("2.5")

    assert len(hits) == 1
    assert (
        hits[0].chunk.doc.file
        == "dailymed-warfarin.md"
    )


@pytest.mark.parametrize(
    "value",
    [None, 123, b"text"],
)
def test_tokenizer_rejects_nontext(value):
    with pytest.raises(TypeError):
        tokenize(value)


@pytest.mark.parametrize(
    "question",
    ["", " ", None, 123],
)
def test_invalid_questions(index, question):
    with pytest.raises(ValueError, match="question"):
        index.search(question)


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 1.5, True],
)
def test_invalid_top_k(index, top_k):
    with pytest.raises(ValueError, match="top_k"):
        index.search(
            "alpha",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "score",
    [
        -1,
        float("nan"),
        float("inf"),
        True,
        "1",
        1j,
    ],
)
def test_invalid_threshold(index, score):
    with pytest.raises(ValueError, match="min_score"):
        index.search(
            "alpha",
            min_score=score,
        )


def test_string_filters_are_rejected(index):
    with pytest.raises(TypeError, match="collection"):
        index.search(
            "alpha",
            drugs="warfarin",
        )


def test_invalid_filter_entries_are_rejected(index):
    with pytest.raises(ValueError, match="nonempty"):
        index.search(
            "alpha",
            sources=[""],
        )


@pytest.mark.parametrize(
    "config",
    [
        {"k1": 0},
        {"k1": -1},
        {"k1": True},
        {"k1": float("inf")},
        {"k1": "1.5"},
        {"b": -0.1},
        {"b": 1.1},
        {"b": True},
        {"b": float("nan")},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        BM25Config(**config)


@pytest.mark.parametrize("b", [0, 1])
def test_length_normalization_boundaries_are_accepted(b):
    assert BM25Config(b=b).b == b


def test_duplicate_chunk_ids_are_rejected():
    corpus = make_corpus()
    corpus.chunks.append(corpus.chunks[0])

    with pytest.raises(
        ValueError,
        match="duplicate chunk",
    ):
        BM25Index(corpus)


def test_missing_parent_is_rejected():
    corpus = make_corpus()

    corpus = Corpus(
        corpus.documents,
        corpus.sections[1:],
        corpus.chunks,
    )

    with pytest.raises(
        ValueError,
        match="missing parent",
    ):
        BM25Index(corpus)


def test_parent_document_mismatch_is_rejected():
    corpus = make_corpus()
    corpus.chunks[0] = replace(
        corpus.chunks[0],
        doc=corpus.documents[1],
    )

    with pytest.raises(
        ValueError,
        match="parent document",
    ):
        BM25Index(corpus)


def test_real_corpus_retrieves_expected_contraindication_section():
    corpus = load_corpus(
        Path(__file__).resolve().parents[1] / "corpus"
    )
    index = BM25Index(corpus)

    hits = index.search(
        "warfarin contraindications",
        drugs=["warfarin"],
        sources=["dailymed"],
        top_k=10,
    )

    assert any(
        hit.chunk.section_number == "4"
        for hit in hits
    )

    assert all(
        hit.chunk.doc.drug == "warfarin"
        and hit.chunk.doc.source == "dailymed"
        for hit in hits
    )