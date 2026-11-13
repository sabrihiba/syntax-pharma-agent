"""Offline tests for delivered-evidence consolidation."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent.consolidation import consolidate_evidence
from app.agent.tools import AgentTools
from app.rag.chunker import chunk_document, load_corpus
from app.rag.hybrid import HybridResult


@pytest.fixture
def context():
    _, sections, chunks = chunk_document(
        "dailymed-example.md",
        "# Example\n"
        "Source: https://example.com/label\n"
        "## 4 CONTRAINDICATIONS\n"
        "abcdefghij",
    )

    retriever = Mock()
    retriever.get_section.return_value = sections[0]
    retriever.search.return_value = [
        HybridResult(chunks[0], 0.03)
    ]

    return SimpleNamespace(
        tools=AgentTools(retriever),
        retriever=retriever,
        section=sections[0],
        chunk=chunks[0],
    )


def page(context, start=0, size=8000, call_id=None):
    return {
        "tool": "get_section",
        "call_id": call_id or f"page-{start}-{size}",
        "output": context.tools.get_section(
            context.section.id,
            start_char=start,
            max_chars=size,
        ),
    }


def search(
    context,
    *,
    partial=False,
    score=0.03,
    call_id="search-1",
):
    chunk = (
        replace(context.chunk, n_parts=2)
        if partial
        else context.chunk
    )

    context.retriever.search.return_value = [
        HybridResult(chunk, score)
    ]

    return {
        "tool": "search_corpus",
        "call_id": call_id,
        "output": context.tools.search_corpus("question"),
    }


def test_full_section_preserves_exact_source_and_citation(context):
    item = consolidate_evidence([
        page(context)
    ])[0]

    assert item.complete
    assert item.full_text == context.section.text
    assert item.total_chars == 10
    assert item.missing_ranges == ()
    assert item.citation.file == "dailymed-example.md"
    assert item.citation.url == "https://example.com/label"
    assert len(item.spans) == 1

    json.dumps(item.to_dict(), allow_nan=False)


def test_search_previews_alone_do_not_prove_completeness(context):
    item = consolidate_evidence([
        search(context, partial=True)
    ])[0]

    assert not item.complete
    assert item.full_text is None
    assert item.total_chars is None
    assert item.missing_ranges is None
    assert item.excerpts[0].text == context.chunk.text


def test_already_complete_search_hit_can_establish_coverage(context):
    item = consolidate_evidence([
        search(context)
    ])[0]

    assert item.complete
    assert item.full_text == context.section.text
    assert item.excerpts == ()


def test_adjacent_pages_reconstruct_complete_section(context):
    item = consolidate_evidence([
        page(context, 0, 4),
        page(context, 4, 6),
    ])[0]

    assert item.complete
    assert item.full_text == "abcdefghij"


def test_out_of_order_pages_are_merged(context):
    item = consolidate_evidence([
        page(context, 7, 3),
        page(context, 0, 3),
        page(context, 3, 4),
    ])[0]

    assert item.complete
    assert item.full_text == context.section.text


def test_matching_overlap_is_not_duplicated(context):
    item = consolidate_evidence([
        page(context, 0, 7),
        page(context, 4, 6),
    ])[0]

    assert item.full_text == context.section.text
    assert len(item.spans) == 1


def test_duplicate_and_contained_pages_do_not_duplicate_text(context):
    record = page(context)

    item = consolidate_evidence([
        record,
        copy.deepcopy(record),
        page(context, 2, 3),
    ])[0]

    assert item.full_text == context.section.text
    assert len(item.spans) == 1
    assert len(item.tool_call_ids) == 2


def test_gap_is_reported_even_if_last_page_was_read(context):
    item = consolidate_evidence([
        page(context, 0, 3),
        page(context, 7, 3),
    ])[0]

    assert not item.complete
    assert item.full_text is None
    assert item.missing_ranges == ((3, 7),)
    assert [
        span.text
        for span in item.spans
    ] == ["abc", "hij"]


def test_tail_only_page_reports_missing_beginning(context):
    item = consolidate_evidence([
        page(context, 5, 5)
    ])[0]

    assert not item.complete
    assert item.missing_ranges == ((0, 5),)


def test_duplicate_search_hits_keep_ids_calls_and_highest_score(context):
    item = consolidate_evidence([
        search(context, partial=True, score=0.02),
        search(
            context,
            partial=True,
            score=0.04,
            call_id="search-2",
        ),
    ])[0]

    assert len(item.excerpts) == 1
    assert item.matched_chunk_ids == (context.chunk.id,)
    assert item.tool_call_ids == (
        "search-1",
        "search-2",
    )
    assert item.score == 0.04


def test_full_pages_supersede_redundant_previews(context):
    item = consolidate_evidence([
        search(context, partial=True),
        page(context),
    ])[0]

    assert item.complete
    assert item.excerpts == ()
    assert item.matched_chunk_ids == (context.chunk.id,)
    assert len(item.tool_call_ids) == 2


def test_conflicting_overlapping_text_is_rejected(context):
    second = page(context, 4, 6)
    second["output"]["text"] = "XYZhij"

    with pytest.raises(ValueError, match="overlapping"):
        consolidate_evidence([
            page(context, 0, 7),
            second,
        ])


def test_conflicting_section_lengths_are_rejected(context):
    second = page(context, 6, 4)
    second["output"].update(
        total_chars=11,
        has_more=True,
        next_start_char=10,
    )

    with pytest.raises(ValueError, match="lengths"):
        consolidate_evidence([
            page(context, 0, 3),
            second,
        ])


def test_conflicting_citations_are_rejected(context):
    second = page(context, 5, 5)
    second["output"]["citation"]["url"] = (
        "https://different.example"
    )

    with pytest.raises(ValueError, match="metadata"):
        consolidate_evidence([
            page(context, 0, 5),
            second,
        ])


def test_conflicting_duplicate_chunk_text_is_rejected(context):
    first = search(context, partial=True)
    second = copy.deepcopy(first)
    second["output"]["results"][0]["text"] = "different"

    with pytest.raises(ValueError, match="chunk ID"):
        consolidate_evidence([
            first,
            second,
        ])


@pytest.mark.parametrize(
    "change",
    [
        {"start_char": True},
        {"end_char": 9},
        {"total_chars": -1},
        {"section_complete": False},
        {"has_more": True},
        {"next_start_char": 10},
    ],
)
def test_invalid_page_metadata_is_rejected(context, change):
    record = page(context)
    record["output"].update(change)

    with pytest.raises(ValueError):
        consolidate_evidence([record])


def test_citation_id_must_match_section(context):
    record = page(context)
    record["output"]["citation"]["id"] = "wrong"

    with pytest.raises(ValueError, match="Citation ID"):
        consolidate_evidence([record])


def test_truncated_search_cannot_claim_complete_section(context):
    record = search(context)
    record["output"]["results"][0]["text_truncated"] = True

    with pytest.raises(ValueError, match="Truncated"):
        consolidate_evidence([record])


@pytest.mark.parametrize(
    "score",
    [-1, float("nan"), float("inf"), True],
)
def test_invalid_search_scores(context, score):
    record = search(context)
    record["output"]["results"][0]["score"] = score

    with pytest.raises(ValueError, match="score"):
        consolidate_evidence([record])


def test_empty_and_failed_results_produce_no_evidence():
    assert consolidate_evidence([]) == ()

    assert consolidate_evidence([
        {
            "tool": "search_corpus",
            "call_id": "empty",
            "output": {
                "ok": True,
                "results": [],
            },
        }
    ]) == ()

    assert consolidate_evidence([
        {
            "tool": "search_corpus",
            "call_id": "failed",
            "output": {
                "ok": False,
                "error": {
                    "code": "context_budget",
                },
            },
        }
    ]) == ()


def test_unknown_successful_tool_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        consolidate_evidence([
            {
                "tool": "invented",
                "call_id": "a",
                "output": {"ok": True},
            }
        ])


def test_records_are_not_modified(context):
    records = [
        search(context, partial=True),
        page(context),
    ]
    before = copy.deepcopy(records)

    consolidate_evidence(records)

    assert records == before


def test_unicode_page_offsets_preserve_exact_characters(context):
    section = replace(
        context.section,
        text="A\u00e9\U0001f48aBC",
    )
    context.retriever.get_section.return_value = section

    item = consolidate_evidence([
        page(context, 0, 3),
        page(context, 3, 2),
    ])[0]

    assert item.complete
    assert item.full_text == section.text


def test_empty_section_is_complete_only_after_explicit_read(context):
    context.retriever.get_section.return_value = replace(
        context.section,
        text="",
    )

    item = consolidate_evidence([
        page(context)
    ])[0]

    assert item.complete
    assert item.full_text == ""
    assert item.total_chars == 0


def test_long_real_metformin_section_reconstructs_without_loss():
    corpus = load_corpus(
        Path(__file__).resolve().parents[1] / "corpus"
    )

    section_id = (
        "dailymed-metformin.md#5.1-lactic-acidosis"
    )
    section = corpus.section(section_id)

    assert section is not None

    retriever = Mock()
    retriever.get_section.side_effect = corpus.section
    tools = AgentTools(retriever)

    records = []
    start = 0

    while True:
        output = tools.get_section(
            section_id,
            start_char=start,
            max_chars=700,
        )

        records.append({
            "tool": "get_section",
            "call_id": f"page-{start}",
            "output": output,
        })

        if not output["has_more"]:
            break

        start = output["next_start_char"]

    # Include a duplicate and reverse the delivery order.
    item = consolidate_evidence([
        *reversed(records),
        records[0],
    ])[0]

    assert item.complete
    assert item.full_text == section.text
    assert item.total_chars == len(section.text)
    assert item.missing_ranges == ()