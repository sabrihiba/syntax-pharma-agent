"""Offline answer-generation tests: no API calls or embedding requests."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent.answer import (
    ANSWER_PROMPT,
    DISCLAIMER,
    AnswerBudgetError,
    generate_answer,
)
from app.agent.consolidation import SectionEvidence, SectionSpan
from app.agent.evidence_review import CoverageReview, EvidenceCheck
from app.agent.planner import AgentProtocolError
from app.agent.reviewed_research import ReviewedResearchResult
from app.rag.chunker import Citation


SID = "dailymed-example.md#4-contraindications"
TEXT = "Example source text, including a qualification."


@pytest.fixture
def research():
    citation = Citation(
        id=SID,
        file="dailymed-example.md",
        source_label="DailyMed FDA label",
        url="https://example.com/label",
        section_number="4",
        section_path="Example > 4 CONTRAINDICATIONS",
    )

    section = SectionEvidence(
        section_id=SID,
        parent_section_id=None,
        drug="example",
        source="dailymed",
        citation=citation,
        spans=(SectionSpan(0, len(TEXT), TEXT),),
        excerpts=(),
        total_chars=len(TEXT),
        missing_ranges=(),
        complete=True,
        matched_chunk_ids=(),
        tool_call_ids=("call-1",),
        score=0.03,
    )

    review = CoverageReview((
        EvidenceCheck(
            1,
            True,
            True,
            (SID,),
            "Source coverage is sufficient.",
        ),
    ))

    return ReviewedResearchResult(
        question="What does the example label state?",
        plan=("Read the relevant section.",),
        evidence=(),
        trace=(),
        status="evidence_reviewed",
        notes="Coverage accepted.",
        sections=(section,),
        reviews=(review,),
        collection_status="model_finished",
    )


def payload():
    return {
        "statements": [
            {
                "text": "Example source-based statement.",
                "section_ids": [SID],
            },
            {
                "text": "A qualification is preserved.",
                "section_ids": [SID],
            },
        ],
        "limitations": [],
    }


def fake_client(
    data=None,
    *,
    content=None,
    finish_reason="stop",
):
    client = Mock()

    if content is None:
        content = json.dumps(
            payload() if data is None else data
        )

    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    tool_calls=None,
                ),
            )
        ]
    )
    return client


def draft(research, client=None, **kwargs):
    return generate_answer(
        research,
        client=client if client is not None else fake_client(),
        model="configured-test-model",
        **kwargs,
    )


def test_valid_draft_has_trusted_deduplicated_citations(research):
    result = draft(research)

    assert result.status == "draft"
    assert len(result.statements) == 2
    assert result.citations == (
        research.sections[0].citation,
    )

    text = result.to_markdown()

    assert text.count("[S1]") == 3
    assert DISCLAIMER in text
    assert "grounding review pending" in text
    assert research.sections[0].citation.label in text


def test_request_contains_complete_text_and_excludes_collection_notes(
    research,
):
    client = fake_client()

    draft(
        replace(
            research,
            notes="PRIVATE COLLECTION SUMMARY",
        ),
        client,
    )

    request = client.chat.completions.create.call_args.kwargs
    data = json.loads(
        request["messages"][1]["content"]
    )

    assert (
        data["delivered_evidence"][0]["spans"][0]["text"]
        == TEXT
    )
    assert "PRIVATE COLLECTION SUMMARY" not in str(request)
    assert request["messages"][0]["content"] == ANSWER_PROMPT
    assert request["max_completion_tokens"] == 4000

    client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize(
    "status",
    [
        "model_finished",
        "review_limit",
        "context_limit",
    ],
)
def test_unreviewed_status_blocks_api(research, status):
    client = fake_client()

    with pytest.raises(
        ValueError,
        match="reviewed evidence",
    ):
        draft(
            replace(research, status=status),
            client,
        )

    client.chat.completions.create.assert_not_called()


def test_negative_or_missing_review_blocks_api(research):
    negative = CoverageReview((
        EvidenceCheck(
            1,
            False,
            True,
            (),
            "Missing evidence.",
        ),
    ))

    for reviews in ((), (negative,)):
        with pytest.raises(
            ValueError,
            match="reviewed evidence",
        ):
            draft(
                replace(research, reviews=reviews)
            )


def test_oversized_request_preserves_sources_and_skips_api(
    research,
):
    client = fake_client()

    with pytest.raises(
        AnswerBudgetError,
        match="no evidence was truncated",
    ):
        draft(
            research,
            client,
            max_context_bytes=1,
        )

    client.chat.completions.create.assert_not_called()
    assert research.sections[0].full_text == TEXT


@pytest.mark.parametrize(
    "ids",
    [
        [],
        ["invented"],
        [SID, SID],
        [42],
        SID,
    ],
)
def test_invalid_citations_rejected(research, ids):
    data = payload()
    data["statements"][0]["section_ids"] = ids

    with pytest.raises(AgentProtocolError):
        draft(research, fake_client(data))


@pytest.mark.parametrize(
    "data",
    [
        {},
        {
            "statements": [],
            "limitations": [],
        },
        {
            "statements": "wrong",
            "limitations": [],
        },
        {
            "statements": [{}],
            "limitations": [],
        },
        {
            "statements": [
                {
                    "text": "",
                    "section_ids": [SID],
                }
            ],
            "limitations": [],
        },
        {
            "statements": [
                {
                    "text": "Claim [S99]",
                    "section_ids": [SID],
                }
            ],
            "limitations": [],
        },
        {
            "statements": [
                {
                    "text": "Claim",
                    "section_ids": [SID],
                }
            ],
            "limitations": "wrong",
        },
        {
            "statements": [
                {
                    "text": "Claim",
                    "section_ids": [SID],
                }
            ],
            "limitations": [42],
        },
    ],
)
def test_invalid_answer_schema_rejected(research, data):
    with pytest.raises(AgentProtocolError):
        draft(research, fake_client(data))


@pytest.mark.parametrize(
    "reason",
    [
        "length",
        "content_filter",
        "tool_calls",
    ],
)
def test_incomplete_or_blocked_generation_rejected(
    research,
    reason,
):
    with pytest.raises(AgentProtocolError):
        draft(
            research,
            fake_client(finish_reason=reason),
        )


def test_malformed_json_rejected(research):
    with pytest.raises(
        AgentProtocolError,
        match="valid JSON",
    ):
        draft(
            research,
            fake_client(content="Not JSON"),
        )


def test_empty_choices_rejected(research):
    client = fake_client()
    client.chat.completions.create.return_value.choices = []

    with pytest.raises(
        AgentProtocolError,
        match="no choices",
    ):
        draft(research, client)


def test_partial_text_keeps_coverage_metadata(research):
    partial = replace(
        research.sections[0],
        spans=(SectionSpan(0, 7, TEXT[:7]),),
        complete=False,
        missing_ranges=((7, len(TEXT)),),
    )

    client = fake_client()

    draft(
        replace(research, sections=(partial,)),
        client,
    )

    request = client.chat.completions.create.call_args.kwargs
    section = json.loads(
        request["messages"][1]["content"]
    )["delivered_evidence"][0]

    assert section["complete"] is False
    assert section["missing_ranges"] == [
        [7, len(TEXT)]
    ]
    assert section["spans"][0]["text"] == TEXT[:7]


def test_empty_or_duplicate_sections_rejected(research):
    for sections in (
        (),
        research.sections * 2,
    ):
        with pytest.raises(ValueError):
            draft(
                replace(
                    research,
                    sections=sections,
                )
            )


def test_citation_identity_must_match_section(research):
    section = research.sections[0]

    bad = replace(
        section,
        citation=replace(
            section.citation,
            id="other",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Citation ID",
    ):
        draft(
            replace(research, sections=(bad,))
        )


def test_limitations_rendered(research):
    data = payload()
    data["limitations"] = [
        "The corpus is a snapshot."
    ]

    result = draft(research, fake_client(data))

    assert (
        "The corpus is a snapshot."
        in result.to_markdown()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_context_bytes": 0},
        {"max_context_bytes": True},
        {"max_output_tokens": 0},
    ],
)
def test_invalid_budgets(research, kwargs):
    with pytest.raises(ValueError):
        draft(research, **kwargs)