"""Offline coverage-review and reviewed-LangGraph tests."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent.consolidation import consolidate_evidence
from app.agent.evidence_review import (
    CoverageReview,
    EvidenceCheck,
    ReviewBudgetError,
    review_evidence,
)
from app.agent.planner import AgentProtocolError
from app.agent.research import AgentConfig
from app.agent.reviewed_research import (
    ReviewedResearchAgent,
    ReviewConfig,
)
from app.agent.tools import AgentTools
from app.rag.chunker import chunk_document
from app.rag.hybrid import HybridResult


def reply(content=None, calls=None):
    return SimpleNamespace(choices=[
        SimpleNamespace(
            finish_reason="tool_calls" if calls else "stop",
            message=SimpleNamespace(
                content=content,
                tool_calls=calls,
            ),
        )
    ])


@pytest.fixture
def context():
    _, sections, chunks = chunk_document(
        "dailymed-example.md",
        "# Example\n"
        "Source: https://example.com/label\n"
        "## 4 CONTRAINDICATIONS\n"
        "Complete source passage.",
    )

    retriever = Mock()
    retriever.get_section.return_value = sections[0]
    retriever.search.return_value = [
        HybridResult(chunks[0], 0.03)
    ]

    return SimpleNamespace(
        tools=AgentTools(retriever),
        section=sections[0],
    )


def package(context, partial=False):
    output = context.tools.get_section(
        context.section.id,
        max_chars=3 if partial else 8000,
    )

    return consolidate_evidence([
        {
            "tool": "get_section",
            "call_id": "page",
            "output": output,
        }
    ])


def check_payload(section_id, **changes):
    return {
        "step_index": 1,
        "supported": True,
        "requires_complete": True,
        "section_ids": [section_id],
        "explanation": "Relevant source evidence was delivered.",
        **changes,
    }


def reviewer_client(payload):
    client = Mock()
    client.chat.completions.create.return_value = reply(
        json.dumps(payload)
    )
    return client


def verdict(
    supported,
    section_id,
    reason="Find the missing evidence.",
):
    return CoverageReview((
        EvidenceCheck(
            step_index=1,
            supported=supported,
            requires_complete=True,
            section_ids=(section_id,) if supported else (),
            explanation=reason,
        ),
    ))


def test_reviewer_accepts_supported_complete_source(context):
    client = reviewer_client({
        "checks": [
            check_payload(context.section.id)
        ]
    })

    result = review_evidence(
        "question",
        ("Find evidence.",),
        package(context),
        client=client,
        model="test",
    )

    assert result.sufficient
    assert result.gaps == ()
    client.chat.completions.create.assert_called_once()


def test_partial_section_cannot_pass_complete_section_check(context):
    client = reviewer_client({
        "checks": [
            check_payload(context.section.id)
        ]
    })

    result = review_evidence(
        "List all conditions",
        ("Find full list.",),
        package(context, True),
        client=client,
        model="test",
    )

    assert not result.sufficient
    assert "complete cited section" in result.gaps[0]


def test_partial_text_can_support_a_nonexhaustive_fact(context):
    client = reviewer_client({
        "checks": [
            check_payload(
                context.section.id,
                requires_complete=False,
            )
        ]
    })

    result = review_evidence(
        "question",
        ("Find a passage.",),
        package(context, True),
        client=client,
        model="test",
    )

    assert result.sufficient


def test_explicit_missing_evidence_produces_actionable_gap(context):
    client = reviewer_client({
        "checks": [
            check_payload(
                context.section.id,
                supported=False,
                section_ids=[],
                explanation=(
                    "Search the other drug label "
                    "for the named interaction."
                ),
            )
        ]
    })

    result = review_evidence(
        "question",
        ("Check both labels.",),
        package(context),
        client=client,
        model="test",
    )

    assert not result.sufficient
    assert "other drug label" in result.gaps[0]


@pytest.mark.parametrize(
    "change",
    [
        {"section_ids": ["invented"]},
        {"section_ids": []},
        {"step_index": 2},
        {"step_index": True},
        {"supported": "true"},
        {"requires_complete": "yes"},
        {"explanation": ""},
        {"section_ids": ["duplicate", "duplicate"]},
    ],
)
def test_invalid_review_checks_are_rejected(context, change):
    client = reviewer_client({
        "checks": [
            check_payload(
                context.section.id,
                **change,
            )
        ]
    })

    with pytest.raises(AgentProtocolError):
        review_evidence(
            "question",
            ("Find evidence.",),
            package(context),
            client=client,
            model="test",
        )


def test_every_plan_step_must_be_checked(context):
    client = reviewer_client({
        "checks": [
            check_payload(context.section.id)
        ]
    })

    with pytest.raises(
        AgentProtocolError,
        match="every plan step",
    ):
        review_evidence(
            "question",
            ("First step.", "Second step."),
            package(context),
            client=client,
            model="test",
        )


def test_empty_evidence_is_rejected_without_model_call():
    client = Mock()

    result = review_evidence(
        "question",
        ("First.", "Second."),
        (),
        client=client,
        model="test",
    )

    assert not result.sufficient
    assert len(result.gaps) == 2
    client.chat.completions.create.assert_not_called()


def test_review_budget_rejects_without_truncating_or_calling_model(context):
    client = Mock()
    sections = package(context)
    original = sections[0].full_text

    with pytest.raises(ReviewBudgetError):
        review_evidence(
            "question",
            ("Find evidence.",),
            sections,
            client=client,
            model="test",
            max_context_bytes=1,
        )

    assert sections[0].full_text == original
    client.chat.completions.create.assert_not_called()


def test_invalid_review_json_is_rejected(context):
    client = Mock()
    client.chat.completions.create.return_value = reply(
        "not JSON"
    )

    with pytest.raises(AgentProtocolError, match="JSON"):
        review_evidence(
            "question",
            ("Find evidence.",),
            package(context),
            client=client,
            model="test",
        )


def call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        ),
    )


def search_call(call_id="search"):
    return call(
        call_id,
        "search_corpus",
        {
            "question": "example contraindications",
            "top_k": 3,
            "drugs": ["example"],
            "sources": ["dailymed"],
        },
    )


def graph_agent(
    context,
    *responses,
    reviewer,
    max_reviews=3,
    **config,
):
    client = Mock()
    client.chat.completions.create.side_effect = [
        reply(
            '{"steps":["Find the relevant evidence."]}'
        ),
        *responses,
    ]

    agent = ReviewedResearchAgent(
        context.tools,
        AgentConfig(model="test", **config),
        review_config=ReviewConfig(
            max_reviews=max_reviews
        ),
        client=client,
        reviewer=reviewer,
    )

    return agent, client


def test_reviewed_graph_has_consolidation_and_review_nodes(context):
    agent, _ = graph_agent(
        context,
        reviewer=Mock(),
    )

    assert {
        "planner",
        "model",
        "tools",
        "consolidate",
        "review",
    }.issubset(agent.graph.get_graph().nodes)


def test_graph_finishes_when_coverage_is_supported(context):
    reviewer = Mock(
        return_value=verdict(
            True,
            context.section.id,
        )
    )

    agent, _ = graph_agent(
        context,
        reply(calls=[search_call()]),
        reply("Collected."),
        reviewer=reviewer,
    )

    result = agent.collect("question")

    assert result.status == "evidence_reviewed"
    assert len(result.reviews) == 1
    assert (
        result.sections[0].full_text
        == context.section.text
    )
    assert result.collection_status == "model_finished"


def test_gap_returns_to_model_and_preserves_previous_evidence(context):
    reviewer = Mock(side_effect=[
        verdict(
            False,
            context.section.id,
            "Read the complete relevant section.",
        ),
        verdict(True, context.section.id),
    ])

    agent, client = graph_agent(
        context,
        reply(calls=[search_call()]),
        reply("Collected."),
        reply(calls=[
            call(
                "page",
                "get_section",
                {
                    "section_id": context.section.id,
                    "start_char": 0,
                    "max_chars": 8000,
                },
            )
        ]),
        reply("Follow-up collected."),
        reviewer=reviewer,
    )

    result = agent.collect("question")

    assert result.status == "evidence_reviewed"
    assert len(result.reviews) == 2
    assert len(result.evidence) == 2
    assert len(result.sections) == 1
    assert (
        result.sections[0].full_text
        == context.section.text
    )

    follow_up = (
        client.chat.completions.create
        .call_args_list[3].kwargs["messages"]
    )

    assert any(
        message["role"] == "user"
        and "evidence_review_feedback"
        in message.get("content", "")
        for message in follow_up
    )


def test_review_retries_are_bounded(context):
    reviewer = Mock(
        return_value=verdict(
            False,
            context.section.id,
        )
    )

    agent, client = graph_agent(
        context,
        reply("Nothing found."),
        reply("Still missing."),
        reviewer=reviewer,
        max_reviews=2,
    )

    result = agent.collect("question")

    assert result.status == "review_limit"
    assert len(result.reviews) == 2
    assert reviewer.call_count == 2
    assert client.chat.completions.create.call_count == 3


def test_reviewer_cannot_bypass_research_turn_limit(context):
    reviewer = Mock(
        return_value=verdict(
            False,
            context.section.id,
        )
    )

    agent, client = graph_agent(
        context,
        reply(calls=[search_call()]),
        reviewer=reviewer,
        max_turns=1,
    )

    result = agent.collect("question")

    assert result.status == "turn_limit"
    assert len(result.evidence) == 1
    assert result.sections[0].complete
    assert client.chat.completions.create.call_count == 2


def test_review_budget_failure_preserves_all_collected_evidence(context):
    reviewer = Mock(
        side_effect=ReviewBudgetError(
            "Evidence too large; retained."
        )
    )

    agent, _ = graph_agent(
        context,
        reply(calls=[search_call()]),
        reply("Collected."),
        reviewer=reviewer,
    )

    result = agent.collect("question")

    assert result.status == "review_context_limit"
    assert len(result.evidence) == 1
    assert (
        result.sections[0].full_text
        == context.section.text
    )
    assert result.reviews == ()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_review_limits(value):
    with pytest.raises(ValueError):
        ReviewConfig(max_reviews=value)