"""Offline critic, workflow, and bounded revision tests."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from test_agent_answer import fake_client, payload, research

from app.agent.answer import (
    DISCLAIMER,
    AnswerBudgetError,
    generate_answer,
)
from app.agent.grounding import (
    GroundingBudgetError,
    review_answer,
)
from app.agent.planner import AgentProtocolError
from app.agent.research import AgentConfig
from app.agent.workflow import AnswerWorkflow


def verdict():
    return {
        "claims": [
            {
                "index": index,
                "passed": True,
                "requires_complete": False,
                "explanation": "The cited text supports this statement.",
            }
            for index in (1, 2)
        ],
        "coverage": [
            {
                "index": 1,
                "passed": True,
                "explanation": "The answer addresses the requested topic.",
            }
        ],
        "limitations_supported": True,
        "limitations_explanation": "No unsupported limitations.",
    }


@pytest.fixture
def draft(research):
    return generate_answer(
        research,
        client=fake_client(),
        model="test-model",
    )


def audit(draft, research, data=None, **kwargs):
    return review_answer(
        draft,
        research,
        client=fake_client(
            verdict() if data is None else data
        ),
        model="test-model",
        **kwargs,
    )


def test_supported_draft_passes(draft, research):
    review = audit(draft, research)
    assert review.sufficient
    assert review.issues == ()


@pytest.mark.parametrize("group", ["claims", "coverage"])
def test_failed_claim_or_omission_blocks_answer(
    draft, research, group
):
    data = verdict()
    data[group][0]["passed"] = False
    data[group][0]["explanation"] = (
        "A source qualification is missing."
    )
    review = audit(draft, research, data)
    assert not review.sufficient
    assert "qualification" in review.issues[0]


def test_unsupported_limitations_block_answer(draft, research):
    data = verdict()
    data["limitations_supported"] = False
    review = audit(draft, research, data)
    assert not review.sufficient
    assert review.issues[0].startswith("Limitations:")


def test_partial_section_cannot_support_complete_list(
    draft, research
):
    partial = replace(
        research.sections[0],
        complete=False,
    )
    data = verdict()
    data["claims"][0]["requires_complete"] = True
    review = audit(
        draft,
        replace(research, sections=(partial,)),
        data,
    )
    assert not review.sufficient
    assert not review.claims[0].passed


@pytest.mark.parametrize("problem", [
    "missing_claim",
    "duplicate_index",
    "bool_index",
    "unknown_index",
    "string_verdict",
    "missing_coverage",
    "extra_field",
    "bad_completeness",
    "bad_limitations",
    "empty_explanation",
])
def test_malformed_verdict_rejected(draft, research, problem):
    data = verdict()

    if problem == "missing_claim":
        data["claims"].pop()
    elif problem == "duplicate_index":
        data["claims"][1]["index"] = 1
    elif problem == "bool_index":
        data["claims"][0]["index"] = True
    elif problem == "unknown_index":
        data["claims"][0]["index"] = 99
    elif problem == "string_verdict":
        data["claims"][0]["passed"] = "true"
    elif problem == "missing_coverage":
        data["coverage"] = []
    elif problem == "extra_field":
        data["unexpected"] = True
    elif problem == "bad_completeness":
        data["claims"][0]["requires_complete"] = "false"
    elif problem == "bad_limitations":
        data["limitations_supported"] = "true"
    else:
        data["claims"][0]["explanation"] = ""

    with pytest.raises(AgentProtocolError):
        audit(draft, research, data)


def test_invented_citation_rejected_before_api(draft, research):
    client = fake_client(verdict())
    bad_statement = replace(
        draft.statements[0],
        section_ids=("invented",),
    )
    with pytest.raises(ValueError, match="source references"):
        review_answer(
            replace(draft, statements=(bad_statement,)),
            research,
            client=client,
            model="test-model",
        )
    client.chat.completions.create.assert_not_called()


def test_changed_citation_metadata_rejected(draft, research):
    bad_citation = replace(
        draft.citations[0],
        url="https://wrong.example",
    )
    with pytest.raises(ValueError, match="metadata"):
        audit(
            replace(draft, citations=(bad_citation,)),
            research,
        )


def test_question_mismatch_rejected(draft, research):
    with pytest.raises(ValueError, match="must match"):
        audit(
            replace(draft, question="Different question"),
            research,
        )


def test_budget_overflow_keeps_evidence_and_skips_api(
    draft, research
):
    client = fake_client(verdict())
    with pytest.raises(
        GroundingBudgetError,
        match="no evidence was truncated",
    ):
        review_answer(
            draft,
            research,
            client=client,
            model="test-model",
            max_context_bytes=1,
        )
    client.chat.completions.create.assert_not_called()
    assert research.sections[0].full_text is not None


@pytest.mark.parametrize(
    "reason",
    ["length", "content_filter", "tool_calls"],
)
def test_abnormal_completion_rejected(draft, research, reason):
    with pytest.raises(AgentProtocolError):
        review_answer(
            draft,
            research,
            client=fake_client(
                verdict(),
                finish_reason=reason,
            ),
            model="test-model",
        )


def test_invalid_json_rejected(draft, research):
    with pytest.raises(AgentProtocolError, match="valid JSON"):
        review_answer(
            draft,
            research,
            client=fake_client(content="not JSON"),
            model="test-model",
        )


def workflow(
    research,
    *,
    critic_data=None,
    critic=None,
    drafter=None,
):
    researcher = Mock()
    researcher.config = AgentConfig(model="test-model")
    researcher.collect.return_value = research

    client = Mock()
    client.chat.completions.create.side_effect = [
        fake_client(payload()).chat.completions.create.return_value,
        fake_client(
            verdict() if critic_data is None else critic_data
        ).chat.completions.create.return_value,
    ]

    # These earlier tests exercise a single attempt.
    agent = AnswerWorkflow(
        researcher,
        client=client,
        max_revisions=0,
        critic=critic,
        drafter=drafter,
    )
    return agent, client, researcher


def test_real_langgraph_runs_draft_then_critic(research):
    agent, client, researcher = workflow(research)
    result = agent.run(research.question)

    assert result.status == "answer_reviewed"
    assert result.review.sufficient
    assert client.chat.completions.create.call_count == 2

    researcher.collect.assert_called_once_with(
        research.question
    )

    text = result.to_markdown()
    assert DISCLAIMER in text
    assert "[S1]" in text
    assert "grounding review pending" not in text


def test_failed_review_retains_draft_but_blocks_final_render(
    research,
):
    data = verdict()
    data["claims"][0]["passed"] = False

    agent, client, _ = workflow(
        research,
        critic_data=data,
    )
    result = agent.run(research.question)

    assert result.status == "grounding_failed"
    assert result.draft is not None
    assert client.chat.completions.create.call_count == 2

    with pytest.raises(ValueError, match="No reviewed answer"):
        result.to_markdown()


def test_incomplete_research_skips_draft_and_critic(research):
    agent, client, _ = workflow(
        replace(research, status="review_limit")
    )
    result = agent.run(research.question)

    assert result.status == "research_incomplete"
    assert result.draft is None
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("stage", ["draft", "critic"])
def test_protocol_failure_stops_without_retry(research, stage):
    failing = Mock(
        side_effect=AgentProtocolError(
            "Malformed model output"
        )
    )
    agent, client, _ = workflow(
        research,
        drafter=failing if stage == "draft" else None,
        critic=failing if stage == "critic" else None,
    )
    result = agent.run(research.question)

    expected = (
        "draft_protocol_error"
        if stage == "draft"
        else "grounding_protocol_error"
    )

    assert result.status == expected
    assert failing.call_count == 1
    assert client.chat.completions.create.call_count == (
        0 if stage == "draft" else 1
    )


def test_grounding_budget_failure_retains_sources_and_draft(
    research,
):
    critic = Mock(
        side_effect=GroundingBudgetError(
            "Too much evidence"
        )
    )
    agent, _, _ = workflow(research, critic=critic)
    result = agent.run(research.question)

    assert result.status == "grounding_context_limit"
    assert result.draft is not None
    assert result.research.sections == research.sections


def revision_agent(
    research,
    drafts,
    reviews,
    *,
    limit=2,
    reviser=None,
):
    researcher = Mock()
    researcher.config = AgentConfig(model="test-model")
    researcher.collect.return_value = research

    drafter = Mock(return_value=drafts[0])
    reviser = (
        reviser
        if reviser is not None
        else Mock(side_effect=drafts[1:])
    )
    critic = Mock(side_effect=reviews)

    agent = AnswerWorkflow(
        researcher,
        client=Mock(),
        max_revisions=limit,
        drafter=drafter,
        critic=critic,
        reviser=reviser,
    )
    return agent, researcher, reviser, critic


def verdict_object(draft, research, passed):
    data = verdict()
    data["claims"][0]["passed"] = passed
    data["claims"][0]["explanation"] = (
        "Supported."
        if passed
        else "Cite both compared sections."
    )
    return audit(draft, research, data)


def test_failed_draft_is_revised_and_entire_new_draft_reviewed(
    draft, research
):
    revised = replace(
        draft,
        statements=(
            replace(
                draft.statements[0],
                text="Revised statement.",
            ),
            draft.statements[1],
        ),
    )

    failed = verdict_object(draft, research, False)
    passed = verdict_object(revised, research, True)

    agent, researcher, reviser, critic = revision_agent(
        research,
        [draft, revised],
        [failed, passed],
    )
    result = agent.run(research.question)

    assert result.status == "answer_reviewed"
    assert result.revisions == 1
    assert result.draft == revised
    assert len(result.attempts) == 2
    assert result.attempts[0].draft == draft
    assert not result.attempts[0].review.sufficient

    assert critic.call_args_list[1].args == (
        revised,
        research,
    )
    assert reviser.call_args.kwargs["previous_draft"] == draft
    assert reviser.call_args.kwargs["feedback"] == failed.issues
    assert (
        reviser.call_args.kwargs["max_context_bytes"]
        == 48000
    )

    researcher.collect.assert_called_once()
    assert result.research is research


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_repeated_failure_stops_at_exact_revision_limit(
    draft, research, limit
):
    failed = verdict_object(draft, research, False)

    agent, researcher, reviser, critic = revision_agent(
        research,
        [draft] * (limit + 1),
        [failed] * (limit + 1),
        limit=limit,
    )
    result = agent.run(research.question)

    assert result.status == "grounding_failed"
    assert result.revisions == limit
    assert reviser.call_count == limit
    assert critic.call_count == limit + 1
    assert len(result.attempts) == limit + 1

    researcher.collect.assert_called_once()

    with pytest.raises(ValueError):
        result.to_markdown()


def test_initial_success_never_revises(draft, research):
    passed = verdict_object(draft, research, True)
    agent, _, reviser, _ = revision_agent(
        research,
        [draft],
        [passed],
    )
    result = agent.run(research.question)

    assert result.status == "answer_reviewed"
    assert result.revisions == 0
    reviser.assert_not_called()


@pytest.mark.parametrize("error,status", [
    (
        AnswerBudgetError("Too much context"),
        "revision_context_limit",
    ),
    (
        AgentProtocolError("Invalid revision"),
        "revision_protocol_error",
    ),
])
def test_failed_revision_keeps_last_draft_review_and_sources(
    draft, research, error, status
):
    failed = verdict_object(draft, research, False)
    reviser = Mock(side_effect=error)

    agent, _, _, critic = revision_agent(
        research,
        [draft],
        [failed],
        reviser=reviser,
    )
    result = agent.run(research.question)

    assert result.status == status
    assert result.draft == draft
    assert result.review == failed
    assert result.research.sections == research.sections
    assert len(result.attempts) == 1
    assert critic.call_count == 1
    assert result.revisions == 1


def test_old_review_is_cleared_before_reviewing_new_draft(
    draft, research
):
    failed = verdict_object(draft, research, False)

    agent, _, _, _ = revision_agent(
        research,
        [draft, draft],
        [
            failed,
            AgentProtocolError(
                "Reviewer response invalid"
            ),
        ],
    )
    result = agent.run(research.question)

    assert result.status == "grounding_protocol_error"
    assert result.review is None
    assert len(result.attempts) == 1

    with pytest.raises(ValueError):
        result.to_markdown()


def test_revision_request_contains_full_sources_previous_draft_and_feedback(
    draft, research
):
    import json

    client = fake_client()

    generate_answer(
        research,
        client=client,
        model="test-model",
        previous_draft=draft,
        feedback=("Cite both compared sections.",),
    )

    request = client.chat.completions.create.call_args.kwargs
    data = json.loads(
        request["messages"][1]["content"]
    )

    assert data["delivered_evidence"] == [
        json.loads(json.dumps(section.to_dict()))
        for section in research.sections
    ]
    assert data["revision"]["critic_feedback"] == [
        "Cite both compared sections."
    ]
    assert (
        data["revision"]["previous_draft"]["statements"][0]["text"]
        == draft.statements[0].text
    )


def test_revision_feedback_is_included_in_actual_budget_check(
    draft, research
):
    from app.agent.planner import serialized_size

    client = fake_client()

    generate_answer(
        research,
        client=client,
        model="test-model",
    )

    initial_size = serialized_size(
        client.chat.completions.create.call_args.kwargs
    )
    client.reset_mock()

    with pytest.raises(AnswerBudgetError):
        generate_answer(
            research,
            client=client,
            model="test-model",
            previous_draft=draft,
            feedback=("Fix citation coverage.",),
            max_context_bytes=initial_size + 5,
        )

    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    "limit",
    [-1, 3, True, 1.5],
)
def test_invalid_revision_limits(research, limit):
    researcher = Mock()

    with pytest.raises(
        ValueError,
        match="max_revisions",
    ):
        AnswerWorkflow(
            researcher,
            client=Mock(),
            max_revisions=limit,
        )