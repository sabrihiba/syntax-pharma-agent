"""Offline scope, stopping-rule, and grounding regressions."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_evidence_review import (
    context,
    package,
    check_payload,
    reviewer_client,
    reply,
)
from test_agent_answer import research, fake_client
from test_agent_grounding import draft, verdict

from app.agent.evidence_review import review_evidence
from app.agent.grounding import review_answer
from app.agent.planner import AgentProtocolError
from app.agent.research import AgentConfig
from app.agent.reviewed_research import ReviewedResearchAgent


def evidence_payload(
    section_id,
    *,
    core=True,
    extra_required=False,
):
    return {
        "checks": [
            check_payload(
                section_id,
                required=True,
                supported=core,
                section_ids=[section_id] if core else [],
                explanation=(
                    "Required answer evidence."
                    if core
                    else "Find required evidence."
                ),
            ),
            {
                "step_index": 2,
                "required": extra_required,
                "supported": False,
                "requires_complete": False,
                "section_ids": [],
                "explanation": (
                    "Measured laboratory effects were not requested "
                    "and are not necessary for this risk-and-monitoring answer."
                    if not extra_required
                    else "The user explicitly requests measured effects."
                ),
            },
        ]
    }


def evidence_review(context, data, *, partial=False):
    return review_evidence(
        "What is the risk and recommended monitoring?",
        (
            "Required risk and monitoring.",
            "Optional measured laboratory effects.",
        ),
        package(context, partial),
        client=reviewer_client(data),
        model="test",
    )


def test_optional_missing_evidence_does_not_block_stopping(context):
    review = evidence_review(
        context,
        evidence_payload(context.section.id),
    )

    assert review.sufficient
    assert review.gaps == ()
    assert not review.checks[1].supported
    assert not review.checks[1].required


def test_missing_required_evidence_still_blocks(context):
    review = evidence_review(
        context,
        evidence_payload(
            context.section.id,
            core=False,
        ),
    )

    assert not review.sufficient
    assert len(review.gaps) == 1
    assert review.gaps[0].startswith("Step 1:")


def test_explicitly_required_measured_effect_still_blocks(context):
    review = evidence_review(
        context,
        evidence_payload(
            context.section.id,
            extra_required=True,
        ),
    )

    assert not review.sufficient
    assert review.gaps[0].startswith("Step 2:")


def test_required_complete_section_cannot_be_bypassed(context):
    review = evidence_review(
        context,
        evidence_payload(context.section.id),
        partial=True,
    )

    assert not review.sufficient
    assert "complete cited section" in review.gaps[0]


def test_evidence_reviewer_cannot_mark_everything_optional(context):
    data = evidence_payload(context.section.id)

    for check in data["checks"]:
        check["required"] = False

    with pytest.raises(
        AgentProtocolError,
        match="at least one required",
    ):
        evidence_review(context, data)


@pytest.mark.parametrize("value", [None, 1, "false"])
def test_evidence_required_flag_is_boolean(context, value):
    data = evidence_payload(context.section.id)
    data["checks"][1]["required"] = value

    with pytest.raises(AgentProtocolError):
        evidence_review(context, data)


def test_legacy_evidence_checks_default_to_required(context):
    review = review_evidence(
        "question",
        ("Required evidence.",),
        package(context),
        client=reviewer_client({
            "checks": [
                check_payload(context.section.id)
            ]
        }),
        model="test",
    )

    assert review.checks[0].required
    assert review.sufficient


def grounding_payload(*, extra_required=False):
    data = verdict()
    data["coverage"][0]["required"] = True

    data["coverage"].append({
        "index": 2,
        "required": extra_required,
        "passed": False,
        "explanation": "Optional background is omitted.",
    })

    return data


def grounding_review(draft, research, data):
    return review_answer(
        draft,
        replace(
            research,
            plan=(
                *research.plan,
                "Optional laboratory measurements.",
            ),
        ),
        client=fake_client(data),
        model="test",
    )


def test_grounding_ignores_omitted_optional_background(
    draft,
    research,
):
    review = grounding_review(
        draft,
        research,
        grounding_payload(),
    )

    assert review.sufficient
    assert review.issues == ()
    assert not review.coverage[1].passed


def test_grounding_requires_explicit_user_need(draft, research):
    review = grounding_review(
        draft,
        research,
        grounding_payload(extra_required=True),
    )

    assert not review.sufficient
    assert review.issues[0].startswith("Plan step 2:")


def test_optional_content_in_answer_must_still_be_grounded(
    draft,
    research,
):
    data = grounding_payload()
    data["claims"][0]["passed"] = False
    data["claims"][0]["explanation"] = (
        "Unsupported optional assertion."
    )

    review = grounding_review(draft, research, data)

    assert not review.sufficient
    assert review.issues[0].startswith("Statement 1:")


def test_grounding_cannot_exempt_claims_from_review(
    draft,
    research,
):
    data = grounding_payload()
    data["claims"][0]["required"] = False

    with pytest.raises(AgentProtocolError):
        grounding_review(draft, research, data)


def test_grounding_cannot_mark_all_coverage_optional(
    draft,
    research,
):
    data = grounding_payload()
    data["coverage"][0]["required"] = False

    with pytest.raises(
        AgentProtocolError,
        match="at least one required",
    ):
        grounding_review(draft, research, data)


@pytest.mark.parametrize("value", [None, 1, "false"])
def test_grounding_required_flag_is_boolean(
    draft,
    research,
    value,
):
    data = grounding_payload()
    data["coverage"][1]["required"] = value

    with pytest.raises(AgentProtocolError):
        grounding_review(draft, research, data)


def test_research_graph_stops_when_only_optional_evidence_is_missing(
    context,
):
    coverage = evidence_review(
        context,
        evidence_payload(context.section.id),
    )

    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(
            name="get_section",
            arguments=json.dumps({
                "section_id": context.section.id,
                "start_char": 0,
                "max_chars": 8000,
            }),
        ),
    )

    client = Mock()
    client.chat.completions.create.side_effect = [
        reply(json.dumps({
            "steps": [
                "Required risk and monitoring.",
                "Optional measured effects.",
            ]
        })),
        reply(calls=[tool_call]),
        reply("Required evidence collected."),
    ]

    reviewer = Mock(return_value=coverage)

    agent = ReviewedResearchAgent(
        context.tools,
        AgentConfig(
            model="test",
            max_turns=5,
        ),
        client=client,
        reviewer=reviewer,
    )

    result = agent.collect(
        "What is the risk and recommended monitoring?"
    )

    assert result.status == "evidence_reviewed"
    assert len(result.trace) == 1
    assert len(result.reviews) == 1
    assert result.reviews[0].sufficient
    assert client.chat.completions.create.call_count == 3

    reviewer.assert_called_once()