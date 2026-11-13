"""Offline regressions for the required two-label interaction gate."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent.consolidation import SectionEvidence, SectionSpan
from app.agent.evidence_review import review_evidence
from app.agent.interaction_evidence import interaction_evidence_checks
from app.agent.research import AgentConfig
from app.agent.reviewed_research import ReviewedResearchAgent
from app.agent.tools import AgentTools
from app.rag.chunker import Citation, chunk_document

from test_evidence_review import (
    check_payload,
    reviewer_client,
    reply,
)


QUESTION = (
    "A patient currently on warfarin is prescribed sertraline. "
    "What is the risk of bleeding and what monitoring is recommended?"
)


def section(drug, number, title, text):
    sid = f"dailymed-{drug}.md#{number}"

    return SectionEvidence(
        section_id=sid,
        parent_section_id=None,
        drug=drug,
        source="dailymed",
        citation=Citation(
            id=sid,
            file=f"dailymed-{drug}.md",
            source_label="DailyMed FDA label",
            url=None,
            section_number=number,
            section_path=title,
        ),
        spans=(
            SectionSpan(0, len(text), text),
        ),
        excerpts=(),
        total_chars=len(text),
        missing_ranges=(),
        complete=True,
        matched_chunk_ids=(),
        tool_call_ids=(),
        score=1.0,
    )


@pytest.fixture
def evidence():
    return SimpleNamespace(
        sertraline=section(
            "sertraline",
            "5.3",
            "Increased Risk of Bleeding",
            "Concomitant warfarin use may increase bleeding risk. Monitor INR.",
        ),
        generic=section(
            "warfarin",
            "5.1",
            "Hemorrhage",
            "Warfarin can cause major bleeding. Monitor INR when starting drugs.",
        ),
        specific=section(
            "warfarin",
            "7.3",
            "Drugs that Increase Bleeding Risk",
            "Serotonin reuptake inhibitors, including sertraline, "
            "increase bleeding risk with warfarin. Closely monitor patients.",
        ),
    )


def positive_model_review(sections, *, optional=False):
    checks = [
        check_payload(
            sections[0].section_id,
            required=True,
        )
    ]
    plan = ["Named interaction risk and monitoring."]

    if optional:
        plan.append("Optional measured INR effects.")
        checks.append({
            "step_index": 2,
            "required": False,
            "supported": False,
            "requires_complete": False,
            "section_ids": [],
            "explanation": "Measured effects were not requested.",
        })

    return review_evidence(
        QUESTION,
        tuple(plan),
        tuple(sections),
        client=reviewer_client({"checks": checks}),
        model="test",
    )


def test_generic_warfarin_warnings_cannot_pass_positive_model_review(
    evidence,
):
    review = positive_model_review((
        evidence.sertraline,
        evidence.generic,
    ))

    assert review.checks[0].supported
    assert not review.sufficient
    assert len(review.interactions) == 2

    missing = [
        item
        for item in review.interactions
        if not item.supported
    ]

    assert len(missing) == 1
    assert missing[0].source_drug == "warfarin"
    assert missing[0].target_drug == "sertraline"
    assert "Search warfarin drug interactions" in review.gaps[0]


def test_complete_named_evidence_from_both_labels_passes(evidence):
    review = positive_model_review((
        evidence.sertraline,
        evidence.generic,
        evidence.specific,
    ))

    assert review.sufficient
    assert review.gaps == ()
    assert all(
        item.match_kind == "drug"
        for item in review.interactions
    )


def test_missing_optional_lab_details_do_not_block_complete_pair(
    evidence,
):
    review = positive_model_review(
        (
            evidence.sertraline,
            evidence.specific,
        ),
        optional=True,
    )

    assert review.sufficient
    assert not review.checks[1].supported
    assert review.gaps == ()


def test_optional_scope_flag_cannot_disable_two_label_gate(evidence):
    client = reviewer_client({
        "checks": [
            check_payload(
                evidence.sertraline.section_id,
                required=True,
            ),
            {
                "step_index": 2,
                "required": False,
                "supported": False,
                "requires_complete": False,
                "section_ids": [],
                "explanation": (
                    "The model incorrectly calls the other label optional."
                ),
            },
        ]
    })

    review = review_evidence(
        QUESTION,
        ("Risk.", "Other label."),
        (
            evidence.sertraline,
            evidence.generic,
        ),
        client=client,
        model="test",
    )

    assert not review.sufficient
    assert "Required interaction evidence" in review.gaps[0]


def test_unread_or_partial_specific_section_does_not_count(evidence):
    partial = replace(
        evidence.specific,
        complete=False,
    )

    review = positive_model_review((
        evidence.sertraline,
        evidence.generic,
        partial,
    ))

    assert not review.sufficient


def test_explicit_mapped_class_can_supply_counterpart_mention(evidence):
    text = "SSRIs increase bleeding risk."

    class_section = replace(
        evidence.specific,
        spans=(
            SectionSpan(0, len(text), text),
        ),
        total_chars=len(text),
    )

    review = positive_model_review((
        evidence.sertraline,
        class_section,
    ))

    assert review.sufficient
    assert any(
        item.match_kind == "class"
        for item in review.interactions
    )


def test_shared_class_self_description_is_not_counterpart_evidence():
    sections = (
        section(
            "warfarin",
            "5.1",
            "Hemorrhage",
            "Warfarin is an anticoagulant.",
        ),
        section(
            "apixaban",
            "5.2",
            "Bleeding",
            "Apixaban is an anticoagulant.",
        ),
    )

    checks = interaction_evidence_checks(
        "Can warfarin and apixaban be used together?",
        sections,
    )

    assert len(checks) == 2
    assert not any(item.supported for item in checks)


def test_counseling_only_is_not_relevant_interaction_section(evidence):
    counseling = replace(
        evidence.specific,
        citation=replace(
            evidence.specific.citation,
            section_number="17",
            section_path="Patient Counseling",
        ),
    )

    review = positive_model_review((
        evidence.sertraline,
        counseling,
    ))

    assert not review.sufficient


@pytest.mark.parametrize(
    "question",
    [
        "Compare the contraindications of warfarin and apixaban.",
        "What are the main contraindications for warfarin?",
        "What is the metformin renal threshold?",
    ],
)
def test_unrelated_questions_do_not_activate_gate(evidence, question):
    assert interaction_evidence_checks(
        question,
        (
            evidence.sertraline,
            evidence.generic,
        ),
    ) == ()


def test_second_label_cannot_disappear_completely(evidence):
    review = positive_model_review((
        evidence.sertraline,
    ))

    assert not review.sufficient
    assert any(
        check.source_drug == "warfarin"
        and not check.supported
        for check in review.interactions
    )


def test_gap_is_cleared_after_targeted_parent_is_delivered(evidence):
    before = positive_model_review((
        evidence.sertraline,
        evidence.generic,
    ))
    after = positive_model_review((
        evidence.sertraline,
        evidence.generic,
        evidence.specific,
    ))

    assert not before.sufficient
    assert before.gaps
    assert after.sufficient
    assert not after.gaps


def test_real_research_graph_follows_up_on_code_detected_gap():
    _, warfarin_sections, _ = chunk_document(
        "dailymed-warfarin.md",
        "# Warfarin\n"
        "## 5.1 Hemorrhage\n"
        "Warfarin can cause bleeding. Perform more frequent INR monitoring "
"when starting other drugs.\n"
        "## 7.3 Drugs that Increase Bleeding Risk\n"
        "Sertraline increases bleeding risk with warfarin. Monitor closely.",
    )

    _, sertraline_sections, _ = chunk_document(
        "dailymed-sertraline.md",
        "# Sertraline\n"
        "## 5.3 Increased Risk of Bleeding\n"
        "Warfarin may increase bleeding risk. Monitor INR.",
    )

    source_sections = {
        item.id: item
        for item in (
            *warfarin_sections,
            *sertraline_sections,
        )
    }

    generic = next(
        item
        for item in warfarin_sections
        if item.number == "5.1"
    )
    specific = next(
        item
        for item in warfarin_sections
        if item.number == "7.3"
    )
    sertraline = next(
        item
        for item in sertraline_sections
        if item.number == "5.3"
    )

    retriever = Mock()
    retriever.get_section.side_effect = source_sections.get

    def tool_call(call_id, section_id):
        return SimpleNamespace(
            id=call_id,
            type="function",
            function=SimpleNamespace(
                name="get_section",
                arguments=json.dumps({
                    "section_id": section_id,
                    "start_char": 0,
                    "max_chars": 8000,
                }),
            ),
        )

    client = Mock()
    client.chat.completions.create.side_effect = [
        reply(json.dumps({
            "steps": [
                "Named interaction risk and monitoring."
            ]
        })),
        reply(calls=[
            tool_call("one", sertraline.id)
        ]),
        reply(calls=[
            tool_call("two", generic.id)
        ]),
        reply("Evidence collected."),
        reply(calls=[
            tool_call("three", specific.id)
        ]),
        reply("Missing interaction section collected."),
    ]

    def reviewer(question, plan, sections, **kwargs):
        return review_evidence(
            question,
            plan,
            sections,
            client=reviewer_client({
                "checks": [
                    check_payload(
                        sertraline.id,
                        required=True,
                    )
                ]
            }),
            model="test",
        )

    agent = ReviewedResearchAgent(
        AgentTools(retriever),
        AgentConfig(
            model="test",
            max_turns=8,
        ),
        client=client,
        reviewer=reviewer,
    )

    result = agent.collect(QUESTION)

    assert result.status == "evidence_reviewed"
    assert len(result.reviews) == 2
    assert not result.reviews[0].sufficient
    assert result.reviews[1].sufficient
    assert len(result.trace) == 3

    assert any(
        item["role"] == "user"
        and "Required interaction evidence" in item["content"]
        for item in (
            client.chat.completions.create
            .call_args.kwargs["messages"]
        )
    )