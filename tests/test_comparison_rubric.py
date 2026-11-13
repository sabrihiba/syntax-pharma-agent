"""Offline regressions for valid bleeding-comparison wording."""

from dataclasses import replace

import pytest

from app.agent.answer import AnswerDraft, AnswerStatement
from test_agent_quality import (
    APIXABAN_SECTION,
    CASES,
    WARFARIN_SECTION,
    citation_for,
    corpus,
    matching_sources,
    missing_answer,
)


@pytest.fixture
def comparison_context(corpus):
    warfarin = citation_for(
        matching_sources(corpus, WARFARIN_SECTION)[0]
    )
    apixaban = citation_for(
        matching_sources(corpus, APIXABAN_SECTION)[0]
    )

    rule = next(
        rule
        for rule in CASES[1].answer
        if rule.label
        == "bleeding wording compared without assuming equivalence"
    )

    case = replace(CASES[1], answer=(rule,))
    return case, warfarin, apixaban


def make_draft(
    context,
    text,
    *,
    include_warfarin=True,
    include_apixaban=True,
):
    case, warfarin, apixaban = context

    ids = (
        *((warfarin.id,) if include_warfarin else ()),
        *((apixaban.id,) if include_apixaban else ()),
    )

    return AnswerDraft(
        question=case.question,
        statements=(
            AnswerStatement(text, ids),
        ),
        limitations=(),
        citations=(warfarin, apixaban),
    )


@pytest.mark.parametrize(
    "warfarin_wording",
    [
        "hemorrhagic tendencies",
        "blood dyscrasias",
        "bleeding tendencies associated with active ulceration or overt bleeding",
    ],
)
def test_valid_label_wording_is_accepted(
    comparison_context,
    warfarin_wording,
):
    case, _, _ = comparison_context

    draft = make_draft(
        comparison_context,
        "Apixaban lists active pathological bleeding; warfarin lists "
        f"{warfarin_wording}. The wording differs.",
    )

    assert missing_answer(case, draft) == []


def test_exact_reported_comparison_bullet_is_accepted(
    comparison_context,
):
    case, _, _ = comparison_context

    draft = make_draft(
        comparison_context,
        "Apixaban lists active pathological bleeding as a contraindication; "
        "warfarin lists bleeding tendencies associated with active ulceration "
        "or overt bleeding of the gastrointestinal, genitourinary, or respiratory "
        "tract, among other specified conditions. "
        "The bleeding wording overlaps but is not identical.",
    )

    assert missing_answer(case, draft) == []


@pytest.mark.parametrize(
    "text",
    [
        "Apixaban lists bleeding; warfarin lists bleeding tendencies. "
        "The wording differs.",
        "Apixaban lists active pathological bleeding; warfarin lists monitoring. "
        "The wording differs.",
        "Apixaban lists active pathological bleeding; "
        "warfarin lists bleeding tendencies.",
    ],
)
def test_required_comparison_components_still_matter(
    comparison_context,
    text,
):
    case, _, _ = comparison_context
    draft = make_draft(comparison_context, text)

    assert missing_answer(case, draft) == [
        case.answer[0].label
    ]


@pytest.mark.parametrize(
    "missing_source",
    ["warfarin", "apixaban"],
)
def test_both_label_citations_are_still_required(
    comparison_context,
    missing_source,
):
    case, _, _ = comparison_context

    draft = make_draft(
        comparison_context,
        "Apixaban lists active pathological bleeding; warfarin lists "
        "bleeding tendencies. The wording differs.",
        include_warfarin=missing_source != "warfarin",
        include_apixaban=missing_source != "apixaban",
    )

    assert missing_answer(case, draft) == [
        case.answer[0].label
    ]