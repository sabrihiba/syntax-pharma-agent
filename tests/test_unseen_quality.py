"""Practice unseen-question tests, not Syntax's actual hidden questions.

Offline checks run by default.
Set RUN_UNSEEN_QUALITY=1 for live API calls.

Requires the existing test_agent_quality.py evaluation helpers.
Production code is unchanged.
"""

from dataclasses import replace
import os
from pathlib import Path
import time

import pytest

from app.agent.answer import DISCLAIMER
from app.rag.chunker import load_corpus
from app.service import build_service

from test_agent_quality import (
    AnswerRule,
    EvidenceRule,
    PublicCase,
    citation_errors,
    matching_sources,
    missing_answer,
    missing_evidence,
    synthetic_example,
)


ROOT = Path(__file__).resolve().parents[1]
LIVE = os.getenv("RUN_UNSEEN_QUALITY") == "1"

AMOX = "dailymed-amoxicillin.md"
IBU = "dailymed-ibuprofen.md"
ATOR = "dailymed-atorvastatin.md"
ACET = "dailymed-acetaminophen.md"
APIX = "dailymed-apixaban.md"


AMOX_ALLERGY = EvidenceRule(
    "amoxicillin serious hypersensitivity",
    AMOX,
    ("4",),
    (
        r"serious hypersensitivity",
        r"penicillins",
        r"cephalosporins",
    ),
)

IBU_CONTRA = EvidenceRule(
    "ibuprofen contraindications",
    IBU,
    (None,),
    (
        r"contraindicated",
        r"coronary artery bypass",
        r"hypersensitivity",
        r"aspirin",
    ),
)


CASES = (
    PublicCase(
        "brand-eliquis",
        "According to the corpus, what are the contraindications "
        "for Eliquis?",
        (
            EvidenceRule(
                "apixaban contraindications",
                APIX,
                ("4",),
                (
                    r"active pathological bleeding",
                    r"severe hypersensitivity",
                ),
            ),
        ),
        (
            AnswerRule(
                "active pathological bleeding",
                (r"active pathological bleeding",),
                (APIX,),
            ),
            AnswerRule(
                "severe hypersensitivity",
                (r"severe hypersensitivity|anaphyla",),
                (APIX,),
            ),
        ),
    ),
    PublicCase(
        "amoxicillin-allergy",
        "Does the amoxicillin label contraindicate use after a serious "
        "reaction to a cephalosporin? Explain the relevant allergy category.",
        (AMOX_ALLERGY,),
        (
            AnswerRule(
                "serious allergy includes cephalosporins",
                (
                    r"cephalosporin",
                    r"hypersensitiv|allerg|anaphyla",
                    r"contraindicat|must not|should not|do not",
                ),
                (AMOX,),
            ),
        ),
    ),
    PublicCase(
        "ibuprofen-cabg",
        "What does the ibuprofen label say about its use in the setting "
        "of coronary artery bypass graft surgery?",
        (IBU_CONTRA,),
        (
            AnswerRule(
                "CABG contraindication",
                (
                    r"cabg|coronary artery bypass",
                    r"contraindicat|must not|should not|do not",
                ),
                (IBU,),
            ),
        ),
    ),
    PublicCase(
        "atorvastatin-interaction",
        "According to the atorvastatin label, what risk and dose restriction "
        "apply when clarithromycin is taken with atorvastatin?",
        (
            EvidenceRule(
                "atorvastatin interaction table",
                ATOR,
                ("7.1", "7"),
                (
                    r"clarithromycin",
                    r"20 mg",
                    r"myopathy",
                    r"rhabdomyolysis",
                ),
            ),
        ),
        (
            AnswerRule(
                "muscle toxicity risk",
                (r"myopath|rhabdomyolysis",),
                (ATOR,),
            ),
            AnswerRule(
                "20 mg ceiling with clarithromycin",
                (
                    r"clarithromycin",
                    r"20\s*mg",
                    r"not exceed|no more than|maximum|at most|limit|up to",
                ),
                (ATOR,),
            ),
        ),
    ),
    PublicCase(
        "acetaminophen-duplicate-products",
        "Can this acetaminophen product be used with another medicine "
        "containing acetaminophen? What harm does its label warn about?",
        (
            EvidenceRule(
                "acetaminophen duplicate-product and liver warnings",
                ACET,
                (None,),
                (
                    r"severe liver damage",
                    r"other drug containing acetaminophen",
                ),
            ),
        ),
        (
            AnswerRule(
                "avoid another acetaminophen-containing medicine",
                (
                    r"acetaminophen",
                    r"other|another|multiple",
                    r"do not|should not|must not|avoid|not use|not take|not be",
                ),
                (ACET,),
            ),
            AnswerRule(
                "liver damage",
                (
                    r"liver",
                    r"damage|injury|toxic|failure",
                ),
                (ACET,),
            ),
        ),
    ),
    PublicCase(
        "two-drug-new-comparison",
        "Compare amoxicillin and ibuprofen: what allergy-related "
        "contraindications are listed for each?",
        (
            AMOX_ALLERGY,
            IBU_CONTRA,
        ),
        (
            AnswerRule(
                "amoxicillin beta-lactam allergy",
                (
                    r"penicillin|cephalosporin|lactam",
                    r"hypersensitiv|allerg|anaphyla",
                ),
                (AMOX,),
            ),
            AnswerRule(
                "ibuprofen hypersensitivity",
                (
                    r"ibuprofen",
                    r"hypersensitiv|allerg",
                ),
                (IBU,),
            ),
            AnswerRule(
                "aspirin or NSAID-associated reactions",
                (
                    r"aspirin|nsaid",
                    r"asthma|urticaria|allergic",
                ),
                (IBU,),
            ),
        ),
    ),
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(ROOT / "corpus")


@pytest.fixture(scope="module")
def service():
    if not LIVE:
        pytest.skip("Set RUN_UNSEEN_QUALITY=1 for live calls")

    instance = build_service()

    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_gold_evidence_exists(case, corpus):
    for rule in case.evidence:
        assert matching_sources(corpus, rule), rule.label


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_rubric_accepts_cited_source_text(case, corpus):
    draft, sections = synthetic_example(case, corpus)

    assert not missing_evidence(case, sections, corpus)
    assert not missing_answer(case, draft)
    assert not citation_errors(draft, sections, corpus)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_rubric_rejects_uncited_source_text(case, corpus):
    draft, _ = synthetic_example(case, corpus)

    uncited = replace(
        draft,
        statements=tuple(
            replace(statement, section_ids=())
            for statement in draft.statements
        ),
    )

    assert len(missing_answer(case, uncited)) == len(case.answer)


def test_unavailable_drug_has_no_monograph(corpus):
    assert "naproxen" not in {
        doc.drug
        for doc in corpus.documents
    }


def check_execution_bounds(result, service):
    workflow = service.workflow

    assert len(result.research.trace) <= (
        workflow.researcher.config.max_tool_calls
    )

    assert 0 <= result.revisions <= workflow.max_revisions

    assert all(
        entry.name in {"search_corpus", "get_section"}
        for entry in result.research.trace
    )

    delivered = {
        entry.call_id
        for entry in result.research.trace
        if entry.ok and entry.delivered
    }

    assert all(
        set(section.tool_call_ids) <= delivered
        for section in result.research.sections
    )


@pytest.mark.skipif(
    not LIVE,
    reason="Set RUN_UNSEEN_QUALITY=1",
)
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_live_unseen_question(case, corpus, service):
    started = time.perf_counter()

    result = service.ask(case.question)

    print(f"\nCASE: {case.name}")
    print("Question:", case.question)
    print("Status:", result.status)
    print("Research:", result.research.status)
    print("Tool calls:", len(result.research.trace))
    print(
        "Elapsed:",
        round(time.perf_counter() - started, 1),
        "seconds",
    )
    print("Notes:", result.notes)

    check_execution_bounds(result, service)

    assert result.status == "answer_reviewed", result.notes
    assert result.research.status == "evidence_reviewed"
    assert result.review is not None and result.review.sufficient
    assert result.draft is not None

    missing_sources = missing_evidence(
        case,
        result.research.sections,
        corpus,
    )

    missing_facts = missing_answer(
        case,
        result.draft,
    )

    bad_citations = citation_errors(
        result.draft,
        result.research.sections,
        corpus,
    )

    print("Missing evidence:", missing_sources)
    print("Missing answer requirements:", missing_facts)
    print("Citation errors:", bad_citations)
    print("\nANSWER:\n", result.to_markdown())

    assert DISCLAIMER in result.to_markdown()
    assert not missing_sources, missing_sources
    assert not missing_facts, missing_facts
    assert not bad_citations, bad_citations


@pytest.mark.skipif(
    not LIVE,
    reason="Set RUN_UNSEEN_QUALITY=1",
)
def test_live_unavailable_drug_is_not_released(service):
    """Safety fallback check, not user-friendly abstention wording."""
    result = service.ask(
        "Using only the supplied corpus, list naproxen's contraindications. "
        "Do not substitute another NSAID or use outside medical knowledge."
    )

    print("\nCASE: unavailable-naproxen")
    print("Status:", result.status)
    print("Research:", result.research.status)
    print("Notes:", result.notes)

    check_execution_bounds(result, service)

    # No naproxen monograph exists in this corpus.
    # A fabricated approved answer must not be released.
    assert result.status == "research_incomplete", result.notes

    # Protocol/context errors are not successful safety fallbacks.
    assert result.research.status in {
        "no_evidence",
        "model_finished",
        "review_limit",
        "research_limit",
        "tool_limit",
        "turn_limit",
    }, result.research.status

    assert any(
        entry.name == "search_corpus"
        and entry.ok
        and entry.delivered
        for entry in result.research.trace
    ), "The agent must actually check the corpus"

    with pytest.raises(
        ValueError,
        match="No reviewed answer",
    ):
        result.to_markdown()