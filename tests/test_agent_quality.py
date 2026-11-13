"""Four public-question evaluations. Live agent runs are opt-in.

Default: local corpus/rubric checks only.
Live: RUN_AGENT_QUALITY=1.
Regex checks are regression checks, not a clinical correctness oracle.
No reports are written and failed live cases are not automatically retried.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from app import config as settings
from app.agent.answer import DISCLAIMER, AnswerDraft, AnswerStatement
from app.agent.consolidation import SectionEvidence, SectionSpan
from app.agent.research import AgentConfig
from app.agent.reviewed_research import ReviewedResearchAgent, ReviewConfig
from app.agent.tools import AgentTools
from app.agent.workflow import AnswerWorkflow
from app.rag.cache import CachedEmbedder
from app.rag.chunker import Citation, load_corpus
from app.rag.hybrid import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]
LIVE = os.getenv("RUN_AGENT_QUALITY") == "1"


@dataclass(frozen=True)
class EvidenceRule:
    label: str
    file: str
    numbers: tuple[str, ...]
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class AnswerRule:
    label: str
    patterns: tuple[str, ...]
    files: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicCase:
    name: str
    question: str
    evidence: tuple[EvidenceRule, ...]
    answer: tuple[AnswerRule, ...]


W = "dailymed-warfarin.md"
A = "dailymed-apixaban.md"
S = "dailymed-sertraline.md"
M = "dailymed-metformin.md"

WARFARIN_SECTION = EvidenceRule(
    "complete warfarin contraindications",
    W,
    ("4",),
    (
        r"pregnan",
        r"mechanical heart valves",
        r"malignant hypertension",
    ),
)

APIXABAN_SECTION = EvidenceRule(
    "complete apixaban contraindications",
    A,
    ("4",),
    (
        r"active pathological bleeding",
        r"severe hypersensitivity",
    ),
)

WARFARIN_INTERACTION = EvidenceRule(
    "warfarin interaction naming sertraline",
    W,
    ("7", "7.3"),
    (r"sertraline", r"bleeding", r"monitor"),
)

SERTRALINE_INTERACTION = EvidenceRule(
    "sertraline interaction and INR monitoring",
    S,
    ("5.3", "7.1", "7"),
    (
        r"warfarin",
        r"bleeding",
        r"monitor",
        r"international normalized ratio",
    ),
)

METFORMIN_RENAL = EvidenceRule(
    "metformin renal cutoff",
    M,
    ("2.3", "4", "5.1", "8.6", "5", "8"),
    (
        r"egfr",
        r"contraindicat",
        r"(?:below|less than)\s+30\b",
    ),
)

# Independent conditions may appear in separate cited statements.
VASCULAR_ANSWERS = (
    AnswerRule(
        "central nervous system hemorrhage",
        (r"central nervous|\bcns\b", r"hemorrhag"),
        (W,),
    ),
    AnswerRule(
        "cerebral aneurysms",
        (r"cerebral.{0,20}aneurysm|brain.{0,20}aneurysm",),
        (W,),
    ),
    AnswerRule(
        "dissecting aorta",
        (r"dissect", r"aort"),
        (W,),
    ),
    AnswerRule(
        "pericarditis",
        (r"pericarditis",),
        (W,),
    ),
    AnswerRule(
        "pericardial effusions",
        (r"pericardial effusion",),
        (W,),
    ),
    AnswerRule(
        "bacterial endocarditis",
        (r"bacterial endocarditis|infective endocarditis",),
        (W,),
    ),
)

WARFARIN_ANSWERS = (
    AnswerRule(
        "pregnancy exception and high-risk mechanical valves",
        (
            r"pregnan",
            r"except|unless|exception",
            r"mechanical.{0,20}(?:heart\s+)?valve",
            r"high.{0,35}risk|risk.{0,20}high",
            r"thromboembol",
        ),
        (W,),
    ),
    AnswerRule(
        "hemorrhagic tendencies or blood dyscrasias",
        (r"hemorrhagic tendenc|blood dyscrasia",),
        (W,),
    ),
    AnswerRule(
        "central nervous system surgery",
        (r"central nervous|\bcns\b", r"surg"),
        (W,),
    ),
    AnswerRule(
        "eye surgery",
        (r"\beye\b", r"surg"),
        (W,),
    ),
    AnswerRule(
        "traumatic surgery with large open surfaces",
        (r"trauma", r"surg", r"large open surface"),
        (W,),
    ),
    AnswerRule(
        "gastrointestinal bleeding or ulceration",
        (r"gastrointestinal", r"bleed|ulcer"),
        (W,),
    ),
    AnswerRule(
        "genitourinary bleeding",
        (r"genitourinary", r"bleed"),
        (W,),
    ),
    AnswerRule(
        "respiratory tract bleeding",
        (r"respiratory", r"bleed"),
        (W,),
    ),
    *VASCULAR_ANSWERS,
    AnswerRule(
        "threatened abortion",
        (r"threatened abortion",),
        (W,),
    ),
    AnswerRule(
        "eclampsia",
        (r"\beclampsia\b",),
        (W,),
    ),
    AnswerRule(
        "preeclampsia",
        (r"\bpre[- ]?eclampsia\b",),
        (W,),
    ),
    AnswerRule(
        "unsupervised patients and noncompliance",
        (r"unsupervised", r"non.?compliance|non.?adheren"),
        (W,),
    ),
    AnswerRule(
        "spinal puncture",
        (r"spinal puncture",),
        (W,),
    ),
    AnswerRule(
        "procedures with potential uncontrolled bleeding",
        (r"procedur", r"uncontroll|uncontrol", r"bleed"),
        (W,),
    ),
    AnswerRule(
        "hypersensitivity",
        (r"hypersensitiv",),
        (W,),
    ),
    AnswerRule(
        "regional or lumbar anesthesia",
        (r"regional|lumbar", r"an(?:ae|e)sthesia"),
        (W,),
    ),
    AnswerRule(
        "malignant hypertension",
        (r"malignant hypertension",),
        (W,),
    ),
)

COMPARISON_ANSWERS = WARFARIN_ANSWERS + (
    AnswerRule(
        "shared hypersensitivity with citations to both sections",
        (r"hypersensitiv", r"both|shared|each"),
        (W, A),
    ),
    AnswerRule(
        "bleeding wording compared without assuming equivalence",
        (
            r"active pathological bleeding",
            r"hemorrhagic tendenc|blood dyscrasia|bleeding tendenc",
            r"differ|not identical|not equivalent|not.*same|distinct",
        ),
        (W, A),
    ),
    AnswerRule(
        "a label-specific distinction supported by both sections",
        (r"not.{0,35}list|list.{0,25}only|but not|unique",),
        (W, A),
    ),
    AnswerRule(
        "apixaban severe hypersensitivity",
        (r"severe hypersensitiv",),
        (A,),
    ),
)

INTERACTION_ANSWERS = (
    AnswerRule(
        "increased bleeding risk",
        (
            r"bleed",
            r"increas|potentiat|additive|add to|heighten",
        ),
        (S,),
    ),
    AnswerRule(
        "INR monitoring",
        (
            r"\binr\b|international normalized ratio",
            r"monitor|test",
        ),
        (S,),
    ),
    AnswerRule(
        "warfarin monitoring when medication changes",
        (
            r"\binr\b|international normalized ratio",
            r"monitor|test",
            r"start|initiat|stop|discontinu|dose.?chang|dosage",
            r"frequen|additional|extra|clos",
        ),
        (W,),
    ),
)

UNIT = (
    r"\bml\s*/\s*min(?:ute)?s?\s*/\s*"
    r"1\.73\s*m\s*\^?\s*2\b"
)

STRICT_BELOW_30 = (
    r"(?:below|less than|under)\s+30\b|<\s*30\b"
)

BAD_CUTOFF = (
    r"(?:below|less than|under|<)\s*25\b",
    r"(?:at or below|less than or equal to|<=)\s*30\b",
)

METFORMIN_ANSWERS = (
    AnswerRule(
        "contraindication at the patient value of 25",
        (r"\b25\b", r"contraindicat"),
        (M,),
        forbidden=(
            r"not\s+(?:\w+\s+){0,2}contraindicat",
            *BAD_CUTOFF,
        ),
    ),
    AnswerRule(
        "strict below-30 eGFR cutoff with normalized units",
        (
            r"egfr",
            r"contraindicat",
            STRICT_BELOW_30,
            UNIT,
        ),
        (M,),
        forbidden=BAD_CUTOFF,
    ),
)

CASES = (
    PublicCase(
        "warfarin",
        "What are the main contraindications for warfarin?",
        (WARFARIN_SECTION,),
        WARFARIN_ANSWERS,
    ),
    PublicCase(
        "comparison",
        "Compare the contraindications of warfarin and apixaban. "
        "What conditions are shared and which are unique to each drug?",
        (WARFARIN_SECTION, APIXABAN_SECTION),
        COMPARISON_ANSWERS,
    ),
    PublicCase(
        "interaction",
        "A patient currently on warfarin is prescribed sertraline. "
        "What is the risk of bleeding and what monitoring is recommended?",
        (WARFARIN_INTERACTION, SERTRALINE_INTERACTION),
        INTERACTION_ANSWERS,
    ),
    PublicCase(
        "metformin",
        "A patient with an eGFR of 25 mL/min/1.73 m² is being considered "
        "for metformin. Is metformin contraindicated at this level of "
        "renal function, and what is the threshold?",
        (METFORMIN_RENAL,),
        METFORMIN_ANSWERS,
    ),
)


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("≤", "<=").replace("≥", ">=")
    text = text.replace("−", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text)


def contains(text, patterns):
    text = normalized(text)
    return all(
        re.search(pattern, text)
        for pattern in patterns
    )


def matching_sources(corpus, rule):
    return [
        section
        for section in corpus.sections
        if section.doc.file == rule.file
        and section.number in rule.numbers
        and contains(section.text, rule.patterns)
    ]


def missing_evidence(case, sections, corpus):
    missing = []

    for rule in case.evidence:
        candidates = {
            section.id: section
            for section in matching_sources(corpus, rule)
        }

        covered = any(
            section.section_id in candidates
            and section.complete
            and section.full_text
            == candidates[section.section_id].text
            for section in sections
        )

        if not covered:
            missing.append(rule.label)

    return missing


def missing_answer(case, draft):
    references = {
        citation.id: citation
        for citation in draft.citations
    }
    missing = []

    for rule in case.answer:
        covered = False

        for statement in draft.statements:
            files = {
                references[section_id].file
                for section_id in statement.section_ids
                if section_id in references
            }

            if (
                set(rule.files) <= files
                and contains(statement.text, rule.patterns)
                and not any(
                    re.search(
                        pattern,
                        normalized(statement.text),
                    )
                    for pattern in rule.forbidden
                )
            ):
                covered = True
                break

        if not covered:
            missing.append(rule.label)

    return missing


def citation_for(section):
    return Citation(
        id=section.id,
        file=section.doc.file,
        source_label=section.doc.source_label,
        url=section.doc.url,
        section_number=section.number,
        section_path=section.path_str,
    )


def citation_errors(draft, sections, corpus):
    errors = []
    delivered = {
        section.section_id: section
        for section in sections
    }
    references = {
        citation.id: citation
        for citation in draft.citations
    }

    if len(references) != len(draft.citations):
        errors.append("Duplicate citation IDs")

    used = set()

    for index, statement in enumerate(
        draft.statements,
        start=1,
    ):
        if not statement.section_ids:
            errors.append(
                f"Statement {index} has no citation"
            )

        for section_id in statement.section_ids:
            used.add(section_id)
            source = corpus.section(section_id)

            if (
                section_id not in references
                or section_id not in delivered
                or source is None
            ):
                errors.append(
                    f"Statement {index} cites unknown evidence: "
                    f"{section_id}"
                )
            elif (
                references[section_id]
                != delivered[section_id].citation
                or references[section_id]
                != citation_for(source)
            ):
                errors.append(
                    f"Statement {index} has incorrect citation metadata"
                )

    if set(references) != used:
        errors.append(
            "Reference list does not match statement citations"
        )

    return errors


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(ROOT / "corpus")


@pytest.fixture(scope="module")
def workflow(corpus):
    if not LIVE:
        pytest.skip(
            "Set RUN_AGENT_QUALITY=1 for live agent evaluation"
        )

    if (
        not settings.API_KEY
        or not settings.MODEL
        or not settings.EMBEDDING_MODEL
    ):
        pytest.fail(
            "API key, chat model, and embedding model "
            "must be configured"
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL or None,
        timeout=45.0,
        max_retries=1,
    )

    retriever = HybridRetriever.build(
        corpus,
        CachedEmbedder(),
    )

    researcher = ReviewedResearchAgent(
        AgentTools(retriever),
        AgentConfig(max_turns=12),
        review_config=ReviewConfig(max_reviews=3),
        client=client,
    )

    return AnswerWorkflow(
        researcher,
        client=client,
        max_revisions=2,
    )


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_public_gold_evidence_exists(case, corpus):
    for rule in case.evidence:
        assert matching_sources(corpus, rule), rule.label


def synthetic_example(case, corpus):
    """Exercise the rubric using source text, not a generated answer."""
    sections = []
    statements = []
    citations = []

    for rule in case.evidence:
        section = matching_sources(corpus, rule)[0]

        sections.append(
            SectionEvidence(
                section_id=section.id,
                parent_section_id=section.parent_id,
                drug=section.doc.drug,
                source=section.doc.source,
                citation=citation_for(section),
                spans=(
                    SectionSpan(
                        0,
                        len(section.text),
                        section.text,
                    ),
                ),
                excerpts=(),
                total_chars=len(section.text),
                missing_ranges=(),
                complete=True,
                matched_chunk_ids=(),
                tool_call_ids=(),
                score=1.0,
            )
        )

        citations.append(citation_for(section))
        statements.append(
            AnswerStatement(
                section.text,
                (section.id,),
            )
        )

    ids = tuple(citation.id for citation in citations)

    if case.name == "comparison":
        statements.append(
            AnswerStatement(
                "Both list hypersensitivity. Bleeding wording differs: "
                "active pathological bleeding versus hemorrhagic tendencies "
                "or blood dyscrasias. Pregnancy is not listed in one "
                "supplied section.",
                ids,
            )
        )

    if case.name == "metformin":
        statements.append(
            AnswerStatement(
                "An eGFR of 25 mL/min/1.73 m² is below "
                "30 mL/min/1.73 m²; metformin is contraindicated "
                "at this patient value.",
                ids,
            )
        )

    return (
        AnswerDraft(
            case.question,
            tuple(statements),
            (),
            tuple(citations),
        ),
        tuple(sections),
    )


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_rubric_accepts_source_examples(case, corpus):
    draft, sections = synthetic_example(case, corpus)

    assert not missing_evidence(case, sections, corpus)
    assert not missing_answer(case, draft)
    assert not citation_errors(draft, sections, corpus)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_rubric_rejects_uncited_content_even_if_words_match(
    case,
    corpus,
):
    draft, _ = synthetic_example(case, corpus)

    uncited = replace(
        draft,
        statements=tuple(
            replace(statement, section_ids=())
            for statement in draft.statements
        ),
    )

    assert len(
        missing_answer(case, uncited)
    ) == len(case.answer)


def test_rubric_requires_pregnancy_exception(corpus):
    case = CASES[0]
    draft, _ = synthetic_example(case, corpus)

    incomplete = replace(
        draft,
        statements=(
            AnswerStatement(
                "Warfarin is contraindicated in pregnancy.",
                draft.statements[0].section_ids,
            ),
        ),
    )

    assert case.answer[0].label in missing_answer(
        case,
        incomplete,
    )


@pytest.mark.parametrize(
    "wording",
    [
        "eGFR 25 mL/min/1.73 m² is not contraindicated "
        "below 30 mL/min/1.73 m².",
        "eGFR 25 mL/min/1.73 m² is contraindicated "
        "at or below 30 mL/min/1.73 m².",
        "eGFR 25 is contraindicated below 30.",
    ],
)
def test_rubric_rejects_wrong_or_incomplete_numeric_claim(
    corpus,
    wording,
):
    case = CASES[3]
    draft, _ = synthetic_example(case, corpus)

    wrong = replace(
        draft,
        statements=(
            AnswerStatement(
                wording,
                draft.statements[0].section_ids,
            ),
        ),
    )

    assert missing_answer(case, wrong)


@pytest.fixture
def split_condition_draft(corpus):
    source = matching_sources(
        corpus,
        WARFARIN_SECTION,
    )[0]
    citation = citation_for(source)

    return AnswerDraft(
        question=CASES[1].question,
        statements=(
            AnswerStatement(
                "Both labels contain bleeding-related contraindications, "
                "with different wording. Warfarin lists bleeding tendencies "
                "associated with central nervous system hemorrhage.",
                (citation.id,),
            ),
            AnswerStatement(
                "Warfarin lists bleeding tendencies associated with cerebral "
                "aneurysms, dissecting aorta, pericarditis and pericardial "
                "effusions, or bacterial endocarditis.",
                (citation.id,),
            ),
        ),
        limitations=(),
        citations=(citation,),
    )


def test_rubric_accepts_conditions_split_across_cited_statements(
    split_condition_draft,
):
    case = replace(
        CASES[1],
        answer=VASCULAR_ANSWERS,
    )

    assert not missing_answer(
        case,
        split_condition_draft,
    )


@pytest.mark.parametrize(
    "omitted",
    range(len(VASCULAR_ANSWERS)),
    ids=[
        rule.label
        for rule in VASCULAR_ANSWERS
    ],
)
def test_each_condition_is_still_required(corpus, omitted):
    citation = citation_for(
        matching_sources(
            corpus,
            WARFARIN_SECTION,
        )[0]
    )

    facts = (
        "Central nervous system hemorrhage.",
        "Cerebral aneurysms.",
        "Dissecting aorta.",
        "Pericarditis.",
        "Pericardial effusions.",
        "Bacterial endocarditis.",
    )

    draft = AnswerDraft(
        question=CASES[1].question,
        statements=tuple(
            AnswerStatement(
                text,
                (citation.id,),
            )
            for index, text in enumerate(facts)
            if index != omitted
        ),
        limitations=(),
        citations=(citation,),
    )

    case = replace(
        CASES[1],
        answer=VASCULAR_ANSWERS,
    )

    assert missing_answer(case, draft) == [
        VASCULAR_ANSWERS[omitted].label
    ]


def test_split_facts_still_need_the_correct_source(
    corpus,
    split_condition_draft,
):
    apixaban = citation_for(
        matching_sources(
            corpus,
            APIXABAN_SECTION,
        )[0]
    )

    incorrect = replace(
        split_condition_draft,
        citations=(
            *split_condition_draft.citations,
            apixaban,
        ),
        statements=(
            split_condition_draft.statements[0],
            replace(
                split_condition_draft.statements[1],
                section_ids=(apixaban.id,),
            ),
        ),
    )

    case = replace(
        CASES[1],
        answer=VASCULAR_ANSWERS,
    )

    assert missing_answer(case, incorrect) == [
        rule.label
        for rule in VASCULAR_ANSWERS[1:]
    ]


@pytest.mark.skipif(
    not LIVE,
    reason="Set RUN_AGENT_QUALITY=1",
)
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_live_public_agent_quality(
    case,
    corpus,
    workflow,
):
    started = time.perf_counter()
    result = workflow.run(case.question)
    elapsed = time.perf_counter() - started

    print(f"\nCASE: {case.name}")
    print(
        f"Status: {result.status}; "
        f"research: {result.research.status}"
    )
    print(
        f"Tool calls: {len(result.research.trace)}; "
        f"revisions: {result.revisions}; "
        f"elapsed: {elapsed:.1f}s"
    )
    print("Notes:", result.notes)

    assert result.status == "answer_reviewed", result.notes
    assert result.draft is not None
    assert (
        result.review is not None
        and result.review.sufficient
    )
    assert result.research.status == "evidence_reviewed"
    assert 0 <= result.revisions <= workflow.max_revisions

    assert (
        len(result.research.trace)
        <= workflow.researcher.config.max_tool_calls
    )
    assert (
        len(result.attempts)
        == result.revisions + 1
    )
    assert result.attempts[-1].draft == result.draft
    assert result.attempts[-1].review == result.review

    delivered_ids = {
        entry.call_id
        for entry in result.research.trace
        if entry.ok and entry.delivered
    }

    assert all(
        set(section.tool_call_ids) <= delivered_ids
        for section in result.research.sections
    )

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

    errors = (
        [
            f"Evidence: {item}"
            for item in missing_sources
        ]
        + [
            f"Answer: {item}"
            for item in missing_facts
        ]
        + bad_citations
    )

    assert DISCLAIMER in result.to_markdown()
    assert not errors, "\n".join(errors)