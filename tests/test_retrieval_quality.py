"""End-to-end retrieval quality checks.

Default run: validate corpus evidence; live tests are skipped.
Set RUN_RETRIEVAL_QUALITY=1 to use real, cached embeddings.

These are retrieval checks, not clinical-answer or agent evaluations.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pytest

from app import config as settings
from app.rag.cache import CachedEmbedder
from app.rag.chunker import Section, load_corpus
from app.rag.evidence import expand_evidence
from app.rag.hybrid import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]

# Retrieve enough chunks to allow deduplication during expansion.
RETRIEVAL_K = 20
EVIDENCE_K = 5


@dataclass(frozen=True)
class Requirement:
    label: str
    file: str
    section_numbers: tuple[str, ...]
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class QualityCase:
    name: str
    question: str
    requirements: tuple[Requirement, ...]
    drugs: tuple[str, ...] | None = None
    sources: tuple[str, ...] | None = None


WARFARIN_CONTRA = Requirement(
    label="complete warfarin contraindications",
    file="dailymed-warfarin.md",
    section_numbers=("4",),
    patterns=(
        r"pregnan",
        r"mechanical heart valves",
        r"hypersensitivity",
        r"malignant hypertension",
    ),
)

APIXABAN_CONTRA = Requirement(
    label="complete apixaban contraindications",
    file="dailymed-apixaban.md",
    section_numbers=("4",),
    patterns=(
        r"active pathological bleeding",
        r"hypersensitivity",
    ),
)

WARFARIN_INTERACTION = Requirement(
    label="warfarin interaction evidence naming sertraline",
    file="dailymed-warfarin.md",
    section_numbers=("7", "7.3"),
    patterns=(
        r"sertraline",
        r"bleeding",
        r"monitor",
    ),
)

SERTRALINE_INTERACTION = Requirement(
    label="sertraline bleeding risk and INR monitoring",
    file="dailymed-sertraline.md",
    section_numbers=("5.3", "7.1", "7"),
    patterns=(
        r"warfarin",
        r"bleeding",
        r"monitor",
        r"international normalized ratio",
    ),
)

METFORMIN_RENAL = Requirement(
    label="metformin renal contraindication threshold",
    file="dailymed-metformin.md",
    section_numbers=(
        "2.3",
        "4",
        "5.1",
        "5",
        "8.6",
        "8",
    ),
    patterns=(
        r"egfr",
        r"contraindicat",
        r"(?:below|less than)\s+30\b",
    ),
)


CASES = (
    QualityCase(
        name="original-warfarin",
        question=(
            "What are the main contraindications for warfarin?"
        ),
        requirements=(WARFARIN_CONTRA,),
    ),
    QualityCase(
        name="original-comparison",
        question=(
            "Compare the contraindications of warfarin and apixaban. "
            "What conditions are shared and which are unique to each drug?"
        ),
        requirements=(
            WARFARIN_CONTRA,
            APIXABAN_CONTRA,
        ),
    ),
    QualityCase(
        name="original-interaction",
        question=(
            "A patient currently on warfarin is prescribed sertraline. "
            "What is the risk of bleeding and what monitoring is recommended?"
        ),
        requirements=(
            WARFARIN_INTERACTION,
            SERTRALINE_INTERACTION,
        ),
    ),
    QualityCase(
        name="original-metformin",
        question=(
            "A patient with an eGFR of 25 mL/min is being considered for "
            "metformin. Is metformin contraindicated at this level of "
            "renal function, and what is the threshold?"
        ),
        requirements=(METFORMIN_RENAL,),
    ),
    QualityCase(
        name="filtered-warfarin",
        question=(
            "What are the main contraindications for warfarin?"
        ),
        requirements=(WARFARIN_CONTRA,),
        drugs=("warfarin",),
        sources=("dailymed",),
    ),
    QualityCase(
        name="filtered-apixaban",
        question=(
            "What are the contraindications for apixaban?"
        ),
        requirements=(APIXABAN_CONTRA,),
        drugs=("apixaban",),
        sources=("dailymed",),
    ),
    QualityCase(
        name="filtered-warfarin-interaction",
        question=(
            "Which serotonin reuptake inhibitors increase bleeding risk "
            "with warfarin, and what monitoring is recommended?"
        ),
        requirements=(WARFARIN_INTERACTION,),
        drugs=("warfarin",),
        sources=("dailymed",),
    ),
    QualityCase(
        name="filtered-sertraline-interaction",
        question=(
            "What is the bleeding risk of sertraline with warfarin and "
            "what international normalized ratio monitoring is recommended?"
        ),
        requirements=(SERTRALINE_INTERACTION,),
        drugs=("sertraline",),
        sources=("dailymed",),
    ),
    QualityCase(
        name="filtered-metformin",
        question=(
            "When is metformin contraindicated because of renal impairment, "
            "and what eGFR threshold applies?"
        ),
        requirements=(METFORMIN_RENAL,),
        drugs=("metformin",),
        sources=("dailymed",),
    ),
)


def matches(
    section: Section,
    requirement: Requirement,
) -> bool:
    text = unicodedata.normalize(
        "NFKC",
        section.text,
    ).casefold()

    return (
        section.doc.file == requirement.file
        and section.number in requirement.section_numbers
        and all(
            re.search(pattern, text)
            for pattern in requirement.patterns
        )
    )


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(ROOT / "corpus")


@pytest.fixture(scope="module")
def retriever(corpus):
    if os.getenv("RUN_RETRIEVAL_QUALITY") != "1":
        pytest.skip(
            "Set RUN_RETRIEVAL_QUALITY=1 for real embedding tests"
        )

    assert settings.EMBEDDING_MODEL, (
        "OPENAI_EMBEDDING_MODEL is required"
    )
    assert settings.API_KEY, (
        "OPENAI_API_KEY is required for a live run"
    )

    embedder = CachedEmbedder()

    print(
        f"\nBuilding cached index: "
        f"{len(corpus.documents)} documents, "
        f"{len(corpus.chunks)} chunks; "
        f"model={embedder.config.model}"
    )

    return HybridRetriever.build(
        corpus,
        embedder,
    )


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_expected_evidence_exists_in_corpus(corpus, case):
    """Verify evaluation expectations against the actual local corpus."""
    missing = [
        requirement.label
        for requirement in case.requirements
        if not any(
            matches(section, requirement)
            for section in corpus.sections
        )
    ]

    assert not missing, (
        f"Evaluation expectation missing from corpus: {missing}"
    )


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_live_retrieval_quality(corpus, retriever, case):
    hits = retriever.search(
        case.question,
        top_k=RETRIEVAL_K,
        drugs=case.drugs,
        sources=case.sources,
    )

    evidence = expand_evidence(
        hits,
        corpus,
        max_sections=EVIDENCE_K,
    )

    total_bytes = sum(
        len(item.section.text.encode("utf-8"))
        for item in evidence
    )

    lines = [
        f"\nCASE: {case.name}",
        f"Question: {case.question}",
        (
            f"Retrieved chunks: {len(hits)}; "
            f"expanded sections: {len(evidence)}"
        ),
        f"Expanded UTF-8 bytes: {total_bytes}",
        "Top chunks:",
    ]

    lines.extend(
        (
            f"  {rank}. {hit.chunk.id} "
            f"RRF={hit.score:.6f} "
            f"vector_rank={hit.vector_rank} "
            f"bm25_rank={hit.bm25_rank}"
        )
        for rank, hit in enumerate(
            hits[:10],
            start=1,
        )
    )

    lines.append("Expanded sections:")

    lines.extend(
        (
            f"  {rank}. {item.section.id} "
            f"chars={len(item.section.text)}"
        )
        for rank, item in enumerate(
            evidence,
            start=1,
        )
    )

    missing = []

    for requirement in case.requirements:
        covered = any(
            matches(item.section, requirement)
            for item in evidence
        )

        # Distinguish retrieval misses from evidence-selection misses.
        first_rank = next(
            (
                rank
                for rank, hit in enumerate(
                    hits,
                    start=1,
                )
                if matches(
                    corpus.parent(hit.chunk),
                    requirement,
                )
            ),
            None,
        )

        lines.append(
            f"Requirement: {requirement.label}; "
            f"covered={covered}; "
            f"first_matching_chunk_rank={first_rank}"
        )

        if not covered:
            missing.append(requirement.label)

    covered_count = len(case.requirements) - len(missing)

    lines.append(
        "Evidence requirement coverage: "
        f"{covered_count}/{len(case.requirements)}"
    )

    report = "\n".join(lines)
    print(report)

    assert hits, report
    assert evidence, report

    # Evidence sections must be unique and complete.
    assert len({
        item.section.id
        for item in evidence
    }) == len(evidence), report

    hits_by_id = {
        hit.chunk.id: hit
        for hit in hits
    }

    for item in evidence:
        assert item.section == corpus.section(
            item.section.id
        ), report

        assert (
            item.citation.id == item.section.id
        ), report
        assert (
            item.citation.file == item.section.doc.file
        ), report
        assert (
            item.citation.url == item.section.doc.url
        ), report
        assert item.citation.url, report
        assert item.matched_chunk_ids, report

        for chunk_id in item.matched_chunk_ids:
            assert (
                hits_by_id[chunk_id].chunk.section_id
                == item.section.id
            ), report

        if case.drugs is not None:
            assert (
                item.section.doc.drug in case.drugs
            ), report

        if case.sources is not None:
            assert (
                item.section.doc.source in case.sources
            ), report

    assert not missing, (
        f"Missing evidence: {missing}\n{report}"
    )