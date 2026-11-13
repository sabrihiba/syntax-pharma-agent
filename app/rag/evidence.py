"""Expand retrieved chunks into complete parent-section evidence.

Multiple matches from the same section become one Evidence item.
The section receives its highest matched-chunk score.

Expansion uses the chunk's immediate section, not automatically an ancestor.
No truncation or context-window budgeting is performed here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from app.rag.chunker import Citation, Corpus, Section
from app.rag.hybrid import HybridResult


@dataclass(frozen=True)
class Evidence:
    section: Section
    score: float
    matched_chunk_ids: tuple[str, ...]

    @property
    def citation(self) -> Citation:
        return Citation(
            id=self.section.id,
            file=self.section.doc.file,
            source_label=self.section.doc.source_label,
            url=self.section.doc.url,
            section_number=self.section.number,
            section_path=self.section.path_str,
        )


def expand_evidence(
    results: Sequence[HybridResult],
    corpus: Corpus,
    max_sections: int = 5,
) -> list[Evidence]:
    """Expand matches, group by section, rank, and limit the evidence.

    All results are validated before limiting. Foreign or stale chunks,
    missing parents, and invalid scores raise ValueError.

    Scores are retrieval ranking signals, not clinical confidence.
    Exact section-score ties are resolved by section ID.
    """
    if type(max_sections) is not int or max_sections < 1:
        raise ValueError(
            "max_sections must be a positive integer"
        )

    # Build fresh lookups so validation reflects the supplied corpus lists.
    sections_by_id: dict[str, Section] = {}

    for section in corpus.sections:
        if section.id in sections_by_id:
            raise ValueError(
                f"Duplicate section id: {section.id}"
            )
        sections_by_id[section.id] = section

    chunks_by_id = {}

    for chunk in corpus.chunks:
        if chunk.id in chunks_by_id:
            raise ValueError(
                f"Duplicate chunk id: {chunk.id}"
            )
        chunks_by_id[chunk.id] = chunk

    evidence_by_section: dict[str, Evidence] = {}

    for result in results:
        chunk = result.chunk
        section_id = chunk.section_id
        section = sections_by_id.get(section_id)

        if section is None:
            raise ValueError(
                f"Missing parent section: {section_id}"
            )

        if chunks_by_id.get(chunk.id) != chunk:
            raise ValueError(
                f"Chunk is outside the corpus or stale: {chunk.id}"
            )

        if section.doc != chunk.doc:
            raise ValueError(
                f"Parent document mismatch: {chunk.id}"
            )

        if (
            not isinstance(result.score, Real)
            or isinstance(result.score, bool)
            or not math.isfinite(result.score)
            or result.score < 0
        ):
            raise ValueError(
                "Evidence scores must be nonnegative finite numbers"
            )

        score = float(result.score)
        existing = evidence_by_section.get(section_id)

        if existing is None:
            evidence_by_section[section_id] = Evidence(
                section=section,
                score=score,
                matched_chunk_ids=(chunk.id,),
            )
            continue

        matched_ids = existing.matched_chunk_ids

        if chunk.id not in matched_ids:
            matched_ids = (*matched_ids, chunk.id)

        evidence_by_section[section_id] = Evidence(
            section=section,
            score=max(existing.score, score),
            matched_chunk_ids=matched_ids,
        )

    return sorted(
        evidence_by_section.values(),
        key=lambda item: (-item.score, item.section.id),
    )[:max_sections]