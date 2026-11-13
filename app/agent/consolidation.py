"""Consolidate evidence actually delivered to the research model.

This module never fetches missing source text or truncates medical passages.
Unknown-offset search excerpts do not prove complete section coverage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from numbers import Real

from app.rag.chunker import Citation


@dataclass(frozen=True)
class SectionSpan:
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class SearchExcerpt:
    chunk_id: str
    text: str
    text_truncated: bool


@dataclass(frozen=True)
class SectionEvidence:
    section_id: str
    parent_section_id: str | None
    drug: str | None
    source: str
    citation: Citation
    spans: tuple[SectionSpan, ...]
    excerpts: tuple[SearchExcerpt, ...]
    total_chars: int | None
    missing_ranges: tuple[tuple[int, int], ...] | None
    complete: bool
    matched_chunk_ids: tuple[str, ...]
    tool_call_ids: tuple[str, ...]
    score: float

    @property
    def full_text(self) -> str | None:
        if not self.complete:
            return None

        return "".join(
            span.text
            for span in self.spans
        )

    def to_dict(self) -> dict:
        # Spans already contain the text, so full_text is not duplicated.
        return asdict(self)


@dataclass
class _Builder:
    section_id: str
    parent_section_id: str | None
    drug: str | None
    source: str
    citation: Citation
    total_chars: int | None = None
    spans: list[SectionSpan] = field(default_factory=list)
    excerpts: dict[str, SearchExcerpt] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    call_ids: list[str] = field(default_factory=list)
    score: float = 0.0


def _integer(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            f"{name} must be a nonnegative integer"
        )

    return value


def _text(
    value,
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value.strip())
    ):
        raise ValueError(
            f"{name} must be a string"
        )

    return value


def _boolean(value, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{name} must be a boolean"
        )

    return value


def _merge_spans(
    spans: list[SectionSpan],
) -> tuple[SectionSpan, ...]:
    merged: list[SectionSpan] = []

    for span in sorted(
        spans,
        key=lambda item: (
            item.start_char,
            item.end_char,
        ),
    ):
        if (
            not merged
            or span.start_char > merged[-1].end_char
        ):
            merged.append(span)
            continue

        previous = merged[-1]

        overlap_end = min(
            previous.end_char,
            span.end_char,
        )
        overlap_size = overlap_end - span.start_char
        offset = span.start_char - previous.start_char

        if (
            previous.text[offset:offset + overlap_size]
            != span.text[:overlap_size]
        ):
            raise ValueError(
                "Conflicting text in overlapping section pages"
            )

        suffix = span.text[
            max(0, previous.end_char - span.start_char):
        ]

        merged[-1] = SectionSpan(
            start_char=previous.start_char,
            end_char=max(
                previous.end_char,
                span.end_char,
            ),
            text=previous.text + suffix,
        )

    return tuple(merged)


def _missing_ranges(
    spans: tuple[SectionSpan, ...],
    total_chars: int | None,
) -> tuple[tuple[int, int], ...] | None:
    if total_chars is None:
        return None

    gaps = []
    cursor = 0

    for span in spans:
        if span.start_char > cursor:
            gaps.append(
                (cursor, span.start_char)
            )

        cursor = max(
            cursor,
            span.end_char,
        )

    if cursor < total_chars:
        gaps.append(
            (cursor, total_chars)
        )

    return tuple(gaps)


def consolidate_evidence(
    records: Sequence[dict],
) -> tuple[SectionEvidence, ...]:
    """Group search hits and section pages by section ID.

    Pass ResearchResult.evidence, which contains delivered outputs only.

    Scores retain the highest search score. They are ranking signals,
    not clinical confidence scores.
    """
    builders: dict[str, _Builder] = {}

    def get_builder(
        item: dict,
        call_id: str,
    ) -> _Builder:
        section_id = _text(
            item.get("section_id"),
            "section_id",
        )

        try:
            citation = Citation(**item["citation"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid evidence citation"
            ) from exc

        if citation.id != section_id:
            raise ValueError(
                "Citation ID does not match section ID"
            )

        parent_id = item.get("parent_section_id")

        if parent_id is not None:
            _text(parent_id, "parent_section_id")

        drug = item.get("drug")

        if drug is not None:
            _text(drug, "drug")

        source = _text(
            item.get("source"),
            "source",
        )

        builder = builders.get(section_id)

        if builder is None:
            builder = _Builder(
                section_id=section_id,
                parent_section_id=parent_id,
                drug=drug,
                source=source,
                citation=citation,
            )
            builders[section_id] = builder

        elif (
            builder.citation != citation
            or builder.drug != drug
            or builder.source != source
            or builder.parent_section_id != parent_id
        ):
            raise ValueError(
                "Conflicting metadata for the same section"
            )

        if call_id not in builder.call_ids:
            builder.call_ids.append(call_id)

        return builder

    def add_span(
        builder: _Builder,
        start: int,
        end: int,
        total: int,
        text: str,
    ) -> None:
        if (
            start > end
            or end > total
            or len(text) != end - start
        ):
            raise ValueError(
                "Section page offsets do not match its text"
            )

        if (
            builder.total_chars is not None
            and builder.total_chars != total
        ):
            raise ValueError(
                "Conflicting section lengths"
            )

        builder.total_chars = total
        builder.spans.append(
            SectionSpan(start, end, text)
        )

    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                "Each evidence record must be an object"
            )

        output = record.get("output")

        if not isinstance(output, dict):
            raise ValueError(
                "Evidence output must be an object"
            )

        if output.get("ok") is not True:
            continue

        call_id = _text(
            record.get("call_id"),
            "call_id",
        )
        tool = record.get("tool")

        if tool == "get_section":
            builder = get_builder(output, call_id)

            text = _text(
                output.get("text"),
                "text",
                allow_empty=True,
            )
            start = _integer(
                output.get("start_char"),
                "start_char",
            )
            end = _integer(
                output.get("end_char"),
                "end_char",
            )
            total = _integer(
                output.get("total_chars"),
                "total_chars",
            )

            claimed_complete = _boolean(
                output.get("section_complete"),
                "section_complete",
            )

            if claimed_complete != (
                start == 0 and end == total
            ):
                raise ValueError(
                    "Incorrect section_complete flag"
                )

            has_more = _boolean(
                output.get("has_more"),
                "has_more",
            )

            if has_more != (end < total):
                raise ValueError(
                    "Incorrect has_more flag"
                )

            next_start = output.get("next_start_char")

            if has_more:
                if (
                    type(next_start) is not int
                    or next_start != end
                ):
                    raise ValueError(
                        "Incorrect next_start_char"
                    )
            elif next_start is not None:
                raise ValueError(
                    "Unexpected next_start_char"
                )

            add_span(
                builder,
                start,
                end,
                total,
                text,
            )

        elif tool == "search_corpus":
            results = output.get("results")

            if not isinstance(results, list):
                raise ValueError(
                    "Search results must be a list"
                )

            for item in results:
                builder = get_builder(
                    item,
                    call_id,
                )

                text = _text(
                    item.get("text"),
                    "text",
                    allow_empty=True,
                )
                chunk_id = _text(
                    item.get("chunk_id"),
                    "chunk_id",
                )
                truncated = _boolean(
                    item.get("text_truncated"),
                    "text_truncated",
                )
                complete = _boolean(
                    item.get("section_complete"),
                    "section_complete",
                )

                score = item.get("score")

                if (
                    not isinstance(score, Real)
                    or isinstance(score, bool)
                    or not math.isfinite(score)
                    or score < 0
                ):
                    raise ValueError(
                        "Invalid search score"
                    )

                builder.score = max(
                    builder.score,
                    float(score),
                )

                excerpt = SearchExcerpt(
                    chunk_id=chunk_id,
                    text=text,
                    text_truncated=truncated,
                )

                previous = builder.excerpts.get(chunk_id)

                if (
                    previous is not None
                    and previous != excerpt
                ):
                    raise ValueError(
                        "Conflicting text for the same chunk ID"
                    )

                builder.excerpts[chunk_id] = excerpt

                if chunk_id not in builder.chunk_ids:
                    builder.chunk_ids.append(chunk_id)

                if complete:
                    if truncated:
                        raise ValueError(
                            "Truncated search text cannot be complete"
                        )

                    add_span(
                        builder,
                        0,
                        len(text),
                        len(text),
                        text,
                    )

        else:
            raise ValueError(
                f"Unsupported evidence tool: {tool}"
            )

    consolidated = []

    for builder in builders.values():
        spans = _merge_spans(builder.spans)

        missing = _missing_ranges(
            spans,
            builder.total_chars,
        )

        complete = (
            builder.total_chars is not None
            and missing == ()
        )

        # Full source text supersedes redundant search previews.
        excerpts = (
            ()
            if complete
            else tuple(
                excerpt
                for excerpt in builder.excerpts.values()
                if not any(
                    excerpt.text in span.text
                    for span in spans
                )
            )
        )

        consolidated.append(
            SectionEvidence(
                section_id=builder.section_id,
                parent_section_id=builder.parent_section_id,
                drug=builder.drug,
                source=builder.source,
                citation=builder.citation,
                spans=spans,
                excerpts=excerpts,
                total_chars=builder.total_chars,
                missing_ranges=missing,
                complete=complete,
                matched_chunk_ids=tuple(builder.chunk_ids),
                tool_call_ids=tuple(builder.call_ids),
                score=builder.score,
            )
        )

    return tuple(
        sorted(
            consolidated,
            key=lambda item: (
                -item.score,
                item.section_id,
            ),
        )
    )