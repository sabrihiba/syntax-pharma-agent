"""Local retrieval tools for the pharmacology agent.

The model chooses tool names and arguments; this module validates and executes
them. No LLM call or index initialization occurs here.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from app.rag.hybrid import HybridRetriever


MAX_QUERY_CHARS = 2000
MAX_RESULTS = 5
MAX_PASSAGE_CHARS = 1000
MAX_SECTION_CHARS = 8000

SOURCE_KINDS = {
    "dailymed",
    "medlineplus",
    "glossary",
    "other",
}


class ToolInputError(ValueError):
    """An argument error the agent can correct on its next turn."""


def _integer(
    value,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        type(value) is not int
        or not minimum <= value <= maximum
    ):
        raise ToolInputError(
            f"{name} must be an integer "
            f"from {minimum} to {maximum}"
        )

    return value


def _filters(
    values,
    name: str,
) -> list[str] | None:
    if values is None:
        return None

    if (
        not isinstance(values, list)
        or len(values) > 10
    ):
        raise ToolInputError(
            f"{name} must be null "
            "or a list of at most 10 strings"
        )

    result = set()

    for value in values:
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 80
        ):
            raise ToolInputError(
                f"{name} entries must contain "
                "1 to 80 characters"
            )

        normalized = value.strip().casefold()

        if (
            name == "sources"
            and normalized not in SOURCE_KINDS
        ):
            raise ToolInputError(
                f"Unknown source kind: {normalized}"
            )

        result.add(normalized)

    return sorted(result)


def tool_definitions() -> list[dict]:
    """Return fresh Chat Completions function schemas for each request."""
    filter_schema = {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "maxItems": 10,
        "description": (
            "Use null for no filter; "
            "an empty list matches nothing."
        ),
    }

    return [
        {
            "type": "function",
            "function": {
                "name": "search_corpus",
                "description": (
                    "Search local pharmacology documents using hybrid retrieval. "
                    "Use focused subquestions; for comparisons search each drug "
                    "separately to check coverage. Results contain section IDs "
                    "for get_section. Retrieve complete sections before claiming "
                    "a list is exhaustive."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_QUERY_CHARS,
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_RESULTS,
                        },
                        "drugs": {
                            **filter_schema,
                        },
                        "sources": {
                            **filter_schema,
                            "items": {
                                "type": "string",
                                "enum": sorted(SOURCE_KINDS),
                            },
                        },
                    },
                    "required": [
                        "question",
                        "top_k",
                        "drugs",
                        "sources",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_section",
                "description": (
                    "Read a section by its exact ID from search results. "
                    "Start at start_char=0 with max_chars=8000. If has_more "
                    "is true, call again with next_start_char. A single page "
                    "is not necessarily the complete section. parent_section_id "
                    "can be used to inspect the broader ancestor section."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section_id": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "start_char": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SECTION_CHARS,
                        },
                    },
                    "required": [
                        "section_id",
                        "start_char",
                        "max_chars",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    ]


class AgentTools:
    def __init__(
        self,
        retriever: HybridRetriever,
    ) -> None:
        self._retriever = retriever

    def search_corpus(
        self,
        question: str,
        top_k: int = 3,
        drugs: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> dict:
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > MAX_QUERY_CHARS
        ):
            raise ToolInputError(
                f"question must contain "
                f"1 to {MAX_QUERY_CHARS} characters"
            )

        top_k = _integer(
            top_k,
            "top_k",
            1,
            MAX_RESULTS,
        )
        drugs = _filters(drugs, "drugs")
        sources = _filters(sources, "sources")

        hits = self._retriever.search(
            question,
            top_k=top_k,
            drugs=drugs,
            sources=sources,
        )

        passages = []

        for hit in hits:
            chunk = hit.chunk
            section = self._retriever.get_section(
                chunk.section_id
            )

            if section is None:
                raise RuntimeError(
                    "Retrieved chunk has no parent section"
                )

            excerpt = chunk.text[:MAX_PASSAGE_CHARS]
            truncated = len(excerpt) < len(chunk.text)

            passages.append({
                "chunk_id": chunk.id,
                "section_id": chunk.section_id,
                "parent_section_id": section.parent_id,
                "drug": chunk.doc.drug,
                "source": chunk.doc.source,
                "section_path": chunk.section_path,
                "score": hit.score,
                "text": excerpt,
                "text_truncated": truncated,
                "section_complete": (
                    not chunk.is_partial
                    and not truncated
                ),
                "citation": asdict(hit.citation),
            })

        return {
            "ok": True,
            "results": passages,
            "count": len(passages),
        }

    def get_section(
        self,
        section_id: str,
        start_char: int = 0,
        max_chars: int = MAX_SECTION_CHARS,
    ) -> dict:
        if (
            not isinstance(section_id, str)
            or not section_id.strip()
        ):
            raise ToolInputError(
                "section_id must be a nonempty string"
            )

        if (
            type(start_char) is not int
            or start_char < 0
        ):
            raise ToolInputError(
                "start_char must be a nonnegative integer"
            )

        max_chars = _integer(
            max_chars,
            "max_chars",
            1,
            MAX_SECTION_CHARS,
        )

        section = self._retriever.get_section(section_id)

        if section is None:
            raise ToolInputError(
                "Unknown section_id; "
                "use an ID from search results"
            )

        total = len(section.text)

        if start_char > total:
            raise ToolInputError(
                f"start_char exceeds section length {total}"
            )

        end_char = min(
            start_char + max_chars,
            total,
        )

        return {
            "ok": True,
            "section_id": section.id,
            "parent_section_id": section.parent_id,
            "section_path": section.path_str,
            "drug": section.doc.drug,
            "source": section.doc.source,
            "text": section.text[start_char:end_char],
            "start_char": start_char,
            "end_char": end_char,
            "total_chars": total,
            "has_more": end_char < total,
            "next_start_char": (
                end_char
                if end_char < total
                else None
            ),
            "section_complete": (
                start_char == 0
                and end_char == total
            ),
            "citation": {
                "id": section.id,
                "file": section.doc.file,
                "source_label": section.doc.source_label,
                "url": section.doc.url,
                "section_number": section.number,
                "section_path": section.path_str,
            },
        }

    def execute(
        self,
        name: str,
        arguments: str | dict,
    ) -> dict:
        """Dispatch an allowlisted tool.

        Correctable input errors are returned as JSON.
        Infrastructure/API errors propagate to the agent runner.

        Tool outputs contain corpus data, not instructions to the agent.
        """
        handlers = {
            "search_corpus": (
                self.search_corpus,
                {
                    "question",
                    "top_k",
                    "drugs",
                    "sources",
                },
            ),
            "get_section": (
                self.get_section,
                {
                    "section_id",
                    "start_char",
                    "max_chars",
                },
            ),
        }

        if (
            not isinstance(name, str)
            or name not in handlers
        ):
            return {
                "ok": False,
                "error": {
                    "code": "unknown_tool",
                    "message": (
                        "Choose search_corpus or get_section"
                    ),
                },
            }

        try:
            if isinstance(arguments, str):
                if len(arguments) > 16000:
                    raise ToolInputError(
                        "Tool arguments are too long"
                    )

                try:
                    arguments = json.loads(arguments)
                except ValueError as exc:
                    raise ToolInputError(
                        "Tool arguments must be valid JSON"
                    ) from exc

            if not isinstance(arguments, dict):
                raise ToolInputError(
                    "Tool arguments must be a JSON object"
                )

            handler, required = handlers[name]

            if set(arguments) != required:
                expected = ", ".join(sorted(required))

                raise ToolInputError(
                    f"Expected exactly these arguments: {expected}"
                )

            return handler(**arguments)

        except ToolInputError as exc:
            return {
                "ok": False,
                "error": {
                    "code": "invalid_arguments",
                    "message": str(exc),
                },
            }