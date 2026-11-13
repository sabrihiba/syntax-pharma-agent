"""Offline tests for the agent's local retrieval tools."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from app.agent.tools import (
    AgentTools,
    MAX_PASSAGE_CHARS,
    MAX_SECTION_CHARS,
    ToolInputError,
    tool_definitions,
)
from app.rag.chunker import chunk_document
from app.rag.hybrid import HybridResult


@pytest.fixture
def context():
    _, sections, chunks = chunk_document(
        "dailymed-warfarin.md",
        "# Warfarin\n"
        "Source: https://example.com/warfarin\n"
        "## 5 WARNINGS\n"
        "Overview.\n"
        "### 5.1 Bleeding\n"
        "First sentence. Second sentence.",
    )

    section_map = {
        section.id: section
        for section in sections
    }

    retriever = Mock()
    retriever.get_section.side_effect = section_map.get
    retriever.search.return_value = [
        HybridResult(
            chunk=chunks[1],
            score=0.03,
        )
    ]

    return (
        AgentTools(retriever),
        retriever,
        sections,
        chunks,
    )


def search_args(**changes):
    return {
        "question": "warfarin bleeding",
        "top_k": 5,
        "drugs": None,
        "sources": None,
        **changes,
    }


def section_args(section_id, **changes):
    return {
        "section_id": section_id,
        "start_char": 0,
        "max_chars": MAX_SECTION_CHARS,
        **changes,
    }


def test_search_preserves_ids_text_and_citations(context):
    tools, _, sections, chunks = context

    output = tools.execute(
        "search_corpus",
        search_args(),
    )

    assert output["ok"]
    assert output["count"] == 1

    item = output["results"][0]

    assert item["chunk_id"] == chunks[1].id
    assert item["section_id"] == sections[1].id
    assert item["parent_section_id"] == sections[0].id
    assert item["text"] == chunks[1].text
    assert item["citation"]["file"] == "dailymed-warfarin.md"
    assert (
        item["citation"]["url"]
        == "https://example.com/warfarin"
    )

    json.dumps(output, allow_nan=False)


def test_search_normalizes_and_passes_filters(context):
    tools, retriever, _, _ = context

    tools.execute(
        "search_corpus",
        search_args(
            drugs=[" WARFARIN "],
            sources=["DailyMed"],
        ),
    )

    retriever.search.assert_called_once_with(
        "warfarin bleeding",
        top_k=5,
        drugs=["warfarin"],
        sources=["dailymed"],
    )


def test_empty_search_results(context):
    tools, retriever, _, _ = context
    retriever.search.return_value = []

    assert tools.execute(
        "search_corpus",
        search_args(),
    ) == {
        "ok": True,
        "results": [],
        "count": 0,
    }


def test_search_marks_partial_sections(context):
    tools, retriever, _, chunks = context

    retriever.search.return_value = [
        HybridResult(
            chunk=replace(
                chunks[1],
                n_parts=2,
            ),
            score=0.03,
        )
    ]

    item = tools.search_corpus("question")["results"][0]

    assert not item["section_complete"]


def test_search_truncation_is_explicit(context):
    tools, retriever, _, chunks = context

    long_chunk = replace(
        chunks[1],
        text="x" * (MAX_PASSAGE_CHARS + 50),
    )

    retriever.search.return_value = [
        HybridResult(
            chunk=long_chunk,
            score=0.03,
        )
    ]

    item = tools.search_corpus("question")["results"][0]

    assert len(item["text"]) == MAX_PASSAGE_CHARS
    assert item["text_truncated"]
    assert not item["section_complete"]


def test_short_leaf_chunk_is_complete(context):
    tools, _, _, _ = context

    item = tools.search_corpus("question")["results"][0]

    assert item["section_complete"]
    assert not item["text_truncated"]


def test_get_section_returns_complete_short_section(context):
    tools, _, sections, _ = context

    result = tools.execute(
        "get_section",
        section_args(sections[1].id),
    )

    assert result["ok"]
    assert result["text"] == sections[1].text
    assert result["section_complete"]
    assert not result["has_more"]
    assert result["next_start_char"] is None
    assert result["citation"]["id"] == sections[1].id


def test_section_paging_reconstructs_exact_text(context):
    tools, _, sections, _ = context
    section = sections[1]

    parts = []
    start = 0

    while True:
        result = tools.get_section(
            section.id,
            start_char=start,
            max_chars=7,
        )

        parts.append(result["text"])

        assert len(result["text"]) <= 7
        assert (
            result["end_char"] - result["start_char"]
            == len(result["text"])
        )
        assert not result["section_complete"]

        if not result["has_more"]:
            break

        assert result["next_start_char"] > start
        start = result["next_start_char"]

    assert "".join(parts) == section.text


def test_ancestor_section_contains_descendant_content(context):
    tools, _, sections, _ = context

    result = tools.get_section(sections[0].id)

    assert sections[1].text in result["text"]
    assert result["section_complete"]


def test_large_section_requires_paging(context):
    tools, retriever, sections, _ = context

    long_section = replace(
        sections[1],
        text="x" * (MAX_SECTION_CHARS + 1),
    )

    retriever.get_section.side_effect = None
    retriever.get_section.return_value = long_section

    result = tools.get_section(long_section.id)

    assert len(result["text"]) == MAX_SECTION_CHARS
    assert result["has_more"]
    assert not result["section_complete"]
    assert result["next_start_char"] == MAX_SECTION_CHARS


def test_unknown_section_is_correctable_error(context):
    tools, _, _, _ = context

    result = tools.execute(
        "get_section",
        section_args("unknown"),
    )

    assert not result["ok"]
    assert result["error"]["code"] == "invalid_arguments"


def test_unknown_tool_is_not_executed(context):
    tools, retriever, _, _ = context

    result = tools.execute(
        "__getattribute__",
        {},
    )

    assert result["error"]["code"] == "unknown_tool"
    retriever.search.assert_not_called()


def test_json_string_arguments_are_supported(context):
    tools, _, _, _ = context

    result = tools.execute(
        "search_corpus",
        json.dumps(search_args()),
    )

    assert result["ok"]


@pytest.mark.parametrize(
    "arguments",
    ["{bad", "[]", "null", 123],
)
def test_invalid_argument_objects(context, arguments):
    tools, retriever, _, _ = context

    result = tools.execute(
        "search_corpus",
        arguments,
    )

    assert result["error"]["code"] == "invalid_arguments"
    retriever.search.assert_not_called()


def test_unexpected_arguments_are_rejected(context):
    tools, retriever, _, _ = context

    result = tools.execute(
        "search_corpus",
        search_args(unexpected=True),
    )

    assert not result["ok"]
    retriever.search.assert_not_called()


def test_missing_required_arguments_are_rejected(context):
    tools, _, _, _ = context

    result = tools.execute(
        "search_corpus",
        {"question": "question"},
    )

    assert not result["ok"]


@pytest.mark.parametrize(
    "changes",
    [
        {"question": ""},
        {"question": None},
        {"question": "x" * 2001},
        {"top_k": 0},
        {"top_k": 11},
        {"top_k": True},
        {"drugs": "warfarin"},
        {"drugs": [""]},
        {"sources": ["unknown"]},
        {"sources": [None]},
    ],
)
def test_search_argument_validation(context, changes):
    tools, retriever, _, _ = context

    result = tools.execute(
        "search_corpus",
        search_args(**changes),
    )

    assert result["error"]["code"] == "invalid_arguments"
    retriever.search.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"start_char": -1},
        {"start_char": True},
        {"start_char": 100000},
        {"max_chars": 0},
        {"max_chars": 8001},
        {"max_chars": True},
        {"section_id": ""},
    ],
)
def test_section_argument_validation(context, changes):
    tools, _, sections, _ = context

    arguments = section_args(sections[1].id)
    arguments.update(changes)

    result = tools.execute(
        "get_section",
        arguments,
    )

    assert result["error"]["code"] == "invalid_arguments"


def test_api_failures_propagate(context):
    tools, retriever, _, _ = context

    retriever.search.side_effect = RuntimeError(
        "upstream unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="upstream unavailable",
    ):
        tools.execute(
            "search_corpus",
            search_args(),
        )


def test_direct_methods_raise_input_errors(context):
    tools, _, _, _ = context

    with pytest.raises(ToolInputError):
        tools.get_section("missing")


def test_schemas_define_two_strict_functions():
    definitions = tool_definitions()

    assert {
        item["function"]["name"]
        for item in definitions
    } == {
        "search_corpus",
        "get_section",
    }

    for item in definitions:
        function = item["function"]
        parameters = function["parameters"]

        assert item["type"] == "function"
        assert function["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(
            parameters["properties"]
        )

    json.dumps(definitions)


def test_schemas_are_fresh_objects():
    first = tool_definitions()

    first[0]["function"]["parameters"]["properties"]["drugs"][
        "type"
    ] = "bad"

    assert tool_definitions()[0]["function"]["parameters"][
        "properties"
    ]["drugs"]["type"] == [
        "array",
        "null",
    ]