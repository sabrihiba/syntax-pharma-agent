"""Offline planner and LangGraph research tests. No API calls."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent import research as module
from app.agent.planner import (
    AgentProtocolError,
    create_plan,
    serialized_size,
)
from app.agent.research import AgentConfig, ResearchAgent
from app.agent.tools import AgentTools
from app.rag.chunker import chunk_document
from app.rag.hybrid import HybridResult


def reply(content=None, calls=None, finish_reason=None):
    return SimpleNamespace(choices=[
        SimpleNamespace(
            finish_reason=(
                finish_reason
                or ("tool_calls" if calls else "stop")
            ),
            message=SimpleNamespace(
                content=content,
                tool_calls=calls,
            ),
        )
    ])


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def plan_reply():
    return reply(json.dumps({
        "steps": [
            "Find the relevant contraindications section."
        ]
    }))


def make_agent(*responses, tools=None, **config):
    client = Mock()
    client.chat.completions.create.side_effect = [
        plan_reply(),
        *responses,
    ]

    tools = tools or Mock()

    agent = ResearchAgent(
        tools,
        AgentConfig(model="test-model", **config),
        client=client,
    )

    return agent, client, tools


@pytest.fixture(autouse=True)
def prevent_real_clients(monkeypatch):
    monkeypatch.setattr(
        module,
        "OpenAI",
        Mock(side_effect=AssertionError(
            "Unexpected real client"
        )),
    )


def test_planner_returns_validated_short_steps():
    client = Mock()
    client.chat.completions.create.return_value = reply(
        '{"steps":[" Search drug A. ","Search drug B."]}'
    )

    assert create_plan(
        "Compare drugs", client=client, model="test"
    ) == ("Search drug A.", "Search drug B.")

    request = client.chat.completions.create.call_args.kwargs
    assert "tools" not in request
    assert request["model"] == "test"


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        "[]",
        "{}",
        '{"steps":[]}',
        '{"steps":[""]}',
        '{"steps":[123]}',
        json.dumps({"steps": ["x"] * 7}),
        json.dumps({"steps": ["x" * 501]}),
        '{"steps":["one"],"answer":"unsupported"}',
    ],
)
def test_invalid_plans_are_rejected(content):
    client = Mock()
    client.chat.completions.create.return_value = reply(content)

    with pytest.raises(AgentProtocolError):
        create_plan("question", client=client, model="test")


def test_planner_budget_is_checked_before_api_call():
    client = Mock()

    with pytest.raises(ValueError, match="byte budget"):
        create_plan(
            "question",
            client=client,
            model="test",
            max_context_bytes=1,
        )

    client.chat.completions.create.assert_not_called()


def test_planner_output_limit_is_explicit():
    client = Mock()
    client.chat.completions.create.return_value = reply(
        "", finish_reason="length"
    )

    with pytest.raises(AgentProtocolError, match="token limit"):
        create_plan("question", client=client, model="test")


def test_model_selects_tools_and_receives_correlated_results():
    first = tool_call("call-1", "search_corpus", {
        "question": "warfarin contraindications",
        "top_k": 5,
        "drugs": ["warfarin"],
        "sources": ["dailymed"],
    })
    second = tool_call("call-2", "get_section", {
        "section_id": "label#4",
        "start_char": 0,
        "max_chars": 8000,
    })

    agent, client, tools = make_agent(
        reply(calls=[first]),
        reply(calls=[second]),
        reply("Collected."),
    )

    tools.execute.side_effect = [
        {
            "ok": True,
            "results": [{
                "section_id": "label#4",
                "text": "passage",
            }],
        },
        {
            "ok": True,
            "text": "Complete section.",
            "section_complete": True,
        },
    ]

    result = agent.collect(
        "What are warfarin contraindications?"
    )

    assert result.status == "model_finished"
    assert len(result.evidence) == 2
    assert [entry.name for entry in result.trace] == [
        "search_corpus", "get_section"
    ]
    assert all(entry.delivered for entry in result.trace)

    tools.execute.assert_any_call(
        "search_corpus", first.function.arguments
    )
    tools.execute.assert_any_call(
        "get_section", second.function.arguments
    )

    messages = (
        client.chat.completions.create.call_args
        .kwargs["messages"]
    )

    assert [
        message["tool_call_id"]
        for message in messages
        if message["role"] == "tool"
    ] == ["call-1", "call-2"]


def test_multiple_model_chosen_searches_are_not_replaced_by_fixed_queries():
    calls = [
        tool_call("a", "search_corpus", {"question": "drug A"}),
        tool_call("b", "search_corpus", {"question": "drug B"}),
    ]

    agent, _, tools = make_agent(
        reply(calls=calls), reply("Done")
    )
    tools.execute.return_value = {
        "ok": True,
        "results": [{"text": "evidence"}],
    }

    result = agent.collect("Compare drugs")

    assert result.status == "model_finished"
    assert tools.execute.call_count == 2
    assert (
        tools.execute.call_args_list[0].args[1]
        == calls[0].function.arguments
    )
    assert (
        tools.execute.call_args_list[1].args[1]
        == calls[1].function.arguments
    )


def test_correctable_tool_error_can_be_followed_by_a_retry():
    agent, _, tools = make_agent(
        reply(calls=[tool_call("a", "unknown", {})]),
        reply(calls=[tool_call("b", "search_corpus", {})]),
        reply("Collected"),
    )

    tools.execute.side_effect = [
        {"ok": False, "error": {"code": "unknown_tool"}},
        {"ok": True, "results": [{"text": "evidence"}]},
    ]

    result = agent.collect("question")

    assert [entry.ok for entry in result.trace] == [False, True]
    assert len(result.evidence) == 1


def test_empty_search_results_do_not_count_as_evidence():
    agent, _, tools = make_agent(
        reply(calls=[tool_call("a", "search_corpus", {})]),
        reply("Not found"),
    )
    tools.execute.return_value = {
        "ok": True, "results": [], "count": 0,
    }

    result = agent.collect("question")

    assert result.status == "no_evidence"
    assert result.evidence == ()


def test_model_stopping_without_tools_is_not_successful_research():
    agent, _, tools = make_agent(
        reply("I know the answer")
    )

    result = agent.collect("question")

    assert result.status == "no_evidence"
    tools.execute.assert_not_called()


def test_tool_limit_stops_additional_execution():
    agent, client, tools = make_agent(
        reply(calls=[
            tool_call("a", "search_corpus", {}),
            tool_call("b", "get_section", {}),
        ]),
        max_tool_calls=1,
        max_batch_retries=0,
    )
    tools.execute.return_value = {
        "ok": True,
        "results": [{"text": "evidence"}],
    }

    result = agent.collect("question")

    assert result.status == "tool_limit"
    assert tools.execute.call_count == 0
    assert client.chat.completions.create.call_count == 2
    assert result.trace == ()
    assert "No calls in this batch ran" in result.notes


def test_turn_limit_is_explicit():
    agent, _, tools = make_agent(
        reply(calls=[tool_call("a", "search_corpus", {})]),
        max_turns=1,
    )
    tools.execute.return_value = {
        "ok": False, "error": {"code": "bad"},
    }

    assert agent.collect("question").status == "turn_limit"


def test_output_limit_is_explicit():
    agent, _, tools = make_agent(
        reply("", finish_reason="length")
    )

    assert agent.collect("question").status == "output_limit"
    tools.execute.assert_not_called()


def test_large_tool_output_is_not_delivered_or_retained():
    agent, client, tools = make_agent(
        reply(calls=[tool_call("a", "get_section", {})]),
        reply("Requested evidence could not be collected."),
        max_context_bytes=12000,
    )
    tools.execute.return_value = {
        "ok": True, "text": "x" * 20000,
    }

    result = agent.collect("question")

    assert result.status == "no_evidence"
    assert result.evidence == ()
    assert not result.trace[0].delivered

    request = client.chat.completions.create.call_args.kwargs
    tool_messages = [
        message for message in request["messages"]
        if message["role"] == "tool"
    ]
    feedback = json.loads(tool_messages[0]["content"])

    assert feedback["error"]["code"] == "context_budget"
    assert "x" * 100 not in tool_messages[0]["content"]

    for call in client.chat.completions.create.call_args_list:
        assert serialized_size(call.kwargs) <= 12000


def test_partial_section_remains_explicitly_partial():
    agent, _, tools = make_agent(
        reply(calls=[tool_call("a", "get_section", {})]),
        reply("More pages are missing."),
    )
    tools.execute.return_value = {
        "ok": True,
        "text": "partial",
        "section_complete": False,
        "has_more": True,
        "next_start_char": 8000,
    }

    result = agent.collect("question")

    assert not result.evidence[0]["output"]["section_complete"]
    assert result.status == "model_finished"
    assert "missing" in result.notes


def test_repeated_tool_call_ids_are_rejected():
    agent, _, tools = make_agent(
        reply(calls=[tool_call("same", "search_corpus", {})]),
        reply(calls=[tool_call("same", "get_section", {})]),
    )
    tools.execute.return_value = {"ok": True, "results": []}

    with pytest.raises(AgentProtocolError, match="tool-call ID"):
        agent.collect("question")


def test_infrastructure_errors_propagate():
    agent, _, tools = make_agent(
        reply(calls=[tool_call("a", "search_corpus", {})])
    )
    tools.execute.side_effect = RuntimeError(
        "upstream unavailable"
    )

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        agent.collect("question")


def test_real_local_tools_integrate_with_fake_model():
    _, sections, chunks = chunk_document(
        "dailymed-warfarin.md",
        "# Warfarin\n## 4 CONTRAINDICATIONS\nExample passage.",
    )

    retriever = Mock()
    retriever.search.return_value = [
        HybridResult(chunks[0], 0.03)
    ]
    retriever.get_section.return_value = sections[0]
    tools = AgentTools(retriever)

    agent, _, _ = make_agent(
        reply(calls=[tool_call("a", "search_corpus", {
            "question": "warfarin contraindications",
            "top_k": 5,
            "drugs": ["warfarin"],
            "sources": ["dailymed"],
        })]),
        reply(calls=[tool_call("b", "get_section", {
            "section_id": sections[0].id,
            "start_char": 0,
            "max_chars": 8000,
        })]),
        reply("Evidence collected."),
        tools=tools,
    )

    result = agent.collect(
        "What are warfarin contraindications?"
    )

    assert result.status == "model_finished"
    assert (
        result.evidence[1]["output"]["citation"]["file"]
        == "dailymed-warfarin.md"
    )
    assert (
        result.evidence[1]["output"]["text"]
        == sections[0].text
    )


@pytest.mark.parametrize(
    "question",
    ["", " ", None, 123, "x" * 4001],
)
def test_invalid_questions_do_not_create_client(question):
    agent = ResearchAgent(
        Mock(), AgentConfig(model="test")
    )

    with pytest.raises(ValueError, match="question"):
        agent.collect(question)

    module.OpenAI.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        {"model": ""},
        {"max_turns": 0},
        {"max_tool_calls": -1},
        {"max_context_bytes": True},
        {"max_output_tokens": 0},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        AgentConfig(**{"model": "test", **config})


def test_missing_api_key_is_explicit(monkeypatch):
    monkeypatch.setattr(module.settings, "API_KEY", None)
    agent = ResearchAgent(
        Mock(), AgentConfig(model="test")
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        agent.collect("question")

    module.OpenAI.assert_not_called()


def test_real_langgraph_has_explicit_nodes():
    agent, _, _ = make_agent(reply("No evidence"))
    nodes = agent.graph.get_graph().nodes

    assert {"planner", "model", "tools"}.issubset(nodes)


def test_multiple_collections_have_independent_graph_state():
    agent, client, _ = make_agent(
        reply("First run"),
        plan_reply(),
        reply("Second run"),
    )

    first = agent.collect("First question")
    second = agent.collect("Second question")

    assert first.question == "First question"
    assert second.question == "Second question"
    assert first.trace == second.trace == ()
    assert first.status == second.status == "no_evidence"
    assert first.rejected_batches == second.rejected_batches == ()
    assert client.chat.completions.create.call_count == 4


def test_oversized_tool_batch_outputs_are_discarded_together():
    agent, client, tools = make_agent(
        reply(calls=[
            tool_call("a", "get_section", {}),
            tool_call("b", "get_section", {}),
        ]),
        reply("Evidence is incomplete."),
        max_context_bytes=12000,
    )
    tools.execute.return_value = {
        "ok": True, "text": "x" * 8500,
    }

    result = agent.collect("question")

    assert tools.execute.call_count == 2
    assert result.status == "no_evidence"
    assert result.evidence == ()
    assert len(result.trace) == 2
    assert all(not item.delivered for item in result.trace)

    messages = (
        client.chat.completions.create.call_args
        .kwargs["messages"]
    )
    feedback = [
        message for message in messages
        if message["role"] == "tool"
    ]

    assert [
        message["tool_call_id"] for message in feedback
    ] == ["a", "b"]
    assert all(
        json.loads(message["content"])["error"]["code"]
        == "context_budget"
        for message in feedback
    )


def test_graph_recursion_limit_allows_configured_turn_limit():
    responses = [
        reply(calls=[
            tool_call(f"call-{index}", "search_corpus", {})
        ])
        for index in range(15)
    ]

    agent, _, tools = make_agent(
        *responses,
        max_turns=15,
        max_tool_calls=20,
        max_context_bytes=100000,
    )
    tools.execute.return_value = {"ok": True, "results": []}

    result = agent.collect("question")

    assert result.status == "turn_limit"
    assert len(result.trace) == 15


def test_model_requests_single_tool_call_per_turn():
    agent, client, _ = make_agent(reply("No evidence"))
    agent.collect("question")

    assert (
        client.chat.completions.create.call_args
        .kwargs["parallel_tool_calls"]
        is False
    )


def test_budget_feedback_allows_a_smaller_retry():
    agent, _, tools = make_agent(
        reply(calls=[tool_call("a", "get_section", {})]),
        reply(calls=[tool_call("b", "get_section", {})]),
        reply("Collected a smaller page."),
        max_context_bytes=12000,
    )
    tools.execute.side_effect = [
        {"ok": True, "text": "x" * 20000},
        {
            "ok": True,
            "text": "Small evidence passage.",
            "section_complete": False,
        },
    ]

    result = agent.collect("question")

    assert result.status == "model_finished"
    assert len(result.evidence) == 1
    assert result.evidence[0]["call_id"] == "b"
    assert [
        item.delivered for item in result.trace
    ] == [False, True]


def test_budget_stops_when_even_feedback_cannot_fit():
    agent, _, tools = make_agent(
        reply("Unused"), max_context_bytes=1
    )
    tools.execute.return_value = {
        "ok": True, "text": "large output",
    }

    state = {
        "pending_calls": [{
            "id": "a",
            "function": {
                "name": "get_section",
                "arguments": "{}",
            },
        }],
        "messages": [],
        "trace": [],
        "evidence": [],
        "budget_retries": 0,
    }

    update = agent._tools_node(state)

    assert update["status"] == "context_limit"
    assert not update["trace"][0].delivered
    assert "evidence" not in update


@pytest.mark.parametrize("retry_limit", [0, 1, 2])
def test_budget_retries_have_an_explicit_limit(retry_limit):
    responses = [
        reply(calls=[
            tool_call(f"large-{index}", "get_section", {})
        ])
        for index in range(retry_limit + 1)
    ]

    agent, client, tools = make_agent(
        *responses,
        max_context_bytes=12000,
        max_budget_retries=retry_limit,
    )
    tools.execute.return_value = {
        "ok": True, "text": "x" * 20000,
    }

    result = agent.collect("question")

    assert result.status == "context_limit"
    assert "Budget-retry limit" in result.notes
    assert tools.execute.call_count == retry_limit + 1
    assert (
        client.chat.completions.create.call_count
        == retry_limit + 2
    )
    assert result.evidence == ()
    assert all(not item.delivered for item in result.trace)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_budget_retry_configuration(value):
    with pytest.raises(ValueError, match="max_budget_retries"):
        AgentConfig(
            model="test",
            max_budget_retries=value,
        )


def test_original_context_budget_is_preserved():
    assert AgentConfig(model="test").max_context_bytes == 48000


def test_small_previews_preserve_full_parent_and_all_pages():
    body = (
        "- "
        + "Important condition. " * 90
        + "\n\n"
        + "Additional qualification. " * 100
    )

    _, sections, chunks = chunk_document(
        "dailymed-example.md",
        "# Example\n## 4 CONTRAINDICATIONS\n" + body,
    )

    retriever = Mock()
    retriever.search.return_value = [
        HybridResult(chunks[0], 0.03)
    ]
    retriever.get_section.return_value = sections[0]
    tools = AgentTools(retriever)

    preview = tools.search_corpus(
        "contraindications"
    )["results"][0]

    assert preview["text_truncated"]
    assert not preview["section_complete"]

    parts = []
    start = 0

    while True:
        page = tools.get_section(
            sections[0].id,
            start_char=start,
            max_chars=700,
        )
        parts.append(page["text"])

        if not page["has_more"]:
            break

        start = page["next_start_char"]

    reconstructed = "".join(parts)

    assert reconstructed == sections[0].text
    assert "Important condition." in reconstructed
    assert "Additional qualification." in reconstructed


def oversized_calls(prefix="large"):
    return [
        tool_call(f"{prefix}-1", "search_corpus", {}),
        tool_call(f"{prefix}-2", "get_section", {}),
    ]


def test_oversized_batch_can_be_revised_without_consuming_allowance():
    revised = tool_call(
        "large-1", "search_corpus", {}
    )

    agent, _, tools = make_agent(
        reply(calls=oversized_calls()),
        reply(calls=[revised]),
        max_tool_calls=1,
    )
    tools.execute.return_value = {
        "ok": True,
        "results": [{"text": "evidence"}],
    }

    result = agent.collect("question")

    assert tools.execute.call_count == 1
    assert len(result.trace) == 1
    assert result.trace[0].call_id == "large-1"
    assert len(result.rejected_batches) == 1

    rejection = result.rejected_batches[0]

    assert rejection.requested_calls == 2
    assert rejection.remaining_calls == 1
    assert rejection.retry_scheduled

    # The successful call uses the final available tool call.
    assert result.status == "tool_limit"
    assert len(result.evidence) == 1


def test_controller_feedback_is_temporary_and_rejected_calls_are_absent():
    agent, client, tools = make_agent(
        reply(calls=[
            tool_call("rejected-1", "search_corpus", {}),
            tool_call("rejected-2", "get_section", {}),
            tool_call("rejected-3", "get_section", {}),
        ]),
        reply(calls=[
            tool_call("accepted", "search_corpus", {})
        ]),
        reply("Collected."),
        max_tool_calls=2,
    )

    tools.execute.return_value = {
        "ok": True,
        "results": [{"text": "evidence"}],
    }

    result = agent.collect("question")
    requests = client.chat.completions.create.call_args_list

    retry_messages = requests[2].kwargs["messages"]

    assert retry_messages[-1]["role"] == "user"
    assert (
        "Controller feedback:"
        in retry_messages[-1]["content"]
    )
    assert (
        "allowance of 2"
        in retry_messages[-1]["content"]
    )

    final_messages = requests[3].kwargs["messages"]

    assert not any(
        isinstance(message.get("content"), str)
        and message["content"].startswith(
            "Controller feedback:"
        )
        for message in final_messages
    )

    history_ids = [
        call["id"]
        for message in final_messages
        for call in message.get("tool_calls", [])
    ]

    assert history_ids == ["accepted"]
    assert result.status == "model_finished"


def test_repeated_oversized_batches_have_two_retries():
    agent, client, tools = make_agent(
        *(
            reply(calls=oversized_calls(f"batch-{index}"))
            for index in range(3)
        ),
        max_tool_calls=1,
    )

    result = agent.collect("question")

    assert result.status == "tool_limit"
    assert "Batch-retry limit" in result.notes
    assert result.trace == ()
    tools.execute.assert_not_called()
    assert client.chat.completions.create.call_count == 4

    assert [
        item.retry_scheduled
        for item in result.rejected_batches
    ] == [True, True, False]


def test_batch_retries_count_toward_model_turn_limit():
    agent, client, tools = make_agent(
        reply(calls=oversized_calls("one")),
        reply(calls=oversized_calls("two")),
        max_tool_calls=1,
        max_turns=2,
    )

    result = agent.collect("question")

    assert result.status == "turn_limit"
    assert client.chat.completions.create.call_count == 3
    assert len(result.rejected_batches) == 2
    tools.execute.assert_not_called()


def test_failed_batch_preserves_previously_collected_evidence():
    agent, _, tools = make_agent(
        reply(calls=[
            tool_call("valid", "search_corpus", {})
        ]),
        reply(calls=oversized_calls()),
        max_tool_calls=2,
        max_batch_retries=0,
    )

    tools.execute.return_value = {
        "ok": True,
        "results": [{"text": "existing evidence"}],
    }

    result = agent.collect("question")

    assert result.status == "tool_limit"
    assert len(result.trace) == 1
    assert len(result.evidence) == 1
    assert result.evidence[0]["call_id"] == "valid"
    assert len(result.rejected_batches) == 1


def test_batch_feedback_must_fit_context_budget():
    agent, client, tools = make_agent(
        reply("unused"),
        max_context_bytes=1,
    )

    state = {
        "messages": [],
        "batch_retries": 0,
        "rejected_batches": [],
    }

    update = agent._reject_batch(
        state,
        turns=1,
        requested=2,
        remaining=1,
    )

    assert update["status"] == "context_limit"
    assert not update["rejected_batches"][0].retry_scheduled
    assert update["controller_feedback"] == ""
    client.chat.completions.create.assert_not_called()
    tools.execute.assert_not_called()


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_batch_retry_configuration(value):
    with pytest.raises(ValueError, match="max_batch_retries"):
        AgentConfig(
            model="test",
            max_batch_retries=value,
        )