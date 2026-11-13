"""Offline API and service tests."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from test_agent_answer import research
from test_agent_grounding import draft

from app.agent.answer import DISCLAIMER
from app.agent.grounding import GroundingCheck, GroundingReview
from app.agent.planner import AgentProtocolError
from app.agent.research import RejectedBatch, ToolTrace
from app.agent.workflow import AnswerAttempt, AnswerResult
from app.main import create_app
from app.service import (
    AskService,
    ServiceBusyError,
    ServiceClosedError,
)
import app.service as service_module


@pytest.fixture
def reviewed_result(research, draft):
    review = GroundingReview(
        claims=tuple(
            GroundingCheck(
                index,
                True,
                "Supported.",
            )
            for index in range(
                1,
                len(draft.statements) + 1,
            )
        ),
        coverage=(
            GroundingCheck(
                1,
                True,
                "Covered.",
            ),
        ),
        limitations_supported=True,
        limitations_explanation="No unsupported limitations.",
    )

    research = replace(
        research,
        trace=(
            ToolTrace(
                "call-1",
                "get_section",
                '{"section_id":"example"}',
                True,
                True,
            ),
        ),
        rejected_batches=(
            RejectedBatch(
                1,
                3,
                2,
                True,
            ),
        ),
    )

    return AnswerResult(
        research=research,
        draft=draft,
        review=review,
        status="answer_reviewed",
        notes="Review passed.",
        attempts=(
            AnswerAttempt(draft, review),
        ),
    )


@pytest.fixture
def fake_service(reviewed_result):
    return SimpleNamespace(
        ask=Mock(return_value=reviewed_result),
        close=Mock(),
    )


@pytest.fixture
def api_client(fake_service):
    application = create_app(
        service_factory=lambda: fake_service
    )

    with TestClient(application) as client:
        yield client


def test_success_returns_reviewed_answer_and_aligned_references(
    api_client,
    fake_service,
    reviewed_result,
):
    response = api_client.post(
        "/ask",
        json={
            "question": reviewed_result.research.question,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "answer_reviewed"
    assert "[S1]" in data["answer"]
    assert data["citations"][0]["reference"] == "S1"
    assert (
        data["citations"][0]["id"]
        == reviewed_result.draft.citations[0].id
    )
    assert (
        data["citations"][0]["label"]
        == reviewed_result.draft.citations[0].label
    )
    assert data["disclaimer"] == DISCLAIMER
    assert data["trace"]["research_status"] == "evidence_reviewed"
    assert (
        data["trace"]["tool_calls"][0]["arguments"]
        == {"section_id": "example"}
    )
    assert data["trace"]["evidence"][0]["complete"] is True
    assert (
        data["trace"]["answer_reviews"][0]["sufficient"]
        is True
    )
    assert (
        data["trace"]["rejected_batches"][0]["requested_calls"]
        == 3
    )
    assert data["trace"]["elapsed_ms"] >= 0
    assert "messages" not in data["trace"]


def test_whitespace_is_normalized(
    api_client,
    fake_service,
):
    response = api_client.post(
        "/ask",
        json={"question": "  Example question?  "},
    )

    assert response.status_code == 200

    fake_service.ask.assert_called_once_with(
        "Example question?"
    )

    assert response.json()["question"] == "Example question?"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": 42},
        {"question": None},
        {"question": "x" * 4001},
        {
            "question": "Example?",
            "model": "override",
        },
    ],
)
def test_invalid_requests_never_run_agent(
    api_client,
    fake_service,
    body,
):
    response = api_client.post(
        "/ask",
        json=body,
    )

    assert response.status_code == 422
    fake_service.ask.assert_not_called()


def test_invalid_json_never_runs_agent(
    api_client,
    fake_service,
):
    response = api_client.post(
        "/ask",
        content="{",
        headers={
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 422
    fake_service.ask.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        "research_incomplete",
        "grounding_failed",
        "revision_context_limit",
    ],
)
def test_unapproved_draft_is_not_returned(
    api_client,
    fake_service,
    reviewed_result,
    status,
):
    fake_service.ask.return_value = replace(
        reviewed_result,
        status=status,
        notes="Required evidence remains missing.",
    )

    response = api_client.post(
        "/ask",
        json={"question": "Example?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == status
    assert data["answer"] is None
    assert data["citations"] == []
    assert "draft" not in data
    assert "Example source-based statement." not in response.text


def test_bad_tool_arguments_remain_visible_without_crashing(
    api_client,
    fake_service,
    reviewed_result,
):
    research = replace(
        reviewed_result.research,
        trace=(
            ToolTrace(
                "bad",
                "search_corpus",
                "{invalid",
                False,
                True,
            ),
        ),
    )

    fake_service.ask.return_value = replace(
        reviewed_result,
        research=research,
    )

    response = api_client.post(
        "/ask",
        json={"question": "Example?"},
    )

    assert response.status_code == 200
    assert (
        response.json()["trace"]["tool_calls"][0]["arguments"]
        == "{invalid"
    )


def test_factory_runs_once_and_closes_on_shutdown(
    fake_service,
):
    factory = Mock(return_value=fake_service)
    application = create_app(
        service_factory=factory
    )

    factory.assert_not_called()

    with TestClient(application) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "ready": True,
        }

        client.post(
            "/ask",
            json={"question": "First?"},
        )
        client.post(
            "/ask",
            json={"question": "Second?"},
        )

        factory.assert_called_once()
        fake_service.close.assert_not_called()

    fake_service.close.assert_called_once()


def test_request_without_startup_is_unavailable(
    fake_service,
):
    client = TestClient(
        create_app(
            service_factory=lambda: fake_service
        )
    )

    response = client.post(
        "/ask",
        json={"question": "Example?"},
    )

    assert response.status_code == 503
    fake_service.ask.assert_not_called()
    client.close()


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("busy", 503),
        ("closed", 503),
        ("timeout", 504),
        ("connection", 502),
        ("rate_limit", 503),
        ("provider_status", 502),
        ("protocol", 502),
        ("unexpected", 500),
    ],
)
def test_errors_have_controlled_http_status_without_leaking_details(
    api_client,
    fake_service,
    kind,
    expected,
):
    request = httpx.Request(
        "POST",
        "https://provider.example/v1",
    )

    errors = {
        "busy": ServiceBusyError("SECRET"),
        "closed": ServiceClosedError("SECRET"),
        "timeout": APITimeoutError(
            request=request,
        ),
        "connection": APIConnectionError(
            message="SECRET",
            request=request,
        ),
        "rate_limit": RateLimitError(
            "SECRET",
            response=httpx.Response(
                429,
                request=request,
            ),
            body=None,
        ),
        "provider_status": APIStatusError(
            "SECRET",
            response=httpx.Response(
                500,
                request=request,
            ),
            body=None,
        ),
        "protocol": AgentProtocolError("SECRET"),
        "unexpected": ValueError("SECRET"),
    }

    fake_service.ask.side_effect = errors[kind]

    response = api_client.post(
        "/ask",
        json={"question": "Example?"},
    )

    assert response.status_code == expected
    assert "SECRET" not in response.text
    assert "answer" not in response.json()


def test_service_prevents_overlapping_runs_and_releases_lock():
    workflow = Mock()
    service = AskService(workflow, Mock())

    def run(question):
        with pytest.raises(ServiceBusyError):
            service.ask(
                "Overlapping question"
            )

        return question

    workflow.run.side_effect = run

    assert service.ask("First") == "First"
    assert service.ask("Second") == "Second"


def test_service_releases_lock_after_failure():
    workflow = Mock()
    workflow.run.side_effect = [
        RuntimeError("failed"),
        "recovered",
    ]

    service = AskService(workflow, Mock())

    with pytest.raises(RuntimeError):
        service.ask("First")

    assert service.ask("Second") == "recovered"


def test_close_is_idempotent_and_closed_service_cannot_run():
    workflow = Mock()
    client = Mock()
    service = AskService(workflow, client)

    service.close()
    service.close()

    client.close.assert_called_once()

    with pytest.raises(ServiceClosedError):
        service.ask("Example?")

    workflow.run.assert_not_called()


def mocked_builder(monkeypatch, *, fail=False):
    monkeypatch.setattr(
        service_module.settings,
        "API_KEY",
        "test-key",
    )
    monkeypatch.setattr(
        service_module.settings,
        "MODEL",
        "test-model",
    )
    monkeypatch.setattr(
        service_module.settings,
        "EMBEDDING_MODEL",
        "test-embedding",
    )

    client = Mock()

    monkeypatch.setattr(
        service_module,
        "OpenAI",
        Mock(return_value=client),
    )

    corpus = SimpleNamespace(
        chunks=[object()],
    )

    monkeypatch.setattr(
        service_module,
        "load_corpus",
        Mock(return_value=corpus),
    )
    monkeypatch.setattr(
        service_module,
        "CachedEmbedder",
        Mock(),
    )
    monkeypatch.setattr(
        service_module.HybridRetriever,
        "build",
        Mock(
            side_effect=(
                RuntimeError("failed")
                if fail
                else None
            )
        ),
    )
    monkeypatch.setattr(
        service_module,
        "ReviewedResearchAgent",
        Mock(),
    )

    workflow = Mock()

    monkeypatch.setattr(
        service_module,
        "AnswerWorkflow",
        Mock(return_value=workflow),
    )

    return client, workflow


def test_builder_reuses_client_and_builds_index_only_once(
    monkeypatch,
):
    client, workflow = mocked_builder(
        monkeypatch
    )

    service = service_module.build_service()

    service_module.HybridRetriever.build.assert_called_once()
    service_module.CachedEmbedder.assert_called_once_with(
        client=client
    )
    service_module.ReviewedResearchAgent.assert_called_once()
    service_module.AnswerWorkflow.assert_called_once()

    service.ask("First")
    service.ask("Second")

    service_module.HybridRetriever.build.assert_called_once()
    assert workflow.run.call_count == 2

    service.close()
    client.close.assert_called_once()


def test_builder_closes_client_if_initialization_fails(
    monkeypatch,
):
    client, _ = mocked_builder(
        monkeypatch,
        fail=True,
    )

    with pytest.raises(
        RuntimeError,
        match="failed",
    ):
        service_module.build_service()

    client.close.assert_called_once()


def test_startup_failure_is_not_hidden():
    factory = Mock(
        side_effect=RuntimeError(
            "startup failed"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="startup failed",
    ):
        with TestClient(
            create_app(
                service_factory=factory
            )
        ):
            pass


@pytest.mark.parametrize(
    "field",
    ["draft", "review"],
)
def test_mislabeled_success_does_not_expose_an_unapproved_answer(
    api_client,
    fake_service,
    reviewed_result,
    field,
):
    fake_service.ask.return_value = replace(
        reviewed_result,
        **{field: None},
    )

    response = api_client.post(
        "/ask",
        json={"question": "Example?"},
    )

    assert response.status_code == 500
    assert "Example source-based statement." not in response.text


def test_missing_key_fails_before_client_or_index_creation(
    monkeypatch,
):
    monkeypatch.setattr(
        service_module.settings,
        "API_KEY",
        None,
    )

    client_factory = Mock()

    monkeypatch.setattr(
        service_module,
        "OpenAI",
        client_factory,
    )

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY",
    ):
        service_module.build_service()

    client_factory.assert_not_called()