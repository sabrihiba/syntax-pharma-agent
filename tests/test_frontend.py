"""Offline HTTP checks for the frontend and its API integration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from test_agent_answer import research
from test_agent_grounding import draft
from test_api import reviewed_result

import app.main as main_module


@pytest.fixture
def web(reviewed_result):
    service = SimpleNamespace(
        ask=Mock(return_value=reviewed_result),
        close=Mock(),
    )

    factory = Mock(return_value=service)
    app = main_module.create_app(
        service_factory=factory
    )

    with TestClient(app) as client:
        yield client, service, factory


def test_home_page_is_utf8_html(web):
    client, service, _ = web

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/html"
    )
    assert "charset=utf-8" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache"
    assert "Pharma Research Assistant" in response.text
    assert "<textarea" in response.text
    assert "⌘" in response.text

    service.ask.assert_not_called()


def test_home_and_health_reuse_the_same_service(web):
    client, service, factory = web

    assert client.get("/").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {
        "status": "ok",
        "ready": True,
    }

    factory.assert_called_once()
    service.ask.assert_not_called()


def test_ask_remains_available_with_frontend(web):
    client, service, _ = web

    assert client.get("/").status_code == 200

    response = client.post(
        "/ask",
        json={"question": "Example question?"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "answer_reviewed"

    service.ask.assert_called_once_with(
        "Example question?"
    )


def test_missing_frontend_does_not_disable_api(
    web,
    monkeypatch,
):
    client, service, _ = web

    monkeypatch.setattr(
        main_module,
        "FRONTEND_FILE",
        Path(__file__).resolve().parent
        / "__missing_frontend_test__.html",
    )

    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200

    response = client.post(
        "/ask",
        json={"question": "Example?"},
    )

    assert response.status_code == 200