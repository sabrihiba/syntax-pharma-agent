"""Build and reuse the LangGraph assistant for the HTTP application."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from openai import OpenAI

from app import config as settings
from app.agent.research import AgentConfig
from app.agent.reviewed_research import ReviewedResearchAgent, ReviewConfig
from app.agent.tools import AgentTools
from app.agent.workflow import AnswerWorkflow
from app.rag.cache import CachedEmbedder
from app.rag.chunker import load_corpus
from app.rag.hybrid import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]


class ServiceBusyError(RuntimeError):
    pass


class ServiceClosedError(RuntimeError):
    pass


class AskService:
    """One shared index and workflow; one active question per process."""

    def __init__(self, workflow: AnswerWorkflow, client):
        self.workflow = workflow
        self._client = client
        self._lock = Lock()
        self._closed = False

    def ask(self, question: str):
        if not self._lock.acquire(blocking=False):
            raise ServiceBusyError(
                "The assistant is processing another question"
            )

        try:
            if self._closed:
                raise ServiceClosedError(
                    "The assistant is closed"
                )

            return self.workflow.run(question)
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._client.close()


def build_service(
    corpus_dir: Path | None = None,
) -> AskService:
    if (
        not isinstance(settings.API_KEY, str)
        or not settings.API_KEY.strip()
    ):
        raise ValueError(
            "OPENAI_API_KEY must be configured"
        )

    config = AgentConfig(max_turns=12)

    corpus = load_corpus(
        corpus_dir
        if corpus_dir is not None
        else ROOT / "corpus"
    )

    if not corpus.chunks:
        raise ValueError(
            "The corpus contains no searchable chunks"
        )

    client = OpenAI(
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL or None,
        timeout=45.0,
        max_retries=1,
    )

    try:
        retriever = HybridRetriever.build(
            corpus,
            CachedEmbedder(client=client),
        )

        researcher = ReviewedResearchAgent(
            AgentTools(retriever),
            config,
            review_config=ReviewConfig(
                max_reviews=3
            ),
            client=client,
        )

        workflow = AnswerWorkflow(
            researcher,
            client=client,
            max_revisions=2,
        )

        return AskService(workflow, client)

    except Exception:
        client.close()
        raise