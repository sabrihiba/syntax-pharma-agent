"""FastAPI interface and frontend for the reviewed LangGraph assistant."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from starlette.concurrency import run_in_threadpool

from app.agent.answer import DISCLAIMER
from app.agent.planner import AgentProtocolError
from app.service import (
    ServiceBusyError,
    ServiceClosedError,
    build_service,
)


logger = logging.getLogger(__name__)
FRONTEND_FILE = (
    Path(__file__).resolve().parent
    / "static"
    / "index.html"
)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        strict=True,
        min_length=1,
        max_length=4000,
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "question must not be blank"
            )

        return value


class CitationResponse(BaseModel):
    reference: str
    id: str
    file: str
    source_label: str
    url: str | None
    section_number: str | None
    section_path: str
    label: str


class ToolCallResponse(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] | str
    ok: bool
    delivered: bool


class EvidenceResponse(BaseModel):
    section_id: str
    file: str
    drug: str | None
    source: str
    section_path: str
    complete: bool
    missing_ranges: list[tuple[int, int]] | None
    matched_chunk_ids: list[str]


class ReviewResponse(BaseModel):
    sufficient: bool
    issues: list[str]


class TraceResponse(BaseModel):
    plan: list[str]
    tool_calls: list[ToolCallResponse]
    evidence: list[EvidenceResponse]
    evidence_reviews: list[ReviewResponse]
    answer_reviews: list[ReviewResponse]
    rejected_batches: list[dict[str, Any]]
    research_status: str
    collection_status: str
    revisions: int
    notes: str
    elapsed_ms: float


class AskResponse(BaseModel):
    question: str
    status: str
    answer: str | None
    citations: list[CitationResponse]
    trace: TraceResponse
    disclaimer: str


def _arguments(raw: str):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return raw

    return value if isinstance(value, dict) else raw


def make_response(
    result,
    question: str,
    elapsed_ms: float,
) -> AskResponse:
    research = result.research
    answer = None
    citations = []

    if result.status == "answer_reviewed":
        answer = result.to_markdown()

        citations = [
            CitationResponse(
                reference=f"S{index}",
                **asdict(citation),
                label=citation.label,
            )
            for index, citation in enumerate(
                result.draft.citations,
                start=1,
            )
        ]

    trace = TraceResponse(
        plan=list(research.plan),
        tool_calls=[
            ToolCallResponse(
                call_id=entry.call_id,
                name=entry.name,
                arguments=_arguments(
                    entry.arguments
                ),
                ok=entry.ok,
                delivered=entry.delivered,
            )
            for entry in research.trace
        ],
        evidence=[
            EvidenceResponse(
                section_id=section.section_id,
                file=section.citation.file,
                drug=section.drug,
                source=section.source,
                section_path=section.citation.section_path,
                complete=section.complete,
                missing_ranges=(
                    list(section.missing_ranges)
                    if section.missing_ranges is not None
                    else None
                ),
                matched_chunk_ids=list(
                    section.matched_chunk_ids
                ),
            )
            for section in research.sections
        ],
        evidence_reviews=[
            ReviewResponse(
                sufficient=review.sufficient,
                issues=list(review.gaps),
            )
            for review in research.reviews
        ],
        answer_reviews=[
            ReviewResponse(
                sufficient=attempt.review.sufficient,
                issues=list(attempt.review.issues),
            )
            for attempt in result.attempts
        ],
        rejected_batches=[
            asdict(item)
            for item in research.rejected_batches
        ],
        research_status=research.status,
        collection_status=research.collection_status,
        revisions=result.revisions,
        notes=result.notes,
        elapsed_ms=elapsed_ms,
    )

    return AskResponse(
        question=question,
        status=result.status,
        answer=answer,
        citations=citations,
        trace=trace,
        disclaimer=DISCLAIMER,
    )


def create_app(
    *,
    service_factory=build_service,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = await run_in_threadpool(
            service_factory
        )
        application.state.service = service

        try:
            yield
        finally:
            application.state.service = None
            await run_in_threadpool(
                service.close
            )

    application = FastAPI(
        title="Pharma Research Assistant",
        description=(
            "Cited pharmacology answers using a reviewed LangGraph workflow"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.state.service = None

    @application.get(
        "/",
        response_class=FileResponse,
        include_in_schema=False,
    )
    def home():
        if not FRONTEND_FILE.is_file():
            raise HTTPException(
                404,
                "Frontend file is missing: app/static/index.html",
            )

        return FileResponse(
            FRONTEND_FILE,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache",
            },
        )

    @application.get("/health")
    def health_check():
        return {
            "status": "ok",
            "ready": application.state.service is not None,
        }

    @application.post(
        "/ask",
        response_model=AskResponse,
    )
    def ask(
        payload: AskRequest,
        request: Request,
    ):
        service = request.app.state.service

        if service is None:
            raise HTTPException(
                503,
                "Assistant is not ready",
            )

        started = perf_counter()

        try:
            result = service.ask(
                payload.question
            )

            return make_response(
                result,
                payload.question,
                round(
                    (perf_counter() - started) * 1000,
                    2,
                ),
            )

        except ServiceBusyError:
            raise HTTPException(
                503,
                "Assistant is processing another question; try again shortly",
            ) from None

        except ServiceClosedError:
            raise HTTPException(
                503,
                "Assistant is not available",
            ) from None

        except (APITimeoutError, TimeoutError):
            raise HTTPException(
                504,
                "The model provider request timed out",
            ) from None

        except RateLimitError:
            raise HTTPException(
                503,
                "The model provider is rate-limiting requests",
            ) from None

        except (
            APIConnectionError,
            APIStatusError,
            AgentProtocolError,
        ):
            raise HTTPException(
                502,
                "The model provider did not return a usable response",
            ) from None

        except Exception as exc:
            logger.error(
                "Assistant request failed: %s",
                type(exc).__name__,
            )

            raise HTTPException(
                500,
                "The assistant could not complete this request",
            ) from None

    return application


app = create_app()