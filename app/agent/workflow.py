"""LangGraph answer workflow with bounded, fully reviewed revisions."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agent.answer import (
    DISCLAIMER,
    AnswerBudgetError,
    AnswerDraft,
    generate_answer,
)
from app.agent.grounding import (
    GroundingBudgetError,
    GroundingReview,
    review_answer,
)
from app.agent.planner import (
    AgentProtocolError,
    validate_question,
)
from app.agent.reviewed_research import (
    ReviewedResearchAgent,
    ReviewedResearchResult,
)


@dataclass(frozen=True)
class AnswerAttempt:
    draft: AnswerDraft
    review: GroundingReview


@dataclass(frozen=True)
class AnswerResult:
    research: ReviewedResearchResult
    draft: AnswerDraft | None
    review: GroundingReview | None
    status: str
    notes: str
    revisions: int = 0
    attempts: tuple[AnswerAttempt, ...] = ()

    def to_markdown(self) -> str:
        if (
            self.status != "answer_reviewed"
            or self.draft is None
            or self.review is None
            or not self.review.sufficient
        ):
            raise ValueError(
                "No reviewed answer is available"
            )

        labels = {
            citation.id: f"S{index}"
            for index, citation in enumerate(
                self.draft.citations,
                start=1,
            )
        }

        parts = []

        for statement in self.draft.statements:
            refs = " ".join(
                f"[{labels[section_id]}]"
                for section_id in statement.section_ids
            )
            parts.append(
                f"- {statement.text} {refs}"
            )

        if self.draft.limitations:
            parts.extend([
                "",
                "Evidence limitations:",
            ])
            parts.extend(
                f"- {text}"
                for text in self.draft.limitations
            )

        parts.extend(["", "Sources:"])

        for citation in self.draft.citations:
            parts.append(
                f"- [{labels[citation.id]}] "
                f"{citation.label}"
            )

        parts.extend(["", DISCLAIMER])
        return "\n".join(parts)


class AnswerState(TypedDict):
    question: str
    research: ReviewedResearchResult | None
    draft: AnswerDraft | None
    review: GroundingReview | None
    status: str
    notes: str
    revisions: int
    attempts: list[AnswerAttempt]


class AnswerWorkflow:
    def __init__(
        self,
        researcher: ReviewedResearchAgent,
        *,
        client,
        max_output_tokens: int = 4000,
        max_revisions: int = 2,
        drafter=None,
        critic=None,
        reviser=None,
    ):
        if (
            type(max_output_tokens) is not int
            or max_output_tokens < 1
        ):
            raise ValueError(
                "max_output_tokens must be a positive integer"
            )

        if (
            type(max_revisions) is not int
            or not 0 <= max_revisions <= 2
        ):
            raise ValueError(
                "max_revisions must be 0, 1, or 2"
            )

        self.max_revisions = max_revisions
        self.reviser = reviser or generate_answer
        self.researcher = researcher
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.drafter = drafter or generate_answer
        self.critic = critic or review_answer

        builder = StateGraph(AnswerState)

        builder.add_node("research", self._research)
        builder.add_node("draft_answer", self._draft)
        builder.add_node("grounding_review", self._review)
        builder.add_node("revise_answer", self._revise)

        builder.add_edge(START, "research")

        builder.add_conditional_edges(
            "research",
            lambda state: (
                END
                if state["status"]
                else "draft_answer"
            ),
        )
        builder.add_conditional_edges(
            "draft_answer",
            lambda state: (
                END
                if state["status"]
                else "grounding_review"
            ),
        )
        builder.add_conditional_edges(
            "grounding_review",
            lambda state: (
                END
                if state["status"]
                else "revise_answer"
            ),
        )
        builder.add_conditional_edges(
            "revise_answer",
            lambda state: (
                END
                if state["status"]
                else "grounding_review"
            ),
        )

        self.graph = builder.compile()

    def _request_options(self):
        config = self.researcher.config
        return {
            "client": self.client,
            "model": config.model,
            "max_context_bytes": config.max_context_bytes,
            "max_output_tokens": self.max_output_tokens,
        }

    def _research(self, state: AnswerState):
        research = self.researcher.collect(
            state["question"]
        )

        if (
            research.status != "evidence_reviewed"
            or not research.reviews
            or not research.reviews[-1].sufficient
        ):
            return {
                "research": research,
                "status": "research_incomplete",
                "notes": (
                    f"Research stopped with "
                    f"{research.status}: {research.notes}"
                ),
            }

        return {"research": research}

    def _draft(self, state: AnswerState):
        try:
            draft = self.drafter(
                state["research"],
                **self._request_options(),
            )
        except AnswerBudgetError as exc:
            return {
                "status": "draft_context_limit",
                "notes": str(exc),
            }
        except AgentProtocolError as exc:
            return {
                "status": "draft_protocol_error",
                "notes": str(exc),
            }

        return {"draft": draft}

    def _review(self, state: AnswerState):
        try:
            review = self.critic(
                state["draft"],
                state["research"],
                **self._request_options(),
            )
        except GroundingBudgetError as exc:
            return {
                "status": "grounding_context_limit",
                "notes": str(exc),
            }
        except AgentProtocolError as exc:
            return {
                "status": "grounding_protocol_error",
                "notes": str(exc),
            }   
        attempts = [
            *state["attempts"],
            AnswerAttempt(
                state["draft"],
                review,
            ),
        ]

        if review.sufficient:
            return {
                "review": review,
                "attempts": attempts,
                "status": "answer_reviewed",
                "notes": (
                    "Draft passed source-grounding "
                    "and answer-coverage review "
                    f"after {state['revisions']} revision(s)."
                ),
            }

        exhausted = (
            state["revisions"]
            >= self.max_revisions
        )

        return {
            "review": review,
            "attempts": attempts,
            "status": (
                "grounding_failed"
                if exhausted
                else ""
            ),
            "notes": "\n".join(review.issues),
        }

    def _revise(self, state: AnswerState):
        if (
            state["revisions"]
            >= self.max_revisions
        ):
            return {
                "status": "grounding_failed",
                "notes": "Revision limit reached.",
            }

        count = state["revisions"] + 1

        try:
            draft = self.reviser(
                state["research"],
                previous_draft=state["draft"],
                feedback=state["review"].issues,
                **self._request_options(),
            )
        except AnswerBudgetError as exc:
            return {
                "revisions": count,
                "status": "revision_context_limit",
                "notes": str(exc),
            }
        except AgentProtocolError as exc:
            return {
                "revisions": count,
                "status": "revision_protocol_error",
                "notes": str(exc),
            }

        return {
            "draft": draft,
            "review": None,
            "revisions": count,
            "notes": "",
        }

    def run(self, question: str) -> AnswerResult:
        validate_question(question)

        state = self.graph.invoke(
            {
                "question": question.strip(),
                "research": None,
                "draft": None,
                "review": None,
                "status": "",
                "notes": "",
                "revisions": 0,
                "attempts": [],
            },
            config={
                "recursion_limit": (
                    10 + 2 * self.max_revisions
                )
            },
        )

        return AnswerResult(
            research=state["research"],
            draft=state["draft"],
            review=state["review"],
            status=state["status"],
            notes=state["notes"],
            revisions=state["revisions"],
            attempts=tuple(state["attempts"]),
        )