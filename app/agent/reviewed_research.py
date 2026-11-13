"""LangGraph research with consolidation and bounded coverage review.

Extends ResearchAgent without changing its implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from app.agent.consolidation import (
    SectionEvidence,
    consolidate_evidence,
)
from app.agent.evidence_review import (
    CoverageReview,
    ReviewBudgetError,
    review_evidence,
)
from app.agent.planner import validate_question
from app.agent.research import (
    AgentConfig,
    ResearchAgent,
    ResearchResult,
    ResearchState,
)
from app.agent.tools import AgentTools


@dataclass(frozen=True)
class ReviewConfig:
    max_reviews: int = 3

    def __post_init__(self):
        if (
            type(self.max_reviews) is not int
            or self.max_reviews < 1
        ):
            raise ValueError(
                "max_reviews must be a positive integer"
            )


@dataclass(frozen=True)
class ReviewedResearchResult(ResearchResult):
    sections: tuple[SectionEvidence, ...] = ()
    reviews: tuple[CoverageReview, ...] = ()
    collection_status: str = ""


class ReviewedState(ResearchState):
    sections: list[SectionEvidence]
    reviews: list[CoverageReview]
    collection_status: str


class ReviewedResearchAgent(ResearchAgent):
    def __init__(
        self,
        tools: AgentTools,
        config: AgentConfig | None = None,
        *,
        review_config: ReviewConfig | None = None,
        client=None,
        reviewer=None,
    ):
        self.review_config = review_config or ReviewConfig()
        self._reviewer = reviewer or review_evidence

        super().__init__(
            tools,
            config,
            client=client,
        )

    def _build_graph(self):
        builder = StateGraph(ReviewedState)

        builder.add_node("planner", self._plan_node)
        builder.add_node("model", self._model_node)
        builder.add_node("tools", self._tools_node)
        builder.add_node("consolidate", self._consolidate_node)
        builder.add_node("review", self._review_node)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "model")

        builder.add_conditional_edges(
            "model",
            self._after_reviewed_model,
            {
                "model": "model",
                "tools": "tools",
                "consolidate": "consolidate",
            },
        )

        builder.add_conditional_edges(
            "tools",
            self._after_reviewed_tools,
            {
                "model": "model",
                "consolidate": "consolidate",
            },
        )

        builder.add_edge("consolidate", "review")

        builder.add_conditional_edges(
            "review",
            self._after_review,
            {
                "model": "model",
                END: END,
            },
        )

        return builder.compile()

    @staticmethod
    def _after_reviewed_model(state: ReviewedState):
        if state["status"]:
            return "consolidate"

        return (
            "tools"
            if state["pending_calls"]
            else "model"
        )

    @staticmethod
    def _after_reviewed_tools(state: ReviewedState):
        return (
            "consolidate"
            if state["status"]
            else "model"
        )

    @staticmethod
    def _after_review(state: ReviewedState):
        return END if state["status"] else "model"

    def _consolidate_node(self, state: ReviewedState):
        return {
            "sections": list(
                consolidate_evidence(state["evidence"])
            ),
            "collection_status": state["status"],
        }

    def _review_node(self, state: ReviewedState):
        if len(state["reviews"]) >= self.review_config.max_reviews:
            return {
                "status": "review_limit",
                "notes": (
                    "Evidence-review limit reached; "
                    "evidence was preserved."
                ),
            }

        try:
            review = self._reviewer(
                state["question"],
                tuple(state["plan"]),
                tuple(state["sections"]),
                client=self._get_client(),
                model=self.config.model,
                max_context_bytes=self.config.max_context_bytes,
                max_output_tokens=self.config.max_output_tokens,
            )
        except ReviewBudgetError as exc:
            return {
                "status": "review_context_limit",
                "notes": str(exc),
            }

        reviews = [
            *state["reviews"],
            review,
        ]

        if review.sufficient:
            return {
                "reviews": reviews,
                "status": "evidence_reviewed",
                "notes": (
                    "Reviewer marked plan coverage sufficient. "
                    "Final answer drafting and grounding review "
                    "are still pending."
                ),
            }

        notes = "\n".join(review.gaps)

        if len(reviews) >= self.review_config.max_reviews:
            return {
                "reviews": reviews,
                "status": "review_limit",
                "notes": notes,
            }

        # Review cannot bypass research execution limits.
        if state["collection_status"] not in {
            "model_finished",
            "no_evidence",
        }:
            return {
                "reviews": reviews,
                "status": state["collection_status"],
                "notes": notes,
            }

        if (
            state["turns"] >= self.config.max_turns
            or len(state["trace"]) >= self.config.max_tool_calls
        ):
            return {
                "reviews": reviews,
                "status": "research_limit",
                "notes": notes,
            }

        feedback = {
            "type": "evidence_review_feedback",
            "gaps": review.gaps,
            "instruction": (
                "Collect source evidence for these gaps. "
                "Choose the next tool calls yourself. "
                "Preserve original question scope. "
                "Read relevant full sections/pages where needed."
            ),
        }

        messages = [
            *state["messages"],
            {
                "role": "user",
                "content": json.dumps(feedback),
            },
        ]

        if not self._fits(messages):
            return {
                "reviews": reviews,
                "status": "context_limit",
                "notes": (
                    "Review feedback does not fit. "
                    "Remaining gaps: " + notes
                ),
            }

        return {
            "reviews": reviews,
            "status": "",
            "controller_feedback": "",
            "messages": messages,
            "notes": notes,
        }

    def collect(
        self,
        question: str,
    ) -> ReviewedResearchResult:
        validate_question(question)

        state = self.graph.invoke(
            {
                "question": question,
                "plan": (),
                "messages": [],
                "evidence": [],
                "trace": [],
                "pending_calls": [],
                "call_ids": [],
                "turns": 0,
                "budget_retries": 0,
                "batch_retries": 0,
                "controller_feedback": "",
                "rejected_batches": [],
                "status": "",
                "notes": "",
                "sections": [],
                "reviews": [],
                "collection_status": "",
            },
            config={
                "recursion_limit": (
                    2 * self.config.max_turns
                    + 3 * self.review_config.max_reviews
                    + 10
                )
            },
        )

        return ReviewedResearchResult(
            question=question,
            plan=tuple(state["plan"]),
            evidence=tuple(state["evidence"]),
            trace=tuple(state["trace"]),
            status=state["status"],
            notes=state["notes"],
            rejected_batches=tuple(
                state["rejected_batches"]
            ),
            sections=tuple(state["sections"]),
            reviews=tuple(state["reviews"]),
            collection_status=state["collection_status"],
        )