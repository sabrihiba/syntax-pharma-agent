"""Live check: research -> drafting -> grounding review."""

import argparse
from pathlib import Path

from openai import OpenAI

from app import config as settings
from app.agent.research import AgentConfig
from app.agent.reviewed_research import (
    ReviewedResearchAgent,
    ReviewConfig,
)
from app.agent.tools import AgentTools
from app.agent.workflow import AnswerWorkflow
from app.rag.cache import CachedEmbedder
from app.rag.chunker import load_corpus
from app.rag.hybrid import HybridRetriever


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        nargs="?",
        default=(
            "Compare the contraindications of warfarin and apixaban. "
            "What conditions are shared and which are unique to each drug?"
        ),
    )

    args = parser.parse_args()

    if not settings.API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing"
        )

    root = Path(__file__).resolve().parents[1]
    corpus = load_corpus(root / "corpus")

    print("Loading cached retrieval index...")

    retriever = HybridRetriever.build(
        corpus,
        CachedEmbedder(),
    )

    client = OpenAI(
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL or None,
        timeout=45.0,
        max_retries=1,
    )

    researcher = ReviewedResearchAgent(
        AgentTools(retriever),
        AgentConfig(max_turns=12),
        review_config=ReviewConfig(
            max_reviews=3
        ),
        client=client,
    )

    result = AnswerWorkflow(
        researcher,
        client=client,
    ).run(args.question)

    print(
        "\nQUESTION:",
        result.research.question,
    )
    print(
        "RESEARCH STATUS:",
        result.research.status,
    )
    print(
        "TOOL CALLS:",
        len(result.research.trace),
    )
    print(
        "FINAL STATUS:",
        result.status,
    )
    print(
        "NOTES:",
        result.notes,
    )

    if result.review:
        for check in result.review.claims:
            print(
                f"Statement {check.index}: "
                f"passed={check.passed} — "
                f"{check.explanation}"
            )

        for check in result.review.coverage:
            print(
                f"Plan step {check.index}: "
                f"passed={check.passed} — "
                f"{check.explanation}"
            )

        print(
            "LIMITATIONS SUPPORTED:",
            result.review.limitations_supported,
        )

    if result.status == "answer_reviewed":
        print("\nREVIEWED ANSWER:\n")
        print(result.to_markdown())

    elif result.draft:
        print(
            "\nUNAPPROVED DRAFT "
            "FOR DEBUGGING ONLY:\n"
        )
        print(result.draft.to_markdown())


if __name__ == "__main__":
    main()